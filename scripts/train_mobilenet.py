#!/usr/bin/env python3
"""Train a classifier on hand-object crops.

Supports 10 architectures (see ``--model``).  Dataset and output directories
are auto-versioned by default so you can iterate quickly without overwriting
previous runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from hand_object_demo.io_utils import latest_versioned_dir, next_versioned_dir
from hand_object_demo.training import (
    DEFAULT_AUGMENT,
    SUPPORTED_MODELS,
    build_dataloaders,
    build_scheduler,
    create_model,
    export_model,
    freeze_backbone,
    get_param_groups,
    run_epoch,
    save_checkpoint,
    set_head_dropout,
    set_head_mlp,
    unfreeze_backbone,
    write_history_csv,
    write_json,
)


_TRAINING_ROOT = Path("data/training")
_MODELS_ROOT = Path("data/models")


def _resolve_dataset_root(raw: Path | None) -> Path:
    """If the user did not pass ``--dataset-root``, pick the latest
    ``data/training/dataset_*`` directory automatically."""
    if raw is not None:
        return raw.resolve()
    latest = latest_versioned_dir(_TRAINING_ROOT, "dataset")
    if latest is None:
        raise RuntimeError(
            f"No dataset found under {_TRAINING_ROOT.resolve()}. "
            "Run prepare_dataset.py first or pass --dataset-root explicitly."
        )
    return latest.resolve()


def _resolve_output_dir(raw: Path | None, model_name: str) -> Path:
    """If the user did not pass ``--output-dir``, create the next
    ``data/models/train_{model}_{N}`` directory automatically."""
    if raw is not None:
        return raw.resolve()
    prefix = f"train_{model_name}"
    out = next_versioned_dir(_MODELS_ROOT, prefix)
    return out.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier on the extracted hand-object crops.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Dataset root created by prepare_dataset.py. "
             "Default: latest data/training/dataset_* version.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Default: data/models/train_{model}_{version}.",
    )
    parser.add_argument(
        "--model", type=str, default="mobilenet_v3_small",
        choices=SUPPORTED_MODELS,
        help="Architecture to train (default: mobilenet_v3_small).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--dropout", type=float, default=0.4,
        help="Dropout probability in the classifier head (default: 0.4).",
    )
    parser.add_argument(
        "--head", type=str, default="linear", choices=["linear", "mlp"],
        help="Head type: 'linear' (single layer) or 'mlp' (3-layer MLP, "
             "best with --freeze-epochs -1).",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.3,
        help="Mixup interpolation strength. 0 to disable (default: 0.3).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience: epochs without val_acc improvement. 0 to disable.",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=3,
        help="Freeze backbone for the first N epochs. Use -1 to freeze forever "
             "(recommended with --head mlp).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing factor for CrossEntropyLoss.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=3,
        help="Linear warmup epochs for LR scheduler.",
    )
    parser.add_argument(
        "--backbone-lr-factor", type=float, default=0.1,
        help="LR multiplier for backbone params after unfreezing. "
             "Backbone LR = lr * factor (default: 0.1, i.e. backbone gets "
             "10x lower LR than head). Set to 1.0 to disable discriminative LR.",
    )
    parser.add_argument(
        "--refreeze-epoch", type=int, default=0,
        help="Re-freeze backbone at this epoch and continue head-only training. "
             "0 = disabled. Example: --freeze-epochs 5 --refreeze-epoch 20 "
             "gives 3 phases: head -> full -> head.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True,
        help="Use ImageNet-pretrained weights (default: True). "
             "Use --no-pretrained to train from scratch.",
    )

    # ── Data augmentation overrides ─────────────────────────────────────
    aug = parser.add_argument_group(
        "data augmentation",
        "Override any augmentation parameter.  Values of 0 disable the "
        "corresponding transform.  Defaults match DEFAULT_AUGMENT in training.py.",
    )
    _D = DEFAULT_AUGMENT  # shorthand
    # Geometry
    aug.add_argument("--crop-scale-lo",   type=float, default=None, metavar="F", help=f"RandomResizedCrop min scale (default {_D['crop_scale_lo']}).")
    aug.add_argument("--crop-scale-hi",   type=float, default=None, metavar="F", help=f"RandomResizedCrop max scale (default {_D['crop_scale_hi']}).")
    aug.add_argument("--hflip",           type=float, default=None, metavar="P", help=f"Horizontal flip prob (default {_D['hflip']}). 0=off.")
    aug.add_argument("--rotation",        type=float, default=None, metavar="DEG", help=f"Max rotation degrees (default {_D['rotation']}). 0=off.")
    aug.add_argument("--translate",       type=float, default=None, metavar="F", help=f"Max translate fraction (default {_D['translate']}). 0=off.")
    aug.add_argument("--shear",           type=float, default=None, metavar="DEG", help=f"Shear range degrees (default {_D['shear']}). 0=off.")
    aug.add_argument("--perspective",     type=float, default=None, metavar="F", help=f"Perspective distortion scale (default {_D['perspective']}).")
    aug.add_argument("--perspective-p",   type=float, default=None, metavar="P", help=f"Perspective prob (default {_D['perspective_p']}). 0=off.")
    # Color
    aug.add_argument("--brightness",      type=float, default=None, metavar="F", help=f"ColorJitter brightness (default {_D['brightness']}). 0=off.")
    aug.add_argument("--contrast",        type=float, default=None, metavar="F", help=f"ColorJitter contrast (default {_D['contrast']}). 0=off.")
    aug.add_argument("--saturation",      type=float, default=None, metavar="F", help=f"ColorJitter saturation (default {_D['saturation']}). 0=off.")
    aug.add_argument("--hue",             type=float, default=None, metavar="F", help=f"ColorJitter hue (default {_D['hue']}). 0=off.")
    aug.add_argument("--grayscale-p",     type=float, default=None, metavar="P", help=f"RandomGrayscale prob (default {_D['grayscale_p']}). 0=off.")
    aug.add_argument("--equalize-p",      type=float, default=None, metavar="P", help=f"RandomEqualize prob (default {_D['equalize_p']}). 0=off.")
    aug.add_argument("--autocontrast-p",  type=float, default=None, metavar="P", help=f"RandomAutocontrast prob (default {_D['autocontrast_p']}). 0=off.")
    aug.add_argument("--sharpness-factor", type=float, default=None, metavar="F", help=f"Sharpness factor (default {_D['sharpness_factor']}).")
    aug.add_argument("--sharpness-p",     type=float, default=None, metavar="P", help=f"RandomAdjustSharpness prob (default {_D['sharpness_p']}). 0=off.")
    # Blur
    aug.add_argument("--blur-sigma-lo",   type=float, default=None, metavar="F", help=f"GaussianBlur sigma min (default {_D['blur_sigma_lo']}).")
    aug.add_argument("--blur-sigma-hi",   type=float, default=None, metavar="F", help=f"GaussianBlur sigma max (default {_D['blur_sigma_hi']}).")
    aug.add_argument("--blur-p",          type=float, default=None, metavar="P", help=f"GaussianBlur prob (default {_D['blur_p']}). 0=off.")
    # Erasing
    aug.add_argument("--erasing-p",       type=float, default=None, metavar="P", help=f"RandomErasing prob (default {_D['erasing_p']}). 0=off.")
    aug.add_argument("--erasing-scale-lo", type=float, default=None, metavar="F", help=f"RandomErasing min area fraction (default {_D['erasing_scale_lo']}).")
    aug.add_argument("--erasing-scale-hi", type=float, default=None, metavar="F", help=f"RandomErasing max area fraction (default {_D['erasing_scale_hi']}).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve auto-versioned paths
    args.dataset_root = _resolve_dataset_root(args.dataset_root)
    args.output_dir = _resolve_output_dir(args.output_dir, args.model)

    print(f"Dataset root : {args.dataset_root}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Model        : {args.model}")
    print(f"Head         : {args.head}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Collect augmentation overrides (only flags explicitly set by the user)
    _AUG_KEYS = list(DEFAULT_AUGMENT.keys())
    augment_cfg: dict[str, float] = {}
    for key in _AUG_KEYS:
        cli_attr = key  # argparse stores with underscores already
        val = getattr(args, cli_attr, None)
        if val is not None:
            augment_cfg[key] = val

    datasets_map, loaders, class_names = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=augment_cfg or None,
    )
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("At least 2 classes are required to train the classifier.")

    model = create_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    if args.head == "mlp":
        set_head_mlp(model, num_classes=num_classes, dropout=args.dropout)
    else:
        set_head_dropout(model, args.dropout)
    model.to(device)

    # Freeze backbone: -1 = freeze forever, >0 = freeze for N epochs
    freeze_forever = args.freeze_epochs < 0
    if (args.freeze_epochs > 0 or freeze_forever) and args.pretrained:
        freeze_backbone(model)

    if args.refreeze_epoch > 0:
        if args.refreeze_epoch <= args.freeze_epochs:
            raise ValueError(
                f"--refreeze-epoch ({args.refreeze_epoch}) must be > "
                f"--freeze-epochs ({args.freeze_epochs})"
            )
        if args.refreeze_epoch > args.epochs:
            raise ValueError(
                f"--refreeze-epoch ({args.refreeze_epoch}) must be <= "
                f"--epochs ({args.epochs})"
            )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_str = "forever" if freeze_forever else f"{args.freeze_epochs} epochs"
    print(f"Backbone     : frozen {frozen_str}" if (args.freeze_epochs != 0 and args.pretrained) else "Backbone     : trainable")
    if not freeze_forever and args.freeze_epochs > 0 and args.pretrained:
        bb_lr = args.lr * args.backbone_lr_factor
        print(f"Backbone LR  : {bb_lr:.1e} (factor {args.backbone_lr_factor})")
    if args.refreeze_epoch > 0:
        print(f"Re-freeze at : epoch {args.refreeze_epoch}")
    print(f"Parameters   : {trainable:,} trainable / {total_params:,} total")
    # Show training phases
    if args.freeze_epochs > 0 and args.pretrained and not freeze_forever:
        phase1 = f"  Phase 1 (epochs 1-{args.freeze_epochs}): head only"
        if args.refreeze_epoch > 0:
            phase2 = f"  Phase 2 (epochs {args.freeze_epochs+1}-{args.refreeze_epoch}): full model (backbone LR x{args.backbone_lr_factor})"
            phase3 = f"  Phase 3 (epochs {args.refreeze_epoch+1}-{args.epochs}): head only (re-frozen)"
            print(f"\nTraining schedule:\n{phase1}\n{phase2}\n{phase3}")
        else:
            phase2 = f"  Phase 2 (epochs {args.freeze_epochs+1}-{args.epochs}): full model (backbone LR x{args.backbone_lr_factor})"
            print(f"\nTraining schedule:\n{phase1}\n{phase2}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    history = []
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze period (skip if frozen forever)
        if not freeze_forever and epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0 and args.pretrained:
            unfreeze_backbone(model)
            # Rebuild optimizer with discriminative LR (lower LR for backbone)
            param_groups = get_param_groups(
                model, lr=args.lr,
                backbone_lr_factor=args.backbone_lr_factor,
                weight_decay=args.weight_decay,
            )
            optimizer = AdamW(param_groups)
            end_epoch = args.refreeze_epoch if args.refreeze_epoch > 0 else args.epochs
            remaining = end_epoch - args.freeze_epochs
            scheduler = build_scheduler(
                optimizer, epochs=remaining,
                warmup_epochs=min(args.warmup_epochs, remaining),
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            bb_lr = args.lr * args.backbone_lr_factor
            print(f"\n>>> Epoch {epoch}: backbone UNFROZEN  "
                  f"(backbone LR={bb_lr:.1e}, head LR={args.lr:.1e}, "
                  f"{n_trainable:,} trainable)\n")

        # Re-freeze backbone after fine-tuning window
        if args.refreeze_epoch > 0 and epoch == args.refreeze_epoch + 1:
            freeze_backbone(model)
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, weight_decay=args.weight_decay,
            )
            remaining = args.epochs - args.refreeze_epoch
            scheduler = build_scheduler(
                optimizer, epochs=remaining,
                warmup_epochs=min(args.warmup_epochs, remaining),
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n>>> Epoch {epoch}: backbone RE-FROZEN  "
                  f"(head LR={args.lr:.1e}, {n_trainable:,} trainable)\n")

        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True,
            mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(row)

        payload = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "args": vars(args),
            "val_acc": val_acc,
        }
        save_checkpoint(args.output_dir / "last.pt", payload)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(args.output_dir / "best.pt", payload)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val_acc improvement for {args.patience} epochs).")
                break

    write_history_csv(history, args.output_dir / "history.csv")
    write_json(
        args.output_dir / "training_config.json",
        {
            "args": vars(args),
            "class_names": class_names,
            "dataset_sizes": {k: len(v) for k, v in datasets_map.items()},
            "best_val_acc": best_val_acc,
        },
    )

    print("\nTraining finished.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")

    # --- Export best model for deployment (TorchScript + ONNX) ---
    print("\nExporting best model for deployment …")
    best_ckpt = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    export_model_inst = create_model(args.model, num_classes=num_classes, pretrained=False)
    if args.head == "mlp":
        set_head_mlp(export_model_inst, num_classes=num_classes, dropout=0.0)
    export_model_inst.load_state_dict(best_ckpt["model_state_dict"])
    export_model(
        export_model_inst,
        class_names=class_names,
        output_dir=args.output_dir,
        image_size=args.image_size,
        prefix="best",
    )


if __name__ == "__main__":
    main()
