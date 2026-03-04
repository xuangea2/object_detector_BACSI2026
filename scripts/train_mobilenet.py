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
    SUPPORTED_MODELS,
    build_dataloaders,
    build_scheduler,
    create_model,
    freeze_backbone,
    run_epoch,
    save_checkpoint,
    set_head_dropout,
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
        "--mixup-alpha", type=float, default=0.3,
        help="Mixup interpolation strength. 0 to disable (default: 0.3).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience: epochs without val_acc improvement. 0 to disable.",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=3,
        help="Freeze backbone for the first N epochs (fine-tune head only).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing factor for CrossEntropyLoss.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=3,
        help="Linear warmup epochs for LR scheduler.",
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
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets_map, loaders, class_names = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("At least 2 classes are required to train the classifier.")

    model = create_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    set_head_dropout(model, args.dropout)
    model.to(device)

    # Freeze backbone for the first N epochs so only the head trains
    if args.freeze_epochs > 0 and args.pretrained:
        freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    history = []
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze period
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0 and args.pretrained:
            unfreeze_backbone(model)
            # Rebuild optimizer so all params get proper LR
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            remaining = args.epochs - args.freeze_epochs
            scheduler = build_scheduler(
                optimizer, epochs=remaining,
                warmup_epochs=min(args.warmup_epochs, remaining),
            )

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


if __name__ == "__main__":
    main()
