#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from hand_object_demo.training import (
    build_dataloaders,
    create_model,
    run_epoch,
    save_checkpoint,
    write_history_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNet classifier on the extracted crops.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root created by prepare_dataset.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use ImageNet-pretrained weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_acc = -1.0
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)

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
            best_payload = payload
            save_checkpoint(args.output_dir / "best.pt", payload)

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
