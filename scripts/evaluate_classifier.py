#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from hand_object_demo.training import build_dataloaders, create_model, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on the test split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt["class_names"]
    model_name = ckpt["model_name"]

    _, loaders, _ = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = create_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, targets in loaders["test"]:
            images = images.to(args.device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(targets.tolist())

    report = classification_report(all_true, all_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true, all_pred).tolist()
    payload = {
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    write_json(args.output_dir / "evaluation.json", payload)
    print(f"Saved evaluation to: {args.output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
