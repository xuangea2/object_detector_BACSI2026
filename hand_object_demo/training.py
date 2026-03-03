from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class CSVCropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], class_to_idx: dict[str, int], transform=None):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        target = self.class_to_idx[row["label"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            # Geometry — aggressive rotations, flips, perspective
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=20),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            # Color — brightness, contrast, saturation
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.RandomGrayscale(p=0.1),
            # Blur — light, probabilistic
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name={model_name!r}")


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[dict[str, CSVCropDataset], dict[str, DataLoader], list[str]]:
    samples_csv = dataset_root / "metadata" / "samples.csv"
    if not samples_csv.exists():
        raise FileNotFoundError(f"Missing samples CSV: {samples_csv}")

    df = pd.read_csv(samples_csv)
    if df.empty:
        raise RuntimeError("samples.csv is empty. Run prepare_dataset.py first.")

    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_tfms, eval_tfms = build_transforms(image_size=image_size)
    datasets_map = {}
    for split, tfm in [("train", train_tfms), ("val", eval_tfms), ("test", eval_tfms)]:
        split_rows = df[df["split"] == split].to_dict(orient="records")
        datasets_map[split] = CSVCropDataset(split_rows, class_to_idx=class_to_idx, transform=tfm)

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets_map.items()
    }
    return datasets_map, loaders, class_names


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, running_correct / total


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
