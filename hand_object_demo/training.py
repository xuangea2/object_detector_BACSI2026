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
            # Geometry — mild, realistic distortions only
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=8),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            # Color — brightness, contrast, saturation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            transforms.RandomGrayscale(p=0.05),
            # Blur — light, probabilistic
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
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
    """Instantiate a classifier, replacing the final head with *num_classes* outputs.

    Supported architectures (all accept ImageNet-pretrained weights):

    ── MobileNet ──────────────────────────────────────────────────────────
    mobilenet_v3_small   ~2.5 M params   very fast, edge-friendly
    mobilenet_v3_large   ~5.4 M params   stronger than small

    ── EfficientNet ───────────────────────────────────────────────────────
    efficientnet_b0      ~5.3 M params   best accuracy/size trade-off
    efficientnet_v2_s    ~21 M params    fast training, strong accuracy

    ── ConvNeXt ───────────────────────────────────────────────────────────
    convnext_tiny        ~28 M params    modern "ConvNet strikes back"
    convnext_small       ~50 M params    higher capacity

    ── ResNet ─────────────────────────────────────────────────────────────
    resnet18             ~11 M params    lightweight classic
    resnet50             ~25 M params    strong baseline

    ── Lightweight ────────────────────────────────────────────────────────
    shufflenet_v2        ~2.3 M params   fastest, very small
    squeezenet           ~1.2 M params   tiny footprint
    """
    _registry: dict[str, tuple] = {
        # (factory, weights, head_attr, head_type)
        # head_type: "linear_last" → last element of nn.Sequential is a Linear
        #            "linear_attr" → attribute is itself a Linear
        #            "conv_attr"   → attribute is a Conv2d (SqueezeNet)
        "mobilenet_v3_small": (
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "efficientnet_v2_s": (
            models.efficientnet_v2_s,
            models.EfficientNet_V2_S_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "convnext_tiny": (
            models.convnext_tiny,
            models.ConvNeXt_Tiny_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "convnext_small": (
            models.convnext_small,
            models.ConvNeXt_Small_Weights.DEFAULT,
            "classifier.-1",
            "linear_last",
        ),
        "resnet18": (
            models.resnet18,
            models.ResNet18_Weights.DEFAULT,
            "fc",
            "linear_attr",
        ),
        "resnet50": (
            models.resnet50,
            models.ResNet50_Weights.DEFAULT,
            "fc",
            "linear_attr",
        ),
        "shufflenet_v2": (
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
            "fc",
            "linear_attr",
        ),
        "squeezenet": (
            models.squeezenet1_1,
            models.SqueezeNet1_1_Weights.DEFAULT,
            "classifier.1",
            "conv_attr",
        ),
    }

    if model_name not in _registry:
        supported = ", ".join(sorted(_registry))
        raise ValueError(f"Unsupported model_name={model_name!r}. Choose from: {supported}")

    factory, weights_enum, head_path, head_type = _registry[model_name]
    model = factory(weights=weights_enum if pretrained else None)

    # --- replace the classification head ---
    parts = head_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.lstrip("-").isdigit() else parent[int(p)]
    last_key = parts[-1]

    if head_type == "linear_last":
        container = parent if len(parts) == 1 else parent
        idx = int(last_key)
        old_linear = container[idx]
        container[idx] = nn.Linear(old_linear.in_features, num_classes)
    elif head_type == "linear_attr":
        old_linear = getattr(parent, last_key)
        setattr(parent, last_key, nn.Linear(old_linear.in_features, num_classes))
    elif head_type == "conv_attr":
        # SqueezeNet uses Conv2d(512, num_classes, 1, 1) as its classifier
        old_conv = parent[int(last_key)]
        parent[int(last_key)] = nn.Conv2d(old_conv.in_channels, num_classes, kernel_size=1)
    return model


# All valid model names for CLI choices
SUPPORTED_MODELS: list[str] = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "convnext_tiny",
    "convnext_small",
    "resnet18",
    "resnet50",
    "shufflenet_v2",
    "squeezenet",
]


def set_head_dropout(model: nn.Module, p: float) -> None:
    """Set or insert dropout in the classifier head to probability *p*.

    Works across all supported architectures:

    - Sequential ``classifier`` (MobileNet, EfficientNet, ConvNeXt, SqueezeNet):
      modifies existing Dropout layers or inserts one before the final layer.
    - Direct ``fc`` Linear (ResNet, ShuffleNet): wraps in Sequential with
      Dropout.
    """
    if p < 0:
        return

    # --- Sequential classifier ---
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        found = False
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Dropout):
                model.classifier[i] = nn.Dropout(p=p)
                found = True
        if not found:
            layers = list(model.classifier.children())
            layers.insert(-1, nn.Dropout(p=p))
            model.classifier = nn.Sequential(*layers)
        return

    # --- Plain Linear fc (ResNet, ShuffleNet) ---
    if hasattr(model, "fc"):
        fc = model.fc
        if isinstance(fc, nn.Linear):
            model.fc = nn.Sequential(nn.Dropout(p=p), fc)
        elif isinstance(fc, nn.Sequential):
            found = False
            for i, layer in enumerate(fc):
                if isinstance(layer, nn.Dropout):
                    fc[i] = nn.Dropout(p=p)
                    found = True
            if not found:
                layers = list(fc.children())
                layers.insert(0, nn.Dropout(p=p))
                model.fc = nn.Sequential(*layers)


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


def _mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup: returns ``(mixed_x, y_a, y_b, lam)``."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def run_epoch(
    model, loader, criterion, optimizer, device, train: bool,
    *, mixup_alpha: float = 0.0,
) -> tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if train and mixup_alpha > 0:
                images, targets_a, targets_b, lam = _mixup_data(images, targets, mixup_alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
            else:
                targets_a = targets
                targets_b = targets
                lam = 1.0
                outputs = model(images)
                loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        if train and mixup_alpha > 0:
            running_correct += (
                lam * (preds == targets_a).sum().item()
                + (1.0 - lam) * (preds == targets_b).sum().item()
            )
        else:
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


class _PathEncoder(json.JSONEncoder):
    """JSON encoder that converts Path objects to strings."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_PathEncoder)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the final classifier head.

    Handles different head attribute names across architectures:
    - ``classifier`` (MobileNet, EfficientNet, ConvNeXt, SqueezeNet)
    - ``fc`` (ResNet, ShuffleNet)
    """
    head_names = {"classifier", "fc"}
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level not in head_names:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Cosine-annealing scheduler with linear warmup."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )
