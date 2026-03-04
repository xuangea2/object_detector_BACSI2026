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


# ── Default augmentation parameters ──────────────────────────────────────────
# Collected in a dict so every value can be overridden from the CLI.

DEFAULT_AUGMENT: dict[str, Any] = {
    # Geometry
    "crop_scale_lo": 0.8,
    "crop_scale_hi": 1.0,
    "hflip": 0.5,
    "rotation": 180,
    "translate": 0.1,
    "shear": 8.0,
    "perspective": 0.15,
    "perspective_p": 0.3,
    # Color
    "brightness": 0.3,
    "contrast": 0.3,
    "saturation": 0.15,
    "hue": 0.02,
    "grayscale_p": 0.15,
    "equalize_p": 0.1,
    "autocontrast_p": 0.15,
    "sharpness_factor": 2.0,
    "sharpness_p": 0.2,
    # Blur
    "blur_sigma_lo": 0.1,
    "blur_sigma_hi": 1.0,
    "blur_p": 0.1,
    # Erasing
    "erasing_p": 0.15,
    "erasing_scale_lo": 0.02,
    "erasing_scale_hi": 0.15,
}


def build_transforms(
    image_size: int = 224,
    augment: dict[str, Any] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build train and eval transform pipelines.

    Parameters
    ----------
    image_size:
        Spatial resolution (default 224).
    augment:
        Dict overriding any key in ``DEFAULT_AUGMENT``.  Pass ``None`` to
        use the built-in defaults.  Keys with a probability of 0 disable
        the corresponding transform entirely.
    """
    a = {**DEFAULT_AUGMENT, **(augment or {})}

    train_layers: list = []

    # ── Geometry ─────────────────────────────────────────────────────────
    train_layers.append(
        transforms.RandomResizedCrop(image_size, scale=(a["crop_scale_lo"], a["crop_scale_hi"]))
    )
    if a["hflip"] > 0:
        train_layers.append(transforms.RandomHorizontalFlip(p=a["hflip"]))
    if a["rotation"] > 0:
        train_layers.append(transforms.RandomRotation(degrees=a["rotation"]))
    if a["translate"] > 0 or a["shear"] > 0:
        train_layers.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(a["translate"], a["translate"]) if a["translate"] > 0 else None,
                shear=a["shear"] if a["shear"] > 0 else None,
            )
        )
    if a["perspective_p"] > 0:
        train_layers.append(
            transforms.RandomPerspective(distortion_scale=a["perspective"], p=a["perspective_p"])
        )

    # ── Color ────────────────────────────────────────────────────────────
    if any(a[k] > 0 for k in ("brightness", "contrast", "saturation", "hue")):
        train_layers.append(
            transforms.ColorJitter(
                brightness=a["brightness"],
                contrast=a["contrast"],
                saturation=a["saturation"],
                hue=a["hue"],
            )
        )
    if a["grayscale_p"] > 0:
        train_layers.append(transforms.RandomGrayscale(p=a["grayscale_p"]))
    if a["equalize_p"] > 0:
        train_layers.append(transforms.RandomEqualize(p=a["equalize_p"]))
    if a["autocontrast_p"] > 0:
        train_layers.append(transforms.RandomAutocontrast(p=a["autocontrast_p"]))
    if a["sharpness_p"] > 0:
        train_layers.append(
            transforms.RandomAdjustSharpness(sharpness_factor=a["sharpness_factor"], p=a["sharpness_p"])
        )

    # ── Blur ─────────────────────────────────────────────────────────────
    if a["blur_p"] > 0:
        train_layers.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(a["blur_sigma_lo"], a["blur_sigma_hi"]))],
                p=a["blur_p"],
            )
        )

    # ── Tensor conversion + post-tensor augmentations ────────────────────
    train_layers.append(transforms.ToTensor())
    if a["erasing_p"] > 0:
        train_layers.append(
            transforms.RandomErasing(p=a["erasing_p"], scale=(a["erasing_scale_lo"], a["erasing_scale_hi"]))
        )
    train_layers.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    train_tfms = transforms.Compose(train_layers)
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


def set_head_mlp(
    model: nn.Module,
    num_classes: int,
    dropout: float = 0.4,
    hidden_sizes: tuple[int, ...] = (256,),
) -> None:
    """Replace the single classification layer with a multi-layer MLP.

    Call right after ``create_model`` (instead of ``set_head_dropout``).
    The MLP provides more capacity for learning non-linear decision boundaries
    when the backbone is kept frozen as a fixed feature extractor.

    Parameters
    ----------
    hidden_sizes:
        Sizes of hidden layers.  Default ``(256,)`` gives a single hidden
        layer;  ``(256, 128)`` gives the original two-hidden-layer design.
    """
    def _mlp(in_f: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_f
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        return nn.Sequential(*layers)

    # --- Sequential classifier (MobileNet, EfficientNet, ConvNeXt, SqueezeNet) ---
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for i in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[i]
            if isinstance(layer, nn.Linear):
                model.classifier[i] = _mlp(layer.in_features)
                return
            if isinstance(layer, nn.Conv2d):
                # SqueezeNet: replace Conv2d onward with pool → flatten → MLP
                pre = list(model.classifier.children())[:i]
                model.classifier = nn.Sequential(
                    *pre,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    *list(_mlp(layer.in_channels).children()),
                )
                return

    # --- Direct fc attribute (ResNet, ShuffleNet) ---
    if hasattr(model, "fc"):
        fc = model.fc
        if isinstance(fc, nn.Linear):
            model.fc = _mlp(fc.in_features)
            return
        if isinstance(fc, nn.Sequential):
            for layer in fc:
                if isinstance(layer, nn.Linear):
                    model.fc = _mlp(layer.in_features)
                    return


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    augment: dict[str, Any] | None = None,
) -> tuple[dict[str, CSVCropDataset], dict[str, DataLoader], list[str]]:
    samples_csv = dataset_root / "metadata" / "samples.csv"
    if not samples_csv.exists():
        raise FileNotFoundError(f"Missing samples CSV: {samples_csv}")

    df = pd.read_csv(samples_csv)
    if df.empty:
        raise RuntimeError("samples.csv is empty. Run prepare_dataset.py first.")

    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_tfms, eval_tfms = build_transforms(image_size=image_size, augment=augment)
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


def get_param_groups(
    model: nn.Module,
    lr: float,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 5e-4,
) -> list[dict[str, Any]]:
    """Create optimizer parameter groups with discriminative learning rates.

    Assigns ``lr * backbone_lr_factor`` to backbone parameters and ``lr``
    to the classification head.  This prevents the pre-trained backbone from
    changing too rapidly after unfreezing — the primary cause of overfitting
    in fine-tuning scenarios.

    Typical workflow::

        # Phase 1: head-only (backbone frozen)
        freeze_backbone(model)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        # Phase 2: full model with discriminative LR
        unfreeze_backbone(model)
        groups = get_param_groups(model, lr=3e-4, backbone_lr_factor=0.1)
        optimizer = AdamW(groups)  # backbone -> 3e-5, head -> 3e-4
    """
    head_names = {"classifier", "fc"}
    backbone_params: list[torch.Tensor] = []
    head_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top_level = name.split(".")[0]
        if top_level in head_names:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": lr * backbone_lr_factor,
            "weight_decay": weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    return groups


def export_model(
    model: nn.Module,
    class_names: list[str],
    output_dir: Path,
    image_size: int = 224,
    prefix: str = "best",
) -> dict[str, Path]:
    """Export a trained model to TorchScript and ONNX for deployment.

    Produces three files inside *output_dir*:

    - ``{prefix}.torchscript`` – TorchScript traced model (PyTorch-native,
      works across PyTorch versions).
    - ``{prefix}.onnx`` – ONNX graph (opset 11, compatible with TensorRT /
      onnxruntime on Jetson Nano and similar edge devices).
    - ``class_names.json`` – ordered list of class labels so the inference
      script can map argmax indices back to category names.

    Parameters
    ----------
    model:
        A trained ``nn.Module`` already in eval mode and on CPU.
    class_names:
        Ordered list of class labels (index 0 → first name, etc.).
    output_dir:
        Destination folder (created if needed).
    image_size:
        Spatial size used during training (default 224).
    prefix:
        Filename stem for the exported artefacts.

    Returns
    -------
    dict mapping format name to the written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.cpu()

    dummy = torch.randn(1, 3, image_size, image_size)
    paths: dict[str, Path] = {}

    # --- TorchScript (traced) ---
    ts_path = output_dir / f"{prefix}.torchscript"
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(str(ts_path))
        paths["torchscript"] = ts_path
        print(f"  ✔ TorchScript : {ts_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ TorchScript export failed: {exc}")

    # --- ONNX (opset 11 for broad Jetson / TensorRT compatibility) ---
    onnx_path = output_dir / f"{prefix}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            dynamo=False,  # legacy exporter → clean opset 11
        )
        paths["onnx"] = onnx_path
        print(f"  ✔ ONNX        : {onnx_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ ONNX export failed: {exc}")

    # --- Class-name mapping ---
    names_path = output_dir / "class_names.json"
    with names_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    paths["class_names"] = names_path
    print(f"  ✔ Class names  : {names_path}")

    return paths


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
