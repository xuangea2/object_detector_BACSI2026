#!/usr/bin/env python3
"""Export a trained ``.pt`` checkpoint to TorchScript and ONNX.

Useful when you already have a trained checkpoint and want to (re-)export
it without retraining.  The script reads the checkpoint metadata (model
name, class names, args) to reconstruct the architecture automatically.

Examples
--------
Export the best checkpoint from the latest model run::

    python -m scripts.export_model data/models/train_resnet50_3/best.pt

Export with a specific image size::

    python -m scripts.export_model data/models/train_resnet50_3/best.pt --image-size 224
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from hand_object_demo.training import (
    create_model,
    export_model,
    set_head_mlp,
)


def _detect_mlp_hidden_sizes(state_dict: dict, head_prefix: str) -> tuple[int, ...]:
    """Infer MLP hidden-layer sizes from checkpoint state_dict keys.

    The MLP is stored as ``{head_prefix}.0.weight``, ``{head_prefix}.1.*``
    (BN), …  Linear layers at even-multiple-of-4 indices hold the hidden
    sizes in their ``out_features`` (= weight.shape[0]).
    """
    pattern = re.compile(rf"^{re.escape(head_prefix)}\.(\d+)\.weight$")
    linear_sizes: dict[int, int] = {}
    for key, val in state_dict.items():
        m = pattern.match(key)
        if m and val.dim() == 2:  # Linear weight is 2-D
            linear_sizes[int(m.group(1))] = val.shape[0]

    if len(linear_sizes) <= 1:
        return ()  # single Linear → not an MLP

    # All but the last Linear are hidden layers
    sorted_indices = sorted(linear_sizes)
    return tuple(linear_sizes[i] for i in sorted_indices[:-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained .pt checkpoint to TorchScript & ONNX.",
    )
    parser.add_argument(
        "checkpoint", type=Path,
        help="Path to the .pt checkpoint (e.g. data/models/train_resnet50_3/best.pt).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for exported files (default: same folder as checkpoint).",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Input image spatial size (default: 224).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output_dir = args.output_dir or args.checkpoint.parent
    prefix = args.checkpoint.stem  # e.g. "best" or "last"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_name: str = ckpt["model_name"]
    class_names: list[str] = ckpt["class_names"]
    num_classes = len(class_names)
    ckpt_args: dict = ckpt.get("args", {})
    sd = ckpt["model_state_dict"]

    print(f"Model            : {model_name}")
    print(f"Classes ({num_classes})     : {class_names}")

    # Reconstruct the architecture (no pretrained weights needed)
    model = create_model(model_name, num_classes=num_classes, pretrained=False)

    # Reconstruct MLP head if the checkpoint was trained with --head mlp
    head_type = ckpt_args.get("head", "linear")
    if head_type == "mlp":
        # Find the MLP prefix by looking for groups of 2-D (Linear) weights
        # under the head attribute.  E.g. "fc.0.weight", "fc.4.weight" → prefix "fc"
        # or "classifier.3.0.weight", "classifier.3.4.weight" → prefix "classifier.3"
        head_attr = "fc" if any(k.startswith("fc.") for k in sd) else "classifier"
        candidate_prefixes: set[str] = set()
        for k, v in sd.items():
            if k.startswith(f"{head_attr}.") and k.endswith(".weight") and v.dim() == 2:
                # strip the trailing ".N.weight" to get the MLP parent prefix
                prefix_candidate = k.rsplit(".", 2)[0]
                candidate_prefixes.add(prefix_candidate)

        # If the head_attr itself is a prefix (e.g. "fc"), include it
        for k, v in sd.items():
            if k.startswith(f"{head_attr}.") and k.endswith("weight") and v.dim() == 2:
                parts = k.split(".")
                if len(parts) == 2:  # e.g. "fc.weight" — single Linear, no MLP
                    candidate_prefixes.add(head_attr)
                    break

        # Pick the prefix that has the most Linear weights (that's the MLP)
        best_prefix = head_attr
        best_count = 0
        for cp in candidate_prefixes:
            hidden = _detect_mlp_hidden_sizes(sd, cp)
            count = len(hidden)
            if count > best_count:
                best_count = count
                best_prefix = cp
                best_hidden = hidden

        if best_count > 0:
            hidden = best_hidden
        else:
            hidden = (256,)  # fallback default

        print(f"Head             : mlp  hidden_sizes={hidden}")
        set_head_mlp(model, num_classes=num_classes, dropout=0.0, hidden_sizes=hidden)
    else:
        print("Head             : linear")

    model.load_state_dict(sd)

    print(f"\nExporting to     : {output_dir}")
    export_model(
        model,
        class_names=class_names,
        output_dir=output_dir,
        image_size=args.image_size,
        prefix=prefix,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
