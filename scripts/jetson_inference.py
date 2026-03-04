#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference script for Jetson Nano (and any other device).

Supports three backends:

  1. TorchScript  (.torchscript)  — needs ``torch`` only
  2. ONNX         (.onnx)         — needs ``onnxruntime-gpu``
  3. TensorRT     (.engine/.trt)  — needs ``tensorrt`` + ``pycuda``

Usage examples
--------------
# TorchScript (simplest — PyTorch already ships with JetPack)
python jetson_inference.py best.torchscript class_names.json images/

# ONNX runtime
python jetson_inference.py best.onnx class_names.json images/

# TensorRT engine  (convert first: trtexec --onnx=best.onnx --saveEngine=best.engine --fp16)
python jetson_inference.py best.engine class_names.json images/

# Single image
python jetson_inference.py best.torchscript class_names.json photo.jpg

# Force CPU (for testing on a laptop without GPU)
python jetson_inference.py best.torchscript class_names.json images/ --device cpu

Compatible with Python >= 3.6 and old JetPack PyTorch builds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Image preprocessing (matches torchvision defaults used during training)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr, image_size=224):
    # type: (np.ndarray, int) -> np.ndarray
    """Resize + center-crop + normalise.  Returns float32 [1,3,H,W] in RGB."""
    h, w = image_bgr.shape[:2]

    # Resize so the short side == image_size * 1.14 (same as eval transforms)
    new_size = int(image_size * 1.14)
    if h < w:
        new_h, new_w = new_size, int(w * new_size / h)
    else:
        new_h, new_w = int(h * new_size / w), new_size
    img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    y0 = (new_h - image_size) // 2
    x0 = (new_w - image_size) // 2
    img = img[y0 : y0 + image_size, x0 : x0 + image_size]

    # BGR → RGB, uint8 → float32 [0,1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Normalise (ImageNet stats)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW, add batch dim → [1,3,H,W]
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# Backend 1 — TorchScript
# ---------------------------------------------------------------------------

class TorchScriptBackend:
    """Load a ``.torchscript`` file with ``torch.jit.load``."""

    def __init__(self, model_path, device="cuda"):
        # type: (str, str) -> None
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("[TorchScript] Loading {} …".format(model_path))
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("[TorchScript] Device: {}".format(self.device))

    def predict(self, input_np):
        # type: (np.ndarray) -> np.ndarray
        """Run inference. *input_np*: float32 [1,3,H,W].  Returns logits [1,C]."""
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(input_np).to(self.device)
            output = self.model(tensor)
            return output.cpu().numpy()


# ---------------------------------------------------------------------------
# Backend 2 — ONNX Runtime
# ---------------------------------------------------------------------------

class ONNXBackend:
    """Load ``.onnx`` with ``onnxruntime`` (GPU or CPU)."""

    def __init__(self, model_path, device="cuda"):
        # type: (str, str) -> None
        import onnxruntime as ort
        print("[ONNX] Loading {} …".format(model_path))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        active = self.session.get_providers()
        print("[ONNX] Active providers: {}".format(active))

    def predict(self, input_np):
        # type: (np.ndarray) -> np.ndarray
        output = self.session.run(None, {"input": input_np})
        return output[0]


# ---------------------------------------------------------------------------
# Backend 3 — TensorRT engine
# ---------------------------------------------------------------------------

class TensorRTBackend:
    """Load a serialised TensorRT ``.engine`` / ``.trt`` file.

    Convert first with::

        trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
    """

    def __init__(self, model_path, device="cuda"):
        # type: (str, str) -> None
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context

        self.cuda = cuda

        print("[TensorRT] Loading {} …".format(model_path))
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate device buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": dev_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": dev_mem, "shape": shape})

        print("[TensorRT] Ready.  Input shape: {}".format(self.inputs[0]["shape"]))

    def predict(self, input_np):
        # type: (np.ndarray) -> np.ndarray
        cuda = self.cuda
        np.copyto(self.inputs[0]["host"], input_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path, device="cuda"):
    """Auto-detect backend from file extension and return a backend object."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".torchscript":
        return TorchScriptBackend(model_path, device)
    elif ext == ".onnx":
        return ONNXBackend(model_path, device)
    elif ext in (".engine", ".trt"):
        return TensorRTBackend(model_path, device)
    else:
        # Try TorchScript as fallback
        print("[WARNING] Unknown extension '{}', trying TorchScript…".format(ext))
        return TorchScriptBackend(model_path, device)


def load_class_names(path):
    # type: (str) -> list
    with open(path, "r") as f:
        return json.load(f)


def softmax(logits):
    # type: (np.ndarray) -> np.ndarray
    """Numerically-stable softmax over the last axis."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def classify_image(backend, image_path, class_names, image_size=224):
    # type: (...) -> dict
    """Run inference on a single image.  Returns a result dict."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"image": image_path, "error": "could not read image"}

    input_np = preprocess(img_bgr, image_size)
    t0 = time.time()
    logits = backend.predict(input_np)
    dt = time.time() - t0

    probs = softmax(logits)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx])

    return {
        "image": os.path.basename(image_path),
        "label": label,
        "confidence": confidence,
        "time_ms": round(dt * 1000, 2),
        "all_probs": {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(path):
    # type: (str) -> list
    """Return a sorted list of image paths from *path* (file or directory)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if os.path.isfile(path):
        return [path]
    images = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(root, f))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model on the Jetson (or any device).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model", type=str,
        help="Path to model file (.torchscript | .onnx | .engine/.trt).",
    )
    parser.add_argument(
        "class_names", type=str,
        help="Path to class_names.json.",
    )
    parser.add_argument(
        "images", type=str,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Spatial size used during training (default: 224).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Show top-K predictions per image (default: 3).",
    )
    args = parser.parse_args()

    # --- Load model and class names ---
    class_names = load_class_names(args.class_names)
    backend = load_model(args.model, device=args.device)

    # --- Warmup (first inference is slow on GPU due to CUDA init) ---
    print("\nWarmup inference …")
    dummy = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)
    backend.predict(dummy)
    print("Ready.\n")

    # --- Collect images ---
    image_paths = collect_images(args.images)
    if not image_paths:
        print("No images found at '{}'.".format(args.images))
        sys.exit(1)
    print("Found {} image(s).\n".format(len(image_paths)))

    # --- Inference loop ---
    total_time = 0.0
    correct_header_printed = False
    results = []

    for img_path in image_paths:
        result = classify_image(backend, img_path, class_names, args.image_size)
        results.append(result)

        if "error" in result:
            print("  [SKIP] {} — {}".format(result["image"], result["error"]))
            continue

        if not correct_header_printed:
            print("{:<30s}  {:<8s}  {:>6s}  {:>8s}".format("Image", "Label", "Conf", "Time"))
            print("-" * 60)
            correct_header_printed = True

        print("{:<30s}  {:<8s}  {:>5.1f}%  {:>6.1f}ms".format(
            result["image"],
            result["label"],
            result["confidence"] * 100,
            result["time_ms"],
        ))

        # Show top-K
        if args.top_k > 1:
            sorted_probs = sorted(result["all_probs"].items(), key=lambda x: -x[1])
            for rank, (cls, prob) in enumerate(sorted_probs[:args.top_k], 1):
                bar = "#" * int(prob * 30)
                print("    {}. {:<6s} {:>5.1f}%  {}".format(rank, cls, prob * 100, bar))
            print()

        total_time += result["time_ms"]

    # --- Summary ---
    n = len([r for r in results if "error" not in r])
    if n > 0:
        print("-" * 60)
        print("Total: {} images | Avg: {:.1f} ms/image | {:.1f} FPS".format(
            n, total_time / n, 1000.0 * n / total_time if total_time > 0 else 0,
        ))


if __name__ == "__main__":
    main()
