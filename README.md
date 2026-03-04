# Hand-held object demo: dataset builder + classifier training

This project builds a hand-centered image dataset from class-organized videos and trains a lightweight classifier for a real-time edge demo.

## Goal

You have videos organized by category:

- `A/`
- `B/`
- `C/`
- `D/`
- `E/` (`E` = no object / unknown / negative class)

Each video contains a single hand. The pipeline:

1. Runs MediaPipe Hands on the videos.
2. Computes an object ROI from the detected hand landmarks.
3. Saves crops at about 1 fps.
4. Splits videos by session into train / val / test.
5. Trains a classifier on the extracted crops.

The split is done **by video**, never by frame, so adjacent frames from the same session do not leak across train / val / test.

## Project structure

```text
object_detector_BACSI2026/
├── README.md
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py
│   ├── preview_video_rois.py
│   ├── train_mobilenet.py
│   └── evaluate_classifier.py
└── hand_object_demo/
    ├── __init__.py
    ├── io_utils.py
    ├── roi.py
    ├── splitting.py
    └── training.py
```

## Expected input layout

Put the videos under a root folder, one folder per class:

```text
data/raw_videos/
├── A/
│   ├── session_001.mp4
│   ├── session_002.mp4
│   └── ...
├── B/
├── C/
├── D/
└── E/
```

Accepted extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.mpeg`, `.mpg`

## Installation

Create a virtual environment with Python 3.12 and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1) Preview ROI detection on one video

Before generating the whole dataset, inspect how the hand ROI behaves on a single video:

```bash
python -m scripts.preview_video_rois \
  --video data/raw_videos/A/session_001.mp4 \
  --output-dir outputs/preview_A
```

---

## 2) Build the dataset from videos

### Auto-versioned (recommended)

```bash
# Creates data/training/dataset_1  (first run)
# Creates data/training/dataset_2  (second run)
python -m scripts.prepare_dataset --input-root data/raw_videos
```

### Overwrite the latest version

```bash
# Deletes data/training/dataset_2 and recreates it
python -m scripts.prepare_dataset --input-root data/raw_videos --overwrite
```

### Explicit output path

```bash
python -m scripts.prepare_dataset --input-root data/raw_videos --output-root data/training/my_custom_dataset
```

The script prints the destination directory before starting.

### What it creates

```text
data/training/dataset_N/
├── images/
│   ├── train/{A,B,C,D,E}/
│   ├── val/{A,B,C,D,E}/
│   └── test/{A,B,C,D,E}/
├── debug/
├── metadata/
│   ├── sessions.csv
│   ├── samples.csv
│   ├── split_summary.json
│   └── config.yaml
└── logs/
```

### Split logic

- target ratio: `90% / 5% / 5%`
- at least 2 videos for validation and 2 for test when possible
- always at least 1 video for train

---

## 3) Train a classifier

### Auto-versioned (recommended)

Both `--dataset-root` and `--output-dir` are resolved automatically:

```bash
# Uses latest dataset, saves to data/models/train_mobilenet_v3_small_1/
python -m scripts.train_mobilenet --model mobilenet_v3_small --epochs 25
```

Run it again and it creates `train_mobilenet_v3_small_2/`, etc.

### Explicit paths

```bash
python -m scripts.train_mobilenet \
  --dataset-root data/training/dataset_1 \
  --output-dir data/models/my_experiment \
  --model efficientnet_b0 --epochs 30
```

ImageNet-pretrained weights are used by default. Pass `--no-pretrained` to train from scratch (not recommended for small datasets).

### Supported models

| Model | Params | Speed | Notes |
|---|---|---|---|
| `mobilenet_v3_small` | ~2.5 M | ★★★★★ | Default. Very fast, edge-friendly |
| `mobilenet_v3_large` | ~5.4 M | ★★★★ | Stronger than small |
| `shufflenet_v2` | ~2.3 M | ★★★★★ | Fastest, very small |
| `squeezenet` | ~1.2 M | ★★★★★ | Tiny footprint |
| `efficientnet_b0` | ~5.3 M | ★★★★ | Best accuracy/size trade-off |
| `efficientnet_v2_s` | ~21 M | ★★★ | Fast training, strong accuracy |
| `resnet18` | ~11 M | ★★★★ | Lightweight classic |
| `resnet50` | ~25 M | ★★★ | Strong baseline |
| `convnext_tiny` | ~28 M | ★★★ | Modern "ConvNet strikes back" |
| `convnext_small` | ~50 M | ★★ | Highest capacity |

### Quick examples for each model

```bash
# ── Lightweight / edge ──────────────────────────────────────────────────
python -m scripts.train_mobilenet --model mobilenet_v3_small --epochs 25
python -m scripts.train_mobilenet --model mobilenet_v3_large --epochs 25
python -m scripts.train_mobilenet --model shufflenet_v2      --epochs 25
python -m scripts.train_mobilenet --model squeezenet          --epochs 30

# ── Best accuracy/size balance ──────────────────────────────────────────
python -m scripts.train_mobilenet --model efficientnet_b0     --epochs 30
python -m scripts.train_mobilenet --model efficientnet_v2_s   --epochs 25

# ── Classic / strong baselines ──────────────────────────────────────────
python -m scripts.train_mobilenet --model resnet18            --epochs 25
python -m scripts.train_mobilenet --model resnet50            --epochs 25

# ── Modern high-capacity ────────────────────────────────────────────────
python -m scripts.train_mobilenet --model convnext_tiny       --epochs 25
python -m scripts.train_mobilenet --model convnext_small      --epochs 25 --lr 1e-4
```

Each run auto-creates its output in `data/models/train_{model}_{version}/`.

### Training features

- **Backbone freezing**: First 3 epochs only train the classifier head (`--freeze-epochs`)
- **LR scheduler**: Linear warmup + cosine annealing (`--warmup-epochs`)
- **Label smoothing**: 0.1 by default (`--label-smoothing`)
- **Pretrained**: ImageNet weights by default (`--pretrained` / `--no-pretrained`)

### Outputs

- `best.pt` — best validation accuracy checkpoint
- `last.pt` — last epoch checkpoint
- `history.csv` — per-epoch metrics
- `training_config.json` — full config + class names

---

## 4) Evaluate on test

```bash
python -m scripts.evaluate_classifier \
  --dataset-root data/training/dataset_1 \
  --checkpoint data/models/train_mobilenet_v3_small_1/best.pt \
  --output-dir data/evals/eval_mobilenet_v3_small_1
```

---

## Notes on ROI generation

The ROI is derived from MediaPipe hand landmarks:

1. Detect one hand
2. Compute a landmarks bounding box
3. Estimate the palm direction from wrist to finger MCP center
4. Shift the crop center slightly toward the fingers
5. Enlarge the crop so it covers both the hand and the held object

## Recommended workflow

1. Start with a few videos per class
2. Run `python -m scripts.preview_video_rois` on several sessions
3. Adjust ROI parameters if needed
4. Build the dataset
5. Train several models and compare
6. Inspect mistakes (hand-only negatives, occlusions, small objects, unusual grips)

## Practical tips for recording videos

For each category, record multiple sessions with:

- different users / hands
- different backgrounds
- different distances
- different orientations
- partial occlusions
- varied lighting

For class `E`, include:

- empty hand
- gestures without object
- unrelated objects
- confusing hand poses
