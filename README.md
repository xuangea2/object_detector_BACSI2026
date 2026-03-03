# Hand-held object demo: dataset builder + MobileNet training

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
5. Trains a MobileNet classifier on the extracted crops.

The split is done **by video**, never by frame, so adjacent frames from the same session do not leak across train / val / test.

## Project structure

```text
hand_object_demo_dataset_project/
├── README.md
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py
│   ├── preview_video_rois.py
│   ├── train_mobilenet.py
│   └── evaluate_classifier.py
└── src/
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

Accepted extensions:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.mpeg`
- `.mpg`

## Installation

Create a virtual environment with Python 3.12 and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Preview ROI detection on one video

Before generating the whole dataset, inspect how the hand ROI behaves on a single video:

```bash
python -m scripts.preview_video_rois \
  --video data/raw_videos/A/session_001.mp4 \
  --output-dir outputs/preview_A
```

This writes annotated frames to `outputs/preview_A/` so you can visually verify:

- hand detection
- landmarks
- ROI crop box
- crop thumbnails

## 2) Build the dataset from videos

```bash
python -m scripts.prepare_dataset --input-root data/raw_videos --output-root data/training/dataset_v1 
```

### What it creates

```text
data/training/dataset_v1/
├── images/
│   ├── train/
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   ├── D/
│   │   └── E/
│   ├── val/
│   └── test/
├── debug/
│   └── ... optional annotated frames ...
├── metadata/
│   ├── sessions.csv
│   ├── samples.csv
│   ├── split_summary.json
│   └── config.yaml
└── logs/
```

### Split logic

The split is computed dynamically from the total number of videos:

- target ratio: `70% / 15% / 15%`
- force at least `2` videos for validation and `2` videos for test **when the total number of videos makes that possible**
- always keep at least `1` video for train
- if the dataset is too small to satisfy that exactly, the code degrades gracefully and records the final split in `split_summary.json`

For your expected dataset size of about 20 videos, this will produce roughly:

- train: 14
- val: 3
- test: 3

with the exact allocation depending on rounding.

## 3) Train MobileNet

```bash
python -m scripts.train_mobilenet --dataset-root data/training/dataset_v1 --output-dir data/models/train_mobilenet_v1  --model mobilenet_v3_small --epochs 20 --batch-size 32 --lr 1e-3
```

```bash
python -m scripts.train_mobilenet --dataset-root data/training/dataset_v1 --output-dir data/models/train_mobilenet_large --model mobilenet_v3_large --epochs 25 --batch-size 32 --lr 5e-4
```

Outputs:

- best checkpoint
- last checkpoint
- class names
- training history CSV
- config JSON

## 4) Evaluate on test

```bash
python -m scripts.evaluate_classifier \
  --dataset-root data/training/dataset_v1 --checkpoint data/models/train_mobilenet_v1/best.pt --output-dir data/evals/eval_mobilenet_v1
```

## Notes on ROI generation

The ROI is derived from MediaPipe hand landmarks.

The current strategy is intentionally simple and robust:

1. detect one hand
2. compute a landmarks bounding box
3. estimate the palm direction from wrist to finger MCP center
4. shift the crop center slightly toward the fingers
5. enlarge the crop so it covers both the hand and the held object

This is usually a good starting point for hand-held object classification. If needed later, you can make the ROI more aggressive or use multiple candidate crops per frame.

## Recommended workflow

1. Start with a few videos per class.
2. Run `python -m scripts.preview_video_rois` on several sessions.
3. Adjust ROI parameters if needed.
4. Build the dataset.
5. Train MobileNet.
6. Inspect mistakes, especially:
   - hand-only negatives
   - severe occlusions
   - very small objects
   - unusual grips

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
