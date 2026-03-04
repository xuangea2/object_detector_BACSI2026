#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from hand_object_demo.io_utils import (
    ensure_dir,
    list_videos,
    next_versioned_dir,
    overwrite_latest_versioned_dir,
    write_json,
    write_yaml,
)
from hand_object_demo.roi import crop_from_roi, compute_hand_object_roi, draw_hand_and_roi, landmarks_to_array
from hand_object_demo.splitting import assign_splits, summarize_split


mp_hands = mp.solutions.hands

_TRAINING_ROOT = Path("data/training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hand-centered dataset from class-organized videos.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder with one subfolder per class.")
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Output dataset folder. Default: data/training/dataset_{next_version}.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Delete the latest dataset version and regenerate it instead of creating a new one.",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Approximate number of crops saved per second of video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-num-hands", type=int, default=1)
    parser.add_argument(
        "--save-debug-every",
        type=int,
        default=0,
        help="If > 0, save one annotated debug frame every N saved samples per video.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for stored crops, 0-100.",
    )
    return parser.parse_args()


def sample_frame_indices(num_frames: int, video_fps: float, target_fps: float) -> set[int]:
    if num_frames <= 0:
        return set()
    if video_fps <= 0 or not math.isfinite(video_fps):
        video_fps = 30.0
    step = max(video_fps / max(target_fps, 1e-6), 1.0)
    idx = 0.0
    out: set[int] = set()
    while int(round(idx)) < num_frames:
        out.add(int(round(idx)))
        idx += step
    return out


def process_video(session: dict, args: argparse.Namespace, hands, paths: dict[str, Path]) -> list[dict]:
    video_path = Path(session["video_path"])
    label = session["label"]
    split = session["split"]
    session_id = session["session_id"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_indices = sample_frame_indices(num_frames, video_fps, args.fps)

    rows: list[dict] = []
    image_dir = ensure_dir(paths["images"] / split / label)
    debug_dir = ensure_dir(paths["debug"] / split / label / session_id)

    frame_idx = 0
    saved = 0
    detected = 0
    with tqdm(total=max(num_frames, 0), desc=session_id, leave=False) as pbar:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx not in frame_indices:
                frame_idx += 1
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                pts = landmarks_to_array(hand_landmarks, frame_bgr.shape[1], frame_bgr.shape[0])
                roi = compute_hand_object_roi(pts, frame_bgr.shape[1], frame_bgr.shape[0])
                crop = crop_from_roi(frame_bgr, roi)
                if crop.size > 0 and roi.width >= 16 and roi.height >= 16:
                    filename = f"{session_id}__f{frame_idx:06d}.jpg"
                    image_path = image_dir / filename
                    cv2.imwrite(
                        str(image_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                    )
                    timestamp_sec = frame_idx / (video_fps if video_fps > 0 else 30.0)
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "session_id": session_id,
                            "video_path": str(video_path),
                            "video_name": video_path.name,
                            "frame_idx": frame_idx,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "image_path": str(image_path.resolve()),
                            "roi_x1": roi.x1,
                            "roi_y1": roi.y1,
                            "roi_x2": roi.x2,
                            "roi_y2": roi.y2,
                            "roi_w": roi.width,
                            "roi_h": roi.height,
                        }
                    )
                    if args.save_debug_every > 0 and saved % args.save_debug_every == 0:
                        vis = draw_hand_and_roi(frame_bgr, pts, roi, label=f"{label} | {split}")
                        cv2.imwrite(str(debug_dir / f"{session_id}__f{frame_idx:06d}_vis.jpg"), vis)
                    saved += 1
                    detected += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    session["num_saved_crops"] = saved
    session["num_detected_samples"] = detected
    session["num_frames"] = num_frames
    session["video_fps"] = video_fps
    return rows


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()

    # Resolve auto-versioned output root
    if args.output_root is not None:
        output_root = args.output_root.resolve()
    elif args.overwrite:
        output_root = overwrite_latest_versioned_dir(_TRAINING_ROOT, "dataset").resolve()
    else:
        output_root = next_versioned_dir(_TRAINING_ROOT, "dataset").resolve()

    print(f"Input root  : {input_root}")
    print(f"Output root : {output_root}")
    print()

    paths = {
        "images": ensure_dir(output_root / "images"),
        "debug": ensure_dir(output_root / "debug"),
        "metadata": ensure_dir(output_root / "metadata"),
        "logs": ensure_dir(output_root / "logs"),
    }

    sessions = list_videos(input_root)
    if not sessions:
        raise RuntimeError(f"No videos found under: {input_root}")

    sessions = assign_splits(sessions, seed=args.seed)
    split_summary = summarize_split(sessions)

    config = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "fps": args.fps,
        "seed": args.seed,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "max_num_hands": args.max_num_hands,
        "save_debug_every": args.save_debug_every,
        "jpeg_quality": args.jpeg_quality,
    }

    all_rows: list[dict] = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        for session in tqdm(sessions, desc="Videos"):
            rows = process_video(session, args, hands, paths)
            all_rows.extend(rows)

    sessions_df = pd.DataFrame(sessions)
    samples_df = pd.DataFrame(all_rows)

    sessions_df.to_csv(paths["metadata"] / "sessions.csv", index=False)
    samples_df.to_csv(paths["metadata"] / "samples.csv", index=False)
    write_json(split_summary, paths["metadata"] / "split_summary.json")
    write_yaml(config, paths["metadata"] / "config.yaml")

    per_split_counts = {}
    if not samples_df.empty:
        per_split_counts = (
            samples_df.groupby(["split", "label"]).size().rename("num_images").reset_index().to_dict(orient="records")
        )
    write_json({"samples_per_split_and_label": per_split_counts}, paths["metadata"] / "sample_summary.json")

    print("\nDone.")
    print(f"Sessions CSV: {paths['metadata'] / 'sessions.csv'}")
    print(f"Samples CSV:  {paths['metadata'] / 'samples.csv'}")
    print(f"Summary JSON: {paths['metadata'] / 'split_summary.json'}")


if __name__ == "__main__":
    main()
