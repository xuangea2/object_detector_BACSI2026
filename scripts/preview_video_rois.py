#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

from hand_object_demo.io_utils import ensure_dir
from hand_object_demo.roi import crop_from_roi, compute_hand_object_roi, draw_hand_and_roi, landmarks_to_array


mp_hands = mp.solutions.hands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview MediaPipe hand ROI extraction on one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to a single input video.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save annotated preview frames.")
    parser.add_argument("--sample-every", type=int, default=15, help="Save one annotated frame every N frames.")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum number of frames to inspect.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        frame_idx = 0
        saved = 0
        pbar = tqdm(total=args.max_frames, desc="Preview")
        while frame_idx < args.max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % args.sample_every == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)
                vis = frame_bgr.copy()
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    pts = landmarks_to_array(hand_landmarks, vis.shape[1], vis.shape[0])
                    roi = compute_hand_object_roi(pts, vis.shape[1], vis.shape[0])
                    crop = crop_from_roi(frame_bgr, roi)
                    vis = draw_hand_and_roi(vis, pts, roi, label="ROI")
                    cv2.imwrite(str(args.output_dir / f"frame_{frame_idx:06d}_vis.jpg"), vis)
                    cv2.imwrite(str(args.output_dir / f"frame_{frame_idx:06d}_crop.jpg"), crop)
                else:
                    cv2.putText(
                        vis,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imwrite(str(args.output_dir / f"frame_{frame_idx:06d}_vis.jpg"), vis)
                saved += 1

            frame_idx += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    print(f"Saved {saved} preview steps to: {args.output_dir}")


if __name__ == "__main__":
    main()
