from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


def landmarks_to_array(landmarks, image_width: int, image_height: int) -> np.ndarray:
    pts = []
    for lm in landmarks.landmark:
        x = np.clip(lm.x * image_width, 0, image_width - 1)
        y = np.clip(lm.y * image_height, 0, image_height - 1)
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


def compute_hand_object_roi(
    pts: np.ndarray,
    image_width: int,
    image_height: int,
    scale: float = 2.2,
    shift_along_palm: float = 0.35,
    min_box_frac: float = 0.15,
) -> ROIBox:
    """
    Compute an axis-aligned ROI intended to cover the hand and the held object.

    Strategy:
    - start from the landmarks bounding box
    - estimate palm direction from wrist to average MCP center
    - shift the crop center slightly toward the fingers
    - enlarge the crop so it remains usable at different distances
    """
    if pts.shape != (21, 2):
        raise ValueError(f"Expected hand landmarks with shape (21, 2), got {pts.shape}")

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_w = max(1.0, float(x_max - x_min))
    bbox_h = max(1.0, float(y_max - y_min))
    bbox_center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)

    wrist = pts[0]
    mcp_center = pts[[5, 9, 13, 17]].mean(axis=0)
    palm_vec = mcp_center - wrist
    palm_norm = float(np.linalg.norm(palm_vec))
    if palm_norm < 1e-6:
        palm_dir = np.array([0.0, -1.0], dtype=np.float32)
    else:
        palm_dir = palm_vec / palm_norm

    radial_extent = float(np.max(np.linalg.norm(pts - bbox_center[None, :], axis=1)))
    side = max(bbox_w, bbox_h, radial_extent * 2.4)
    side *= scale
    side = max(side, min(image_width, image_height) * min_box_frac)

    shifted_center = bbox_center + palm_dir * max(bbox_w, bbox_h) * shift_along_palm
    half_side = side * 0.5

    x1 = int(round(shifted_center[0] - half_side))
    y1 = int(round(shifted_center[1] - half_side))
    x2 = int(round(shifted_center[0] + half_side))
    y2 = int(round(shifted_center[1] + half_side))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    if x2 <= x1 or y2 <= y1:
        side = max(64.0, min(image_width, image_height) * 0.2)
        cx, cy = bbox_center
        x1 = max(0, int(round(cx - side / 2)))
        y1 = max(0, int(round(cy - side / 2)))
        x2 = min(image_width, int(round(cx + side / 2)))
        y2 = min(image_height, int(round(cy + side / 2)))

    return ROIBox(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_from_roi(image: np.ndarray, roi: ROIBox) -> np.ndarray:
    return image[roi.y1 : roi.y2, roi.x1 : roi.x2].copy()


def draw_hand_and_roi(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    roi: ROIBox,
    label: str | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    for x, y in pts.astype(int):
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.rectangle(out, (roi.x1, roi.y1), (roi.x2, roi.y2), (0, 0, 255), 2)
    if label:
        cv2.putText(
            out,
            label,
            (roi.x1, max(20, roi.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out
