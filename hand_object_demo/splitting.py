from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any


SPLITS = ("train", "val", "test")


def compute_split_targets(total_videos: int) -> dict[str, int]:
    if total_videos <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if total_videos == 1:
        return {"train": 1, "val": 0, "test": 0}
    if total_videos == 2:
        return {"train": 1, "val": 1, "test": 0}
    if total_videos == 3:
        return {"train": 1, "val": 1, "test": 1}
    if total_videos == 4:
        return {"train": 2, "val": 1, "test": 1}

    val = max(2, int(round(total_videos * 0.05)))
    test = max(2, int(round(total_videos * 0.05)))
    train = total_videos - val - test

    if train < 1:
        deficit = 1 - train
        while deficit > 0 and (val > 1 or test > 1):
            if val >= test and val > 1:
                val -= 1
            elif test > 1:
                test -= 1
            deficit -= 1
        train = total_videos - val - test

    return {"train": train, "val": val, "test": test}


def assign_splits(sessions: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """
    Assign videos to train/val/test using **stratified splitting per category**.

    Each category is split independently so that every category is guaranteed
    to be represented in every split (as long as the category has at least 3
    videos).  Proportions are computed per-category using
    ``compute_split_targets``, which ensures a minimum of 1 video per split
    for small categories.

    This avoids the problem of categories being absent from val/test that can
    happen with a single global split.
    """
    rng = random.Random(seed)
    sessions = [dict(s) for s in sessions]

    # Group by label (category).
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_label[s["label"]].append(s)

    # For each label independently: shuffle, compute targets, assign splits.
    for label in sorted(by_label.keys()):
        group = by_label[label]
        rng.shuffle(group)
        targets = compute_split_targets(len(group))

        idx = 0
        for split in SPLITS:
            count = targets[split]
            for s in group[idx : idx + count]:
                s["split"] = split
            idx += count

    return sessions


def summarize_split(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(sessions)
    summary: dict[str, Any] = {
        "total_videos": total,
        "targets": compute_split_targets(total),
        "actual_counts": Counter(s["split"] for s in sessions),
        "class_distribution": {},
    }

    labels = sorted({s["label"] for s in sessions})
    for label in labels:
        label_sessions = [s for s in sessions if s["label"] == label]
        summary["class_distribution"][label] = {
            "total": len(label_sessions),
            **Counter(s["split"] for s in label_sessions),
        }
    return summary
