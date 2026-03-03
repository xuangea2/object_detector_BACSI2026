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

    val = max(2, int(round(total_videos * 0.15)))
    test = max(2, int(round(total_videos * 0.15)))
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
    Assign videos to train/val/test using a greedy strategy that tries to match:
    - global split sizes
    - class balance across splits
    - deterministic reproducibility with a seed

    This works even on tiny datasets where strict stratified splitting would fail.
    """
    rng = random.Random(seed)
    sessions = [dict(s) for s in sessions]
    rng.shuffle(sessions)

    total = len(sessions)
    targets = compute_split_targets(total)
    label_counts = Counter(s["label"] for s in sessions)

    desired_per_label_split: dict[str, dict[str, float]] = defaultdict(dict)
    for label, cnt in label_counts.items():
        desired_per_label_split[label] = {
            split: cnt * targets[split] / total if total > 0 else 0.0 for split in SPLITS
        }

    assigned_count = {split: 0 for split in SPLITS}
    assigned_label_count = {split: Counter() for split in SPLITS}

    sessions.sort(key=lambda s: (label_counts[s["label"]], s["label"], s["session_id"]))

    for session in sessions:
        label = session["label"]
        best_split = None
        best_score = None
        for split in SPLITS:
            if assigned_count[split] >= targets[split]:
                continue

            size_term = ((assigned_count[split] + 1) - targets[split]) ** 2
            desired = desired_per_label_split[label][split]
            label_term = ((assigned_label_count[split][label] + 1) - desired) ** 2

            # Slightly prefer filling train first for rare classes.
            train_bonus = -0.15 if split == "train" and assigned_label_count["train"][label] == 0 else 0.0
            score = size_term * 2.0 + label_term + train_bonus

            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        if best_split is None:
            remaining = [s for s in SPLITS if assigned_count[s] < targets[s]]
            best_split = remaining[0] if remaining else "train"

        session["split"] = best_split
        assigned_count[best_split] += 1
        assigned_label_count[best_split][label] += 1

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
