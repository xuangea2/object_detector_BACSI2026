from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_videos(input_root: Path) -> list[dict]:
    sessions: list[dict] = []
    for class_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                session_id = f"{label}__{video_path.stem}"
                sessions.append(
                    {
                        "label": label,
                        "video_path": str(video_path.resolve()),
                        "video_name": video_path.name,
                        "session_id": session_id,
                    }
                )
    return sessions


def write_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_yaml(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_text_lines(lines: Iterable[str], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
