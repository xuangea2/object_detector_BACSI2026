from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import yaml

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Versioned directory helpers
# ---------------------------------------------------------------------------

def _find_max_version(parent: Path, prefix: str) -> int:
    """Return the highest version number found under *parent* for dirs named
    ``{prefix}_{N}`` or ``{prefix}_v{N}`` (legacy), or 0 if none exist."""
    if not parent.is_dir():
        return 0
    pattern = re.compile(rf"^{re.escape(prefix)}_v?(\d+)$")
    versions = [
        int(m.group(1))
        for d in parent.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    return max(versions, default=0)


def next_versioned_dir(parent: Path, prefix: str) -> Path:
    """Return ``parent / {prefix}_{N+1}`` where N is the current max version."""
    v = _find_max_version(parent, prefix) + 1
    return parent / f"{prefix}_{v}"


def latest_versioned_dir(parent: Path, prefix: str) -> Path | None:
    """Return the latest ``parent / {prefix}_{N}`` or *None* if none exist."""
    v = _find_max_version(parent, prefix)
    if v == 0:
        return None
    return parent / f"{prefix}_{v}"


def overwrite_latest_versioned_dir(parent: Path, prefix: str) -> Path:
    """Delete the latest version (if any) and return its path so it can be
    re-created.  If no versions exist, return ``{prefix}_1``."""
    latest = latest_versioned_dir(parent, prefix)
    if latest is not None and latest.exists():
        shutil.rmtree(latest)
        return latest
    return parent / f"{prefix}_1"


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
