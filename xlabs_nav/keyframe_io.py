"""Shared keyframe manifest + PNG I/O for recording (single source of truth)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_MANIFEST_RE = re.compile(r"kf_(\d+)\.png$", re.IGNORECASE)


def parse_kf_num_from_path(image_path: str) -> int | None:
    m = _MANIFEST_RE.search(image_path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def next_keyframe_index(entries: list[dict[str, Any]], keyframes_dir: Path) -> int:
    nums: list[int] = []
    for row in entries:
        if isinstance(row.get("id"), int):
            nums.append(int(row["id"]))
        ip = row.get("image_path")
        if isinstance(ip, str):
            n = parse_kf_num_from_path(ip)
            if n is not None:
                nums.append(n)
    if keyframes_dir.is_dir():
        for p in keyframes_dir.glob("kf_*.png"):
            m = re.match(r"kf_(\d+)\.png", p.name, re.IGNORECASE)
            if m:
                nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def load_keyframe_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON array: {path}")
    return data


def write_keyframe_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def append_keyframe_png(
    bgr: np.ndarray,
    *,
    manifest_path: Path,
    keyframes_dir: Path,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    """
    Write kf_NNN.png and append manifest row (id, image_path, sequence_index, description).
    Returns (new_entries_list, saved_path_or_none, error_or_none).
    """
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    next_idx = next_keyframe_index(entries, keyframes_dir)
    fname = f"kf_{next_idx:03d}.png"
    out_path = keyframes_dir / fname
    image_rel = f"data/keyframes/{fname}"
    if not cv2.imwrite(str(out_path), bgr):
        return entries, None, f"Failed to write {out_path}"
    new_row: dict[str, Any] = {
        "id": next_idx,
        "image_path": image_rel,
        "sequence_index": len(entries),
        "description": "",
    }
    updated = [*entries, new_row]
    write_keyframe_manifest(manifest_path, updated)
    return updated, out_path, None
