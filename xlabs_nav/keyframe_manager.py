from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xlabs_nav.keyframe_prefilter import compute_global_descriptor
from xlabs_nav.mission_config import MissionConfig
from xlabs_nav.orb_matcher import extract_orb_features
from xlabs_nav.orb_matcher import extract_orb_features


class KeyframeManager:
    """Loads ordered keyframes from manifest JSON and tracks active index."""

    def __init__(self, mission: MissionConfig):
        manifest_path = mission.keyframes_manifest
        base = mission.keyframes_base_dir
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Keyframes manifest missing: {manifest_path}\n"
                "Record keyframes first or add keyframes.json under data/keyframes/."
            )
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError(f"Keyframes manifest must be a non-empty list: {manifest_path}")

        entries.sort(key=lambda e: int(e.get("sequence_index", e.get("id", 0))))

        # Global-descriptor pre-filter is read here so it can be cached once
        # per keyframe at load time. Falls back to a 32x32 thumbnail when the
        # config block is absent so older mission configs still work.
        ks_cfg = mission.raw.get("keyframe_selection") or {}
        gdesc_size = int(ks_cfg.get("global_descriptor_size", 32))

        self._gdesc_size = gdesc_size
        self._keyframes: list[dict[str, Any]] = []
        for row in entries:
            img_rel = row.get("image_path")
            if not img_rel:
                raise ValueError(f"Keyframe entry missing image_path: {row}")
            img_path = (base / img_rel).resolve() if not Path(img_rel).is_absolute() else Path(img_rel)
            if not img_path.is_file():
                raise FileNotFoundError(f"Keyframe image not found: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise RuntimeError(f"Failed to read keyframe image: {img_path}")
            orb_kp, orb_des, orb_hw = extract_orb_features(bgr, mission.raw)
            gdesc = compute_global_descriptor(bgr, gdesc_size)
            self._keyframes.append(
                {
                    **row,
                    "_path": str(img_path),
                    "_bgr": bgr,
                    "_orb_kp": orb_kp,
                    "_orb_des": orb_des,
                    "_orb_hw": orb_hw,
                    "_gdesc": gdesc,
                }
            )

        self._active_index = 0
        self._mission_complete = False

    @property
    def total(self) -> int:
        return len(self._keyframes)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def global_descriptor_size(self) -> int:
        """Side length used when computing cached keyframe global descriptors."""
        return self._gdesc_size

    def is_complete(self) -> bool:
        return self._mission_complete

    def get_active_keyframe(self) -> dict[str, Any] | None:
        if self._mission_complete or self._active_index >= len(self._keyframes):
            return None
        return self._keyframes[self._active_index]

    def set_active_index(self, index: int) -> None:
        """Jump active keyframe for re-localization (clamped to valid range)."""
        if len(self._keyframes) == 0:
            return
        self._active_index = max(0, min(int(index), len(self._keyframes) - 1))

    def get_keyframe_by_index(self, index: int) -> dict[str, Any] | None:
        if index < 0 or index >= len(self._keyframes):
            return None
        return self._keyframes[index]

    def mark_mission_complete(self) -> None:
        self._mission_complete = True

    def advance(self) -> None:
        if self._active_index + 1 < len(self._keyframes):
            self._active_index += 1
        else:
            self.mark_mission_complete()
