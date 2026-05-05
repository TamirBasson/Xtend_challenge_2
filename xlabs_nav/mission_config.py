from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MissionConfig:
    raw: dict[str, Any]
    repo_root: Path

    @property
    def keyframes_manifest(self) -> Path:
        rel = self.raw["keyframes"]["manifest_path"]
        return (self.repo_root / rel).resolve()

    @property
    def keyframes_base_dir(self) -> Path:
        rel = self.raw["keyframes"]["base_dir"]
        return (self.repo_root / rel).resolve()


def load_mission_config(config_path: str | Path) -> MissionConfig:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Mission config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Mission config must be a YAML mapping")
    root = path.parent.parent
    return MissionConfig(raw=data, repo_root=root)
