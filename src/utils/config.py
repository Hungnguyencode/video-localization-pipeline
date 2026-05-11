from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_dir(path_value: str | Path) -> Path:
    path = resolve_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path