from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must parse to a dictionary.")
    return config


def resolve_repo_relative(path_value: str | Path, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    repo_root = config_path.resolve().parents[2]
    candidate = (repo_root / path).resolve()
    return candidate


def dump_config(config: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
