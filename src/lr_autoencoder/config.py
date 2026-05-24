from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_dir: str = "/path/to/your/train_data"
    val_dir: str | None = None
    image_size: int = 192
    val_fraction_if_no_val_dir: float = 0.1
    seed: int = 42


@dataclass
class TrainConfig:
    output_dir: str = "outputs/simple_lr_autoencoder"
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    in_channels: int = 4
    base_channels: int = 32
    amp: bool = True
    save_every: int = 1


@dataclass
class FullConfig:
    data: DataConfig
    train: TrainConfig


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    defaults = {
        "data": DataConfig().__dict__.copy(),
        "train": TrainConfig().__dict__.copy(),
    }
    merged = _deep_update(defaults, user_cfg)

    # Be tolerant to string placeholders from YAML edits.
    val_dir = merged.get("data", {}).get("val_dir")
    if isinstance(val_dir, str) and val_dir.strip().lower() in {"none", "null", "", "~"}:
        merged["data"]["val_dir"] = None

    return FullConfig(data=DataConfig(**merged["data"]), train=TrainConfig(**merged["train"]))