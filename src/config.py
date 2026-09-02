"""Configuration loading.

The whole pipeline is driven by a single YAML file (see ``configs/baseline.yaml``).
This module loads it, resolves paths relative to the project root, and exposes a
few small dataclasses so the rest of the code has typed, documented access to the
settings instead of passing raw dicts around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = one level above this file's parent (…/src/config.py -> project/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_MODELS = ("lightgbm", "xgboost", "catboost")


@dataclass
class SplitConfig:
    train_end: str
    val_end: str
    test_end: str


@dataclass
class FeatureConfig:
    horizon: int = 14
    target_smoothing: int = 20
    heavy_rain_quantile: float = 0.75


@dataclass
class Config:
    """Fully resolved pipeline configuration for a single run."""

    seed: int
    data_path: Path
    split: SplitConfig
    features: FeatureConfig
    model_params: dict[str, dict[str, Any]]
    output_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)

    def params_for(self, model_name: str) -> dict[str, Any]:
        """Hyper-parameters for the requested model (a shallow copy)."""
        if model_name not in self.model_params:
            raise KeyError(
                f"No hyper-parameters for '{model_name}' in config; "
                f"available: {sorted(self.model_params)}"
            )
        return dict(self.model_params[model_name])


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(config_path: str | Path) -> Config:
    """Read a YAML config file and return a validated :class:`Config`."""
    config_path = _resolve(str(config_path))
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return Config(
        seed=int(raw.get("seed", 42)),
        data_path=_resolve(raw["data"]["path"]),
        split=SplitConfig(**raw["split"]),
        features=FeatureConfig(**raw.get("features", {})),
        model_params=raw["model"],
        output_dir=_resolve(raw.get("output", {}).get("dir", "artifacts")),
        raw=raw,
    )
