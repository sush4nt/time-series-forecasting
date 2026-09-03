"""Classical (tree) backend: LightGBM / XGBoost / CatBoost.

Thin adapter over the existing ``features`` -> ``splits`` -> ``models`` modules.
Returns per-split ``(meta, y_pred)`` so the shared runner evaluates + persists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..features import build_features
from ..models import TrainedModel, train_model
from ..splits import Splits, make_splits
from .base import SplitPredictions


@dataclass
class ClassicalPrepared:
    splits: Splits
    feature_cols: list[str]


@dataclass
class ClassicalFitted:
    trained: TrainedModel
    params: dict


class ClassicalBackend:
    uses_torch = False

    def __init__(self, model_name: str):
        self.name = model_name

    def summary_title(self) -> str:
        return f"Part B baseline — {self.name}"

    def prepare(self, df: pd.DataFrame, cfg, seed: int) -> ClassicalPrepared:
        df, feature_cols = build_features(df, cfg.features, cfg.split, seed)
        splits = make_splits(df, feature_cols, cfg.split)
        return ClassicalPrepared(splits=splits, feature_cols=feature_cols)

    def fit(self, prepared: ClassicalPrepared, cfg, seed: int, max_epochs=None) -> ClassicalFitted:
        params = cfg.params_for(self.name)
        trained = train_model(self.name, params, prepared.splits, seed)
        return ClassicalFitted(trained=trained, params=params)

    def predict(self, fitted: ClassicalFitted, prepared: ClassicalPrepared) -> SplitPredictions:
        s = prepared.splits
        return {
            name: (ds.meta, fitted.trained.predict(ds.X))
            for name, ds in [("train", s.train), ("val", s.val), ("test", s.test)]
        }

    def model_info(self, fitted: ClassicalFitted, prepared: ClassicalPrepared) -> dict:
        return {
            "params": fitted.params,
            "best_iteration": fitted.trained.best_iteration,
            "n_features": len(prepared.feature_cols),
        }

    def run_meta_extra(self, fitted: ClassicalFitted, prepared: ClassicalPrepared) -> dict:
        s = prepared.splits
        return {
            "feature_cols": prepared.feature_cols,
            "row_counts": {
                "train": int(len(s.train.y)),
                "val": int(len(s.val.y)),
                "test": int(len(s.test.y)),
            },
        }

    def save_model(self, fitted: ClassicalFitted, prepared: ClassicalPrepared, run_dir: Path) -> None:
        fitted.trained.save(run_dir / "model")

    def extra_artifacts(self, fitted: ClassicalFitted, prepared: ClassicalPrepared, run_dir: Path) -> None:
        fitted.trained.feature_importance().to_csv(run_dir / "feature_importance.csv", index=False)
