"""Step 4 — Model training (LightGBM / XGBoost / CatBoost).

One thin wrapper (:class:`TrainedModel`) hides the three libraries' APIs behind a
common ``predict`` / ``feature_importance`` / ``save`` interface, so the rest of
the pipeline is model-agnostic and candidates can be benchmarked apples-to-apples
on the same feature matrix. All three:

* train with early stopping on the validation split, and
* use the same integer/target-encoded feature matrix (no per-model feature
  differences), keeping the comparison fair.

Predictions are clipped at 0 (demand cannot be negative).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import SUPPORTED_MODELS
from .splits import Splits


@dataclass
class TrainedModel:
    """A fitted model plus everything evaluation/serialization needs."""

    name: str
    model: Any
    feature_cols: list[str]
    best_iteration: int | None
    _predict_fn: Callable[[pd.DataFrame], np.ndarray]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._predict_fn(X[self.feature_cols])
        return np.clip(preds, 0, None)  # demand is non-negative

    def feature_importance(self) -> pd.DataFrame:
        if self.name == "lightgbm":
            imp = self.model.feature_importances_
        elif self.name == "xgboost":
            imp = self.model.feature_importances_
        else:  # catboost
            imp = self.model.get_feature_importance()
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.name == "lightgbm":
            self.model.booster_.save_model(str(path.with_suffix(".txt")))
            return path.with_suffix(".txt")
        if self.name == "xgboost":
            self.model.save_model(str(path.with_suffix(".json")))
            return path.with_suffix(".json")
        self.model.save_model(str(path.with_suffix(".cbm")))  # catboost
        return path.with_suffix(".cbm")


def _train_lightgbm(params, splits, seed) -> TrainedModel:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(params)
    rounds = params.pop("early_stopping_rounds", 100)
    model = LGBMRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(
        splits.train.X, splits.train.y,
        eval_set=[(splits.val.X, splits.val.y)],
        eval_metric=params.get("metric", "mae"),
        callbacks=[early_stopping(rounds), log_evaluation(100)],
    )
    return TrainedModel(
        name="lightgbm",
        model=model,
        feature_cols=splits.feature_cols,
        best_iteration=getattr(model, "best_iteration_", None),
        _predict_fn=model.predict,
    )


def _train_xgboost(params, splits, seed) -> TrainedModel:
    from xgboost import XGBRegressor

    params = dict(params)
    rounds = params.pop("early_stopping_rounds", 100)
    model = XGBRegressor(
        random_state=seed, n_jobs=-1, early_stopping_rounds=rounds, **params
    )
    model.fit(
        splits.train.X, splits.train.y,
        eval_set=[(splits.val.X, splits.val.y)],
        verbose=100,
    )
    return TrainedModel(
        name="xgboost",
        model=model,
        feature_cols=splits.feature_cols,
        best_iteration=getattr(model, "best_iteration", None),
        _predict_fn=model.predict,
    )


def _train_catboost(params, splits, seed) -> TrainedModel:
    from catboost import CatBoostRegressor, Pool

    params = dict(params)
    rounds = params.pop("early_stopping_rounds", 100)
    model = CatBoostRegressor(
        random_seed=seed, early_stopping_rounds=rounds, verbose=100, **params
    )
    train_pool = Pool(splits.train.X, splits.train.y)
    val_pool = Pool(splits.val.X, splits.val.y)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return TrainedModel(
        name="catboost",
        model=model,
        feature_cols=splits.feature_cols,
        best_iteration=model.get_best_iteration(),
        _predict_fn=lambda X: model.predict(X),
    )


_TRAINERS = {
    "lightgbm": _train_lightgbm,
    "xgboost": _train_xgboost,
    "catboost": _train_catboost,
}


def train_model(
    model_name: str, params: dict[str, Any], splits: Splits, seed: int
) -> TrainedModel:
    """Train the requested model with early stopping on the validation split."""
    if model_name not in _TRAINERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of {SUPPORTED_MODELS}."
        )
    return _TRAINERS[model_name](params, splits, seed)
