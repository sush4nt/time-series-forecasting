"""Step 3 — Temporal train / val / test split.

Chronological split (never shuffle time series):

    train : <= train_end                 (stockout rows dropped from training)
    val   : (train_end, val_end]         used for early stopping / tuning
    test  : (val_end,   test_end]        held out, reported once

Training drops censored (stockout) rows because their ``units_sold`` is
supply-capped and would teach the model a downward-biased demand. Val/test keep
all rows so evaluation reflects real operating conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitConfig
from .features import TARGET_COL

# Identifier/context columns carried alongside predictions for evaluation
# breakdowns and downstream failure-mode analysis (sparse SKUs, promo, cold start).
META_COLS = [
    "date", "store_id", "sku_id", "channel", "category", "subcategory", "brand",
    "country", "promo_flag", "stock_out_flag", "units_sold",
    "list_price", "purchase_cost", "margin_pct",
]


@dataclass
class Dataset:
    """A single split: feature matrix ``X``, target ``y``, and ``meta`` context."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    feature_cols: list[str]


def _subset(df: pd.DataFrame, mask: pd.Series, feature_cols: list[str]) -> Dataset:
    return Dataset(
        X=df.loc[mask, feature_cols].reset_index(drop=True),
        y=df.loc[mask, TARGET_COL].reset_index(drop=True),
        meta=df.loc[mask, META_COLS].reset_index(drop=True),
    )


def make_splits(
    df: pd.DataFrame, feature_cols: list[str], split_cfg: SplitConfig
) -> Splits:
    """Slice the feature frame into train/val/test :class:`Dataset` objects."""
    train_end = pd.Timestamp(split_cfg.train_end)
    val_end = pd.Timestamp(split_cfg.val_end)
    test_end = pd.Timestamp(split_cfg.test_end)

    train_mask = df["date"] <= train_end
    train_mask_clean = train_mask & (df["stock_out_flag"] == 0)  # drop censored
    val_mask = (df["date"] > train_end) & (df["date"] <= val_end)
    test_mask = (df["date"] > val_end) & (df["date"] <= test_end)

    return Splits(
        train=_subset(df, train_mask_clean, feature_cols),
        val=_subset(df, val_mask, feature_cols),
        test=_subset(df, test_mask, feature_cols),
        feature_cols=feature_cols,
    )
