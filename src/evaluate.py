"""Step 5 — Evaluation.

Reports the Part D metric suite:

* **Overall**: WAPE, MAE, RMSE (plus bias and per-split row counts).
* **Breakdowns**: by channel, category, and promo vs. non-promo.
* **Business proxy**: overstock cost (over-forecast tied up as inventory at
  ``purchase_cost``) and stockout / lost-margin cost (under-forecast valued at
  unit margin ``list_price * margin_pct``).

The main entry point, :func:`evaluate_split`, returns a metrics dict and an
enriched per-row prediction frame that downstream failure-mode analysis
(sparse SKUs, promo spikes, cold starts) can slice however it likes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|y - yhat| / sum|y|."""
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (yhat - y). Positive => over-forecast on average."""
    return float((y_pred - y_true).mean())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "actual_sum": float(y_true.sum()),
        "pred_sum": float(y_pred.sum()),
    }


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group WAPE/MAE/RMSE table, sorted by group size (descending)."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        m = compute_metrics(g["units_sold"].values, g["y_pred"].values)
        rows.append({by: key, **m})
    return (
        pd.DataFrame(rows)
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def promo_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    out = breakdown(df, "promo_flag")
    out["segment"] = out["promo_flag"].map({0: "non_promo", 1: "promo"})
    return out


# --------------------------------------------------------------------------- #
# Business proxy
# --------------------------------------------------------------------------- #
def business_proxy(df: pd.DataFrame) -> dict:
    """Turn forecast error into euro over/under-stock costs.

    * over-forecast  -> extra units ordered, capital tied up at ``purchase_cost``
    * under-forecast -> lost sales, valued at unit margin ``list_price * margin_pct``
    """
    error = df["y_pred"].values - df["units_sold"].values
    overstock_units = np.clip(error, 0, None)
    understock_units = np.clip(-error, 0, None)

    unit_margin = (df["list_price"] * df["margin_pct"]).values
    overstock_cost = float((overstock_units * df["purchase_cost"].values).sum())
    stockout_cost = float((understock_units * unit_margin).sum())

    return {
        "overstock_units": float(overstock_units.sum()),
        "understock_units": float(understock_units.sum()),
        "overstock_cost": overstock_cost,
        "stockout_cost_lost_margin": stockout_cost,
        "total_business_cost": overstock_cost + stockout_cost,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def build_eval_frame(meta: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Attach predictions (and signed error) to the split's context frame."""
    out = meta.copy()
    out["y_pred"] = y_pred
    out["error"] = out["y_pred"] - out["units_sold"]
    out["abs_error"] = out["error"].abs()
    return out


def evaluate_split(meta: pd.DataFrame, y_pred: np.ndarray, split_name: str) -> tuple[dict, pd.DataFrame]:
    """Full metric suite for one split. Returns ``(metrics_dict, eval_frame)``."""
    df = build_eval_frame(meta, y_pred)
    results = {
        "split": split_name,
        "overall": compute_metrics(df["units_sold"].values, df["y_pred"].values),
        "by_channel": breakdown(df, "channel").to_dict(orient="records"),
        "by_category": breakdown(df, "category").to_dict(orient="records"),
        "by_promo": promo_breakdown(df).to_dict(orient="records"),
        "business_proxy": business_proxy(df),
    }
    return results, df
