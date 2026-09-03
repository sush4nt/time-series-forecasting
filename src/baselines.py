"""Seasonal-naive baseline — the reference *floor* for the metric suite.

Every learned model should beat a trivial "carry a past observation forward" rule;
without that floor a WAPE of 0.24 is uninterpretable ("0.24 vs. *what*?"). This
module produces that floor and writes it in the **exact same artifact shape** as the
trees/GRU, so it drops straight into ``model_comparison`` alongside the real models.

Rule (horizon-safe seasonal naive)
----------------------------------
For each ``(store, sku, date)`` the prediction is the series' own ``units_sold`` from
``lag`` days earlier. To respect the **same** leakage constraint as the models — no
feature may read inside the 14-day forecast window — the lag must be a multiple of 7
(to land on the *same weekday*) and ``>= horizon``. The tightest such choice is
``lag = 14`` ("same weekday, two weeks ago"). A ``lag = 7`` naive would peek inside
the window for horizons 8..14, so it is *not* a fair floor and is deliberately avoided.

Run it:
    uv run python -m src.baselines                 # lag=14 (default)
    uv run python -m src.baselines --lag 28
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import artifacts
from .config import Config, load_config
from .data import SERIES_KEY, load_data
from .evaluate import evaluate_split
from .splits import META_COLS


def seasonal_naive_predictions(df: pd.DataFrame, lag: int) -> pd.Series:
    """Predict ``units_sold`` from ``lag`` rows (== days) earlier, per series.

    Rows are contiguous daily per series (see ``data.load_data``), so a positional
    ``shift(lag)`` equals a ``lag``-day calendar shift. The first ``lag`` rows of each
    series are NaN (no history) and are dropped from evaluation.
    """
    return df.groupby(SERIES_KEY)["units_sold"].shift(lag)


def run_naive(
    config_path: str | Path = "configs/baseline.yaml",
    lag: int = 14,
    run_name: str | None = None,
) -> Path:
    """Compute the seasonal-naive floor for train/val/test and persist artifacts."""
    cfg: Config = load_config(config_path)
    artifacts.set_seed(cfg.seed)
    horizon = cfg.features.horizon
    if lag < horizon or lag % 7 != 0:
        raise ValueError(
            f"lag={lag} is not a fair floor: it must be a multiple of 7 (same weekday) "
            f"and >= horizon ({horizon}). Use 14 or 28."
        )

    t0 = time.time()
    df = load_data(cfg.data_path)
    df["y_naive"] = seasonal_naive_predictions(df, lag)
    timings = {"load_s": round(time.time() - t0, 2)}

    train_end = pd.Timestamp(cfg.split.train_end)
    val_end = pd.Timestamp(cfg.split.val_end)
    test_end = pd.Timestamp(cfg.split.test_end)
    masks = {
        # Train mirrors the models: censored (stockout) rows dropped.
        "train": (df["date"] <= train_end) & (df["stock_out_flag"] == 0),
        "val": (df["date"] > train_end) & (df["date"] <= val_end),
        "test": (df["date"] > val_end) & (df["date"] <= test_end),
    }

    metrics_by_split: dict[str, dict] = {}
    frames: dict[str, object] = {}
    for name, mask in masks.items():
        sub = df.loc[mask & df["y_naive"].notna()]
        meta = sub[META_COLS].reset_index(drop=True)
        y_pred = sub["y_naive"].to_numpy()
        metrics_by_split[name], frames[name] = evaluate_split(meta, y_pred, name)
    timings["evaluate_s"] = round(time.time() - t0, 2)

    # ---- persist (same shape as a model run) ------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / (run_name or f"naive_seasonal_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    metrics = {
        "model": "naive_seasonal",
        "lag": lag,
        "train": metrics_by_split["train"],
        "val": metrics_by_split["val"],
        "test": metrics_by_split["test"],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "model": "naive_seasonal",
        "timestamp": stamp,
        "seed": cfg.seed,
        "lag": lag,
        "split": {
            "train_end": cfg.split.train_end,
            "val_end": cfg.split.val_end,
            "test_end": cfg.split.test_end,
        },
        "timings": timings,
        "device": "cpu",
        "versions": artifacts.versions(use_torch=False),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    artifacts.write_breakdowns(run_dir, metrics_by_split["test"])
    artifacts.write_predictions(run_dir, frames)
    artifacts.write_summary(
        run_dir / "metrics_summary.txt",
        f"Seasonal-naive baseline — same weekday, lag={lag}d",
        {"lag": lag},
        {k: metrics_by_split[k] for k in ("train", "val", "test")},
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal-naive baseline (metric floor)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--lag", type=int, default=14,
                        help="Same-weekday lag in days; multiple of 7 and >= horizon (14 or 28).")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_dir = run_naive(args.config, lag=args.lag, run_name=args.run_name)
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)["test"]
    ex = m["overall_ex_stockout"]
    print("\n" + "=" * 60)
    print(f"Done: naive_seasonal (lag={args.lag})")
    print(f"Artifacts -> {run_dir}")
    print(f"TEST (in-stock) WAPE={ex['wape']:.4f}  MAE={ex['mae']:.3f}  RMSE={ex['rmse']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
