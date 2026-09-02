"""End-to-end orchestrator.

Wires the five steps together and writes a self-contained, reproducible artifact
folder per run:

    load -> features -> split -> train -> evaluate

Artifacts (under ``artifacts/<model>_<timestamp>/``):

    config.yaml               exact config used
    run_meta.json             seed, versions, timings, best_iteration, row counts
    metrics.json              overall + breakdowns + business proxy (train, val & test)
    metrics_summary.txt       human-readable summary
    feature_importance.csv    ranked feature importances
    breakdown_*_test.csv      per-channel / per-category / per-promo / per-stockout tables
    predictions_test.parquet  per-row test predictions (for failure-mode analysis)
    predictions_val.parquet   per-row val predictions
    model.{txt,json,cbm}      serialized model
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config, load_config
from .data import load_data
from .evaluate import evaluate_split
from .features import build_features
from .models import train_model
from .splits import make_splits


@dataclass
class RunResult:
    model_name: str
    run_dir: Path
    metrics: dict


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _versions() -> dict:
    import sklearn

    vers = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for name in ("lightgbm", "xgboost", "catboost"):
        try:
            vers[name] = __import__(name).__version__
        except Exception:
            vers[name] = None
    return vers


def _fmt_overall(tag: str, m: dict) -> str:
    return (
        f"{tag:<6} n={m['n']:>8,}  "
        f"WAPE={m['wape']:.4f}  MAE={m['mae']:.3f}  "
        f"RMSE={m['rmse']:.3f}  bias={m['bias']:+.3f}"
    )


def _write_summary(path: Path, model_name: str, train: dict, val: dict, test: dict, best_iter) -> None:
    lines = [
        "=" * 70,
        f"Part B baseline — {model_name}",
        "=" * 70,
        f"best_iteration: {best_iter}",
        "",
        "OVERALL (all rows)",
        _fmt_overall("train", train["overall"]),
        _fmt_overall("val", val["overall"]),
        _fmt_overall("test", test["overall"]),
        "",
        "OVERALL (in-stock rows only — excludes censored stockout days)",
        _fmt_overall("train", train["overall_ex_stockout"]),
        _fmt_overall("val", val["overall_ex_stockout"]),
        _fmt_overall("test", test["overall_ex_stockout"]),
        "",
        "TEST — by stockout",
    ]
    for r in test["by_stockout"]:
        lines.append(
            f"  {r['segment']:<10} n={r['n']:>8,}  "
            f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
            f"bias={r['bias']:+.3f}"
        )
    lines.append("")
    lines.append("TEST — by promo")
    for r in test["by_promo"]:
        lines.append(
            f"  {r['segment']:<10} n={r['n']:>8,}  "
            f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )
    lines.append("")
    lines.append("TEST — by channel")
    for r in test["by_channel"]:
        lines.append(
            f"  {r['channel']:<12} n={r['n']:>8,}  "
            f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )
    lines.append("")
    lines.append("TEST — by category")
    for r in test["by_category"]:
        lines.append(
            f"  {r['category']:<16} n={r['n']:>8,}  "
            f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )
    lines.append("")
    bp = test["business_proxy"]
    lines.append("TEST — business proxy")
    lines.append(f"  overstock_cost            : {bp['overstock_cost']:>14,.0f}")
    lines.append(f"  stockout_cost_lost_margin : {bp['stockout_cost_lost_margin']:>14,.0f}")
    lines.append(f"  total_business_cost       : {bp['total_business_cost']:>14,.0f}")
    lines.append("")
    path.write_text("\n".join(lines))


def run(
    config_path: str | Path,
    model_name: str,
    run_name: str | None = None,
) -> RunResult:
    """Execute the full pipeline for one model and persist all artifacts."""
    cfg: Config = load_config(config_path)
    _set_seed(cfg.seed)

    timings: dict[str, float] = {}

    # 1. Load ---------------------------------------------------------------
    t = time.time()
    df = load_data(cfg.data_path)
    timings["load_s"] = round(time.time() - t, 2)

    # 2. Features -----------------------------------------------------------
    t = time.time()
    df, feature_cols = build_features(df, cfg.features, cfg.split, cfg.seed)
    timings["features_s"] = round(time.time() - t, 2)

    # 3. Split --------------------------------------------------------------
    t = time.time()
    splits = make_splits(df, feature_cols, cfg.split)
    timings["split_s"] = round(time.time() - t, 2)

    # 4. Train --------------------------------------------------------------
    t = time.time()
    params = cfg.params_for(model_name)
    trained = train_model(model_name, params, splits, cfg.seed)
    timings["train_s"] = round(time.time() - t, 2)

    # 5. Evaluate -----------------------------------------------------------
    t = time.time()
    train_pred = trained.predict(splits.train.X)
    val_pred = trained.predict(splits.val.X)
    test_pred = trained.predict(splits.test.X)
    train_metrics, _ = evaluate_split(splits.train.meta, train_pred, "train")
    val_metrics, val_frame = evaluate_split(splits.val.meta, val_pred, "val")
    test_metrics, test_frame = evaluate_split(splits.test.meta, test_pred, "test")
    timings["evaluate_s"] = round(time.time() - t, 2)

    # --- Persist artifacts -------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / (run_name or f"{model_name}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config so the run is reproducible.
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    metrics = {
        "model": model_name,
        "params": params,
        "best_iteration": trained.best_iteration,
        "n_features": len(feature_cols),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "model": model_name,
        "timestamp": stamp,
        "seed": cfg.seed,
        "feature_cols": feature_cols,
        "split": {
            "train_end": cfg.split.train_end,
            "val_end": cfg.split.val_end,
            "test_end": cfg.split.test_end,
        },
        "row_counts": {
            "train": int(len(splits.train.y)),
            "val": int(len(splits.val.y)),
            "test": int(len(splits.test.y)),
        },
        "timings": timings,
        "versions": _versions(),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    trained.feature_importance().to_csv(run_dir / "feature_importance.csv", index=False)

    # Breakdown tables (test) as CSVs for quick eyeballing / benchmarking.
    pd.DataFrame(test_metrics["by_channel"]).to_csv(run_dir / "breakdown_channel_test.csv", index=False)
    pd.DataFrame(test_metrics["by_category"]).to_csv(run_dir / "breakdown_category_test.csv", index=False)
    pd.DataFrame(test_metrics["by_promo"]).to_csv(run_dir / "breakdown_promo_test.csv", index=False)
    pd.DataFrame(test_metrics["by_stockout"]).to_csv(run_dir / "breakdown_stockout_test.csv", index=False)

    # Per-row predictions for downstream failure-mode analysis.
    test_frame.to_parquet(run_dir / "predictions_test.parquet", index=False)
    val_frame.to_parquet(run_dir / "predictions_val.parquet", index=False)

    _write_summary(run_dir / "metrics_summary.txt", model_name, train_metrics, val_metrics,
                   test_metrics, trained.best_iteration)

    # Serialize the model itself.
    trained.save(run_dir / "model")

    return RunResult(model_name=model_name, run_dir=run_dir, metrics=metrics)
