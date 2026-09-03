"""Unified run orchestrator.

One model-agnostic flow drives all backends:

    ingest -> validate -> prepare -> fit -> predict -> evaluate -> persist
       │         │          └── backend ──┘                │          │
       └──────── shared ─────┘                    shared evaluate  shared persist

The backend only supplies the three model-specific stages and its own artifacts;
everything else (seeding, evaluation, metrics/run_meta/summary/breakdowns/preds)
is shared here so classical and GRU runs are byte-for-byte consistent in shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import artifacts
from .backends import get_backend
from .config import Config, load_config
from .data import load_data
from .evaluate import evaluate_split


@dataclass
class RunResult:
    model_name: str
    run_dir: Path
    metrics: dict


def _limit_series(df, n: int):
    keep = df[["store_id", "sku_id"]].drop_duplicates().head(n)
    return df.merge(keep, on=["store_id", "sku_id"], how="inner")


def run(
    config_path: str | Path,
    model: str,
    run_name: str | None = None,
    max_epochs: int | None = None,
    limit_series: int | None = None,
) -> RunResult:
    """Execute the full pipeline for one model and persist all artifacts."""
    cfg: Config = load_config(config_path)
    backend = get_backend(model)
    artifacts.set_seed(cfg.seed, use_torch=backend.uses_torch)

    timings: dict[str, float] = {}

    # 1-2. Ingest + validate (gap check lives in load_data) -----------------
    t = time.time()
    df = load_data(cfg.data_path)
    if limit_series is not None:
        df = _limit_series(df, limit_series)
    timings["load_s"] = round(time.time() - t, 2)

    # 3. Prepare (model-specific) -------------------------------------------
    t = time.time()
    prepared = backend.prepare(df, cfg, cfg.seed)
    timings["prepare_s"] = round(time.time() - t, 2)

    # 4. Fit ----------------------------------------------------------------
    t = time.time()
    fitted = backend.fit(prepared, cfg, cfg.seed, max_epochs=max_epochs)
    timings["train_s"] = round(time.time() - t, 2)

    # 5-6. Predict + evaluate (shared evaluator) ----------------------------
    t = time.time()
    split_preds = backend.predict(fitted, prepared)
    metrics_by_split: dict[str, dict] = {}
    frames: dict[str, object] = {}
    for split_name, (meta, y_pred) in split_preds.items():
        metrics_by_split[split_name], frames[split_name] = evaluate_split(meta, y_pred, split_name)
    timings["evaluate_s"] = round(time.time() - t, 2)

    # 7. Persist (shared shape + backend extras) ----------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / (run_name or f"{model}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    model_info = backend.model_info(fitted, prepared)
    metrics = {
        "model": model,
        **model_info,
        "train": metrics_by_split["train"],
        "val": metrics_by_split["val"],
        "test": metrics_by_split["test"],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "model": model,
        "timestamp": stamp,
        "seed": cfg.seed,
        "split": {
            "train_end": cfg.split.train_end,
            "val_end": cfg.split.val_end,
            "test_end": cfg.split.test_end,
        },
        "timings": timings,
        "versions": artifacts.versions(use_torch=backend.uses_torch),
        **backend.run_meta_extra(fitted, prepared),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    artifacts.write_breakdowns(run_dir, metrics_by_split["test"])
    artifacts.write_predictions(run_dir, frames)

    headers = {k: v for k, v in model_info.items() if isinstance(v, (int, float, str))}
    artifacts.write_summary(
        run_dir / "metrics_summary.txt", backend.summary_title(), headers,
        {"train": metrics_by_split["train"], "val": metrics_by_split["val"], "test": metrics_by_split["test"]},
    )

    backend.save_model(fitted, prepared, run_dir)
    backend.extra_artifacts(fitted, prepared, run_dir)

    return RunResult(model_name=model, run_dir=run_dir, metrics=metrics)
