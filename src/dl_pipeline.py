"""Part C — end-to-end orchestrator for the GRU encoder-decoder.

Mirrors the classical ``src.pipeline`` so both models emit the *same* artifact shape
and are scored by the *same* ``evaluate_split`` on the *same* splits:

    load -> window -> train (early-stop on real val WAPE) -> evaluate -> persist

Key fairness choices (see the Part C plan):
* early stopping uses the reshaped, inverse-transformed **val WAPE** (identical metric
  definition to the baseline), not the raw training loss;
* predictions are inverse-transformed to real ``units_sold`` before any metric;
* the in-stock view is the head-to-head number vs. the tree baseline;
* a per-horizon breakdown exposes the lead-time profile.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data import load_data
from .dl_dataset import (
    WindowDataset,
    build_dl_data,
    eval_frame,
)
from .dl_model import GRUEncoderDecoder
from .evaluate import evaluate_split
from .splits import META_COLS


@dataclass
class DLRunResult:
    model_name: str
    run_dir: Path
    metrics: dict


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _versions() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def _predict(model: nn.Module, ds: WindowDataset, batch_size: int, device: str) -> np.ndarray:
    """Forward the whole dataset in order, returning scaled predictions [n, H]."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            preds = model(
                b["enc"].to(device), b["fut"].to(device),
                b["scat"].to(device), b["snum"].to(device),
            )
            out.append(preds.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, ds.H))


def _to_units(preds_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert (log1p + standardise); demand is non-negative."""
    return np.clip(np.expm1(preds_scaled * std + mean), 0, None)


def _evaluate(
    data, series, samples, preds_scaled, df_meta: pd.DataFrame, split_name: str
) -> tuple[dict, pd.DataFrame]:
    long = eval_frame(series, samples, data.horizon)
    long["y_pred"] = _to_units(preds_scaled, data.target_mean, data.target_std).reshape(-1)
    merged = long.merge(df_meta, on=["store_id", "sku_id", "date"], how="left")
    y_pred = merged["y_pred"].to_numpy()
    return evaluate_split(merged.drop(columns=["y_pred"]), y_pred, split_name)


def _masked_huber(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = nn.functional.smooth_l1_loss(pred, y, reduction="none") * mask
    denom = mask.sum().clamp_min(1.0)
    return per.sum() / denom


def _fmt(tag: str, m: dict) -> str:
    return (
        f"{tag:<6} n={m['n']:>8,}  WAPE={m['wape']:.4f}  "
        f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  bias={m['bias']:+.3f}"
    )


def _write_summary(path: Path, train: dict, val: dict, test: dict, best_epoch: int) -> None:
    lines = [
        "=" * 70,
        "Part C — GRU encoder-decoder",
        "=" * 70,
        f"best_epoch: {best_epoch}",
        "",
        "OVERALL (all rows)",
        _fmt("train", train["overall"]),
        _fmt("val", val["overall"]),
        _fmt("test", test["overall"]),
        "",
        "OVERALL (in-stock rows only — head-to-head vs. baseline)",
        _fmt("train", train["overall_ex_stockout"]),
        _fmt("val", val["overall_ex_stockout"]),
        _fmt("test", test["overall_ex_stockout"]),
        "",
        "TEST — by horizon (lead time 1..H)",
    ]
    for r in test.get("by_horizon", []):
        lines.append(
            f"  h={r['horizon']:>2}  n={r['n']:>7,}  WAPE={r['wape']:.4f}  "
            f"MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )
    lines.append("")
    lines.append("TEST — by promo")
    for r in test["by_promo"]:
        lines.append(
            f"  {r['segment']:<10} n={r['n']:>8,}  "
            f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
        )
    path.write_text("\n".join(lines))


def run(
    config_path: str | Path,
    run_name: str | None = None,
    max_epochs: int | None = None,
    limit_series: int | None = None,
) -> DLRunResult:
    """Train the GRU and persist a self-contained artifact folder."""
    cfg: Config = load_config(config_path)
    dl = cfg.dl
    _set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = max_epochs if max_epochs is not None else dl.max_epochs

    timings: dict[str, float] = {}

    t = time.time()
    df = load_data(cfg.data_path)
    if limit_series is not None:
        keep = (
            df[["store_id", "sku_id"]].drop_duplicates().head(limit_series)
        )
        df = df.merge(keep, on=["store_id", "sku_id"], how="inner")
    timings["load_s"] = round(time.time() - t, 2)

    t = time.time()
    data = build_dl_data(df, dl, cfg.split, cfg.seed)
    df_meta = df[META_COLS].copy()
    timings["window_s"] = round(time.time() - t, 2)

    train_ds = WindowDataset(data.series, data.train_samples, data.encoder_len, data.horizon)
    val_ds = WindowDataset(data.series, data.val_samples, data.encoder_len, data.horizon)
    test_ds = WindowDataset(data.series, data.test_samples, data.encoder_len, data.horizon)

    model = GRUEncoderDecoder(
        cat_cardinalities=data.cat_cardinalities,
        n_enc_feat=data.n_enc_feat,
        n_fut_feat=data.n_fut_feat,
        n_stat_num=data.n_stat_num,
        hidden_size=dl.hidden_size,
        num_layers=dl.num_layers,
        dropout=dl.dropout,
        embedding_dim=dl.embedding_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=dl.lr)
    train_loader = DataLoader(train_ds, batch_size=dl.batch_size, shuffle=True)

    # --- train with early stopping on the real (reshaped) val WAPE ---------
    t = time.time()
    best_wape = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = dl.patience
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for b in train_loader:
            opt.zero_grad()
            pred = model(
                b["enc"].to(device), b["fut"].to(device),
                b["scat"].to(device), b["snum"].to(device),
            )
            loss = _masked_huber(pred, b["y"].to(device), b["mask"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), dl.grad_clip)
            opt.step()
            running += loss.item()

        val_preds = _predict(model, val_ds, dl.batch_size, device)
        val_metrics, _ = _evaluate(data, data.series, data.val_samples, val_preds, df_meta, "val")
        val_wape = val_metrics["overall_ex_stockout"]["wape"]
        print(
            f"epoch {epoch:>3}  train_loss={running / max(1, len(train_loader)):.4f}  "
            f"val_WAPE(in-stock)={val_wape:.4f}"
        )

        if val_wape < best_wape - 1e-5:
            best_wape, best_epoch = val_wape, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = dl.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    timings["train_s"] = round(time.time() - t, 2)

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- final evaluation --------------------------------------------------
    t = time.time()
    val_preds = _predict(model, val_ds, dl.batch_size, device)
    test_preds = _predict(model, test_ds, dl.batch_size, device)
    train_preds = _predict(model, train_ds, dl.batch_size, device)
    val_metrics, val_frame = _evaluate(data, data.series, data.val_samples, val_preds, df_meta, "val")
    test_metrics, test_frame = _evaluate(data, data.series, data.test_samples, test_preds, df_meta, "test")
    train_metrics, _ = _evaluate(data, data.series, data.train_samples, train_preds, df_meta, "train")
    timings["evaluate_s"] = round(time.time() - t, 2)

    # --- persist -----------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / (run_name or f"gru_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    metrics = {
        "model": "gru",
        "best_epoch": best_epoch,
        "encoder_len": data.encoder_len,
        "horizon": data.horizon,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "model": "gru",
        "timestamp": stamp,
        "seed": cfg.seed,
        "device": device,
        "dl_config": vars(dl),
        "split": {
            "train_end": cfg.split.train_end,
            "val_end": cfg.split.val_end,
            "test_end": cfg.split.test_end,
        },
        "sample_counts": {
            "train": len(data.train_samples),
            "val": len(data.val_samples),
            "test": len(data.test_samples),
        },
        "timings": timings,
        "versions": _versions(),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    for name in ("by_channel", "by_category", "by_promo", "by_stockout", "by_horizon"):
        if name in test_metrics:
            pd.DataFrame(test_metrics[name]).to_csv(
                run_dir / f"breakdown_{name.replace('by_', '')}_test.csv", index=False
            )

    test_frame.to_parquet(run_dir / "predictions_test.parquet", index=False)
    val_frame.to_parquet(run_dir / "predictions_val.parquet", index=False)

    torch.save(
        {"state_dict": model.state_dict(), "dl_config": vars(dl),
         "target_mean": data.target_mean, "target_std": data.target_std},
        run_dir / "model.pt",
    )

    _write_summary(run_dir / "metrics_summary.txt", train_metrics, val_metrics, test_metrics, best_epoch)

    return DLRunResult(model_name="gru", run_dir=run_dir, metrics=metrics)
