"""GRU backend: windowed sequence model.

Wraps ``dl_dataset`` (windowing + train-only transforms) and ``dl_model`` (the GRU
encoder-decoder), holds the training loop, and returns per-split ``(meta, y_pred)``
in real ``units_sold`` so the shared runner scores it with the same evaluator.

Flag-3 fix: the fitted preprocessor (scalers + vocabs + target log-stats) is saved
alongside the weights so inference can replay the exact training transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..dl_dataset import DLData, WindowDataset, build_dl_data, eval_frame
from ..dl_model import GRUEncoderDecoder
from ..evaluate import evaluate_split
from ..splits import META_COLS
from .base import SplitPredictions


@dataclass
class GRUPrepared:
    data: DLData
    df_meta: pd.DataFrame
    datasets: dict[str, WindowDataset]
    dl: object  # DLConfig


@dataclass
class GRUFitted:
    model: nn.Module
    best_epoch: int
    device: str


# --------------------------------------------------------------------------- #
# Helpers (shared by fit's early-stopping check and predict)
# --------------------------------------------------------------------------- #
def _predict_scaled(model: nn.Module, ds: WindowDataset, batch_size: int, device: str) -> np.ndarray:
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
    return np.clip(np.expm1(preds_scaled * std + mean), 0, None)


def _split_frame(prepared: GRUPrepared, samples, preds_scaled: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Reshape [n, H] preds into a per-(store, sku, date, horizon) context frame."""
    data = prepared.data
    long = eval_frame(data.series, samples, data.horizon)
    long["y_pred"] = _to_units(preds_scaled, data.target_mean, data.target_std).reshape(-1)
    merged = long.merge(prepared.df_meta, on=["store_id", "sku_id", "date"], how="left")
    return merged.drop(columns=["y_pred"]), merged["y_pred"].to_numpy()


def _masked_huber(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = nn.functional.smooth_l1_loss(pred, y, reduction="none") * mask
    return per.sum() / mask.sum().clamp_min(1.0)


class GRUBackend:
    uses_torch = True
    name = "gru"

    def summary_title(self) -> str:
        return "Part C — GRU encoder-decoder"

    def prepare(self, df: pd.DataFrame, cfg, seed: int) -> GRUPrepared:
        data = build_dl_data(df, cfg.dl, cfg.split, seed)
        datasets = {
            "train": WindowDataset(data.series, data.train_samples, data.encoder_len, data.horizon),
            "val": WindowDataset(data.series, data.val_samples, data.encoder_len, data.horizon),
            "test": WindowDataset(data.series, data.test_samples, data.encoder_len, data.horizon),
        }
        return GRUPrepared(data=data, df_meta=df[META_COLS].copy(), datasets=datasets, dl=cfg.dl)

    def fit(self, prepared: GRUPrepared, cfg, seed: int, max_epochs: int | None = None) -> GRUFitted:
        dl = prepared.dl
        data = prepared.data
        device = "cuda" if torch.cuda.is_available() else "cpu"
        epochs = max_epochs if max_epochs is not None else dl.max_epochs

        model = GRUEncoderDecoder(
            cat_cardinalities=data.cat_cardinalities,
            n_enc_feat=data.n_enc_feat, n_fut_feat=data.n_fut_feat, n_stat_num=data.n_stat_num,
            hidden_size=dl.hidden_size, num_layers=dl.num_layers,
            dropout=dl.dropout, embedding_dim=dl.embedding_dim,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=dl.lr)
        train_loader = DataLoader(prepared.datasets["train"], batch_size=dl.batch_size, shuffle=True)

        best_wape, best_epoch, best_state = float("inf"), -1, None
        patience_left = dl.patience
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for b in train_loader:
                opt.zero_grad()
                pred = model(b["enc"].to(device), b["fut"].to(device),
                             b["scat"].to(device), b["snum"].to(device))
                loss = _masked_huber(pred, b["y"].to(device), b["mask"].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), dl.grad_clip)
                opt.step()
                running += loss.item()

            val_scaled = _predict_scaled(model, prepared.datasets["val"], dl.batch_size, device)
            meta, y_pred = _split_frame(prepared, data.val_samples, val_scaled)
            val_metrics, _ = evaluate_split(meta, y_pred, "val")
            val_wape = val_metrics["overall_ex_stockout"]["wape"]
            print(f"epoch {epoch:>3}  train_loss={running / max(1, len(train_loader)):.4f}  "
                  f"val_WAPE(in-stock)={val_wape:.4f}")

            if val_wape < best_wape - 1e-5:
                best_wape, best_epoch = val_wape, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = dl.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return GRUFitted(model=model, best_epoch=best_epoch, device=device)

    def predict(self, fitted: GRUFitted, prepared: GRUPrepared) -> SplitPredictions:
        dl = prepared.dl
        samples = {"train": prepared.data.train_samples,
                   "val": prepared.data.val_samples,
                   "test": prepared.data.test_samples}
        out: SplitPredictions = {}
        for name, ds in prepared.datasets.items():
            scaled = _predict_scaled(fitted.model, ds, dl.batch_size, fitted.device)
            out[name] = _split_frame(prepared, samples[name], scaled)
        return out

    def model_info(self, fitted: GRUFitted, prepared: GRUPrepared) -> dict:
        return {
            "best_epoch": fitted.best_epoch,
            "encoder_len": prepared.data.encoder_len,
            "horizon": prepared.data.horizon,
        }

    def run_meta_extra(self, fitted: GRUFitted, prepared: GRUPrepared) -> dict:
        return {
            "device": fitted.device,
            "dl_config": vars(prepared.dl),
            "sample_counts": {
                "train": len(prepared.data.train_samples),
                "val": len(prepared.data.val_samples),
                "test": len(prepared.data.test_samples),
            },
        }

    def save_model(self, fitted: GRUFitted, prepared: GRUPrepared, run_dir: Path) -> None:
        torch.save(
            {"state_dict": fitted.model.state_dict(), "dl_config": vars(prepared.dl),
             "target_mean": prepared.data.target_mean, "target_std": prepared.data.target_std},
            run_dir / "model.pt",
        )

    def extra_artifacts(self, fitted: GRUFitted, prepared: GRUPrepared, run_dir: Path) -> None:
        # Persist fitted transforms so inference preprocesses identically to training.
        joblib.dump(prepared.data.preprocessor, run_dir / "preprocessor.joblib")
