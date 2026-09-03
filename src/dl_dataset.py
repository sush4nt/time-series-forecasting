"""Part C — dataset construction for the GRU encoder-decoder.

Turns the flat daily table into sliding windows:

    encoder = last ``encoder_len`` days   ->   decoder = next ``horizon`` days

Each window (a "flashcard") carries three column roles:

* **static**       — constant per series (ids, channel, category, lat/long);
                     categoricals become embedding indices, numerics are scaled.
* **known-future** — available for the forecast days too (calendar, promo, price,
                     weather); fed to both encoder (past) and decoder (future).
* **observed**     — only known in the past (scaled demand history + stockout flag);
                     fed to the encoder only.

Leakage controls: scalers, target stats and categorical vocabularies are all fit on
TRAIN rows only. A window is assigned to a split by its forecast origin (the whole
14-day target window must fall inside that split's date range). Stockout target days
are masked out of the loss so a correct demand forecast is not punished for beating a
supply-capped actual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .config import DLConfig, SplitConfig
from .data import SERIES_KEY

TARGET_COL = "units_sold"

STATIC_CAT_COLS = [
    "store_id", "sku_id", "channel", "category",
    "subcategory", "brand", "country", "city",
]
STATIC_NUM_COLS = ["latitude", "longitude"]

# Known-future numeric inputs (planned/announced ahead, so valid for the horizon).
KNOWN_NUM_COLS = [
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "weekofyear_sin", "weekofyear_cos", "is_weekend", "is_holiday",
    "promo_flag", "discount_pct", "list_price", "temperature", "rain_mm",
]


def _add_calendar_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 53)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 53)
    return df


@dataclass
class DLData:
    """Everything the model + evaluation need, split-aware."""

    series: list[dict]
    train_samples: list[tuple[int, int]]
    val_samples: list[tuple[int, int]]
    test_samples: list[tuple[int, int]]
    cat_cardinalities: list[int]
    n_enc_feat: int
    n_fut_feat: int
    n_stat_num: int
    target_mean: float
    target_std: float
    encoder_len: int
    horizon: int


class WindowDataset(Dataset):
    """Serves ``(encoder, future, static, target, mask)`` tensors per window."""

    def __init__(self, series: list[dict], samples: list[tuple[int, int]], L: int, H: int):
        self.series = series
        self.samples = samples
        self.L = L
        self.H = H

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        sid, t = self.samples[i]
        d = self.series[sid]
        L, H = self.L, self.H
        enc_known = d["known"][t - L + 1 : t + 1]                 # [L, F_fut]
        enc_tgt = d["tgt_hist"][t - L + 1 : t + 1][:, None]        # [L, 1]
        enc_so = d["stockout"][t - L + 1 : t + 1][:, None]         # [L, 1]
        enc = np.concatenate([enc_tgt, enc_so, enc_known], axis=1)
        fut = d["known"][t + 1 : t + H + 1]                        # [H, F_fut]
        y = d["y_scaled"][t + 1 : t + H + 1]                       # [H]
        mask = d["mask"][t + 1 : t + H + 1]                        # [H]
        return {
            "enc": torch.from_numpy(enc).float(),
            "fut": torch.from_numpy(fut).float(),
            "scat": torch.from_numpy(d["scat"]).long(),
            "snum": torch.from_numpy(d["snum"]).float(),
            "y": torch.from_numpy(y).float(),
            "mask": torch.from_numpy(mask).float(),
            "idx": i,
        }


def eval_frame(series: list[dict], samples: list[tuple[int, int]], H: int) -> pd.DataFrame:
    """One row per (window, lead time) with keys to rejoin the full context frame."""
    store, sku, dates, horizons = [], [], [], []
    for sid, t in samples:
        d = series[sid]
        win_dates = d["dates"][t + 1 : t + H + 1]
        store.append(np.full(H, d["store_id"]))
        sku.append(np.full(H, d["sku_id"]))
        dates.append(win_dates)
        horizons.append(np.arange(1, H + 1))
    return pd.DataFrame({
        "store_id": np.concatenate(store),
        "sku_id": np.concatenate(sku),
        "date": np.concatenate(dates),
        "horizon": np.concatenate(horizons),
    })


def _fit_cat_vocab(values: pd.Series) -> dict:
    """Map train categories to indices 1..K; index 0 is reserved for unseen."""
    uniques = pd.unique(values)
    return {v: i + 1 for i, v in enumerate(uniques)}


def build_dl_data(df: pd.DataFrame, dl: DLConfig, split: SplitConfig, seed: int) -> DLData:
    """Build train/val/test window samples with train-only scalers and vocabs."""
    L, H = dl.encoder_len, dl.horizon
    df = df.sort_values(SERIES_KEY + ["date"]).reset_index(drop=True)
    df = _add_calendar_cyclic(df)

    train_end = pd.Timestamp(split.train_end)
    val_end = pd.Timestamp(split.val_end)
    test_end = pd.Timestamp(split.test_end)

    train_mask = df["date"] <= train_end
    train_clean = train_mask & (df["stock_out_flag"] == 0)

    # --- fit transforms on TRAIN only -------------------------------------
    known_scaler = StandardScaler().fit(df.loc[train_mask, KNOWN_NUM_COLS].to_numpy())
    stat_scaler = StandardScaler().fit(df.loc[train_mask, STATIC_NUM_COLS].to_numpy())

    tgt_log = np.log1p(df[TARGET_COL].to_numpy(dtype=float))
    train_tgt_log = np.log1p(df.loc[train_clean, TARGET_COL].to_numpy(dtype=float))
    tmean, tstd = float(train_tgt_log.mean()), float(train_tgt_log.std() or 1.0)

    vocabs = {c: _fit_cat_vocab(df.loc[train_mask, c]) for c in STATIC_CAT_COLS}
    cat_cardinalities = [len(vocabs[c]) + 1 for c in STATIC_CAT_COLS]

    # --- transform whole frame --------------------------------------------
    known_all = known_scaler.transform(df[KNOWN_NUM_COLS].to_numpy()).astype(np.float32)

    demand_adj = df[TARGET_COL].where(df["stock_out_flag"] == 0)
    tgt_hist_log = np.log1p(demand_adj.to_numpy(dtype=float))
    tgt_hist_log = np.where(np.isnan(tgt_hist_log), tmean, tgt_hist_log)
    tgt_hist = ((tgt_hist_log - tmean) / tstd).astype(np.float32)
    y_scaled = ((tgt_log - tmean) / tstd).astype(np.float32)
    loss_mask = (1 - df["stock_out_flag"].to_numpy(dtype=float)).astype(np.float32)
    stockout = df["stock_out_flag"].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()

    cat_codes = {
        c: df[c].map(vocabs[c]).fillna(0).to_numpy(dtype=np.int64)
        for c in STATIC_CAT_COLS
    }
    stat_num_all = stat_scaler.transform(df[STATIC_NUM_COLS].to_numpy()).astype(np.float32)

    # --- build per-series arrays + window samples -------------------------
    series: list[dict] = []
    train_s: list[tuple[int, int]] = []
    val_s: list[tuple[int, int]] = []
    test_s: list[tuple[int, int]] = []

    for _, idx in df.groupby(SERIES_KEY, sort=False).indices.items():
        idx = np.asarray(idx)
        sid = len(series)
        first = idx[0]
        series.append({
            "known": known_all[idx],
            "tgt_hist": tgt_hist[idx],
            "stockout": stockout[idx],
            "y_scaled": y_scaled[idx],
            "mask": loss_mask[idx],
            "dates": dates[idx],
            "scat": np.array([cat_codes[c][first] for c in STATIC_CAT_COLS], dtype=np.int64),
            "snum": stat_num_all[first],
            "store_id": df["store_id"].iat[first],
            "sku_id": df["sku_id"].iat[first],
        })

        s_dates = dates[idx]
        n = len(idx)
        tr, va, te = [], [], []
        for t in range(L - 1, n - H):
            tstart = s_dates[t + 1]
            tend = s_dates[t + H]
            if tend <= train_end:
                tr.append(t)
            elif tstart > train_end and tend <= val_end:
                va.append(t)
            elif tstart > val_end and tend <= test_end:
                te.append(t)

        train_s += [(sid, t) for t in tr[:: dl.train_stride]]
        val_s += [(sid, t) for t in va[::H]]     # tile: each val day scored once
        test_s += [(sid, t) for t in te[::H]]    # tile: each test day scored once

    return DLData(
        series=series,
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
        cat_cardinalities=cat_cardinalities,
        n_enc_feat=2 + len(KNOWN_NUM_COLS),
        n_fut_feat=len(KNOWN_NUM_COLS),
        n_stat_num=len(STATIC_NUM_COLS),
        target_mean=tmean,
        target_std=tstd,
        encoder_len=L,
        horizon=H,
    )
