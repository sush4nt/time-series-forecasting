"""Step 2 — Feature engineering (leakage-aware).

Ports the Part A / EDA feature logic into a single reusable ``build_features``
call. Every feature obeys two rules:

* **Leakage-free**: statistics that "learn" from the target (target encoding,
  price medians, weather thresholds) are fitted on TRAIN rows only.
* **Horizon-safe**: for the fixed 14-day horizon, no feature may read a value
  from inside the forecast window. Point lags use ``k >= HORIZON`` and rolling
  windows are shifted by ``HORIZON`` (see ``add_lag_rolling_features``).

Feature families: calendar, lag/rolling demand, price/promo, weather, and
store/category/channel encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import FeatureConfig, SplitConfig
from .data import SERIES_KEY

TARGET_COL = "units_sold"

LABEL_COLS = [
    "store_id", "sku_id", "channel", "category",
    "subcategory", "brand", "country", "city",
]

TARGET_ENC_COLS = {
    "store_id": "store_target_enc",
    "sku_id": "sku_target_enc",
    "channel": "channel_target_enc",
    "category": "category_target_enc",
    "subcategory": "subcategory_target_enc",
}


# --------------------------------------------------------------------------- #
# Censored demand
# --------------------------------------------------------------------------- #
def add_censored_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Mask stockout days so they do not pollute lag/rolling demand features.

    When ``stock_out_flag == 1`` the observed ``units_sold`` is supply-capped,
    not true demand. We set ``demand_adj = NaN`` on those rows so that lags and
    rolling windows skip the censored value instead of propagating it.
    """
    df["demand_adj"] = df[TARGET_COL].where(df["stock_out_flag"] == 0, other=np.nan)
    return df


# --------------------------------------------------------------------------- #
# Calendar / holiday / weekend
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> list[str]:
    """Calendar features — all known in advance, so valid future covariates."""
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 53)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 53)

    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    # Country-aware holiday proximity (holidays can differ by country).
    hol = (
        df[["country", "date", "is_holiday"]]
        .drop_duplicates()
        .sort_values(["country", "date"])
        .copy()
    )
    hol["last_holiday_date"] = (
        hol["date"].where(hol["is_holiday"] == 1).groupby(hol["country"]).ffill()
    )
    hol["days_since_holiday"] = (hol["date"] - hol["last_holiday_date"]).dt.days
    hol["is_pre_holiday"] = (
        hol.groupby("country")["is_holiday"].shift(-1).fillna(0).astype(int)
    )
    df_merged = df.merge(
        hol[["country", "date", "days_since_holiday", "is_pre_holiday"]],
        on=["country", "date"],
        how="left",
    )
    df["days_since_holiday"] = df_merged["days_since_holiday"].fillna(999).values
    df["is_pre_holiday"] = df_merged["is_pre_holiday"].values

    return [
        "month", "day", "weekday", "weekofyear",
        "month_sin", "month_cos", "weekday_sin", "weekday_cos",
        "weekofyear_sin", "weekofyear_cos",
        "quarter", "is_month_end", "is_month_start",
        "is_holiday", "is_weekend", "is_pre_holiday", "days_since_holiday",
    ]


# --------------------------------------------------------------------------- #
# Lag & rolling demand
# --------------------------------------------------------------------------- #
def add_lag_rolling_features(df: pd.DataFrame, horizon: int) -> list[str]:
    """Point lags + rolling stats of censored-adjusted demand, horizon-safe.

    A feature is horizon-safe if, at every forecast step ``s`` in ``[1, horizon]``,
    it only reads data at or before the last known day ``t``:

    * ``lag_k`` is safe iff ``k >= horizon``  -> we keep lag_14, lag_28.
    * ``rolling_w`` shifted by ``horizon`` covers a fully historical window.
    """

    def _rolling(col: str, shift_n: int, window: int, agg: str, min_p: int | None = None):
        assert shift_n >= horizon, (
            f"shift_n={shift_n} exposes future data for a {horizon}-day horizon"
        )
        min_periods = min_p if min_p is not None else max(1, window // 2)
        fns = {
            "mean": lambda x: x.shift(shift_n).rolling(window, min_periods=min_periods).mean(),
            "std": lambda x: x.shift(shift_n).rolling(window, min_periods=min_periods).std(),
            "max": lambda x: x.shift(shift_n).rolling(window, min_periods=min_periods).max(),
            "min": lambda x: x.shift(shift_n).rolling(window, min_periods=min_periods).min(),
        }
        return df.groupby(SERIES_KEY)[col].transform(fns[agg])

    for lag in (14, 28):
        df[f"lag_{lag}"] = df.groupby(SERIES_KEY)["demand_adj"].transform(
            lambda x, l=lag: x.shift(l)
        )

    for window in (7, 14, 28):
        df[f"rolling_mean_{window}"] = _rolling("demand_adj", horizon, window, "mean")
        df[f"rolling_std_{window}"] = _rolling("demand_adj", horizon, window, "std")

    df["rolling_max_7"] = _rolling("demand_adj", horizon, 7, "max", min_p=4)
    df["rolling_min_7"] = _rolling("demand_adj", horizon, 7, "min", min_p=4)
    df["demand_trend_7v28"] = df["rolling_mean_7"] - df["rolling_mean_28"]

    return (
        [f"lag_{l}" for l in (14, 28)]
        + [f"rolling_mean_{w}" for w in (7, 14, 28)]
        + [f"rolling_std_{w}" for w in (7, 14, 28)]
        + ["rolling_max_7", "rolling_min_7", "demand_trend_7v28"]
    )


# --------------------------------------------------------------------------- #
# Price & promotion
# --------------------------------------------------------------------------- #
def add_price_promo_features(df: pd.DataFrame, train_mask: pd.Series) -> list[str]:
    """Price / promo dynamics. Promo calendars are known ahead -> future covariates."""
    df["effective_price"] = df["list_price"] * (1 - df["discount_pct"])

    # Reference price fitted on TRAIN only (compare net vs. net, per market).
    med = (
        df.loc[train_mask]
        .groupby(["sku_id", "country"])["effective_price"]
        .median()
        .rename("sku_country_median_price")
    )
    df_joined = df.join(med, on=["sku_id", "country"])
    df["price_vs_sku_median"] = (
        df["effective_price"] / df_joined["sku_country_median_price"]
    ).fillna(1.0)

    def days_since_promo(g):
        last_promo = g["date"].where(g["promo_flag"] == 1).shift(1).ffill()
        return (g["date"] - last_promo).dt.days

    df["days_since_promo"] = (
        df.groupby(SERIES_KEY, group_keys=False)[["date", "promo_flag"]]
        .apply(days_since_promo)
    )
    df["no_prior_promo"] = df["days_since_promo"].isna().astype(int)
    df["days_since_promo"] = df["days_since_promo"].fillna(999)

    for window in (7, 28):
        df[f"rolling_promo_count_{window}"] = df.groupby(SERIES_KEY)["promo_flag"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
    for lag in (1, 7):
        df[f"lag_promo_{lag}"] = df.groupby(SERIES_KEY)["promo_flag"].transform(
            lambda x, l=lag: x.shift(l)
        )

    return [
        "effective_price", "price_vs_sku_median", "list_price", "discount_pct",
        "promo_flag", "days_since_promo", "no_prior_promo",
        "rolling_promo_count_7", "rolling_promo_count_28",
        "lag_promo_1", "lag_promo_7",
    ]


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def add_weather_features(df: pd.DataFrame, train_mask: pd.Series, heavy_rain_q: float) -> list[str]:
    """Weather features; assumed known for the forecast date (planning input)."""
    df["rain_flag"] = (df["rain_mm"] > 0).astype(int)
    heavy_rain_thr = df.loc[train_mask, "rain_mm"].quantile(heavy_rain_q)  # train-only
    df["heavy_rain_flag"] = (df["rain_mm"] >= heavy_rain_thr).astype(int)
    df["temp_bin"] = pd.cut(
        df["temperature"],
        bins=[-np.inf, 7, 12, 17, 22, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)
    return ["temperature", "rain_mm", "rain_flag", "heavy_rain_flag", "temp_bin"]


# --------------------------------------------------------------------------- #
# Encodings
# --------------------------------------------------------------------------- #
def add_encodings(df: pd.DataFrame, train_mask_clean: pd.Series, smoothing: int) -> list[str]:
    """Label encodings (leakage-free) + smoothed target encodings (train-only means)."""
    for col in LABEL_COLS:
        df[f"{col}_enc"] = LabelEncoder().fit_transform(df[col])

    global_mean = df.loc[train_mask_clean, TARGET_COL].mean()

    def smoothed_target_encode(col: str, new_col: str) -> None:
        stats = df.loc[train_mask_clean].groupby(col)[TARGET_COL].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        df[new_col] = df[col].map(smooth).fillna(global_mean)

    for col, new_col in TARGET_ENC_COLS.items():
        smoothed_target_encode(col, new_col)

    # Series-level mean demand (store x SKU), smoothed, train-only.
    gstats = df.loc[train_mask_clean].groupby(SERIES_KEY)[TARGET_COL].agg(["mean", "count"])
    series_smooth = (gstats["mean"] * gstats["count"] + global_mean * smoothing) / (
        gstats["count"] + smoothing
    )
    df_joined = df.join(series_smooth.rename("series_mean_demand"), on=SERIES_KEY)
    df["series_mean_demand"] = df_joined["series_mean_demand"].fillna(global_mean).values

    static_numeric = ["latitude", "longitude"]

    return (
        [f"{c}_enc" for c in LABEL_COLS]
        + list(TARGET_ENC_COLS.values())
        + ["series_mean_demand"]
        + static_numeric
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame, feat_cfg: FeatureConfig, split_cfg: SplitConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Run all feature families and return ``(df_with_features, feature_cols)``.

    Train-only statistics use the same chronological TRAIN window as the final
    split, and target encodings additionally exclude stockout (censored) rows.
    """
    train_end = pd.Timestamp(split_cfg.train_end)
    train_mask = df["date"] <= train_end
    train_mask_clean = train_mask & (df["stock_out_flag"] == 0)

    df = add_censored_demand(df)

    cal = add_calendar_features(df)
    lag = add_lag_rolling_features(df, feat_cfg.horizon)
    promo = add_price_promo_features(df, train_mask)
    weather = add_weather_features(df, train_mask, feat_cfg.heavy_rain_quantile)
    enc = add_encodings(df, train_mask_clean, feat_cfg.target_smoothing)

    feature_cols = cal + lag + promo + weather + enc
    return df, feature_cols
