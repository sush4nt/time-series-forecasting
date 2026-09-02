"""Step 1 — Load data.

Reads ``data.csv``, parses dates, and sorts by series + date so that all
downstream lag/rolling features are computed in the correct chronological order.

Assumption: each ``(store_id, sku_id)`` series has one contiguous row per
calendar day (lag/rolling features shift by *row position*, so gaps would break
the "k rows ago == k days ago" guarantee). We do not reindex; instead we log how
many within-series date gaps exist so the assumption can be verified at a glance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Grain of a single time series: one (store, SKU) pair.
SERIES_KEY = ["store_id", "sku_id"]


def _report_date_gaps(df: pd.DataFrame) -> None:
    """Print how many rows sit more than 1 day after their series predecessor."""
    day_diff = df.groupby(SERIES_KEY)["date"].diff().dt.days
    gap_rows = int((day_diff > 1).sum())
    total_missing_days = int((day_diff[day_diff > 1] - 1).sum())
    n_series = df.groupby(SERIES_KEY).ngroups
    print(
        f"[data] {n_series:,} series, {len(df):,} rows | "
        f"date-gap rows (diff>1d): {gap_rows:,} | "
        f"implied missing calendar days: {total_missing_days:,}"
    )


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw sales data, typed and sorted for time-series feature building.

    Parameters
    ----------
    path : str | Path
        Path to ``data.csv``.

    Returns
    -------
    pd.DataFrame
        Sorted by ``[store_id, sku_id, date]`` with a proper datetime ``date``.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(SERIES_KEY + ["date"]).reset_index(drop=True)
    _report_date_gaps(df)
    return df
