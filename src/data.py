"""Step 1 — Load data.

Reads ``data.csv``, parses dates, and sorts by series + date so that all
downstream lag/rolling features are computed in the correct chronological order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Grain of a single time series: one (store, SKU) pair.
SERIES_KEY = ["store_id", "sku_id"]


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
    return df
