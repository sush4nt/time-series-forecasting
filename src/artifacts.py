"""Shared run artifacts: seeding, versions, summary, and file persistence.

Both model families (classical trees, GRU) funnel through these helpers so every
run writes the *same* artifact shape regardless of backend. Model-specific extras
(feature importance, preprocessor, serialized model) are written by the backend.
"""

from __future__ import annotations

import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Keys inside a breakdown row that are metrics (everything else is the group label).
_METRIC_KEYS = {"n", "wape", "mae", "rmse", "bias", "actual_sum", "pred_sum"}
# Preferred order + which breakdowns to render in the human summary.
_BREAKDOWN_ORDER = ["by_stockout", "by_horizon", "by_promo", "by_channel", "by_category"]


def set_seed(seed: int, use_torch: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)


def versions(use_torch: bool = False) -> dict:
    import sklearn

    vers = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if use_torch:
        import torch

        vers["torch"] = torch.__version__
    else:
        for name in ("lightgbm", "xgboost", "catboost"):
            try:
                vers[name] = __import__(name).__version__
            except Exception:
                vers[name] = None
    return vers


def _row_label(row: dict) -> str:
    """A breakdown row's display label: prefer ``segment``, else the group column."""
    if "segment" in row:
        return str(row["segment"])
    for k, v in row.items():
        if k not in _METRIC_KEYS and k != "segment":
            return str(v)
    return ""


def _fmt_overall(tag: str, m: dict) -> str:
    return (
        f"{tag:<6} n={m['n']:>8,}  WAPE={m['wape']:.4f}  "
        f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  bias={m['bias']:+.3f}"
    )


def write_summary(path: Path, title: str, headers: dict, metrics: dict) -> None:
    """Generic human-readable summary; renders whatever breakdowns are present."""
    train, val, test = metrics["train"], metrics["val"], metrics["test"]
    lines = ["=" * 70, title, "=" * 70]
    for k, v in headers.items():
        lines.append(f"{k}: {v}")
    lines += [
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
    ]
    for key in _BREAKDOWN_ORDER:
        if key in test:
            lines += ["", f"TEST — {key.replace('by_', 'by ')}"]
            for r in test[key]:
                lines.append(
                    f"  {_row_label(r):<12} n={r['n']:>8,}  "
                    f"WAPE={r['wape']:.4f}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}"
                )
    if "business_proxy" in test:
        bp = test["business_proxy"]
        lines += [
            "",
            "TEST — business proxy",
            f"  overstock_cost            : {bp['overstock_cost']:>14,.0f}",
            f"  stockout_cost_lost_margin : {bp['stockout_cost_lost_margin']:>14,.0f}",
            f"  total_business_cost       : {bp['total_business_cost']:>14,.0f}",
        ]
    lines.append("")
    Path(path).write_text("\n".join(lines))


def write_breakdowns(run_dir: Path, test_metrics: dict) -> None:
    """One CSV per breakdown present in the test metrics (channel/category/…/horizon)."""
    for key in test_metrics:
        if key.startswith("by_"):
            name = key.replace("by_", "")
            pd.DataFrame(test_metrics[key]).to_csv(
                run_dir / f"breakdown_{name}_test.csv", index=False
            )


def write_predictions(run_dir: Path, frames: dict) -> None:
    """Persist per-row val/test prediction frames for failure-mode analysis."""
    for split in ("test", "val"):
        if split in frames:
            frames[split].to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
