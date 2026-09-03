"""Backend contract — the small interface every model family implements.

A backend isolates the three genuinely model-specific stages (prepare / fit /
predict) plus its own artifacts, and returns predictions as per-split
``(meta, y_pred)`` so the shared runner can score them with the *same*
``evaluate_split`` and persist the *same* artifact shape.

``Prepared`` and ``Fitted`` are deliberately opaque (``Any``): a tree backend
carries tabular ``Splits``; the GRU backend carries windowed datasets. Neither is
forced into the other's shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Per split name -> (context frame incl. units_sold [+ optional horizon], predictions).
SplitPredictions = dict[str, tuple[pd.DataFrame, np.ndarray]]


@runtime_checkable
class Backend(Protocol):
    name: str
    uses_torch: bool

    def summary_title(self) -> str:
        """Header line for metrics_summary.txt."""
        ...

    def prepare(self, df: pd.DataFrame, cfg: Any, seed: int) -> Any:
        """Stage 3: build model-specific inputs (features+splits OR windows)."""
        ...

    def fit(self, prepared: Any, cfg: Any, seed: int, max_epochs: int | None = None) -> Any:
        """Stage 4: train and return a fitted-model payload."""
        ...

    def predict(self, fitted: Any, prepared: Any) -> SplitPredictions:
        """Stage 5: predict each split, aligned to a context frame for evaluation."""
        ...

    def model_info(self, fitted: Any, prepared: Any) -> dict:
        """Extra top-level keys spread into metrics.json (params/best_iteration/…)."""
        ...

    def run_meta_extra(self, fitted: Any, prepared: Any) -> dict:
        """Extra keys spread into run_meta.json (feature_cols/dl_config/…)."""
        ...

    def save_model(self, fitted: Any, prepared: Any, run_dir: Path) -> None:
        """Serialize the model itself (model.txt/.json/.cbm or model.pt)."""
        ...

    def extra_artifacts(self, fitted: Any, prepared: Any, run_dir: Path) -> None:
        """Any other artifacts (feature_importance.csv, preprocessor.joblib)."""
        ...
