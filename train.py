"""CLI entry point for the Part B classical ML baseline pipeline.

Examples
--------
    uv run python train.py --model lightgbm
    uv run python train.py --model xgboost  --config configs/baseline.yaml
    uv run python train.py --model catboost --run-name catboost_v1

Each run writes a self-contained artifact folder under ``artifacts/`` containing
the config snapshot, metrics (overall + breakdowns + business proxy), per-row
predictions, feature importances, and the serialized model.
"""

from __future__ import annotations

import argparse

from src.config import SUPPORTED_MODELS
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B — classical ML baseline")
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Which tree-based model to train.",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML config (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact folder name (default: <model>_<timestamp>).",
    )
    args = parser.parse_args()

    result = run(config_path=args.config, model_name=args.model, run_name=args.run_name)

    test = result.metrics["test"]["overall"]
    test_ex = result.metrics["test"]["overall_ex_stockout"]
    print("\n" + "=" * 60)
    print(f"Done: {result.model_name}")
    print(f"Artifacts -> {result.run_dir}")
    print(
        f"TEST (all)      WAPE={test['wape']:.4f}  MAE={test['mae']:.3f}  RMSE={test['rmse']:.3f}"
    )
    print(
        f"TEST (in-stock) WAPE={test_ex['wape']:.4f}  MAE={test_ex['mae']:.3f}  RMSE={test_ex['rmse']:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
