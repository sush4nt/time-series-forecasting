"""CLI entry point for the unified forecasting pipeline.

Examples
--------
    uv run python train.py --model lightgbm
    uv run python train.py --model xgboost  --config configs/baseline.yaml
    uv run python train.py --model gru      --run-name gru_v1
    uv run python train.py --model gru      --max-epochs 2 --limit-series 40  # smoke

Each run writes a self-contained artifact folder under ``artifacts/`` with the same
shape regardless of model family (config snapshot, metrics + breakdowns + business
proxy, per-row predictions, serialized model, and model-specific extras).
"""

from __future__ import annotations

import argparse

from src.config import SUPPORTED_MODELS
from src.runner import run

MODELS = SUPPORTED_MODELS + ("gru",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified forecasting pipeline")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Which model to train (tree baseline or GRU).",
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
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="GRU only: override dl.max_epochs.",
    )
    parser.add_argument(
        "--limit-series", type=int, default=None,
        help="Train on only the first N series (smoke testing).",
    )
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        model=args.model,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        limit_series=args.limit_series,
    )

    test = result.metrics["test"]
    test_all, test_ex = test["overall"], test["overall_ex_stockout"]
    print("\n" + "=" * 60)
    print(f"Done: {result.model_name}")
    print(f"Artifacts -> {result.run_dir}")
    print(
        f"TEST (all)      WAPE={test_all['wape']:.4f}  "
        f"MAE={test_all['mae']:.3f}  RMSE={test_all['rmse']:.3f}"
    )
    print(
        f"TEST (in-stock) WAPE={test_ex['wape']:.4f}  "
        f"MAE={test_ex['mae']:.3f}  RMSE={test_ex['rmse']:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
