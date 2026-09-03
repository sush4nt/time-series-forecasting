"""CLI entry point for the Part C GRU encoder-decoder.

Examples
--------
    uv run python train_dl.py
    uv run python train_dl.py --run-name gru_v1
    uv run python train_dl.py --max-epochs 2 --limit-series 40   # fast smoke test

Writes the same artifact folder shape as the classical baseline, so the exploration
notebook and comparison tooling work unchanged.
"""

from __future__ import annotations

import argparse

from src.dl_pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Part C — GRU encoder-decoder")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override dl.max_epochs.")
    parser.add_argument(
        "--limit-series", type=int, default=None,
        help="Train on only the first N series (smoke testing).",
    )
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        limit_series=args.limit_series,
    )

    test = result.metrics["test"]
    print("\n" + "=" * 60)
    print(f"Done: {result.model_name}")
    print(f"Artifacts -> {result.run_dir}")
    print(
        f"TEST (all)      WAPE={test['overall']['wape']:.4f}  "
        f"MAE={test['overall']['mae']:.3f}  RMSE={test['overall']['rmse']:.3f}"
    )
    print(
        f"TEST (in-stock) WAPE={test['overall_ex_stockout']['wape']:.4f}  "
        f"MAE={test['overall_ex_stockout']['mae']:.3f}  "
        f"RMSE={test['overall_ex_stockout']['rmse']:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
