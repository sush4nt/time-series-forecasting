# FMCG Demand Forecasting — Classical ML Baseline (Part B)

Reproducible, config-driven pipeline that forecasts daily `units_sold` per
(store × SKU) and benchmarks tree-based models (**LightGBM / XGBoost / CatBoost**)
on the same feature matrix. Feature engineering is ported from `notebooks/eda.ipynb`
(Part A) with leakage and 14-day horizon controls intact.

## Pipeline (5 steps)

| Step | Module | What it does |
|------|--------|--------------|
| 1. Load | `src/data.py` | Read `data.csv`, parse dates, sort by `(store, sku, date)`. |
| 2. Features | `src/features.py` | Leakage-aware features: calendar, lag/rolling demand, price/promo, weather, encodings. Censored (stockout) demand masked; all train-only stats. |
| 3. Split | `src/splits.py` | Chronological train/val/test. Train drops stockout rows; val/test keep all. |
| 4. Train | `src/models.py` | LightGBM/XGBoost/CatBoost behind one interface; early stopping on val; preds clipped ≥ 0. |
| 5. Evaluate | `src/evaluate.py` | WAPE/MAE/RMSE overall + breakdowns + business proxy; writes artifacts. |

`src/pipeline.py` orchestrates steps 1→5; `train.py` is the CLI. Everything is
driven by `configs/baseline.yaml` (paths, split dates, feature params, per-model
hyperparameters, seed).

**Leakage controls:** only lags `≥ 14`; rolling windows shifted by the horizon;
target encoding, price medians and weather thresholds fit on TRAIN only;
stockout `units_sold` masked to `NaN` so it never leaks into lag/rolling features.

**Temporal split:** train `≤ 2022-12-31` · val `→ 2023-09-30` (early stopping) ·
test `→ 2023-12-31` (reported once).

## Setup

```bash
uv sync   # installs deps into .venv from uv.lock (see how-to-setup.md)
```

## Run

```bash
uv run python train.py --model lightgbm
uv run python train.py --model xgboost
uv run python train.py --model catboost

# optional flags
uv run python train.py --model lightgbm --config configs/baseline.yaml --run-name lgbm_v1
```

Each run creates a self-contained folder `artifacts/<model>_<timestamp>/`:

```
config.yaml               exact config used (reproducibility)
run_meta.json             seed, package versions, timings, row counts, best_iteration
metrics.json              overall + breakdowns + business proxy (val & test)
metrics_summary.txt       human-readable summary
feature_importance.csv    ranked feature importances
breakdown_{channel,category,promo}_test.csv
predictions_{test,val}.parquet   per-row preds + context (for failure-mode analysis)
model.{txt|json|cbm}      serialized model
```

## Evaluation assessment

Metrics (Part D): **WAPE / MAE / RMSE** overall, broken down by **channel**,
**category**, and **promo vs. non-promo**. Business proxy converts forecast error
to euros: over-forecast → overstock cost at `purchase_cost`; under-forecast →
lost-margin (stockout) cost at `list_price × margin_pct`.

Smoke-test results on the held-out **test** split (92,368 rows):

| Model | WAPE | MAE | RMSE |
|-------|------|-----|------|
| LightGBM | 0.2636 | 15.79 | 24.07 |
| XGBoost  | 0.2643 | 15.83 | 24.15 |
| CatBoost | 0.2630 | 15.75 | 24.00 |

All three are within noise of each other — the feature set, not the learner, is
the current bottleneck. Notable signals: **promo rows carry ~2× the MAE** of
non-promo (promo spikes), and **stockout rows show inflated error** because the
model predicts *true* demand while observed units are censored. `predictions_*.parquet`
retains `sku_id`, `promo_flag`, `stock_out_flag`, `date` and cost fields so the
next step (failure modes: sparse SKUs, promo spikes, cold starts) can slice these
directly.

## Directory structure

```
Assignment_ML_Engineer_v3/
├── configs/
│   └── baseline.yaml          # single source of truth for a run
├── data/
│   └── data.csv               # ~1.1M daily store×SKU rows
├── notebooks/
│   └── eda.ipynb              # Part A: EDA + feature engineering logic
├── src/
│   ├── config.py              # YAML -> typed Config
│   ├── data.py                # step 1: load
│   ├── features.py            # step 2: leakage-aware features
│   ├── splits.py              # step 3: temporal split
│   ├── models.py              # step 4: LGBM/XGB/CatBoost factory
│   ├── evaluate.py            # step 5: metrics + breakdowns + business proxy
│   └── pipeline.py            # orchestrator (steps 1→5, writes artifacts)
├── train.py                   # CLI entry point
├── artifacts/                 # per-run outputs (git-ignored)
├── problem_statement.pdf
├── how-to-setup.md
├── pyproject.toml / uv.lock
└── README.md
```
