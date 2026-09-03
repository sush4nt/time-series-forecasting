# FMCG Demand Forecasting — Classical ML + Deep Learning

Reproducible, config-driven pipeline that forecasts the next **14 days** of daily
`units_sold` per (store × SKU). It benchmarks two model families on the **same
splits and the same metrics**:

- **Part B — Classical baseline:** tree models (**LightGBM / XGBoost / CatBoost**) on
  engineered tabular features.
- **Part C — Deep learning:** a global **GRU encoder-decoder** on windowed sequences
  (shared weights across all series, static covariates, known-future inputs).

Both families run through **one unified pipeline** (`src/runner.py`) with pluggable
**backends**, so every run writes the same artifact shape and is scored by the same
`src/evaluate.py`.

## Architecture

```
                         src/runner.py  (ingest → validate → prepare → fit → predict → evaluate → persist)
                                │                    │                          │            │
             ┌──────────────────┴───────┐     backend-specific          shared evaluate  shared persist
             ▼                          ▼        (3 stages)              (evaluate.py)   (artifacts.py)
   backends/classical.py         backends/gru.py
   (features → splits → models)  (dl_dataset → dl_model → train loop)
```

The backend (`src/backends/base.py` protocol) implements only the genuinely
model-specific stages — **prepare / fit / predict** — plus its own artifacts, and
returns per-split `(meta, y_pred)`. Everything else (seeding, evaluation,
metrics/run_meta/summary/breakdowns/predictions) is shared.

| Concern | Module |
|---|---|
| Ingest + daily-gap check | `src/data.py` |
| Classical features (leakage-aware) | `src/features.py` |
| Temporal split (tabular) | `src/splits.py` |
| Tree models behind one interface | `src/models.py` |
| Sequence windowing + train-only scalers/vocabs | `src/dl_dataset.py` |
| GRU encoder-decoder | `src/dl_model.py` |
| Metrics + breakdowns + business proxy | `src/evaluate.py` |
| Shared seed/versions/summary/persistence | `src/artifacts.py` |
| Unified orchestrator | `src/runner.py` |
| Backend registry + adapters | `src/backends/` |

**Leakage controls (shared intent):** classical uses only lags `≥ 14` with rolling
windows shifted by the horizon; the GRU lags demand history and feeds only
known-future covariates to the decoder. Scalers, target encodings/stats, price
medians, weather thresholds and categorical vocabularies are fit on **TRAIN only**.
Stockout `units_sold` is treated as **censored** — masked from classical
lag/rolling features and from the GRU loss.

**Temporal split:** train `≤ 2022-12-31` · val `→ 2023-09-30` (early stopping) ·
test `→ 2023-12-31` (reported once).

## Setup

```bash
uv sync   # installs deps (incl. CPU PyTorch) into .venv from uv.lock — see how-to-setup.md
```

## Run

One CLI, `--model` selects the backend:

```bash
uv run python train.py --model lightgbm
uv run python train.py --model xgboost
uv run python train.py --model catboost
uv run python train.py --model gru                       # Part C

# useful flags
uv run python train.py --model gru --run-name gru_v1
uv run python train.py --model gru --max-epochs 2 --limit-series 40   # fast smoke test
```

Each run creates a self-contained folder `artifacts/<model>_<timestamp>/`:

```
config.yaml               exact config used (reproducibility)
run_meta.json             seed, versions, timings, split, + model-specific meta
metrics.json              overall + breakdowns + business proxy (train, val & test)
metrics_summary.txt       human-readable summary
predictions_{test,val}.parquet   per-row preds + context (for failure-mode analysis)
breakdown_{channel,category,promo,stockout}_test.csv

# classical only
feature_importance.csv    ranked feature importances
model.{txt|json|cbm}      serialized tree model

# GRU only
breakdown_horizon_test.csv   per-lead-time (1..14) WAPE/MAE/RMSE
model.pt                     weights + config + target stats
preprocessor.joblib          fitted scalers + vocabs (train==inference transforms)
```

Both families share the top-level `metrics.json` structure; model-specific keys
(`params`/`best_iteration`/`n_features` vs. `best_epoch`/`encoder_len`/`horizon`)
are contributed by each backend, so tooling works across models.

## Evaluation

Metrics (Part D): **WAPE / MAE / RMSE** overall, broken down by **channel**,
**category**, **promo vs. non-promo**, and **in-stock vs. stockout**; the GRU adds a
**per-horizon** breakdown. Two overall views are reported: **all rows** and
**in-stock only** (the fair, censoring-free number — see the write-ups). Business
proxy converts error to euros: over-forecast → overstock at `purchase_cost`;
under-forecast → lost margin at `list_price × margin_pct`.

Results on the held-out **test** split (in-stock view):

| Model | WAPE | MAE | RMSE | train time |
|-------|------|-----|------|-----------|
| LightGBM | 0.2407 | 14.77 | 22.12 | seconds |
| GRU encoder-decoder | 0.2437 | 14.86 | 22.30 | ~16 min (CPU) |

The GRU **matches** the tuned tree on accuracy at far higher training/serving cost
→ **ship LightGBM v1, keep the GRU as v2**. Notable failure modes (both models):
**promo rows carry ~2× the MAE** (promo spikes), **stockout rows show inflated
positive error** (censored demand), and **sparse/low-volume SKUs** have the worst
per-series WAPE.

## Notebooks & write-ups

| File | Purpose |
|---|---|
| `notebooks/eda.ipynb` | Part A: EDA + feature-engineering logic |
| `notebooks/artifact_exploration.ipynb` | Deep-dive on one run (metrics, breakdowns, failure modes) |
| `notebooks/model_comparison.ipynb` | Baseline vs. deep learning on the agreed metrics |
| `classical-model-training-and-evaluation.md` | How the tree forecast is made & evaluated |
| `gru-model-training-and-evaluation.md` | GRU architecture, training, evaluation |

## Directory structure

```
time-series-forecasting/
├── configs/
│   └── baseline.yaml          # single source of truth (paths, split, features, model + dl params)
├── data/
│   └── data.csv               # ~1.1M daily store×SKU rows
├── notebooks/
│   ├── eda.ipynb
│   ├── artifact_exploration.ipynb
│   └── model_comparison.ipynb
├── src/
│   ├── config.py              # YAML -> typed Config (+ DLConfig)
│   ├── data.py                # ingest + daily-gap check
│   ├── features.py            # classical leakage-aware features (OOF target encoding)
│   ├── splits.py              # temporal split (tabular)
│   ├── models.py              # LGBM/XGB/CatBoost factory
│   ├── dl_dataset.py          # sequence windowing + Preprocessor (train-only transforms)
│   ├── dl_model.py            # GRU encoder-decoder
│   ├── evaluate.py            # shared metrics + breakdowns + business proxy
│   ├── artifacts.py           # shared seed/versions/summary/persistence
│   ├── runner.py              # unified orchestrator
│   └── backends/              # base protocol + classical + gru adapters + registry
├── train.py                   # unified CLI (--model lightgbm|xgboost|catboost|gru)
├── artifacts/                 # per-run outputs (git-ignored)
├── classical-model-training-and-evaluation.md
├── gru-model-training-and-evaluation.md
├── problem_statement.pdf
├── how-to-setup.md
├── pyproject.toml / uv.lock
└── README.md
```
