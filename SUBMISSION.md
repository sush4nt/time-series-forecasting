# FMCG Multi-Store Demand Forecasting — Submission Write-Up

**Goal:** forecast the next **14 days** of daily `units_sold` for every active
`(store_id, sku_id)` pair, as a production-oriented system (not a notebook demo).
**Data:** `data.csv`, ~1.1M daily store×SKU rows · 7 countries · 13 stores · 4 channels
· 5 categories. One unified, config-driven pipeline (`src/runner.py`) runs both model
families through the **same splits, same metrics, same evaluator**.

> Companion docs: [`README.md`](README.md) (architecture),
> [`classical-model-training-and-evaluation.md`](classical-model-training-and-evaluation.md),
> [`gru-model-training-and-evaluation.md`](gru-model-training-and-evaluation.md),
> [`walk-forward-vs-fixed-origin.md`](walk-forward-vs-fixed-origin.md) (evaluation validity).

---

## 1. Approach

| Part | What we built |
|---|---|
| **A — Features** | Leakage-aware families: calendar/holiday, lag & rolling demand (shifted ≥ horizon), price/promo dynamics, weather, and store/category/channel encodings (out-of-fold target encoding). Censored (stockout) demand is masked so empty-shelf days don't teach low demand. |
| **B — Classical** | Direct 14-day tabular regression (one row = store×SKU×day). **LightGBM / XGBoost / CatBoost** behind one interface. Early-stopped on a temporal validation window. |
| **C — Deep learning** | One **global GRU encoder–decoder** (shared weights across all ~1,000 series): reads 28 past days → predicts 14 future days directly. Static covariates via embeddings; known-future covariates (promo/calendar/weather) fed to the decoder. |
| **D — Evaluation** | Shared `evaluate.py`: WAPE/MAE/RMSE overall + by channel/category/promo/stockout (+ per-horizon for the GRU), plus a € business proxy. |

**Forecasting frame (both models):** *direct* multi-horizon — all 14 days come out
without feeding predictions back in, so there is no recursive error snowball. Every
demand feature is lagged by **≥ 14 days** (the horizon), which makes a 14-day-ahead
forecast leakage-free by construction (see the walk-forward doc).

---

## 2. Assumptions

- **Known-future covariates.** Promotions, price, calendar/holidays, and weather are
  *planned or announced ahead*, so they are treated as known on the forecast date. Stated,
  not hidden — this is standard for retail planning.
- **Censored demand.** On `stock_out_flag == 1`, `units_sold` is supply-capped, not true
  demand. Those days are (a) **dropped from training**, (b) **masked** from lag/rolling
  features and from the GRU loss, and (c) reported **separately** at evaluation.
- **Temporal split (chronological, no shuffling).** train ≤ 2022-12-31 · val → 2023-09-30
  (early stopping) · test → 2023-12-31 (reported once). Test is a single Q4 window
  (holiday-heavy — a caveat, see §6).
- **Headline metric = WAPE on in-stock rows.** WAPE is scale-free (comparable across big
  and tiny SKUs); the in-stock view removes the mechanical penalty of forecasting true
  demand on capped stockout days. Both "all rows" and "in-stock" are reported.
- **Train-only statistics.** Scalers, target/price/weather stats, and categorical
  vocabularies are fit on **train only**; the GRU's preprocessor is serialized so
  inference replays identical transforms.

---

## 3. Results — model comparison (TEST, in-stock view)

| Model | WAPE | MAE | RMSE | bias | Train time | Notes |
|---|---:|---:|---:|---:|---:|---|
| **CatBoost** | **0.2399** | 14.72 | 22.02 | +0.03 | 62 s | best accuracy, well-centred |
| **LightGBM** | 0.2407 | 14.77 | 22.13 | −0.06 | **13 s** | ~tied, **fastest** |
| **XGBoost** | 0.2410 | 14.79 | 22.21 | +0.00 | 35 s | ~tied |
| **GRU enc-dec** | 0.2435 | 14.85 | 22.19 | −2.26 | **611 s** (CPU) | matches trees, ~47× slower, slight under-bias |

*All three trees are within 0.001 WAPE of each other; the GRU is within ~0.004.* On
accuracy this is effectively a **four-way tie**, with the GRU costing ~50× more to train
and far more to serve.

**Key breakdowns (LightGBM, representative):**

| Segment | Signal |
|---|---|
| **Promo vs non-promo** | Promo WAPE ≈ non-promo (~0.26) but **promo MAE ~2× higher** (28.3 vs 14.8) — spikes are big in absolute units and under-shot. |
| **Stockout vs in-stock** | Stockout rows: WAPE 4.1, large **positive** bias (~+50) — the censoring artifact, not a model failure (model predicts true demand > capped sales). |
| **By channel** | Tight band (0.259–0.265); Convenience easiest in MAE (low volume), Hypermarket hardest (high volume). |
| **By category** | Tight band (0.259–0.267); Snacks hardest (spiky). |

**Business proxy (€, directional).** Over-forecast → overstock at `purchase_cost`;
under-forecast → lost margin at `list_price × margin_pct`. Trees total ≈ **€5.44M**
(overstock €3.57M + lost margin €1.87M) over the test window. The GRU's total looks
lower (≈ €4.81M) mainly because its slight **under-bias** trades overstock for lost
margin — but ⚠️ **the GRU evaluates on fewer test rows (84.3k vs 92.4k)** due to the
28-day encoder warm-up, so absolute € are **not** directly comparable; WAPE/MAE are.

---

## 4. Classical vs. Deep Learning — when each wins, what to ship

**When the tree wins (today):** essentially everywhere that matters operationally —
equal-or-better WAPE, **~50× cheaper** to train, trivial to serve (a single model file,
millisecond CPU inference), transparent feature importances, and easy to debug. With
strong lag/rolling features, the tree already captures most of the signal.

**When the GRU wins — empirically, on this data: nowhere.** The segment-level head-to-head
(see `model_comparison.ipynb`) is decisive: **CatBoost wins all 11 segments** (every
channel, every category, promo and non-promo), and across series-volume terciles the GRU
wins **no** bucket — even **sparse/low-volume** series (the classic DL stronghold) go to a
tree. Its only neutral positive is a **flat per-horizon WAPE** (~0.26 from day 1 to 14, no
decay from the direct decoder). So the case for the GRU is **forward-looking, not
current**:
- **Cold-start / new SKUs** — shared weights + embeddings *should* let a thin series borrow
  from similar ones; not yet observable here because all series have long history.
- **Richer regimes at scale** — as promo interactions, cannibalisation, and cross-series
  structure grow, the sequence model has more headroom than hand-built lags.

**Ship-first call:** **LightGBM v1.** It is tied-best on accuracy (with CatBoost, within
noise), the **cheapest** to run (13 s train, ms CPU inference), and the easiest to monitor
and explain. **CatBoost** is the accuracy leader by a hair and the natural fallback if
native categorical handling is preferred. Keep the **GRU as v2** — a strategic bet that
only pays off as store/SKU count and promo complexity grow; today it earns its keep on
*none* of the segments.

---

## 5. Part E — Engineering Judgment

**Scaling to more stores & SKUs.**
- Classical scales near-linearly in rows; the bottleneck is *feature build*, not fit.
  Partition feature engineering by series (embarrassingly parallel — Spark/Dask/Ray) and
  train on the pooled table. Model size is independent of #series.
- The **GRU scales the best conceptually**: one global model regardless of series count;
  add series by adding rows/embeddings, not new models. Watch embedding cardinality
  (hash rare ids) and use GPU batching for training.
- **Avoid** per-series models (13 stores × hundreds of SKUs = thousands of models to
  train, deploy, and monitor) — both approaches here are single-model by design.

**Online vs. batch inference.**
- **Batch is the right default.** The horizon is 14 days and demand planning is a daily/
  weekly cycle → a **nightly batch job** scores all `(store, SKU)` pairs and writes
  forecasts to a table the planning system reads. Simple, cheap, reproducible.
- **Online/on-demand** only where justified: ad-hoc "what-if" promo simulations or new-SKU
  onboarding. Same serialized model + preprocessor behind a thin service; keep features
  precomputed in a feature store to hit latency.
- Known-future inputs (promo calendar, price, weather forecast) must be available to the
  batch job at run time — a data-contract dependency, not a modelling one.

**Monitoring for drift & promo-regime change.**
- **Input drift:** PSI/KS on key features (price, discount depth, promo frequency,
  weather, per-channel volume) vs. the training distribution.
- **Output/accuracy drift:** track rolling WAPE/MAE/bias **overall and per segment**
  (channel, category, promo) as actuals land (≤14-day lag); alert on threshold breach and
  on **persistent bias** (systematic over/under-forecast).
- **Promo-regime change specifically:** monitor promo share, average discount depth, and
  **promo-segment WAPE** separately; a shift in promo strategy is the most likely cause of
  silent degradation. Trigger **retrain** on drift or on a scheduled cadence
  (e.g. monthly), with a champion/challenger gate on held-out WAPE.
- Log every run's config + versions (already persisted per artifact) for reproducibility.

**What I would NOT put in a v1 release.**
- **The GRU** — no accuracy win today for ~50× the cost/complexity. Ship it as v2.
- **Probabilistic / quantile forecasts** — valuable for safety stock, but v1 ships point
  forecasts; add quantiles (or conformal intervals) next.
- **Forecast → auto-order automation** — v1 *recommends*; a human/planning system decides.
  No closed-loop ordering until monitoring is trusted.
- **Recursive multi-step beyond 14 days**, hyper-granular per-store-per-SKU custom models,
  and real-time streaming inference — all unnecessary for the stated 14-day batch problem.

---

## 6. Known limitations & next steps

1. **Prove evaluation validity** — add a **rolling-origin 14-day backtest** (should match
   the reported ~0.24 WAPE) to formally close the walk-forward vs. fixed-origin question
   (see [`walk-forward-vs-fixed-origin.md`](walk-forward-vs-fixed-origin.md)).
2. **Add a naive floor** — seasonal-naive (same-weekday-last-week) WAPE as a reference so
   "0.24" has meaning ("models beat naive by X%").
3. **Single Q4 test window** is holiday-heavy → add a **rolling/multi-origin backtest** over
   several windows for a stability estimate.
4. **Uncertainty** — add quantile or conformal intervals to drive safety stock / service
   levels; the current business proxy is directional only (full `purchase_cost`, no holding
   fraction).
5. **Promo spikes** remain the top error source (~2× MAE) — candidate for promo-specific
   features or a dedicated uplift component.
6. **GRU headroom** — light tuning (encoder length, hidden size, attention) and GPU
   training if it graduates to v2.

**Bottom line:** a reproducible, leakage-controlled pipeline that benchmarks four models
on identical footing. **CatBoost/LightGBM lead on accuracy at a fraction of the cost — ship
LightGBM v1, keep the GRU as the v2 growth bet.**
