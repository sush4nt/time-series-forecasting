# Walk-Forward vs. Fixed-Origin Evaluation — a Visual Guide

This note explains one subtle-but-important point that a reviewer *will* probe:

> **"Your test WAPE is 0.24 — but is that a real 14-day-ahead forecast, or did the
> model quietly peek at data it wouldn't have at forecast time?"**

Short answer: **the numbers are honest**, and this doc shows exactly why, using
pictures. It also pins down the one scenario where the two evaluation styles *do*
diverge, and what to measure to prove it.

The confusing paragraph this explains lives in
[`src/splits.py`](src/splits.py) lines 13–23.

---

## 1. First, the one fact that makes everything click

Our classical model predicts **one day's `units_sold` from a row of features**. The
features that come from *past demand* are the risky ones (they could leak the future).
So the pipeline shifts **every** demand feature back by **at least the 14-day horizon**
([`src/features.py`](src/features.py) lines 105–147):

| Feature | Reads demand from | Shift |
|---|---|---|
| `lag_14` | 14 days ago | 14 |
| `lag_28` | 28 days ago | 28 |
| `rolling_mean_7/14/28` | a window ending **14 days ago** | 14 |
| `rolling_std / min / max` | a window ending **14 days ago** | 14 |

> **The golden rule:** to forecast day `T + k` (for `k = 1 … 14`), the model only ever
> reads demand from day `T + k − 14` **or earlier** — i.e. day `T` or before. It
> **never** reads a value from inside the 14-day window it is predicting.

Keep this picture in your head:

```
                 the 14 days we forecast
   ┌───────────────────────────────────────────┐
   │                                            │
  T-14 ... T-1  T │ T+1  T+2  ...  T+13  T+14 │
   ▲              │  └──────────────────────┘  │
   │              │        forecast window      │
   │  lag_14 for T+1  reads T-13   ✅ known (≤ T)
   │  lag_14 for T+14 reads T      ✅ known (≤ T)
   └─ every demand feature points to here or further left — always ≤ T
```

**Consequence:** a *single* 14-day-ahead forecast is **100% leakage-free by
construction**. There is no way for it to cheat, because nothing it reads lives inside
the window. Hold onto this — it is the whole answer.

---

## 2. The two "styles" — and why they only differ past day 14

The test window is **not** 14 days; it is **~92 days** (Oct 1 – Dec 31, 2023). So the
question becomes: *how were those 92 days of predictions produced?* There are two
mental models.

### Style A — Fixed-origin (a single "forecast and walk away")

You stand at **one origin `T`** (say Sep 30). You forecast forward and never look at
new sales again.

```
 origin T = Sep 30
   │
   ▼
   ├── T+1 ... T+14   ← uses only actuals ≤ T   ✅ clean (rule from §1)
   │
   ├── T+15 ... T+28  ← lag_14 now needs T+1..T+14, which are IN the future!
   │                     you don't have actuals → must feed the model's OWN
   │                     predictions back in → recursive → errors compound
   ▼
   ... it gets progressively shakier the further you go
```

- **Days 1–14:** perfect, no leakage (§1).
- **Days 15+:** `lag_14` for day `T+15` wants the actual of day `T+1` — but from origin
  `T` you don't know it yet. You'd substitute the model's **prediction** of `T+1`.
  Predictions-feeding-predictions is **recursive forecasting**, and small errors
  **snowball**. This is why a single-origin forecast of a *long* horizon degrades.

**But our horizon is fixed at 14.** So a *correct* fixed-origin forecast simply stops
at day 14. To cover the 92-day test window this way, you don't extrapolate one origin
forever — you **re-plant the origin every 14 days** (Style B).

### Style B — Walk-forward / rolling-origin (what we actually report)

Re-plant the origin every 14 days. Each origin forecasts its **own** next 14 days,
using the actuals that have arrived by then. This mirrors production: *you re-run the
model as fresh sales come in.*

```
 test window: Oct 1 ─────────────────────────────────────────────► Dec 31

 origin 1 (Sep 30):   forecast Oct 1–14   ┐ each block reads only
 origin 2 (Oct 14):   forecast Oct 15–28  │ actuals ≤ its own origin
 origin 3 (Oct 28):   forecast Oct 29–Nov 11 │ → every block obeys §1
 origin 4 (Nov 11):   forecast Nov 12–25  │ → every block is leakage-free
 ...                                       ┘
        ▲ by the time origin 3 runs, Oct 14's actuals ARE known
          (14 days have passed) — so reading them is legitimate, not cheating
```

Each 14-day block is individually clean (§1). Stitch the blocks together and you have
92 days of **genuine 14-day-ahead** predictions.

---

## 3. So what does the pipeline actually compute?

The pipeline builds features for **all** test days in one pass over the full
(train + val + test) series. For a test day deep in the window — say Nov 20 — its
`lag_14` reads the **actual** `units_sold` of Nov 6.

- **Is Nov 6 "the future"?** No. Nov 6 is 14 days *before* Nov 20. Under Style B, by
  the time you forecast Nov 20 you have long since observed Nov 6. Reading its actual is
  exactly what a rolling redeployment does.
- **Therefore** the single-pass computation is **mathematically identical to the
  rolling-origin (Style B) backtest** — precisely *because* all lags are ≥ 14. The
  reported WAPE **is** a true 14-day-ahead number, presented as a "re-scored every 14
  days" deployment.

```
 What "reading actual Nov 6 to predict Nov 20" means:

  Style A (single origin Sep 30):  ❌ illegal — Nov 6 unknown from Sep 30
  Style B (rolling, re-plant q14): ✅ legal   — Nov 6 known by Nov 20
  Our pipeline:                    ≡ Style B  (because every lag is ≥ 14 days)
```

---

## 4. Where the honesty caveat actually applies

The `splits.py` note is being **conservative**. It flags that the reported numbers are
**not** the numbers for *"forecast the entire 92-day test window from one fixed origin
and walk away"* (Style A extended past day 14). That scenario:

- is **not** what the assignment asks (horizon is fixed at **14**, not 92), **and**
- would be **worse**, because past day 14 it must feed predictions into its own lags
  (recursive snowball, §2 Style A).

So there is no leak and no inflated score. There is only a **naming/interpretation**
risk: a reader who assumes "92-day test = one continuous forecast" would be wrong about
*how* those predictions are generated.

---

## 5. The one number worth adding (to end the debate)

To convert this from "trust the argument" to "here's the evidence", run a **rolling-origin
14-day backtest** and show it matches the headline:

```
for origin in [Sep30, Oct14, Oct28, ...]:        # every 14 days
    build features using ONLY data ≤ origin       # hard cut, no peeking
    predict origin+1 … origin+14
    score against actuals
aggregate WAPE/MAE/RMSE over all blocks
```

Expected result: **≈ the reported 0.24 WAPE** (confirming §3 — no leakage). Optionally
also run the **pessimistic** version — a single fixed origin driven *recursively* across
all 92 days — to quantify how much worse "forecast once and walk away" would be. The gap
between the two is a crisp, quantified answer to the reviewer's question, and doubles as
the business case for **re-ingesting actuals on a ≤14-day cadence** in production.

---

## 6. TL;DR

1. Every demand feature is lagged **≥ 14 days** (the horizon), so a **single 14-day
   forecast cannot leak** — it never reads inside its own window.
2. **Fixed-origin** and **walk-forward** give the *same* answer **for the first 14
   days**. They only diverge if you push a *single* origin **past** day 14 (then Style A
   goes recursive and degrades).
3. The 92-day test metric is computed as a **rolling / walk-forward** redeployment
   (re-scored every 14 days) — a legitimate, realistic **14-day-ahead** number, **not**
   a leaky one.
4. To *prove* it: add a **rolling-origin 14-day backtest** (should match ~0.24) and,
   optionally, a recursive single-origin run to show the (worse) "forecast-and-walk-away"
   number.
