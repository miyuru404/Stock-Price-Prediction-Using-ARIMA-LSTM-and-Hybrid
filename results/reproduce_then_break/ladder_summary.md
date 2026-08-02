# Reproduce, then break it — where does the "macro improves accuracy" gain go?

Papers report that adding macroeconomic variables improves forecasting. Every test in this project
says it does not. Rather than argue, this reproduces the **paper-style setup** and then adds our
guards **one at a time**, recording where the improvement dies.

## The ladder

| Level | What changes |
|---|---|
| **L0 PAPER STYLE** | single 80/20 split · macro aligned by **reference date (no publication lag)** · scaler fitted on **train+test** · price-level target |
| **L1** | + scaler fitted on **training data only** |
| **L2** | + macro lagged to when it was actually **published** (rates 35d, CPI 21d) |
| **L3** | + **walk-forward** instead of one split |
| **L4** | + the **naive baseline**, which most papers never include (the `macro_vs_naive` column) |

`improvement_%` is the headline a paper would report:
`100 x (MAPE_price_only - MAPE_price_macro) / MAPE_price_only`. Positive = "macro helped".

## Results

| target | level | windows | MAPE_price_only | MAPE_price_macro | MAPE_naive | improvement_pct | R2 | macro_vs_naive | DM_p |
|---|---|---|---|---|---|---|---|---|---|
| BANKS | L0 PAPER STYLE | 1 | 0.8323 | 0.8309 | 0.8514 | 0.174 | 0.9984 | 0.976 | 0.6255 |
| BANKS | L1 + train-only scaling | 1 | 0.8323 | 0.8306 | 0.8514 | 0.216 | 0.9984 | 0.9756 | 0.5947 |
| BANKS | L2 + publication lag | 1 | 0.836 | 0.8381 | 0.855 | -0.248 | 0.9984 | 0.9802 | 0.4831 |
| BANKS | L3 + walk-forward | 7 | 1.007 | 0.995 | 0.9545 | -0.798 | 0.9886 | 1.0266 | 0.2879 |
| SPSL20 | L0 PAPER STYLE | 1 | 0.7691 | 0.7694 | 0.7918 | -0.041 | 0.9978 | 0.9717 | 0.8614 |
| SPSL20 | L1 + train-only scaling | 1 | 0.7691 | 0.7694 | 0.7918 | -0.042 | 0.9978 | 0.9717 | 0.8614 |
| SPSL20 | L2 + publication lag | 1 | 0.7691 | 0.7635 | 0.7918 | 0.723 | 0.9978 | 0.9643 | 0.9857 |
| SPSL20 | L3 + walk-forward | 7 | 0.9026 | 0.9478 | 0.9036 | -1.769 | 0.9835 | 1.0256 | 0.0611 |

**Did the paper-style setup reproduce a gain? YES.**

## Reading it

- **`improvement_%`** is the claim. Follow it down the ladder.
- **`macro_vs_naive`** is L4: below 1.0 means the model actually beats "tomorrow = today".
  This column is the one most papers never compute — and it is the one that matters.
- **`R2`** stays around 0.99 at every level, for every configuration. That is the trap: on a price
  LEVEL, R2 near 1.0 is what a random walk scores. It is not evidence of anything.

## Caveats
- The models here are Ridge and XGBoost (best of the two reported, as papers routinely report their
  best). ARIMA/LSTM/hybrid variants are covered in `hybrid_arima_lstm_macro_news.py`.
- L0's full-sample scaler is a mild leak — it leaks distributional information, not the target.
  It is included because it is extremely common in published code.
- "No publication lag" at L0/L1 is the honest reconstruction of a widespread practice: aligning
  month-end macro to month-end prices means the model sees a figure weeks before it existed.
