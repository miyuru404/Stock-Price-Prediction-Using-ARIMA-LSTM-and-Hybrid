# Price forecasting with macro + news

The project's Stage 1 forecast price from price history alone and found nothing beat the naive
"tomorrow = today". That was **univariate**. This run asks the follow-up the project title implies:

> **does adding macroeconomic variables and news sentiment improve PRICE forecasting?**

**Targets:** ['HNB', 'COMB', 'SAMP', 'LOFC', 'LOLC', 'LFIN', 'CFIN'] + indices ['SPSL20', 'ASPI'] · **Horizons:** [1, 5] day(s) · walk-forward 6-month folds,
refit every fold, per stock.

**Feature sets (nested):** `P` price/technical → `P+M` + rates, inflation, FX →
`P+M+N` + news sentiment.
**Models:** naive · ARIMA(1,1,1) · Ridge · XGBoost.

**Target framing:** models predict the next-h-day RETURN, which is converted back to a price
(`price_hat = close_today x (1 + return_hat)`). Predicting the price level directly would let any
model score well by echoing today's price — it would simply reproduce the naive baseline while
appearing accurate. MAPE is still reported on the reconstructed PRICE so the numbers stay
comparable to Stage 1.

## Results vs naive

| horizon_days | model | median_MAPE_% | median_ret_RMSE_pp | median_dir_acc_% | MAPE_vs_naive | beats_naive | beats_naive_p |
|---|---|---|---|---|---|---|---|
| 1 | naive (no change) | 1.174 | 1.698 | 8.1 | 1.0 | 0/169 | 1.0 |
| 1 | ARIMA(1,1,1) | 1.197 | 1.711 | 45.2 | 1.004 | 39/169 | 1.0 |
| 1 | Ridge P (price only) | 1.215 | 1.728 | 48.0 | 1.0188 | 51/169 | 1.0 |
| 1 | Ridge P+M (+macro) | 1.232 | 1.754 | 47.9 | 1.0308 | 38/169 | 1.0 |
| 1 | Ridge P+M+N (+macro+news) | 1.245 | 1.76 | 47.8 | 1.0414 | 36/169 | 1.0 |
| 1 | XGBoost P+M (+macro) | 1.272 | 1.783 | 47.4 | 1.0679 | 24/169 | 1.0 |
| 1 | XGBoost P+M+N (+macro+news) | 1.275 | 1.801 | 47.8 | 1.0824 | 23/169 | 1.0 |
| 1 | XGBoost P (price only) | 1.292 | 1.791 | 49.6 | 1.0602 | 31/169 | 1.0 |
| 5 | naive (no change) | 2.921 | 3.895 | 2.6 | 1.0 | 0/169 | 1.0 |
| 5 | ARIMA(1,1,1) | 2.93 | 3.985 | 49.2 | 1.0 | 69/169 | 0.9932 |
| 5 | Ridge P (price only) | 2.999 | 3.892 | 52.1 | 1.029 | 57/169 | 1.0 |
| 5 | Ridge P+M (+macro) | 3.129 | 4.152 | 49.6 | 1.0548 | 36/169 | 1.0 |
| 5 | XGBoost P (price only) | 3.158 | 4.232 | 50.0 | 1.1006 | 24/169 | 1.0 |
| 5 | Ridge P+M+N (+macro+news) | 3.194 | 4.16 | 50.0 | 1.0601 | 34/169 | 1.0 |
| 5 | XGBoost P+M+N (+macro+news) | 3.296 | 4.308 | 47.6 | 1.113 | 16/169 | 1.0 |
| 5 | XGBoost P+M (+macro) | 3.311 | 4.261 | 47.5 | 1.132 | 12/169 | 1.0 |

`MAPE_vs_naive` below 1.0 means better than doing nothing. `beats_naive` counts stock-fold cases.

**Best per horizon:**

| horizon_days | model | median_MAPE_% | MAPE_vs_naive | beats_naive | beats_naive_p |
|---|---|---|---|---|---|
| 1 | naive (no change) | 1.174 | 1.0 | 0/169 | 1.0 |
| 5 | naive (no change) | 2.921 | 1.0 | 0/169 | 1.0 |

**No model beats the naive baseline significantly at either horizon.**

## Does macro / news improve on price-only? (the ablation)

| horizon_days | algo | added | median_MAPE_improvement_pp | cases_improved | sign_test_p |
|---|---|---|---|---|---|
| 1 | Ridge | P+M (+macro) | -0.007 | 44/169 | 1.0 |
| 1 | Ridge | P+M+N (+macro+news) | -0.015 | 43/169 | 1.0 |
| 1 | XGBoost | P+M (+macro) | -0.007 | 76/169 | 0.917 |
| 1 | XGBoost | P+M+N (+macro+news) | -0.019 | 60/169 | 0.9999 |
| 5 | Ridge | P+M (+macro) | -0.041 | 51/169 | 1.0 |
| 5 | Ridge | P+M+N (+macro+news) | -0.055 | 48/169 | 1.0 |
| 5 | XGBoost | P+M (+macro) | -0.085 | 67/169 | 0.9973 |
| 5 | XGBoost | P+M+N (+macro+news) | -0.048 | 72/169 | 0.9774 |

`median_MAPE_improvement_pp` is positive when the added data REDUCED error.

## Reading it simply

- Every model sits at or above `MAPE_vs_naive = 1.0`, i.e. at or worse than "tomorrow = today".
- Adding macro and news moves MAPE by fractions of a percentage point, in both directions, with no
  consistency across stocks or folds.
- This mirrors the direction results exactly: the same information that fails to predict DIRECTION
  also fails to improve PRICE forecasting.

## Leak scan

Every feature is checked against both the SAME-DAY and the NEXT-DAY return. A legitimate predictor
should track today's move at least as closely as tomorrow's; the reverse means it carries future
information. This check was added after the ASPI forward-fill bug (see below) and now runs on every
target automatically — result in `leak_scan.csv`.

**A bug this caught:** ASPI was originally forward-filled onto each target's calendar. The target
and ASPI do not share a calendar, so the fill let ASPI carry a move that the target only recorded on
its next row. `aspi_ret_1` then correlated **0.742** with SPSL20's FUTURE return but only **0.374**
with its own same-day return, and halved MAPE (ratio 0.496 vs a true 0.953). Fixed by removing the
forward-fill and dropping ASPI features for index targets entirely.

## Caveats
- News sentiment only exists from 2016 to 2022-06, so the `P+M+N` rows cover a shorter span than
  `P` and `P+M`; compare within a row, not across feature sets on different windows.
- ARIMA is fitted once per fold with a fixed (1,1,1) order and applies a constant drift across the
  fold — it is a reference point, not a tuned competitor. Stage 1 already found auto-ARIMA
  collapsing to (0,1,0), i.e. the naive model, on this data.
- MAPE on a price level is dominated by the level itself; the `ret_RMSE_pp` and `dir_acc_%` columns
  are the more honest views of forecast skill.
- **The naive row's `dir_acc_%` (10.5% at 1 day, 3.3% at 5 days) is a metric artefact, not a
  result.** Naive predicts a return of exactly 0, and `sign(0)` matches only on days the price did
  not move at all — so that column is really "share of perfectly flat days". It is still worth
  noting on its own terms: **~10% of CSE bank/finance trading days close exactly unchanged**, which
  is an illiquidity fact and part of why the naive baseline is so hard to beat here.
