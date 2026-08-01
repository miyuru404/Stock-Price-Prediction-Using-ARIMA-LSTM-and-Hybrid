# Multi-Horizon Direction + Return — Phase A (Tier-1 technical only)

**Stock:** HNB daily (2012-02-22 → 2026-07-27, 3270 rows)
**Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252] trading days · **Method:** DIRECT (separate model per horizon)
**Features:** Tier-1 only — ret_1, ret_5, ret_10, ma5_ratio, ma10_ratio, ma20_ratio, momentum_10, vol_10, vol_20
**Split:** 80/20 chronological, with a purge gap of h bars (no label leaks across the split)
**Dead-zone:** ±0.5%·√h (grows with horizon, else Hold disappears at long horizons)

## BOTTOM LINE (caveman)

- **Direction: 0 of 9 horizons beat the baseline. Zero. No edge anywhere.**
- Longer horizon does **NOT** help. It gets *worse*: −3.5 pp at 1 day,
  −47.1 pp at 1 year.
- Why worse: at long horizons the stock just drifts one way, so the dumb baseline gets huge
  (majority 78% at 6 months, persistence 77% at 1 year). Easy to be dumb, hard to beat dumb.
- **Return %: RMSE beats naive at 8/9 horizons — but that is fake.**
  Ridge is only ~1% better than "predict the average past return". It learned *drift*, not signal.
- **Return % sign accuracy: 55-57% looks good, is not.** "Always guess up" gets the same or more.
  Real sign wins: 2/9, both +0.2 pp = noise.
- **Verdict: Tier-1 technical features carry no information at any horizon.** Same answer as
  Step 1, now proven across 1 day → 1 year and on both targets. This is the floor.

## Direction — accuracy vs horizon

| horizon | deadzone_% | n_test | indep_test_windows | acc_logistic_% | acc_xgboost_% | acc_majority_% | acc_persistence_% | edge_pp | beats_baseline |
|---|---|---|---|---|---|---|---|---|---|
| 1 day | 0.5 | 650 | 650.0 | 39.5 | 34.0 | 39.5 | 43.1 | -3.5 | No |
| 1 week | 1.12 | 649 | 129.8 | 36.1 | 31.3 | 28.4 | 37.3 | -1.2 | No |
| 2 weeks | 1.58 | 648 | 64.8 | 34.0 | 33.2 | 31.2 | 36.9 | -2.9 | No |
| 3 weeks | 1.94 | 647 | 43.1 | 36.0 | 30.6 | 30.9 | 43.0 | -7.0 | No |
| 1 month | 2.35 | 646 | 29.4 | 40.1 | 33.1 | 28.6 | 42.1 | -2.0 | No |
| 2 months | 3.32 | 637 | 14.5 | 37.4 | 34.7 | 50.5 | 31.6 | -13.2 | No |
| 3 months | 4.06 | 628 | 9.5 | 26.4 | 31.7 | 55.9 | 33.4 | -24.2 | No |
| 6 months | 5.74 | 602 | 4.6 | 36.5 | 32.2 | 78.2 | 69.8 | -41.7 | No |
| 1 year | 7.94 | 554 | 2.2 | 29.8 | 22.6 | 1.8 | 76.9 | -47.1 | No |

**Horizons that beat both baselines: 0 of 9** (none).
Best edge: **1 week**, -1.2 pp (model 36.1% vs baseline 37.3%).

## Return % — error vs two nulls

Two nulls, because they answer different questions:
- **naive "no change" (predict 0%)** — the weak null.
- **train-mean drift (predict the average past h-day return)** — the TOUGHER null. It already
  contains the market's upward drift, so only beating *this* proves the **features** did work.

| horizon | rmse_ridge | rmse_xgb | rmse_naive0 | rmse_trainmean | rmse_ratio_ridge_vs_naive | rmse_ratio_ridge_vs_trainmean | beats_naive0 | beats_trainmean |
|---|---|---|---|---|---|---|---|---|
| 1 day | 1.54 | 1.65 | 1.55 | 1.55 | 0.99 | 0.991 | Yes | Yes |
| 1 week | 4.2 | 4.82 | 4.26 | 4.25 | 0.99 | 0.99 | Yes | Yes |
| 2 weeks | 6.52 | 7.16 | 6.64 | 6.58 | 0.98 | 0.99 | Yes | Yes |
| 3 weeks | 8.42 | 9.22 | 8.57 | 8.47 | 0.98 | 0.994 | Yes | Yes |
| 1 month | 10.74 | 11.87 | 10.99 | 10.8 | 0.98 | 0.994 | Yes | Yes |
| 2 months | 16.51 | 17.39 | 17.21 | 16.58 | 0.96 | 0.995 | Yes | Yes |
| 3 months | 21.19 | 22.87 | 22.61 | 21.61 | 0.94 | 0.98 | Yes | Yes |
| 6 months | 31.95 | 36.32 | 33.27 | 32.33 | 0.96 | 0.988 | Yes | Yes |
| 1 year | 70.73 | 75.77 | 69.15 | 70.68 | 1.02 | 1.001 | No | No |

**Beat naive "no change": 8 of 9** (1 day, 1 week, 2 weeks, 3 weeks, 1 month, 2 months, 3 months, 6 months).
**Beat train-mean drift: 8 of 9** (1 day, 1 week, 2 weeks, 3 weeks, 1 month, 2 months, 3 months, 6 months).

Read it simply: Ridge's RMSE sits within ~1% of the train-mean line at every horizon, so the
apparent win over "no change" is **drift, not skill** — the model learned "HNB usually goes up a
bit," not a signal.

### Return model — sign accuracy vs the "always guess the winning side" null

| horizon | dir_acc_ridge_% | dir_acc_xgb_% | up_share_test_% | dir_acc_alwaysguess_% | dir_edge_pp | beats_sign_null |
|---|---|---|---|---|---|---|
| 1 day | 47.2 | 47.5 | 43.8 | 56.2 | -8.6 | No |
| 1 week | 50.5 | 49.5 | 48.2 | 51.8 | -1.2 | No |
| 2 weeks | 54.8 | 50.6 | 54.6 | 54.6 | 0.2 | Yes |
| 3 weeks | 55.0 | 48.8 | 57.7 | 57.7 | -2.6 | No |
| 1 month | 56.0 | 47.7 | 59.1 | 59.1 | -3.1 | No |
| 2 months | 57.3 | 51.0 | 57.1 | 57.1 | 0.2 | Yes |
| 3 months | 55.9 | 56.4 | 64.6 | 64.6 | -8.3 | No |
| 6 months | 77.6 | 39.5 | 89.2 | 89.2 | -11.6 | No |
| 1 year | 20.4 | 24.0 | 95.1 | 95.1 | -71.1 | No |

**Beat the sign null: 2 of 9** (2 weeks, 2 months).

This is the trap to avoid: Ridge's sign accuracy looks decent (55-57% at 1-3 months, 78% at 6
months) — but the test window simply went UP that often. A constant "up" guess matches or beats
the model at every horizon. **No real sign skill anywhere.**

## Class balance (sanity)

| horizon_days | Buy | Hold | Sell |
|---|---|---|---|
| 1.0 | 30.8 | 39.5 | 29.7 |
| 5.0 | 37.8 | 28.4 | 33.9 |
| 10.0 | 42.4 | 26.4 | 31.2 |
| 15.0 | 44.2 | 24.9 | 30.9 |
| 22.0 | 44.7 | 26.6 | 28.6 |
| 44.0 | 50.5 | 24.3 | 25.1 |
| 66.0 | 55.9 | 24.0 | 20.1 |
| 132.0 | 78.2 | 15.6 | 6.1 |
| 252.0 | 93.7 | 4.5 | 1.8 |

## Caveats
- Long horizons overlap heavily. `indep_test_windows` = n_test / h is the honest sample size:
  at 252 days there are only ~2.2 independent windows — **do not
  claim significance there**, even if accuracy looks high.
- One stock, one split, Tier-1 features only. This is the Phase A floor to improve on.
- A high baseline accuracy at long horizons is NOT a good result — it just means the test window
  drifted one way (e.g. majority 78% at 6 months, persistence 77% at 1 year). It makes the
  baseline nearly unbeatable, which is exactly why the models look so bad there.
- Correlation, not causation.

## Next
Phase B — add Tier-2 (RSI, MACD, volume) and measure the accuracy GAIN per horizon.
