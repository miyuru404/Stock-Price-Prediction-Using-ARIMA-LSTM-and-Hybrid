# Phase Ablation — A (Tier-1) vs B (Tier-1 + Tier-2)

**Stock:** HNB daily · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252] trading days · **Method:** DIRECT (model per horizon)
**Phase A:** 9 features — ret_1, ret_5, ret_10, ma5_ratio, ma10_ratio, ma20_ratio, momentum_10, vol_10, vol_20
**Phase B:** 15 features — A + rsi_14, macd, macd_signal, macd_hist, volchg_5, volchg_20
**Identical** rows, split (80/20 chrono + h-bar purge), seeds and baselines in both phases.
Only the feature set changes, so `gain_pp` is the clean effect of Tier-2.

## BOTTOM LINE (caveman)

- **Tier-2 gain: -1.1 pp on average.** Positive at 1 of 9 horizons.
- **Horizons where Phase B beats the baseline: 0 of 9.**
- XGBoost spends **41%** of its importance on the Tier-2 features — it *uses* them,
  it just does not get *paid* for using them.
- Verdict: RSI, MACD and volume add essentially nothing. Noise-level wiggle, no horizon crosses the baseline. Technical indicators are exhausted — the missing information is not in the price chart.

## The gain table (the finding)

| horizon | baseline_% | A_best_% | B_best_% | gain_pp | A_edge_pp | B_edge_pp | B_beats_baseline |
|---|---|---|---|---|---|---|---|
| 1 day | 43.1 | 39.6 | 38.8 | -0.8 | -3.6 | -4.3 | No |
| 1 week | 37.2 | 36.1 | 35.0 | -1.1 | -1.1 | -2.2 | No |
| 2 weeks | 37.1 | 34.6 | 33.0 | -1.6 | -2.5 | -4.0 | No |
| 3 weeks | 43.2 | 36.2 | 36.0 | -0.2 | -7.0 | -7.1 | No |
| 1 month | 42.3 | 39.2 | 36.9 | -2.3 | -3.1 | -5.4 | No |
| 2 months | 50.5 | 37.4 | 36.7 | -0.7 | -13.2 | -13.8 | No |
| 3 months | 55.9 | 30.7 | 33.0 | 2.3 | -25.2 | -22.9 | No |
| 6 months | 78.4 | 36.6 | 34.3 | -2.3 | -41.8 | -44.1 | No |
| 1 year | 76.9 | 29.8 | 26.8 | -3.0 | -47.0 | -50.1 | No |

`gain_pp` = Phase B best model − Phase A best model. `edge_pp` = model − best baseline.
**Edge is what counts; gain only matters if it pushes edge above 0.**

## Full direction detail (both phases)

| horizon | phase | n_features | acc_logistic_% | acc_xgboost_% | acc_majority_% | acc_persistence_% | edge_pp | beats_baseline |
|---|---|---|---|---|---|---|---|---|
| 1 day | A (Tier-1) | 9 | 39.6 | 34.8 | 39.7 | 43.1 | -3.6 | No |
| 1 day | B (Tier-1+2) | 15 | 38.8 | 38.0 | 39.7 | 43.1 | -4.3 | No |
| 1 week | A (Tier-1) | 9 | 36.1 | 29.1 | 28.5 | 37.2 | -1.1 | No |
| 1 week | B (Tier-1+2) | 15 | 35.0 | 32.5 | 28.5 | 37.2 | -2.2 | No |
| 2 weeks | A (Tier-1) | 9 | 34.6 | 32.9 | 31.3 | 37.1 | -2.5 | No |
| 2 weeks | B (Tier-1+2) | 15 | 33.0 | 32.4 | 31.3 | 37.1 | -4.0 | No |
| 3 weeks | A (Tier-1) | 9 | 36.2 | 30.3 | 31.1 | 43.2 | -7.0 | No |
| 3 weeks | B (Tier-1+2) | 15 | 36.0 | 30.1 | 31.1 | 43.2 | -7.1 | No |
| 1 month | A (Tier-1) | 9 | 39.2 | 33.4 | 28.8 | 42.3 | -3.1 | No |
| 1 month | B (Tier-1+2) | 15 | 36.9 | 29.5 | 28.8 | 42.3 | -5.4 | No |
| 2 months | A (Tier-1) | 9 | 37.4 | 33.4 | 50.5 | 31.6 | -13.2 | No |
| 2 months | B (Tier-1+2) | 15 | 36.7 | 29.2 | 50.5 | 31.6 | -13.8 | No |
| 3 months | A (Tier-1) | 9 | 26.6 | 30.7 | 55.9 | 33.4 | -25.2 | No |
| 3 months | B (Tier-1+2) | 15 | 31.2 | 33.0 | 55.9 | 33.4 | -22.9 | No |
| 6 months | A (Tier-1) | 9 | 36.6 | 32.3 | 78.4 | 69.9 | -41.8 | No |
| 6 months | B (Tier-1+2) | 15 | 34.3 | 29.5 | 78.4 | 69.9 | -44.1 | No |
| 1 year | A (Tier-1) | 9 | 29.8 | 22.4 | 1.8 | 76.9 | -47.0 | No |
| 1 year | B (Tier-1+2) | 15 | 26.8 | 20.4 | 1.8 | 76.9 | -50.1 | No |

## Return % — did Tier-2 help there?

| horizon | A_ret_ratio | B_ret_ratio | ret_ratio_gain | A_sign_edge_pp | B_sign_edge_pp |
|---|---|---|---|---|---|
| 1 day | 0.991 | 0.991 | 0.0 | -8.5 | -7.1 |
| 1 week | 0.99 | 0.991 | 0.001 | -1.4 | -1.7 |
| 2 weeks | 0.989 | 0.996 | 0.007 | 0.2 | -2.5 |
| 3 weeks | 0.994 | 0.996 | 0.002 | -2.6 | -4.7 |
| 1 month | 0.994 | 0.995 | 0.001 | -3.0 | -6.5 |
| 2 months | 0.995 | 0.997 | 0.002 | 0.5 | 0.6 |
| 3 months | 0.98 | 0.987 | 0.007 | -8.8 | -12.4 |
| 6 months | 0.988 | 1.034 | 0.046 | -11.6 | -43.6 |
| 1 year | 1.001 | 1.025 | 0.024 | -71.1 | -71.2 |

`ret_ratio` = model RMSE ÷ train-mean-drift RMSE. **Below 1.0 = features helped.**
`ret_ratio_gain` **negative = Tier-2 improved it.**
`sign_edge_pp` = sign accuracy − "always guess the winning side". **Above 0 = real.**

## Where XGBoost put its attention in Phase B

| horizon | tier2_share_of_importance_% | top_feature | top_feature_is_tier2 |
|---|---|---|---|
| 1 day | 38.79999923706055 | vol_20 | False |
| 1 week | 40.0 | vol_20 | False |
| 2 weeks | 41.400001525878906 | macd_signal | True |
| 3 weeks | 41.900001525878906 | macd_signal | True |
| 1 month | 41.900001525878906 | vol_20 | False |
| 2 months | 40.79999923706055 | macd_signal | True |
| 3 months | 41.599998474121094 | macd_signal | True |
| 6 months | 42.70000076293945 | macd_signal | True |
| 1 year | 44.0 | macd_signal | True |

## Caveats
- Phase A numbers here differ slightly from `results/direction/multi_horizon/` because MACD's
  35-bar warmup removes ~15 extra early rows from BOTH phases. This table is the fair A/B.
- Long horizons overlap: at 252 days only ~2.2 independent
  windows exist. No significance claims there.
- A -1.1 pp average shift is inside noise for this sample size. The honest reading is
  "no effect", not "Tier-2 actively hurts".
- One stock, one split. Correlation, not causation.

## Next
Phase C — macro (interest rates / spread, then inflation, FX, M2). Rates data is already in
`cleaned_data/interest_rates_monthly.csv`. Monthly data suits the LONG horizons (22d+), which is
exactly where technicals are weakest.
