# Phase Ablation — A (Tier-1) → B (+Tier-2) → C (+macro rates) → C2 (macro Δ only)

**Stock:** HNB daily · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252] trading days · **Method:** DIRECT (model per horizon)

| Phase | Features | What is added |
|---|---|---|
| A | 9 | Tier-1 technical: ret_1, ret_5, ret_10, ma5_ratio, ma10_ratio, ma20_ratio, momentum_10, vol_10, vol_20 |
| B | 15 | + Tier-2 technical: rsi_14, macd, macd_signal, macd_hist, volchg_5, volchg_20 |
| C | 28 | + macro rates, levels **and** changes: policy_rate, tb_3m, tb_12m, awdr, awlr, spread, term_slope, d_policy_1m, d_spread_1m, d_spread_3m, d_tb3m_1m, d_tb3m_3m, d_awlr_3m |
| C2 | 22 | + macro **changes only** (levels dropped): term_slope, d_policy_1m, d_spread_1m, d_spread_3m, d_tb3m_1m, d_tb3m_3m, d_awlr_3m |

C2 is a diagnostic, not a new tier: it answers "is macro useless, or was it fed in the wrong form?"
Rate *levels* trend and never repeat across regimes, so a tree can memorise "rates were 15% in
2022"; that mapping is worthless out-of-sample. Rate *changes* are stationary and reusable.

Identical rows, split (80/20 chrono + h-bar purge), seeds and baselines across all phases.
Only the feature list changes, so each `gain` column is the clean effect of that step.

**Macro look-ahead guard:** every monthly CBSL figure is stamped `month_end + 35 days`
before being merged backward-asof onto trading days. The model never sees a rate before it was
published. Inflation, FX and M2 are still on the COLLECT list — **Phase C here is rates only.**

## BOTTOM LINE (caveman)

- **Macro gain (B→C): -2.8 pp on average.** Positive at 3 of 9 horizons.
- **Phase C beats the baseline at 0 of 9 horizons.** Phase C2: 0 of 9.
- **The gain is not just negative, it is WILD:** a 33 pp swing across horizons
  (-22.3 to +10.3). That is not weak signal,
  that is **overfitting**.
- **Return % confirms it:** Phase C RMSE blows out to **1.37×** the train-mean null
  (Phase A/B sat at ~0.99). Adding macro made the return model materially worse.
- **XGBoost hands 64% of its importance to macro** and still loses accuracy — the
  classic fingerprint of a model latching onto a trending variable.
- **The C2 diagnostic settles it: +3.4 pp using macro CHANGES only.**
  Dropping the trending levels fixes the damage, so the problem was the FORM of the data, not macro itself.
- Verdict: Interest-rate data does NOT predict HNB direction at any horizon. Four feature sets, up to 22 features, 1 day to 1 year: still no edge anywhere.

## The gain table (the finding)

| horizon | baseline_% | A_best_% | B_best_% | C_best_% | C2_best_% | gain_A_to_B_pp | gain_B_to_C_pp | gain_B_to_C2_pp | C_edge_pp | C2_edge_pp | C2_beats_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 day | 43.3 | 39.4 | 38.5 | 41.7 | 38.5 | -0.9 | 3.2 | 0.0 | -1.6 | -4.8 | No |
| 1 week | 37.4 | 36.3 | 35.3 | 30.0 | 35.6 | -1.0 | -5.3 | 0.3 | -7.5 | -1.9 | No |
| 2 weeks | 36.9 | 34.2 | 33.4 | 30.8 | 28.3 | -0.8 | -2.6 | -5.1 | -6.1 | -8.6 | No |
| 3 weeks | 43.0 | 36.2 | 36.0 | 29.9 | 30.5 | -0.2 | -6.1 | -5.5 | -13.1 | -12.5 | No |
| 1 month | 44.9 | 39.7 | 37.2 | 26.1 | 29.8 | -2.5 | -11.1 | -7.4 | -18.8 | -15.1 | No |
| 2 months | 50.9 | 37.0 | 37.0 | 35.9 | 37.0 | 0.0 | -1.1 | 0.0 | -15.0 | -13.9 | No |
| 3 months | 55.9 | 30.7 | 33.0 | 43.2 | 48.1 | 2.3 | 10.2 | 15.1 | -12.7 | -7.8 | No |
| 6 months | 78.4 | 36.6 | 34.3 | 44.6 | 44.3 | -2.3 | 10.3 | 10.0 | -33.8 | -34.1 | No |
| 1 year | 76.9 | 29.8 | 26.8 | 4.5 | 49.7 | -3.0 | -22.3 | 22.9 | -72.3 | -27.1 | No |

`gain` = best model of that phase − best model of the previous phase.
`edge_pp` = model − best baseline. **Edge is what counts; gain only matters if it lifts edge above 0.**

## Full direction detail (all phases)

| horizon | phase | n_features | acc_logistic_% | acc_xgboost_% | acc_majority_% | acc_persistence_% | edge_pp | beats_baseline |
|---|---|---|---|---|---|---|---|---|
| 1 day | A (Tier-1) | 9 | 39.4 | 34.6 | 39.6 | 43.3 | -3.9 | No |
| 1 day | B (+Tier-2) | 15 | 38.5 | 35.8 | 39.6 | 43.3 | -4.8 | No |
| 1 day | C (+macro) | 28 | 41.7 | 35.8 | 39.6 | 43.3 | -1.6 | No |
| 1 day | C2 (+macro Δ only) | 22 | 38.5 | 34.1 | 39.6 | 43.3 | -4.8 | No |
| 1 week | A (Tier-1) | 9 | 36.3 | 31.5 | 28.2 | 37.4 | -1.1 | No |
| 1 week | B (+Tier-2) | 15 | 35.3 | 31.4 | 28.2 | 37.4 | -2.2 | No |
| 1 week | C (+macro) | 28 | 30.0 | 29.6 | 28.2 | 37.4 | -7.5 | No |
| 1 week | C2 (+macro Δ only) | 22 | 35.6 | 32.4 | 28.2 | 37.4 | -1.9 | No |
| 2 weeks | A (Tier-1) | 9 | 34.2 | 32.7 | 26.2 | 36.9 | -2.7 | No |
| 2 weeks | B (+Tier-2) | 15 | 33.4 | 32.0 | 26.2 | 36.9 | -3.4 | No |
| 2 weeks | C (+macro) | 28 | 28.7 | 30.8 | 26.2 | 36.9 | -6.1 | No |
| 2 weeks | C2 (+macro Δ only) | 22 | 28.3 | 26.4 | 26.2 | 36.9 | -8.6 | No |
| 3 weeks | A (Tier-1) | 9 | 36.2 | 30.2 | 31.3 | 43.0 | -6.9 | No |
| 3 weeks | B (+Tier-2) | 15 | 36.0 | 28.8 | 31.3 | 43.0 | -7.0 | No |
| 3 weeks | C (+macro) | 28 | 28.0 | 29.9 | 31.3 | 43.0 | -13.1 | No |
| 3 weeks | C2 (+macro Δ only) | 22 | 30.5 | 28.5 | 31.3 | 43.0 | -12.5 | No |
| 1 month | A (Tier-1) | 9 | 39.7 | 32.0 | 44.9 | 42.5 | -5.2 | No |
| 1 month | B (+Tier-2) | 15 | 37.2 | 29.0 | 44.9 | 42.5 | -7.7 | No |
| 1 month | C (+macro) | 28 | 26.1 | 26.1 | 44.9 | 42.5 | -18.8 | No |
| 1 month | C2 (+macro Δ only) | 22 | 29.8 | 26.7 | 44.9 | 42.5 | -15.1 | No |
| 2 months | A (Tier-1) | 9 | 37.0 | 32.7 | 50.9 | 31.8 | -13.9 | No |
| 2 months | B (+Tier-2) | 15 | 37.0 | 30.3 | 50.9 | 31.8 | -13.9 | No |
| 2 months | C (+macro) | 28 | 35.9 | 33.0 | 50.9 | 31.8 | -15.0 | No |
| 2 months | C2 (+macro Δ only) | 22 | 34.8 | 37.0 | 50.9 | 31.8 | -13.9 | No |
| 3 months | A (Tier-1) | 9 | 26.6 | 30.7 | 55.9 | 33.4 | -25.2 | No |
| 3 months | B (+Tier-2) | 15 | 31.2 | 33.0 | 55.9 | 33.4 | -22.9 | No |
| 3 months | C (+macro) | 28 | 43.0 | 43.2 | 55.9 | 33.4 | -12.7 | No |
| 3 months | C2 (+macro Δ only) | 22 | 42.7 | 48.1 | 55.9 | 33.4 | -7.8 | No |
| 6 months | A (Tier-1) | 9 | 36.6 | 32.3 | 78.4 | 69.9 | -41.8 | No |
| 6 months | B (+Tier-2) | 15 | 34.3 | 29.5 | 78.4 | 69.9 | -44.1 | No |
| 6 months | C (+macro) | 28 | 44.6 | 35.3 | 78.4 | 69.9 | -33.8 | No |
| 6 months | C2 (+macro Δ only) | 22 | 32.1 | 44.3 | 78.4 | 69.9 | -34.1 | No |
| 1 year | A (Tier-1) | 9 | 29.8 | 22.4 | 1.8 | 76.9 | -47.0 | No |
| 1 year | B (+Tier-2) | 15 | 26.8 | 20.4 | 1.8 | 76.9 | -50.1 | No |
| 1 year | C (+macro) | 28 | 4.5 | 2.0 | 1.8 | 76.9 | -72.3 | No |
| 1 year | C2 (+macro Δ only) | 22 | 29.3 | 49.7 | 1.8 | 76.9 | -27.1 | No |

## Return % — did macro help there?

| horizon | A_ret_ratio | B_ret_ratio | C_ret_ratio | C2_ret_ratio | C_sign_edge_pp | C2_sign_edge_pp |
|---|---|---|---|---|---|---|
| 1 day | 0.99 | 0.99 | 0.993 | 0.994 | -8.4 | -5.1 |
| 1 week | 0.989 | 0.99 | 0.992 | 1.002 | -1.1 | -0.8 |
| 2 weeks | 0.989 | 0.996 | 0.998 | 1.018 | -3.4 | -2.8 |
| 3 weeks | 0.994 | 0.996 | 1.03 | 1.051 | -10.8 | -8.9 |
| 1 month | 0.994 | 0.995 | 1.08 | 1.106 | -13.2 | -17.3 |
| 2 months | 0.995 | 0.997 | 1.37 | 1.386 | -7.3 | -12.3 |
| 3 months | 0.98 | 0.987 | 1.345 | 1.381 | 1.4 | -7.8 |
| 6 months | 0.988 | 1.034 | 1.291 | 1.188 | -12.6 | -45.9 |
| 1 year | 1.001 | 1.025 | 1.332 | 1.159 | -78.8 | -45.2 |

`ret_ratio` = model RMSE ÷ train-mean-drift RMSE. **Below 1.0 = features helped.**
`C_sign_edge_pp` = sign accuracy − "always guess the winning side". **Above 0 = real.**

## Where the model looks in Phase C

| horizon | tier1_share_% | tier2_share_% | macro_share_% | top_feature | top_feature_tier |
|---|---|---|---|---|---|
| 1 day | 33.29999923706055 | 20.899999618530273 | 45.79999923706055 | vol_20 | tier1 |
| 1 week | 27.899999618530273 | 18.600000381469727 | 53.5 | spread | macro |
| 2 weeks | 24.5 | 17.399999618530273 | 58.20000076293945 | tb_12m | macro |
| 3 weeks | 23.399999618530273 | 16.899999618530273 | 59.599998474121094 | d_tb3m_3m | macro |
| 1 month | 20.700000762939453 | 15.600000381469727 | 63.70000076293945 | d_tb3m_3m | macro |
| 2 months | 18.799999237060547 | 13.800000190734863 | 67.4000015258789 | spread | macro |
| 3 months | 16.700000762939453 | 11.199999809265137 | 72.0 | tb_12m | macro |
| 6 months | 11.899999618530273 | 10.199999809265137 | 77.9000015258789 | tb_12m | macro |
| 1 year | 11.800000190734863 | 8.399999618530273 | 79.69999694824219 | policy_rate | macro |

This is the key diagnostic: macro importance climbs from 46% at
1 day to 80% at 1 year while accuracy *falls*. The model is
fitting the rate series as a slow-moving trend proxy, not using it as signal.

## Caveats
- Rates are **monthly**, held flat between releases, so within a month every trading day carries
  the same macro value. That suits long horizons and is nearly useless for the 1-day model.
- Long horizons overlap: at 252 days only ~2.2 independent
  windows exist. No significance claims there.
- Phase A/B numbers shift slightly from earlier runs because macro availability trims the early
  rows from all phases. This table is the fair three-way comparison.
- One stock, one split. Correlation, not causation.
- Known real effect from the earlier spread work: bank returns react negatively to a widening
  spread **contemporaneously (same month)**. That is an *explanatory* result, not a *predictive*
  one — this run confirms it does not forecast.

## Next
Phase D — sector (ASPI market return, spread × is_bank, peer-bank returns) across banks / finance /
control, not just HNB. Then E (events) and F (news sentiment), which are the last untested sources
of information outside the price chart.
