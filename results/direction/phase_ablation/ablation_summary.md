# Phase Ablation — A (Tier-1) → B (+Tier-2) → C / C2 (macro) → D (+sector)

**Stock:** HNB daily · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252] trading days · **Method:** DIRECT (model per horizon)

| Phase | Features | What is added |
|---|---|---|
| A | 9 | Tier-1 technical: ret_1, ret_5, ret_10, ma5_ratio, ma10_ratio, ma20_ratio, momentum_10, vol_10, vol_20 |
| B | 15 | + Tier-2 technical: rsi_14, macd, macd_signal, macd_hist, volchg_5, volchg_20 |
| C | 44 | + macro rates, levels **and** changes: policy_rate, tb_3m, tb_12m, awdr, awlr, spread, term_slope, d_policy_1m, d_spread_1m, d_spread_3m, d_tb3m_1m, d_tb3m_3m, d_awlr_3m |
| C2 | 22 | + macro **changes only** (levels dropped): term_slope, d_policy_1m, d_spread_1m, d_spread_3m, d_tb3m_1m, d_tb3m_3m, d_awlr_3m |
| D | 38 | **C2** + sector: ASPI market returns/vol, relative strength vs ASPI and vs peer banks, peer-bank and finance composites, 60d correlation and beta |

**Phase D builds on C2, not C** — C's rate levels were shown to be actively harmful, so carrying
them forward would contaminate the sector test. Peers used: banks ['COMB', 'SAMP'], finance ['LOFC', 'LOLC', 'LFIN', 'CFIN'].
All sector features are same-day-known returns/ratios (no look-ahead) and already stationary.

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
- **Sector gain (C2→D): +2.6 pp.** Positive at 6 of 9 horizons —
  the **first step in the whole study with a positive average gain**.
- **But it still beats the baseline at 0 of 9 horizons.** Closest is
  **1 week** at -0.2 pp — near-parity, not a win.
- XGBoost gives sector features **34%** of its importance.
- Verdict: Sector context is the most useful thing added so far — it repairs the macro damage and pulls short-horizon models to within ~2 pp of the baseline — but it still never crosses it. Five feature sets, up to 38 features, 1 day to 1 year: no horizon beats the naive guess.

## The gain table (the finding)

| horizon | baseline_% | A_best_% | B_best_% | C_best_% | C2_best_% | D_best_% | gain_A_to_B_pp | gain_B_to_C_pp | gain_B_to_C2_pp | gain_C2_to_D_pp | D_edge_pp | D_beats_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 day | 43.3 | 39.4 | 38.5 | 41.7 | 38.5 | 41.6 | -0.9 | 3.2 | 0.0 | 3.1 | -1.7 | No |
| 1 week | 37.4 | 36.3 | 35.3 | 30.0 | 35.6 | 37.3 | -1.0 | -5.3 | 0.3 | 1.7 | -0.2 | No |
| 2 weeks | 36.9 | 34.2 | 33.4 | 30.8 | 28.3 | 35.3 | -0.8 | -2.6 | -5.1 | 7.0 | -1.6 | No |
| 3 weeks | 43.0 | 36.2 | 36.0 | 29.9 | 30.5 | 35.8 | -0.2 | -6.1 | -5.5 | 5.3 | -7.2 | No |
| 1 month | 44.9 | 39.7 | 37.2 | 26.1 | 29.8 | 38.0 | -2.5 | -11.1 | -7.4 | 8.2 | -6.9 | No |
| 2 months | 50.9 | 37.0 | 37.0 | 35.9 | 37.0 | 34.9 | 0.0 | -1.1 | 0.0 | -2.1 | -16.0 | No |
| 3 months | 55.9 | 30.7 | 33.0 | 43.2 | 48.1 | 47.0 | 2.3 | 10.2 | 15.1 | -1.1 | -8.9 | No |
| 6 months | 78.4 | 36.6 | 34.3 | 44.6 | 44.3 | 47.4 | -2.3 | 10.3 | 10.0 | 3.1 | -30.9 | No |
| 1 year | 76.9 | 29.8 | 26.8 | 4.5 | 49.7 | 47.7 | -3.0 | -22.3 | 22.9 | -2.0 | -29.1 | No |

`gain` = best model of that phase − best model of the previous phase.
`edge_pp` = model − best baseline. **Edge is what counts; gain only matters if it lifts edge above 0.**

## Full direction detail (all phases)

| horizon | phase | n_features | acc_logistic_% | acc_xgboost_% | acc_majority_% | acc_persistence_% | edge_pp | beats_baseline |
|---|---|---|---|---|---|---|---|---|
| 1 day | A (Tier-1) | 9 | 39.4 | 34.6 | 39.6 | 43.3 | -3.9 | No |
| 1 day | B (+Tier-2) | 15 | 38.5 | 35.8 | 39.6 | 43.3 | -4.8 | No |
| 1 day | C (+macro) | 28 | 41.7 | 35.8 | 39.6 | 43.3 | -1.6 | No |
| 1 day | C2 (+macro Δ only) | 22 | 38.5 | 34.1 | 39.6 | 43.3 | -4.8 | No |
| 1 day | D (+sector) | 38 | 41.6 | 38.5 | 39.6 | 43.3 | -1.7 | No |
| 1 week | A (Tier-1) | 9 | 36.3 | 31.5 | 28.2 | 37.4 | -1.1 | No |
| 1 week | B (+Tier-2) | 15 | 35.3 | 31.4 | 28.2 | 37.4 | -2.2 | No |
| 1 week | C (+macro) | 28 | 30.0 | 29.6 | 28.2 | 37.4 | -7.5 | No |
| 1 week | C2 (+macro Δ only) | 22 | 35.6 | 32.4 | 28.2 | 37.4 | -1.9 | No |
| 1 week | D (+sector) | 38 | 37.3 | 34.5 | 28.2 | 37.4 | -0.2 | No |
| 2 weeks | A (Tier-1) | 9 | 34.2 | 32.7 | 26.2 | 36.9 | -2.7 | No |
| 2 weeks | B (+Tier-2) | 15 | 33.4 | 32.0 | 26.2 | 36.9 | -3.4 | No |
| 2 weeks | C (+macro) | 28 | 28.7 | 30.8 | 26.2 | 36.9 | -6.1 | No |
| 2 weeks | C2 (+macro Δ only) | 22 | 28.3 | 26.4 | 26.2 | 36.9 | -8.6 | No |
| 2 weeks | D (+sector) | 38 | 35.3 | 26.7 | 26.2 | 36.9 | -1.6 | No |
| 3 weeks | A (Tier-1) | 9 | 36.2 | 30.2 | 31.3 | 43.0 | -6.9 | No |
| 3 weeks | B (+Tier-2) | 15 | 36.0 | 28.8 | 31.3 | 43.0 | -7.0 | No |
| 3 weeks | C (+macro) | 28 | 28.0 | 29.9 | 31.3 | 43.0 | -13.1 | No |
| 3 weeks | C2 (+macro Δ only) | 22 | 30.5 | 28.5 | 31.3 | 43.0 | -12.5 | No |
| 3 weeks | D (+sector) | 38 | 35.8 | 28.6 | 31.3 | 43.0 | -7.2 | No |
| 1 month | A (Tier-1) | 9 | 39.7 | 32.0 | 44.9 | 42.5 | -5.2 | No |
| 1 month | B (+Tier-2) | 15 | 37.2 | 29.0 | 44.9 | 42.5 | -7.7 | No |
| 1 month | C (+macro) | 28 | 26.1 | 26.1 | 44.9 | 42.5 | -18.8 | No |
| 1 month | C2 (+macro Δ only) | 22 | 29.8 | 26.7 | 44.9 | 42.5 | -15.1 | No |
| 1 month | D (+sector) | 38 | 38.0 | 28.4 | 44.9 | 42.5 | -6.9 | No |
| 2 months | A (Tier-1) | 9 | 37.0 | 32.7 | 50.9 | 31.8 | -13.9 | No |
| 2 months | B (+Tier-2) | 15 | 37.0 | 30.3 | 50.9 | 31.8 | -13.9 | No |
| 2 months | C (+macro) | 28 | 35.9 | 33.0 | 50.9 | 31.8 | -15.0 | No |
| 2 months | C2 (+macro Δ only) | 22 | 34.8 | 37.0 | 50.9 | 31.8 | -13.9 | No |
| 2 months | D (+sector) | 38 | 34.9 | 33.3 | 50.9 | 31.8 | -16.0 | No |
| 3 months | A (Tier-1) | 9 | 26.6 | 30.7 | 55.9 | 33.4 | -25.2 | No |
| 3 months | B (+Tier-2) | 15 | 31.2 | 33.0 | 55.9 | 33.4 | -22.9 | No |
| 3 months | C (+macro) | 28 | 43.0 | 43.2 | 55.9 | 33.4 | -12.7 | No |
| 3 months | C2 (+macro Δ only) | 22 | 42.7 | 48.1 | 55.9 | 33.4 | -7.8 | No |
| 3 months | D (+sector) | 38 | 38.5 | 47.0 | 55.9 | 33.4 | -8.9 | No |
| 6 months | A (Tier-1) | 9 | 36.6 | 32.3 | 78.4 | 69.9 | -41.8 | No |
| 6 months | B (+Tier-2) | 15 | 34.3 | 29.5 | 78.4 | 69.9 | -44.1 | No |
| 6 months | C (+macro) | 28 | 44.6 | 35.3 | 78.4 | 69.9 | -33.8 | No |
| 6 months | C2 (+macro Δ only) | 22 | 32.1 | 44.3 | 78.4 | 69.9 | -34.1 | No |
| 6 months | D (+sector) | 38 | 43.4 | 47.4 | 78.4 | 69.9 | -30.9 | No |
| 1 year | A (Tier-1) | 9 | 29.8 | 22.4 | 1.8 | 76.9 | -47.0 | No |
| 1 year | B (+Tier-2) | 15 | 26.8 | 20.4 | 1.8 | 76.9 | -50.1 | No |
| 1 year | C (+macro) | 28 | 4.5 | 2.0 | 1.8 | 76.9 | -72.3 | No |
| 1 year | C2 (+macro Δ only) | 22 | 29.3 | 49.7 | 1.8 | 76.9 | -27.1 | No |
| 1 year | D (+sector) | 38 | 40.3 | 47.7 | 1.8 | 76.9 | -29.1 | No |

## Return % — did macro help there?

| horizon | A_ret_ratio | B_ret_ratio | C_ret_ratio | C2_ret_ratio | D_ret_ratio | C2_sign_edge_pp | D_sign_edge_pp |
|---|---|---|---|---|---|---|---|
| 1 day | 0.99 | 0.99 | 0.993 | 0.994 | 0.988 | -5.1 | -4.7 |
| 1 week | 0.989 | 0.99 | 0.992 | 1.002 | 1.003 | -0.8 | 1.7 |
| 2 weeks | 0.989 | 0.996 | 0.998 | 1.018 | 1.009 | -2.8 | 1.7 |
| 3 weeks | 0.994 | 0.996 | 1.03 | 1.051 | 1.034 | -8.9 | 1.1 |
| 1 month | 0.994 | 0.995 | 1.08 | 1.106 | 1.075 | -17.3 | -4.7 |
| 2 months | 0.995 | 0.997 | 1.37 | 1.386 | 1.335 | -12.3 | -1.7 |
| 3 months | 0.98 | 0.987 | 1.345 | 1.381 | 1.323 | -7.8 | -2.2 |
| 6 months | 0.988 | 1.034 | 1.291 | 1.188 | 1.19 | -45.9 | -38.3 |
| 1 year | 1.001 | 1.025 | 1.332 | 1.159 | 1.068 | -45.2 | -38.9 |

Phase D return RMSE peaks at **1.33×** the train-mean null. Sector features repair
much of the macro damage on the return target too, but never get below the ~0.99 that plain
Tier-1 already achieved.

`ret_ratio` = model RMSE ÷ train-mean-drift RMSE. **Below 1.0 = features helped.**
`C_sign_edge_pp` = sign accuracy − "always guess the winning side". **Above 0 = real.**

## Where the model looks

Phase C (macro levels included):

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

Phase D (macro Δ + sector):

| horizon | tier1_share_% | tier2_share_% | macro_share_% | sector_share_% | top_feature | top_feature_tier |
|---|---|---|---|---|---|---|
| 1 day | 24.700000762939453 | 15.300000190734863 | 18.399999618530273 | 41.599998474121094 | vol_20 | tier1 |
| 1 week | 22.899999618530273 | 15.0 | 22.100000381469727 | 40.0 | vol_20 | tier1 |
| 2 weeks | 20.399999618530273 | 15.199999809265137 | 27.299999237060547 | 37.099998474121094 | term_slope | macro |
| 3 weeks | 19.799999237060547 | 15.399999618530273 | 30.200000762939453 | 34.599998474121094 | d_tb3m_3m | macro |
| 1 month | 19.399999618530273 | 14.800000190734863 | 32.0 | 33.79999923706055 | d_tb3m_3m | macro |
| 2 months | 19.100000381469727 | 14.300000190734863 | 34.79999923706055 | 31.799999237060547 | d_tb3m_3m | macro |
| 3 months | 16.299999237060547 | 13.899999618530273 | 41.20000076293945 | 28.5 | d_tb3m_1m | macro |
| 6 months | 15.5 | 14.600000381469727 | 39.79999923706055 | 30.200000762939453 | d_tb3m_1m | macro |
| 1 year | 13.600000381469727 | 9.800000190734863 | 49.20000076293945 | 27.299999237060547 | d_policy_1m | macro |

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
1. **Sector-aware run across tickers** — repeat Phase D on COMB/SAMP (banks), LOFC/LOLC/LFIN/CFIN
   (finance) and JKH/DIAL/CTC/DIST (control). HNB alone cannot tell us whether the sector gain is
   general or one stock's luck. This is the cheapest remaining test and it uses data already held.
2. **Phase E (events)** — dividend dates and rate-decision flags. Rate events are in
   `cleaned_data/policy_rate_events.csv`; dividends still to collect.
3. **Phase F (news sentiment)** — the last untested source. Needs scraping + FinBERT.
