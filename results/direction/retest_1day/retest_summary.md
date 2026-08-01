# Re-test of the 1-day claim — walk-forward, 21 independent windows

## The claim being tested (pre-registered)

> *"Pooled over the 7 banking/finance stocks, the model beats each stock's own baseline at the
> 1-day horizon: 6/7 stocks positive, median +2.5 pp, sign-test p = 0.062."*

It came from **one** train/test split, p was **above** 0.05, and it was 1 horizon out of 9 tested.

## Method

- Expanding training window → fresh **6-month** test window → roll forward.
  **21 folds**, non-overlapping test windows, retrained from scratch each time.
- Scored **per stock against that stock's own baseline** (the fairness lesson from the last run).
- **Both models reported separately.** Taking max(Logistic, XGBoost) each fold would cherry-pick a
  winner every time — that optimism is part of how the original +2.5 pp arose.
- **1 week carried as a control.** It already failed; it should stay flat.
- **Verdict rule fixed in advance:** survives only if fold medians are positive in clearly more
  than half the folds AND the sign test gives p < 0.05.

## VERDICT

| horizon | model | folds_positive | median_of_fold_medians | mean_of_fold_medians | sign_test_p | wilcoxon_p | SURVIVES |
|---|---|---|---|---|---|---|---|
| 1 day | Logistic | 7/21 | -1.8 | -1.8 | 0.9608 | 0.9164 | NO |
| 1 day | XGBoost | 6/21 | -1.9 | -1.79 | 0.9867 | 0.966 | NO |
| 1 week (control) | Logistic | 7/21 | -0.9 | -3.01 | 0.9608 | 0.9797 | NO |
| 1 week (control) | XGBoost | 7/21 | -6.0 | -5.39 | 0.9608 | 0.9938 | NO |

**The 1-day claim DOES NOT SURVIVE.**

## Fold-by-fold (1 day)

| test_from | test_to | median_edge_logistic | median_edge_xgboost | stocks_pos_logistic | stocks_pos_xgboost | n_stocks |
|---|---|---|---|---|---|---|
| 2016-01 | 2016-07 | -4.5 | -1.7 | 3 | 3 | 7 |
| 2016-07 | 2017-01 | -10.5 | -7.6 | 1 | 1 | 7 |
| 2017-01 | 2017-07 | -9.6 | -4.0 | 1 | 2 | 7 |
| 2017-07 | 2018-01 | -5.3 | -5.3 | 2 | 3 | 7 |
| 2018-01 | 2018-07 | -1.9 | -4.4 | 2 | 1 | 7 |
| 2018-07 | 2019-01 | 1.9 | -1.9 | 4 | 2 | 7 |
| 2019-01 | 2019-07 | -4.3 | -7.0 | 2 | 2 | 7 |
| 2019-07 | 2020-01 | -1.8 | -3.5 | 3 | 2 | 7 |
| 2020-01 | 2020-07 | -1.3 | -9.8 | 3 | 2 | 7 |
| 2020-07 | 2021-01 | -7.2 | -4.0 | 1 | 3 | 7 |
| 2021-01 | 2021-07 | 4.4 | 1.8 | 4 | 4 | 7 |
| 2021-07 | 2022-01 | -3.2 | -1.6 | 3 | 3 | 7 |
| 2022-01 | 2022-07 | 2.1 | -3.2 | 4 | 2 | 7 |
| 2022-07 | 2023-01 | 5.1 | 9.4 | 5 | 5 | 7 |
| 2023-01 | 2023-07 | 3.4 | -0.8 | 7 | 3 | 7 |
| 2023-07 | 2024-01 | -1.6 | 4.1 | 3 | 4 | 7 |
| 2024-01 | 2024-07 | 3.5 | 6.1 | 4 | 4 | 7 |
| 2024-07 | 2025-01 | 3.2 | 1.6 | 4 | 5 | 7 |
| 2025-01 | 2025-07 | -7.9 | 0.9 | 1 | 4 | 7 |
| 2025-07 | 2026-01 | -2.4 | -1.6 | 3 | 2 | 7 |
| 2026-01 | 2026-07 | 0.0 | -5.1 | 3 | 3 | 7 |

## Per stock, across all folds (1 day)

| ticker | folds | median_edge_logistic | median_edge_xgboost | pct_folds_pos_xgb |
|---|---|---|---|---|
| LOFC | 21 | 0.8 | 4.9 | 61.9 |
| CFIN | 21 | 3.0 | 2.6 | 61.9 |
| LFIN | 21 | 0.0 | -1.0 | 42.9 |
| COMB | 21 | -4.3 | -1.7 | 38.1 |
| LOLC | 21 | -2.4 | -3.5 | 28.6 |
| HNB | 21 | -1.9 | -4.4 | 19.0 |
| SAMP | 21 | -5.2 | -4.9 | 33.3 |

## Caveats
- 6-month test windows at a 1-day horizon are effectively independent, but the 7 stocks inside a
  fold are not independent of each other — that is why the fold, not the stock, is the unit of the
  sign test here.
- Walk-forward retrains ~21 times per model, so this is a much harder test than the single
  80/20 split every earlier phase used. A result that survives this is worth believing.
