# LSTM / GRU for direction — the last untested model class

**Why this is a different test:** every earlier model read a FLAT ROW of summary features
(`ret_5`, `vol_20`, `rsi_14`, …), which throws away the ORDER of recent days. A recurrent net reads
the raw **sequence** of the last **30 days**, so it can react to shape that a summary cannot
express. Different hypothesis, not just a different algorithm.

**Setup:** pooled over ['HNB', 'COMB', 'SAMP', 'LOFC', 'LOLC', 'LFIN', 'CFIN']; sequence features ['ret_1', 'vol_chg', 'hl_range', 'aspi_ret_1', 'peer_ret_1', 'rel_aspi'];
3 seeds per model (majority vote — recurrent nets swing with seed and this project has
been fooled by seed noise before); walk-forward 12-month test windows; scored per stock
against that stock's own baseline. Monthly macro deliberately excluded — inside a 30-day
window it is a flat line and tells a sequence model nothing.

**Logistic(flat)** is the control: the SAME sequences, flattened into one long row. If the recurrent
nets beat it, sequence structure genuinely matters. If not, the ordering carries nothing.

## VERDICT

| horizon | model | folds_positive | median_of_fold_medians_pp | sign_test_p | beats_baseline |
|---|---|---|---|---|---|
| 1 day | LSTM | 5/7 | 2.9 | 0.2266 | NO |
| 1 day | GRU | 5/7 | 1.3 | 0.2266 | NO |
| 1 day | Logistic(flat) | 3/7 | -1.7 | 0.7734 | NO |
| 1 week | LSTM | 2/7 | -0.4 | 0.9375 | NO |
| 1 week | GRU | 3/7 | 0.0 | 0.7734 | NO |
| 1 week | Logistic(flat) | 0/7 | -2.1 | 1.0 | NO |
| 1 month | LSTM | 1/7 | -5.4 | 0.9922 | NO |
| 1 month | GRU | 2/7 | -8.3 | 0.9375 | NO |
| 1 month | Logistic(flat) | 2/7 | -9.9 | 0.9375 | NO |

**No recurrent model beats the naive baseline at any horizon. Sequence structure adds nothing.**

## Fold by fold (median per-stock edge, pp)

| horizon_days | fold | test_from | med_LSTM | med_GRU | med_Logistic(flat) |
|---|---|---|---|---|---|
| 1 | 0 | 2019-01 | -1.3 | -2.8 | -5.4 |
| 1 | 1 | 2020-01 | -2.0 | -2.2 | -3.0 |
| 1 | 2 | 2021-01 | 2.5 | 0.8 | -1.7 |
| 1 | 3 | 2022-01 | 3.6 | 6.2 | 3.6 |
| 1 | 4 | 2023-01 | 7.9 | 7.9 | 5.0 |
| 1 | 5 | 2024-01 | 5.0 | 3.8 | 2.9 |
| 1 | 6 | 2025-01 | 2.9 | 1.3 | -2.1 |
| 5 | 0 | 2019-01 | 3.3 | 0.0 | -1.6 |
| 5 | 1 | 2020-01 | -1.0 | -1.0 | 0.0 |
| 5 | 2 | 2021-01 | 0.0 | -5.9 | -8.0 |
| 5 | 3 | 2022-01 | -1.6 | 4.7 | 0.0 |
| 5 | 4 | 2023-01 | -6.6 | -8.7 | -5.8 |
| 5 | 5 | 2024-01 | -0.4 | 3.3 | -2.5 |
| 5 | 6 | 2025-01 | 0.8 | 0.4 | -2.1 |
| 22 | 0 | 2019-01 | 0.0 | 5.0 | 1.3 |
| 22 | 1 | 2020-01 | -12.2 | -8.3 | -16.8 |
| 22 | 2 | 2021-01 | 5.5 | 0.4 | -9.8 |
| 22 | 3 | 2022-01 | -17.8 | -11.4 | 1.6 |
| 22 | 4 | 2023-01 | -10.7 | -18.6 | -9.9 |
| 22 | 5 | 2024-01 | -5.4 | -8.3 | -20.0 |
| 22 | 6 | 2025-01 | -5.0 | -3.8 | -12.2 |

## Reading it simply

- If **LSTM/GRU ≈ Logistic(flat)**, the order of the last 30 days carries no extra
  information — the summary features were already enough (and they were already useless).
- If **LSTM/GRU < Logistic(flat)**, the extra flexibility is just fitting noise, which is the same
  story as XGBoost losing to Logistic throughout this project.

## Caveats
- ~3 seeds averaged, but recurrent nets remain the noisiest models here.
- Pooled sequences from 7 correlated stocks are not independent samples.
- 30-day window chosen a priori; a longer window was not searched, because searching window
  lengths until one works is exactly how false findings are manufactured.
