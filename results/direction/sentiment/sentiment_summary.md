# Phase H — news sentiment

**Source:** 99,957 Dailymirror/Newsfirst articles from 2016-01; **17,953**
market-relevant after keyword filtering (18%).
**Feed ends 2022-06-20**, so the whole study window is 2016-01 → 2022-06.

**Lexicons:** VADER (general) and compact LM-style (157 pos / 213 neg).
Correlation between the two scores: **0.416**.

**Look-ahead guard:** CSE closes 14:30. Articles published after the close, or on non-trading days,
are pushed to the **next** trading day (49% of articles were pushed).
Without this the model would be reading the answer.

**Evaluation:** walk-forward, 7 × 6-month folds, scored per stock against
that stock's own baseline. A single 80/20 split was deliberately NOT used: it would put the entire
test set inside the 2022 crisis, a period this project has three times shown makes every model —
price-only included — look good.

| Phase | Features |
|---|---|
| A | Tier-1 technical (floor) |
| D | + Tier-2 + rate changes + sector |
| **H** | **+ news sentiment** |
| S | Tier-1 + sentiment only (sentiment's standalone value) |

## VERDICT

| horizon_days | phase | folds_positive | median_edge_pp | sign_test_p | beats_baseline |
|---|---|---|---|---|---|
| 1 | A | 4/7 | 0.9 | 0.5 | NO |
| 1 | D | 5/7 | 1.3 | 0.2266 | NO |
| 1 | H | 4/7 | 0.8 | 0.5 | NO |
| 1 | S | 2/7 | 0.0 | 0.9375 | NO |
| 1 | GAIN D->H | 2/7 | -0.8 | 0.9375 | — |
| 5 | A | 3/7 | 0.0 | 0.7734 | NO |
| 5 | D | 1/7 | -3.4 | 0.9922 | NO |
| 5 | H | 2/7 | -3.3 | 0.9375 | NO |
| 5 | S | 3/7 | 0.0 | 0.7734 | NO |
| 5 | GAIN D->H | 2/7 | -0.7 | 0.9375 | — |
| 10 | A | 3/7 | -3.2 | 0.7734 | NO |
| 10 | D | 4/7 | 4.3 | 0.5 | NO |
| 10 | H | 4/7 | 4.5 | 0.5 | NO |
| 10 | S | 3/7 | 0.0 | 0.7734 | NO |
| 10 | GAIN D->H | 4/7 | 2.4 | 0.5 | — |
| 22 | A | 1/7 | -5.6 | 0.9922 | NO |
| 22 | D | 4/7 | 2.4 | 0.5 | NO |
| 22 | H | 4/7 | 2.4 | 0.5 | NO |
| 22 | S | 3/7 | -7.3 | 0.7734 | NO |
| 22 | GAIN D->H | 4/7 | 1.6 | 0.5 | — |

**Sentiment gain (D→H):**

| horizon_days | folds_positive | median_edge_pp | sign_test_p |
|---|---|---|---|
| 1 | 2/7 | -0.8 | 0.9375 |
| 5 | 2/7 | -0.7 | 0.9375 |
| 10 | 4/7 | 2.4 | 0.5 |
| 22 | 4/7 | 1.6 | 0.5 |

**No phase beats the naive baseline at any horizon. News sentiment does not improve direction forecasting.**

XGBoost gives the sentiment block **29.8%** of its
importance — the same "uses it, gains nothing" fingerprint seen with Tier-2 indicators (41%),
monthly macro (64%) and daily macro (27%).

## Regime split — does sentiment only "work" during the crisis?

| period | folds | median_gain_D_to_H | positive |
|---|---|---|---|
| 2019-2020 (calm) | 16 | 0.85 | 9/16 |
| 2021-2022 (crisis) | 12 | -0.45 | 3/12 |

## Fold by fold

| horizon_days | test_from | med_A | med_D | med_H | med_S | gain_D_to_H |
|---|---|---|---|---|---|---|
| 1 | 2019-01 | -1.7 | -8.5 | -9.6 | -4.3 | -1.1 |
| 1 | 2019-07 | 0.9 | 0.8 | 0.0 | -0.9 | -0.8 |
| 1 | 2020-01 | -2.5 | 1.3 | 4.2 | 0.0 | 2.9 |
| 1 | 2020-07 | -7.2 | -4.8 | -4.8 | -1.6 | 0.0 |
| 1 | 2021-01 | 0.9 | 3.5 | 2.6 | 2.7 | -0.9 |
| 1 | 2021-07 | 3.2 | 1.6 | 0.8 | 0.0 | -0.8 |
| 1 | 2022-01 | 3.5 | 2.3 | 4.6 | 6.1 | 2.3 |
| 5 | 2019-01 | 1.1 | -2.6 | 1.1 | 2.6 | 3.7 |
| 5 | 2019-07 | 0.0 | -3.4 | -3.3 | 0.0 | 0.1 |
| 5 | 2020-01 | -12.9 | -4.9 | -7.3 | -9.8 | -2.4 |
| 5 | 2020-07 | 3.2 | 5.6 | 2.4 | 1.6 | -3.2 |
| 5 | 2021-01 | -14.9 | -3.5 | -8.8 | -12.3 | -5.3 |
| 5 | 2021-07 | 0.8 | 0.0 | 0.0 | 1.6 | 0.0 |
| 5 | 2022-01 | -2.4 | -3.6 | -4.3 | -4.8 | -0.7 |
| 10 | 2019-01 | -3.2 | 4.3 | 8.2 | 8.5 | 3.9 |
| 10 | 2019-07 | 0.8 | -10.1 | -4.4 | 0.0 | 5.7 |
| 10 | 2020-01 | -4.2 | -2.4 | 0.0 | -3.7 | 2.4 |
| 10 | 2020-07 | 6.4 | 15.2 | 11.2 | 3.3 | -4.0 |
| 10 | 2021-01 | -3.5 | 6.1 | 4.5 | -6.4 | -1.6 |
| 10 | 2021-07 | 5.6 | 11.3 | 5.6 | 2.4 | -5.7 |
| 10 | 2022-01 | -13.0 | -25.6 | -21.8 | -32.1 | 3.8 |
| 22 | 2019-01 | -3.2 | -8.6 | -6.5 | -11.7 | 2.1 |
| 22 | 2019-07 | 5.9 | 7.6 | 4.9 | 0.8 | -2.7 |
| 22 | 2020-01 | -13.4 | -15.0 | -12.5 | -7.3 | 2.5 |
| 22 | 2020-07 | -5.6 | 15.2 | 16.8 | -8.0 | 1.6 |
| 22 | 2021-01 | 0.0 | 5.3 | 8.8 | 11.4 | 3.5 |
| 22 | 2021-07 | -8.1 | 2.4 | 2.4 | 4.8 | 0.0 |
| 22 | 2022-01 | -28.8 | -15.4 | -15.6 | -51.5 | -0.2 |

## Caveats
- **Feed ends 2022-06**, so Phase H cannot be tested on the same window as every other phase.
  Compare phases *within* this table only.
- Median ~2-3 relevant articles per day: a daily sentiment score built on so few items is noisy.
- 51% of source articles have empty body text, so those contribute headline sentiment only.
- General-news outlets, not a financial wire — relevance filtering is keyword-based and imperfect.
- The compact finance lexicon can be replaced by the full Loughran-McDonald master dictionary by
  dropping it into `cleaned_data/loughran_mcdonald_master.csv` (see `src/finance_lexicon.py`).
