# Pooled panel, co-movement and readable rules

**Sector:** ['HNB', 'COMB', 'SAMP', 'LOFC', 'LOLC', 'LFIN', 'CFIN'] (banking + finance) · **Panel:** 22,677 stock-days,
2012-02 → 2026-07 · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252]

---

## PART 1 — Co-movement: "if the indicator goes up, does the stock go up?"

No model. Pure counting, pooled over all 7 stocks.
`lift` = P(stock up | indicator up) − P(stock up | indicator down), in percentage points.
**Lift near 0 = the indicator tells you nothing about direction.**

| indicator | 1 day | 1 week | 2 weeks | 3 weeks | 1 month | 2 months | 3 months | 6 months | 1 year |
|---|---|---|---|---|---|---|---|---|---|
| ASPI (market) 1d | 8.3 | 8.4 | 7.1 | 7.6 | 7.1 | 4.6 | 3.8 | 2.8 | 1.9 |
| ASPI (market) 5d | 6.9 | 8.6 | 8.2 | 10.2 | 9.2 | 3.7 | 3.9 | 2.7 | 2.7 |
| Peer banks 5d | 6.1 | 7.6 | 6.4 | 7.2 | 6.4 | 3.0 | 2.1 | 1.0 | -1.6 |
| Own momentum 10d | 3.4 | 6.2 | 8.4 | 9.6 | 6.4 | 0.2 | 0.1 | 3.4 | 1.9 |
| RSI(14) above 50 | 3.5 | 7.3 | 8.9 | 9.1 | 6.0 | -1.2 | -0.6 | 2.7 | 1.5 |
| MACD histogram | 2.0 | 5.2 | 7.0 | 9.5 | 9.3 | 2.7 | 0.6 | 2.6 | 2.2 |
| Volume vs 5d avg | 2.7 | 1.9 | 2.0 | 2.0 | 2.0 | 1.9 | 1.0 | 0.8 | 0.6 |
| Policy rate change | -0.7 | -4.4 | -5.1 | -6.6 | -8.8 | -11.1 | -10.0 | -8.1 | 1.5 |
| Lending-deposit spread Δ | 0.6 | 1.2 | 0.4 | 0.7 | 2.3 | 0.8 | -3.9 | -7.9 | -11.4 |
| T-bill 3m change (3m) | 0.1 | -2.3 | -3.5 | -5.3 | -6.5 | -8.5 | -11.8 | -15.3 | -19.3 |
| USD/LKR 5d | -1.0 | -3.9 | -8.3 | -10.0 | -10.3 | -10.8 | -9.5 | -7.6 | -5.0 |
| USD/LKR 20d | -1.6 | -4.2 | -8.1 | -9.9 | -11.8 | -14.0 | -15.3 | -12.5 | -12.7 |
| Inflation YoY Δ (3m) | -0.3 | -0.7 | -0.5 | 0.8 | 2.2 | 5.6 | 3.7 | -8.4 | -10.4 |
| Money supply M2 YoY Δ | -0.4 | -1.6 | -1.3 | -1.5 | -1.7 | 0.9 | 6.0 | 15.0 | 17.0 |
| Industrial production YoY | -0.0 | 2.0 | 0.6 | 2.0 | 0.9 | 4.3 | 3.8 | -5.1 | -12.0 |

- **Median |lift| across all 135 cells: 3.9 pp.**
- Cells with |lift| ≥ 5 pp AND p < 0.01: **63 of 135**.
- Largest single lift: **T-bill 3m change (3m)** at 1 year,
  49.2% vs 68.6%
  (**-19.3 pp**, p = 0.0).

⚠ These p-values are **too generous**: overlapping forward windows mean the rows are not
independent (a 252-day window shares 251 days with the next one). Treat large-lift long-horizon
cells with suspicion — the effective sample there is tiny.

---

## PART 2 — Pooled panel model (17,770 training rows vs ~2,700 per stock)

Split by **date**, not by row, so no stock's future leaks into another's past.

| horizon | n_train | n_test | acc_logistic_% | acc_xgboost_% | acc_majority_% | acc_persistence_% | edge_pp | beats_baseline |
|---|---|---|---|---|---|---|---|---|
| 1 day | 17770 | 4445 | 40.1 | 41.3 | 38.0 | 37.7 | 3.3 | True |
| 1 week | 17721 | 4438 | 38.8 | 36.3 | 34.3 | 36.1 | 2.7 | True |
| 2 weeks | 17651 | 4431 | 35.8 | 30.9 | 33.4 | 37.6 | -1.8 | False |
| 3 weeks | 17588 | 4424 | 37.1 | 31.9 | 33.0 | 41.7 | -4.6 | False |
| 1 month | 17504 | 4410 | 39.3 | 27.7 | 44.0 | 41.5 | -4.8 | False |
| 2 months | 17224 | 4382 | 40.0 | 25.6 | 47.4 | 36.1 | -7.5 | False |
| 3 months | 16924 | 4347 | 44.5 | 32.8 | 50.5 | 34.6 | -6.0 | False |
| 6 months | 15743 | 4158 | 41.8 | 37.3 | 64.7 | 56.2 | -22.8 | False |
| 1 year | 13844 | 3821 | 41.4 | 41.7 | 79.4 | 59.1 | -37.7 | False |

**Horizons where the pooled model beats the pooled baseline: 2 of 9.**

### ⚠ Fairness check — the pooled baseline is not a fair benchmark

Pooling stocks with different class balances can WEAKEN the majority baseline and manufacture an
edge out of nothing. So the pooled model was also scored **per stock, against that stock's own
baseline**, on the same test window, then sign-tested across the 7 stocks:

| horizon | horizon_days | stocks_positive | median_edge_pp | sign_test_p |
|---|---|---|---|---|
| 1 day | 1 | 6/7 | 2.5 | 0.062 |
| 1 week | 5 | 3/7 | -0.5 | 0.773 |
| 2 weeks | 10 | 1/7 | -6.0 | 0.992 |
| 3 weeks | 15 | 0/7 | -6.8 | 1.0 |
| 1 month | 22 | 0/7 | -4.1 | 1.0 |
| 2 months | 44 | 1/7 | -6.2 | 0.992 |
| 3 months | 66 | 2/7 | -6.4 | 0.938 |
| 6 months | 132 | 2/7 | -34.7 | 0.938 |
| 1 year | 252 | 1/7 | -38.9 | 0.992 |

| ticker | 1 day | 1 week | 2 weeks | 3 weeks | 1 month | 2 months | 3 months | 6 months | 1 year |
|---|---|---|---|---|---|---|---|---|---|
| CFIN | 4.4 | 5.8 | -3.5 | -6.8 | -10.4 | -5.9 | -8.2 | -31.9 | -23.0 |
| COMB | -3.0 | 0.6 | 2.9 | -4.1 | -3.3 | 7.1 | -2.3 | -34.7 | -40.3 |
| HNB | 0.2 | -0.5 | -3.3 | -6.6 | -4.0 | -7.5 | -14.0 | -38.9 | -38.9 |
| LFIN | 3.3 | -4.6 | -6.0 | -14.5 | -18.2 | -22.2 | -19.0 | -40.2 | -54.1 |
| LOFC | 5.4 | -1.4 | -11.7 | -2.8 | -4.1 | -6.2 | 8.9 | 23.2 | 20.5 |
| LOLC | 1.4 | -1.9 | -8.8 | -9.7 | -7.1 | -1.6 | 10.5 | 9.9 | -21.4 |
| SAMP | 2.5 | 1.4 | -8.7 | -7.6 | -4.1 | -9.6 | -6.4 | -37.2 | -45.4 |

**What this changes:**
- The pooled table's **+2.7 pp at 1 week was an artifact** — against each stock's own baseline it
  is **-0.5 pp** (3/7 stocks).
- **1 day survives the fair test**: 6/7 stocks positive,
  median **+2.5 pp**,
  p = **0.062**.
  That is the **strongest result in the entire project** — but it is still not significant at 0.05,
  it is 1 horizon out of 9 tested, and it comes from a single test window. **Not a finding yet.**

7× more training data therefore did NOT produce a significant edge. The per-stock models were not starved of data.

---

## PART 3 — What rules did the model actually learn?

Depth-3 decision tree on the pooled training data, every branch scored on **unseen** test data.

| horizon | rule | says | fired_n | correct_% | vs_baseline_pp |
|---|---|---|---|---|---|
| 1 day | vol_20 <= 0.0123 AND aspi_ret_1 <= 0.0018 AND ret_1 > -0.0102 | Hold | 692 | 49.3 | 11.3 |
| 1 day | vol_20 <= 0.0123 AND aspi_ret_1 > 0.0018 AND ma5_ratio <= 0.0122 | Hold | 457 | 46.4 | 8.4 |
| 1 day | vol_20 <= 0.0123 AND aspi_ret_1 > 0.0018 AND ma5_ratio > 0.0122 | Buy | 153 | 43.1 | 5.2 |
| 1 day | vol_20 > 0.0123 AND ma5_ratio <= -0.0145 AND vol_10 <= 0.0391 | Buy | 504 | 41.1 | 3.1 |
| 1 day | vol_20 > 0.0123 AND ma5_ratio > -0.0145 AND peer_ret_1 > 0.0050 | Buy | 943 | 40.5 | 2.5 |
| 1 day | vol_20 > 0.0123 AND ma5_ratio > -0.0145 AND peer_ret_1 <= 0.0050 | Sell | 1526 | 36.5 | -1.5 |
| 1 day | vol_20 <= 0.0123 AND aspi_ret_1 <= 0.0018 AND ret_1 <= -0.0102 | Buy | 150 | 29.3 | -8.6 |
| 1 week | vol_20 > 0.0127 AND d_ccpi_yoy_3m <= 5.2350 AND ccpi_yoy_pct <= 9.8700 | Buy | 2887 | 40.4 | 4.3 |
| 1 week | vol_20 <= 0.0127 AND aspi_ret_5 > 0.0052 AND d_ccpi_yoy_3m > 2.6300 | Buy | 123 | 35.8 | -0.3 |
| 1 week | vol_20 <= 0.0127 AND aspi_ret_5 <= 0.0052 AND peer_ret_1 > -0.0080 | Hold | 643 | 33.7 | -2.4 |
| 1 week | vol_20 <= 0.0127 AND aspi_ret_5 > 0.0052 AND d_ccpi_yoy_3m <= 2.6300 | Hold | 591 | 32.7 | -3.5 |
| 1 week | vol_20 <= 0.0127 AND aspi_ret_5 <= 0.0052 AND peer_ret_1 <= -0.0080 | Hold | 194 | 29.9 | -6.2 |
| 1 month | vol_10 <= 0.0158 AND d_ccpi_yoy_3m <= -3.2750 AND ccpi_yoy_pct <= 18.6000 | Sell | 317 | 68.5 | 24.4 |
| 1 month | vol_10 > 0.0158 AND d_ccpi_yoy_3m <= 5.2350 AND term_slope <= 0.6100 | Buy | 1421 | 41.4 | -2.6 |
| 1 month | vol_10 <= 0.0158 AND d_ccpi_yoy_3m > -3.2750 AND d_tb3m_3m > -0.8250 | Hold | 1537 | 25.0 | -19.0 |
| 1 month | vol_10 <= 0.0158 AND d_ccpi_yoy_3m > -3.2750 AND d_tb3m_3m <= -0.8250 | Hold | 639 | 20.8 | -23.2 |
| 1 month | vol_10 > 0.0158 AND d_ccpi_yoy_3m <= 5.2350 AND term_slope > 0.6100 | Hold | 496 | 19.4 | -24.7 |

`vs_baseline_pp` is the honest column: how much better than the naive guess that rule is.
Best rule: **vol_10 <= 0.0158 AND d_ccpi_yoy_3m <= -3.2750 AND ccpi_yoy_pct <= 18.6000** → says Sell, correct 68.5% (+24.4 pp vs baseline).

---

## Caveats
- Overlapping forward windows inflate significance at long horizons; the 1-day and 1-week rows are
  the only ones with a decent effective sample.
- Pooling assumes the 7 stocks share one pattern. If each stock behaved differently, pooling would
  blur them — but the per-stock runs already failed too, so that is not the explanation here.
- Correlation, not causation.
