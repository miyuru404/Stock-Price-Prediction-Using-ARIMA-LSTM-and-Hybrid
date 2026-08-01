# Three diagnostics — is the null real, or is the pipeline blind?

Every phase in this project returned null. Before that becomes a finding it must be separated from
the alternative: that the evaluation cannot detect signal even when one exists.

## TEST 1 — POSITIVE CONTROL (the decisive one)

Same pipeline, same stocks, same folds. Only the timing changes.

| Setup | Features | Median accuracy | Baseline | Edge |
|---|---|---|---|---|
| **Contemporaneous** | TODAY's market, macro, news | **68.6%** | 57.4% | **+8.2 pp** |
| **Lagged** | YESTERDAY's information only | 54.9% | 57.4% | -4.1 pp |

**Gap: +13.7 percentage points.**

The stock's own same-day return is EXCLUDED from the contemporaneous features — otherwise the
answer would be an input. Only market, peers, macro and sentiment are used.

**Verdict: the pipeline WORKS. It finds a large, obvious signal the moment the information is contemporaneous, and loses it entirely one day earlier. That gap IS the finding: this information EXPLAINS returns but does not PREDICT them.**

## TEST 2 — Binary up/down (removes the dead-zone / Hold-majority problem)

| test | median_acc_% | median_baseline_% | median_edge_pp | folds_positive | sign_test_p | verdict |
|---|---|---|---|---|---|---|
| 2-BINARY h=1 | 54.9 | 57.9 | -4.8 | 1/19 | 1.0 | no |
| 2-BINARY h=22 | 55.3 | 56.5 | -2.1 | 4/19 | 0.9978 | no |
| 2-BINARY h=5 | 55.3 | 57.7 | -1.1 | 7/19 | 0.9165 | no |

## TEST 3 — Magnitude-weighted value: is the model right on the BIG moves?

| test | median_edge_pp | median_mag_weighted_edge_pp |
|---|---|---|
| 2-BINARY h=1 | -4.3 | 9.3 |
| 2-BINARY h=22 | -2.5 | 9.0 |
| 2-BINARY h=5 | -2.4 | 7.3 |

`median_mag_weighted_edge_pp` weights each day by |return|, so being right on an 8% day counts 80x
a 0.1% day.

### Long/short backtest — h = 1 only

Only the 1-day horizon is summable: at 5 and 22 days consecutive forward returns OVERLAP, so adding
them counts every move ~h times over. Those backtests are omitted rather than reported wrong.

| metric | strategy_gross_% | buy_and_hold_% | net_10bps_% | net_25bps_% | net_50bps_% | median_position_flips | beats_buy_hold_gross | beats_buy_hold_net_25bps |
|---|---|---|---|---|---|---|---|---|
| median across stock-folds | 22.19 | 6.49 | 17.35 | 11.24 | 0.05 | 40 | 88/133 | 71/133 |

Each position flip is a round trip. CSE round-trip costs (brokerage + levies + spread) are
realistically **50-100 bps**, so the `net_50bps` column is the closest to reality — and it is the
column that matters.

## Full verdict table

| test | folds | median_acc_% | median_baseline_% | median_edge_pp | folds_positive | sign_test_p | verdict |
|---|---|---|---|---|---|---|---|
| 1-CONTEMPORANEOUS (today from today) | 19 | 68.6 | 57.4 | 8.2 | 17/19 | 0.0004 | BEATS BASELINE |
| 1-LAGGED (today from yesterday) | 19 | 54.9 | 57.4 | -4.1 | 3/19 | 0.9996 | no |
| 2-BINARY h=1 | 19 | 54.9 | 57.9 | -4.8 | 1/19 | 1.0 | no |
| 2-BINARY h=22 | 19 | 55.3 | 56.5 | -2.1 | 4/19 | 0.9978 | no |
| 2-BINARY h=5 | 19 | 55.3 | 57.7 | -1.1 | 7/19 | 0.9165 | no |

## Caveats
- The long/short backtest ignores transaction costs, bid-ask spread and liquidity. On the CSE those
  are large. Treat any positive strategy return as an upper bound.
- Contemporaneous accuracy is NOT a forecast and must never be reported as one — it uses same-day
  information that a trader would not have before acting.
