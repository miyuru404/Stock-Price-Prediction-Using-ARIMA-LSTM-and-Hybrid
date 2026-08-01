# Sector Sweep — does HNB's Phase D result replicate?

**Stocks:** 11 — banks ['HNB', 'COMB', 'SAMP'], finance ['LOFC', 'LOLC', 'LFIN', 'CFIN'], control ['JKH', 'DIAL', 'CTC', 'DIST']
(MELS excluded: history starts 2016-12 and would truncate every other series.)
**Phases:** A (Tier-1 floor) vs D (Tier-1 + Tier-2 + macro Δ + sector) · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252]
**Protocol:** identical to the ablation runner — direct per-horizon models, 80/20 chronological
split with an h-bar purge gap, train-only scaling, macro publication-lagged 35 days.
**Peer composites exclude the target stock itself**, so no stock predicts itself.

## BOTTOM LINE (caveman)

- **Phase D beats the baseline in 27 of 99 stock×horizon cells** (Phase A: 14).
- **Sector gain is real and general: +3.8 pp on average, positive in
  66 of 99 cells.** HNB was not a fluke — adding sector context helps almost
  everywhere.
- **But it still does not produce an edge.** Helping ≠ winning.
- The three HNB claims, re-tested:
  1. *"1 week is near parity"* → 7 of 11 stocks land within 1 pp of the baseline at 1 week.
  2. *"sign edge positive at 1-3 weeks"* → positive in 6 of 33 cells
     (18%, coin-flip would be ~50%).
  3. *"sector gain is positive"* → 66 of 99 cells (67%). **Replicates.**
- Claim 2 **fails**: 6/33 is *below* the ~50% a coin flip would give. HNB's
  positive 1-3 week sign edge was luck. Good that we checked it by name.

## Significance screen — is the short-horizon clustering real?

Per horizon, the 11 stocks are 11 tries. Sign test on "edge > 0" against H0: p = 0.5.

| horizon_days | horizon | stocks_with_positive_edge | median_edge_pp | sign_test_p | wilcoxon_p | verdict |
|---|---|---|---|---|---|---|
| 1 | 1 day | 6/11 | 0.9 | 0.5 | 0.35 | noise |
| 5 | 1 week | 6/11 | 0.2 | 0.5 | 0.626 | noise |
| 10 | 2 weeks | 3/11 | -3.0 | 0.967 | 0.938 | noise |
| 15 | 3 weeks | 3/11 | -6.2 | 0.967 | 0.938 | noise |
| 22 | 1 month | 2/11 | -7.5 | 0.994 | 0.972 | noise |
| 44 | 2 months | 2/11 | -9.4 | 0.994 | 0.995 | noise |
| 66 | 3 months | 3/11 | -9.1 | 0.967 | 0.938 | noise |
| 132 | 6 months | 2/11 | -20.3 | 0.994 | 0.988 | noise |
| 252 | 1 year | 0/11 | -28.4 | 1.0 | 1.0 | noise |

Strongest cell: **1 day**, 6/11 stocks positive,
median edge +0.9 pp, sign-test p = 0.5.

**The stocks are not independent** — they share the CSE market factor, so these p-values are
optimistic. Nothing here survives a multiple-testing correction across 9 horizons. Read this as
*"the only place worth looking again is 1 day to 1 week"*, not as a discovery.

## Edge by horizon (Phase D, averaged over all stocks)

| horizon_days | horizon | mean_edge_pp | median_edge_pp | n_beat | n | mean_sign_edge_pp |
|---|---|---|---|---|---|---|
| 1 | 1 day | -1.2 | 0.9 | 6 | 11 | -9.5 |
| 5 | 1 week | -0.7 | 0.2 | 6 | 11 | -3.9 |
| 10 | 2 weeks | -2.6 | -3.0 | 3 | 11 | -5.0 |
| 15 | 3 weeks | -4.1 | -6.2 | 3 | 11 | -7.0 |
| 22 | 1 month | -6.5 | -7.5 | 2 | 11 | -6.5 |
| 44 | 2 months | -12.0 | -9.4 | 2 | 11 | -7.9 |
| 66 | 3 months | -8.9 | -9.1 | 3 | 11 | -6.8 |
| 132 | 6 months | -19.6 | -20.3 | 2 | 11 | -13.3 |
| 252 | 1 year | -25.6 | -28.4 | 0 | 11 | -28.7 |

## Edge and gain by group

| group | mean_edge_pp | mean_gain_pp | n_beat | n |
|---|---|---|---|---|
| bank | -10.3 | 5.3 | 4 | 27 |
| control | -13.9 | -1.4 | 11 | 36 |
| finance | -3.2 | 7.7 | 12 | 36 |

## Sector gain by stock

| group | ticker | mean_gain_pp | best_edge_pp | n_beat |
|---|---|---|---|---|
| finance | LFIN | 13.2 | 5.6 | 1 |
| bank | SAMP | 9.4 | 2.4 | 2 |
| finance | LOFC | 8.2 | 24.7 | 5 |
| finance | CFIN | 7.5 | 6.2 | 4 |
| bank | HNB | 4.8 | 0.3 | 1 |
| control | DIST | 4.1 | 9.7 | 5 |
| control | JKH | 3.3 | 8.4 | 5 |
| finance | LOLC | 1.9 | 4.6 | 2 |
| bank | COMB | 1.8 | 4.6 | 1 |
| control | DIAL | -1.8 | 0.9 | 1 |
| control | CTC | -11.0 | -1.2 | 0 |

## How to read the win count

11 stocks × 9 horizons = **99 tries**. With no skill at all, a
few cells beat the baseline by luck. So a handful of wins is the *expected* result of testing this
much, not evidence. What would count as evidence is wins **clustered at one horizon across many
stocks** — a pattern luck does not produce.

## Caveats
- One chronological split per stock. Long horizons overlap heavily (few independent windows).
- The control group is not a clean control: JKH, DIAL, CTC and DIST are still CSE stocks driven by
  the same market factor (ASPI), so "sector" information partly overlaps for them too.
- Correlation, not causation.

## Next
Phase E (events: dividend dates, rate-decision flags) and Phase F (news sentiment). These are the
last untested information sources outside the price chart. Rate events are already in
`cleaned_data/policy_rate_events.csv`; dividends and news still need collecting.
