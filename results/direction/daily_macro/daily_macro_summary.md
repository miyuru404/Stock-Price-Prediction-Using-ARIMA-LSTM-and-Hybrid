# Phase E — daily macro (does frequency fix what Phase C got wrong?)

**Stocks:** all 11 (banks/finance/control) · **Horizons:** [1, 5, 10, 15, 22, 44, 66, 132, 252] · **Cells:** 99 per phase
**Window:** set by the global data (2015-01 → 2026-04); every phase uses identical rows.

| Phase | Adds | Features |
|---|---|---|
| A | Tier-1 technical | 9 |
| D | + Tier-2 + monthly rate Δ + sector | 38 |
| E | **+ daily USD/LKR** returns & vol | 42 |
| E2 | + daily oil / US 10Y / DXY + CPI Δ | 54 |

Money supply is excluded on purpose — the CBSL export ends 2024-08, inside the test window.

## BOTTOM LINE (caveman)

- **Daily FX gain (D→E): +0.0 pp**, positive in 57 of 99 cells.
- **Global + CPI gain (E→E2): -1.9 pp.**
- **Cells beating the baseline:** A 17/99 · D 27/99 ·
  E 24/99 · **E2 25/99**.
- Significance: **0 of 36 phase×horizon combinations reach p < 0.05.**
  Best for E2: 1 day, 6/11 stocks positive, p = 0.5.
- The model does *use* the new data — FX gets 7% of XGBoost's importance,
  global 11%, CPI 9% — and still gains nothing.
- **Verdict: frequency was NOT the problem. Macro fails daily exactly as it failed monthly. The information is not there.**

## Median edge by phase and horizon

| horizon | A | D | E | E2 |
|---|---|---|---|---|
| 1 day | -0.6 | 1.9 | 1.2 | 0.9 |
| 1 week | -2.7 | -3.2 | -2.8 | -1.7 |
| 2 weeks | -3.6 | -3.6 | -5.5 | -5.7 |
| 3 weeks | -8.9 | -5.5 | -6.6 | -6.2 |
| 1 month | -5.0 | -6.0 | -5.4 | -6.9 |
| 2 months | -5.7 | -6.1 | -3.8 | -5.7 |
| 3 months | -12.3 | -5.0 | -4.6 | -9.8 |
| 6 months | -30.9 | -20.3 | -17.2 | -21.2 |
| 1 year | -49.1 | -27.3 | -28.3 | -28.0 |

## Significance screen (sign test across the 11 stocks)

| horizon | positive | median_edge_pp | sign_test_p |
|---|---|---|---|
| 1 day | 6/11 | 0.9 | 0.5 |
| 1 week | 4/11 | -1.7 | 0.887 |
| 2 weeks | 2/11 | -5.7 | 0.994 |
| 3 weeks | 3/11 | -6.2 | 0.967 |
| 1 month | 1/11 | -6.9 | 1.0 |
| 2 months | 3/11 | -5.7 | 0.967 |
| 3 months | 3/11 | -9.8 | 0.967 |
| 6 months | 2/11 | -21.2 | 0.994 |
| 1 year | 1/11 | -28.0 | 1.0 |

## Caveats
- Window starts 2015 (global data), so these numbers are not directly comparable to the earlier
  2012-start ablation — compare phases *within* this table only.
- Stocks share the market factor, so the p-values are optimistic already.
- Monthly CPI is still flat within a month; only FX, oil, US 10Y and DXY truly vary daily.

## Next
The untested sources left are all **event-timed or textual**: foreign investor daily net flows,
dividend/XD dates, earnings dates + EPS, and news sentiment. Also worth more than any of them:
switch the target from direction to **volatility**, which is known to be predictable and needs no
new data.
