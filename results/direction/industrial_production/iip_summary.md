# Phase G — industrial production (IIP): the last gap in the classic macro set

**Window:** IIP starts 2016-01, so the whole sample starts there (identical rows for every phase).
Sample ends **2026-05-31**. Test window: **2024-10-08 → 2026-05-27**.

| Phase | Adds | Features |
|---|---|---|
| A | Tier-1 technical (the plain time-series floor) | 9 |
| D | + Tier-2 + monthly rate Δ + sector | 38 |
| **I** | **+ industrial production (YoY based)** | 42 |
| FULL | + USD/LKR + CPI Δ = **full classic macro set** | 49 |

IIP lagged **50 days** (DCS publishes ~6-7 weeks after the reference month).
All macro fed as **changes**, never levels.

**Seasonality guard:** the DCS index is not seasonally adjusted — April falls 10-34 points every
single year (Sinhala/Tamil New Year) and March always peaks. Month-on-month change would therefore
be a calendar signal, so **only year-on-year based features are used**. Feeding mom here would let
the model "predict" the holiday and look clever for no reason.

## BOTTOM LINE (caveman)

- **Industrial production gain (D→I): -6.4 pp**, positive in 25 of 99 cells.
- **Full macro set gain (I→FULL): +0.2 pp.**
- **Cells beating the baseline:** A 16/99 · D 27/99 ·
  **I 25/99** · FULL 25/99.
- Significance: **0 of 36 phase×horizon combinations reach p < 0.05**
  (chance alone gives ~1.8).
- XGBoost gives IIP **15%** of its importance.
- **Verdict: industrial production adds nothing. It was the strongest variable in the reference paper, and it still does not help predict DIRECTION here.**

## Why the reference paper found IIP significant and we do not

Naik & Padhi (2012) found industrial production positive and significant for the BSE Sensex. That
is not a contradiction — it is a **different question**:

| | Reference paper | This project |
|---|---|---|
| Method | Johansen cointegration / VECM | supervised direct forecasting |
| Question | is there a long-run *relationship*? | can you *predict* the next move? |
| Data used | full sample, in-sample fit | train past → grade on unseen future |
| Target | index **level** | **direction** of the move |

A cointegrating relationship says two series drift together over years. It does **not** say you can
forecast tomorrow's, or even next year's, direction out of sample. This project keeps finding the
same thing: macro **explains**, it does not **predict**.

## Median edge by phase and horizon

| horizon | A | D | FULL | I |
|---|---|---|---|---|
| 1 day | -1.3 | -0.3 | -0.8 | -0.7 |
| 1 week | -3.3 | -0.5 | -3.0 | -2.6 |
| 2 weeks | -3.6 | -1.8 | -10.8 | -5.0 |
| 3 weeks | -8.1 | -6.3 | -11.9 | -10.5 |
| 1 month | -6.5 | -1.7 | -6.1 | -4.7 |
| 2 months | -6.2 | -10.8 | -16.4 | -11.5 |
| 3 months | -19.0 | -12.2 | -25.2 | -21.8 |
| 6 months | -41.0 | -24.6 | -38.4 | -46.9 |
| 1 year | -56.4 | -20.2 | -41.6 | -51.9 |

## Significance screen (sign test across the 11 stocks)

| horizon | positive | median_edge_pp | sign_test_p |
|---|---|---|---|
| 1 day | 5/11 | -0.7 | 0.726 |
| 1 week | 2/11 | -2.6 | 0.994 |
| 2 weeks | 3/11 | -5.0 | 0.967 |
| 3 weeks | 3/11 | -10.5 | 0.967 |
| 1 month | 3/11 | -4.7 | 0.967 |
| 2 months | 4/11 | -11.5 | 0.887 |
| 3 months | 1/11 | -21.8 | 1.0 |
| 6 months | 2/11 | -46.9 | 0.994 |
| 1 year | 2/11 | -51.9 | 0.994 |

## The macro chapter is now COMPLETE

| Reference-paper variable (Naik & Padhi 2012) | Tested here | Result |
|---|---|---|
| Industrial production | ✅ **this run** | -6.4 pp |
| Money supply (M1/M2/M2b/M4) | ✅ Phase F | −0.7 pp |
| Treasury bill / policy rate | ✅ Phase C | −2.8 pp (levels harmful) |
| Exchange rate | ✅ Phase E (daily) | +0.0 pp |
| Inflation (WPI / here CCPI) | ✅ Phase E | included in −1.9 pp |

All five classic macro variables are tested. **None of them beats plain price history.**

## Caveats
- Sample starts 2016 (IIP coverage), so numbers are not comparable to the 2012-start phases —
  compare *within* this table only.
- IIP is monthly and lagged ~7 weeks, so at a 1-day horizon it is nearly constant. Its fair test is
  the long horizons — where it also fails.
- Stocks share the market factor, so the p-values are already optimistic.
