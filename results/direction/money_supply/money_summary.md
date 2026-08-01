# Phase F — money supply and the full classic macro set

**Window MOVED:** sample cut at **2024-08-31**, where the CBSL money supply export
genuinely ends. Test window is therefore roughly **2022-03-09 → 2024-08-29**, which
**contains the 2022 Sri Lankan crisis** — a harder test than the calm 2024-2026 window.

| Phase | Adds | Features |
|---|---|---|
| A | Tier-1 technical (the plain time-series floor) | 9 |
| D | + Tier-2 + monthly rate Δ + sector | 38 |
| **M** | **+ money supply Δ (M1, M2, M2b, M4)** | 47 |
| F | + USD/LKR + CPI Δ = **full classic macro set** | 54 |

Money supply lagged **65 days** — the CBSL file itself says "Update frequency: Two
months after the end of Month". Reserve Money dropped (series ends 2024-02, six months short).
All macro fed as **changes**, never levels (Phase C lesson).

## BOTTOM LINE (caveman)

- **Money supply gain (D→M): -0.7 pp**, positive in 50 of 99 cells.
- **Full macro set gain (M→F): +1.1 pp.**
- **Cells beating the baseline:** A 43/99 · D 44/99 ·
  **M 51/99** · F 42/99.
- Significance: **2 of 36 phase×horizon combinations reach p < 0.05.**
- XGBoost gives money supply **29%** of its importance.
- **Verdict: money supply adds nothing, even on a crisis-era test window. The full classic macro set (money + rates + FX + inflation) does not beat plain price history at any horizon.**

## ⚠ The 1-day cell — read this before getting excited

2 cells reach p < 0.05 (chance alone would give ~1.8 out of 36). Both are at 1 day:

| Phase | What it contains | Stocks positive | Median edge | p |
|---|---|---|---|---|
| **A** | **price only, NO macro** | 9/11 | **+3.9 pp** | **0.033** |
| M | + money supply + everything else | 9/11 | +3.2 pp | 0.033 |

**The effect is strongest in the model with NO macro in it at all.** Adding money supply makes it
*smaller* (+3.9 → +3.2 pp). So this is not a macro result —
it is a property of the **crisis test window**, where 1-day moves became more persistent and the
naive baselines got easier to beat. Note the overall win counts jumped from ~17-27/99 on the calm
2024-2026 window to 43-51/99 here, for price-only features too.

Honest reading: **the window, not the variable.** And 2 hits out of 36 tests is what luck
produces anyway.

## Median edge by phase and horizon

| horizon | A | D | F | M |
|---|---|---|---|---|
| 1 day | 3.9 | 3.8 | 1.6 | 3.2 |
| 1 week | 0.2 | -4.6 | -1.0 | -2.7 |
| 2 weeks | -0.2 | -2.1 | -3.2 | -3.6 |
| 3 weeks | -2.1 | -3.8 | -5.4 | -3.6 |
| 1 month | -3.3 | -4.4 | -0.2 | 0.2 |
| 2 months | -4.7 | 2.7 | 0.2 | 1.6 |
| 3 months | -13.6 | 1.0 | -6.7 | 3.3 |
| 6 months | -19.6 | -2.9 | -4.0 | -6.6 |
| 1 year | -3.7 | 7.3 | 4.8 | 6.3 |

## Significance screen (sign test across the 11 stocks)

| horizon | positive | median_edge_pp | sign_test_p |
|---|---|---|---|
| 1 day | 9/11 | 3.2 | 0.033 |
| 1 week | 5/11 | -2.7 | 0.726 |
| 2 weeks | 5/11 | -3.6 | 0.726 |
| 3 weeks | 3/11 | -3.6 | 0.967 |
| 1 month | 6/11 | 0.2 | 0.5 |
| 2 months | 6/11 | 1.6 | 0.5 |
| 3 months | 6/11 | 3.3 | 0.5 |
| 6 months | 4/11 | -6.6 | 0.887 |
| 1 year | 7/11 | 6.3 | 0.274 |

## Why this test matters for the write-up

This closes the macro chapter properly. The full classic macro set is now tested:

| Reference-paper variable (Naik & Padhi 2012) | Tested here? |
|---|---|
| Money supply (M3 / here M1, M2, M2b, M4) | ✅ this run |
| Treasury bill / policy rate | ✅ Phase C |
| Exchange rate | ✅ Phase E (daily) + here |
| Inflation (WPI / here CCPI) | ✅ Phase E + here |
| Industrial production | ❌ **not collected — the one remaining gap** |

## Caveats
- Different window from the earlier phases, so compare phases *within* this table only.
- Stocks share the market factor, so p-values are already optimistic.
- Money supply is monthly AND lagged two months, so at a 1-day horizon it is nearly a constant.
  Its fair test is the long horizons — where it also fails.
