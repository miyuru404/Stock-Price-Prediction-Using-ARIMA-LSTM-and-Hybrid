# ARDL / VECM Spread Test — result (honest, simple)

## Key stationarity fact (shapes everything)
- **spread = stationary / I(0)** (ADF p=0.001) — it mean-reverts, does not wander.
- **bank & finance relative price index = I(1)** (trending, non-stationary).
- Mixed orders -> **ARDL is the correct tool** (handles I(0)+I(1)); Johansen/VECM cointegration is not really valid here, so its output is treated as indicative only.

## ARDL bounds test (the right test for long-run link)
- Banks:   F = 0.66  -> **NO long-run link**
- Finance: F = 3.42  -> below the 5% I(1) bound (~4.16) -> **no clear long-run link either**

=> **There is NO stable long-run (cointegrating) relationship between the spread and stock performance for either group.**

## Long-run direction (indicative) + short-run + causality
| Group | long-run spread effect | Granger spread->stocks (p) |
|---|---|---|
| Banks   | -0.31 (negative) | 0.082 (weak, ~10%) |
| Finance | +0.11 (positive) | 0.474 (no) |

- Direction still matches earlier: **banks negative, finance positive** — but long-run is weak/not significant.
- Granger: spread -> banks is only marginal; spread -> finance is nothing.

## Plain-English takeaway
The spread does NOT have a strong long-run bond with stock prices (spread mean-reverts; prices trend).
So the bank-vs-finance effect we found before is a **SHORT-RUN reaction**, not a long-run equilibrium.
This is consistent with the OLS result (banks react negatively to spread *changes*), and with Sri Lankan
bank literature (rising rates hurt banks) — but the honest ARDL verdict is: **short-run effect real, long-run link weak/absent.**

*Caveats: 173 monthly points; spread is I(0) so VECM cointegration is indicative only; correlation not causation beyond Granger.*
