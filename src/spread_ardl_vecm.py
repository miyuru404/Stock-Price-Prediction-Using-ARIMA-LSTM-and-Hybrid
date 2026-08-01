#!/usr/bin/env python3
"""
ARDL / VECM spread test. Splits the bank-vs-finance response to the lending-deposit
spread into LONG-RUN and SHORT-RUN, and tests causality (does the spread drive stock
performance?). Follow-up to the OLS spread regression.

Groups (corrected, spec B): BANKS = HNB,COMB,SAMP ; FINANCE = LOFC,LFIN,CFIN.
Market-adjusted: y = mean(log price of group) - log(ASPI)  (relative performance level).
Nothing tuned. Outputs to results/spread_regression/ardl_vecm/.
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "spread_regression" / "ardl_vecm"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
FINANCE = ["LOFC", "LFIN", "CFIN"]
KDIFF = 2  # lags of differences in VECM

rate = pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"]).sort_values("date").set_index("date")
spread = rate["spread"]

def mlog(sym):
    d = pd.read_csv(DATA / f"{sym}_daily_clean.csv", parse_dates=["date"]).sort_values("date")
    return np.log(d.set_index("date")["close"].resample("ME").last())

aspi = mlog("ASPI")
def group_rel(members):
    lp = pd.concat([mlog(m) for m in members], axis=1).mean(axis=1)
    return (lp - aspi).rename("y")

def adf(x, name):
    r = adfuller(x.dropna(), autolag="AIC")
    return {"series": name, "adf": r[0], "p": r[1], "stationary(5%)": r[1] < 0.05}

rows_adf = [adf(spread, "spread")]
results = {}
lines = []

for gname, members in [("Banks", BANKS), ("Finance", FINANCE)]:
    y = group_rel(members)
    data = pd.concat([y, spread.rename("spread")], axis=1).dropna()
    rows_adf.append(adf(data["y"], f"{gname}_relIndex"))

    # Johansen cointegration
    joh = coint_johansen(data, det_order=0, k_ar_diff=KDIFF)
    trace = joh.lr1[0]; crit95 = joh.cvt[0, 1]
    cointegrated = trace > crit95

    # VECM
    vecm = VECM(data, k_ar_diff=KDIFF, coint_rank=1, deterministic="ci")
    vres = vecm.fit()
    beta = vres.beta[:, 0]                      # cointegration vector [y, spread]
    lr_spread = -beta[1] / beta[0]             # long-run effect of spread on y (normalised on y)
    alpha_y = vres.alpha[0, 0]                 # speed of adjustment (ECM term) for y
    # Granger causality: does spread cause y?
    gc = vres.test_granger_causality(caused="y", causing="spread")

    results[gname] = {"group": gname, "n": len(data),
                      "coint(Johansen 95%)": cointegrated,
                      "trace_stat": trace, "trace_crit95": crit95,
                      "longrun_spread_effect": lr_spread,
                      "adjust_speed_alpha": alpha_y, "alpha_sig": abs(vres.tvalues_alpha[0, 0]) > 1.96,
                      "granger_spread->y_p": gc.pvalue}
    lines.append(f"[{gname}] n={len(data)} | cointegrated={cointegrated} (trace {trace:.1f} vs crit {crit95:.1f}) "
                 f"| long-run spread effect = {lr_spread:+.3f} | ECM speed alpha = {alpha_y:+.3f} "
                 f"({'sig' if abs(vres.tvalues_alpha[0,0])>1.96 else 'ns'}) | spread->y Granger p = {gc.pvalue:.3f}")

adf_df = pd.DataFrame(rows_adf); adf_df.round(3).to_csv(OUT / "adf_tests.csv", index=False)
res_df = pd.DataFrame(results.values()); res_df.round(4).to_csv(OUT / "ardl_vecm_results.csv", index=False)

print("=== ADF stationarity (level) ===")
print(adf_df.round(3).to_string(index=False))
print("\n=== VECM: long-run, short-run (ECM), causality ===")
for l in lines:
    print(l)

# ---- ARDL bounds test (robustness, per group) ----
try:
    from statsmodels.tsa.ardl import UECM
    for gname, members in [("Banks", BANKS), ("Finance", FINANCE)]:
        y = group_rel(members)
        data = pd.concat([y, spread.rename("spread")], axis=1).dropna()
        u = UECM(data["y"], 2, data[["spread"]], 2, trend="c").fit()
        bt = u.bounds_test(case=3)
        print(f"[ARDL bounds] {gname}: F={bt.stat:.2f}  (cointegration if F above upper I(1) bound)")
        results[gname]["ardl_bounds_F"] = float(bt.stat)
except Exception as e:
    print(f"ARDL bounds test skipped: {str(e)[:100]}")

# ---- simple summary md ----
b, f = results["Banks"], results["Finance"]
md = f"""# ARDL / VECM Spread Test — result (simple)

**Question:** over the long run and short run, how does bank / finance stock performance
react to the lending-deposit **spread**? Market-adjusted (minus ASPI). Groups: banks HNB/COMB/SAMP,
finance LOFC/LFIN/CFIN.

## Banks
- Long-run link with spread exists (cointegrated): **{b['coint(Johansen 95%)']}**
- **Long-run effect of spread on bank performance = {b['longrun_spread_effect']:+.3f}**
  (negative = wider spread -> banks do worse over time)
- Short-run pull back to equilibrium (ECM speed) = {b['adjust_speed_alpha']:+.3f} ({'significant' if b['alpha_sig'] else 'not sig'})
- Spread Granger-causes bank performance? p = {b['granger_spread->y_p']:.3f} ({'yes' if b['granger_spread->y_p']<0.05 else 'no'})

## Finance
- Cointegrated: **{f['coint(Johansen 95%)']}**
- **Long-run effect of spread = {f['longrun_spread_effect']:+.3f}**
- ECM speed = {f['adjust_speed_alpha']:+.3f} ({'significant' if f['alpha_sig'] else 'not sig'})
- Spread Granger-causes finance? p = {f['granger_spread->y_p']:.3f} ({'yes' if f['granger_spread->y_p']<0.05 else 'no'})

## Plain-English takeaway
Banks long-run spread effect {b['longrun_spread_effect']:+.2f} vs finance {f['longrun_spread_effect']:+.2f}.
If banks are clearly more negative, it confirms (with a proper long-run method) that a widening
spread / rising rates hurt Sri Lankan banks more than finance — matching the local banking literature.

*Caveat: 173 monthly points; VECM k_ar_diff={KDIFF}; market-adjusted relative index; correlation not causation beyond Granger sense.*
"""
(OUT / "ardl_vecm_summary.md").write_text(md)
print("\nSaved to", OUT)
