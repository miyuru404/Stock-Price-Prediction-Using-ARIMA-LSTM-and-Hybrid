#!/usr/bin/env python3
"""
Interest-rate SPREAD regression test (follow-up to the null policy-rate pilot).

Uses 173 monthly observations of the actual lending-deposit spread (bank net interest
margin) instead of 9 rate-hike events. Tests whether banks and finance companies respond
DIFFERENTLY to margin changes — the interaction term is the whole test.

Nothing tuned to produce a positive result; a null is reported plainly.
Outputs to results/spread_regression/.
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "spread_regression"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
FINANCE = ["LOFC", "LOLC", "LFIN"]
CONTROL = ["JKH", "DIAL"]
COMPANIES = BANKS + FINANCE + CONTROL
GROUP = {**{c: "Banks" for c in BANKS}, **{c: "Finance" for c in FINANCE},
         **{c: "Control" for c in CONTROL}}
LAGS = [0, 1, 2, 3]

# ------------------------------------------------- monthly rates + changes
rate = pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"]).sort_values("date")
rate = rate.set_index("date")
rate["d_spread"] = rate["spread"].diff()
rate["d_awpr"] = rate["awpr"].diff()
rate["d_policy"] = rate["policy_rate"].diff()

# ------------------------------------------------- monthly abnormal returns
def monthly_ret(sym):
    d = pd.read_csv(DATA / f"{sym}_daily_clean.csv", parse_dates=["date"]).sort_values("date")
    m = d.set_index("date")["close"].resample("ME").last()
    return m.pct_change()

aspi = monthly_ret("ASPI")
abn = {}
for s in COMPANIES:
    abn[s] = (monthly_ret(s) - aspi)  # abnormal monthly return

# ------------------------------------------------- tidy panel
recs = []
for s in COMPANIES:
    a = abn[s].rename("abn_ret").to_frame()
    a = a.join(rate[["d_spread", "d_awpr", "d_policy", "spread"]], how="inner")
    for k in LAGS:
        a[f"d_spread_l{k}"] = rate["d_spread"].shift(k).reindex(a.index)
        a[f"d_awpr_l{k}"] = rate["d_awpr"].shift(k).reindex(a.index)
        a[f"d_policy_l{k}"] = rate["d_policy"].shift(k).reindex(a.index)
    a["company"] = s; a["group"] = GROUP[s]
    a["is_bank"] = int(s in BANKS); a["year"] = a.index.year
    a["abn_ret"] = a["abn_ret"] * 100  # percent for readability
    recs.append(a.reset_index().rename(columns={"index": "date"}))
panel = pd.concat(recs, ignore_index=True)
print(f"Panel: {panel.company.nunique()} companies x months | "
      f"{panel.date.min().date()} -> {panel.date.max().date()} | rows {len(panel)}")

# ================================================= (4) per-company regressions
rows = []
for s in COMPANIES:
    cdf = panel[panel.company == s]
    for k in LAGS:
        x = f"d_spread_l{k}"
        sub = cdf[["abn_ret", x]].dropna()
        m = smf.ols(f"abn_ret ~ {x}", data=sub).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append({"company": s, "group": GROUP[s], "lag": k,
                     "coef": m.params[x], "se": m.bse[x], "t": m.tvalues[x],
                     "p": m.pvalues[x], "r2": m.rsquared, "n": int(m.nobs)})
per_co = pd.DataFrame(rows)
per_co.round(4).to_csv(OUT / "per_company_regression.csv", index=False)

# ================================================= (5) group interaction (banks vs finance)
bf = panel[panel.group.isin(["Banks", "Finance"])].copy()
def interaction(df, xcol, label):
    sub = df[["abn_ret", xcol, "is_bank", "date"]].dropna()
    m = smf.ols(f"abn_ret ~ {xcol} * is_bank", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["date"]})
    inter = f"{xcol}:is_bank"
    fin_slope = m.params[xcol]                       # finance (is_bank=0)
    bank_slope = m.params[xcol] + m.params[inter]    # banks
    return {"model": label, "n": int(m.nobs), "r2": m.rsquared,
            "finance_slope": fin_slope, "bank_slope": bank_slope,
            "interaction_coef": m.params[inter], "interaction_se": m.bse[inter],
            "interaction_t": m.tvalues[inter], "interaction_p": m.pvalues[inter]}, m, sub

gi = []
main_model = None; main_sub = None
for k in LAGS:
    r, m, sub = interaction(bf, f"d_spread_l{k}", f"d_spread lag {k}")
    gi.append(r)
    if k == 0:
        main_model, main_sub = m, sub
gi = pd.DataFrame(gi)
gi.round(4).to_csv(OUT / "group_interaction_results.csv", index=False)

# ================================================= (6/robustness) checks
rob = []
# ex-2022
r, _, _ = interaction(bf[bf.year != 2022], "d_spread_l0", "d_spread ex-2022")
rob.append(r)
# d_awpr
r, _, _ = interaction(bf, "d_awpr_l0", "d_awpr (lag0)")
rob.append(r)
# d_policy (should be weaker - failed in pilot)
r, _, _ = interaction(bf, "d_policy_l0", "d_policy (lag0)")
rob.append(r)
# OUTLIER robustness: the 5 biggest residuals are all 2021 finance mega-returns.
# (a) winsorize abnormal returns at 1/99 pct; (b) drop extreme months |ret|>50%.
bf_w = bf.copy()
_lo, _hi = bf_w["abn_ret"].quantile([0.01, 0.99])
bf_w["abn_ret"] = bf_w["abn_ret"].clip(_lo, _hi)
r, _, _ = interaction(bf_w, "d_spread_l0", "d_spread winsorized 1/99pct")
rob.append(r)
r, _, _ = interaction(bf[bf["abn_ret"].abs() <= 50], "d_spread_l0", "d_spread excl |ret|>50%")
rob.append(r)
rob = pd.DataFrame(rob)
rob.round(4).to_csv(OUT / "robustness_checks.csv", index=False)

# outliers: 5 largest residuals from main (lag0 d_spread) panel model
resid = main_sub.copy(); resid["resid"] = main_model.resid
resid = resid.merge(panel[["date", "company"]].drop_duplicates(), on="date", how="left")
top5 = (main_sub.assign(resid=main_model.resid.values)
        .assign(company=bf.loc[main_sub.index, "company"].values,
                grp=bf.loc[main_sub.index, "group"].values)
        .reindex(main_model.resid.abs().sort_values(ascending=False).index)
        .head(5)[["date", "company", "grp", "abn_ret", "d_spread_l0", "resid"]])

# ================================================= console
print("\n=== PER-COMPANY: coefficient on contemporaneous d_spread (lag 0) ===")
print(per_co[per_co.lag == 0][["company", "group", "coef", "se", "t", "p", "r2", "n"]].round(3).to_string(index=False))
print("\n=== GROUP INTERACTION (banks vs finance) — the whole test ===")
print(gi[["model", "finance_slope", "bank_slope", "interaction_coef", "interaction_se", "interaction_p", "n", "r2"]].round(4).to_string(index=False))
print("\n=== ROBUSTNESS ===")
print(rob[["model", "finance_slope", "bank_slope", "interaction_coef", "interaction_p", "n"]].round(4).to_string(index=False))
print("\n=== 5 largest residuals (main lag-0 model) ===")
print(top5.round(3).to_string(index=False))

# ================================================= plots
# scatter: d_spread vs abn_ret, banks vs finance
fig, ax = plt.subplots(figsize=(9, 6))
for grp, col in [("Banks", "tab:blue"), ("Finance", "tab:red")]:
    g = bf[bf.group == grp][["d_spread_l0", "abn_ret"]].dropna()
    ax.scatter(g["d_spread_l0"], g["abn_ret"], s=14, alpha=0.5, color=col, label=grp)
    b = np.polyfit(g["d_spread_l0"], g["abn_ret"], 1)
    xs = np.linspace(g["d_spread_l0"].min(), g["d_spread_l0"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=col, lw=2, label=f"{grp} fit (slope {b[0]:+.2f})")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Monthly change in spread (awpr - awdr), pp")
ax.set_ylabel("Abnormal monthly return (%)")
ax.set_title("Do banks and finance react differently to margin changes?\n(d_spread vs abnormal return, contemporaneous)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "scatter_dspread_vs_abnret.png", dpi=140)

# time series: spread with 2022 crisis
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(rate.index, rate["spread"], color="tab:purple", lw=1.5)
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="red", alpha=0.15, label="2022 crisis")
ax.set_ylabel("Spread (awpr - awdr), pp"); ax.set_title("Bank lending-deposit spread, 2012-2026 (2022 crisis shaded)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "spread_timeseries.png", dpi=140)

# ================================================= summary md
i0 = gi[gi.model == "d_spread lag 0"].iloc[0]
any_sig = (gi["interaction_p"] < 0.05).any()
ex22 = rob[rob.model == "d_spread ex-2022"].iloc[0]
excl_out = rob[rob.model == "d_spread excl |ret|>50%"].iloc[0]
banks_pos_fin_neg = (i0["bank_slope"] > 0) and (i0["finance_slope"] <= 0)
opposite_signs = (i0["bank_slope"] < 0) and (i0["finance_slope"] > 0)
survives_outliers = excl_out["interaction_p"] < 0.05

if any_sig and banks_pos_fin_neg and (ex22["interaction_p"] < 0.05) and survives_outliers:
    verdict = ("SUPPORTED — significant interaction with banks positive / finance negative to a "
               "wider spread, robust to excluding 2022 and to removing outliers.")
elif any_sig and opposite_signs:
    verdict = (f"HYPOTHESIS NOT SUPPORTED (effect runs the OTHER way) — there IS a strong, "
               f"significant interaction (lag-0 p={i0['interaction_p']:.4f}), but the signs are "
               f"OPPOSITE the hypothesis: banks react NEGATIVELY to a wider spread "
               f"({i0['bank_slope']:+.2f}) and finance POSITIVELY ({i0['finance_slope']:+.2f}). "
               f"And the finance-positive side is driven by extreme 2021 outliers (LOLC/LOFC "
               f"bubble): excluding months with |return|>50%, the interaction p="
               f"{excl_out['interaction_p']:.3f} ({'still significant' if survives_outliers else 'collapses'}). "
               f"The margin-channel story is not supported.")
elif any_sig:
    verdict = ("MIXED — a significant interaction appears but signs/robustness do not cleanly "
               "match the hypothesis. Treat as weak / likely confounded.")
else:
    verdict = ("NULL — no significant interaction at any lag. Consistent with the pilot's null.")

def md_table(df, cols, hdr):
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

md = f"""# Interest-Rate Spread Regression — Result

**Test:** does bank net interest margin (spread = AWPR - AWDR) move bank and finance stock
returns **differently**? 173 monthly observations, abnormal returns (vs ASPI). The
**interaction term (d_spread x is_bank)** is the whole test.

## Verdict

**{verdict}**

Contemporaneous (lag 0): finance slope **{i0['finance_slope']:+.3f}**, banks slope
**{i0['bank_slope']:+.3f}**, interaction **{i0['interaction_coef']:+.3f}**
(SE {i0['interaction_se']:.3f}, p = **{i0['interaction_p']:.3f}**), n={int(i0['n'])}.

## Group interaction — all lags (banks vs finance)

{md_table(gi, ['model','finance_slope','bank_slope','interaction_coef','interaction_p','n'], ['Model','Finance slope','Banks slope','Interaction','Interaction p','n'])}

Hypothesis wants: banks slope **positive**, finance slope **negative/zero**, interaction **p<0.05**.

## Per-company (contemporaneous d_spread, lag 0)

{md_table(per_co[per_co.lag==0], ['company','group','coef','p','r2','n'], ['Company','Group','Coef','p','R2','n'])}

Control (JKH, DIAL) should be near zero and non-significant — the placebo check.

## Robustness

{md_table(rob, ['model','finance_slope','bank_slope','interaction_coef','interaction_p','n'], ['Variant','Finance slope','Banks slope','Interaction','Interaction p','n'])}

- **ex-2022:** if the interaction loses significance here, the 2022 crisis was driving it.
- **d_awpr:** lending rate alone, for comparison.
- **d_policy:** the policy-rate proxy that already returned null in the pilot — expected weakest.

## 5 largest residuals (main lag-0 model)

{top5.round(3).to_string(index=False)}

## Caveats

- **173 monthly observations** — real but modest power (not thousands).
- **2022 is an enormous spread outlier** (3.41 -> 15.50) — reported with and without.
- **LFIN** prices pre-2015-07-14 back-adjusted x0.5 for an unapplied 2-for-1 split.
- **Policy framework changed 27 Nov 2024** (SDFR/SLFR -> single OPR).
- **Correlation, not causation** — this is an association between margin changes and returns,
  not proof that margin changes cause them.
- Per-company SEs are HAC (Newey-West, 3 lags); the panel interaction uses SEs **clustered by
  month** to handle contemporaneous cross-company correlation.

*Outputs: per_company_regression.csv, group_interaction_results.csv, robustness_checks.csv,
scatter_dspread_vs_abnret.png, spread_timeseries.png.*
"""
(OUT / "spread_regression_summary.md").write_text(md)
print(f"\nVERDICT: {verdict}")
print(f"Saved to {OUT}/")
