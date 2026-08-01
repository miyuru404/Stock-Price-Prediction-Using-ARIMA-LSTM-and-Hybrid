#!/usr/bin/env python3
"""
Spread-regression GROUPING check. Same method as spread_regression_test.py; only the
group membership changes. Tests whether the significant bank-vs-finance interaction
survives when LOLC Holdings (a conglomerate, not a finance company) is removed.

Four specifications (A original, B drop conglomerate, C funding structure, D pure NBFI).
Nothing tuned. If B kills the result, that is reported plainly.
Outputs to results/spread_regression/grouping_check/.
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
OUT = ROOT / "results" / "spread_regression" / "grouping_check"
OUT.mkdir(parents=True, exist_ok=True)
LAGS = [0, 1, 2, 3]

# four specifications: (name, group1(=1), group2(=0), control, g1label, g2label)
SPECS = [
    ("A_original", ["HNB", "COMB", "SAMP"], ["LOFC", "LOLC", "LFIN"], ["JKH", "DIAL"],
     "Banks", "Finance"),
    ("B_drop_conglomerate", ["HNB", "COMB", "SAMP"], ["LOFC", "LFIN", "CFIN"],
     ["JKH", "DIAL", "LOLC"], "Banks", "Finance"),
    ("C_funding_structure", ["HNB", "COMB", "SAMP", "LFIN"], ["LOFC", "CFIN"],
     ["JKH", "DIAL", "LOLC"], "Deposit-funded", "Borrowing-funded"),
    ("D_pure_nbfi", ["HNB", "COMB", "SAMP"], ["LOFC", "CFIN"], ["JKH", "DIAL", "LOLC"],
     "Banks", "Finance(NBFI)"),
]
ALLCO = sorted(set(sum([s[1] + s[2] + s[3] for s in SPECS], [])))

# ---------------- monthly rates + abnormal returns
rate = pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"]).sort_values("date").set_index("date")
rate["d_spread"] = rate["spread"].diff()

def monthly_ret(sym):
    d = pd.read_csv(DATA / f"{sym}_daily_clean.csv", parse_dates=["date"]).sort_values("date")
    return d.set_index("date")["close"].resample("ME").last().pct_change()

aspi = monthly_ret("ASPI")
CO = {}
for s in ALLCO:
    a = (monthly_ret(s) - aspi).rename("abn_ret").to_frame()
    a = a.join(rate[["d_spread"]], how="inner")
    for k in LAGS:
        a[f"d_spread_l{k}"] = rate["d_spread"].shift(k).reindex(a.index)
    a["abn_ret"] *= 100
    a["year"] = a.index.year
    CO[s] = a.reset_index().rename(columns={"index": "date"})

# ---------------- per-company regressions (spec-independent coefficients)
def company_reg(s, k):
    x = f"d_spread_l{k}"
    sub = CO[s][["abn_ret", x]].dropna()
    m = smf.ols(f"abn_ret ~ {x}", data=sub).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {"coef": m.params[x], "se": m.bse[x], "t": m.tvalues[x], "p": m.pvalues[x],
            "r2": m.rsquared, "n": int(m.nobs)}

percomp_cache = {(s, k): company_reg(s, k) for s in ALLCO for k in LAGS}

def interaction(pooled, xcol):
    sub = pooled[["abn_ret", xcol, "is_g1", "date"]].dropna()
    m = smf.ols(f"abn_ret ~ {xcol} * is_g1", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["date"]})
    inter = f"{xcol}:is_g1"
    return {"g2_slope": m.params[xcol], "g1_slope": m.params[xcol] + m.params[inter],
            "interaction": m.params[inter], "se": m.bse[inter], "p": m.pvalues[inter],
            "n": int(m.nobs), "r2": m.rsquared}

# ---------------- run each spec
comp_rows = []
pc_rows = []
bar = []
for name, g1, g2, ctrl, g1l, g2l in SPECS:
    role = {**{c: "G1(" + g1l + ")" for c in g1}, **{c: "G2(" + g2l + ")" for c in g2},
            **{c: "Control" for c in ctrl}}
    # pooled g1+g2
    pooled = pd.concat([CO[c].assign(is_g1=1 if c in g1 else 0) for c in g1 + g2], ignore_index=True)
    lag_res = {k: interaction(pooled, f"d_spread_l{k}") for k in LAGS}
    r0 = lag_res[0]
    ex22 = interaction(pooled[pooled.year != 2022], "d_spread_l0")
    pw = pooled.copy(); lo, hi = pw["abn_ret"].quantile([0.01, 0.99]); pw["abn_ret"] = pw["abn_ret"].clip(lo, hi)
    win = interaction(pw, "d_spread_l0")
    sig_lags = [k for k in LAGS if lag_res[k]["p"] < 0.05]
    ctrl_coefs = [percomp_cache[(c, 0)] for c in ctrl]
    ctrl_mean = np.mean([c["coef"] for c in ctrl_coefs])
    ctrl_maxt = max(abs(c["t"]) for c in ctrl_coefs)
    ctrl_anysig = any(c["p"] < 0.05 for c in ctrl_coefs)
    survives = (r0["p"] < 0.05) and (ex22["p"] < 0.05) and (win["p"] < 0.05)
    comp_rows.append({"spec": name, "g1": g1l, "g2": g2l,
                      "g1_members": "+".join(g1), "g2_members": "+".join(g2),
                      "g1_slope_l0": r0["g1_slope"], "g2_slope_l0": r0["g2_slope"],
                      "interaction_l0": r0["interaction"], "interaction_p_l0": r0["p"],
                      "n": r0["n"], "r2": r0["r2"], "sig_lags": ",".join(map(str, sig_lags)) or "none",
                      "ex2022_p": ex22["p"], "winsor_p": win["p"],
                      "control_mean_coef": ctrl_mean, "control_max_abs_t": ctrl_maxt,
                      "control_any_sig": ctrl_anysig, "survives_all": survives})
    bar.append((name, r0["interaction"], r0["p"], survives))
    for c in g1 + g2 + ctrl:
        for k in LAGS:
            pc = percomp_cache[(c, k)]
            pc_rows.append({"spec": name, "company": c, "role": role[c], "lag": k, **pc})

comp = pd.DataFrame(comp_rows)
comp.round(4).to_csv(OUT / "specification_comparison.csv", index=False)
pd.DataFrame(pc_rows).round(4).to_csv(OUT / "per_company_by_spec.csv", index=False)

# ---------------- console
pd.set_option("display.width", 200)
print("=== SPECIFICATION COMPARISON (contemporaneous d_spread, lag 0) ===")
print(comp[["spec", "g2_members", "g1_slope_l0", "g2_slope_l0", "interaction_l0",
            "interaction_p_l0", "ex2022_p", "winsor_p", "control_any_sig", "survives_all"]].round(4).to_string(index=False))
print("\n=== KEY QUESTION: does the interaction survive dropping LOLC (spec B)? ===")
B = comp[comp.spec == "B_drop_conglomerate"].iloc[0]
print(f"Spec B interaction p = {B['interaction_p_l0']:.4f}  "
      f"(ex-2022 p={B['ex2022_p']:.4f}, winsor p={B['winsor_p']:.4f})  -> "
      f"{'SURVIVES' if B['survives_all'] else 'DOES NOT SURVIVE'}")

# ---------------- bar chart
fig, ax = plt.subplots(figsize=(9, 5.5))
names = [b[0] for b in bar]; coefs = [b[1] for b in bar]; ps = [b[2] for b in bar]; surv = [b[3] for b in bar]
colors = ["tab:green" if s else "tab:orange" for s in surv]
bars = ax.bar(range(len(names)), coefs, color=colors, alpha=0.85)
for i, (c, p) in enumerate(zip(coefs, ps)):
    ax.text(i, c + (0.1 if c >= 0 else -0.3), f"p={p:.3f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
ax.set_ylabel("Interaction coefficient (d_spread x is_g1), lag 0")
ax.set_title("Bank-vs-finance interaction across group specifications\n(green = survives ex-2022 + winsorise; orange = does not)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "interaction_by_spec.png", dpi=140)

# ---------------- recommendation + summary md
A = comp[comp.spec == "A_original"].iloc[0]
Bs = comp[comp.spec == "B_drop_conglomerate"].iloc[0]
Cs = comp[comp.spec == "C_funding_structure"].iloc[0]
Ds = comp[comp.spec == "D_pure_nbfi"].iloc[0]

if not Bs["survives_all"]:
    rec = ("**Spec B collapses.** Dropping LOLC Holdings kills the interaction — the original "
           "result was largely driven by one misclassified conglomerate. This materially weakens "
           "the finding. Report B as the main specification and treat the original (A) as an error.")
elif Cs["interaction_p_l0"] < Bs["interaction_p_l0"] and Cs["survives_all"]:
    rec = ("**Spec C is cleanest.** The distinction is deposit-funded vs borrowing-funded, not "
           "bank vs non-bank — a sharper, more defensible grouping. Report C as main, B as support.")
elif Bs["survives_all"]:
    rec = ("**Spec B holds.** The interaction survives removing the conglomerate, so the effect is "
           "about genuine finance companies. Report B as the main specification, A as a footnote.")
else:
    rec = "Mixed — see per-spec results; no single clean specification dominates."

def slopes_line(r):
    return (f"g1({r['g1']}) slope {r['g1_slope_l0']:+.2f}, g2({r['g2']}) slope {r['g2_slope_l0']:+.2f}, "
            f"interaction {r['interaction_l0']:+.2f} (p={r['interaction_p_l0']:.4f}), "
            f"ex-2022 p={r['ex2022_p']:.4f}, winsor p={r['winsor_p']:.4f}, n={int(r['n'])}, "
            f"control {'OK (ns)' if not r['control_any_sig'] else 'PROBLEM (a control is significant)'}")

md = f"""# Spread Regression — Grouping Robustness Check

The original run put LOLC **Holdings** (a diversified conglomerate) in the finance group; its
+256% month in 2021 was asset-sale rerating, not lending margin. LFIN also behaved like a bank.
CFIN (Central Finance, a genuine licensed NBFI) is added. Four specifications tested.

## Recommendation

{rec}

## Specification results (contemporaneous, lag 0)

| Spec | Finance/G2 group | Interaction | p | ex-2022 p | winsor p | Survives all? | Control clean? |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r['spec']} | {r['g2_members']} | {r['interaction_l0']:+.3f} | {r['interaction_p_l0']:.4f} | "
    f"{r['ex2022_p']:.4f} | {r['winsor_p']:.4f} | {'YES' if r['survives_all'] else 'NO'} | "
    f"{'yes' if not r['control_any_sig'] else 'NO — a control is sig'} |"
    for _, r in comp.iterrows()) + f"""

## Per specification

- **A (original):** {slopes_line(A)}
- **B (drop conglomerate — LOLC to control):** {slopes_line(Bs)}
- **C (deposit- vs borrowing-funded):** {slopes_line(Cs)}
- **D (pure NBFI, LOFC+CFIN only):** {slopes_line(Ds)}

Significant lags per spec: A={A['sig_lags']}, B={Bs['sig_lags']}, C={Cs['sig_lags']}, D={Ds['sig_lags']}.

## Key question — does the interaction survive removing LOLC Holdings?

Spec B interaction p = **{Bs['interaction_p_l0']:.4f}** (ex-2022 {Bs['ex2022_p']:.4f}, winsor {Bs['winsor_p']:.4f})
→ **{'SURVIVES' if Bs['survives_all'] else 'DOES NOT SURVIVE'}**.

## Per-company coefficients (contemporaneous d_spread, lag 0)

| Company | Coef | p | n |
|---|---|---|---|
""" + "\n".join(
    f"| {c} | {percomp_cache[(c,0)]['coef']:+.3f} | {percomp_cache[(c,0)]['p']:.3f} | {percomp_cache[(c,0)]['n']} |"
    for c in ["HNB","COMB","SAMP","LFIN","LOFC","CFIN","LOLC","JKH","DIAL"]) + f"""

(LFIN {percomp_cache[('LFIN',0)]['coef']:+.2f} sits between the groups; LOLC {percomp_cache[('LOLC',0)]['coef']:+.2f} is the conglomerate.)

## Caveats

- **LOLC Holdings is a conglomerate** — including it in a finance group was a classification error
  in the original run.
- **LFIN (LB Finance)** is a deposit-taking LFC that sits structurally between the two groups.
- Only **173 monthly observations**, 2-4 companies per group — n reported per cell.
- **Correlation, not causation.** Note also the sign is *opposite* the margin-channel hypothesis
  (banks negative, finance positive) — this check is about whether the differential is real, not
  about which direction confirms the theory.

*Outputs: specification_comparison.csv, per_company_by_spec.csv, interaction_by_spec.png.*
"""
(OUT / "grouping_check_summary.md").write_text(md)
print(f"\nRECOMMENDATION: {rec}")
print(f"Saved to {OUT}/")
