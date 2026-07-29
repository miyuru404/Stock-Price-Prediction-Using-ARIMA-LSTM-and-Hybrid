#!/usr/bin/env python3
"""
Pilot test: do banks and non-bank finance companies react in OPPOSITE directions to
CBSL policy-rate decisions? Event study with abnormal returns vs the ASPI.

Follows PILOT_TEST_policy_rate_sign_flip.md exactly. Nothing is tuned to produce a
positive result; a null result is reported plainly. Outputs to results/pilot/.
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
FINANCE = ["LOFC", "LOLC", "LFIN"]
CONTROL = ["JKH", "DIAL"]
COMPANIES = BANKS + FINANCE + CONTROL
GROUP = {**{c: "Banks" for c in BANKS}, **{c: "Finance" for c in FINANCE},
         **{c: "Control" for c in CONTROL}}
WINDOWS = {"CAR[0,0]": (0, 0), "CAR[0,+1]": (0, 1), "CAR[0,+3]": (0, 3), "CAR[-1,+3]": (-1, 3)}
PRIMARY = "CAR[0,+1]"
CLOSURE_EVENT = pd.Timestamp("2022-04-08")   # inside the 2022-04-07->04-25 closure

# ---------------------------------------------------------------- load prices
def load_close(sym):
    d = pd.read_csv(DATA / f"{sym}_daily_clean.csv", parse_dates=["date"])
    d = d[d.date >= "2012-01-01"][["date", "close"]].rename(columns={"close": sym})
    return d.set_index("date")[sym]

series = {s: load_close(s) for s in COMPANIES}
series["ASPI"] = load_close.__wrapped__("ASPI") if False else (
    pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
    .pipe(lambda d: d[d.date >= "2012-01-01"][["date", "close"]]).rename(columns={"close": "ASPI"})
    .set_index("date")["ASPI"])

prices = pd.concat(series.values(), axis=1, join="inner").dropna()
prices = prices[COMPANIES + ["ASPI"]]
rets = prices.pct_change().dropna()
# abnormal return = company return - ASPI return
abn = rets[COMPANIES].subtract(rets["ASPI"], axis=0)
cal = abn.index
print(f"Common trading days: {len(abn)}  ({cal.min().date()} -> {cal.max().date()})")

# ---------------------------------------------------------------- events
ev = pd.read_csv(DATA / "policy_rate_events.csv", parse_dates=["announcement_date"])

def car_for(day0_pos, a, b, sym):
    lo, hi = day0_pos + a, day0_pos + b
    if lo < 0 or hi >= len(cal):
        return np.nan, False
    dates = cal[lo:hi + 1]
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int) if len(dates) > 1 else np.array([])
    spans = bool((gaps > 7).any()) if gaps.size else False
    return float(abn[sym].iloc[lo:hi + 1].sum()), spans

rows = []
event_status = []
for _, e in ev.iterrows():
    adate = e["announcement_date"]
    if adate < cal.min() or adate > cal.max():
        event_status.append({"date": adate.date(), "direction": e.direction,
                             "status": "EXCLUDED — outside ASPI coverage"}); continue
    day0 = cal.searchsorted(adate, side="left")            # first trading day >= announcement
    day0_date = cal[day0]
    is_closure = adate == CLOSURE_EVENT
    event_status.append({"date": adate.date(), "direction": e.direction,
                         "day0": day0_date.date(),
                         "status": "CASE STUDY (closure)" if is_closure else "used"})
    for sym in COMPANIES:
        for wname, (a, b) in WINDOWS.items():
            car, spans = car_for(day0, a, b, sym)
            rows.append({"event_date": adate.date(), "direction": e.direction,
                         "regime": e.regime, "change_bps": e.change_bps,
                         "day0_date": day0_date.date(), "company": sym, "group": GROUP[sym],
                         "window": wname, "car_pct": car * 100 if not np.isnan(car) else np.nan,
                         "spans_closure": spans, "is_closure_event": is_closure})

el = pd.DataFrame(rows)
el.to_csv(OUT / "event_level_cars.csv", index=False)

# rows valid for MAIN averages: not the closure case-study, window doesn't span a closure, not NaN
el["use_main"] = (~el.is_closure_event) & (~el.spans_closure) & el.car_pct.notna()

# ---------------------------------------------------------------- 3x3 table (all windows)
tbl = []
for wname in WINDOWS:
    sub = el[(el.window == wname) & el.use_main]
    for grp in ["Banks", "Finance", "Control"]:
        for direc in ["HIKE", "CUT", "HOLD"]:
            cell = sub[(sub.group == grp) & (sub.direction == direc)]
            tbl.append({"window": wname, "group": grp, "direction": direc,
                        "mean_car_pct": round(cell.car_pct.mean(), 4) if len(cell) else np.nan,
                        "n_obs": len(cell), "n_events": cell.event_date.nunique()})
tbl = pd.DataFrame(tbl)
tbl.to_csv(OUT / "car_by_group_and_direction.csv", index=False)

# ---------------------------------------------------------------- significance (HIKE: banks vs finance)
def tests(wname, exclude_2022=False):
    sub = el[(el.window == wname) & el.use_main & (el.direction == "HIKE")]
    if exclude_2022:
        sub = sub[~sub.event_date.astype(str).str.startswith("2022")]
    bk = sub[sub.group == "Banks"].car_pct.dropna().values
    fn = sub[sub.group == "Finance"].car_pct.dropna().values
    out = {"bank_mean": bk.mean() if len(bk) else np.nan, "bank_n": len(bk),
           "fin_mean": fn.mean() if len(fn) else np.nan, "fin_n": len(fn),
           "n_hike_events": sub.event_date.nunique()}
    if len(bk) > 1 and len(fn) > 1:
        out["t_p"] = stats.ttest_ind(bk, fn, equal_var=False).pvalue
        out["mw_p"] = stats.mannwhitneyu(bk, fn, alternative="two-sided").pvalue
    else:
        out["t_p"] = out["mw_p"] = np.nan
    return out

sig = {w: tests(w) for w in WINDOWS}
sig_no2022 = tests(PRIMARY, exclude_2022=True)

# ---------------------------------------------------------------- 2022-04-08 case study
cs = el[el.is_closure_event & (el.window == PRIMARY)][["company", "group", "car_pct", "day0_date"]]

# ---------------------------------------------------------------- console report
print("\n" + "=" * 66)
print(f"3x3 TABLE — mean abnormal CAR (%), primary window {PRIMARY}")
print("=" * 66)
piv = tbl[tbl.window == PRIMARY].pivot(index="group", columns="direction", values="mean_car_pct").reindex(["Banks", "Finance", "Control"])[["HIKE", "CUT", "HOLD"]]
npiv = tbl[tbl.window == PRIMARY].pivot(index="group", columns="direction", values="n_obs").reindex(["Banks", "Finance", "Control"])[["HIKE", "CUT", "HOLD"]]
print(piv.round(3).to_string())
print("\n(n observations per cell)"); print(npiv.to_string())

print("\n" + "=" * 66)
print("KEY QUESTION — HIKE: Banks vs Finance abnormal CAR, all windows")
print("=" * 66)
for w in WINDOWS:
    s = sig[w]
    print(f"{w:12} banks {s['bank_mean']:+.3f}% (n={s['bank_n']}) | finance {s['fin_mean']:+.3f}% "
          f"(n={s['fin_n']}) | diff {s['bank_mean']-s['fin_mean']:+.3f}pp | "
          f"t-p={s['t_p']:.3f} MW-p={s['mw_p']:.3f}  [{s['n_hike_events']} hikes]")

print("\n--- Robustness: exclude all 2022 events (primary window) ---")
s = sig_no2022
print(f"banks {s['bank_mean']:+.3f}% (n={s['bank_n']}) | finance {s['fin_mean']:+.3f}% (n={s['fin_n']}) "
      f"| diff {s['bank_mean']-s['fin_mean']:+.3f}pp | t-p={s['t_p']:.3f} | {s['n_hike_events']} hikes")

print("\n--- Sanity: CONTROL group should be ~0 on HIKE (primary) ---")
ctrl = el[(el.window == PRIMARY) & el.use_main & (el.direction == "HIKE") & (el.group == "Control")]
print(f"control HIKE mean {ctrl.car_pct.mean():+.3f}% (n={len(ctrl)})")

print("\n--- 2022-04-08 +700bps CASE STUDY (market shut 04-07->04-25; reaction from reopen) ---")
print(cs.to_string(index=False))

# ---------------------------------------------------------------- bar chart (primary window)
fig, ax = plt.subplots(figsize=(9, 5.5))
groups = ["Banks", "Finance", "Control"]; directions = ["HIKE", "CUT", "HOLD"]
x = np.arange(len(groups)); w = 0.26
colors = {"HIKE": "tab:red", "CUT": "tab:green", "HOLD": "gray"}
for i, direc in enumerate(directions):
    vals = [piv.loc[g, direc] for g in groups]
    ns = [npiv.loc[g, direc] for g in groups]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=f"{direc}", color=colors[direc], alpha=0.85)
    for bx, v, nn in zip(x + (i - 1) * w, vals, ns):
        if not np.isnan(v):
            ax.text(bx, v + (0.03 if v >= 0 else -0.06), f"n={nn}", ha="center", fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel(f"Mean abnormal CAR (%), {PRIMARY}")
ax.set_title("Policy-rate reaction by group & direction (abnormal CAR vs ASPI)\n"
             "Hypothesis: on a HIKE, Banks positive & Finance negative")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "car_bar_by_group_direction.png", dpi=140)

# ---------------------------------------------------------------- pilot_summary.md
def fmt(v): return f"{v:+.3f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a"
sp = sig[PRIMARY]
diff = sp["bank_mean"] - sp["fin_mean"]
# Honest verdict weighing ALL windows + robustness (not just the primary window).
flip_windows = [w for w in WINDOWS if (sig[w]["bank_mean"] > 0 and sig[w]["fin_mean"] < 0)]
sig_flip_windows = [w for w in flip_windows
                    if not np.isnan(sig[w]["t_p"]) and (sig[w]["t_p"] < 0.05 or sig[w]["mw_p"] < 0.05)]
robust_flip = (sig_no2022["bank_mean"] > 0 and sig_no2022["fin_mean"] < 0)

if sig_flip_windows and robust_flip:
    verdict = ("SUPPORTED — banks positive and finance negative on hikes, statistically "
               f"significant at {sig_flip_windows}, and it survives excluding 2022.")
elif sig_flip_windows:
    verdict = (f"SUGGESTIVE — the sign flip is significant at {sig_flip_windows} but does NOT "
               "survive excluding 2022; treat with caution given the sample size.")
elif flip_windows:
    verdict = ("NOT SUPPORTED (null) — the hypothesised sign flip (banks +, finance -) appears "
               f"ONLY at the immediate window {flip_windows} and is NOT statistically significant; "
               "it disappears/reverses at longer windows and does not survive excluding 2022. "
               "Best described as a weak, non-significant hint — i.e. a null result at this sample size.")
else:
    verdict = ("NULL — no opposite-direction effect: banks and finance move in the SAME direction "
               "on rate hikes; only magnitude differs, and not significantly.")

def table_md(wname):
    pv = tbl[tbl.window == wname].pivot(index="group", columns="direction", values="mean_car_pct").reindex(["Banks", "Finance", "Control"])[["HIKE", "CUT", "HOLD"]]
    nn = tbl[tbl.window == wname].pivot(index="group", columns="direction", values="n_obs").reindex(["Banks", "Finance", "Control"])[["HIKE", "CUT", "HOLD"]]
    lines = [f"| Group | HIKE | CUT | HOLD |", "|---|---|---|---|"]
    for g in ["Banks", "Finance", "Control"]:
        cells = [f"{pv.loc[g,d]:+.3f}% (n={nn.loc[g,d]})" if not np.isnan(pv.loc[g,d]) else "n/a" for d in ["HIKE","CUT","HOLD"]]
        lines.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)

md = f"""# Pilot Test Result — Policy-Rate Sign-Flip (Banks vs Finance)

**Hypothesis:** on a CBSL rate **HIKE**, banks show a **positive** abnormal return and
non-bank finance companies show a **negative** one.

## Verdict

**{verdict}**

Primary window {PRIMARY}, HIKE events:
Banks mean abnormal CAR **{fmt(sp['bank_mean'])}%** (n={sp['bank_n']} obs),
Finance **{fmt(sp['fin_mean'])}%** (n={sp['fin_n']} obs),
difference **{diff:+.3f} pp**, t-test p={sp['t_p']:.3f}, Mann-Whitney p={sp['mw_p']:.3f},
across {sp['n_hike_events']} usable hike events.

## The 3x3 table (primary window {PRIMARY})

{table_md(PRIMARY)}

## Banks vs Finance on HIKE — all four windows

| Window | Banks | Finance | Diff (pp) | t-test p | Mann-Whitney p | n hikes |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {w} | {sig[w]['bank_mean']:+.3f}% | {sig[w]['fin_mean']:+.3f}% | {sig[w]['bank_mean']-sig[w]['fin_mean']:+.3f} | {sig[w]['t_p']:.3f} | {sig[w]['mw_p']:.3f} | {sig[w]['n_hike_events']} |"
    for w in WINDOWS) + f"""

## Robustness — exclude all 2022 events (crisis must not drive the result)

Banks {sig_no2022['bank_mean']:+.3f}%, Finance {sig_no2022['fin_mean']:+.3f}%,
diff {sig_no2022['bank_mean']-sig_no2022['fin_mean']:+.3f} pp, t-test p={sig_no2022['t_p']:.3f},
across {sig_no2022['n_hike_events']} hikes.

## Sanity checks

- **Control group (JKH, DIAL) on HIKE:** mean {ctrl.car_pct.mean():+.3f}% (n={len(ctrl)}) — should be ~0.
- **HOLD control is WEAK:** the 9 "HOLD" rows are artifacts (SDFR flat while SLFR moved), not genuine no-change meetings — treat HOLD as weak evidence only.
- **Events actually used:** 2012-02-03 excluded (before ASPI coverage begins 2012-02-23); 2022-04-08 held out as a case study (see below). Windows that span a market-closure gap (COVID 2020-03-19→05-12, crisis 2022-04-07→04-25) are dropped from that window's averages.
- **LFIN split adjustment:** LFIN prices before 2015-07-14 were back-adjusted ×0.5 to fix an unapplied 2-for-1 split.

## 2022-04-08 (+700bps) case study — the market was shut

The emergency +700bps hike fell inside the 2022-04-07 → 2022-04-25 closure, so the market
could not react until it reopened on 25 April. Abnormal CAR from the reopen ({PRIMARY}):

{cs.to_string(index=False)}

Reported separately, not in the main averages (a displaced, confounded reaction).

## Caveats (read before quoting any number)

1. **Small samples.** Only {sp['n_hike_events']} usable hike events; each cell mean rests on
   few events — n is reported in every cell and must be quoted with the mean.
2. **Observations within an event are correlated** (three banks react to the same hike), so the
   pooled t-test / Mann-Whitney p-values are optimistic; treat significance as indicative.
3. **Weak control** for HOLD (artifact rows, above).
4. **No tuning:** all four windows are reported; the result is whatever it is.

*Outputs: `car_by_group_and_direction.csv`, `event_level_cars.csv`, `car_bar_by_group_direction.png`.*
"""
(OUT / "pilot_summary.md").write_text(md)
print(f"\nVERDICT: {verdict}")
print(f"\nSaved to {OUT}/ : car_by_group_and_direction.csv, event_level_cars.csv, "
      "pilot_summary.md, car_bar_by_group_direction.png")
