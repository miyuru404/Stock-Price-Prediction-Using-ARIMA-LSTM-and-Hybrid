#!/usr/bin/env python3
"""
Clean the newly collected CBSL macro sources into machine-readable CSVs in cleaned_data/.

Sources (all from ~/Downloads, originals copied to raw_exports/cbsl/):
  1. 5 CBSL "Movements of CCPI/NCPI" PDFs  -> monthly inflation
  2. Data.xls (actually an HTML table)      -> monthly money supply (M1, M2, M2b, M4, reserve money)
  3. data.csv                               -> DAILY USD/LKR exchange rate

Key jobs:
  * CCPI arrives in FOUR different base years (2002, 2006/07, 2013, 2021). Each rebase restarts
    the index at 100, so the raw series cannot be concatenated. They DO overlap, so this script
    CHAINS them into one continuous index on the newest base (2021=100) using the mean ratio
    over each overlap window. Raw per-base values are also kept as an audit trail.
  * Month-on-month and year-on-year % are RECOMPUTED from the spliced index (the PDFs leave YoY
    blank for the first year of each new base). The PDF-reported figures are kept alongside and
    the largest disagreement is printed as a quality check.
  * Everything lands as tidy long/wide CSVs with a `date` column at month-end (or trading day),
    matching the conventions of the other files in cleaned_data/.

Run:  python src/clean_macro_sources.py
"""
import html
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Downloads"
OUT = ROOT / "cleaned_data"
RAW = ROOT / "raw_exports" / "cbsl"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_NO = {m: i + 1 for i, m in enumerate(MONTHS)}

# (file, series, base label, sort order for chaining)
CPI_FILES = [
    ("MovementsofCCPI2002.pdf", "CCPI", "2002=100", 1),
    ("MovementsofCCPI_200607.pdf", "CCPI", "2006/07=100", 2),
    ("MOVEMENTS_of_CCPI_with_MV_Base2013-100.pdf", "CCPI", "2013=100", 3),
    ("MOVEMENTS_of_CCPI_with_MV_Base2021.pdf", "CCPI", "2021=100", 4),
    ("MovementsOf-NCPI-2013.pdf", "NCPI", "2013=100", 1),
]

NUM = r"-?[\d,]+\.?\d*%?"
ROW_RE = re.compile(rf"^\s*(?:(\d{{4}})\s+)?({'|'.join(MONTHS)})\s+({NUM})"
                    rf"(?:\s+({NUM}))?(?:\s+({NUM}))?(?:\s+({NUM}))?\s*$")


def num(x):
    if x is None or str(x).strip() in ("", "-"):
        return np.nan
    return float(str(x).replace(",", "").replace("%", ""))


def parse_cpi_pdf(path, series, base):
    """CBSL layout: [year] Month  index  [mom%]  [yoy%]  [12m-avg%]. Year only on January rows."""
    text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    rows, year = [], None
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        y, mon, v1, v2, v3, v4 = m.groups()
        if y:
            year = int(y)
        elif year is None:
            continue                       # data before the first year marker
        elif mon == "January":
            year += 1                      # new year with no explicit label
        rows.append({
            "date": pd.Timestamp(year=year, month=MONTH_NO[mon], day=1) + pd.offsets.MonthEnd(0),
            "series": series, "base": base,
            "index": num(v1), "mom_pct_reported": num(v2),
            "yoy_pct_reported": num(v3), "ma12_pct_reported": num(v4),
        })
    df = pd.DataFrame(rows)
    # PDFs repeat the last row of a page at the top of the next -> drop duplicates
    return df.drop_duplicates(subset=["date"], keep="first").sort_values("date").reset_index(drop=True)


def splice(frames):
    """Chain rebased index series onto the NEWEST base using the mean ratio over each overlap."""
    frames = sorted(frames, key=lambda t: t[0])           # oldest -> newest
    newest = frames[-1][1].set_index("date")["index"]
    spliced = {d: v for d, v in newest.items()}
    factor_log = []
    running = newest
    for _, older in reversed(frames[:-1]):
        old = older.set_index("date")["index"]
        overlap = old.index.intersection(running.index)
        if len(overlap) == 0:
            raise ValueError("no overlap to chain on — cannot splice safely")
        ratio = float((running[overlap] / old[overlap]).mean())
        factor_log.append({"base": older["base"].iloc[0], "overlap_months": len(overlap),
                           "scale_factor": round(ratio, 6),
                           "overlap_from": str(overlap.min().date()),
                           "overlap_to": str(overlap.max().date())})
        rescaled = old * ratio
        for d, v in rescaled.items():
            spliced.setdefault(d, v)                      # newer base always wins
        running = rescaled
    s = pd.Series(spliced).sort_index()
    s.index.name = "date"
    return s, pd.DataFrame(factor_log)


print("=" * 78)
print("1. INFLATION (CBSL CCPI + NCPI PDFs)")
print("=" * 78)
per_base = []
for fname, series, base, order in CPI_FILES:
    p = SRC / fname
    if not p.exists():
        print(f"  MISSING: {fname}")
        continue
    shutil.copy2(p, RAW / fname)
    df = parse_cpi_pdf(p, series, base)
    per_base.append((order, df))
    print(f"  {series:5s} {base:12s} {len(df):4d} months  "
          f"{df.date.min():%Y-%m} -> {df.date.max():%Y-%m}  ({fname})")

RAWCPI = pd.concat([d for _, d in per_base], ignore_index=True)
RAWCPI.to_csv(OUT / "inflation_cpi_by_base.csv", index=False)

ccpi_frames = [(o, d) for o, d in per_base if d["series"].iloc[0] == "CCPI"]
ncpi_frames = [(o, d) for o, d in per_base if d["series"].iloc[0] == "NCPI"]

ccpi, factors = splice(ccpi_frames)
print("\n  CCPI chaining (onto 2021=100):")
for _, r in factors.iterrows():
    print(f"    {r['base']:12s} x {r['scale_factor']:<10.6f} "
          f"({r['overlap_months']} overlap months {r['overlap_from']} .. {r['overlap_to']})")
factors.to_csv(OUT / "inflation_ccpi_splice_factors.csv", index=False)

INF = pd.DataFrame({"date": ccpi.index, "ccpi_index_2021base": ccpi.values})
INF["ccpi_mom_pct"] = (INF["ccpi_index_2021base"].pct_change() * 100).round(2)
INF["ccpi_yoy_pct"] = (INF["ccpi_index_2021base"].pct_change(12) * 100).round(2)
INF["ccpi_ma12_yoy_pct"] = INF["ccpi_yoy_pct"].rolling(12).mean().round(2)

ncpi = ncpi_frames[0][1].set_index("date")["index"] if ncpi_frames else pd.Series(dtype=float)
INF = INF.merge(ncpi.rename("ncpi_index_2013base").reset_index(), on="date", how="left")
INF["ncpi_mom_pct"] = (INF["ncpi_index_2013base"].pct_change() * 100).round(2)
INF["ncpi_yoy_pct"] = (INF["ncpi_index_2013base"].pct_change(12) * 100).round(2)
INF["ccpi_index_2021base"] = INF["ccpi_index_2021base"].round(3)

# quality check: our recomputed YoY vs what the PDF printed
chk = RAWCPI[(RAWCPI.series == "CCPI")][["date", "yoy_pct_reported"]].dropna()
chk = chk.merge(INF[["date", "ccpi_yoy_pct"]], on="date", how="inner")
chk["diff"] = (chk.ccpi_yoy_pct - chk.yoy_pct_reported).abs()
print(f"\n  Check vs PDF-reported YoY: {len(chk)} months compared, "
      f"mean gap {chk['diff'].mean():.2f} pp, max {chk['diff'].max():.2f} pp")
worst = chk.loc[chk['diff'].idxmax()]
print(f"    largest gap {worst['date']:%Y-%m}: ours {worst.ccpi_yoy_pct:.1f} vs reported {worst.yoy_pct_reported:.1f}")

INF.to_csv(OUT / "inflation_monthly.csv", index=False)
print(f"\n  -> cleaned_data/inflation_monthly.csv  ({len(INF)} months, "
      f"{INF.date.min():%Y-%m} -> {INF.date.max():%Y-%m})")

print("\n" + "=" * 78)
print("2. MONEY SUPPLY (Data.xls — HTML table, not a real .xls)")
print("=" * 78)
xls = next((SRC / n for n in ["Data.xls", "Data (1).xls"] if (SRC / n).exists()), None)
if xls is None:
    print("  MISSING: Data.xls")
else:
    shutil.copy2(xls, RAW / "money_supply_Data.xls")
    raw = xls.read_text(encoding="utf-8", errors="replace")
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", raw, re.S | re.I)

    def cells(tr):
        return [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]

    header = cells(trs[0])
    dates, keep = [], []
    for i, h in enumerate(header[4:], start=4):
        try:
            dates.append(pd.Timestamp(h) + pd.offsets.MonthEnd(0))
            keep.append(i)
        except Exception:
            pass

    NAME = {"Reserve Money": "reserve_money", "Narrow Money Supply M1": "m1",
            "Broad Money Supply M2": "m2", "Broad Money M2b": "m2b",
            "Broad Money Supply M4": "m4"}
    data = {}
    for tr in trs[1:]:
        c = cells(tr)
        if len(c) < 5 or c[1] not in NAME:
            continue
        data[NAME[c[1]]] = [num(c[i]) if i < len(c) else np.nan for i in keep]

    MS = pd.DataFrame({"date": dates, **data}).dropna(how="all", subset=list(data))
    for col in data:
        MS[f"{col}_mom_pct"] = (MS[col].pct_change() * 100).round(3)
        MS[f"{col}_yoy_pct"] = (MS[col].pct_change(12) * 100).round(3)
    MS.to_csv(OUT / "money_supply_monthly.csv", index=False)
    print(f"  columns: {list(data)}  (LKR million)")
    print(f"  -> cleaned_data/money_supply_monthly.csv  ({len(MS)} months, "
          f"{MS.date.min():%Y-%m} -> {MS.date.max():%Y-%m})")

print("\n" + "=" * 78)
print("3. USD/LKR EXCHANGE RATE (data.csv — DAILY)")
print("=" * 78)
fx_src = SRC / "data.csv"
if not fx_src.exists():
    print("  MISSING: data.csv")
else:
    shutil.copy2(fx_src, RAW / "usd_lkr_data.csv")
    fx = pd.read_csv(fx_src)
    fx.columns = [c.strip().strip('"') for c in fx.columns]
    fx = fx.rename(columns={"Date": "date", "Exchange Rate": "usd_lkr", "Currency": "currency"})
    fx["date"] = pd.to_datetime(fx["date"])
    fx = (fx[fx.currency == "USD"][["date", "usd_lkr"]]
          .dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True))
    fx["usd_lkr_ret_1"] = fx["usd_lkr"].pct_change().round(6)
    fx["usd_lkr_ret_5"] = fx["usd_lkr"].pct_change(5).round(6)
    fx["usd_lkr_ret_20"] = fx["usd_lkr"].pct_change(20).round(6)
    fx["usd_lkr_vol_20"] = fx["usd_lkr_ret_1"].rolling(20).std().round(6)
    fx.to_csv(OUT / "usd_lkr_daily.csv", index=False)
    gap = fx["date"].diff().dt.days
    print(f"  -> cleaned_data/usd_lkr_daily.csv  ({len(fx)} days, "
          f"{fx.date.min():%Y-%m-%d} -> {fx.date.max():%Y-%m-%d})")
    print(f"  largest gap between observations: {int(gap.max())} days "
          f"(weekends/holidays expected)")

# ---------------------------------------------------------------- reference paper
paper = SRC / "34.md"
if paper.exists():
    dest = ROOT / "references" / "naik_padhi_2012_macro_stock_india.md"
    shutil.copy2(paper, dest)
    print(f"\n  reference paper -> references/{dest.name} (Naik & Padhi, Indian macro vs BSE Sensex)")

print("\n" + "=" * 78)
print("4. DATA-QUALITY / USABILITY REPORT")
print("=" * 78)
# The modelling test window is roughly 2024-2026 (last 20% of the daily stock data), so a series
# that stops before then is unusable for the test set no matter how long its history is.
TEST_WINDOW_START = pd.Timestamp("2024-01-01")
STOCK_END = pd.Timestamp("2026-07-27")          # last HNB trading day

qrows = []


def assess(name, dates, freq, expected_per_year, note=""):
    dates = pd.Series(pd.to_datetime(dates)).sort_values()
    covers_test = dates.max() >= TEST_WINDOW_START
    months_short = (STOCK_END.to_period("M") - dates.max().to_period("M")).n
    per_year = dates.groupby(dates.dt.year).size()
    dense_from = next((int(y) for y, n in per_year.items() if n >= expected_per_year * 0.9), None)
    if not covers_test:
        verdict = "UNUSABLE for the test window"
    elif months_short > 2:
        verdict = f"usable but ends {months_short} months early"
    else:
        verdict = "OK"
    qrows.append({"file": name, "freq": freq, "n": len(dates),
                  "from": f"{dates.min():%Y-%m-%d}", "to": f"{dates.max():%Y-%m-%d}",
                  "dense_from": dense_from, "months_behind_stock_data": months_short,
                  "covers_2024_test_window": covers_test, "verdict": verdict, "note": note})


assess("inflation_monthly.csv", INF["date"], "monthly", 12,
       "CCPI chained across 4 base years; NCPI single base 2014-2022 only")
if xls is not None:
    assess("money_supply_monthly.csv", MS["date"], "monthly", 12,
           "source export has 24 BLANK trailing months (2024-Sep..2026-Aug)")
if fx_src.exists():
    assess("usd_lkr_daily.csv", fx["date"], "daily", 240,
           "2010-2013 is sparse (30-162 obs/yr); genuinely daily from 2014")

Q = pd.DataFrame(qrows)
Q.to_csv(OUT / "_macro_quality_report.csv", index=False)
print(Q[["file", "freq", "n", "from", "to", "dense_from", "verdict"]].to_string(index=False))

print("\n  READ THIS BEFORE MODELLING:")
print("  * usd_lkr_daily.csv  -> BEST of the three. Daily, current to 2026-07-31.")
print("                          Use from 2014 onward; 2010-2013 is patchy.")
print("  * inflation_monthly.csv -> good history AND current (to 2026-07). Monthly, so it is")
print("                          flat within a month — same weakness that sank Phase C rates.")
print("  * money_supply_monthly.csv -> WARNING: ends 2024-08, but the test window is 2024-2026.")
print("                          Including M1/M2 would drop nearly the whole test set.")
print("                          Use it for explanation/long-run work, NOT for the direction test.")
print("  * Phase C lesson still applies: feed these as CHANGES (yoy/mom/returns), never levels.")

print("\n" + "=" * 78)
print("SUMMARY — new files in cleaned_data/")
print("=" * 78)
for f in ["inflation_monthly.csv", "inflation_cpi_by_base.csv",
          "inflation_ccpi_splice_factors.csv", "money_supply_monthly.csv",
          "usd_lkr_daily.csv", "_macro_quality_report.csv"]:
    p = OUT / f
    if p.exists():
        d = pd.read_csv(p)
        print(f"  {f:38s} {len(d):5d} rows x {d.shape[1]:2d} cols")
print("\nOriginals archived in raw_exports/cbsl/")
