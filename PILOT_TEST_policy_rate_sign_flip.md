# Pilot Test — Do Banks and Finance Companies React in Opposite Directions to CBSL Policy Rate Decisions?

**Purpose:** Test the core hypothesis of the FYP in miniature, using only data that is free and already partly collected. One week of work. The result decides whether the main project proceeds as planned.

**Hypothesis (write this down before you look at any results):**

> When the CBSL **raises** the policy rate, banks show a **positive** abnormal return and non-bank finance companies show a **negative** abnormal return, because banks' lending rates reprice faster than their deposit costs (margin widens), while finance companies fund themselves through borrowings whose cost rises immediately.

**Why this matters:** If the sign flip exists, the sector-conditional gap is real and empirically demonstrated on own data. If it does not, this is learned in week 1 rather than month 6.

---

## PART 1 — DATA (COLLECTED — see status below)

All data is collected and cleaned. Everything lives in `cleaned_data/` with the
schema `date, ticker, open, high, low, close, volume`.

### 1.1 Price data — ALL DONE

| Group | Company | File | Rows | Coverage |
|---|---|---|---|---|
| Bank | Hatton National Bank | `HNB_daily_clean.csv` | 3,270 | 2012-02-22 → 2026-07-27 |
| Bank | Commercial Bank of Ceylon | `COMB_daily_clean.csv` | 3,360 | 2012-02-23 → 2026-07-27 |
| Bank | Sampath Bank | `SAMP_daily_clean.csv` | 3,404 | 2012-02-23 → 2026-07-27 |
| Finance | LOLC Finance | `LOFC_daily_clean.csv` | 3,208 | 2012-02-23 → 2026-07-27 |
| Finance | LOLC Holdings | `LOLC_daily_clean.csv` | 3,188 | 2012-02-22 → 2026-07-27 |
| Finance | LB Finance | `LFIN_daily_clean.csv` | 3,104 | 2012-02-23 → 2026-07-28 |
| Control | John Keells Holdings | `JKH_daily_clean.csv` | 3,363 | 2012-02-22 → 2026-07-27 |
| Control | Dialog Axiata | `DIAL_daily_clean.csv` | 3,336 | 2012-02-23 → 2026-07-27 |
| **Benchmark** | **ASPI (All Share Index)** | `ASPI_daily_clean.csv` | **3,419** | **2012-02-23 → 2026-07-28** |

Also present but **not used in this pilot**: `CTC`, `DIST`, `MELS`.

**Data quality notes carried forward:**
- All files verified: zero OHLC violations, zero duplicate dates.
- **LFIN prices before 2015-07-14 were back-adjusted by ×0.5** to correct an
  unadjusted 2-for-1 share split that TradingView had not applied
  (57.00 → 29.25 overnight). All other files were already adjusted correctly.
- ASPI verified against known index levels (7,348 at Jan-2015; 4,572 at
  Mar-2020; 12,226 at Dec-2021; 8,490 at Dec-2022) — all plausible.
- Two genuine market closures exist in every series:
  **2020-03-19 → 2020-05-12** (COVID, 54 days) and
  **2022-04-07 → 2022-04-25** (economic crisis, 18 days).

### 1.2 CBSL policy rate decisions — DONE

**File:** `cleaned_data/policy_rate_events.csv`
**Source:** CBSL `historical_policy_interest_rates.xlsx` (official, sheets
"Historical Policy Rates" and "OPR")

| Column | Description | Example |
|---|---|---|
| `announcement_date` | Date the decision was publicly announced | 2022-04-08 |
| `rate_new` | New policy rate (%) — SDFR pre-Nov-2024, OPR after | 13.50 |
| `rate_old` | Previous policy rate (%) | 6.50 |
| `change_bps` | Change in basis points | 700 |
| `direction` | `HIKE` / `CUT` / `HOLD` | HIKE |
| `regime` | `SDFR_SLFR` or `OPR` | SDFR_SLFR |

**Actual count: 35 events (2012 onwards)** — not the ~110 originally assumed.

| Direction | Count |
|---|---|
| CUT | 15 |
| HIKE | 11 |
| HOLD | 9 |

**Three things to know about this file:**

1. **CBSL's own spreadsheet omitted the 8 April 2022 emergency hike**
   (SDFR 6.50 → 13.50, +700bps) — it jumps straight from 04.03.2022 to
   07.07.2022. This event has been **added back manually** and the
   2022-07-07 `rate_old` corrected to 13.50.

2. **The 9 "HOLD" rows are artifacts, not genuine policy holds.** They are
   dates where SDFR stayed flat while SLFR moved. CBSL's sheet only records
   dates where a rate *changed*, so true no-change policy meetings are absent.
   **The control group is therefore weak** — state this in the writeup. To get
   genuine holds you would need the Monetary Policy Review press-release dates.

3. **Policy framework changed on 27 November 2024** — SDFR/SLFR were replaced
   by a single Overnight Policy Rate (OPR). The `regime` column flags this.
   31 events are `SDFR_SLFR`, 3 are `OPR`.

A full history back to 2000 (90 events) is also available in
`policy_rate_events_full_since2000.csv` if a longer window is ever needed.

### 1.3 Deliberately NOT collected for this pilot

AWPLR, USD/LKR, inflation, oil, news and company announcements are **out of
scope**. They belong to Layer 1 onwards. Collecting them now would delay the
test without changing its result.

---

## PART 2 — METHOD

### Step 1 — Build the daily return series

For every company, and for the ASPI:

```
return_t = (close_t / close_{t-1}) - 1
```

Use the cleaned CSV files (`date, ticker, open, high, low, close, volume`).

### Step 2 — Compute abnormal return

This is the step that makes the test valid. It removes the effect of the whole market moving.

```
abnormal_return_t = company_return_t - ASPI_return_t
```

Without this, a rate hike that happens on a day the whole market rallied would look like a positive bank reaction when it was really just the market.

### Step 3 — Define the event window

For each policy decision, let **day 0** = announcement date (or next trading day if announced after close).

Compute the **cumulative abnormal return (CAR)** over several windows:

| Window | Days | What it captures |
|---|---|---|
| `CAR[0,0]` | Day 0 only | Immediate reaction |
| `CAR[0,+1]` | Day 0 to +1 | Reaction plus next-day follow-through |
| `CAR[0,+3]` | Day 0 to +3 | Short-term absorption |
| `CAR[-1,+3]` | Day −1 to +3 | Includes possible information leakage |

```
CAR[a,b] = sum of abnormal_return_t for t = a to b
```

Test all four. Report which window shows the clearest effect.

### Step 4 — Group and average

For each event window, split the events by direction and the companies by type, then average the CAR:

| | Banks (HNB, COMB, SAMP) | Finance (LOFC, LOLC, LFIN) | Control (JKH, DIAL) |
|---|---|---|---|
| **HIKE events** | mean CAR = ? | mean CAR = ? | should be ≈ 0 |
| **CUT events** | mean CAR = ? | mean CAR = ? | should be ≈ 0 |
| **HOLD events** | mean CAR = ? | mean CAR = ? | should be ≈ 0 |

**This 3×3 table is the entire result of the pilot.** Report `n` in every cell.

### Step 5 — Test whether the difference is real

Averages alone are not enough — a difference could be noise.

- Run **both** a two-sample t-test and a Mann-Whitney U test comparing bank CARs vs finance CARs on HIKE events. Report both, since with this sample size normality cannot be assumed.
- Report the **p-value**. `p < 0.05` means the difference is unlikely to be chance.
- **Report `n` for every cell.** With only 35 events split across three directions — and 11 HIKEs in total — some cells will be small. Say so honestly rather than quoting a mean from four observations as if it were solid.

### Step 6 — Sanity checks (do not skip)

- [ ] **CONTROL group (JKH, DIAL) should show CAR near zero.** These are non-financial companies with no direct rate-margin channel. If they move as much as the banks, the method is picking up something other than the policy effect and the result is invalid.
- [ ] **HOLD events should show near-zero CAR for all groups** — but remember these 9 "holds" are artifacts, not genuine no-change meetings, so treat this check as weak evidence only.
- [ ] **ASPI covers 2012-02-23 → 2026-07-28. 34 of the 35 events are covered; only 2012-02-03 falls outside.** Exclude it and report how many events were actually used.
- [ ] **The 2022-04-08 event (+700bps) falls inside the 2022-04-07 → 2022-04-25 market closure.** The market could not react until it reopened on 25 April. Report it as a separate case study, not in the main averages.
- [ ] Check whether any other event window overlaps the closures (2020-03-19 → 2020-05-12, 2022-04-07 → 2022-04-25). Exclude or flag them.
- [ ] Re-run excluding all of 2022 and confirm the pattern survives — the crisis must not be single-handedly driving the result.

---

## PART 3 — WHAT SUCCESS LOOKS LIKE

### Strong result (hypothesis supported)

```
                    HIKE events
Banks:              +0.6%   (mean CAR[0,+1])
Non-bank finance:   -0.9%
Difference:          1.5 percentage points,  p = 0.02
HOLD events:        ~0.0% for both groups
```

Opposite signs, statistically significant, and holds show nothing. **The gap is proven. Proceed with the full project.**

### Partial result (still useful)

Same sign for both groups but clearly different **magnitude** — e.g. banks +0.2%, finance companies −0.1% but not significant. Reframe from "opposite direction" to "differential magnitude of response". Weaker but still a real conditional effect, and still unaddressed in the literature.

### Null result (no effect found)

No meaningful difference between groups. **This is not a failed project.** Options:
1. Try longer windows — the effect may take a week to appear, not a day.
2. Try the AWPLR instead of the policy rate — it is closer to actual bank margins.
3. Pivot the gap to the parts that do not depend on this: calibrated confidence, explainability, and the deployed live pipeline. All three stand alone.

Record the null result either way — "the differential response was not detectable at this sample size on CSE data" is a legitimate, reportable finding that no paper has established.

---

## PART 4 — OUTPUT TO PRODUCE

Write everything to `results/pilot/`:

1. `car_by_group_and_direction.csv` — the 3×3 table, for all four event windows, with `n` per cell
2. `event_level_cars.csv` — one row per (event × company), so results can be audited
3. `pilot_summary.md` — the tables plus one paragraph stating plainly whether banks and finance companies moved in **opposite directions** on rate hikes
4. A bar chart of mean CAR by group and direction
5. A short **caveats** section listing: number of events actually used, `n` per cell, the weak HOLD control, the 2022-04-08 closure case, and the LFIN split adjustment

---

## COMMON MISTAKES TO AVOID

| Mistake | Consequence |
|---|---|
| Using raw returns instead of abnormal returns | Measuring the market, not the announcement |
| Using effective date instead of announcement date | Off by days or weeks; effect disappears |
| Treating the 9 "HOLD" rows as genuine policy holds | False confidence in a control group that is really an artifact |
| Event window spanning a market closure | Meaningless CAR values |
| Including 2022-04-08 in the main averages | The market was shut; reaction is displaced to 25 April |
| Choosing the best window *after* seeing results and reporting only that | Cherry-picking — report all four windows |
| Omitting the CONTROL group | Lose the only real check that the method works |
| Quoting a mean without its `n` | A mean of four events reads as if it were solid |
| Tuning anything to produce a positive result | Invalidates the whole exercise |

---

## STATUS

**Data collection: COMPLETE.** All files are cleaned and in `cleaned_data/`.

**Remaining work: the analysis only** — roughly 2–3 hours.

| Task | Status |
|---|---|
| Collect CBSL policy rate decisions | DONE — 35 events |
| Export and clean ASPI | DONE — 3,419 rows, 2012–2026 |
| Export and clean 8 companies | DONE |
| Compute abnormal returns and CARs | TO DO |
| Results table + significance tests | TO DO |
| Sanity checks and writeup | TO DO |
