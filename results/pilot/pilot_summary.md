# Pilot Test Result — Policy-Rate Sign-Flip (Banks vs Finance)

**Hypothesis:** on a CBSL rate **HIKE**, banks show a **positive** abnormal return and
non-bank finance companies show a **negative** one.

## Verdict

**NOT SUPPORTED (null) — the hypothesised sign flip (banks +, finance -) appears ONLY at the immediate window ['CAR[0,0]'] and is NOT statistically significant; it disappears/reverses at longer windows and does not survive excluding 2022. Best described as a weak, non-significant hint — i.e. a null result at this sample size.**

Primary window CAR[0,+1], HIKE events:
Banks mean abnormal CAR **+0.827%** (n=21 obs),
Finance **+0.678%** (n=21 obs),
difference **+0.149 pp**, t-test p=0.921, Mann-Whitney p=0.039,
across 7 usable hike events.

## The 3x3 table (primary window CAR[0,+1])

| Group | HIKE | CUT | HOLD |
|---|---|---|---|
| Banks | +0.827% (n=21) | +0.229% (n=42) | -1.148% (n=24) |
| Finance | +0.678% (n=21) | +0.450% (n=42) | +0.480% (n=24) |
| Control | -0.050% (n=14) | +0.781% (n=28) | +0.044% (n=16) |

## Banks vs Finance on HIKE — all four windows

| Window | Banks | Finance | Diff (pp) | t-test p | Mann-Whitney p | n hikes |
|---|---|---|---|---|---|---|
| CAR[0,0] | +0.274% | -0.527% | +0.802 | 0.235 | 0.119 | 9 |
| CAR[0,+1] | +0.827% | +0.678% | +0.149 | 0.921 | 0.039 | 7 |
| CAR[0,+3] | +0.740% | +0.948% | -0.208 | 0.936 | 0.821 | 7 |
| CAR[-1,+3] | +0.923% | +1.953% | -1.030 | 0.754 | 0.763 | 7 |

## Robustness — exclude all 2022 events (crisis must not drive the result)

Banks +0.442%, Finance +0.323%,
diff +0.119 pp, t-test p=0.929,
across 4 hikes.

## Sanity checks

- **Control group (JKH, DIAL) on HIKE:** mean -0.050% (n=14) — should be ~0.
- **HOLD control is WEAK:** the 9 "HOLD" rows are artifacts (SDFR flat while SLFR moved), not genuine no-change meetings — treat HOLD as weak evidence only.
- **Events actually used:** 2012-02-03 excluded (before ASPI coverage begins 2012-02-23); 2022-04-08 held out as a case study (see below). Windows that span a market-closure gap (COVID 2020-03-19→05-12, crisis 2022-04-07→04-25) are dropped from that window's averages.
- **LFIN split adjustment:** LFIN prices before 2015-07-14 were back-adjusted ×0.5 to fix an unapplied 2-for-1 split.

## 2022-04-08 (+700bps) case study — the market was shut

The emergency +700bps hike fell inside the 2022-04-07 → 2022-04-25 closure, so the market
could not react until it reopened on 25 April. Abnormal CAR from the reopen (CAR[0,+1]):

company   group   car_pct  day0_date
    HNB   Banks 12.562249 2022-04-25
   COMB   Banks  6.456292 2022-04-25
   SAMP   Banks  2.034477 2022-04-25
   LOFC Finance  7.064989 2022-04-25
   LOLC Finance 19.355463 2022-04-25
   LFIN Finance  8.666814 2022-04-25
    JKH Control  6.898784 2022-04-25
   DIAL Control 10.767831 2022-04-25

Reported separately, not in the main averages (a displaced, confounded reaction).

## Caveats (read before quoting any number)

1. **Small samples.** Only 7 usable hike events; each cell mean rests on
   few events — n is reported in every cell and must be quoted with the mean.
2. **Observations within an event are correlated** (three banks react to the same hike), so the
   pooled t-test / Mann-Whitney p-values are optimistic; treat significance as indicative.
3. **Weak control** for HOLD (artifact rows, above).
4. **No tuning:** all four windows are reported; the result is whatever it is.

*Outputs: `car_by_group_and_direction.csv`, `event_level_cars.csv`, `car_bar_by_group_direction.png`.*
