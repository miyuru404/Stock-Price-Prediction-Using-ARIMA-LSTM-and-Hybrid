#!/usr/bin/env python3
"""
Loughran-McDonald style finance sentiment lexicon.

WHY NOT VADER ALONE: VADER was tuned on social media. In financial reporting the polarity of many
words flips or vanishes — "liability", "provision", "exposure" and "volatile" are neutral technical
terms in a bank report but negative in everyday English, while "beat", "upgrade" and "surplus" are
strongly positive in finance and near-neutral in general text. Loughran & McDonald (2011) built
their word lists precisely because general-purpose lexicons misclassify financial documents.

WHAT THIS FILE IS: a COMPACT, hand-built lexicon in the LM style (~180 positive, ~230 negative
terms covering the vocabulary that actually appears in Sri Lankan market/economy reporting). It is
NOT the full LM dictionary (~4,000 terms).

TO USE THE REAL THING: download the official Loughran-McDonald master dictionary from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and save it as

    cleaned_data/loughran_mcdonald_master.csv

with columns `Word`, `Positive`, `Negative` (the standard format). load_lm() picks it up
automatically and the compact list is ignored. The results file records which one was used, so a
run is never ambiguous about its lexicon.
"""
from pathlib import Path
import re

import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
LM_PATH = _HERE / "cleaned_data" / "loughran_mcdonald_master.csv"

COMPACT_POSITIVE = """
achieve achieved achievement advance advanced advantage advantageous appreciate appreciated
appreciation attractive beat beats benefit benefited beneficial best better boom boost boosted
boosting breakthrough brisk buoyant certain collaborate confidence confident constructive
conducive delight delighted deliver delivered dividend durable ease eased easing efficient
efficiency encourage encouraged encouraging enhance enhanced enhancement enjoy enthusiasm
excellent exceed exceeded exceeding exceptional expand expanded expanding expansion favourable
favorable gain gained gaining good grew grow growing growth healthy high higher highest improve
improved improvement improving incentive increase increased increasing inflow innovate innovation
lead leading led lucrative optimism optimistic opportunity outperform outperformed outstanding
pickup pleased plentiful positive premier profit profitable profitability progress prosperity
rally rallied rebound rebounded recover recovered recovery reward rewarding rise risen rising
robust rose satisfactory soar soared solid sound stabilise stabilised stabilize stabilized stable
stability strength strengthen strengthened strong stronger strongest succeed success successful
supportive surge surged surplus thrive top transparent upbeat upgrade upgraded upside upturn
upward valuable versatile viable win winner winning
""".split()

COMPACT_NEGATIVE = """
adverse adversely against aggravate alarm alarming allegation arrears bad bankrupt bankruptcy
bearish below breach breached burden cancel cancelled caution cautious cease challenge challenging
collapse collapsed concern concerned concerns constrain constrained contraction correction crash
crashed crisis critical curtail curtailed cut cuts damage damaging danger decline declined
declining decrease decreased default defaulted deficit deflation delay delayed depreciate
depreciated depreciation depress depressed deteriorate deteriorated deterioration devalue
devaluation difficult difficulty dip dipped disappoint disappointing disappointment discontinue
dispute disrupt disrupted disruption distress downgrade downgraded downturn downward drag drop
dropped drought erode eroded erosion fail failed failure fall fallen falling fear fell forced
fraud freeze halt halted hamper hardship hazard hike hurdle hurt illiquid impair impaired
impairment inability inadequate inefficiency inflationary insolvency instability insufficient
interruption lack lag lagged lawsuit layoff liquidation litigation loss losses lost low lower
lowest mismanagement negative obstacle overdue panic penalty plunge plunged poor postpone
postponed pressure protest queue recession record-low reduce reduced reduction restrict
restricted restriction risk risky sanction scarcity setback severe shortage shrink shrank shut
shutdown slash slashed slow slowdown slowed sluggish slump slumped stagnant stagnation stall
stalled strike struggle struggling subdued suffer suffered suspend suspended threat tighten
tightening trouble turmoil unable uncertain uncertainty underperform undermine unemployment
unfavourable unfavorable unrest unstable volatile volatility vulnerable weak weaken weakened
weakness worse worsen worsened worst write-off writedown
""".split()

_TOKEN = re.compile(r"[a-z][a-z\-']+")


def load_lm():
    """Return (positive_set, negative_set, source_label)."""
    if LM_PATH.exists():
        df = pd.read_csv(LM_PATH)
        cols = {c.lower(): c for c in df.columns}
        w, p, n = cols.get("word"), cols.get("positive"), cols.get("negative")
        if w and p and n:
            words = df[w].astype(str).str.lower()
            pos = set(words[pd.to_numeric(df[p], errors="coerce").fillna(0) > 0])
            neg = set(words[pd.to_numeric(df[n], errors="coerce").fillna(0) > 0])
            if pos and neg:
                return pos, neg, f"Loughran-McDonald master ({len(pos)} pos / {len(neg)} neg)"
    return (set(COMPACT_POSITIVE), set(COMPACT_NEGATIVE),
            f"compact LM-style ({len(set(COMPACT_POSITIVE))} pos / {len(set(COMPACT_NEGATIVE))} neg)")


def lm_score(text, pos, neg):
    """Polarity in [-1, 1]: (pos - neg) / (pos + neg). 0.0 when no sentiment word is present."""
    toks = _TOKEN.findall(text.lower())
    p = sum(1 for t in toks if t in pos)
    n = sum(1 for t in toks if t in neg)
    return (p - n) / (p + n) if (p + n) else 0.0


if __name__ == "__main__":
    P, N, label = load_lm()
    print("lexicon:", label)
    for s in ["Bank profits surged and the rupee strengthened",
              "Shares plunged amid default fears and a deepening crisis",
              "The central bank held rates unchanged"]:
        print(f"  {lm_score(s, P, N):+.2f}  {s}")
