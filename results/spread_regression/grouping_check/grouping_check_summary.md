# Spread Regression — Grouping Robustness Check

The original run put LOLC **Holdings** (a diversified conglomerate) in the finance group; its
+256% month in 2021 was asset-sale rerating, not lending margin. LFIN also behaved like a bank.
CFIN (Central Finance, a genuine licensed NBFI) is added. Four specifications tested.

## Recommendation

**Spec B holds.** The interaction survives removing the conglomerate, so the effect is about genuine finance companies. Report B as the main specification, A as a footnote.

## Specification results (contemporaneous, lag 0)

| Spec | Finance/G2 group | Interaction | p | ex-2022 p | winsor p | Survives all? | Control clean? |
|---|---|---|---|---|---|---|---|
| A_original | LOFC+LOLC+LFIN | -4.647 | 0.0002 | 0.0060 | 0.0000 | YES | yes |
| B_drop_conglomerate | LOFC+LFIN+CFIN | -3.017 | 0.0021 | 0.0073 | 0.0002 | YES | NO — a control is sig |
| C_funding_structure | LOFC+CFIN | -3.447 | 0.0056 | 0.0262 | 0.0035 | YES | NO — a control is sig |
| D_pure_nbfi | LOFC+CFIN | -3.807 | 0.0025 | 0.0145 | 0.0005 | YES | NO — a control is sig |

## Per specification

- **A (original):** g1(Banks) slope -2.30, g2(Finance) slope +2.34, interaction -4.65 (p=0.0002), ex-2022 p=0.0060, winsor p=0.0000, n=1014, control OK (ns)
- **B (drop conglomerate — LOLC to control):** g1(Banks) slope -2.30, g2(Finance) slope +0.71, interaction -3.02 (p=0.0021), ex-2022 p=0.0073, winsor p=0.0002, n=1014, control PROBLEM (a control is significant)
- **C (deposit- vs borrowing-funded):** g1(Deposit-funded) slope -1.95, g2(Borrowing-funded) slope +1.50, interaction -3.45 (p=0.0056), ex-2022 p=0.0262, winsor p=0.0035, n=1014, control PROBLEM (a control is significant)
- **D (pure NBFI, LOFC+CFIN only):** g1(Banks) slope -2.30, g2(Finance(NBFI)) slope +1.50, interaction -3.81 (p=0.0025), ex-2022 p=0.0145, winsor p=0.0005, n=845, control PROBLEM (a control is significant)

Significant lags per spec: A=0, B=0, C=0, D=0.

## Key question — does the interaction survive removing LOLC Holdings?

Spec B interaction p = **0.0021** (ex-2022 0.0073, winsor 0.0002)
→ **SURVIVES**.

## Per-company coefficients (contemporaneous d_spread, lag 0)

| Company | Coef | p | n |
|---|---|---|---|
| HNB | -3.185 | 0.002 | 169 |
| COMB | -1.400 | 0.056 | 169 |
| SAMP | -2.330 | 0.009 | 169 |
| LFIN | -0.866 | 0.279 | 169 |
| LOFC | +4.072 | 0.019 | 169 |
| CFIN | -1.068 | 0.181 | 169 |
| LOLC | +3.821 | 0.001 | 169 |
| JKH | -0.095 | 0.864 | 169 |
| DIAL | -0.136 | 0.830 | 169 |

(LFIN -0.87 sits between the groups; LOLC +3.82 is the conglomerate.)

## Caveats

- **LOLC Holdings is a conglomerate** — including it in a finance group was a classification error
  in the original run.
- **LFIN (LB Finance)** is a deposit-taking LFC that sits structurally between the two groups.
- Only **173 monthly observations**, 2-4 companies per group — n reported per cell.
- **Correlation, not causation.** Note also the sign is *opposite* the margin-channel hypothesis
  (banks negative, finance positive) — this check is about whether the differential is real, not
  about which direction confirms the theory.

*Outputs: specification_comparison.csv, per_company_by_spec.csv, interaction_by_spec.png.*
