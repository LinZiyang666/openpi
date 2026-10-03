# Round 7 (opus) — guard-call budget alone on the leading stacks: screen results

Arms in `/home/weiland/trace_runs/os_closed_loop/r09_opus_r7` (inits 20-29): fable's r3c stacks (same-batch control),
and the same stacks with only the round-5 per-episode guard-call budget (gate P off): C = 20 (fitted on inits 0-19 in
round 5) and C = 15 (second dose point). Predictions in `PREDICTION.md` (05:41 CDT, before emit).
Results from the coordinator's timan107 screen, read with `tools/readout.py` (inits 20-29 only):

| arm | SR @ IR | IR change | paired vs stack (arm only / stack only) | calls per episode | budget-gated decisions per episode |
|---|---|---|---|---|---|
| π0.5 stack | .890 @ .188 | | | 3.45 no-progress + 4.13 escalation | — |
| π0.5 C20 | .900 @ .158 | **−16 %** | +6 / −5 | 3.40 + 1.96 | 0.96 |
| π0.5 C15 | .860 @ .148 | −21 % | +6 / −9 | 3.46 + 1.46 | 2.45 |
| GR00T stack | .840 @ .193 | | | 8.22 | — |
| GR00T C20 | .850 @ .175 | **−9 %** | +8 / −7 | 6.86 | 1.21 |
| GR00T C15 | .850 @ .163 | −15 % | +7 / −6 | 6.06 | 1.89 |

Against the prediction (C20: π0.5 IR −6…−10 %, SR 0…−3 pp; GR00T IR −7…−8 %, SR 0…−1 pp), both C20 arms kept SR
(+1 / +1 pp, within churn). The π0.5 IR cut was larger than predicted (−16 %) because this batch escalated a lot
(4.1 escalation calls per episode in the control) and the budget mostly trims long, failing escalations.
C15 lost 3 pp on π0.5 (+6 / −9), as predicted (−1…−5 pp), and nothing on GR00T. C20 is the safe dose; π0.5 C15
starts to eat into escalation rescues.
