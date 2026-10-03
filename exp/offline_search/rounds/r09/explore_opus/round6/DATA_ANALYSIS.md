# Round 6 (opus) — script exhaustion: when the retrieved demo runs out, and whether a takeover then can help

Lane: SR through early detection of script exhaustion. Rules kept: inits 30-49 never read (every ledger through
`iter_jsonl_discovery`, init >= 30 dropped before decoding; holdout roots refused); thresholds fitted on inits 0-19;
inits 20-29 descriptive only; nothing task-indexed; nothing launched; CPUs 2-9,46-53.
Data: `tools/ledger6.py` = the round-5 ledgers plus the round-5 screen (`r09_opus_r5`) and every LIBERO-10/50 run with a
policy takeover (cache + persistent / 24-decision escalation, only-no-progress + escalation).

Signal: the top-1 retrieved row is the **last row of its demonstration** (`rows to end <= k`, k = 0 unless stated).
The episode ends on success, so reaching that row while still running means the script ran out without the goal.

## 1. When does the retrieved demo reach its end, and what happens next? (`tools/exhaust.py`, `out/exhaust.log`)
| model, controller (inits) | SR | failures that reach the last row | median decision | still running >= 6 after the end: share / SR | first guard call after the end (failures) | pace-lag trigger minus end | policy share of decisions after the end (failures) |
|---|---|---|---|---|---|---|---|
| π0.5 cache (0-19, 5 runs) | .72 | .68 | 54 | .19 / .03 | — | −2 | 0 |
| π0.5 only-no-progress + B (0-19, 5 runs) | .85 | .68 | 58 | .13 / .30 | +2 | −8 | .81 |
| π0.5 cache (20-29, 8 runs) | .72 | .59 | 48 | .18 / .07 | — | +4 | 0 |
| π0.5 corrector only (20-29, 2) | .81 | .57 | 76 | .12 / .17 | — | −24 | 0 |
| π0.5 only-no-progress (20-29, 2) | .78 | .84 | 50 | .20 / .12 | +2 | +5 | .84 |
| π0.5 cache + escalation (20-29, 3) | .79 | .77 | 52 | .20 / .18 | +5 | +2 | .46 |
| π0.5 only-no-progress + escalation (20-29) | .85 | .80 | 50 | .12 / .08 | +2 | +4 | .99 |
| **π0.5 leading stack (20-29, 3 runs)** | .90 | .67 | 47 | .10 / **.35** | **+2** | +8 | **.97** |
| GR00T cache (0-19, 5 runs) | .64 | .26 | 48 | .12 / .24 | — | +10 | 0 |
| GR00T only-no-progress + B (0-19, 5 runs) | .73 | .51 | 58 | .16 / .18 | +2 | −10 | .80 |
| GR00T cache (20-29, 8 runs) | .60 | .36 | 46 | .15 / .07 | — | +10 | 0 |
| GR00T corrector only (20-29, 2) | .76 | .35 | 66 | .12 / .26 | — | −4 | 0 |
| GR00T only-no-progress (20-29, 2) | .77 | .47 | 57 | .14 / .29 | +2 | −7 | .75 |
| GR00T cache + escalation (20-29, 3) | .68 | .52 | 48 | .19 / .16 | +6 | −8 | .41 |
| GR00T only-no-progress + escalation (20-29) | .73 | .48 | 76 | .15 / .13 | +2 | −5 | .95 |
| **GR00T leading stack (20-29, 3 runs)** | .88 | .58 | 46 | .08 / **.16** | **+2** | +4 | **.82** |

- **Successes vs failures.** About half of the successful episodes ever retrieve a demo's last row, and when they do
  they end almost immediately (median 0, 90th percentile 2-5 decisions later). Failures that get there stay there:
  the retrieved progress never moves again.
- **How much earlier than the pace-lag trigger?** In the stacks the demo end comes first (pace-lag trigger 4-8
  decisions later, median). In the fit judge runs and the corrector-only runs the pace-lag trigger usually comes
  first (median 4-24 decisions before the end). Across the runs without escalation it fires before the end in 33-65 %
  of the failures that reach the end, because many failures fall behind long before they run out of script.
- **Recovery after that point is poor whoever holds control.** 6 decisions after the end: pure cache 3-24 %,
  guard stacks 16-35 %, full policy takeover by escalation 8-18 %. It falls with the time left (fit
  only-no-progress + B, still running 4 decisions after the end):

  | first end at decision | <= 30 | 30-40 | 40-50 | 50-60 | 60-80 | > 80 |
  |---|---|---|---|---|---|---|
  | π0.5 recovery | .86 | .63 | .43 | .26 | .14 | .00 |
  | GR00T recovery | .75 | .59 | .25 | .17 | .15 | .07 |

- **Key finding: the leading stacks already react to exhaustion.** The no-progress guard cannot see progress at the
  demo's last row, so it calls 2 decisions after the end (median). From then on the policy holds 97 % (π0.5,
  escalation on top) / 82 % (GR00T) of the decisions. In the would-be takeover windows the stacks already call at
  99 % / ~92 % of fresh decisions (`out/predict6.log`).

## 2. The trigger (inits 0-19 only)
Rule stated before applying it: among k in {0,1,2} and dwell d in {0,2,4,6}, keep the pairs whose trigger fires in at
most 10 % of successful fit-judge episodes in both models; take the earliest one (smallest d, then k).

| k \ d | 0 | 2 | 4 | 6 |
|---|---|---|---|---|
| 0: fires in successes π0.5 / GR00T | .48 / .44 | **.08 / .08** | .05 / .04 | .04 / .03 |
| 0: fires in failures | .68 / .51 | .62 / .41 | .57 / .38 | .52 / .33 |
| 1 (successes) | .76 / .73 | .21 / .19 | .08 / .06 | .06 / .04 |
| 2 (successes) | .86 / .87 | .47 / .44 | .13 / .10 | .07 / .06 |

→ **k = 0, d = 2**. It fires at the same look as the guard's first call or later (median +2 decisions after it);
every earlier variant also fires in 44-87 % of successes. Takeover = policy at every fresh decision for 24 decisions
(round-2 window constant; the policy chunk also serves the next blind decision), then back to the stack; once per
episode. Verdicts the stack already forces are kept.

## 3. Arms (`/home/weiland/trace_runs/os_closed_loop/r09_opus_r6`, inits 20-29) and prediction (PREDICTION.md, 05:36 CDT)
`r9o6_pi05_l10_50_stack` (control), `_stack_X` (+ takeover, escalation kept), `_npcorr_X` (escalation replaced by the
takeover); `r9o6_groot_l10_50_stack` (control), `_stack_X`.
Replaying the trigger on the stacks' three screen runs: it fires in 14 % (π0.5) / 10 % (GR00T) of episodes, about half
of them successes finishing inside the window. It adds 0.2 / 1.0 policy calls per firing. **Prediction: SR and IR
indistinguishable from the control for `stack_X` in both models.** For `npcorr_X`: IR −3 % and SR 0 to −4 pp,
because escalation's rescues of episodes that fall behind before running out are lost.

## 4. Correctness
- `tests/test_round6.py` (10 pass): trigger after the dwell at the last row, bounded window, back to the stack, no
  re-trigger; no trigger when retrieval leaves the end; stack verdicts inside the window kept; reset across
  episodes; forced trigger; parameter validation; attribute-clash detection. Real fitted classes on fable's r3c
  kwargs equal fable's artifacts: judge burst 0, CorrectedCacheJ blend .5 (correction on the judge path), same
  library actions/steps, same guard constants, escalation present only where intended. Rows-to-end table equals the
  catalog and the judge's own successor table (asserted in `fit`).
- CPU plugin selftests (`out/selftest_summary.json`), all 11 PASS:
  - Takeover that can never fire vs control: **0 of 48 decisions differ** (both models).
  - Forced takeover at decision 2: 20/20 window decisions are policy calls (18 / 16 with reason 95; the rest are the
    stack's own verdicts, kept), each followed by its policy tail; 0 fresh decisions in the window left to the cache.
  - Production arms PASS (the 12-decision selftest episodes never reach a demo end, so no firing).

## 5. What this means
The failure is produced before the script ends, and the end is already detected within ~2 decisions. A
"script exhausted → takeover" rule cannot add SR to these stacks. The only remaining lever in this direction is to
predict the failure **before** the end — e.g. the pace-lag escalation, which in runs without escalation fires before the
end in 33-65 % of the failures that reach it. The time-left gradient above (≤ 30 → 75-86 % recovery, 40-50 → 25-43 %) says each decision
gained matters.

## 6. Screen results (coordinator's timan107 screen, inits 20-29; `tools/readout.py`, written after the screen)
| arm | SR @ IR | paired vs same-batch stack (arm only / stack only) | calls per episode | takeover |
|---|---|---|---|---|
| π0.5 stack | .860 @ .193 | | 3.89 no-progress + 4.31 escalation | — |
| π0.5 stack + takeover | .880 @ .181 | +5 / −3 | 3.44 + 3.65 | 13 triggers, 110 window decisions |
| π0.5 np + corrector, escalation replaced by takeover | .910 @ .158 | +7 / −2 | 5.38 no-progress + 0.02 takeover-only | 12 triggers, 88 window decisions |
| GR00T stack | .870 @ .190 | | 7.84 | — |
| GR00T stack + takeover | .850 @ .192 | +4 / −6 | 7.99 + 0.14 takeover-only | 15 triggers, 79 window decisions |
As predicted, the takeover is almost entirely redundant with the guard. GR00T added 0.14 calls per episode (predicted
~0.1) and π0.5 only 0.02; both SR changes are within run-to-run churn. The escalation-replaced arm came out better
than predicted (+5 pp SR, −18 % IR vs its control) but at +7/−2 pairs this is not distinguishable from noise; the
coordinator did not carry it forward.
