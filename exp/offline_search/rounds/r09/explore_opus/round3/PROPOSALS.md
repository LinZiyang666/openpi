# Opus round 3 — proposals and frozen screen

Overlap check (re-read 02:0x CDT): astra round 2b = learned detector + bounded takeover with a recovery-score exit
(`_latch`, `_burst1/3`, `_random3`); fable round 2c/3 = cache-side retreat after an empty grasp, sign-corrected
empty-grasp calls, stacks of guard + corrector + empty-grasp + my escalation. Nobody changes *the state from which the
policy takes over*. That is my new lever. All rules task-agnostic; constants and maps from inits 0–19; arms on inits
20–29 only; run root `/home/weiland/trace_runs/os_closed_loop/r09_opus_r3` (emitted, prefits published, CPU plugin
selftests PASS, local dry plan 65 files / 15.9 GiB; not launched).

## Part 1 — fixes to escalation (from the round-2 screen, `DATA_ANALYSIS.md` §1–2)

1. **Use the bounded 24-decision window, not persistence.** Same SR where it matters (π0.5 .80/.80, GR00T .69/.71) at
   30–35% lower IR (.118 vs .172; .143 vs .220).
2. **Do not stack escalation on the no-progress guard.** On the episodes where both runs fell behind pace it changes
   nothing (π0.5 .21 vs .14, GR00T .27 vs .35); the guard has already made ≈ 5 calls there. The headline +8 / −5 were
   run-to-run noise. This also predicts that fable's round-3 `np_corr05_esc` / `np_corr05_gm_esc` gain little from the
   escalation component.
3. **Expect nothing from escalation on 500-demo libraries** (11–17% of episodes trigger; ≤ +2 pp expected).
4. Rejected fixes: futility exit, earlier pace trigger, start-of-episode failure prediction, vetoing on-pace guard calls.

## Part 2 — new lever: take over from the start pose (`HomingEscalation`)

**Rule (frozen).** Pure cache until the round-2 trigger (lag ≥ 12 by decision 80). Then a scripted homing segment back
to the end-effector position recorded at decision 0 (gripper open; up 5 cm, across, down; rotation unchanged; 10-control
segments re-planned each fresh decision; stop at 3 cm or after 8 fresh decisions). Then:
- `_home_w24`: 24 decisions of policy calls (10-control commits with policy tail), then the cache.
- `_home_cache`: the cache directly (no policy; IR ≈ cache).

**Why.** π0.5 takeovers rescue .56–.67 of troubled episodes when the arm is nearer its start than the median vs .12–.32
when farther (AUROC .69–.77 on the round-2 screen; .68–.85 in R8 call arms on inits 0–19); no such dependence for the
pure cache alone or for GR00T. Homing gives every takeover the near-start configuration.

**Predicted (100 pairs, same-run controls):**

| arm | π0.5 long-50 | GR00T long-50 |
|---|---|---|
| `_cache` | .70–.76 @ .076 | .55–.62 @ .074 |
| `_esc_w24` (round-2 rule, re-run) | .76–.82 @ ≈ .12 | .66–.72 @ ≈ .14 |
| `_home_w24` (new) | **.78–.86 @ ≈ .13** | .64–.72 @ ≈ .15 (no gain expected) |
| `_home_cache` (new, no policy) | .70–.77 @ ≈ .077 | .55–.62 @ ≈ .075 |

Adoption bar: `_home_w24` ≥ `_esc_w24` + 4 pp paired on π0.5 at IR within .02, or `_home_cache` ≥ `_cache` + 4 pp
(homing alone as a zero-call lever). Read with the both-crossed cross-tab (`round2/tools/screen_analysis.py` +
`round3/tools/dissect.py`), not the headline alone. Confidence: medium-low (the proximity effect is observational and
could be confounded with how badly the episode went; homing costs ≈ 40–80 controls of a ≈ 260-control remaining budget).

**Failure modes.** Homing path collides with fixtures (straight line at start height + 5 cm); a dropped object lands
badly when the gripper opens (the gripper is usually already open at the trigger); time runs out for first-object
failures that need the whole task redone; for GR00T the time spent homing is pure loss.

## Exact screen (coordinator runs; manifest `manifests/eval_inits20_29.json`)

| batch | arms |
|---|---|
| 1 π0.5 long-50 | `r9o3_pi05_l10_50_cache r9o3_pi05_l10_50_esc_w24 r9o3_pi05_l10_50_home_w24 r9o3_pi05_l10_50_home_cache` (+ optional `r9o3_pi05_l10_P10`) |
| 2 GR00T long-50 | `r9o3_groot_l10_50_cache r9o3_groot_l10_50_esc_w24 r9o3_groot_l10_50_home_w24 r9o3_groot_l10_50_home_cache` (+ optional `r9o3_groot_l10_P10`) |

If only one batch fits, run batch 1 (the proximity effect is π0.5's). Commands in `HANDBACK.md`.

## Outcome (recorded in round 4)

Screen result (coordinator): π0.5 cache .710 / window .760 / homing+window .730 / homing only .640; GR00T .570 / .650 /
.590 / .550. Both homing arms failed their adoption bars; homing is withdrawn. The bounded escalation window
(lag 12, deadline 80, window 24) is kept and is stacked on the half-strength corrector in round 4.
