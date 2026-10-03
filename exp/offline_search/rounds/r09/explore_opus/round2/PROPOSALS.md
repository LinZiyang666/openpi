# Opus round 2 — ranked proposals, frozen specs and exact arms

Written after re-reading `explore_astra/round2/` (shared motion/gripper correction heads, wrist-only head,
call-value gates rejected) and `explore_fable/round2/` (empty-grasp recovery bursts, long-carry trigger, full-strength
corrector withdrawn) at 00:1x CDT. My lever does not overlap theirs: **how long the policy keeps control once the
cache is off its demonstration's pace** (persistent escalation), triggered by a library-pace signal. All rules are
task-agnostic (no task id anywhere), all constants were chosen on inits 0–19, every requested arm evaluates on
**inits 20–29 only** (100 pairs, `manifests/eval_inits20_29.json`). Nothing here is fitted on evaluation inits; the
base controllers are the existing library-only prefits.

Run root (emitted, prefits published, local plan done, nothing launched):
`/home/weiland/trace_runs/os_closed_loop/r09_opus_escalation`. Serving code: `round2/methods.py`.

Ranking = expected gain × confidence. Predictions are the pre-registered simulator values (`DATA_ANALYSIS.md` §3,
`out/preregistration.json`) for EVAL inits; the one unknown is r, the success probability of a policy takeover started
at the trigger (prior .45–.65; the cache by itself recovers .17–.29 of these episodes).

---

## P1. Persistent escalation on the pure cache (all four LIBERO-10 cells)

**Rule (frozen).** Run the pure cache unchanged. At every fresh decision compute
`lag = decision index − library step of the top-1 retrieved row`. At the first fresh decision with
`lag ≥ 12` and decision index `≤ 80`, escalate: every later fresh decision is a policy call committed for 10 controls
(pure-policy pattern with the existing policy-tail lifecycle), until the episode ends. No other calls.
Class `EscalateCalls(lag_threshold=12, deadline=80)`; base = the frozen pure-cache prefit of the cell.

**Why.** (1) Long-task cache failures are blind script-following: after a missed sub-step the cache replays its
demonstration to the end on schedule and then idles; 91–93% of failing episodes reach the demo's end frames at the
same time as successes. (2) Pace lag is therefore a near-perfect, task-agnostic failure detector, but a late one:
FIT inits — π0.5 long-50 detects 95% of failures with 15% false alarms on successes; GR00T long-50 99% / 15%;
π0.5 long-500 73% / 4%; GR00T long-500 90% / 6% (median trigger 190–280 controls of 520). (3) Interventions that touch
every episode break 8–15% of the cache's always-successful long-task episodes (corrector 11–12%, uniform calls
8%, placebo perturbations 5–21%); escalation leaves on-track episodes untouched and spends policy calls only on the
≈ 15–45% of episodes that fall behind. (4) The existing no-progress guard cannot simply be made persistent: it fires
in 89–98% of always-successful episodes.

**Predicted (EVAL, r = .45 / .55 / .65; IR at r = .55):**

| cell | cache | escalation SR | escalation IR | existing points (inits 0–29) |
|---|---|---|---|---|
| π0.5 long-50 | .727 @ .076 | .79 / .83 / .87 | ≈ .16 | B .840 @ .179; uniform .880 @ .301; policy .912 @ .504 |
| GR00T long-50 | .604 @ .074 | .74 / .78 / .83 | ≈ .19 | B .733 @ .220; uniform .797 @ .303; policy .865 @ .504 |
| π0.5 long-500 | .833 @ .077 | .88 / .90 / .92 | ≈ .12 | B .893 @ .156; uniform .920 @ .187; policy .912 @ .504 |
| GR00T long-500 | .867 @ .075 | .89 / .91 / .92 | ≈ .12 | B .870 @ .189; only-no-progress .883 @ .178; policy .865 @ .504 |

Expected gain × confidence: **high on GR00T long-50 and both 500-demo cells** (at r ≥ .45 escalation beats B in SR at
lower IR; on the 500-demo cells it reaches pure-policy SR at ≈ ¼ of its cost and below the uniform-call frontier);
**medium on π0.5 long-50** (needs r ≈ .6 or more to beat B; at r = .55 it only matches B at lower IR).

**Exact arms.** `r9o_{pi05,groot}_l10_{50,500}_esc` vs same-topology controls `r9o_{pi05,groot}_l10_{50,500}_cache`;
optional pure-policy references `r9o_{pi05,groot}_l10_P10`.

**Screen readout (`tools/screen_analysis.py`).** SR, owner IR, paired McNemar vs `_cache`, escalation rate, median
escalation time, **measured r** (success among escalated episodes) and the control arm's self-recovery after the
same rule (both models; GR00T from server logs). Adoption bar per cell: r ≥ .50 and paired ΔSR vs cache ≥ +6 pp (50-demo) / +3 pp (500-demo) with
IR ≤ .20 / ≤ .13. Stronger claim (vs existing methods): SR ≥ B at IR ≤ B's IR (50-demo), SR ≥ policy − 2 pp at IR
≤ .13 (500-demo). 100 pairs cannot establish non-inferiority; that is the coordinator's holdout step.

**Failure modes.** r low because the takeover starts late (≈ 270 controls on π0.5 long-50) or from an
irrecoverable state (knocked objects); false alarms on slow successes (10–15% on 50-demo cells) cost IR and, if the
policy then fails, SR; the policy may need longer than the 520-control limit allows.

## P2. Bounded escalation window (cost variant of P1)

Same trigger; policy calls only during the 24 decisions (120 controls, 12 calls) after the trigger, then the cache
resumes; never re-armed. `EscalateCalls(..., window=24)`. Measures how much of the rescue completes within 120
controls (partial-policy recoveries finish a median 13–20 decisions after the trigger, q75 26–34).
Predicted (EVAL, success ≈ .7·r): π0.5 long-50 .74–.80 @ .130; GR00T long-50 .69–.75 @ .140.
Arms: `r9o_{pi05,groot}_l10_50_esc_w24` (same controls as P1). Adopt only if within 2 pp of P1 at ≥ .02 lower IR.

## P3. Escalation on top of the no-progress guard

The R8 "only no-progress" controller unchanged, plus the P1 rule as a forced policy call (os_reason 91).
`EscalateOnlyNP` / `EscalateOnlyNPGroot` (base = frozen R8 artifacts `r8abl_onlynp_{p,g}_l10_50.pkl`).
Expected gain depends on the base's own recovery after the trigger (π0.5 B ≈ .50, GR00T B ≈ .31–.36), escalation
adds `trigger rate × (r − that)`: π0.5 long-50 predicted .81–.87 @ .20–.21 vs B-only-no-progress .79 @ .19 on these
inits; GR00T long-50 .80–.87 @ .23–.25 vs .75 @ .21 (larger, because GR00T's single calls recover less).
Arms: `r9o_{pi05,groot}_l10_50_onlynp_esc` vs `r9o_{pi05,groot}_l10_50_onlynp`. Lower priority than P1/P2.

## P4. (After the screens; not emitted by me) stack the early and the late trouble signals

Fable's empty-grasp burst catches grasp misses early and cheaply (.790 @ .086 on π0.5 long-50, inits 20–29); its
triggered-late episodes still recover only .44. Applying my simulator to fable's own 20–29 trajectories predicts
.79–.86 @ .16–.18 for "empty-grasp burst + pace-lag escalation". If both screens are positive, a combined class
(`_assignment` = empty-grasp burst OR escalated) is ≈ 20 lines on fable's `GraspMissCalls`; constants stay frozen.

## P5. Implication for the corrector line (not my lever; flagged, not proposed as an arm)

The corrector's long-task null is a selectivity problem, not a fitting problem: it rescues 33–46% of structural
failures but breaks 11–12% of always-successful episodes, concentrated on perturbation-fragile pairs. Any
always-on change of the cache's action pays this. Round-2 closed loop agrees (fable full strength .63/.54, astra
shared heads .63 vs cache .72 on π0.5 long-50). If the correction line continues, gate it on being off-pace
(lag > 0 or after the P1 trigger) so on-track episodes stay byte-identical to the cache.

## Rejected (negative or unsupported this round)

- **Phase-coherent retrieval** (my O4): phase back-jumps and regressions occur after the script is exhausted
  (consequence, not cause). Not built.
- **Earlier pace triggers**: no lag / no-progress combination fires earlier than a median 250 controls on π0.5
  long-50 at ≤ 16% false alarms. The early signal is fable's empty grasp, not pace.
- **Escalation on Spatial**: the trigger arrives at ≈ 170 of 220 controls and the cache's own recovery after it is
  .01; too little time for any takeover. Not emitted.
- **Persistent version of the existing no-progress guard**: fires in 89–98% of always-successful episodes (≈ pure
  policy everywhere).
- **Pure time handoff ("not done by decision 55")**: equal to the lag rule on π0.5 long-50 in simulation, worse on
  500-demo cells (9–14% vs 3–5% false alarms) and later on GR00T.
- **Policy-free retry at script exhaustion** (my O5): not built; the cache recovers .17–.29 of triggered episodes
  on its own and no offline estimate of a forced rewind exists. Fable lists a cache-side open-and-retry for π0.5.

## Exact screen (coordinator runs; 100 pairs each, manifest tasks 0–9 × inits 20–29)

| batch | arms (in order) |
|---|---|
| 1 π0.5 long-50 | `r9o_pi05_l10_50_cache`, `r9o_pi05_l10_50_esc`, `r9o_pi05_l10_50_esc_w24`, `r9o_pi05_l10_50_onlynp`, `r9o_pi05_l10_50_onlynp_esc` (+ optional `r9o_pi05_l10_P10`) |
| 2 GR00T long-50 | `r9o_groot_l10_50_cache`, `r9o_groot_l10_50_esc`, `r9o_groot_l10_50_esc_w24`, `r9o_groot_l10_50_onlynp`, `r9o_groot_l10_50_onlynp_esc` (+ optional `r9o_groot_l10_P10`) |
| 3 500-demo long | `r9o_pi05_l10_500_cache`, `r9o_pi05_l10_500_esc`, `r9o_groot_l10_500_cache`, `r9o_groot_l10_500_esc` |

Commands, deployment of `methods.py` and the analysis call are in `HANDBACK.md`.

**Holdout protocol (coordinator only).** Freeze one candidate per cell from the screen (the P1 `_esc` artifacts as
emitted, or P2/P3 if they win), then run it with its same-topology cache and pure-policy controls on inits 30–49.
Do not re-tune lag / deadline / window on 20–29 before that; if they are changed, the change must be justified on
inits 0–19 data and frozen before the holdout.
