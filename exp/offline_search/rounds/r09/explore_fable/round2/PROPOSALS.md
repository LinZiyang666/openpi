# R9 round 2 — ranked proposals (fable), frozen candidate specs and exact arms

All candidates are task-agnostic at decision time (rule 2), fitted on inits 0–19 only and evaluated on inits 20–29
(rule 1). Serving code: `round2/tools/methods.py` (deployed on the h100 tree under this directory's mirror).
Screening results (100 pairs) are in `DATA_ANALYSIS.md` §3; this file freezes the specs for the coordinator's
300-pair and holdout runs.

## P1. Empty-grasp recovery call: `GraspMissCalls` (observation-keyed call policy)

**What.** Pure cache (optionally corrected, P2) plus one rule read from the robot state alone: when the executed
gripper command has been "close" for the last 2 decisions (10 controls) and the finger aperture is below the
empty-close threshold (fingers met nothing), force a policy call at the next anchor and the one after (2 calls,
20 controls with CU's policy-tail lifecycle); at most 2 triggers per episode; no other calls. IR cost ≈ the
trigger rate × 2 calls / ~65 decisions ≈ +.005–.02 over pure cache.

**Why.** It detects the dominant LIBERO-10 failure (grasp closes on nothing: 53% of π0.5 and 58% of GR00T cache
failures) with 92–100% sensitivity, 0–6% false alarms and 30–70 controls of lead on held-out inits; pure policy
rescues 86–93% of the cache's grasp misses when run from the start (R8 E3), and the no-progress guard, which fires
on the same situations much later, is worth +11 pp at IR .17 on both LIBERO-10-50 cells.

**Frozen spec.** `empty_aperture` = .0010 (π0.5) / .0009 (GR00T) = ½ × p1 of the holding aperture on calibration
rollouts (inits 0–19); `hold_decisions`=2, `burst`=2, `max_calls`=2, `p_uniform`=0, seed 26100202, domain
`R9F/r2/graspmiss`. Base = `CorrectedCache` (P2, blend 1.0 per-task head) or blend 0 (plain cache).

**Exact arms (screened).** `r9f2_{pi05_l10_50,groot_l10_50,pi05_spatial_50}_gm_corr1pt`, `r9f2_{pi05_l10_50,groot_l10_50}_gm_corr0`
(run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_r2`, manifest `manifests/eval100_inits20_29.json`).
**Extension to 300 pairs:** the coordinator may run the same artifacts on inits 0–29 only if the heads are refitted
on B-pool rollouts (the current heads use inits 0–19); the blend-0 variant (`gm_corr0`) has no fitted component and
can be run on all 300 discovery pairs immediately (`OSCL_MANIFEST` = discovery300).

**Decision rule.** Adopt if SR ≥ cache + 5 pp paired (McNemar) at IR ≤ cache + .03 on LIBERO-10-50; compare with
"A + only no_progress" (.832 @ .171 π0.5, .716 @ .216 GR00T, R8 abl on 500 pairs) as the existing trigger.

**Failure modes.** The policy may not recover within 20 controls (then a third trigger is capped); objects knocked
over are unrecoverable; the aperture threshold must be re-calibrated for grippers/objects whose holding aperture
is below 1 mm; false alarms cost .848 each.

**Screening status:** trigger arms pending at hand-back (chain running); see DATA_ANALYSIS §3.

## P2. Corrector at full strength, fitted on disjoint inits: `CorrectedCache(blend=1.0)`

**What.** The round-1 corrector (ridge + 384 random Fourier features on PCA keys, state, cached chunk, decision
index; motion channels only; gripper and 10-control commit unchanged) at **full** strength instead of half, with
the head fitted on inits 0–19 only (all R8 arms of the cell, 20–80k labelled decisions). Two variants: the existing
per-task heads, and a single task-agnostic head with the task one-hot as an input feature.

**Why.** Offline on held-out inits full strength removes 31–41% of the policy gap vs 22–28% at half; the head
generalizes across initial states (the owner's concern about round 1). IR unchanged.

**Frozen spec.** `head_path=round2/out/corrector/head_<cell>_motion_pertask.npz` (or `_motion_single.npz`),
`blend=1.0`, `correct_gripper=False`, base kwargs = A's (`lib current, kref 5, anchor_tail, budget 1, budget_only`),
base fits r5t_p_l10_50 / r5t_p_sp_50 / r5x_g_l10_50.

**Exact arms (screened).** `r9f2_{pi05_l10_50,groot_l10_50,pi05_spatial_50}_corr1pt`, `r9f2_pi05_l10_50_corr1single`.

**Screening result:** full strength is harmful closed loop on π0.5 L10-50 (per-task head .630, single head .540 vs cache .740). **P2 is withdrawn**; keep blend ≤ .5. Offline gap reduction does not license a stronger correction.

## P3. Prepared, not run: corrector + no-progress guard; corrector + uniform call coin

- `CorrectedCache` is a `BlindAWM` subclass, so the R8 `TriggerCommitJudge(disabled_guards=[stuck, terminal,
  overtime])` ("A + only no_progress") can take it as its base by swapping the fitted base in the judge artifact
  (10-line prefit script); expected ≈ onlynp + corrector gains, IR ≈ .17–.22.
- `GraspMissCalls(p_uniform=p)` adds an independent uniform coin (task-agnostic) on top of the trigger for the
  residual failures (place/fixture side); p = .15 would land at IR ≈ .15.

## P4. Not pursued (negative evidence this round)

- Gripper-channel correction: at failure onsets the cache's gripper command already agrees with the policy
  (disagreement .04–.09 vs .10–.17 elsewhere); failures are positioning, not timing.
- Half-strength corrector on LIBERO-10: fitted on 0–19 it still improves the held-out gap, but round 1 showed +1.4
  pp closed loop; full strength is the variant worth testing.
- Anything per task (rule 2).
