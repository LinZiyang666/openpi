# Opus round 2 — direction statement (written 2026-10-01 23:4x CDT, before substantive analysis)

Read before writing this: astra round 1 (REPORT, PROPOSALS, DATA_ANALYSIS, HANDBACK, methods.py), fable round 1
(REPORT, PROPOSALS, DATA_ANALYSIS, FOLLOWUP, PREDICTION_L10, HANDBACK, tools listing), RESULTS_R9.md, and the
current contents of `explore_astra/round2/` (tools/*.py, logs) and `explore_fable/round2/` (tools/corrector.py,
tools/methods.py, out/corrector/*.csv) as of 23:37. I will re-check both round2 directories before freezing PROPOSALS.md.

## (a) Directions already covered or in progress by astra / fable (one line each)

Astra, round 1
1. Shadow-label residual corrector: per-task ridge/RFF head, half strength, motion channels only (the closed-loop corrector).
2. Whole-episode routing between pure policy and pure cache, chosen per task (now forbidden, rule 2).
3. Library-level "do not call" gate for GR00T Spatial-500.
4. Retrieval-metric refit on ten-step action labels.
5. Synthesis variants: top-1, medoid, success-only neighbours, top-k, kernel width, gripper majority, norm expansion (negative).
6. Randomized call-value moderators from the independent-coin arms; generic uncertainty gate (negative / sign-inconsistent).
7. Shadow-MSE proxy falsified by the look-every-5 cadence result.
8. Nearest-neighbour memory of extra shadow labels as a matched-label baseline.

Astra, round 2 (in progress)
9. Task-agnostic shared corrector head (no task id): cache-path / mixed CU+IP paths / temporal-history inputs / linear; gripper-sign correction at a .8 threshold.
10. Learned call gates fitted on inits 0–19, evaluated 20–29 (value / risk / low-motion / distance scores).
11. Physical audit of pure-cache failures with goal predicates (any progress, regression) on inits 20–29.
12. Wrist-only corrector head (cheap-perception distillation).

Fable, round 1
13. Per-task call budgets from a label-free task calibration (now forbidden).
14. Shadow-gap profiler; early-episode disagreement / step-0 retrieval statistics as failure predictors (negative, AUROC ≈ .5).
15. Synthesis re-weighting evaluator (negative).
16. DAgger student MLP replacing retrieval.
17. Library growth with shadow-labelled cache rows (closed loop null on π0.5 L10-50); per-task library choice (forbidden).
18. Corrector + per-task calls combination arms (forbidden direction; ran as context).
19. Explanation of the grown-library null via trajectory consistency (source switching, gripper toggles).
20. Ideas listed: per-task strength, gripper residual with hysteresis, extra blind block on dense libraries.

Fable, round 2 (in progress)
21. Corrector lab: single head with task one-hot vs per-task heads, gripper channel, phase breakdown by executed-gripper history.
22. Grasp-miss-triggered policy calls (closed command held + finger aperture collapsed ⇒ forced call, ≤ 2 per episode) on top of the corrected cache, optional uniform coin.

Existing system mechanisms from earlier rounds (not theirs, but not new levers either): the no-progress guard and the other
B guards, uniform calls + stall trigger, stage-tilted calls, independent call coin, oracle grasp-window calls, look less /
follow lottery, wrist in easy stages, look every 5, CLIP key, R3 "burst 2" call continuation.

## (b) My directions, and why they do not overlap

I avoid every lever above: no shadow-label correction of any kind, no library growth, no metric refit, no synthesis
re-weighting, no per-task anything, no early-failure prediction from disagreement, no learned call-value gate, no
grasp-miss trigger. Everything below is task-agnostic (no task id enters any rule) and fitted/evaluated on disjoint inits.

**O1. Replicate-pooled outcome decomposition (diagnostic tool, new kind of evidence).** Everyone so far reads single
closed-loop runs. There are many replicate runs of the *same* controllers on the *same* (task, init) pairs across rounds
and fleets (pure cache, pure policy, the corrector twice, B/no-progress-only, uniform calls). Pooling them (inits 0–29 only)
gives a per-pair success probability under each controller. Questions: what is the run-to-run churn of pure cache against
itself (the null for "+38/−34")? Is the long-task cache deficit made of pairs that always fail (structural: coverage /
mode errors) or of knife-edge pairs that flip between replicates (robustness)? Does the corrector move structural pairs
on Spatial but only reshuffle knife-edge pairs on LIBERO-10? This directly answers "why the corrector works on Spatial
and not on LIBERO-10" with a mechanism-free, outcome-level test, and tells which lever class can work at all.

**O2. Time / pace anatomy of long-task failures (diagnostic).** From the debug collection's goal predicates (pure cache,
pure policy, call arms; inits 0–29): when each sub-goal first becomes true, how the cache's pace compares with the
policy's and with its own library demos, how much time is left when the cache stalls, and whether cache failures are
"slow but on track", "stuck in a loop", or "irrecoverable". Spatial has one sub-goal; LIBERO-10 has two, so pace and
time budget are a candidate L10-specific mechanism nobody has measured. (Astra's audit counts progress/regression at the
end; I measure timing, stall structure and remaining budget, which determine whether any recovery lever can work.)

**O3. Control-authority persistence ("escalation with hysteresis").** Lever = *how long* the policy keeps control once a
task-agnostic trouble signal fires, not *where* single calls go (items 6, 10, 13, 22, CU/CT/IP). B fires single calls
and immediately returns to the cache, which can drive back into the same stall. Candidate: once the existing
library-progress stall signal (or a pace deficit against the retrieved demos' own timing) fires, hand control to the
policy for a long window or until library progress resumes for k looks. Offline tools: a re-trigger / hand-back analysis
of B, only-no-progress and uniform-call arms (do single calls stick on L10?), and an exact-prefix cost/SR bound
(the arm is identical to pure cache until the first trigger, so the pre-trigger part of SR and IR is exact; only the
rescue rate after the trigger needs closed loop). R3's "burst 2" is the closest precedent (2 calls, old system,
no gain); I will treat it as prior evidence, not ignore it.

**O4. Phase-coherent retrieval (sequential tracking of the library's progress coordinate).** Lever = a temporal prior
across looks (an HMM-like filter over each library demo's progress), so that retrieval cannot jump back to an
already-finished sub-goal or skip ahead. Items 4, 5, 8, 15 change how *one* look is scored or averaged; none
constrains consecutive looks. Motivation: multi-step aliasing is specific to LIBERO-10, and R8's CLIP failure was
explained by back-jumps (16.4% vs 8.6%). Test first: are pure-cache back-/forward-jumps of the top neighbours'
progress more frequent on LIBERO-10 than Spatial, and do they precede failures (inits 0–19 to fit any threshold,
20–29 to check)?

**O5. Policy-free recovery (retry primitive), conditional on O1/O2.** If long-task failures are mostly knife-edge or
stalls with time left, a zero-inference retry — on a stall signal, back off and re-approach by retrieving from an
earlier library phase while excluding the demos used in the failed attempt — could rescue a fraction at pure-cache
IR. No one proposes recovery without the policy. Pursued only if O1/O2 show the precondition (time left + non-structural
failures).

Order of work: O1 and O2 first (they decide whether O3–O5 can work and explain the corrector result), then the one or
two levers the data support, built as tested serving classes with frozen arms on inits 20–29 (anything fitted uses inits
0–19 only; most of these rules have no fitted parameters beyond library statistics). Negative results will be reported.
