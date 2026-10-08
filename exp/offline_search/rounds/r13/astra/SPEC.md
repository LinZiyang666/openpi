# R13 layer-four implementation specification

Frozen 2026-10-04T16:40:30.172320+00:00, before accessing any closed-loop result of the new variants. `freeze.json`, `configs.json`, `test_grid.csv`, `pilot_grid.json`, `static_pilot_grid.json` and `predictions.csv` are the numerical payload. `controller.py` is a CPU reference, not a serving adapter. All new implementation belongs to sol; this directory launches nothing.

## Decision and scope

The primary comparison is unchanged **R11 guard-aware periodic** versus **distance_feedback**, a monotone geometric-distance threshold adjusted by the episode's committed cost ledger. Freeze feedback width at 1.0; no random floor, periodic cap, learned utility predictor, per-task parameter, or cross-episode learning is added. A B-recalibrated static distance threshold is a dev-only diagnostic. No combination arm is in this test grid.

The controller is a candidate target-hitting knob, not yet a demonstrated closed-loop one. Its selected width passed the cost band in 44/44 correlated B development replay checks, while static B recalibration passed 32/44. The untouched B pilot below is mandatory. Size-500 remains an explicit transfer test because it has no dev complement.

## Preserve the existing lower layers and lifecycle

Reuse `rounds/r11/knob/recipe.py:R11Knob` and its unchanged R10 base. Keep both cameras, inherited per-task Mahalanobis geometry, the original only-no-progress guard, distance-attenuated LOEO corrector, ten-control commitment, and the selected nested B subset. Never install the R12 guard or look-half behavior.

At each real vision anchor, call the existing base query once. Retain the corrected cache proposal and its policy-tail gate anchor; read the original guard verdict; compute the new distance/threshold; OR the knob with the guard. A mandatory guard keeps its original reason, flags and progress memo. A knob-only MISS uses one ordinary policy chunk: five controls now and the lifecycle-approved five-control policy tail, then fresh retrieval. Do not reset guard/corrector state or serve the discarded cache tail. The look following a valid HIT-marked policy tail normally uses the inherited stale metric. Include step zero with the inherited early metric. No extra look or shadow policy evaluation is needed by the controller.

`off` must remain action- and decision-equivalent to the existing R10 recipe. Preserve lifecycle fallback behavior. A fallback look/call enters the actual ledger, regardless of why it happened.

## Exact distance_feedback rule

The raw score `d` is the minimum geometric distance from `inner.base._dist(q)[7]`, over the current task's inherited retrieval candidates. Task partitioning here is an existing lower layer; the knob itself has no task lookup. Never use the returned Result scores, recorded `distances`, confidence, corrector ratio or neighbor rank as raw `d`.

At an anchor before the current decision, obtain **accepted** histories exactly as R11 `history(q)` does:

```
N = number of preceding committed five-control decision slots = q.step
V = number of preceding actual vision decisions
M = number of preceding actual policy calls
(cv, cm) = (.152, .848) for pi05; (.148, .852) for groot
D = (cv*V + cm*M - target_IR*N) / cm
u = clip((log1p(d) - lo) / (hi - lo), 0, 1)
theta = bias + D / width
knob_trigger = (u > theta)
miss = guard OR knob_trigger
```

Use float64 for the scalar arithmetic. `width=1.0` is shared across the whole experiment. `lo` and `hi` are cell-wide and shared across doses. `bias` is one scalar per cell and user target. The target is the requested total owner IR, not a knob-only call fraction. Mandatory guards, previous knob calls, blind slots, truncated episodes and actual lifecycle fallbacks all count in the ledger. No invented future tail credit is added; the real terminal ledger determines measured IR.

This is exactly a distance threshold at any fixed ledger: for `0 <= theta < 1`, call iff `d > exp(lo + theta*(hi-lo)) - 1`. If `theta < 0`, call on all finite distances; if `theta >= 1`, call on none. Do not clip `theta` or `D`; clipping would change the specified algorithm. Ties use strict `>` with no coin. A negative or nonfinite raw distance conservatively triggers a policy call and records the error. There is no random-call component.

Episode reset clears the proposal cache and ledger checkpoint. No deficit survives to another episode, connection, task or server. No persistent shared mutable state depends on arrival order. Repeating a proposal at the same decision step returns the exact cached proposal without charging any cost. A decreasing step without reset, inconsistent history length or nonmonotone counters is a contract error handled by the ordinary serving fallback. `controller.py:DistanceFeedback` and `selfcheck.py` specify these semantics; production history extraction must additionally verify correspondence to accepted plugin decisions.

Suggested knob-only reason is a newly allocated `69` (static diagnostic `70`), subject to sol's collision audit; reason numbering is not a parameter. Keep original guard reasons. Put audit fields before low-priority extras: actual raw distance, u, theta, D, target, width, bias, N/V/M, guard, knob-only call, nonfinite flag, and threshold saturation. Log threshold and score before the action is replaced by policy output. Include settings/source SHA in the fit manifest. Shadow-enabled dev recording must retain or exactly reconstruct the corrected cache proposal even on a MISS.

## B-only fitting, exact frozen values

`robust_study.py` is the authoritative fitting procedure, and `robust_fits.json` records its outputs. Reuse the scalar values in `configs.json` without rounded Markdown values. Serving should embed scalars in the ordinary fit pickle and import neither exploration code nor data paths.

| cell | calibration behavior paths | initialization |
| --- | --- | --- |
| pi05_l10_50 | r11_devknob_50:distance; r12_devlooks_50L:off; r12_devlooks_knob:periodic | full served bank, external B inits |
| pi05_l10_200 | r12_devlooks_200:off | full served bank, external B inits |
| pi05_l10_500 | library:heldout | library only |
| pi05_spatial_50 | r11_devknob_50:distance; r12_devlooks_50S:off | full served bank, external B inits |
| groot_l10_50 | r11_devknob_50:distance; r12_devlooks_50L:off; r12_devlooks_knob:periodic | full served bank, external B inits |
| groot_l10_200 | library:heldout | library only |
| groot_l10_500 | library:heldout | library only |
| groot_spatial_50 | r11_devknob_50:distance; r12_devlooks_50S:off | full served bank, external B inits |

For external B data, accept only complete B-labeled episodes disjoint from the actual served subset, reconcile journal attempts against decision logs and N/V/M, and verify look shard hashes. Within all behavior paths use the same identity split: the first eight SHA256 bytes of `R13-cal:task:init`, interpreted big-endian, are calibration iff even. This is an offline split; task/init never index a fitted serving parameter. `development_splits.csv` lists every membership. Width selection used the complementary development split, so that split is **not** an independent final validation set. The final pilot must exclude the entire identity union in `used_b_identities.json`.

For each available behavior path in a cell, use its calibration episodes. Compute log-distance quantiles on guard-eligible real looks at probabilities `linspace(0,1,1001)`; concatenate these equal-length quantile grids across paths; take its pooled tenth and ninetieth percentiles as lo/hi, flooring `hi-lo` at `.05`. This gives each behavior path equal influence on the scalar scale. These quantiles are scale fitting, not a task-dependent normalizer.

At each target and candidate width, replay each calibration path with its observed guard/vision/state sequence. Choose the single bias minimizing the largest absolute total IR error across those paths. `fit_bias_minimax` bisects `[-16,16]` for `37` iterations by whether the midpoint of the largest and smallest predicted IR exceeds the target; among visited candidates choose `(max_abs_error, abs(bias), bias)` lexicographically. It is a monotone fixed-state budget solver, not an environment simulation. The normalizer never sees policy error labels or SR.

When no suitable B dev trajectory exists, use R11's SHA-verified whole-episode-held-out library scores and guard replay. Existing selected-library PCA exposure and reduced fold donor banks remain limitations, as in R11. Split complete packed library episodes by column parity for this calibration/development check. No self-query/full-library state is presented as an external dev episode. This applies to GR00T L10-200 and both L10-500 cells in the current freeze; the GR00T size-200 live pilot checks this transfer. Size-500 has no such pilot.

Global width candidates were `.5, 1, 2, 4, 8`. Width `1.0` was selected from B development evidence for budget robustness with material distance engagement. This is an explicitly selected development result, not a preregistered validation success. Frozen scalars:

| cell | log-distance lo | log-distance hi | bias @ .25 / .32 / .40 / .45 |
| --- | --- | --- | --- |
| pi05_l10_50 | 2.049875 | 2.721843 | 2.208933 / 1.317134 / 0.756655 / 0.500000 |
| pi05_l10_200 | 1.836849 | 2.340101 | 1.437500 / 1.125000 / 0.710938 / 0.515625 |
| pi05_l10_500 | 1.797345 | 2.274967 | 2.140625 / 1.380859 / 0.777545 / 0.531250 |
| pi05_spatial_50 | 2.327692 | 3.120123 | 1.242188 / 0.960938 / 0.651145 / 0.375000 |
| groot_l10_50 | 2.121839 | 2.730634 | 2.726562 / 1.488450 / 0.655729 / 0.352096 |
| groot_l10_200 | 2.016212 | 2.523070 | 2.528076 / 1.515625 / 0.773438 / 0.476562 |
| groot_l10_500 | 1.949735 | 2.361493 | 2.099609 / 1.282042 / 0.671875 / 0.406250 |
| groot_spatial_50 | 2.377626 | 2.990797 | 1.099464 / 0.804688 / 0.449219 / 0.296875 |

Only these scalar layer-four parameters vary by cell/dose. No per-task thresholds, gains, budgets, success gates, episode-length predictions or task-specific rates are allowed. Predictions, test settings and the grid may not be refitted using A.

## Periodic reference and dev-only static ablation

Periodic is exactly R11's dithered gap cap: `cap=floor(K)+Bernoulli(frac(K))` from the unchanged keyed `uniform('R11-gap-v1',0,task,init,last_call_step,'cap')`. Count accepted real-look HITs since the last policy call; the guard resets this count just like any other call. On an eligible anchor call iff `run >= cap`. Keep the R11 keyed-coin behavior, including at episode start. `periodic_fits.json` is recomputed by the unchanged R11 library `ScheduleReplay.solve`, rounded to its existing four-decimal convention. Existing target settings reproduce R11, and the high-dose extension uses the same solver.

`distance_static_b` uses one fixed raw-distance threshold per cell/target and strict `d > threshold`, OR with guard. `static_study.py` bisects the observed calibration range for a minimax threshold across the same B calibration paths. Its exact thresholds are in `static_pilot_grid.json`. It runs only on B at `.32/.40`; it does not replace the primary feedback method based on pilot SR. The static experiment isolates whether better calibration alone fixes the budget problem.

## Dev pilot, pass/stop rules, and release

Run `pilot_grid.json`: 56 arms of `100` episodes, paired K=10/task B inits. It covers both primary methods at every frozen dose in each cell with a nonempty dev complement, plus a second `.32` replicate in the size-50 cells to estimate paired outcome variance and replicate dependence. `static_pilot_grid.json` adds 8 dev-only arms. The total is 6400 pilot episodes. Primary validation uses fresh identities outside **all** development identities used here and outside the served subset; repeats use the same fresh validation identities. Never relabel old K=10 inits as fresh validation. Freeze the actual selection list and artifact hashes before any pilot starts.

The exact fresh pair list is already supplied in `validation_selection.json`, frozen at 2026-10-04T16:51:27.582294+00:00. It uses common pairs across models and tested sizes within each suite, excludes the cross-model union of previously used B identities, and excludes every relevant served subset. `validation_selection.py` found a minimum of 12 available inits per L10 task and 30 per Spatial task, sufficient for the requested selection. Coordinator tooling must build its normal SHA-bound contracts from these lists and reverify the actual prefits; no run root was created here.

Use the same host as the corresponding A cell: all GR00T local 4090; all pi05 H100. Existing R13 frozen-method recordings may improve later diagnostics, but do not silently revise this numerical freeze. The optional requested size-200 calibration recordings are not prerequisites to building these frozen artifacts; current library initialization is explicit.

Pass mechanics first: exact off equivalence, accepted-ledger reconciliation, no double guard/knob call, correct policy-tail lifecycle, same-step idempotence, reset/clone isolation, deterministic output under reordered independent connections, and zero unknown/missing identities. Sol must check these in the real plugin, including invalid-tail fallback and early termination. The explorer CPU reference passed 4800 proposals, 4800 retries and 12 analytic-bound scenarios; that does not replace plugin tests.

For each primary pilot arm require absolute aggregate realized IR error at most `.02`, using `sum(cost)/sum(N)`, and report an episode-cluster bootstrap interval. The paired distance-periodic IR difference must be at most `.01` at each dose for a direct matched-budget label. No average across cells may conceal a failing arm.

Require distance engagement on its own recorded validation states: at least `.40` of non-guard looks with `0 <= theta < 1`, and at least `.20` changed selected-call positions in a same-trajectory replay replacing u with constant `.5`, measured as symmetric difference divided by union of non-guard selected positions. Report within-episode distance shuffling as a second diagnostic. These thresholds screen out a mostly score-insensitive rate controller; they do not establish beneficial placement. Report near-ceiling saturation explicitly. Exclude nonfinite fallback looks from nominal engagement and count them separately.

For the static ablation report the same cost gate; failing static arms stay as a calibration failure result. Do not keep lowering thresholds on validation until they pass.

If a mechanics gate fails, repair the implementation and repeat the affected check with identical frozen arithmetic. If a budget or engagement gate fails, hold the affected primary cell from claims of validated target control; record the failure, retain all results, and hand back for a separately versioned B-only redesign/freeze. Do not tune on the validation set or A. The coordinator may measure a failing frozen method as a **Q1 failure arm**, but must not call it a passed matched-budget comparison. There is no automatic switch to another width or method.

Size-500 can proceed only after the same controller passes on available smaller-library cells and mechanics are validated for its actual fit. It is labeled **library-only transfer**, never B-validated. Its A result either establishes or fails target control without any A-based adjustment.

## Frozen test grid and host/batch plan

All exact arm IDs, configs, pair groups, replicate seeds and fixed randomized order are in `test_grid.csv`. There are 256 arms / 128,000 episode runs: 192 periodic/distance arms, 32 off controls, 32 pure controls.

- Cells: each model's L10 at library sizes `50, 200, 500`, and Spatial at size `50`.
- Both primary methods at `.25, .32, .40, .45` in every cell. `.32/.40` each have four replicates; `.25/.45` each have two. Each arm has `500` A episodes using the full unchanged task/init manifest.
- Off is repeated once per cell per replicate block. Pure C10 and pure C5 are repeated per model/suite/replicate block and shared across library sizes only after action/lifecycle equivalence is verified.
- Pure C10 is the layer-four all-call endpoint: force every real look to policy, retain one five-control tail, resume a real look every ten controls. R11 periodic `K=0` is the reference implementation; do not approximate it by a calibrated target `.50`. Pure C5 is the contextual standard policy called every five controls, owner IR `1`; it is not on the layer-four C10 interpolation curve.
- Keep **all** GR00T arms and controls on the local 4090; **all** pi05 arms and controls on H100. Do not split a cell across hosts when capacity changes. Queue it instead. Paired methods are adjacent within a cell/dose/replicate group; their order and the groups' order are frozen by seed `20261004`. Repeat blocks use different rollout seeds but unchanged R11 knob coin seed `0`. Verify and log whether those rollout seeds actually seed policy/simulator randomness; a metadata label alone is not a new stochastic replicate.
- Complete replicate blocks in order. No SR-based stopping, extra dose selection, dropped cell, best-replicate selection or host migration after viewing A. The estimate from the brief's throughput is 3.7–6.4 test days, excluding the pilot and capacity loss. This larger plan budgets the requested pooled precision rather than treating many correlated arms as independent evidence.

## Budget-matched estimands and power

`PLAN.md` freezes the analysis. Primary contrast is distance minus periodic at realized IR `.32` and `.40`, with equal weight to model × suite, then equal weight to available library sizes within that group, doses and replicates. Keep all nominal-target contrasts and their actual IR differences visible. A direct same-target comparison is called matched only when the realized IR difference is within `.01`; being individually within `.02` is insufficient.

For the fixed realized-IR primary estimand, interpolate each method's measured SR-vs-IR curve linearly between adjacent realized-IR points, with off and pure C10 endpoints. Do not extrapolate, impose increasing SR, choose an x-value from SR outcomes, or interpret interpolated doses as newly selected deployments. Recompute interpolation in every paired bootstrap resample. If common support or the Q1 gate fails, say the intended matched comparison is not identified; do not quietly substitute another cell/dose. Report curve interpolation beside the direct paired measurements and its modeling limitation.

The primary grid contains 32,000 paired episode-run contrasts, but repeated task/init identities are dependent. Cluster by original suite/task/init across models, sizes, doses and replicates; stratify resampling by suite/task. Report both init-cluster intervals conditional on recorded batches and a replicate-block sensitivity analysis. No binomial test on the pooled count of all arm runs.

The B-based variance scenario gives a four-replicate MDE of 1.35 pp with independent replicate noise, 1.79 pp at replicate correlation `.25`, and 2.71 pp under perfect dependence. These use old unequal-budget B method discordance as a planning scenario, not a known variance of the new method. The duplicate B pilot measures this uncertainty. If it implies inadequate precision, record that before A; no finite replicate promise removes an init-specific variance floor. Individual cell arms remain much noisier than the pooled effect.

## Operating limits

This controller resets each episode. Even with correct accounting, finite episodes, behavior-induced score/guard shifts and guard-only infeasible episodes prevent an unconditional exact aggregate budget guarantee. In the no-guard nominal cadence, writing `q=(2*target-cv)/cm`, the pre-look debt updates as `D_next=D+call-q` and stays between `min(0,-width*bias-q)` and `max(0,width*(1-bias)+1-q)` once initialized at zero. Mandatory guards can break the upper bound. They cannot be suppressed to enforce a target.

Periodic has its inherited hard bound `ceil(K)` cache anchors. Feedback has no universal cache-gap bound in the presence of accumulated guard debt: a long burst of mandatory calls can be followed by a long cache interval while repaying cost. Maximum observed gaps and conditional no-guard bounds must be reported; do not advertise feedback as a periodic cap. Both have cheap scalar controller work, but extracting d must reuse the existing geometry without a second vision or policy pass. Sol should benchmark incremental CPU query time separately from overall policy latency.
