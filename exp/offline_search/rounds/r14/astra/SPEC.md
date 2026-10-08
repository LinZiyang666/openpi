# R14 implementation and experiment freeze

Frozen 2026-10-04T17:35:06.095796+00:00, before any closed-loop result of these new variants. The numerical payload is `design.json`, `allocation_fits.json`, `pilot_grid.csv`, `pilot_selection.json` and `test_grid.csv`. Exact floating-point values come from JSON, not rounded tables. This directory is an offline reference; sol owns the production adapter and fitting implementation.

## Part A switches and task identity

Expose independent `scope`, `metric`, `head`, `continuity`, and `guard` switches, each `task|pool`, on top of R10Recipe. Defaults must reproduce the original artifact/actions/decisions. Pool only within the same model, suite and nested B library. The input episode subset, cameras, action horizon, fixed hyperparameters and policy remain unchanged. Task ID is known from the original LIBERO instruction and must never be inferred from a retrieved donor.

| arm | scope | metric | head | continuity | guard |
| --- | --- | --- | --- | --- | --- |
| base | task | task | task | task | task |
| scope | pool | task | task | task | task |
| metric | task | pool | task | task | task |
| head | task | task | pool | task | task |
| scope_metric | pool | pool | task | task | task |
| scope_only | task | pool | pool | pool | pool |
| all_pool | pool | pool | pool | pool | pool |
| continuity | task | task | task | pool | task |

`guard=pool` by itself is an exact behavioral null under the active recipe and is checked on CPU, not a separate test arm. `scope_only` means only the retrieval boundary remains task-specific among the learned behavioral components. `all_pool` shares all of them. Both still use task identity for correct instruction delivery, logging, episode/anchor lifecycle checks and inherited periodic coins if a knob is separately enabled. Part A runs with layer four off. No pooled lower layers are combined with task allocation in the primary grid.

1. **Scope.** `task` retains only donors of the query task. `pool` exposes all rows of the selected library. With task metrics, transform every candidate using the query task's mean/std/main/early transform. Never compare candidates embedded by different task metrics. The median distance, neighbor ordering and kernel reference use the actual allowed candidate set. Use the inherited deterministic row-ID tie policy, donor-to-parent mapping, full action synthesis, corrector-anchor mapping and invalid-tail checks. Pooling scope alone still uses the query's task-specific head and continuity scale.
2. **Metric.** `pool` fits one joint geometry on every selected row: standardize joint PCA/state features; choose the inherited action-similar nearest rows from other episodes, now allowing other tasks; fit the original regularized within-pair covariance and full-rank Cholesky metric. Fit the early metric independently on the inherited early-step subset. Apply one shared transform to every task. The diagonal standardization and early affine-code representation are part of this switch. This does not pool `s_c`; continuity is a separate switch. `partition.py:metric_fit` is a bounded-memory reference to the same main fit equation, and its small exact-equation test is in `selfcheck.json`.
3. **Head.** For each library query, exclude the entire query episode from donor retrieval, use the selected scope/metric/continuity, and synthesize the uncorrected cache proposal with the inherited step-zero/fresh/stale rules. Default fitting uses the original R10 training query convention, with later anchors in the stale regime after a valid policy tail. Build the original PCA/state/cache-action/time features. Task heads preserve the existing task one-hot feature. The pooled head drops those task columns entirely and fits one R10 weighted ridge/RFF head on all tasks. Preserve the original random-feature count/seed, regularizer, action scaling, episode-balanced row weights, blend, unchanged gripper and correction horizon. Use every training row for production, not the economical anchor sample. Do not fit a separate task intercept, scale, embedding or calibrator in the pooled head.
4. **Continuity.** `task` retains each `T.s_c`. `pool` computes the inherited median of nearest normalized tail-to-head action distances across the whole library, excluding each row's own episode, with the inherited epsilon, and aliases that scalar for every query task. This acts only in the fresh branch. It must not alter ordinary stale/early geometry or periodic coins. `continuity.py` supplies its exact frozen library scalar per cell in `continuity_summary.json`; the builder must recompute and check it, not depend on explorer code at serving time.
5. **Guard and inactive auxiliaries.** Pool raw-camera centres over all selected rows, recompute the shared consecutive-frame cosine percentile with those centres, and replace each task median episode length by the median over all selected episodes. Motion threshold is already shared. Alias global versions of inactive blind state scales/motion percentiles and confidence-only state-distance/visual-centering references when producing the fully pooled artifact. Preserve no-progress constants, per-row normalized progress, each donor episode's length, the progress memo, and task/episode identity checks. Do not pool row progress into a global time axis. Disabled stuck/terminal/overtime bits, no events, no burst, no recovery/blending, budget-only cache tails, and lifecycle-only policy tails are invariant. Diagnostics/confidence may change; actions and active verdicts must not change for a guard-only toggle.

**Shared fitted components and descendants.** Reuse the same fresh selected-library PCA for every arm of a cell, once per camera. Action sigma and kref remain cell-wide and byte-identical to the baseline. When scope or metric changes, regenerate its LOEO training cache proposals, refit heads with the same head-sharing choice, and recalibrate the single GC_dist scale with the original complete episode-fold schedule using that scope/metric. This is a structural component ablation: dependent fitted values are regenerated, not left mismatched to their geometry. A head-only or continuity-only switch keeps the same ordinary GC_dist scale. Preserve the original attenuation equation, no step-zero correction and motion-only correction. Confidence fits may be regenerated as descendants but must remain excluded from `guard_only` acceptance. Record every source SHA, subset row SHA, scope and fit fold in the artifact manifest.

Inherited numerical constants are fixed: camera projection dimension 64 each; metric action neighbors 3, covariance ridge factor .1 and full rank; early metric rows have step at most 2; retrieval has 16 neighbors; kref is 5 for the size-50 cells and 8 for the other cells; ridge heads have 384 random Fourier features, alpha 100 and seed 0. The pooled head has 207 input features after dropping the 10 task columns, versus 217 for a task head. Predict the first 10 controls in 6 motion channels; keep the gripper untouched. The effective correction multiplier is `.5 * clip((2 - distance/scale)/1.25, 0, 1)` and is zero at step zero. At serving time distance is the minimum geometric distance among the actual proposal anchor donors, preserving the original fresh-branch nuance; do not substitute normalized kernel scores.

For production GC_dist calibration, assign folds by position in each task's frozen nested episode list exactly as R10 fitting does. Each distance query excludes its heldout episode fold from metric training and donors; pool eligible tasks only according to the switch. Keep the fixed selected-library PCA, fit training-only action sigma, retrieve using the early/main rule, omit step-zero distances from the episode-balanced median and fit one scalar. This is the full calibration, unlike the reduced inner split in the exploratory head diagnostic. At serving time both head and gate must reuse the cached proposal's actual donor rows, including foreign-task donors under pooled scope. Do not silently rerun task-restricted distance lookup inside the corrector.

## Part B: task_periodic on the unchanged recipe

Baseline is the existing R11 periodic controller and its original library-derived cap K. `allocation_fits.json` stores the exact recomputed reference K for each cell/dose. New task-periodic changes only which fixed K the periodic controller uses. No distance feedback, extra model, extra look, per-state action-gap predictor, shared mutable bank or cross-episode adaptation is introduced.

From the SHA-verified R11 whole-episode-held-out library error/guard tables, compute one priority per task:

```
eligible = real library anchor and not mandatory guard
episode_value = log1p(mean(eligible corrected motion MSE / action_sigma**2))
r[t] = mean(episode_value for task t)
n[t] = number of task episodes with at least one eligible finite error
mu = mean(r over represented tasks)
s = max(std(r over represented tasks), 0.1)
z[t] = n[t]/(n[t] + prior_episodes) * clip((r[t] - mu)/s, -2, 2)
z[t] = 0 when a task has no finite evidence
K[t] = max(expm1(log1p(K_reference) + bias - gain*z[t]), 0)
```

The table already uses normalized corrected motion error; do not normalize it twice. `allocation.py:libpack` squares the inherited RMS risk once to obtain MSE. Freeze gain `0.35` and prior episode count `10` globally. Standardize before shrinkage so equal small sample sizes do not cancel the shrinkage. There is no task-SR input or success gate. All final priorities use the complete cross-fitted library; size-full-library uses the identical rule.

Calibrate only one shared `bias` per cell/dose. Use complete, accepted off/periodic episodes from the explicit B roots listed in each fit. Each fixed path includes actual real-look steps, observed mandatory guards and terminal decision count. Replay exact inherited keyed coins with actual original task/init IDs. Seek the smallest worst absolute difference between task-periodic and the cell-wide periodic IR over these paths. IR is always `sum(cv*V + cm*M)/sum(N)`, never a mean of task or episode IRs; `(cv,cm)` is `(.152,.848)` for pi05 and `(.148,.852)` for GR00T. `allocation.py:solve` bisects bias in `[-8,8]` for 32 iterations. Move the lower bound to the midpoint when the midpoint of the largest/smallest signed IR differences is positive, otherwise move the upper bound. Among visited candidates choose `(maximum absolute difference, absolute bias, signed bias)` lexicographically. Discrete keyed schedules make this a numerical search, not a claim of an exact smooth optimum. Use all allowed development episodes for the final refit. Keep sparse identity-holdout failures separate. If no B path exists, use verified library episodes and certified original B init identities. No fit or scalar update may use A, a pilot outcome, or the reserved R13 identities before release.

| cell | dose | reference K | shared bias | task K min | task K max |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 0.32 | 1.0657 | 0.005844 | 0.645394 | 1.364013 |
| pi05_l10_50 | 0.4 | 0.4386 | -0.003906 | 0.134771 | 0.630377 |
| pi05_l10_200 | 0.32 | 0.9305 | 0.009394 | 0.585670 | 1.249579 |
| pi05_l10_200 | 0.4 | 0.4001 | -0.005429 | 0.133090 | 0.607507 |
| pi05_l10_500 | 0.32 | 0.9459 | 0.013889 | 0.550265 | 1.279528 |
| pi05_l10_500 | 0.4 | 0.4057 | 0.001404 | 0.106001 | 0.626278 |
| pi05_spatial_50 | 0.32 | 0.8707 | 0.015608 | 0.615336 | 1.185541 |
| pi05_spatial_50 | 0.4 | 0.3931 | -0.015830 | 0.165703 | 0.577189 |
| groot_l10_50 | 0.32 | 1.0252 | 0.010254 | 0.620264 | 1.265402 |
| groot_l10_50 | 0.4 | 0.4512 | 0.006772 | 0.156999 | 0.617680 |
| groot_l10_200 | 0.32 | 1.0069 | 0.033203 | 0.625688 | 1.391922 |
| groot_l10_200 | 0.4 | 0.4385 | 0.005796 | 0.133753 | 0.668125 |
| groot_l10_500 | 0.32 | 1.0011 | 0.033617 | 0.494282 | 1.473205 |
| groot_l10_500 | 0.4 | 0.4338 | 0.002004 | 0.037345 | 0.716923 |
| groot_spatial_50 | 0.32 | 0.8576 | 0.008098 | 0.674247 | 1.098323 |
| groot_spatial_50 | 0.4 | 0.3931 | -0.000443 | 0.244917 | 0.560246 |

At each real anchor call the lower-layer query once, preserve its cache proposal and guard memo, then take `K[query.task_id]`. Extract only accepted histories with existing `history(q)`. Compute `cap=floor(K)+I[uniform('R11-gap-v1',0,task,init,last_call_step,'cap') < frac(K)]`. Count real-look HITs since the last accepted policy call; call iff guard OR `run >= cap`. A mandatory guard resets the gap exactly like any call and keeps its original reason. A knob-only MISS gets a new collision-audited reason and the ordinary policy chunk/tail lifecycle. Do not change the existing coin seed, use a new coin key, index by donor task, charge a second call, or treat a proposal as acceptance.

The CPU reference is `controller.py`. Same-step retries return the same proposal; reset clears per-episode memo, decreasing steps and malformed histories fail through the ordinary contract fallback. Do not persist state across task, episode, connection, host or replicate. `off` delegates directly to the unchanged recipe; zero gain and zero bias reproduce cell-wide periodic exactly. A task cap of zero means always-call at real looks, not pure C5; long finite caps may yield no knob calls in a short episode while leaving guards intact. Do not clip away mandatory calls or silently normalize episode budgets at termination. A changed success rate changes episode lengths, so a fixed offline cap vector cannot guarantee aggregate live IR without the pilot gate.

Log the original accepted N/V/M, task ID, target, reference K, task K, cap, coin, guard/knob flags, priority, gain, bias, fit SHA and fresh/stale branch. Include actual CPU retrieval/correction latency, parent donor task IDs, corrector norm/gate, active guard reason, and lifecycle fallback counts. Diagnostics must survive the plugin scalar-field limit; use a versioned side record if necessary. Shadow-enabled B arms retain the corrected discarded cache proposal on misses. Never add shadow calls in A.

## B pilot and release gates

Run `pilot_grid.csv`: 68 arms / 3400 episode runs, paired existing B identities from `pilot_selection.json`. Each cell uses the specified same pairs across all methods/replicates. Small cells exercise every A switch and both B doses plus the lower-dose repeat; middle cells exercise baseline/all-pool and both B doses. The list is disjoint from each actual served subset and uses only previously consumed B identities, including cross-model exclusion verification for the middle GR00T cell. These are development implementation checks, not untouched validation. No new identity is selected from or against the reserved R13 list. All pilot runs wait behind the coordinator's R13 scheduling decision.

Before any live run, require default off equivalence, CPU guard-null equivalence, complete parent-row support for pooled donors, query-task metric arithmetic, episode/donor exclusion in fits, step-zero behavior, both fresh and stale branches, accepted-history reconciliation, same-step idempotence, independent reset/clone behavior, deterministic reordering of independent connections, and exact policy-tail/invalid-tail/early-termination handling in the real plugin. The offline reference tests do not replace these checks. The continuity arm must engage in an injected fresh fallback and be identical in nominal stale/early fixtures.

For each B arm require absolute realized total IR error at most `0.02`; each paired task-periodic/reference comparison must additionally differ by at most `0.01`. Report episode-cluster bootstrap intervals, mandatory-guard floor and actual terminal cost. Do not pass a failing cell by averaging across tasks/cells/doses. Guards take precedence when an arm is infeasible. The target band applies to the entire paired cell arm, not separately to tasks, whose rates are intentionally unequal.

Allocation engagement requires both (a) at least `0.05` symmetric-difference/union change in non-guard call positions against the unchanged periodic reference replayed on the allocation arm's own recorded states, and (b) at least `0.02` range across tasks in **allocation-minus-reference** knob-call fraction on those same states. This removes rate differences that existed before allocation. Count every task with observations, flag missing tasks, and report actual task N/V/M descriptively. For A scope/metric/head, require structural activation in CPU fixtures and report live donor/head changes; no minimum SR or minimum action difference is a release criterion. Natural non-engagement of a continuity switch is a valid null result, with fallback counts reported.

Mechanics failures permit implementation repair with frozen mathematics. Budget or allocation-engagement failures stop release of that affected primary cell/dose and are retained as failures; there is no hidden bias refit on pilot data, gain search, arm substitution or A-driven adjustment. Hand back for a separately versioned B-only redesign if necessary. A-only ablations are not blocked by a B allocation failure. Size-full-library can proceed only after the controller passes the smaller long-horizon cells for that model and its own fit passes mechanics; label it library-only transfer. The coordinator may elect to run a failed frozen arm as an explicit failure arm, never as validated matched-budget evidence.

## Economical frozen test grid

`test_grid.csv` contains 128 arms / 64,000 episode runs: 64 A arms and 64 B arms. All cells are pi05_l10_50, pi05_l10_200, pi05_l10_500, pi05_spatial_50, groot_l10_50, groot_l10_200, groot_l10_500, groot_spatial_50. Part A uses one replicate of each structural arm with layer four off. Part B uses the original lower layers, the two periodic methods, doses [0.32, 0.4], and 2 replicates. Every arm runs the full unchanged test manifest of 500 episodes; the explorer has not opened that manifest or any A data. Coordinator tooling binds the ordinary A contracts later.

Every GR00T arm and control is pinned to the local 4090; every pi05 arm and control is pinned to H100. Queue a cell if its host is unavailable. No cell/contrast migrates across hosts. Frozen group/arm order uses seed `20261014`; paired B methods are adjacent. Replicate rollout seeds differ; the inherited coin seed stays fixed. Verify that rollout seeds actually affect simulator/policy randomness; repeated metadata alone is not independent replication. Concurrent episodes are a throughput assumption, not an independence assumption. No additional pure-policy control is required for these contrasts; the A baseline is the newly paired recipe/off control, and B has its own newly paired periodic control. No previous A results are borrowed.

| host model | test episodes | runtime scenario days |
| --- | --- | --- |
| pi05 | 32000 | 1.35–2.15 |
| groot | 32000 | 3.11–4.75 |

These runtime ranges are scenarios from allowed B episode-duration medians/upper quantiles, divided by 16 concurrent clients, with explicit target/pooled-work uplifts in `freeze.py`. They do not measure H100 throughput, include fit time, guarantee scaling, or include queueing behind R13. With both hosts available, elapsed test time is the larger host duration; half the assumed concurrency doubles it. Pilot time, setup, failures and host capacity loss are additional. This grid saves a guard-null arm, omits the unused scope+head corner, limits A to one replicate, and does not repeat pure controls. It does not support a complete interaction factorial or precise per-cell SR claims.

## Frozen analysis

Always report per-cell SR and aggregate realized IR, their paired differences and per-task descriptions. For A, estimate scope/base, metric/base, head/base and continuity/base main contrasts, scope–metric difference-in-differences, scope-only/base and all-pool/base. The factorial interaction is conditional on task heads; head interactions beyond the observed combinations are unidentified. Guard pooling is a structural null, not an empirical claim with a made-up zero-error interval. A's natural guard/IR response belongs to the treatment effect and must be reported, not normalized away.

For B, estimate task-periodic minus cell-periodic SR at each frozen dose with equal total spend. A direct contrast is labelled matched only after both target and pair gates pass. If a live pair fails, retain its SR/IR results as unequal-budget results; do not call the SR difference a task-allocation benefit. Do not interpolate or extrapolate two noisy dose points to repair a failed budget gate. The intended full-grid matched estimand is unestablished if required cells lack valid matches; do not silently average only successful cells.

Use equal weight to model×suite, then equal weight to represented library sizes within that group, then doses and B replicates. Cluster paired bootstrap resampling by original suite/task/init across models, sizes, doses and repeats, stratified by suite/task, with `10000` resamples. Use the same resampled identities for paired arms. Report replicate-block sensitivity for B; A has no estimate of between-run variability. Per-task effects and dose interactions are descriptive. Do not count every repeated episode run as independent. Apply Holm correction to the four pooled A main contrasts; other contrasts are prespecified descriptive estimates with intervals. A pooling claim of practical equivalence requires a lower SR-difference bound above `-0.02` and no material IR increase, not merely a nonsignificant p-value. Insufficient precision means inconclusive. No SR-based stopping, dose selection, best replicate, new arm or task-specific fit follows A.
