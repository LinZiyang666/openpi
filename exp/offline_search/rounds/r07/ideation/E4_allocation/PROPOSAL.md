# E4 — Allocation policy and the budget knob at stage level

## 1. Direction summary

**Recommendation: a small stage-mode table controlled by one cost price, learned by changing one stage at a time.**
Initialize at pure policy; reduce spending where measured terminal-success loss per saved IR is smallest.
Use uniform allocation while the stage ranking is unresolved. Do not manufacture a success-value score from library
action error: R6 already rejected that bridge. Two proposals below specify the allocator and its deployment calibration.

I read the R7 brief, R6 `ANALYSIS.md`, `frontier_final/{frontier_points.csv,outcomes.json}`, the task-routing code/results,
R5 Q3 `HANDBACK.md`, the R6 lottery implementation and budget solver, the stall interface, P3 `SCHEMA_V2.md`,
the closed-loop README/blind contract, R4 owner-cost table, and ledger §§5.1–5.3.
I inspected B-val table schemas, library arrays, and accepted journals/decision logs for all eight R6 uniform arms.
The analysis below audits **4,000 episodes, 163,316 decisions, and 82,534 vision anchors**.

Two surprises matter for the controller:

- “Uniform” uses an **episode-dose lottery**, followed by independent anchor coins. Some episodes have dose 1,
  so using the arm's average probability for every anchor would invent treatment support.
- A coarse stage allocator selected on other initial states does **not** reliably improve SR: all eight refitted
  bootstrap intervals cross zero. This is evidence against immediately shipping progress-based call priorities,
  not evidence that properly identified grasp/transport/place stages cannot help.

This is exploration, not a deployed fit. Existing R6 benchmark outcomes are used only for a diagnostic feasibility
study. None of its fitted priorities, phase boundaries, or candidate shifts is proposed as a deployment calibration.

## 2. Proposal 1 — One IR knob, a small stage-mode menu, one marginal price

### Hypothesis and the controller in a few lines

Hypothesis: useful stage heterogeneity, if it exists, is easiest to exploit through whole operating modes;
independently turning three knobs risks paying for looks without getting useful calls or interrupting a commitment.

1. Identify the current stage; list its admissible `(call probability, look cadence, cameras)` modes.
2. For target IR `rho`, choose each stage's mode by `value − lambda × budget_cost`; adjust one shared `lambda` to fit rho.
3. Spend upgrades in decreasing marginal terminal-SR gain per additional IR; randomize only at a boundary if needed.
4. On deviation/stall, return to the admissible recovery mode and charge its cost; update the table between episodes.

**Knob orientation is explicit:** minimum = cheapest admissible settings; maximum = the chosen pure-policy reference.
For this study the reference is pure L10, nominal IR `.5`; L5 would instead put the maximum at `1`.
The exposed knob is target IR itself. An optional UI maps `u in [0,1]` to
`rho = rho_min + u*(rho_pure-rho_min)`; `u=1` bypasses retrieval decisions and runs pure policy exactly.
`lambda` is internal and increases as the user lowers rho. It is not the exposed knob.

### Mechanism, stages, signals, hardness

Accept E1's causal stage label and confidence from proprioception, gripper/actions, and library alignment.
The allocator needs only `(stage_id, remaining stage extent, eligibility flags)`; it imposes no benchmark-specific labels.
Repeated visits count separately; uncertain/unknown stages use the pooled setting, with the established recovery rule
when the state is outside calibration support. Task identity may index library templates but never a hand-written rule.

Hard/easy means **marginal consequence of spending**, not raw stage failure frequency, duration, or action disagreement.
Estimate terminal-task SR changes for randomized stage-mode changes under a specified continuation controller.
State deviation and calibrated stall flags change which cheap modes are admissible; they are not themselves causal
SR labels. The library determines mode feasibility and cost. Success feedback determines relative value.
Until value differences are supported, use a common call intensity across stages and allocate total cost by occupancy.
Solve that common intensity directly from target C/D; an unresolved value estimate is not evidence to set calls to zero.
Do not divide total cost equally between a long transport and a short contact stage.

Candidate menu, reduced to nondominated settings after calibration:

| Mode family | Fewer calls | Look less | Look half |
|---|---|---|---|
| Pure endpoint/recovery | Call at every reference anchor | Reference commitment only | All reference cameras |
| Standard Commit-Cache | A calibrated call probability | Complete the existing cache/policy commitment | Both cameras |
| Single-camera cache | Same choice of calls | Same commitment | Select a supported one-camera retrieval path; complete the other camera on MISS |
| Longer library following | Calls only at permitted re-observations | Advance valid `next` rows within the identified stage | Both or validated single camera |

Call probability can be continuous by mixing cache/call packages; it needs no hand-tuned probability grid.
Commitment/cadence candidates are complete control blocks supported by the model horizon or library successor chain,
capped at the next stage boundary and at E3's calibrated admissible extent. Each mode specifies separately what happens
on CALL and CACHE: a 15-control cache continuation does not authorize a 10-step policy to execute 15 controls.
Begin with standard Commit-Cache and pure; add single-camera/long-follow modes only after their profile checks.
No shorter denoising, added encoder, or policy replacement is involved.

### Allocation mathematics and what “equal marginal SR per IR” means

Let `x_sj` be the mixture weight for mode j in stage s; sum_j x_sj=1.
Let `g_sj` denote the estimated *unconditional* terminal-SR contribution of that stage change under a frozen continuation.
Let `C_sj,D_sj` be expected cost and active controls/5 attributable to its occupancy, including probability of reaching it.
For a local additive approximation:

```
maximize  sum_sj x_sj g_sj
subject to sum_sj x_sj (C_sj - rho D_sj) <= 0.
choose j maximizing g_sj - lambda*(C_sj - rho D_sj), for each s.
```

With fixed duration, the upper concave envelope of value versus cost gives a fractional knapsack: purchase successive
mode upgrades by `Delta g / Delta C`. All interior allocations share the same marginal price; clamped stages need not.
With duration changes, use `Delta C - rho*Delta D`: a call that ends a stalled trajectory changes both numerator and
denominator. Stage weights must not be applied twice to an already unconditional success contribution.
This is a **local surrogate**, not a claim that task success decomposes additively across stages; Proposal 2 validates
the resulting whole controller. Shared manipulation errors and stage-order effects can invalidate a static table.

For fixed ten-control commitments and two cameras, the simple nominal map is
`IR_s = .076 + .424*p_s` for pi05, and `.074 + .426*p_s` for GR00T.
Thus a common p=.5283 approximately gives pi05 IR=.30. Actual library lengths account for odd terminal blocks.
For a stage consuming one third of decisions, a 0.10 change in p moves pi05 IR by `.01413` before occupancy changes.
With unequal CALL/CACHE commitments the map is a renewal ratio:
`(c_look + p*c_call_extra) / [(1-p)*L_cache/5 + p*L_policy/5]`, not the fixed-cadence formula.

Cost saving below A's roughly `.076` floor requires look less/look half: fewer calls cannot remove A's vision cost.
R4's **assumed** pi05 wrist pricing gives `.055198/2=.027599` for ten-control cache-only use, versus `.076` full vision;
this is a pricing calculation, not a measured R7 SR or portable latency result. Charge `.049890` missing-camera cost
on a wrist MISS in addition to `.848`. Do not price GR00T wrist use until its own encoder path is measured.
The maximum knob position explicitly restores the full reference path even if an estimated cheaper mode dominates it.

### Calibration and generality

- Library-only: obtain stages from E1; measure stage visit frequencies/control lengths; enumerate legal action tails
  and successor chains; derive state-deviation scales and stall thresholds by E3's episode-level rule. Use physical
  control timestamps, normalized/whitened state coordinates, and camera identifiers from the interface manifest.
- At most **10 dedicated non-test recordings per fitted deployment**: distribute coverage across available stage
  families, audit stage tracking and mode eligibility, and adjust occupancy/cost estimates. These recordings may contain
  many anchors but remain at most ten independent terminal-outcome clusters. Do not fit a large SR table from them.
- Values: initialize pooled/tied values; obtain supported changes from Proposal 2's ordinary deployment feedback.
  Pool stages whose uncertainty does not support separate settings. Successful library demos are not a pure-policy SR
  estimate, and reconstructed/open-loop action errors are not labels for a call's success benefit.
- New robot/benchmark: replace interface dimensions, native commitment, stage templates, and measured stage/camera
  prices. The same optimization and calibration rules apply. Current `.152/.148` prices and L10 are inputs, not constants
  to transplant. Any tolerance/confidence level is a declared operating requirement, not a LIBERO-state threshold.

Budget accounting uses counts, not observed wall-clock latency from all-shadow profiling. Re-estimate costs from actual
stage occupancy between episodes. If interventions force spending above rho, report target infeasibility and charged
overrun; an exact hard cap and unrestricted recovery cannot both be promised. At the minimum all *discretionary*
spending is at the cheap end; mandatory eligibility fallbacks define the actual floor.

### Expected effect, cost, hooks, kill criteria

The following are qualitative hypotheses, constrained by the measured A/pure gaps and §4's allocation replay.

| Cell | Expected role of the allocator; numerical anchor |
|---|---|
| pi05 L10-50 | Large need for selective rescue, but coarse-stage allocation is unproved: replay −1.10 pp; A→pure gap 19.0 pp. |
| pi05 L10-500 | Smaller call budget; likely modest upside, unproven: replay −0.46 pp; A→pure gap 7.7 pp. |
| pi05 Spatial-50 | Preserve calls where useful; candidate camera savings merit profiling: replay +0.33 pp; gap 14.9 pp. |
| pi05 Spatial-500 | Main opportunity is lowering the vision floor; replay +0.57 pp is unresolved; A already .976 vs pure .986. |
| GR00T L10-50 | Strong need for recovery; stage ranking unproved: replay +0.58 pp; gap 25.5 pp. |
| GR00T L10-500 | Limited call allocation gains expected: replay +0.40 pp; A .830 vs pure .866. |
| GR00T Spatial-50 | Avoid wasteful stages; replay +0.76 pp is unresolved; guard harm in R6 makes generic “hardness” suspect. |
| GR00T Spatial-500 | Prefer cheap cache when supported; replay −0.63 pp; A .964 exceeds pure .938, so extra calls need justification. |

No numerical frontier improvement is established. In particular the replay does not justify extrapolating local
probability changes to stage-wide pure/cache switches or claiming equality to pure SR.
The earlier task router's .909 SR at .288 IR on pi05 L10-50 remains a useful comparison, not a stage result;
its script averages task IRs, whereas our replay uses aggregate cost/decisions, so the two IRs are not interchangeable.

Cost tier: **T1**, using existing representation/retrieval and small CPU tables; no new neural model.
After stage identification, online selection is O(1) table lookup plus a coin and ledger update. Offline menu solving
is small; additional online CPU should be measured with `runprof` (design target <0.1 ms, **not measured**).
Any stage tracker, successor search, or stall cost is additional and must be included in that measurement.

Implementation sketch: wrap A's `query` to consume the stage and table, preserving its retrieval and issuing
`os_force_miss` through `guard_only`; reuse C10/GR00T policy-tail lifecycle. `blind_step` checks state-only eligibility,
advances allowed successors, and returns `LookReason` before a boundary or unsupported continuation.
Choose camera mode **before stage 1** using previously established stage/state: a post-encoder `query` cannot save that
encoder's cost. R4's configured camera path needs a per-connection pre-encoder selector for stage changes; cache both
camera-specific retrieval fits, and complete the missing camera on MISS. Log invocation counts and all mode overrides.

Kill criteria: drop stage-specific **call allocation** if frozen allocations fail to beat matched-cost uniform calls on
held-out initial states with episode-cluster uncertainty, or cannot transport to held-out task groups. Retain the common
knob only. Drop a new camera/cadence mode if actual cost saving disappears after completion/recovery, or its loss bound
exceeds the declared SR tolerance. Do not rescue a failed allocator by retuning phase cut points on test outcomes.

## 3. Proposal 2 — Start pure, descend one stage at a time, retain a pure reference

### Hypothesis and a short deployment rule

Hypothesis: stage granularity can reduce the size of a risky change and improve credit assignment, even when ten
calibration episodes cannot establish which stage is easy. It does **not** produce more independent success labels.

1. Start every stage at pure policy; freeze the incumbent controller for an evaluation block.
2. Randomize one stage's next cheaper mode against its incumbent mode, keeping other stages fixed.
3. Accept the cheaper whole controller only when its terminal-SR loss versus the pure reference is within tolerance;
   otherwise retain/revert it, and next examine the best remaining saving per estimated loss.
4. Refit stage values, lower the global target to the accepted cost, and repeat; new failures/drift can move it upward.

This is the owner's high-to-low story extended to stage coordinates. The user supplies total tolerance epsilon;
there is no arbitrary equal split `epsilon/K`. In the allocator, shared lambda gives the loss/cost tradeoff;
the acceptance decision checks the **whole controller against pure**, not only the latest local decrement.
Otherwise several individually tolerated stage losses could exceed the task's tolerance.
The manual target-IR knob remains available. Automatic descent changes its selected value only after evidence arrives.

### What is randomized, observed, and calibrated

Use Proposal 1's stage labels/menu. A stage treatment is a complete mode over its visit, assigned before it starts,
with logged propensity and fixed continuation. The primary outcome is eventual task success provided by deployment;
stage-exit progress, gripper agreement and stall rates are auxiliary diagnostics, never assumed object-success labels.
Do not blame every failed task on its final stage. One changed stage per episode yields an identifiable package effect
while holding other stage policies fixed, provided both treatments have support and assignment is exogenous.
Repeated-stage treatment scope (“first visit” or “all visits”) must be declared and logged.

Maintain both an incumbent comparator for local credit and a contemporaneous pure-policy reference for total loss.
An existing trustworthy pure reference can reduce collection initially, but library success=1 cannot replace it.
When the environment changes, old reference SR and old stage effects are invalid until recalibrated.
With binary successes, use episode-level differences and a prespecified sequential confidence procedure or fixed blocks
with alpha spending. For example, decision k can spend `alpha/[k*(k+1)]`; this controls repeated looks by a union bound.
Alpha and epsilon are explicit risk requirements. Show both point estimates and uncertainty; do not lower the knob
merely because a small batch contained no observed failure.

The library supplies settings/occupancy and the same ≤10 initial non-test recordings supply telemetry calibration.
Learning later uses ordinary deployment episodes in the owner's feedback story, **not an undisclosed extra calibration
campaign**. If even future deployment outcomes are capped at ten, fine SR-constrained stage optimization is not supported:
stay at the incumbent/uniform setting and expose uncertainty. No success feedback means no claimed SR convergence.
No model is trained or replaced: only treatment counts, empirical success/cost estimates, and mode tables change.

### Noise, convergence, reference cost, and portability

Ten independent episodes with zero adverse events have a one-sided 95% binomial upper bound **25.89%**.
Even the favorable zero-event calculation needs **149** independent trials to put that bound below 2%; splitting ten
recordings between four stages leaves 2–3 per stage, with bounds **77.64%–63.16%**.
This is a sample-size illustration, not a noninferiority test against an estimated reference. Two independent arms
near SR=.9 need approximately **1,729 episodes each** for a two-sided 95% difference interval with 2 pp half-width;
pairing can help, task variation/drift/selection can hurt. See `calibration_limits.json` for the exact calculations.

For fixed stages, independent stationary episodes and continued randomized support, means converge; distinguishing loss changes
of size delta costs order `1/delta^2` samples. No stage factorization can evade this terminal-feedback limit.
Share evidence across stages only when the pooled treatment response holds on held-out tasks; coarse pooling trades
faster estimates for possible bias. Large obvious losses can reject cheap modes sooner than small losses can be accepted.
The final cost point is a tested discrete-menu/coordinate solution, not a guaranteed global optimum.

Pure-reference and trial episodes consume the same budget: report total charged IR including them. With pure-control
time share eta and equal definitions, the mixture is `(1-eta)*IR_work + eta*IR_pure`; unequal lengths require the exact
cost/control ratio. A fixed sentinel fraction prevents asymptotic attainment of the nominal cheap floor. Decreasing the
fraction reduces overhead but slows reference updating and drift detection. Choose its sampling rate from needed
reference precision and observed variance, not a benchmark-specific constant; publish the resulting overhead.
SR need not be monotone in budget (dense Spatial cache can beat pure), so blind binary search in rho is unjustified.
Retain measured menu points and use local acceptance/reversion, with high/low endpoints defined by cost and mode.

Expected frontier effect by cell: inherit Proposal 1's opportunities, with initial IR at the pure reference and slower
descent wherever the loss margin is small or stages rarely occur. The dense Spatial cells may justify cheap modes
sooner; L10-50 may require retaining expensive stages. These are hypotheses, not measured convergence forecasts.
Portability comes from terminal user/environment success feedback and library-defined stages, not simulator predicates.

Cost tier/CPU: T1; episode-end sufficient-statistic updates and an occasional small menu solve. No additional forward
pass for the allocation decision. Pure sentinels and experiments add real inference cost and must be counted.
Hooks: an episode-level coordinator chooses the frozen table/treatment and sends it at `reset`; per-connection query,
judge and blind hooks execute it; accepted episode-end outcomes update the coordinator, not mutable shared method arrays.
At rho minimum/maximum, the exact same cheap/pure endpoint contract as Proposal 1 applies.

Kill criteria: drop autonomous descent if the deployment cannot report credible success, reference drift overwhelms
the detectable tolerance, required trials exceed the owner's acceptable time/cost, or stage settings oscillate without
stable improvement. Under those conditions retain the manual IR knob and pooled allocation; do not promise convergence.

## 4. Preliminary evidence and reproduction

`analyze_allocation.py` reads the eight `r06_c_validation/runs/r6c_*_U30|U18` arms; audits accepted attempts, unique
decision identities, source coins, library task membership, and N/V/M/SR against summaries. It resolves deployed
library aliases through `frontier_points.csv`. SHA256s of raw decision logs are retained in `allocation_results.json`.
Scratch parsed traces are cached only in `/tmp/r7_E4_allocation/`.

For this diagnostic only, stages are the three quantile intervals of stored-library row progress within each task;
online phase is the **pre-coin top retrieval row's** interval. These are not kinematic grasp/place annotations and do not
use the rollout's future length, final progress, simulator predicates, or eventual success. Three phases and the shift
sizes below are declared exploratory analysis choices, not proposed deployment thresholds. These quantiles weight rows
equally and include the failed source episodes present in the large libraries; success labels do not enter the partition.

First eligible encounter in each phase: use the logged p and score `(Z-p)/(p*(1-p)) * (Y-b)` with a task baseline b fitted
on opposite-parity initial states. This estimates a single-call excursion under the original continuation among reached
histories. All **24/24 nominal 95% intervals include zero**, with **468–500 episodes** per phase/cell.
Dose-1 episodes have no local support and are excluded from these contrasts: counts by sparse cell are 23/5/31/11
for pi05 L10/Spatial and GR00T L10/Spatial respectively. Library-500 arms mix p=.125 and .25, not one common p=.25.

Allocation replay: split initial states by parity within each task. Estimate each phase's full-trajectory score gradient
on one half; solve a three-variable fractional knapsack whose probability shifts have zero training-occupancy-weighted
sum. Evaluate on the other half using **the product of all anchor likelihood ratios**, not final realized call counts.
Keep the episode-dose lottery unchanged and never alter deterministic-dose decisions. Repeat in the other direction.
The main candidate changes supported call probabilities by at most .05; .025 and .10 are sensitivity checks (all shifts
are additionally capped at half the distance to a propensity boundary). Training uses only training outcomes.

Estimate delta-SR with the centered importance contrast `mean[(W-1)*(Y-b_task)]`; estimate IR as `sum(W*C)/sum(W*N)`.
Changing treatment can change duration and phase occupancy, so zero predicted extra calls need not imply exact held-out
IR equality. All reported IR is owner cost per requested five-control decision; standard U logs lack verified partial
terminal-control counts. Success is one outcome per episode, never one independent outcome per anchor.

| Cell | Logged uniform SR @ IR | Reallocated delta-SR, pp | Refit bootstrap 95%, pp | delta-IR | Weight ESS / 500 |
|---|---:|---:|---:|---:|---:|
| pi05 L10-50 | .878 @ .3012 | −1.10 | [−2.65, +1.82] | +.00098 | 410 |
| pi05 L10-500 | .874 @ .1788 | −0.46 | [−3.12, +3.05] | −.00012 | 384 |
| pi05 Spatial-50 | .940 @ .2917 | +0.33 | [−1.35, +1.20] | −.00008 | 465 |
| pi05 Spatial-500 | .968 @ .1794 | +0.57 | [−0.34, +1.03] | +.00167 | 448 |
| GR00T L10-50 | .770 @ .3009 | +0.58 | [−2.96, +3.32] | +.00066 | 408 |
| GR00T L10-500 | .848 @ .1805 | +0.40 | [−3.97, +2.97] | +.00107 | 395 |
| GR00T Spatial-50 | .928 @ .2983 | +0.76 | [−1.56, +1.71] | +.00107 | 460 |
| GR00T Spatial-500 | .954 @ .1811 | −0.63 | [−2.63, +0.96] | +.00044 | 452 |

Intervals resample within task×parity, **refit both allocation folds**, and repeat 2,000 times (seed 20260930).
They are exploratory percentile intervals for a nonsmooth selected policy, not simultaneous or sequential guarantees.
Frozen-selection Wald intervals are also saved; they would misleadingly single out pi05 Spatial-500 as positive.
All eight refitted intervals include zero. The best destination phase agrees across folds in **4/8 cells**
(both GR00T L10 cells, pi05 Spatial-500, GR00T Spatial-50). Sensitivity estimates retain uncertainty; no choice is selected after seeing them.
Importance effective sample sizes are 384–465/500, so the primary obstacle here is noisy stage value, not complete
trajectory-weight collapse. This does not establish support for larger mode changes or different camera/cadence policies.

Commands, from `/home/weiland/projects/openpi`:

```bash
taskset -c 28-29,72-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E4_allocation/analyze_allocation.py --refresh
```

Outputs are `allocation_results.json`, `calibration_limits.json`, and `audit_summary.json`; `--refresh` rereads raw logs.
The script is local and writes only to this directory
and the assigned scratch directory. No servers, simulators, model fits, GPU calls, or git operations were run.

## 5. Profile tools wanted

**Priority 1 — `stage_allocation_audit`: is the allocation ranking real, supported, and budget-neutral?**
Inputs: current randomized journals/coins plus E1's frozen causal stage annotations; owner price table; library manifest;
optional E2 stage treatment estimates. First run **offline on existing traces**, extending this script and existing
`breakdown`/`compare`. Outputs: stage occupancy, propensity/support counts, one-stage excursion effects, package-level
effects where supported, actual ratio-cost derivatives, fold rank agreement, uncertainty, ESS, and equal-cost frontier
comparisons against uniform and task-level allocation. Explicitly distinguish terminal SR from progress proxies.
Decision: whether Proposal 1 deserves any stage-specific priorities, which stages must pool, and whether call allocation
should be dropped in favor of stage-specific look/camera modes. Existing U logs cannot identify untried camera/cadence
packages; flag those rows unsupported instead of extrapolating.

**Priority 2 — `mode_budget_replay`: do the three levers actually buy the advertised IR?**
Inputs: library successors, frozen stage/eligibility annotations, existing B-val/P3 controls and shadow records, mode menu.
Outputs: stage-weighted C/D table, legal commitment coverage, boundary-crossing rate, camera completion cost, intervention
overrun, target-versus-realized IR, and CPU timing for table/tracker hooks. Use `timeline` and `runprof` for discrepancies.
Can run offline for legal/cost feasibility; it cannot certify SR of unexecuted modes. **New telemetry is needed** to audit
an implemented stage-changing camera/cadence path: pre-stage1 mode, chosen stage/confidence, intended/executed controls,
camera invocation counts, stage2/3 dispatch counts, missing-camera completion, override cause, and nominal/charged budget.
Decision: menu pruning, cheap endpoint, cadence restrictions, and feasibility of a per-stage camera selector.

**Priority 3 — `descent_learning_curve`: how long before lowering a knob is defensible?**
Inputs: declared tolerance/confidence, existing paired whole-controller outcomes and package randomization where available;
library stage visitation rates; stated reference/trial traffic allocation. Outputs: learning curves versus independent
episodes and charged IR, unresolved stage count, accepted/reverted settings, reference uncertainty, and sensitivity to
drift. Simulated feedback must be labeled a resampling study, not a new robot result. Existing data suffice to expose
noise and overhead; new package outcomes are required for an actual stage-descent prediction.
Future deployment logs must include controller version, candidate/reference assignment probability, stage visit scope,
accepted outcome/attempt, and reference epoch. Decision: Proposal 2's practicality versus retaining manual control.

## 6. Risks and open questions

- The library and ten recordings identify geometry/cost much more readily than causal SR. This limitation is central,
  not something another disagreement proxy can solve. A finer stage partition usually worsens terminal-outcome precision.
- Phase progress is only a diagnostic stand-in; stage classification mistakes can spend budget on the wrong event.
  Re-run the allocation audit with E1 labels before interpreting these negative coarse-phase results as a stage verdict.
- The additive allocator ignores interactions, altered visitation, and recovery obligations. Small local changes and
  whole-controller checks reduce this risk but do not prove global optimality or monotone SR.
- Conditional on a reached stage is a valid local estimand; conditioning on eventual successful stage completion,
  final call count, or future progress would be selection bias. Sequential propensities must reflect all overrides.
- More compute sometimes hurts. Pure is the required high-knob endpoint, not a universally optimal SR bound.
- Per-stage camera selection must precede encoding; no post-query accounting trick earns that saving. Single-camera
  price transfer across robots is unsupported, and all-shadow latency is not deployed latency.
- The stage-mode controller's cheap minimum may shift after observed guard usage; document infeasible targets.
  Large reductions in nominal IR that merely lengthen failed episodes should also report total cost per attempted task.
- Ongoing terminal feedback and a credible reference are prerequisites for autonomous convergence. If the owner means
  ten episodes for the *entire lifetime*, the defensible deliverable is a calibrated cost knob with unresolved SR risk.
