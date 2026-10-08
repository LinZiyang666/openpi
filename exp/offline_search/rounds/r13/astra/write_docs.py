"""Render handoff documents from the executed B analyses and frozen payload."""
from common import *
fence()
import csv
import numpy as np

def rows(name): return list(csv.DictReader((HERE/name).open()))

def main():
    f=read(HERE/'freeze.json');cnt=read(HERE/'grid_counts.json');sel=f['selection_evidence']
    static=read(HERE/'static_b_summary.json');power=read(HERE/'power.json');tests=read(HERE/'selfcheck.json')
    stamp=f['timestamp']; fits=f['distance_fits']; sig=read(HERE/'signal_quality_summary.json')
    validation=read(HERE/'validation_selection.json')
    sources={r['cell']:r['sources'] for r in fits}
    devrows=[read(p) for p in sorted((HERE/'data').glob('*.json'))]
    old=[r for r in devrows if r['root']=='r11_devknob_50' and r['method'] in ('periodic','distance')]
    selected_rows=[r for r in rows('robust_replay.csv') if float(r['width'])==f['score_width'] and r['split']=='heldout' and r['synthetic']=='False']
    source_table=table(['cell','calibration behavior paths','initialization'],[
        [c,'; '.join(':'.join(x[1:]) for x in sources[c]),'library only' if sources[c][0][1]=='library' else 'full served bank, external B inits'] for c in CELLS])
    scalar_table=table(['cell','log-distance lo','log-distance hi','bias @ .25 / .32 / .40 / .45'],[
        [c,f"{next(r for r in fits if r['cell']==c)['lo']:.6f}",f"{next(r for r in fits if r['cell']==c)['hi']:.6f}",
         ' / '.join(f"{next(r for r in fits if r['cell']==c and r['target']==t)['bias']:.6f}" for t in (.25,.32,.4,.45))] for c in CELLS])
    spec=f'''# R13 layer-four implementation specification

Frozen {stamp}, before accessing any closed-loop result of the new variants. `freeze.json`, `configs.json`, `test_grid.csv`, `pilot_grid.json`, `static_pilot_grid.json` and `predictions.csv` are the numerical payload. `controller.py` is a CPU reference, not a serving adapter. All new implementation belongs to sol; this directory launches nothing.

## Decision and scope

The primary comparison is unchanged **R11 guard-aware periodic** versus **distance_feedback**, a monotone geometric-distance threshold adjusted by the episode's committed cost ledger. Freeze feedback width at {f['score_width']}; no random floor, periodic cap, learned utility predictor, per-task parameter, or cross-episode learning is added. A B-recalibrated static distance threshold is a dev-only diagnostic. No combination arm is in this test grid.

The controller is a candidate target-hitting knob, not yet a demonstrated closed-loop one. Its selected width passed the cost band in {sel['pass02']}/{sel['n']} correlated B development replay checks, while static B recalibration passed {static['passes']}/{static['checks']}. The untouched B pilot below is mandatory. Size-500 remains an explicit transfer test because it has no dev complement.

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

Use float64 for the scalar arithmetic. `width={f['score_width']}` is shared across the whole experiment. `lo` and `hi` are cell-wide and shared across doses. `bias` is one scalar per cell and user target. The target is the requested total owner IR, not a knob-only call fraction. Mandatory guards, previous knob calls, blind slots, truncated episodes and actual lifecycle fallbacks all count in the ledger. No invented future tail credit is added; the real terminal ledger determines measured IR.

This is exactly a distance threshold at any fixed ledger: for `0 <= theta < 1`, call iff `d > exp(lo + theta*(hi-lo)) - 1`. If `theta < 0`, call on all finite distances; if `theta >= 1`, call on none. Do not clip `theta` or `D`; clipping would change the specified algorithm. Ties use strict `>` with no coin. A negative or nonfinite raw distance conservatively triggers a policy call and records the error. There is no random-call component.

Episode reset clears the proposal cache and ledger checkpoint. No deficit survives to another episode, connection, task or server. No persistent shared mutable state depends on arrival order. Repeating a proposal at the same decision step returns the exact cached proposal without charging any cost. A decreasing step without reset, inconsistent history length or nonmonotone counters is a contract error handled by the ordinary serving fallback. `controller.py:DistanceFeedback` and `selfcheck.py` specify these semantics; production history extraction must additionally verify correspondence to accepted plugin decisions.

Suggested knob-only reason is a newly allocated `69` (static diagnostic `70`), subject to sol's collision audit; reason numbering is not a parameter. Keep original guard reasons. Put audit fields before low-priority extras: actual raw distance, u, theta, D, target, width, bias, N/V/M, guard, knob-only call, nonfinite flag, and threshold saturation. Log threshold and score before the action is replaced by policy output. Include settings/source SHA in the fit manifest. Shadow-enabled dev recording must retain or exactly reconstruct the corrected cache proposal even on a MISS.

## B-only fitting, exact frozen values

`robust_study.py` is the authoritative fitting procedure, and `robust_fits.json` records its outputs. Reuse the scalar values in `configs.json` without rounded Markdown values. Serving should embed scalars in the ordinary fit pickle and import neither exploration code nor data paths.

{source_table}

For external B data, accept only complete B-labeled episodes disjoint from the actual served subset, reconcile journal attempts against decision logs and N/V/M, and verify look shard hashes. Within all behavior paths use the same identity split: the first eight SHA256 bytes of `R13-cal:task:init`, interpreted big-endian, are calibration iff even. This is an offline split; task/init never index a fitted serving parameter. `development_splits.csv` lists every membership. Width selection used the complementary development split, so that split is **not** an independent final validation set. The final pilot must exclude the entire identity union in `used_b_identities.json`.

For each available behavior path in a cell, use its calibration episodes. Compute log-distance quantiles on guard-eligible real looks at probabilities `linspace(0,1,1001)`; concatenate these equal-length quantile grids across paths; take its pooled tenth and ninetieth percentiles as lo/hi, flooring `hi-lo` at `.05`. This gives each behavior path equal influence on the scalar scale. These quantiles are scale fitting, not a task-dependent normalizer.

At each target and candidate width, replay each calibration path with its observed guard/vision/state sequence. Choose the single bias minimizing the largest absolute total IR error across those paths. `fit_bias_minimax` bisects `[-16,16]` for `37` iterations by whether the midpoint of the largest and smallest predicted IR exceeds the target; among visited candidates choose `(max_abs_error, abs(bias), bias)` lexicographically. It is a monotone fixed-state budget solver, not an environment simulation. The normalizer never sees policy error labels or SR.

When no suitable B dev trajectory exists, use R11's SHA-verified whole-episode-held-out library scores and guard replay. Existing selected-library PCA exposure and reduced fold donor banks remain limitations, as in R11. Split complete packed library episodes by column parity for this calibration/development check. No self-query/full-library state is presented as an external dev episode. This applies to GR00T L10-200 and both L10-500 cells in the current freeze; the GR00T size-200 live pilot checks this transfer. Size-500 has no such pilot.

Global width candidates were `.5, 1, 2, 4, 8`. Width `{f['score_width']}` was selected from B development evidence for budget robustness with material distance engagement. This is an explicitly selected development result, not a preregistered validation success. Frozen scalars:

{scalar_table}

Only these scalar layer-four parameters vary by cell/dose. No per-task thresholds, gains, budgets, success gates, episode-length predictions or task-specific rates are allowed. Predictions, test settings and the grid may not be refitted using A.

## Periodic reference and dev-only static ablation

Periodic is exactly R11's dithered gap cap: `cap=floor(K)+Bernoulli(frac(K))` from the unchanged keyed `uniform('R11-gap-v1',0,task,init,last_call_step,'cap')`. Count accepted real-look HITs since the last policy call; the guard resets this count just like any other call. On an eligible anchor call iff `run >= cap`. Keep the R11 keyed-coin behavior, including at episode start. `periodic_fits.json` is recomputed by the unchanged R11 library `ScheduleReplay.solve`, rounded to its existing four-decimal convention. Existing target settings reproduce R11, and the high-dose extension uses the same solver.

`distance_static_b` uses one fixed raw-distance threshold per cell/target and strict `d > threshold`, OR with guard. `static_study.py` bisects the observed calibration range for a minimax threshold across the same B calibration paths. Its exact thresholds are in `static_pilot_grid.json`. It runs only on B at `.32/.40`; it does not replace the primary feedback method based on pilot SR. The static experiment isolates whether better calibration alone fixes the budget problem.

## Dev pilot, pass/stop rules, and release

Run `pilot_grid.json`: {cnt['pilot_arms']} arms of `100` episodes, paired K=10/task B inits. It covers both primary methods at every frozen dose in each cell with a nonempty dev complement, plus a second `.32` replicate in the size-50 cells to estimate paired outcome variance and replicate dependence. `static_pilot_grid.json` adds {cnt['static_pilot_arms']} dev-only arms. The total is {cnt['pilot_episodes']} pilot episodes. Primary validation uses fresh identities outside **all** development identities used here and outside the served subset; repeats use the same fresh validation identities. Never relabel old K=10 inits as fresh validation. Freeze the actual selection list and artifact hashes before any pilot starts.

The exact fresh pair list is already supplied in `validation_selection.json`, frozen at {validation['timestamp']}. It uses common pairs across models and tested sizes within each suite, excludes the cross-model union of previously used B identities, and excludes every relevant served subset. `validation_selection.py` found a minimum of {min(r['available'] for r in validation['availability'] if r['suite']=='l10')} available inits per L10 task and {min(r['available'] for r in validation['availability'] if r['suite']=='spatial')} per Spatial task, sufficient for the requested selection. Coordinator tooling must build its normal SHA-bound contracts from these lists and reverify the actual prefits; no run root was created here.

Use the same host as the corresponding A cell: all GR00T local 4090; all pi05 H100. Existing R13 frozen-method recordings may improve later diagnostics, but do not silently revise this numerical freeze. The optional requested size-200 calibration recordings are not prerequisites to building these frozen artifacts; current library initialization is explicit.

Pass mechanics first: exact off equivalence, accepted-ledger reconciliation, no double guard/knob call, correct policy-tail lifecycle, same-step idempotence, reset/clone isolation, deterministic output under reordered independent connections, and zero unknown/missing identities. Sol must check these in the real plugin, including invalid-tail fallback and early termination. The explorer CPU reference passed {tests['proposals']} proposals, {tests['retries']} retries and {tests['bound_scenarios']} analytic-bound scenarios; that does not replace plugin tests.

For each primary pilot arm require absolute aggregate realized IR error at most `.02`, using `sum(cost)/sum(N)`, and report an episode-cluster bootstrap interval. The paired distance-periodic IR difference must be at most `.01` at each dose for a direct matched-budget label. No average across cells may conceal a failing arm.

Require distance engagement on its own recorded validation states: at least `.40` of non-guard looks with `0 <= theta < 1`, and at least `.20` changed selected-call positions in a same-trajectory replay replacing u with constant `.5`, measured as symmetric difference divided by union of non-guard selected positions. Report within-episode distance shuffling as a second diagnostic. These thresholds screen out a mostly score-insensitive rate controller; they do not establish beneficial placement. Report near-ceiling saturation explicitly. Exclude nonfinite fallback looks from nominal engagement and count them separately.

For the static ablation report the same cost gate; failing static arms stay as a calibration failure result. Do not keep lowering thresholds on validation until they pass.

If a mechanics gate fails, repair the implementation and repeat the affected check with identical frozen arithmetic. If a budget or engagement gate fails, hold the affected primary cell from claims of validated target control; record the failure, retain all results, and hand back for a separately versioned B-only redesign/freeze. Do not tune on the validation set or A. The coordinator may measure a failing frozen method as a **Q1 failure arm**, but must not call it a passed matched-budget comparison. There is no automatic switch to another width or method.

Size-500 can proceed only after the same controller passes on available smaller-library cells and mechanics are validated for its actual fit. It is labeled **library-only transfer**, never B-validated. Its A result either establishes or fails target control without any A-based adjustment.

## Frozen test grid and host/batch plan

All exact arm IDs, configs, pair groups, replicate seeds and fixed randomized order are in `test_grid.csv`. There are {cnt['arms']} arms / {cnt['episodes']:,} episode runs: {cnt['active_arms']} periodic/distance arms, {cnt['off_arms']} off controls, {cnt['pure_arms']} pure controls.

- Cells: each model's L10 at library sizes `50, 200, 500`, and Spatial at size `50`.
- Both primary methods at `.25, .32, .40, .45` in every cell. `.32/.40` each have four replicates; `.25/.45` each have two. Each arm has `500` A episodes using the full unchanged task/init manifest.
- Off is repeated once per cell per replicate block. Pure C10 and pure C5 are repeated per model/suite/replicate block and shared across library sizes only after action/lifecycle equivalence is verified.
- Pure C10 is the layer-four all-call endpoint: force every real look to policy, retain one five-control tail, resume a real look every ten controls. R11 periodic `K=0` is the reference implementation; do not approximate it by a calibrated target `.50`. Pure C5 is the contextual standard policy called every five controls, owner IR `1`; it is not on the layer-four C10 interpolation curve.
- Keep **all** GR00T arms and controls on the local 4090; **all** pi05 arms and controls on H100. Do not split a cell across hosts when capacity changes. Queue it instead. Paired methods are adjacent within a cell/dose/replicate group; their order and the groups' order are frozen by seed `20261004`. Repeat blocks use different rollout seeds but unchanged R11 knob coin seed `0`. Verify and log whether those rollout seeds actually seed policy/simulator randomness; a metadata label alone is not a new stochastic replicate.
- Complete replicate blocks in order. No SR-based stopping, extra dose selection, dropped cell, best-replicate selection or host migration after viewing A. The estimate from the brief's throughput is {cnt['projected_days'][0]:.1f}–{cnt['projected_days'][1]:.1f} test days, excluding the pilot and capacity loss. This larger plan budgets the requested pooled precision rather than treating many correlated arms as independent evidence.

## Budget-matched estimands and power

`PLAN.md` freezes the analysis. Primary contrast is distance minus periodic at realized IR `.32` and `.40`, with equal weight to model × suite, then equal weight to available library sizes within that group, doses and replicates. Keep all nominal-target contrasts and their actual IR differences visible. A direct same-target comparison is called matched only when the realized IR difference is within `.01`; being individually within `.02` is insufficient.

For the fixed realized-IR primary estimand, interpolate each method's measured SR-vs-IR curve linearly between adjacent realized-IR points, with off and pure C10 endpoints. Do not extrapolate, impose increasing SR, choose an x-value from SR outcomes, or interpret interpolated doses as newly selected deployments. Recompute interpolation in every paired bootstrap resample. If common support or the Q1 gate fails, say the intended matched comparison is not identified; do not quietly substitute another cell/dose. Report curve interpolation beside the direct paired measurements and its modeling limitation.

The primary grid contains {cnt['primary_pairs']:,} paired episode-run contrasts, but repeated task/init identities are dependent. Cluster by original suite/task/init across models, sizes, doses and replicates; stratify resampling by suite/task. Report both init-cluster intervals conditional on recorded batches and a replicate-block sensitivity analysis. No binomial test on the pooled count of all arm runs.

The B-based variance scenario gives a four-replicate MDE of {cnt['mde4_by_replicate_correlation'][0]['mde_pp']:.2f} pp with independent replicate noise, {cnt['mde4_by_replicate_correlation'][1]['mde_pp']:.2f} pp at replicate correlation `.25`, and {cnt['mde4_by_replicate_correlation'][-1]['mde_pp']:.2f} pp under perfect dependence. These use old unequal-budget B method discordance as a planning scenario, not a known variance of the new method. The duplicate B pilot measures this uncertainty. If it implies inadequate precision, record that before A; no finite replicate promise removes an init-specific variance floor. Individual cell arms remain much noisier than the pooled effect.

## Operating limits

This controller resets each episode. Even with correct accounting, finite episodes, behavior-induced score/guard shifts and guard-only infeasible episodes prevent an unconditional exact aggregate budget guarantee. In the no-guard nominal cadence, writing `q=(2*target-cv)/cm`, the pre-look debt updates as `D_next=D+call-q` and stays between `min(0,-width*bias-q)` and `max(0,width*(1-bias)+1-q)` once initialized at zero. Mandatory guards can break the upper bound. They cannot be suppressed to enforce a target.

Periodic has its inherited hard bound `ceil(K)` cache anchors. Feedback has no universal cache-gap bound in the presence of accumulated guard debt: a long burst of mandatory calls can be followed by a long cache interval while repaying cost. Maximum observed gaps and conditional no-guard bounds must be reported; do not advertise feedback as a periodic cap. Both have cheap scalar controller work, but extracting d must reuse the existing geometry without a second vision or policy pass. Sol should benchmark incremental CPU query time separately from overall policy latency.
'''
    (HERE/'SPEC.md').write_text(spec)
    forecast=[]
    for cell in CELLS:
        for target in (.25,.32,.4,.45):
            d=next(r for r in f['predictions'] if r.get('cell')==cell and r.get('method')=='distance_feedback' and r.get('target')==target)
            p=next(r for r in f['predictions'] if r.get('cell')==cell and r.get('method')=='periodic' and r.get('target')==target)
            rr=[r for r in selected_rows if r['cell']==cell and float(r['target'])==target]
            forecast.append([cell,target,f"{p['predicted_ir']:.4f}",f"{d['predicted_ir']:.4f}",
                f"{min(float(r['ir']) for r in rr):.4f}–{max(float(r['ir']) for r in rr):.4f}",
                f"{d['sr_delta_distance_minus_periodic_pp']:+.2f}"])
    pred=f'''# Prospective prediction

Timestamp: **{stamp}**. Frozen before reading any closed-loop result of distance_feedback or distance_static_b. No A traces, outcomes or query arrays enter this prediction. The R11 aggregate statements supplied by the brief are background; newly computed numeric forecasts use B evidence and explicit subjective priors only.

Every arm, including every replicate and off/pure control, has a row in `predictions.csv`. The point forecast for feedback IR is the user's target; the planning band is target ±`.02`, an acceptance aim **not** a statistical prediction interval. The table's replay range is the observed range over complementary development episodes and behavior paths with fixed states, not a future closed-loop range. Periodic predictions are from the unchanged library solver. Repeated arms share a point forecast; repeats are not promised identical realizations.

{table(['cell','target','periodic IR forecast','feedback IR forecast','feedback fixed-path dev range','distance − periodic SR prior (pp)'],forecast)}

The central value prior favors distance slightly in pi05 L10 at the lower doses, periodic slightly in GR00T L10 and Spatial, and approaches a tie at the highest dose. These are deliberately weak hypotheses: each listed SR difference has a subjective ±`3` pp plausibility range, including zero. They are not fitted success predictions. Pi05's stronger B shadow error ranking motivates its sign; GR00T's weaker eligible-look ranking and periodic's guard-aware spacing motivate the opposite sign. Error ranking cannot establish rescue value, so a null result remains plausible everywhere, including at dense libraries.

I predict B calibration plus cost feedback will substantially improve budget delivery over the old frozen distance rule. I do **not** predict that it will dominate periodic's SR/IR frontier in every cell. High-dose gains should be smaller as both methods approach the common policy-at-every-look endpoint. Size-500's target delivery is especially uncertain because no full-bank B dev calibration is possible. A guard-heavy episode or changed short-episode distribution can still miss the target.

There is no absolute SR forecast for library-only off arms or pure controls: no same-host, same-batch B measurement establishes it here. Available B off SR values in `predictions.csv` are explicitly source-labeled planning anchors, not A estimates. Pure C10 IR includes a library-based terminal-truncation estimate; pure C5 IR is `1` by its ledger definition. These controls must be measured fresh in the same test batch.

The freeze is SHA-bound in `MANIFEST.sha256`; do not overwrite it after a pilot failure. Any redesigned variant needs its own version and prediction timestamp before its new closed-loop results.
'''
    (HERE/'PREDICTION.md').write_text(pred)
    qtable=table(['cell','B source/population','looks','pooled rank correlation','within-episode mean','top-quartile error lift'],[
        [r['cell'],r['source']+'/'+r['population'],r['looks'],f"{r['pooled_spearman']:.3f}",
         f"{r['within_episode_spearman_mean']:.3f}",f"{r['top_quartile_lift']:.3f}"] for r in sig])
    comp=table(['width','B checks within band','mean abs IR error','minimum / mean interior share','constant-score placement change mean','synthetic max error'],[
        [r['width'],f"{r['pass02']}/{r['n']}",f"{r['mae']:.4f}",f"{r['min_interior']:.3f} / {r['mean_interior']:.3f}",
         f"{r['mean_placement_change']:.3f}",f"{r['stress_max']:.4f}"] for r in read(HERE/'robust_summary.json')])
    oldtable=table(['cell','periodic B IR','old distance B IR','periodic B SR','old distance B SR'],[
        [c,*[f"{next(r for r in old if r['cell']==c and r['method']==m)[metric]:.4f}" for metric in ('ir','sr') for m in ('periodic','distance')]]
        for c in CELLS if c.endswith('_50')])
    audit=read(HERE/'shadow_audit.json')
    offline=f'''# B-only offline evidence

Computed here by `extract.py`, `replay_study.py`, `robust_study.py`, `static_study.py`, `signal_quality.py`, `periodic_and_power.py` and `selfcheck.py`. This evidence shaped the freeze at {stamp}. All writes are in this directory; no policy model, GPU, simulator, server, remote command or test-A data was used. `*_reads.json` inventories the data access. The Python audit fence rejects unapproved run roots, all non-library store inputs, mixed historical A/B analysis artifacts and writes outside this directory.

## Existing B evidence and reconstruction audit

`inventory.csv` records the contract-listed roots, completed episodes and look counts. `data/` contains only compact accepted B traces. Attempts are reconciled against accepted journals, episode decision sequences are contiguous, and reconstructed N/V/M equal the root's cost ledger. Pool contracts prove each accepted init is outside the actual served subset; artifact and shard hashes are verified.

At the existing nominal `.32` setting, B independently reproduces the old distance budget shortfall:

{oldtable}

These SR columns are **unequal-budget B measurements**, not evidence about matched-IR superiority. They are not a basis for selecting test cells. The cells/doses cover the contract's requested models, suites and size range uniformly.

The recording field named `distances` is negative wrapper Result score, which in this serving path is a kernel-transformed value with nearest score zero. It is unsuitable for calibrating raw distance. This analysis replays the exact deployed `_dist` on saved camera keys/state with the full served bank and inherited early/stale metric. No image encoder is run. The cache proposal is reconstructed from recorded parent donor rows/weights and the unchanged deployed corrector; cache HITs verify the reconstruction. Maximum absolute hit-action discrepancy among shadow-labeled sources is {max(r['replay_hit_max_abs_error'] for r in audit):.9g}. It is float arithmetic drift, not an alternative cache policy.

Policy MISS `action` and `shadow_policy` can be identical; comparing those would falsely assign zero call value. This analysis always compares shadow policy against the reconstructed corrected cache proposal, including at guard MISSes. The normalized motion label is mean squared difference over the first ten controls and first six action channels divided by the frozen cell action scale. Gripper sign disagreement is retained separately. Shadow policy samples are one stochastic same-stage-one action label, not ground-truth recovery utility.

## Making the distance knob honest

First fit one static raw-distance threshold per cell/dose using only the calibration identities and minimizing worst error across available B behavior paths. On complementary development identities it passes the ±`.02` band on {static['passes']}/{static['checks']} correlated checks; mean absolute error is {static['mae']:.4f}, worst {static['max_error']:.4f}. Better calibration helps, but a single threshold remains sensitive to the behavior path. This supports a dev-only static checkpoint rather than making it the primary knob.

An off-path-only fitted feedback bias also failed a path-transfer check. `replay.csv` preserves that attempt. The final fitting procedure uses one cell-wide minimax bias across available **calibration** paths, without mixing their outcomes into fitting. The complementary identities select one global feedback width:

{comp}

Choose width `{f['score_width']}`: {sel['pass02']}/{sel['n']} nominal B checks, MAE {sel['mae']:.4f}, worst error {sel['max_error']:.4f}. The threshold lies inside score support on at least {100*sel['min_interior']:.1f}% of eligible looks across these checks, averaging {100*sel['mean_interior']:.1f}%. Replacing distance by a constant changes on average {100*sel['mean_placement_change']:.1f}% of the union of selected non-guard positions (minimum {100*sel['min_placement_change']:.1f}%). This is evidence of distance engagement, not evidence of beneficial placement.

The synthetic tests multiply every validation distance by `.5, .75, 1.25, 1.5` without changing guards/actions. For the chosen width, {sel['stress_pass02']}/{sel['stress_n']} such checks stay inside the band; worst error is {sel['stress_max']:.4f}. These deliberately broad distortions include library-only cells. They are scenario tests, not statistical uncertainty bounds. Widths with more interior thresholds are less robust to this shift; the smaller width spends accurately with substantially more saturated, score-insensitive decisions. This tradeoff is why budget control and signal engagement are both pilot gates.

The development checks overlap across doses and paths and helped select width. Their count is not a sample size, and passing them does not establish closed-loop target delivery. No replay changes future states, success, length or guard events in response to the hypothetical actions. The untouched B pilot and then frozen A measurement are needed for those claims.

## Signal quality: library versus live B cache states

{qtable}

Correlation is Spearman on looks; the within-episode column averages episode correlations with enough nonconstant observations. Top-quartile lift is the fraction of episode-weighted action error captured divided by the episode-weighted mass selected by the pooled distance top quartile. `signal_quality.csv` also contains task-demeaned rank correlation. Report all-look and guard-eligible results separately: the guard already claims some high-error looks, so its overlap cannot be credited as new layer-four value.

The present shadow comparison covers both L10 size-50 cells on knob-off trajectories. Distance retains a positive live action-error association there, including on guard-eligible looks; the signal has not simply vanished in closed loop. Within-episode associations are weaker than pooled ones. These measurements do not establish which method rescues failures or whether the shadow disagreement is necessary for task success. Spatial, size-200 and active-method shadow evidence remain requested; they are not invented from labels that are absent.

`distance_paths.csv` separates actual B behavior paths and their score/guard distributions. The normalization and scalar solver use only permitted B evidence. `development_splits.csv` and `used_b_identities.json` make the next genuinely independent selection auditable.

## Limits and next measurements

No closed-loop claim has been made for the new variant. Size-500 has no external B complement. GR00T size-200 currently has library-only initialization. The owner ledger is a normalized cost index, not wall-time latency; incremental controller CPU latency remains sol's integration measurement. Four-replicate precision scenarios in `power.json`/`grid_counts.json` expose the dependence assumption instead of multiplying arm counts into an artificial effective sample size.

The early REQUESTS.md was written at {read(HERE/'request_plan.json')['timestamp']}; the final request addendum binds it to the frozen arm files. `PLAN.md` fixes the later Q1–Q6 analyses. `TEST_DESCRIPTIVE.md` records that no new test-A descriptive analysis was performed in this phase. ANALYSIS.md and REPORT.md are reserved for a later coordinator resume after the frozen measurement grid runs.
'''
    (HERE/'OFFLINE.md').write_text(offline)
    plan=f'''# Measurement and handoff plan

Frozen with the implementation/prediction at {stamp}. Exact jobs are in `pilot_grid.json`, `static_pilot_grid.json` and `test_grid.csv`. The coordinator launches; sol implements and validates the plugin.

1. **Implementation now:** embed `configs.json`; retain the base layers; validate real history/tail/clone/reset behavior and byte-equivalent off. Reconcile source SHA. Verify pure C10 equivalence and actual rollout seed behavior before reusing controls or counting stochastic replicates.
2. **Independent B pilot:** exact gate rules in SPEC. Freeze fresh selection manifests against `used_b_identities.json` and served subsets. Run every primary dose before releasing that cell. Shadow-enabled active `.32` arms provide wasted-call labels. The duplicate pilot assesses paired outcome variance and replicate dependence without changing the winner, dose grid or method based on SR.
3. **Frozen A waves:** replicate blocks in fixed order. First two blocks cover the full dose curve; later two add precision at `.32/.40` with contemporaneous off/pure controls. The full plan is {cnt['arms']} arms; it is not stopped on intermediate A SR. No fit, gain, threshold, host or cell selection uses A. If a dev gate fails, preserve a clearly labeled Q1 failure rather than presenting unmatched spending as a matched frontier.
4. **Analysis after resume:** reconcile every accepted task/init/attempt and N/V/M; retain failures; inventory incomplete/aborted attempts separately; never count an incomplete duplicate as a replicate. Report exact host/checkpoint/fit hashes and missing arms before any outcome estimate.

**Q1.** For each arm show target, aggregate realized IR, signed error, paired method cost difference, guard floor, startup versus later debt, threshold saturation and engagement. Cost is a ratio of pooled costs and decisions, not the mean of episode IRs. Bootstrap identities for uncertainty. Short episodes, mandatory-guard-heavy episodes, each library size and host are separate operating checks. Size-500 has a transfer label even if it succeeds. Report how many independent scalars and what calibration data each method needs.

**Q2.** The primary outcome is distance minus periodic at fixed realized IR `.32/.40`, with weights: equal model × suite; equal library sizes within each group; equal two primary doses; equal four replicates. Estimate the curves by predeclared piecewise linear interpolation of SR against measured IR, bounded by off and pure C10. Sort by IR, average exact duplicate-IR SR points; do not assume monotonic SR or extrapolate. Keep direct paired nominal-dose SR/IR differences beside interpolation. No A-derived threshold or executable dose is selected by interpolation. If there is insufficient common support or Q1 failure, report that primary comparison as unidentified, with the unmatched measurements retained.

For confidence intervals resample original init identities within each suite/task, carrying the complete model/size/method/dose/replicate outcome-and-ledger vector together; recompute aggregate IR and interpolation in each resample. Use `5000` resamples with seed `20261004`. Conditioning on the benchmark tasks avoids treating them as a random sample of all tasks. Add leave-one-replicate-block and two-way init/replicate resampling sensitivity. Model/suite/size/dose contrasts are secondary and must show their own uncertainty; apply Holm correction if declaring a family of subgroup winners. No claim of individual-arm precision at the pooled target.

The grid's {cnt['primary_pairs']:,} primary paired episode runs repeat original init identities. Four-replicate planning MDE ranges from {cnt['mde4_by_replicate_correlation'][0]['mde_pp']:.2f} to {cnt['mde4_by_replicate_correlation'][-1]['mde_pp']:.2f} pp across the specified dependence scenarios. Use the B replicate pilot to audit the assumption before A. If its variance floor prevents the requested precision, say so; do not manufacture independent episodes from repeated doses or data-driven choose easier cells. All cost and outcome analyses retain their different denominator/weighting conventions.

**Q3.** Predefine phase bins by elapsed controls (absolute) and elapsed fraction of the final episode (retrospective only), and separately by retrieved donor progress. Tabulate lag since **any** call and since guard calls, policy-call bursts, cache-only run length, gripper transitions, normalized/absolute episode length, success-time distribution and restricted time to success at each suite's timeout. Distinguish cache HIT actions from policy-tail HIT actions. Pair each active arm with its same-batch off on original inits to report rescue `off fails / method succeeds`, harm `off succeeds / method fails`, and their net difference with paired uncertainty; these are stochastic paired outcomes, not proof of an individually doomed init. Describe task concentration only; never turn it into a task-indexed knob.

For wasted calls, use B shadow labels only. Primary plot is the full distribution of normalized corrected-cache versus policy motion MSE at selected calls; also show gripper sign agreement. Define a low-difference cut from the lower quartile of **calibration B off eligible-look** motion error and freeze it before applying it to independent active B states. Count calls with error below that fixed cut and identical first-ten-control gripper signs; show a threshold sweep as sensitivity, not as a fitted success surrogate. Compare selected versus eligible opportunities at the same cost, separating mandatory guards. MISS-versus-MISS action equality is not a valid cache-error label.

**Q4.** Extend `signal_quality.py` to new B shadows with episode bootstrap uncertainty, top-error capture curves and task/episode adjusted correlations. Compare the same corrected-motion proxy to library-held-out labels; label stochastic policy differences and host effects. No claim that action error identifies causal call value. A causal intervention study would require its own paired B rollouts and prospective protocol; it is outside this frozen grid.

**Q5.** No combination is tested in the primary grid. The feedback mechanism already consumes guard costs but adds no periodic floor or cap. A future combination would need a new B calibration and matched-IR freeze. A flattering post hoc union of calls is not evidence for a combination.

**Q6.** Report calibration sources, scalars, determinism/idempotence, actual CPU overhead and total policy latency separately, target delivery by episode length/guard rate/library size, score saturation, and maximum observed cache intervals. Periodic's analytic cap is stronger than the feedback method's guard-dependent debt bound. Do not extrapolate host equivalence.

**Data boundary.** Future A reads are measurement only and must be in a separate analysis entrypoint. No code in the current B pipeline opens them. The optional frozen-R11 A descriptive exception remains in `TEST_DESCRIPTIVE.md` and cannot select parameters, variants, doses or cells. No such additional A analysis is needed to build this candidate. Later ANALYSIS.md and REPORT.md must show the actual matched-IR result, prediction checks and uncertainty; REPORT begins in plain Chinese for the owner.

**Open dependencies:** sol's real plugin implementation/tests; independent validation selection manifests; the pilot's own induced states; active-method shadow cache proposals; same-host pure controls; actual policy/simulator seed control; full grid completion. The current R13 recording root announced in ../DATA.md contains only frozen R11 methods and does not validate the new controller. No result from it was needed to freeze this proposal.
'''
    (HERE/'PLAN.md').write_text(plan)
    (HERE/'TEST_DESCRIPTIVE.md').write_text('''# Test-A descriptive boundary

No raw test-A trace, query array or newly computed A statistic was used in this design phase. Historical R11 report prose and the brief's aggregate findings were read as contract context; their additional numeric tables were not calibration inputs or new-grid selection criteria. The present reports recompute their measured numbers from permitted B inputs in this directory.

If the coordinator later requests descriptive Q3 work on already-frozen R11 methods, place its code/output here with a separate A-only measurement provenance and no write path to calibration/config/prediction files. It must not choose a new threshold, gain, dose, variant or cell. This file currently contains no A analysis.
''')
    request_add=f'''\n\n# Frozen-request addendum — {stamp}

This supersedes provisional counts/algorithm references above. Exact new jobs: `pilot_grid.json` ({cnt['pilot_arms']} primary validation/replicate arms) and `static_pilot_grid.json` ({cnt['static_pilot_arms']} static diagnostic arms), all `100` episodes each. Primary release gates are in SPEC.md. The static checkpoint is retained for B diagnosis even though its fixed-path transfer is worse. The added size-50 `.32` duplicate is for replicate-correlation/power assessment. Do not use it to pick an SR winner.

Pin GR00T validation to local 4090 and pi05 validation to H100, matching the frozen test host. Fresh B validation must exclude the complete old-identity union in `used_b_identities.json` as well as the actual served subset. Reusing the existing R11/R12 K=10 selection is not independent validation. The extra size-200 calibration recordings in the initial request are optional diagnostics now; sol can build immediately from the frozen current scalar payload. No scalar changes are authorized by these new recordings within this freeze.

The exact feasible fresh B pair lists are frozen in `validation_selection.json`; use them to create normal validated contracts. They are common across models/tested sizes within each suite and exclude the complete cross-model old-identity union. The root announced in ../DATA.md is `r13_devrec_50`, containing frozen methods, and was not read for this freeze. The old recorder's `distances` field is transformed Result score; retain raw geometric `_dist` minimum explicitly if practical, otherwise saved keys permit exact CPU reconstruction. Active `.32` shadows must preserve/reconstruct the corrected cache proposal on every look, including MISSes. Existing L10 off shadows already support the initial signal-quality analysis; Spatial/active-method shadows remain useful.
'''
    req=HERE/'REQUESTS.md';s=req.read_text();s=s.split('\n\n# Frozen-request addendum')[0]+request_add;req.write_text(s)
    hand=f'''# Handoff to coordinator and sol

**Ready to implement and run the independent B pilot.** Algorithm/scalars, forecasts and the exact test grid froze at {stamp}. Start with SPEC.md and `configs.json`; run `pilot_grid.json` before interpreting A arms as target-controlled. No launch or remote operation was performed here.

The proposed distance threshold uses true nearest geometric distance and one episode-local committed-cost feedback. Width is `{f['score_width']}` globally; cell/dose scalar fits use library/B data only. It passed {sel['pass02']}/{sel['n']} correlated development replay budget checks (worst error {sel['max_error']:.4f}) and retained material score-dependent placement. Closed-loop delivery and value are still unmeasured. Static B recalibration alone passed {static['passes']}/{static['checks']} checks. Do not promote replay to a live success claim.

Files: SPEC.md (exact implementation/gates), PREDICTION.md + predictions.csv (prospective forecasts), PLAN.md (measurement/statistics), OFFLINE.md (executed B evidence), REQUESTS.md (early and final requests), configs.json (immutable scalar payload), test_grid.csv (full paired/host-pinned grid), pilot_grid.json + static_pilot_grid.json (new B jobs), validation_selection.json (exact feasible fresh pairs, common across models/sizes per suite), used_b_identities.json + development_splits.csv (selection audit), controller.py/selfcheck.py (reference arithmetic).

Test plan: {cnt['arms']} arms, {cnt['episodes']:,} episode runs; four replicates at the two primary doses and two at the curve extensions, with repeated off and pure controls. Approximate test time {cnt['projected_days'][0]:.1f}–{cnt['projected_days'][1]:.1f} fleet days from the brief's throughput, plus pilot. The frozen cell order keeps GR00T local and pi05 H100. Do not trim replicates after reading outcomes and retain the dependence caveat in the power calculation.

Important integration correction: recorder `distances` are transformed wrapper scores here, not geometric distance. Use `_dist(q)[7].min()`. We reconstructed corrected cache proposals against saved HIT actions to float precision before computing shadow labels. A MISS action compared with its own shadow does not measure wasted-call error.

Open items: real-plugin tests and overhead; true independent B selection; pilot cost/engagement/replicate-variance results; size-500 library-only transfer; active/Spatial shadow labels; pure-control lifecycle and rollout-seed verification. If gates fail, keep the failed candidate and require a separately versioned B-only redesign; never adjust using A. Deep B mechanism work can continue without modifying frozen config/forecast files.

ANALYSIS.md and the Chinese-first REPORT.md are intentionally deferred until the coordinator resumes with the completed frozen grid. TEST_DESCRIPTIVE.md contains the explicit A boundary and no newly computed A results.
'''
    (HERE/'HANDBACK.md').write_text(hand)
    (HERE/'STATUS.md').write_text(f'''# Status — {now()}

SPEC/PREDICTION/PLAN and exact numeric/test-grid payload frozen at {stamp}. Ready for sol implementation and independent B pilot; coordinator owns all launches. REQUESTS.md was written early and now has an exact frozen-job addendum.

Completed: accepted-log/ledger reconciliation; true geometric-distance replay; cache/shadow reconstruction; static versus feedback calibration; independent-identity development split; width/engagement/shift checks; library-only transfer configs; unchanged periodic scalar calibration; initial Q4 B signal table; power scenarios; reference arithmetic checks. All compute CPU-only inside the allocated CPU set; all writes in this directory.

Remaining on resume: actual new-method B pilot/recording analysis, fuller Q3/Q4 shadow/placement uncertainty, and frozen A measurements. ANALYSIS/REPORT await those results. Open questions are in HANDBACK/PLAN; none requires stopping sol's implementation work. No new-variant result or raw A data has been read.
''')
    repro='''# Reproduction

From the repository root, prefix every invocation with:

```bash
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Run these scripts in this directory in order: `inventory.py`, `extract.py`, `replay_study.py`, `robust_study.py`, `static_study.py`, `signal_quality.py`, `periodic_and_power.py`, `selfcheck.py`, `freeze.py`, `validation_selection.py`, `write_docs.py`, `audit.py`. Early `early.py` was run once before calibration and must not be rerun over the final handoff. `freeze.py` refuses a changed payload at an existing freeze timestamp. No command starts a rollout or policy model. Imported R11 library code is read-only; the local audit hook fences writes and disallows A/non-library inputs.

The exploratory off-path replay preceded the robust multi-path scalar fit. Keep both evidence tables. Development holdout results were used for global width selection and are not a final independent validation claim. Repeating extraction uses the immutable completed root summaries and cached numerical traces; source hashes in the trace metadata and audit remain authoritative.
'''
    (HERE/'RUN.md').write_text(repro)
    provenance('docs')
    print('Rendered SPEC, PREDICTION, OFFLINE, PLAN, REQUESTS addendum, HANDBACK, STATUS, RUN, TEST_DESCRIPTIVE')

if __name__=='__main__': main()
