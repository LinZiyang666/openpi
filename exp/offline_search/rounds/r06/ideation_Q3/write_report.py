from analyze import *
from summarize import interval

S=json.loads((OUT/'summary.json').read_text())
E=json.loads((OUT/'event_probe.json').read_text())
V=json.loads((OUT/'verification.json').read_text())
parts=[]
def add(s):parts.append(s.strip())

add('''# R6 Q3: where a MISS is worth its cost

**Conclusion: preserve the first useful reactive intervention; there is no validated portable rule for buying the same SR with fewer calls yet.** The most localized evidence points to the first no-progress rescue, well before the end of π0.5 LIBERO-10 episodes. Coverage and disagreement identify cache trouble, but frequently identify it **after** the existing guard has already intervened. A preventive call near an uncertain library action transition is a testable hypothesis, not an established improvement. Do not remove late guard calls merely because the episodes receiving them often fail.

This report delivers offline measurements and an eight-arm **design only**. Nothing was installed or launched. No server, worker, chain, port, tmux session, remote machine, git operation, or review-test directory was used. Writes were confined to this directory and `/tmp/r6_q3/`.

## 1. What was measured and how to reproduce it

Read first: [R6 brief](../FINDINGS.md), [R5 analysis](../../r05/ANALYSIS.md), [R4 analysis](../../r04/ANALYSIS.md), and [harness contract](../../../harness/README.md). The complete numeric outputs are [TABLES.md](TABLES.md), [summary.json](summary.json), `calibration_*.json`, `causal_*.json`, `paired_*.json`, and [event_probe.json](event_probe.json).

Three evidence levels remain separate:

1. **Randomized intervention evidence:** reloaded K5 raw accepted attempts with the unchanged `rounds/r04/k5_rand/estimate.py:load_arm`, checked assignments, continuity, served heads, compliance and complementary replicas, and recomputed treatment effects. Four arms: `r04_k5/r4k5_p_l10_g{50,500}_{r1,r2}`. These are guard-only π0.5-l10, five executed controls per decision, one randomized first/third guard opportunity. They do not identify preventive calls, committed C10 calls, GR00T, or Spatial.
2. **Paired configuration and temporal evidence:** loaded both accepted trajectories for all eight R6 A/B cells, 16 completed arms, 500 `(task, init)` pairs/cell. Exact source mapping is `analyze.py:PAIRS`; each `paired_*.json` also names its A/B arms and raw files. π0.5 A comes from `r05_ptail`, except Spatial-500 from `r04_blind/r4b3_p_sp_500_tail1uc`; B is `r05_q1/r5q1_c10_*`. GR00T A is `r05_x/r5x_g_*_tail1u`, B is the **periodic** `r05_q2/*_G10`. No running R6 arm was read or extrapolated.
3. **Offline retrieval proxies:** 89,129 rows / 2,199 library episodes over eight model/suite/scale cells. Every row retrieves only other episodes of its own task, using the actually served A fit, top-16 kernel and kref 5/8. This is **retrieval LOEO conditional on the full-library-fitted PCA/metric**, not an inductive refit leaving the episode out of representation learning. It is optimistic as a deployment calibration. Error and neighbour disagreement use the deployed library's own action standard deviations, active recorded dimensions 0–6, and five actually executed controls. The ten-control target concatenates the held-out row's executed head with its contiguous successor's executed head; it does not treat an unused policy tail as ground truth. Terminal/missing successor targets are excluded from that metric. The old fits themselves retain their historical normalization/provenance.

Statistics: 4,000 percentile bootstrap draws, seed `20260928`. A/B resamples whole `(task, init)` pairs. K5 uses the original Horvitz–Thompson estimator at propensity .5, keeping both replicas in each init cluster, including unexposed episodes in ITT. Feature strata use pre-intervention values at exposed landmarks. LOEO error is an episode-macro mean; AUC is the mean of within-episode AUCs among episodes with both high-error and low-error rows, with episodes resampled. A high-error row exceeds its task's library-LOEO error p90. AUC sample counts are in `summary.json`; they are smaller than the library episode count. All intervals are conditional on these tasks/runs and pointwise unless labeled otherwise; no new-task guarantee follows. The K5 18-interaction sensitivity uses a Bonferroni normal bound with bootstrap cluster SE, not an exact simultaneous guarantee. No classifier was trained to call a B-only success a beneficial individual MISS.

Commands, from `/home/weiland/projects/openpi` (one Python process at a time is sufficient):

```bash
mkdir -p /tmp/r6_q3
for stage in calibrate causal paired; do
  taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/analyze.py "$stage"
done
for script in summarize event_probe verify write_report; do
  taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python "exp/offline_search/rounds/r06/ideation_Q3/$script.py"
done
```

Raw parsing caches and per-row LOEO arrays live in `/tmp/r6_q3/`; all input stores, fits and historical rounds remain read-only. The analysis scripts never call a fit/server launcher. The same commands with empty scratch recreate those caches. Logs of this run are `/tmp/r6_q3/{calibrate,causal,paired_final,summary_final,event_probe,verify}.log`.
''')
add(f'''Audit: the 16 A/B arms contain **{V['accepted_decisions']:,} accepted decisions**; all N/V/M counts match their cost ledgers. K5 adds 125,791 decisions, giving 469,266 raw decisions over 20 arms / 10,000 accepted episodes. All examined arms have completion markers. Reconstructed {V['cache_reconstruction_samples']:,} sampled cached heads from logged library rows/weights; maximum absolute error was **{V['max_cache_reconstruction_error']:.3g}**. Thus the B policy-minus-cache action differences below use a checked reconstruction. K5 ITT point estimates exactly equal the coordinator's saved estimates. [verification.json](verification.json) retains the audit. K5 intervals differ slightly from R4/R5 where this analysis uses a different seed and 4,000 rather than 2,000 draws.''')

add('''## 2. Findings

### 2.1 What one randomized call identifies

| Library | CALL−CACHE ITT ΔSR, pp [95%] | Δ total MISS/episode [95%] | First landmark ΔSR, pp [95%] | Third landmark ΔSR, pp [95%] |
|---|---|---|---|---|''')
for n in (50,500):
 c=S['causal'][str(n)];r=c['ITT'];f=c['subgroups']['landmark1'];t=c['subgroups']['landmark3']
 def effectfmt(x,m,scale=1):return interval(dict(mean=x['delta'][m],ci95=x['ci95'][m]),scale)
 parts.append(f"| {n} | {effectfmt(r,'Y',100)} | {effectfmt(r,'M')} | {effectfmt(f,'Y',100)} | {effectfmt(t,'Y',100)} |")
add('''At 500, retaining the randomized opportunity improves episode success and shortens the trajectory: ΔN = −1.582 requests [−2.990, −0.234]. Its **net downstream call-count interval includes zero**, so dividing ΔSR by ΔM would be unstable. +3.4 pp is the value of assigning CALL at one selected landmark under this controller, including episodes that never reach it; it is not +3.4 pp for every guard call. The first-versus-third effect difference is +3.08 pp [−3.25, +9.20], so “first is better than third” is **not** statistically established. The first-landmark estimate nevertheless provides the clearest positive timing evidence. At 50, the total effect is unresolved, not proof that calls are unnecessary.

K5 has **zero pre-guard exposures**. It cannot estimate a MISS before guard evidence, because such decisions were never randomized. It also lacks the joint visual nearest-neighbour distance: logged `dnn` is raw valid robot-state nearest-neighbour distance. The joint visual/state coverage signal is available in A's logs but cannot be recovered from K5's normalized scores or missing keys. Do not rename `dnn` “visual coverage.”

The following are high-minus-low differences in CALL−CACHE effect, in percentage points. They test treatment-effect heterogeneity, rather than whether a feature predicts eventual failure. Full stratum sample/treatment counts, effects and intervals are in `causal_*.json` and `TABLES.md`.

| Online signal / split | Library 50 interaction [95%] | Library 500 interaction [95%] |
|---|---|---|''')
names={'state_coverage':'State NN distance ≥ task LOEO p90','disagreement':'Top-five action disagreement ≥ task LOEO p90','phase':'Matched progress ≥ .5','event_soon':'Library action-change event within current/next transition','mode_switch_soon':'Recorded gripper-command mode switch within that window','stuck':'stuck_n ≥ 2','stall':'noprog_n ≥ 2','overtime':'elapsed / library median duration ≥ 1','since_miss':'since MISS / library median duration ≥ .1 (previous MISS exists)'}
for key,label in names.items():
 vals=[]
 for n in ('50','500'):
  v=S['causal'][n]['multiplicity'][key];vals.append(interval(dict(mean=v['interaction'],ci95=v['interaction_ci95']),100))
 parts.append(f"| {label} | {' | '.join(vals)} |")
add('''A generic action event is a consecutive library executed-head change above its task's p90, in library-normalized action units. It does **not** mean verified grasp/contact/release. The gripper-mode row is a retrospective annotation using the recorded schema's dimension 6, without assigning “open” a common sign across models; it is excluded from the proposed portable rule. Contact is not observed by these logs. The splits .5, 2, 1 and .1 above are exploratory diagnostic conventions, not learned optimal thresholds.

The event interaction at 50 is nominally positive, but its family-18 interval is **[−10.70, +60.57] pp**; every feature interaction includes zero after this sensitivity adjustment. State coverage and stuck effects even change point sign across library scale. “Long since last MISS” has positive interaction point estimates, but intervals remain broad; it does not establish an optimal timer. The first landmark has no preceding MISS and is omitted from that timer contrast. These data do not support a suppression table. The earlier stricter [R5 Q3 audit](../../r05/q3_callvalue/HANDBACK.md) likewise found no supported saving under its 1 pp success-loss constraint; its task-level g500 interval also includes zero. This analysis does not relax that deployment decision.

### 2.2 Configuration gain per call, and where B-only trajectories separate

| Cell | A SR @ IR | B SR @ IR | B−A pp [95%] | B-only / A-only | B calls/episode | ΔSR pp / added call [95%] |
|---|---|---|---|---|---|---|''')
for cell,r in S['cells'].items():
 p=r['paired'];rr=p['configuration_gain_per_added_call'];q=dict(mean=rr['value'],ci95=rr['ci95'])
 parts.append(f"| {cell} | {p['A_SR']:.3f} @ {p['A_IR']:.5f} | {p['B_SR']:.3f} @ {p['B_IR']:.5f} | {interval(p['contrast'],100)} | {p['B_only']}/{p['A_only']} | {p['B_M_per_episode']:.3f} | {interval(q,100)} |")
add('''IR is recomputed as `.152 v + .848 m` for π0.5 and `.148 v + .852 m` for GR00T. The last column is **configuration accounting**: paired success gain divided by total additional calls, bootstrapping numerator and denominator together. It is not a local causal treatment effect. It includes downstream state changes, durations, call repetitions and both positive and negative paired outcomes. The 500-library GR00T periodic calls buy no resolved benefit; Spatial-500 π0.5 also has no observed net gain. That does not identify which particular calls can safely be removed.

For trajectory localization, align A and B by executed five-control slots. “State separation” is the first normalized state difference exceeding the task library's p95 one-transition motion; “action separation” exceeds its LOEO head-error p90. These are coarse library-scaled thresholds, not bitwise common prefixes or verified failure times.

| π0.5 cell | B-only with ≥1 MISS / all B-only | State separation before/at first MISS | Median first MISS slot / matched phase | First MISS reason among B-only with a call |
|---|---|---|---|---|''')
for cell in [c for c in PAIRS if c.startswith('pi05')]:
 r=S['cells'][cell]['B_only_timing'];parts.append(f"| {cell} | {r['with_call']}/{r['B_only']} | {r['state_divergence_before_or_at']} | {r['first_miss_steps'][1]:g} / {r['first_phase'][1]:.3f} | {r['first_reasons']} |")
add('''Reason 1 = stuck, 2 = terminal-row/closed-command, 3 = overtime+lag, 4 = no-progress. At l10-500, all 42 B-only trajectories with a MISS first call on **no-progress**, at median matched phase .198; 41/42 first call before the coarse state-separation threshold. At l10-50, 80/85 first call on no-progress. This is stronger evidence for protecting the first stalled-progress rescue than for spending calls mainly at the final release. It is still a paired-system association: B-only successes with **no MISS** exist (3/88 and 1/43 l10, 2/45 Spatial-50), demonstrating run variability and preventing the label “B-only = rescued by this MISS.” Small state differences below our threshold can precede the call.

Checked examples, selected by lowest `(task, init)` among B-only pairs whose first call precedes coarse state separation:

- **π0.5 l10-50, task 0/init 0:** first no-progress MISS at slot 12, policy/cache head RMS difference .5482; action separation at 12, state separation at 13. B executes the policy at 12 and its tail at 13, returns to cache at 14; retrieved phase changes .190 → .259 → .296 → .340 at slots 12/14/16/18. B eventually succeeds with four calls, A fails. This observes a changed action followed by resumed matched progress, not proven successful contact.
- **π0.5 l10-500, task 0/init 2:** first no-progress MISS at 26, policy/cache RMS .2949; B returns to cache at 28, matched phase .508 → .621. Coarse action/state separation occurs only at 38/40; there are four B calls. Attributing the final success solely to the call at 26 would overstate localization.
- **π0.5 Spatial-50, task 0/init 2:** one MISS at slot 6 and its tail at 7; policy/cache RMS .4804, phase .333 → .563 at the return to cache at 8. Coarse action/state separation is at 8/9. This is a cleaner single-call temporal example, still not a randomized intervention.

For GR00T G10, **every first MISS is at slot 0**, including all B-only successes; later calls recur every fourth anchor. The paired runs therefore cannot locate which later MISS helped. They compare an entire schedule beginning at initialization, not a guard-triggered rescue.

### 2.3 Guard outcomes: repeated failure calls are a diagnostic, not a saving guarantee

For π0.5 l10-50/500, total calls are 3,747/2,869; **2,080/1,362** occur in ultimately failed episodes. Median calls per successful episode are **3/3**, versus **26/21** per failed episode. At l10-50, no-progress is the selected reason for 2,951 calls and touches 449 episodes, 364 eventually successful (81.1%, bootstrap [77.3%,84.9%]); stuck touches 84 episodes, 31 successes (36.9%, [26.2%,47.6%]); overtime touches 32, one success. At 500, no-progress accounts for 2,605 calls, with 395/462 touched episodes successful (85.5%, [82.3%,88.7%]); stuck yields 9/38 (23.7%, [10.5%,36.8%]); overtime 0/3, far too small for an inference.

Episodes can appear under multiple reasons; reasons are the lowest active flag at each call. The numbers above are observational, with severity confounding. Long failed episodes also provide more opportunities to call. They do not show that late calls cause failure or can be removed safely. Among calls with a subsequent vision anchor, another policy call follows 43.6%/26.4% of l10-50/500 no-progress calls, versus 94.3%/81.6% of stuck calls. This locates repeated rescue clusters, but supplies no counterfactual for a longer burst or a suppression rule. Spatial-500 is especially different: 264/529 calls are the last decision of the episode, and the configuration has zero net SR gain.

### 2.4 Which offline signals predict deviation, and do those signals predict useful policy placement?

| Cell | LOEO head RMS [95%] | Executed ten-control RMS [95%] | Coverage AUC [95%] | Disagreement AUC [95%] |
|---|---|---|---|---|''')
for cell,r in S['cells'].items():
 q=r['loeo'];parts.append(f"| {cell} | {interval(q['error'])} | {interval(q['error10'])} | {interval(q['distance_auc'])} | {interval(q['disagreement_auc'])} |")
add('''Disagreement predicts large action deviation better than chance in all eight cells. Generic event AUC ranges .559–.626; inferred-stall AUC is only .500–.542 (intervals and eligible episode counts are in `TABLES.md`/`summary.json`). None of these labels measures a policy advantage. A demonstration can differ from another valid action, and the VLA can make the same error as the cache.

To check against actual closed-loop benefit, take one **A** vision anchor per episode: the first at/after one quarter of the task's library median duration, never a fraction of its eventual rollout length. Regressing is unnecessary: compare B−A success within high/low library-calibrated feature groups. All four π0.5 comparisons retain 500 episodes at this landmark.

- l10-50: high joint coverage-distance group = 362 episodes, B−A **+16.02 pp** versus **+2.90 pp** elsewhere; interaction **+13.12 [4.34,21.53] pp**. High state-distance group = 39, interaction **+31.05 [15.06,47.09] pp**. High disagreement group = 62, interaction **+2.42 [−9.08,14.42] pp**.
- l10-500: high joint-distance group = 57, B−A **+36.84 pp** versus **−0.45 pp** elsewhere; interaction **+37.29 [24.05,50.45] pp**. High disagreement group = 45, interaction **+22.69 [10.13,36.98] pp**. The state-distance high group has only 19 episodes, insufficient to rely on despite its positive point estimate.
- These relations do not transfer uniformly: Spatial-50 coverage interaction **+1.41 [−10.82,12.70] pp**; GR00T l10-50 **−0.17 [−10.05,9.77] pp**. Full eight-feature × eight-cell results are in `TABLES.md`. Extreme phase or motion effects from groups of 1–17 episodes are unsupported; do not use them as thresholds. These exploratory A/B intervals are not multiplicity-adjusted.

This is an association between **A's later state** and the value of an entire different controller. B may already have intervened before that A landmark. Thus it is not a causal estimate of calling at the A landmark. The measured temporal reach below exposes this distinction.

**Closed-loop falsification across existing arms:** beyond the 20 newly parsed arms, checked the 12 external controller predictions in R5-B `log_results.json` and the 12 arms in the six prospective B1 solve/hand comparisons reported in [R5 §7](../../r05/ANALYSIS.md). This gives 44 distinct completed arm results in the evidence review; the older 24 are inherited audited results, not freshly parsed raw data. The R5 failure-risk model has held-task AUC .617/.684 at 50/500, yet predicts the l10-50 phase controller safer than tail (.202 versus .248 failure) while their observed failures are .300 versus .194. Across its 3/9 external controllers, rank correlations are only .500/.644 (p=.667/.061); these small collections do not validate general ranking. All six prospective B1 changes reduce offline action and successor error, but only two improve SR; the pooled change is −0.53 pp [−1.63,+0.57]. The new data provide another warning: GR00T Spatial LOEO head error rises .348 → .349 from 50 to 500 while A SR rises .868 → .964. **No offline loss, AUC or alarm count in this report is accepted as an SR selector.**
''')

add('''### 2.5 Frozen-path timing rejects a simple universal novelty alarm

Let D and U be joint NN distance and top-five action disagreement divided by their task's LOEO p90. Calibrate a library-wide threshold τ as the p90 (`higher` order statistic) of successful-library-episode maxima of max(D,U), on noninitial two-quantum anchors. The table compares OR: max(D,U)>τ, with conservative AND: min(D,U)>max(1,τ), using the **same** threshold. No outcome labels from evaluated A/B runs enter τ. This is conditional library calibration, not a distribution-free false-alarm guarantee.

| Cell | τ | OR flagged A episodes /500 | AND flagged /500 | Successful A flagged by AND | AND alarm before B first MISS |
|---|---|---|---|---|---|''')
for cell,r in S['cells'].items():
 q=r['replay'];parts.append(f"| {cell} | {r['pooled_calibration']['threshold']:.4f} | {q['flagged']} | {q['both_flagged']} | {q['both_success_flagged']} | {q['both_before_B_first_miss']} |")
add('''The OR screen's Spatial-50 alarm rates are 498/500 and 500/500, including 417 and 434 A successes. Nominal library calibration therefore does not preserve its alarm rate on evaluated trajectories. The conservative AND screen is more selective, but at π0.5 l10-500 **none** of its first alarms precedes B's first MISS. At l10-50 only 10 do. It is mainly a late trouble detector here. A separate exploratory calibration using a min-score's own episode-max quantile was also examined; lowering the threshold increased alert reach, so the final conservative screen keeps the common max-score threshold. This is an exploratory design sequence, not a preregistered prospective result.

For a genuinely earlier probe, an **upcoming action event AND disagreement above task p90**, without requiring distance to be extreme, has the following frozen-A-path reach. An event is detectable from stored future library actions; future rollout actions are never inspected online.

| π0.5 cell | Flagged A episodes | A successes flagged | First alarm before B first MISS | B-only pairs in that early subset |
|---|---|---|---|---|''')
for cell in [c for c in PAIRS if c.startswith('pi05')]:
 r=E[cell];parts.append(f"| {cell} | {r['flagged']} | {r['flagged_A_success']} | {r['before_B_first_miss']} | {r['B_only_before']} |")
add('''This supports collecting earlier intervention evidence, **not** a forecast of additional successes. In particular, many successful A trajectories would receive unnecessary-looking calls. At 500 the early subset is only 96/500 in l10; a one-landmark study may be underpowered. The before-B comparison is between separate runs, not a replay of B's guard on the exact A state. GR00T's zero “before B” counts are tautological because its current B calls at initialization. Freeze a GR00T guard baseline before using this definition there.

## 3. Exact proposed placement methods and portable calibration

### Recommended disposition

Keep the current validated A/B choice with the coordinator/Q2. Do not deploy a new suppression gate, clock, extra burst, or a claim of SR preservation from this analysis. For Q3, **test one preventive event call while retaining current reactive rescues**. Treat the conservative coverage/disagreement conjunction as a diagnostic or a separate later reactive-candidate study; its timing does not justify calling it preventive. Budget selection remains Q2's problem.

The new module below is benchmark/robot-independent: it needs a task-conditioned cache metric, recorded active action channels, control timestamps and policy chunk lengths. It does not use task names beyond the retrieval partition, camera count, state width, gripper index/sign, contact labels, or a fixed episode horizon. Active channels must come from the robot adapter; the recorded 32-wide arrays cannot be used blindly. The existing LIBERO guard is a **reference controller input G**, not a newly generalized guard. Porting stock terminal/closed-gripper semantics remains necessary if the application chooses to port that guard; the placement module itself can wrap another benchmark's supervisor, or G=false. This report does not claim the entire inherited AWM/guard stack is already portable.

### Library calibration shared by all candidates

1. Use only the deployed library to set action scales σ_d = std of executed actions in each declared active channel. Drop constant channels; if none remain, disable the optional probe. Recompute the deployed metric on the library when changing benchmark/model; this study's audits intentionally reused historical fits. Require at least two episodes in a retrieval task, otherwise disable the optional probe for that task. No large-library statistic is transferred into a 50-library threshold.
2. Let Δ be the observed control count per recorded decision (adapter execution metadata, **5 in these runs**). Let H = Δ·min(2, floor(H_policy/Δ), floor(H_library/Δ)). If either horizon is below Δ, the controller contract must be repaired before using this rule. Thus H is **10 here**, including GR00T with a 16-control available chunk. Multiplier 2 is a conservative design constant retaining the validated parent commitment; it is not estimated optimal by LOEO. It introduces no fixed physical duration on another robot.
3. At every held-out library row, retrieve the top k=min(16, other-episode candidate count) with the deployed kernel. Compute U as the mean pairwise RMS action distance among its top min(5,k) executed heads, normalized by σ. k=16 and the five-neighbour statistic are fixed design choices matching the existing cache/causal logs, not fitted SR-optimal choices. Let u90_t be the p90 over that task's LOEO U values, with NumPy's default linear quantile. If u90_t=0 or there are fewer than two neighbours, disable the optional probe for that task.
4. For each consecutive same-episode pair of library executed heads, compute J_r = RMS((head[r+1]−head[r])/σ). Let j90_t be its task's p90, linear quantile. Define an action event on edge r→r+1 iff J_r>j90_t. A terminal row has no future event; do not wrap to another episode. No grasp/contact interpretation is assigned.
5. Optional diagnostic only: D is joint metric distance divided by its task's LOEO p90. The conservative diagnostic threshold τ and AND screen are exactly §2.5 (successful episodes; noninitial anchors every H/Δ library decisions; `higher` p90 of pooled episode maxima; require min(D,U/u90)>max(1,τ)). Successful episodes calibrate the alarm envelope, while retrieval remains from the deployed bank including its failures. If success labels are unavailable, use all library episodes and label the calibration accordingly. This diagnostic does not determine the event probe's eligibility.

All quantile levels (.90), neighbour counts (16/5), commitment multiplier (2), and one-opportunity limit below are stated design constants. Library values in `calibration_*.json` are the actual measured thresholds. They are not tuned against evaluated SR; neither a percentile level nor a bootstrap interval proves an optimal decision boundary. Numerical divisions in the analysis use a 1e−8 floor only for pathological variance, not as a physical control threshold.

### E1: preventive event probe, one committed policy chunk

State per episode: `probe_consumed=false`, `ever_policy=false`, actual last-policy timestamp, actual execution/tail provenance; reset all of it per episode. At each real vision anchor:

1. Finish any previously selected H-control chunk before a new decision, except the application's ordinary termination/safety mechanism. Do not fabricate a vision key at a blind step. Evaluate the reference guard G first. If G requests policy, execute the normal committed rescue, set `ever_policy=true`, and permanently retire the preventive opportunity for this episode. This preserves the first reactive MISS and all later reactive MISSes.
2. Otherwise, never probe at initialization, when `ever_policy` or `probe_consumed` is true, or during an unexecuted policy/cache tail. Requiring no previous MISS is the strict preventive/timer rule for this pilot: there is no learned cooldown and no attempt to infer one from K5's noisy .1-duration split.
3. At the current retrieval, compute U from top-five heads. Let r be its top-one matched row. Set E=true iff there is an action event on an edge starting at r or at its first valid successor. This two-edge window follows H=2Δ and is **future library data**, not future robot data. For a different derived H, inspect exactly H/Δ such edges.
4. The first anchor with E=true and U>u90_t is the sole eligible opportunity. Mark it consumed even if the randomized study chooses CACHE. The deterministic proposed E1 controller chooses CALL there. Execute one complete policy chunk of length H (current execution plus its remaining tail), with no forced additional model calls inside it. Policy inference itself is unchanged.
5. At the end of that chunk, require a real vision anchor, refresh retrieval using actual executed history and correct policy-tail provenance, and evaluate G. If G is clear, return to the cache immediately for the normal H-control commitment. If G fires, the ordinary reactive rescue retains control. Do not keep policy control just because coverage remains poor, and do not extend the burst based on an unvalidated error proxy. No extra preventive opportunity remains this episode.

This explicitly fixes policy duration and return behaviour so the experiment changes placement. Current C10 already realizes nonterminal policy tails; extending control further is not justified by the data here. Application safety/termination still preempts ordinary commitments; there is no new robot-specific safety rule in this design.

### U1: placement control

Replace step 3's event condition with one sampled noninitial anchor index. At episode reset, sample uniformly a successful library episode of that task, then uniformly one of its noninitial H-spaced anchor indices. Freeze this index before interacting with the environment. If the evaluation episode terminates or G first calls policy before that anchor, it is unexposed. At that anchor require G=false and consume the sole opportunity; CALL/CACHE, H, and return handling are identical to E1. If no successful eligible library episode exists, disable this optional probe. This is a one-opportunity timing control, **not an assertion of equal realized call counts**; exposure and downstream costs must be measured. It tests whether action events/disagreement target more useful opportunities than a task-duration-derived timer.

On paper, E1 spends at most one additional direct call before the first guard, and could reduce IR only if it averts enough later calls or shortens the trajectory. U1 has the same maximum direct opportunity, but different exposure. Neither has a proved saving. The AND novelty detector often arrives after the first guard and cannot supply the same preventive hypothesis. Current no-progress guards react to stalled matched progress; E1 attempts to intervene before a library action change while neighbours disagree. Semantic release/contact events and periodic calls after a fixed number of robot steps are deliberately unnecessary to this module.

## 4. Closed-loop test plan: exactly eight proposed arms, no launches

Use the two π0.5-l10 scales, holding the complete current C10 parent fixed. These are the cells with causal K5 history and the strongest paired localization. This first study does **not** establish GR00T/Spatial portability; offline cross-cell checks already show that extrapolation is unsafe. [arms_proposed.json](arms_proposed.json) is a design manifest, not an executable installed method.

Shared parent: `exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge`; `base_kwargs={lib: current|big, kref:5|8, serving:anchor_tail, budget:1, gates:budget_only}`, `progress_guard=noprog_span`, `events=none`, `stuck_guard=vision_confirmed`, `policy_tail_gate=lifecycle`, `monitor=off`; full stage-1, `guard_only`, blind cache/policy-tail support, normal policy inference, five-control client requests and H=10 commitment. Reuse the corresponding exact R5 Q1 fit. Match all 500 task/init pairs per arm. No new guard threshold, library fit, action synthesis or execution length is changed.

For each scale and probe type, two replicas use complementary CALL/CACHE assignments at the first eligible opportunity. Propensity=.5; unexposed episodes stay in ITT. Use seed 20260928 with the exact SHA256 scheme in the manifest; keep landmark sampling independent of treatment. All subsequent behaviour is the unchanged C10 parent. CACHE-assigned histories therefore implement baseline B; CALL-assigned histories implement the single-probe controller, up to run variability.

| Arm | Library / probe / replica | Exact paired comparison answered | Qualitative prediction (hypothesis) |
|---|---|---|---|
| r6q3_p_l10_50_event_r1 | 50 / E1 / 1 | Complement of next row; within-init CALL−CACHE value of the first uncertain event | Plausible positive SR, uncertain net IR |
| r6q3_p_l10_50_event_r2 | 50 / E1 / 2 | Same eligibility and controller, opposite treatment coin | Same; not an independent second opportunity |
| r6q3_p_l10_50_uniform_r1 | 50 / U1 / 1 | Complement of next row; E1-versus-U1 treatment-effect contrast | Weaker concentration of benefit than E1, unproved |
| r6q3_p_l10_50_uniform_r2 | 50 / U1 / 2 | Same sampled index, opposite treatment coin | More calls may be spent on easy trajectories |
| r6q3_p_l10_500_event_r1 | 500 / E1 / 1 | Complement of next row; does prevention avert costly later rescues? | Small or unresolved SR gain; cost may rise |
| r6q3_p_l10_500_event_r2 | 500 / E1 / 2 | Same event rule, opposite treatment coin | Expect sparse exposure / wide intervals |
| r6q3_p_l10_500_uniform_r1 | 500 / U1 / 1 | Complement of next row; opportunity cost of a timer with a strong bank | Little benefit, likely extra cost |
| r6q3_p_l10_500_uniform_r2 | 500 / U1 / 2 | Same sampled index, opposite treatment coin | Same; test rather than assume |

These are qualitative predictions, not simulated or invented SR numbers. Log actual pre-treatment D, U, E, matched phase, guard flags, time since prior MISS, candidate trigger reason, assignment, exposure, actual chunk duration, policy/cache return, and whole-episode Y/N/V/M. Record raw joint distance explicitly; K5 cannot answer that feature question because it did not retain it. Define preventive as before any parent guard/MISS; “before irreversible physical failure” remains unobserved without additional state annotations.

Primary analysis: paired-init clustered ITT on ΔSR, ΔN/V/M and actual decision-weighted IR, plus conditional-on-exposure HT estimates as secondary. Bootstrap replicas together; keep unexposed episodes and audit accepted attempts as in K5. Compare E1 and U1 by difference in their randomized effects with common init resampling; do not compare raw success among their different exposed populations. Do not divide by net additional calls if the denominator interval includes zero. Declare the two event-probe ITT contrasts primary and correct their family; timer and feature heterogeneity are secondary. Do not select event thresholds from these evaluation outcomes.

A proposed **design criterion**, not a measured result: adopt only if the upper confidence bound on IR change is below zero and the lower bound on ΔSR exceeds −.01, against paired/repeated B references, with no concentrated task failure concealed by averaging. The .01 noninferiority margin requires owner/coordinator acceptance before an experiment; it is not retroactively applied as evidence. With only 500 initializations and sparse exposure, these arms may be insufficient to establish that criterion. A null/wide result means no promotion, not preserved SR. This offline report asks for no approval and launches nothing.

## 5. Risks, limitations and falsification

- **No proven placement predictor:** all K5 feature interactions fail the family-wise sensitivity; the lone nominal event result motivated an exploratory hypothesis only. It is outcome-informed research, not a demo-only discovered law. No direct causal evidence exists for an event before a guard under committed control.
- **Coverage may be late, disagreement may reflect valid alternatives:** a good AUC against recorded actions is insufficient. E1 is falsified as a useful placement improvement if its randomized effect is no better than U1 at measured cost, if policy repeats the same error, or if SR gain is absent while IR rises. A positive event result only at one task/scale also falsifies a broad portability claim.
- **False-alarm transfer already fails:** near-universal OR alarms on Spatial-50 and successful A episodes flagged by E1 are measured counterexamples to interpreting library quantiles as online risk probabilities. The conjunction's late timing falsifies its use as a universal preventive trigger. Do not retune thresholds on those evaluation labels and then call the result validated.
- **Local matched phase is not true progress:** neighbour episode switches, repeated motions and retries can move phase backward; event rows are action changes, not observed physical contact. A wrong matched trajectory can place the probe before an irrelevant event. Actual contact/grasp/release benefit remains unmeasured.
- **Confounding and uncertainty:** A/B success discordance, initial stochastic differences, tasks with few failures, and severity-dependent repeated calls limit episode-level attribution. The first/third K5 contrast does not prove that the first call is more valuable. Low success after stuck/overtime does not license suppression; delaying no-progress already hurt SR in R4/R5.
- **Calibration limitations:** the representation includes held-out episodes, and row p90s plus episode maxima are finite-library empirical thresholds, not conformal guarantees. Only a few episodes/task exist at scale 50; task pooling stabilizes the diagnostic envelope but does not ensure calibration on new tasks. Strict refit/crossfit calibration and a new benchmark remain required before a transfer claim.
- **Library-scale comparisons also change collection:** R5 Q4's [provenance audit](../../r05/q4_growth/HANDBACK.md) found the π0.5 current libraries are not subsets of bpool_cs. The 50-versus-500 contrasts here therefore change collection as well as size; neither proxy changes nor treatment-effect differences isolate a pure size effect.
- **Cost and control are coupled:** one early call changes all later states, durations, vision anchors and guards. Frozen-path reach supplies neither IR nor SR. Committing longer or returning later would introduce another intervention; those changes are held fixed here. Extra computation for diagnostics is also not credited as free measured wall-clock performance.
- **Test scope:** the eight proposed arms isolate two placement hypotheses at two π0.5 library scales. They do not validate GR00T guard semantics, another robot, or another benchmark, and do not solve the library-quality or total-call-budget questions owned by the other R6 agents.

The defensible current answer is therefore narrow: **protect the first no-progress rescue, complete its existing policy chunk, and hand back after a real observation when the guard clears.** Earlier event-based placement is worth a controlled probe; current evidence cannot claim that it keeps SR while lowering IR.
''')

arms=[]
for scale in (50,500):
 for probe in ('event','uniform'):
  for rep in (1,2):
   arm=f'r6q3_p_l10_{scale}_{probe}_r{rep}'
   arms.append(dict(arm=arm,design_only=True,model='pi05',suite='libero_10',library='current' if scale==50 else 'big',parent_run='r05_q1',parent_arm=f'r5q1_c10_p_l10_{scale}',fit=f'/home/weiland/trace_runs/os_closed_loop/r05_q1/fits/r5q1_c10_p_l10_{scale}.pkl',method='exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge',kwargs=dict(base_kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,serving='anchor_tail',budget=1,gates='budget_only'),progress_guard='noprog_span',events='none',stuck_guard='vision_confirmed',policy_tail_gate='lifecycle',monitor='off'),stage1_mode='full',judge='guard_only',cache_commit_controls=10,policy_commit_controls=10,client_request_controls=5,probe=probe,probe_definition='REPORT.md section 3 E1' if probe=='event' else 'REPORT.md section 3 U1',max_opportunities=1,pre_first_guard_only=True,propensity=.5,seed=SEED,replicate=rep,treatment_assignment='digest=SHA256(UTF8("r6q3:20260928:{scale}:{probe}:{task}:{init}:treatment")); CALL iff ((int.from_bytes(digest,"big") & 1) XOR (replicate==2)) == 1',uniform_landmark_sampling='Use independent SHA256(...+":episode") and SHA256(...+":anchor"); sort successful eligible library episodes by stored episode ID, choose floor(U_episode*n), then floor(U_anchor*m) among its noninitial H-spaced anchors. U=(int.from_bytes(digest,"big")+.5)/2**256. Identical across complementary replicas.',pairs='all task/init pairs of the corresponding completed 500-episode R5 parent; exact pairing retained',prediction='hypothesis only; see REPORT.md section 4'))
dump('arms_proposed.json',dict(status='DESIGN_ONLY_NOT_EXECUTABLE',arms=arms))
report='\n\n'.join(parts)+'\n'
# Keep generated table rows contiguous; paragraphs remain separated.
while '|\n\n|' in report:report=report.replace('|\n\n|','|\n|')
appendix=OUT/'DATA_WISHLIST.md'
if appendix.exists():report+='\n'+appendix.read_text().rstrip()+'\n'
(OUT/'REPORT.md').write_text(report)
print('Wrote REPORT.md and eight-arm design manifest')
