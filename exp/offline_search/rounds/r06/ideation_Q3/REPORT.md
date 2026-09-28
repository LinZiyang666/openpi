# R6 Q3: where a MISS is worth its cost

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

Audit: the 16 A/B arms contain **343,475 accepted decisions**; all N/V/M counts match their cost ledgers. K5 adds 125,791 decisions, giving 469,266 raw decisions over 20 arms / 10,000 accepted episodes. All examined arms have completion markers. Reconstructed 1,024 sampled cached heads from logged library rows/weights; maximum absolute error was **1.91e-07**. Thus the B policy-minus-cache action differences below use a checked reconstruction. K5 ITT point estimates exactly equal the coordinator's saved estimates. [verification.json](verification.json) retains the audit. K5 intervals differ slightly from R4/R5 where this analysis uses a different seed and 4,000 rather than 2,000 draws.

## 2. Findings

### 2.1 What one randomized call identifies

| Library | CALL−CACHE ITT ΔSR, pp [95%] | Δ total MISS/episode [95%] | First landmark ΔSR, pp [95%] | Third landmark ΔSR, pp [95%] |
|---|---|---|---|---|
| 50 | -0.200 [-3.800, 3.400] | 0.472 [-0.748, 1.726] | -0.833 [-6.114, 4.365] | 0.385 [-4.634, 5.448] |
| 500 | 3.400 [0.400, 6.600] | -0.238 [-1.018, 0.570] | 5.000 [0.429, 9.623] | 1.923 [-2.410, 6.224] |

At 500, retaining the randomized opportunity improves episode success and shortens the trajectory: ΔN = −1.582 requests [−2.990, −0.234]. Its **net downstream call-count interval includes zero**, so dividing ΔSR by ΔM would be unstable. +3.4 pp is the value of assigning CALL at one selected landmark under this controller, including episodes that never reach it; it is not +3.4 pp for every guard call. The first-versus-third effect difference is +3.08 pp [−3.25, +9.20], so “first is better than third” is **not** statistically established. The first-landmark estimate nevertheless provides the clearest positive timing evidence. At 50, the total effect is unresolved, not proof that calls are unnecessary.

K5 has **zero pre-guard exposures**. It cannot estimate a MISS before guard evidence, because such decisions were never randomized. It also lacks the joint visual nearest-neighbour distance: logged `dnn` is raw valid robot-state nearest-neighbour distance. The joint visual/state coverage signal is available in A's logs but cannot be recovered from K5's normalized scores or missing keys. Do not rename `dnn` “visual coverage.”

The following are high-minus-low differences in CALL−CACHE effect, in percentage points. They test treatment-effect heterogeneity, rather than whether a feature predicts eventual failure. Full stratum sample/treatment counts, effects and intervals are in `causal_*.json` and `TABLES.md`.

| Online signal / split | Library 50 interaction [95%] | Library 500 interaction [95%] |
|---|---|---|
| State NN distance ≥ task LOEO p90 | 6.833 [-16.705, 30.399] | -17.483 [-65.588, 30.369] |
| Top-five action disagreement ≥ task LOEO p90 | 7.434 [-14.588, 29.723] | 30.695 [-9.214, 70.962] |
| Matched progress ≥ .5 | 1.697 [-13.111, 17.450] | -12.219 [-35.339, 11.160] |
| Library action-change event within current/next transition | 24.939 [1.347, 47.887] | 9.255 [-20.015, 39.073] |
| Recorded gripper-command mode switch within that window | 1.325 [-19.615, 23.052] | 4.325 [-23.113, 31.412] |
| stuck_n ≥ 2 | 8.936 [-19.328, 35.458] | -17.483 [-70.000, 32.752] |
| noprog_n ≥ 2 | -13.808 [-37.412, 10.018] | 12.253 [-25.029, 48.923] |
| elapsed / library median duration ≥ 1 | 21.852 [-15.051, 59.245] | 30.087 [-19.719, 80.946] |
| since MISS / library median duration ≥ .1 (previous MISS exists) | 18.850 [-13.342, 51.630] | 30.627 [-12.571, 73.623] |

A generic action event is a consecutive library executed-head change above its task's p90, in library-normalized action units. It does **not** mean verified grasp/contact/release. The gripper-mode row is a retrospective annotation using the recorded schema's dimension 6, without assigning “open” a common sign across models; it is excluded from the proposed portable rule. Contact is not observed by these logs. The splits .5, 2, 1 and .1 above are exploratory diagnostic conventions, not learned optimal thresholds.

The event interaction at 50 is nominally positive, but its family-18 interval is **[−10.70, +60.57] pp**; every feature interaction includes zero after this sensitivity adjustment. State coverage and stuck effects even change point sign across library scale. “Long since last MISS” has positive interaction point estimates, but intervals remain broad; it does not establish an optimal timer. The first landmark has no preceding MISS and is omitted from that timer contrast. These data do not support a suppression table. The earlier stricter [R5 Q3 audit](../../r05/q3_callvalue/HANDBACK.md) likewise found no supported saving under its 1 pp success-loss constraint; its task-level g500 interval also includes zero. This analysis does not relax that deployment decision.

### 2.2 Configuration gain per call, and where B-only trajectories separate

| Cell | A SR @ IR | B SR @ IR | B−A pp [95%] | B-only / A-only | B calls/episode | ΔSR pp / added call [95%] |
|---|---|---|---|---|---|---|
| pi05_l10_50 | 0.706 @ 0.07640 | 0.830 @ 0.18073 | 12.400 [8.200, 16.400] | 88/26 | 7.494 | 1.655 [1.047, 2.306] |
| pi05_l10_500 | 0.828 @ 0.07650 | 0.866 @ 0.16108 | 3.800 [0.795, 7.000] | 43/24 | 5.738 | 0.662 [0.122, 1.272] |
| pi05_spatial_50 | 0.838 @ 0.07754 | 0.910 @ 0.14145 | 7.200 [4.600, 10.000] | 45/9 | 1.770 | 4.068 [2.500, 5.742] |
| pi05_spatial_500 | 0.982 @ 0.07819 | 0.982 @ 0.12058 | 0.000 [-1.400, 1.400] | 6/6 | 1.058 | 0.000 [-1.209, 1.346] |
| groot_l10_50 | 0.608 @ 0.07428 | 0.718 @ 0.18503 | 11.000 [6.000, 15.600] | 104/49 | 8.676 | 1.268 [0.692, 1.829] |
| groot_l10_500 | 0.830 @ 0.07450 | 0.828 @ 0.18605 | -0.200 [-3.800, 3.400] | 42/43 | 7.836 | -0.026 [-0.468, 0.438] |
| groot_spatial_50 | 0.868 @ 0.07529 | 0.920 @ 0.19794 | 5.200 [1.600, 8.800] | 53/27 | 3.404 | 1.528 [0.466, 2.576] |
| groot_spatial_500 | 0.964 @ 0.07551 | 0.952 @ 0.19897 | -1.200 [-2.800, 0.400] | 6/12 | 3.246 | -0.370 [-0.860, 0.125] |

IR is recomputed as `.152 v + .848 m` for π0.5 and `.148 v + .852 m` for GR00T. The last column is **configuration accounting**: paired success gain divided by total additional calls, bootstrapping numerator and denominator together. It is not a local causal treatment effect. It includes downstream state changes, durations, call repetitions and both positive and negative paired outcomes. The 500-library GR00T periodic calls buy no resolved benefit; Spatial-500 π0.5 also has no observed net gain. That does not identify which particular calls can safely be removed.

For trajectory localization, align A and B by executed five-control slots. “State separation” is the first normalized state difference exceeding the task library's p95 one-transition motion; “action separation” exceeds its LOEO head-error p90. These are coarse library-scaled thresholds, not bitwise common prefixes or verified failure times.

| π0.5 cell | B-only with ≥1 MISS / all B-only | State separation before/at first MISS | Median first MISS slot / matched phase | First MISS reason among B-only with a call |
|---|---|---|---|---|
| pi05_l10_50 | 85/88 | 14 | 16 / 0.305 | {'4': 80, '1': 4, '2': 1} |
| pi05_l10_500 | 42/43 | 1 | 14 / 0.198 | {'4': 42} |
| pi05_spatial_50 | 43/45 | 1 | 14 / 0.579 | {'4': 34, '2': 8, '1': 1} |
| pi05_spatial_500 | 6/6 | 2 | 19 / 1.000 | {'4': 2, '2': 4} |

Reason 1 = stuck, 2 = terminal-row/closed-command, 3 = overtime+lag, 4 = no-progress. At l10-500, all 42 B-only trajectories with a MISS first call on **no-progress**, at median matched phase .198; 41/42 first call before the coarse state-separation threshold. At l10-50, 80/85 first call on no-progress. This is stronger evidence for protecting the first stalled-progress rescue than for spending calls mainly at the final release. It is still a paired-system association: B-only successes with **no MISS** exist (3/88 and 1/43 l10, 2/45 Spatial-50), demonstrating run variability and preventing the label “B-only = rescued by this MISS.” Small state differences below our threshold can precede the call.

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
|---|---|---|---|---|
| pi05_l10_50 | 0.301 [0.281, 0.321] | 0.335 [0.313, 0.358] | 0.710 [0.663, 0.758] | 0.744 [0.693, 0.791] |
| pi05_l10_500 | 0.271 [0.265, 0.278] | 0.309 [0.301, 0.316] | 0.620 [0.600, 0.640] | 0.780 [0.765, 0.795] |
| pi05_spatial_50 | 0.352 [0.326, 0.381] | 0.395 [0.367, 0.426] | 0.821 [0.768, 0.872] | 0.675 [0.609, 0.735] |
| pi05_spatial_500 | 0.294 [0.287, 0.302] | 0.321 [0.313, 0.330] | 0.597 [0.570, 0.623] | 0.809 [0.791, 0.826] |
| groot_l10_50 | 0.363 [0.338, 0.389] | 0.394 [0.367, 0.423] | 0.694 [0.650, 0.739] | 0.761 [0.720, 0.803] |
| groot_l10_500 | 0.359 [0.351, 0.367] | 0.388 [0.380, 0.396] | 0.574 [0.556, 0.593] | 0.763 [0.748, 0.778] |
| groot_spatial_50 | 0.348 [0.328, 0.371] | 0.391 [0.370, 0.415] | 0.798 [0.753, 0.840] | 0.729 [0.660, 0.795] |
| groot_spatial_500 | 0.349 [0.341, 0.356] | 0.372 [0.363, 0.380] | 0.623 [0.599, 0.645] | 0.753 [0.732, 0.773] |

Disagreement predicts large action deviation better than chance in all eight cells. Generic event AUC ranges .559–.626; inferred-stall AUC is only .500–.542 (intervals and eligible episode counts are in `TABLES.md`/`summary.json`). None of these labels measures a policy advantage. A demonstration can differ from another valid action, and the VLA can make the same error as the cache.

To check against actual closed-loop benefit, take one **A** vision anchor per episode: the first at/after one quarter of the task's library median duration, never a fraction of its eventual rollout length. Regressing is unnecessary: compare B−A success within high/low library-calibrated feature groups. All four π0.5 comparisons retain 500 episodes at this landmark.

- l10-50: high joint coverage-distance group = 362 episodes, B−A **+16.02 pp** versus **+2.90 pp** elsewhere; interaction **+13.12 [4.34,21.53] pp**. High state-distance group = 39, interaction **+31.05 [15.06,47.09] pp**. High disagreement group = 62, interaction **+2.42 [−9.08,14.42] pp**.
- l10-500: high joint-distance group = 57, B−A **+36.84 pp** versus **−0.45 pp** elsewhere; interaction **+37.29 [24.05,50.45] pp**. High disagreement group = 45, interaction **+22.69 [10.13,36.98] pp**. The state-distance high group has only 19 episodes, insufficient to rely on despite its positive point estimate.
- These relations do not transfer uniformly: Spatial-50 coverage interaction **+1.41 [−10.82,12.70] pp**; GR00T l10-50 **−0.17 [−10.05,9.77] pp**. Full eight-feature × eight-cell results are in `TABLES.md`. Extreme phase or motion effects from groups of 1–17 episodes are unsupported; do not use them as thresholds. These exploratory A/B intervals are not multiplicity-adjusted.

This is an association between **A's later state** and the value of an entire different controller. B may already have intervened before that A landmark. Thus it is not a causal estimate of calling at the A landmark. The measured temporal reach below exposes this distinction.

**Closed-loop falsification across existing arms:** beyond the 20 newly parsed arms, checked the 12 external controller predictions in R5-B `log_results.json` and the 12 arms in the six prospective B1 solve/hand comparisons reported in [R5 §7](../../r05/ANALYSIS.md). This gives 44 distinct completed arm results in the evidence review; the older 24 are inherited audited results, not freshly parsed raw data. The R5 failure-risk model has held-task AUC .617/.684 at 50/500, yet predicts the l10-50 phase controller safer than tail (.202 versus .248 failure) while their observed failures are .300 versus .194. Across its 3/9 external controllers, rank correlations are only .500/.644 (p=.667/.061); these small collections do not validate general ranking. All six prospective B1 changes reduce offline action and successor error, but only two improve SR; the pooled change is −0.53 pp [−1.63,+0.57]. The new data provide another warning: GR00T Spatial LOEO head error rises .348 → .349 from 50 to 500 while A SR rises .868 → .964. **No offline loss, AUC or alarm count in this report is accepted as an SR selector.**

### 2.5 Frozen-path timing rejects a simple universal novelty alarm

Let D and U be joint NN distance and top-five action disagreement divided by their task's LOEO p90. Calibrate a library-wide threshold τ as the p90 (`higher` order statistic) of successful-library-episode maxima of max(D,U), on noninitial two-quantum anchors. The table compares OR: max(D,U)>τ, with conservative AND: min(D,U)>max(1,τ), using the **same** threshold. No outcome labels from evaluated A/B runs enter τ. This is conditional library calibration, not a distribution-free false-alarm guarantee.

| Cell | τ | OR flagged A episodes /500 | AND flagged /500 | Successful A flagged by AND | AND alarm before B first MISS |
|---|---|---|---|---|---|
| pi05_l10_50 | 1.5979 | 321 | 146 | 17 | 10 |
| pi05_l10_500 | 1.6401 | 86 | 33 | 2 | 0 |
| pi05_spatial_50 | 1.3256 | 498 | 119 | 50 | 53 |
| pi05_spatial_500 | 1.4514 | 28 | 5 | 1 | 0 |
| groot_l10_50 | 1.5060 | 280 | 140 | 11 | 0 |
| groot_l10_500 | 1.5893 | 77 | 2 | 0 | 0 |
| groot_spatial_50 | 1.2447 | 500 | 111 | 70 | 0 |
| groot_spatial_500 | 1.4170 | 30 | 2 | 0 | 0 |

The OR screen's Spatial-50 alarm rates are 498/500 and 500/500, including 417 and 434 A successes. Nominal library calibration therefore does not preserve its alarm rate on evaluated trajectories. The conservative AND screen is more selective, but at π0.5 l10-500 **none** of its first alarms precedes B's first MISS. At l10-50 only 10 do. It is mainly a late trouble detector here. A separate exploratory calibration using a min-score's own episode-max quantile was also examined; lowering the threshold increased alert reach, so the final conservative screen keeps the common max-score threshold. This is an exploratory design sequence, not a preregistered prospective result.

For a genuinely earlier probe, an **upcoming action event AND disagreement above task p90**, without requiring distance to be extreme, has the following frozen-A-path reach. An event is detectable from stored future library actions; future rollout actions are never inspected online.

| π0.5 cell | Flagged A episodes | A successes flagged | First alarm before B first MISS | B-only pairs in that early subset |
|---|---|---|---|---|
| pi05_l10_50 | 420 | 281 | 156 | 23 |
| pi05_l10_500 | 298 | 227 | 96 | 12 |
| pi05_spatial_50 | 174 | 122 | 146 | 15 |
| pi05_spatial_500 | 125 | 119 | 101 | 2 |

This supports collecting earlier intervention evidence, **not** a forecast of additional successes. In particular, many successful A trajectories would receive unnecessary-looking calls. At 500 the early subset is only 96/500 in l10; a one-landmark study may be underpowered. The before-B comparison is between separate runs, not a replay of B's guard on the exact A state. GR00T's zero “before B” counts are tautological because its current B calls at initialization. Freeze a GR00T guard baseline before using this definition there.

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

## 6. Data we want but do not have

**Owner follow-up: requirements for the single superset profiler; analysis/design only.** The planned profiler should replace the separate eight-arm collection proposal in §4 as the next data-collection priority. The earlier measurements remain unchanged. The list below distinguishes gaps the announced profiler would close from additional requirements it would otherwise miss. It is one shared collection protocol with analysis strata, not a request to build separate profilers or launch experiments now.

The most useful target is the vector

`call_value(h, continuation) = E[(terminal success, remaining controls, remaining vision calls, remaining deployment policy calls) | do(CALL at h)] − E[same | do(CACHE at h)]`,

where `h` is the entire pre-decision history and both branches use a specified continuation controller. Shadow policy/cache action difference is an input, not this outcome. Likewise, randomization at every anchor estimates an effect under the profiler's subsequent randomized behaviour; it does not automatically estimate the effect of a single call followed by A, B, or the proposed sparse controller. Sequential off-policy estimation can in principle bridge these regimes when there is support, but long treatment sequences can have unusably large weights. This distinction determines the ranking below.

### Precision and cost conventions used in the ranked list

All new sample counts below are **planning calculations or explicitly proposed pilot sizes**, not observed profiler results, power guarantees, or launch requests. The arithmetic is reproducible with:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r06/ideation_Q3/wishlist_precision.py
```

[wishlist_precision.json](wishlist_precision.json) contains the formulas, assumptions, outputs and reach sensitivity. For a single Bernoulli success contrast at propensity `p`, using one independently selected eligible opportunity per episode, approximate pointwise 95% half-width `h` requires `n = ceil(1.96² s(1−s)/(p(1−p)h²))`. The first column uses **assumed** success probability `s=.85` in both groups; the next is the worst-case `s=.5`. For paired simulator branches from one source state per independent episode, `X=Y_CALL−Y_CACHE`, with **assumed** discordance `q=.10–.20`, use `n = ceil(1.96² q/h²)` near a zero mean effect. Actual branch discordance has not been measured.

| Pointwise ΔSR half-width | Independent eligible episodes, p=.5, s=.85 | Worst-case independent eligible episodes, p=.5 | Paired source states, assumed q=.10–.20 |
|---|---:|---:|---:|
| ±5 pp | 784 | 1,537 | 154–308 |
| ±3 pp | 2,177 | 4,269 | 427–854 |
| ±2 pp | 4,898 | 9,604 | 961–1,921 |
| ±1 pp | 19,592 | 38,415 | 3,842–7,683 |

These counts are **per cell and per claimed context/contrast**, not totals over all cells. Interaction tests, simultaneous intervals, multiple horizons, task generalization and unequal propensities need more support. At `p=.1`, equal-variance sample requirements increase by a computed factor **2.78** relative to `.5`. Many anchor rows sharing one terminal outcome are not independent episodes; cluster by episode and by common init across replicas/cells, and report treatment-specific effective sample sizes. Repeated branches from one state improve Monte Carlo precision, not the number of independent histories. The table is a conservative planning convention using one selected source state per episode, not a claim that extra anchors contain no information.

If collection reuses the same 500 task/init units, repeating them does not create thousands of independent initializations. Separate within-init policy/simulator randomness from between-init and run variation, and estimate precision with those clusters intact. Extrapolating the independent-episode columns beyond the available init set requires additional independent scenes/tasks or an explicit repeated-run variance calculation. The branch columns' assumed discordance is not a universal lower bound: unusually low paired loss/discordance can require fewer samples.

An initial proposed coverage pass is **500 episodes in each of eight existing cells = 4,000 episodes**: two models × two suites × two library scales. Spread the collection across **three independently seeded run blocks** and preserve paired init IDs; this is a design choice motivated by the existing repeat variability, not a guarantee that three blocks characterize all run noise. At 500 independent eligible opportunities, the approximate terminal-effect half-width is already **6.3 pp** under `s=.85`, or **8.8 pp** worst-case, before rare-context filtering. Thus this pass is useful for coverage, proximal responses and variance estimation, not a 1 pp SR-preservation claim.

Rarity dominates the targeted estimates. Existing frozen-path early-event reach was **156/500=.312** and **96/500=.192** in π0.5 l10-50/500. If those rates persisted—which randomization may change—784 eligible episodes would require **2,513/4,084 total episodes**, and 2,177 would require **6,978/11,339**. Those are sensitivity calculations, not predicted profiler reach. A context with 5%/1% episode prevalence needs **4,000/20,000** episodes merely to contain 200 distinct histories on average. Use a few prespecified coarse strata before attempting a large context table.

Incremental costs are stated as model calls, simulator continuations, bytes, or operations. **Serialization time, logging latency, branch throughput and GPU timing overhead have not been measured**, so no millisecond claim is made. All extra rollouts or policy sampling below can be deferred to saved-state analysis; they need not be paid during the collection pass. The coordinator should record measured overhead and missingness rather than silently dropping overloaded or difficult episodes.

### Ranked quantities

#### 1. Actual treatment support and history-specific causal outcomes at both guard and non-guard anchors — highest priority; partly covered by the plan

**What / granularity:** Per decision, the pre-assignment history/features, both candidate chunks computed from that same observation, baseline guard verdict, random assignment, assigned propensity, **effective probability of the action actually served**, availability/veto mask, actual source, and unique episode/attempt/anchor IDs. Per episode, terminal success, termination reason and complete future controls/vision/policy-call counts. Planned shadow chunks, retrieval internals, propensities and outcome joins cover much of this; the effective treatment/support contract must be explicit.

**Question answered:** Does a call at this coverage/disagreement/phase/stall/recency context improve SR, and is it worth its downstream cost? Can a currently forced guard call be withheld? “Random MISS injection” alone answers only additions if ordinary guards always force CALL: at those points the effective propensity is 1, however the injection coin was logged. To learn suppression there must be actual randomized CALL/CACHE support at the relevant guard points, or those contexts must be labeled unsupported. Never manufacture CACHE comparisons from non-guard states. Log random draws and overrides even when no action changes, and preserve pre-treatment values before guard/state updates caused by that action.

**Online measurement / cost:** Emit the existing candidate vectors and scalar controller/assignment state, with final accepted-attempt joins and explicit truncation/error fields. Reusing the already planned shadow chunk adds no further policy inference. Supporting an alternative treatment changes the rollout and its length, even when the model computation has already been paid. No extra inference does not mean no experimental intervention or no extra simulator cost.

**Needed precision / cells:** Collect in all eight cells. The 500/cell pass checks support; a terminal-effect claim with ±5 pp precision needs roughly **784–1,537 independent eligible episodes per claimed stratum**, or the paired-state alternative in item 2. This new dataset could validate a treatment-effect predictor; K5's negative feature result cannot be overturned merely by having more action-error labels.

#### 2. Exact branchable anchor states and paired CALL/CACHE outcome-to-go under a fixed continuation — highest priority; periodic snapshots alone do not supply it

**What / granularity:** Per selected decision, a restorable **pre-actuation** state plus both terminal potential-outcome samples and remaining cost vectors. Periodic simulator coordinates without a matching controller state are insufficient. Capture simulator dynamic state and time, actuator/control state, RNG state, observation/render state needed for replay, environment/termination counters, policy RNG or recurrent state if any, guard/retrieval histories, queued cache/policy tails with consumed offsets, scheduler state and pending randomization. Reference immutable model/library/code/configuration hashes instead of duplicating weights. Freeze both sampled chunks. Record the boundary relative to shadow sampling and treatment assignment so future policy RNG and controller mutations can be reproduced.

**Question answered:** At the **same physical and controller state**, what would the withheld chunk have done? Which calls rescue, harm, or merely duplicate a successful cached continuation? This replaces the coarse cross-run separation analysis and avoids treating every B-only success as a local rescue.

**Online measurement / cost:** Save snapshots immediately before a logged sample of event candidates, first guards, repeated late guards and ordinary anchors; include a random sample of ordinary states rather than only detected trouble. Record snapshot-selection probabilities. If only periodic capture is feasible, retain the entire intervening applied-control/RNG/controller-event trail and prove it can reconstruct the target anchor. Store replay checks: replay the factual action and compare the next observation, simulator state and controller verdict. A failed replay means no valid paired branch, not an approximate identical-state comparison. Online cost is serialization, synchronization, memory and disk; measure snapshot bytes and p50/p95 stalls. Later, each selected state needs **two simulator continuations** to termination under the same declared downstream controller, including later model calls as required. A factual suffix counts as one side only if its continuation actually matches. Matched exogenous randomness can reduce variance; allow and report residual nondeterminism.

**Needed precision / cells:** Start with a proposed **100 distinct source episodes/cell** for feasibility and discordance estimation, without interpreting this as a precise utility estimate. For each important context, plan **154–308 paired source states for ±5 pp**, or **427–854 for ±3 pp**, under the table's discordance assumptions. Repeated seeds at each state do not substitute for those source histories. Cover the eight cells eventually; prioritize π0.5 l10-50/500 and the Spatial-500 no-net-gain cell if branch resources are limited.

#### 3. Call-now versus call-later value and the remaining recovery window — missing even with shadow pairs and terminal outcomes

**What / granularity:** Per sampled trajectory segment, outcomes for a call **now**, after the next anchor, and after a further anchor, with the same preassigned policy allowance and downstream controller; a CACHE/no-extra-call reference; and remaining control budget at each point. Also the probability of recovery under a specified policy continuation from successive states before/after a suspected failure. “Unrecoverable” must mean under that tested controller and budget, not impossible for every conceivable policy.

**Question answered:** How much earlier must the MISS occur? Is a signal preventive, timely reactive, or already too late? The current report's “before first guard” label is not a physical failure/recovery deadline. A shadow chunk at an early anchor does not reveal whether waiting one chunk destroys the opportunity to recover.

**Online measurement / cost:** Retain linked pre-event/guard snapshots and dense intervening control/state records, preferably with a rolling buffer so the state just before the detected event is preserved. Later branch with delays expressed in observed anchor/control quanta, not fixed benchmark seconds. Keep the initial remaining episode deadline common when estimating deployable value; separately label any diagnostic recovery run given a fresh time allowance. Besides CALL-now and CACHE references, each tested delay requires an additional continuation per state; do not generate the full delay×duration×budget Cartesian product during collection.

**Needed precision / cells:** Begin with the same **100-source-episode** branch pilot, emphasizing event/first-stall boundaries in both l10 scales. A selected now-versus-delay contrast needs **154–308 source states for ±5 pp** under the planning assumptions; more for simultaneous delay-curve claims. Extend boundary strata to all eight cells before a universal “MISS before contact/release” rule. These data could reverse the conclusion that novelty is useful only as a late trouble detector, or show that a visually early alarm is already too late.

#### 4. Outcomes for different policy control durations and different handback decisions — still missing if only MISS/CACHE is randomized

**What / granularity:** Per intervention and subsequent handback opportunity, outcome-to-go and remaining cost for executing one control quantum versus the current two-quantum commitment (where the chunk supports both), and for returning to cache versus obtaining another policy chunk. Include real executed length, queued/discarded tail, refreshed observation and cache confidence at each cut point. The full shadow chunk supplies alternative commands, not the dynamics those commands would have produced or the value of an additional call.

**Question answered:** How long should policy keep control, when can cache safely resume, and are repeated guard calls a useful recovery burst or redundant requests? This directly tests the report's fixed-H/immediate-guard-clear handback choice.

**Online measurement / cost:** Either randomize duration/handback choices with their own conditional propensities within the superset protocol, or preserve exact cut-point snapshots for later branches. Holding source and duration separate requires a source×duration comparison; otherwise CALL versus CACHE confounds where to call with how long to act. A shorter look interval adds vision cost; an additional fresh policy chunk adds model inference beyond the original shadow sample. A decision at a formerly blind cut point is not a free existing vision anchor. No need to request policy chunks beyond the model's supported horizon.

**Needed precision / cells:** Use both models and both scales; duration conclusions cannot be transported from π0.5's chunk to GR00T's by naming both “H”. Allocate **154–308 paired source episodes per primary duration/handback contrast for ±5 pp**, or **784–1,537 independent eligible episodes** if randomized without state pairing. An initial 100-source pilot estimates response variance and whether the proposed contrasts are feasible. It cannot select a globally optimal horizon.

#### 5. Ground-truth physical events, subgoal progress and failure onset at control-step resolution — not implied by retrieval/guard logging

**What / granularity:** Per actual control step, simulator object/articulation poses and velocities, robot/end-effector state, object-contact pairs, contact forces or constraint indicators where exposed, commanded and realized gripper/aperture state, task/subgoal predicate values, and timestamps of grasp acquisition, slip/loss, release, task completion and relevant collisions. Preserve entity/frame/unit/channel metadata. Store raw measurements and the versioned event-label adapter; gripper command sign alone is not a grasp label. Label ambiguity explicitly, including intended release versus dropping an object.

**Question answered:** Do matched library phase and action-change events correspond to the real phase? Does a MISS help before grasp/contact/release or recover after a failed transition? Which useful alarms precede the first physical problem? These are evaluation labels, not permission to give the online cache oracle simulator features unavailable on another robot. Recovery *ability* still requires item 3, beyond detecting a contact/predicate change.

**Online measurement / cost:** Read already simulated state/contact/task predicates before/after controls, including blind steps. Log compact numeric data, with sampled synchronized observation clips for auditing disputed labels. No extra VLA inference is needed; physics API reads, predicate evaluation and I/O incur unmeasured overhead. Reconstructing these events from robot-state-only history or an aggregate final success label is not sufficient. The old libraries generally cannot be retroactively given exact object-contact labels from their stored robot state.

**Needed precision / cells:** Collect on **all 4,000 episodes of the proposed eight-cell pass**, not only failed episodes. Audit a proposed **100 clips per cell**, stratified by event type, success and failure, to establish label validity before using them. For event-specific utility use the terminal-effect counts above, not the number of contact ticks. For scale: a 5%/1% episode-level event needs **4,000/20,000 episodes** to supply 200 distinct event histories on average. Rare slips/releases need targeted snapshot sampling or broader collection, not invented certainty from many adjacent frames.

#### 6. The actual applied control stream, chunk lineage and observation chronology — per-anchor tables alone can miss it

**What / granularity:** Per control step and per decision, action issued after transforms/clipping, controller/frame/unit version, timestamps and actual controls elapsed, unique chunk ID and offset, whether the action is cache head, cache tail, policy head or reused policy tail, interruptions/discards, observation ID and age, and causal parent anchor. Per episode, reset/init identity, accepted retry, early success, timeout, error, censoring and last observed control. Missing observations must be distinguishable from a true zero motion signal. Some old logs contain portions of this; the complete joined stream does not exist for all required counterfactual cuts.

**Question answered:** Did MISS actually change the executed controls? Did it act before or after the event? Is “time since last MISS” measured since inference, since first execution, or since the end of policy control? Are guard stalls produced by blindly continuing a stale chunk? It also makes item 2 reproducible and prevents charging a reused tail as a new call.

**Online measurement / cost:** Log actuator-boundary values and lineage as steps are applied, linking to rather than repeating full chunks. Use a monotonic control index in addition to wall time. This is bookkeeping/I/O with no new policy computation. Compact state/control records on blind steps do not require new vision inference. Full observation capture on those steps is a separate byte/latency choice and should not silently change the controller's look schedule.

**Needed precision / cells:** **Every episode in every profiled cell; no subsampling of the control/lineage/termination skeleton.** This is a completeness requirement rather than an SR sample-size problem. Exercise all source and return paths in the first **100 episodes per cell** as a proposed coverage audit; missing rare paths stay unvalidated. More episodes cannot repair an omitted control interval or an ambiguous chunk offset.

#### 7. Deployment-cost outcome-to-go separated from the profiler's all-shadow cost — still missing from a simple MISS counter

**What / granularity:** Per anchor/episode, separate actual profiling compute from the compute a deployment would invoke: stage-1 vision calls, policy calls selected for execution, shadow-only calls, reused policy tails, executed control count, downstream calls, search/guard cost and measured stage timings. Include cold-start/warm state, device/concurrency and timing scope. The required value target is joint ΔSR and Δremaining cost, not just action mismatch or a success/call ratio with a near-zero denominator.

**Question answered:** Does moving a call forward avert later calls or shorten a trajectory enough to lower **deployment** IR? A shadow chunk reused for a randomized MISS is inference-free *incrementally inside the profiler*, but would still cost a real policy call in deployment. Counting all shadow calls as deployed MISSes, or pricing served shadow MISSes at zero, would both invalidate Q3's objective.

**Online measurement / cost:** Maintain two explicit ledgers and exclusive stage/timing scopes. Existing stage-price measurements can anchor model-based deployment repricing; record profiler overhead and distinguish that calculation from a measured non-profiler fast path. Instrumentation itself adds bookkeeping and possibly synchronization; actual sparse-path latency ultimately requires a shadow-disabled timing check, which all-shadow traces alone cannot measure exactly. Longer/shorter control commitments must be normalized by executed controls, not anchor count.

**Needed precision / cells:** Log all **500 initial episodes/cell** in all eight cells. Estimate cluster variance of Δcost, then size additional collection from it; there is no honest fixed episode count for a cost-saving claim before observing this variance and its covariance with success. The ±1 pp success precision requirement can already be **19,592–38,415 independent eligible episodes**, so a cheap-looking call schedule cannot claim SR preservation from a much smaller cost-only study. Paired branches reduce the required success sample only if their measured discordance permits it.

#### 8. History and continuation-regime support for sparse deployment, cooldowns and equal-budget placement — every-anchor randomization does not guarantee practical coverage

**What / granularity:** Per episode, assigned behaviour/rate/budget regime; per decision, prior assignments, executed MISS history, time since inference and since policy control ended, remaining **preassigned** policy allowance, guard history and all conditional propensities. Outcomes are needed under long cache-only prefixes, after the first rescue, after repeated calls, and under different placements of the same allowed policy budget. Final realized call count is a post-treatment outcome; do not condition on it as though it were an assigned budget.

**Question answered:** Is a call useful because it occurs at this event, or merely because the profiler has already used many earlier calls? Should closely spaced calls be combined or delayed? Can the same allowed calls be placed better? This is Q3's placement comparison while holding Q2's budget choice fixed, not a proposal to optimize the total budget here.

**Online measurement / cost:** Within the one protocol, retain a naturally behaving/reference or zero-injection cohort and support for a sparse continuation, or save reference-trajectory states and branch them under those continuations later. Keep shadow sampling even when the policy is not served. Record the exact baseline controller and regime assignment; otherwise all-anchor treatment effects can be transported to B only through an unvalidated model. Sampling multiple continuation regimes changes rollout cost and state distribution, but need not add policy inference per already shadowed anchor. Sequential importance weights and their effective sample sizes should be reported; a formally nonzero probability is not evidence of usable support.

**Needed precision / cells:** Start with the **eight existing cells**, and inspect support after the shared 500/cell pass. For any “long gap versus short gap” utility claim, seek **784–1,537 independent eligible episodes in each major compared stratum for ±5 pp within-stratum effects**; an interaction is less precise than either alone. Rare long-cache histories under frequent randomization may require a deliberately assigned sparse cohort rather than more of the same high-rate episodes. Branch alternatives use **154–308 independent source states per primary contrast** under the assumed discordance range.

#### 9. Synchronized shadow/cache proposals on the actual new-controller states, with score and calibration provenance — existing gap largely closed by the plan

**What / granularity:** Per anchor, the **full** cache candidate and policy candidate before treatment, common observation/timestamp and model/policy-context identity, raw joint NN distance, neighbour actions/rows/distances/weights, per-modality components, matched episode/row/phase, event lookahead, guard inputs/outputs, and exact calibration/metric/library version. Per library, the IDs, active dimensions/scales, candidate set and fitted transforms used by that decision. In the old store, `a_inf/a_hit` pairs exist on specific recorded query distributions; they are not synchronized shadow pairs at all current A/B states. Some older logs preserve only top-k normalized scores and omit joint distance, as K5 demonstrates.

**Question answered:** Does a joint-distance/disagreement/phase/recency feature predict causal benefit after controlling for its scale and history? Did the policy actually propose a different chunk? Can we recompute candidate placement features without another closed-loop run? The plan's “every retrieval internal” should explicitly include raw query keys or a lossless reproducible reference, raw distances, and the candidate action even when policy is served. It should not mean only the winning score or gripper sign.

**Online measurement / cost:** This is mostly already planned: persist the computed buffers before any serving-induced mutation. No additional model call is necessary. Derived LOEO distances, action standard deviations and strict metric-refit LOEO are **not missing measurements** when the existing library inputs are available; compute them offline rather than adding sensors. For storage sizing, the current two float32 vision keys cost **262,144 bytes/anchor**; two uncompressed 224×224 RGB images cost **301,056 bytes/anchor**, if additionally retained. These are schema-based byte calculations, not measured compressed sizes or new-robot constants. Avoid duplicating full library arrays in every row; use versioned references.

**Needed precision / cells:** Capture every anchor in all **eight cells**; the **500 episodes/cell** discovery pass is enough to inspect score distributions and logging consistency. Precision for “predicts a useful MISS” still comes from randomized/branched outcome support in each stratum, not from the larger count of shadow action pairs. The existing within-library action AUCs do not become causal merely because this input gap is filled.

#### 10. Policy-sampling uncertainty and a clean shadow-state contract — one shadow draw misses variance; lower-priority extra computation

**What / granularity:** Per selected anchor, policy RNG/latent seed, persistent-state identifier, a small set of independent policy chunks and, on a smaller branch subset, their outcomes. Preserve whether a shadow query advances any recurrent/context/RNG state even when its action is not served. A single shadow draw suffices for average randomized value under that sampling distribution; repeated draws are needed only to separate sampling noise from a stable advantage and to assess reliability of an action-gap placement feature.

**Question answered:** Is a large policy/cache difference a useful corrective direction or ordinary policy sampling variability? Would sparse serving use the same policy-state distribution as the all-shadow profiler? This prevents inadvertently evaluating a policy whose hidden history changes because shadow queries were added.

**Online measurement / cost:** Seeds/state provenance are cheap metadata. Freeze the planned shadow sample before treatment; do not resample selectively after seeing disagreement. **Optional pilot:** three policy draws at each of 100 saved states/cell, generated later if convenient; after the existing one draw this adds **200 policy inferences/cell**, and branch outcomes add corresponding continuations. Coupled seeds across a CALL/CACHE pair and independent seeds across replicates answer different questions and should both be labeled. Do not request repeats at every anchor without evidence they are needed.

**Needed precision / cells:** Start with the proposed **100 source states/cell**, spanning both models, suites and scales; this estimates a noise decomposition, not a precise SR gain. Use the paired-source-state counts above for any terminal-outcome claim. More samples of the same policy at the same state cannot establish transfer to new states.

#### 11. Treatment-response transport across controller/model/library/benchmark conditions — collection scope, not another scalar log field

**What / granularity:** Per cell/library and per episode, outcome-bearing intervention data for both models, both suites and weak/strong banks; exact library membership, collection origin, candidate/metric hashes and initial full-state identity. For a causal library-size interpretation, require nested or independently assigned comparable bank draws; the existing π0.5 50/500 banks change collection as well as size. For the generality claim, outcome-bearing data on a held-out benchmark and robot are still absent. Existing library quality statistics do not fill that response gap.

**Question answered:** Does the same placement statistic identify valuable calls when the bank, policy or task dynamics changes? Are the apparent scale effects really collection effects? This stays within Q3: measure treatment-response transport conditional on library/controller, rather than duplicate Q1's library-quality estimator.

**Online measurement / cost:** Run the same profiling schema across the **eight existing cells** in the shared experiment. If alternative banks are retrieved in shadow, that adds CPU search but no VLA inference; their action/outcome effects still require actual serving assignment or saved-state branches. A single LIBERO experiment cannot supply another robot/benchmark's outcomes. Such transfer must be a later dataset with a documented adapter, not an inference from the present logs. Initial simulator-state hashes and source episode provenance are needed to detect overlap, not just matching integer init labels.

**Needed precision / cells:** **500 episodes/cell** is the proposed existing-cell coverage pass, with **784–1,537 eligible independent episodes per cell/primary stratum** for approximate ±5 pp local effects. To separate benchmark transfer from robot transfer, a proposed minimum is **two additional weak/strong-bank cells on a new benchmark and two on a new robot setting**; this is scope guidance, not an authorized expansion of the profiler run. Exact precision across new tasks depends on between-task variance and cannot be guaranteed by increasing inits on the same ten tasks. Multiple bank draws likewise need separate uncertainty from episode sampling.

#### 12. Unused validation outcomes and run-to-run reliability for the chosen placement rule — cannot be obtained by relabeling its training trajectories

**What / granularity:** Per episode/run/cell, held-out outcomes for a frozen placement rule and matched reference, seed/run-block identities, complete failures/timeouts/retries, and a record of which histories/tasks were available when thresholds or predictors were chosen. The missing quantity is fresh performance of the selected controller, not another in-sample action-error statistic.

**Question answered:** Does the selected rule actually preserve SR and lower IR, including on tasks/runs not used to select it? Does any learned “skip this call” boundary survive the repeat variation already observed in R4/R5?

**Online measurement / cost:** Reserve init/task groups and run blocks before inspecting outcomes; retain their full data without using them for candidate selection. A frozen-policy value can be estimated on held-out randomized data only where sequential support and effective sample size suffice. Otherwise run held-out complete-policy branches or later normal rollouts; all-shadow randomized trajectories are not direct deployments of the selected rule. This needs additional simulator continuations and their model calls, not additional sensors. Never discard unexposed or failed episodes to obtain an attractive conditional SR.

**Needed precision / cells:** Preserve the proposed **three run blocks in all eight cells**, keeping common-init observations clustered; within each claimed validation cell size by the table using its measured variance/exposure. Even an overall ±1 pp estimate can require **19,592–38,415 independently eligible episodes** in the unpaired calculation, or **3,842–7,683 paired source episodes** at assumed discordance .10–.20. Noninferiority power is a separate calculation with an agreed margin and actual paired variance. Do not budget 500 as sufficient for 1 pp preservation without a valid interval based on the observed paired losses/discordance; a very low discordance rate can reduce the table's sample requirements. Rare adverse/error outcomes also need their own denominator: with zero observed events, the exact one-sided 95% upper frequency bound is **0.597% after 500 independent episodes**, or **0.100% after 3,000**.

### What the profiler already buys, what to add first, and which conclusions can change

The announced shadow chunks, complete retrieval/guard data, randomized actions and outcome joins already address the principal missing **inputs** and provide randomized **average effects under the profiling continuation** (items 1 and 9), provided actual support and compliance are logged. I would spend the next collection effort on **exact replayable controller+simulator state**, **dense compact execution/event traces**, and **separate deployment/profiling cost ledgers**. Those records enable delayed timing, duration and handback branches without repeating the expensive acquisition pass. Periodic snapshots that cannot recover an event-boundary state leave much of that value unavailable. If snapshots prove infeasible, retain the randomization and dense traces, report the lost counterfactual capability, and size prospective terminal-outcome comparisons accordingly; do not treat shadow action error as a replacement.

Expected changes to Q3 conclusions, conditional on the data rather than promised results:

| Current conclusion | Data that could change it | What would justify the change |
|---|---|---|
| First no-progress rescue has the clearest evidence; no validated earlier placement signal | Items 1–3, 5, 9, 12 | An earlier context has a positive randomized/branched value under the intended continuation and improves the frozen controller on held-out episodes. Physical labels could also show that “before guard” was already after the failure. |
| No safe late-call suppression table is supported | Items 1–2, 7–8, 12 | Actual CALL/CACHE support at late guards shows withholding preserves SR while reducing downstream deployment cost, with adequate intervals. Low observed success after late calls alone still cannot establish this. |
| Coverage/disagreement often detects trouble late and library alarm rates do not transfer | Items 3, 5, 8–9, 11 | New causal response and recovery-window data may show which contexts remain rescuable, or validate a conditional calibration. The measured late timings and near-universal Spatial-50 OR alarms in the old runs do **not** disappear; they remain falsification cases. |
| Keep the existing committed chunk and return when the guard clears; extra control duration is unproved | Items 2, 4, 6–7 | A source/duration/handback contrast establishes better SR–cost behaviour, including the cost of extra observations and calls. |
| Reported ΔSR per added call is configuration accounting, not local utility | Items 1–2, 7–8 | Estimate local success and downstream-cost effects with a specified continuation. Continue reporting the vector if the cost denominator is near zero rather than forcing a misleading ratio. |
| No portable useful-MISS predictor has been demonstrated | Items 9–12 plus all-cell randomized outcomes | Independent model/suite/library validation and genuinely new benchmark/robot outcomes support the same rule without threshold tuning on those outcomes. More LIBERO anchor rows alone cannot establish this. |

**What should not change merely because more logs exist:** the previous K5 point estimates and their original scope; the distinction between observational failure risk and treatment benefit; the exclusion of padded action channels; the requirement to cluster outcomes by episode/init; and the prohibition on claiming SR preservation from a lower offline loss. No profiler implementation, runtime changes, or closed-loop experiments were performed for this follow-up.
