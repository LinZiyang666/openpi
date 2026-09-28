import json,pathlib,numpy as np,collections
from score import O,joblist
C=json.loads((O/'rank_correlations.json').read_text());R=json.loads((O/'closed_loop_records.json').read_text());S=json.loads((O/'supplement.json').read_text());V=json.loads((O/'validation.json').read_text());ST=json.loads((O/'state_audit.json').read_text());PH=json.loads((O/'phase_summary.json').read_text());MA=json.loads((O/'metric_refit_audit.json').read_text());CA=json.loads((O/'calibration_eval.json').read_text())
def f(x):return '—' if x is None else f'{x:.3f}'
def ci(v):return 'undefined' if v['rho'] is None else f"{v['rho']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
text='''**Q1 result: useful library diagnostics, but no validated universal success score.**

The owner’s leave-one-episode-out idea detects improvements within a fixed model/suite and exposes where a bank lacks support. It does **not** establish an absolute quality scale that ranks closed-loop A across models/suites. The delivered composite is an interpretable reconstruction/support score, with a global summary, task summaries, and an online anchor implementation. It is **not a success probability, a policy-call utility, or an automatic MISS threshold**. A small independent trajectory sample calibrates its predicted errors, but does not repair its weak pooled SR ranking.

All computations are offline, on CPUs 10–13,54–57, with single-thread BLAS and CUDA disabled. No simulator, server, chain, tmux session, port, remote host, git operation, or shared data mutation was used. The closed-loop data are existing completed results; `r06_paper` was not incorporated while it was running.

**1. What was measured and how**

Read first: `../FINDINGS.md`, `../../r05/ANALYSIS.md`, `../../r04/ANALYSIS.md`, and `../../../harness/README.md`. Retrieval definitions come from `../../r02/g1_awm/awm.py`, growth/demo fit definitions from `../../r05/q4_growth/{method,demo_method}.py`. The earlier R5 conditional-LOEO solver and its negative SR validation informed the leakage checks, not score tuning.

The read-only store is `/home/weiland/trace_runs/offline_search_store`; closed-loop results are `/home/weiland/trace_runs/os_closed_loop`. `input_hashes.json` records fit and library identity hashes; `source_index.json` records summary/journal/DONE hashes and exact arm membership. The R5 analysis JSON is used only to discover arm paths: this analysis rereads accepted journals, excludes failed/error attempts, checks unique task/init identities, verifies completion markers, and reconciles each accepted count/success count with its summary.

Inventory correction: there are **18 actual library directories**, evaluated as **22 bank/fit combinations**. π0.5 has seven names per suite; GR00T has only `current` and `bpool_all`, so GR00T demo100/200/300 and grow250 scores/SRs cannot be invented. π0.5 `bpool_all` is a nominal 50 bank, not the 500 bank (`bpool_cs`). Spatial `current` contains 49 episodes. Grow250 retains successful acquisition episodes, giving 258 l10 / 297 spatial episodes, not 300. Its 208 / 248 acquisition episodes are exact source-file overlaps with the recorded inference trajectories. Both query streams are restricted to inits 25–49 for grown-bank evaluation; the five-step growth comparison restricts its static references to those same inits too. Other source-file overlaps with query trajectories are zero. Numeric B-pool init IDs are not the A-pool query trajectories.

Scored **196,079 library-row evaluations + 947,792 held-query-row evaluations = 1,143,871**, counting repeated candidate fits separately. Every row is scored; no row/episode subsampling is used except the bounded pair-distance scale described below. Equal-episode means followed by equal-task means prevent long failures from dominating global scores. Raw state-level arrays, all progress profiles, and all 660 task/source summaries are delivered.

Retrieval uses the actual serialized AWM fit for 20 combinations, including candidate codes, PCA, main/early metric and deployed kref. The two nondeployed π0.5 `bpool_all` fits use the same fitting recipe on their own library. Every task’s candidate search excludes the whole query episode for library LOEO. Recorded queries always use A’s post-anchor stale branch (`prev_hit=True` after step zero), even on inference recordings: injecting the recording’s MISS history would measure a different controller. The early metric is used at step zero. Each recorded decision is treated as a possible anchor; these are fixed-state probes, not a simulation of A’s changed trajectory or a restriction to its even-numbered anchors.

A uses top-16 kernel synthesis; kref=5 for current/demo/growth and 8 for deployed 500 endpoints. We retain those actual settings. The 50→100 data source changes and 300→500 bandwidth change therefore remain confounds. Pure-cache A is evaluated separately from five-step CL2 growth and from mixed/blind/controller stress tests.

Distances/projections are batched float32 arithmetic with the saved maps. This is the same features, metric and kernel, not a bit-identical scalar replay. Across 400 scalar deployed-query checks, the maximum action-RMS discrepancy was 0.002805 and the largest nearest-distance discrepancy was 0.005722; each check is in the per-fit JSON. The portable online adapter was separately checked on 100 scalar anchors: maximum Q difference was 3.52e-7. All array/edge/holdout invariants passed; independent SciPy checks agreed with all reported ordinary rank coefficients to 2.23e-16 (`validation.json`).

Reproduce from the repository root (no commands launch closed-loop work):

```bash
mkdir -p /tmp/r6_Q1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/r6_Q1/mpl
# Four scorer subprocesses, all inheriting the same CPU/environment restrictions.
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/run_all.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/metric_audit.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/analyze.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/state_audit.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/supplement.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/calibration_eval.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/validate.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/plan.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/plot.py
taskset -c 10-13,54-57 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q1/write_report.py
```

`analyze.py` reads `/tmp/r5_analysis/analysis.json` as the historical index, or falls back to the delivered `source_index.json` when that temporary index is absent. Its indexed paths and audited identities are preserved in that delivered file. `run_all.py` invokes every numbered job in `score.py --list`, at most four scoring subprocesses. Projection caches go only to `/tmp/r6_Q1`. Source/fit artifacts and all raw store arrays remain read-only.

**2. Measured findings**

The table below gives every actual bank/fit combination. Q is larger for lower normalized distance/action/successor error, as defined in §3. `C95` is held-inference coverage under the library’s internal distance threshold; it is not a 95% success claim. Action error uses the whole committed 10-control policy chunk, successor error uses observed two-decision state displacement. Growth rows use 250 held-out episodes; other held-query rows use 500.

| Model/suite | Bank / fit | Episodes | LOEO Q | Held-inf Q | C95 | Action RMS | Successor RMS | Existing A SR / CL2* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
for j in joblist():
 tag='_'.join([j['cell'],j['library'],j['variant']]);a=json.loads((O/(tag+'.json')).read_text());g=a['metrics']['inf']['global'];l=a['metrics']['loeo']['global'];rr=[r for r in R['A'] if r['tag']==tag];star=''
 if not rr and j['library']=='grow250':rr=[r for r in R['A5_growth'] if r['tag']==tag];star='*'
 sr=np.mean([r['y'] for r in rr]) if rr else None
 text+=f"| {j['cell']} | {j['library']} / {j['variant']} | {a['episodes']} | {f(l['q'])} | {f(g['q'])} | {f(g['covered'])} | {f(g['err10'])} | {f(g['succ2'])} | {f(sr)}{star} |\n"
text+='''
Sources: individual `<cell>_<library>_<variant>.json`, `library_scores.csv`, and accepted outcomes in `closed_loop_records.json`. A dash means no matching A closed-loop arm. The two π0.5 `bpool_all` rows are scored but are not assigned current’s outcomes; l10 has the same payload family, while spatial adds the unsuccessful episode missing from current.

**Candidate-only LOEO is optimistic.** It freezes a PCA and supervised action-neighbor metric that already saw the omitted episode. A separate audit refitted the main and early metric after omitting each of all 199 current-library episodes. PCA and original fitting scales remained frozen, so even this audit is not fully inductive pipeline LOEO. Whole-chunk error increased in all four cells:

| Current bank | Frozen-fit LOEO action RMS | Metric-refitted LOEO RMS | Relative increase |
|---|---:|---:|---:|
'''
for r in MA:
 g=r['global'];text+=f"| {r['cell']} | {g['frozen_error']:.4f} | {g['refit_error']:.4f} | {(g['refit_error']/g['frozen_error']-1)*100:.1f}% |\n"
text+='''
Source: `metric_refit_audit.json`. The corresponding distance inflation is even larger, but distances are in changed fitted metrics and are not a clean common-unit performance comparison. Internal C95 is about 0.95 by construction on row-weighted library pseudo-queries; the reported episode/task weighting makes its macro mean slightly different. GR00T spatial-current has held C95=0.0045 while A succeeds at 0.868. Its task 1 has SR=0.98 and C95=0.0124; task 5 has SR=0.92 and C95=0. These directly falsify interpreting internal distance coverage as success probability. Better support is useful comparatively; crossing the internal 95th percentile is not automatically a failure or a worthwhile MISS.

**Rank validation.** Higher-oriented predictors mean Q/coverage directly and negative distance/error. Spearman ρ intervals use 2,000 bootstrap draws, seed 20260928. Each draw resamples ten task clusters within each suite, keeping both models, all library variants, controller outcomes, and paired gains together. Global correlations recompute arm macro-means; task correlations pool arm×task means with the same clustering. These intervals are conditional on the recorded fits, chosen configurations and run seeds. They do not independently resample nested libraries, rerun fits, model seed variation, or certify transfer to an unseen robot. No significance-based metric selection or multiplicity-adjusted superiority claim is made.

Primary A has 16 arms / 160 task cells: eight 50/500 endpoints and eight demo size/fit arms. Five-step growth has 12 arms / 120 task cells, all restricted to inits 25–49. The all-controller stress panel has 60 completed arms / 600 task cells across every requested run root, plus `r04_gblind`; it deliberately assigns the same A bank diagnostic to different controllers and includes the wrist reference as a different-retrieval stress case. It is not 60 independent libraries. Native/R2 CL0–CL1 are not relabeled as A.

| Higher-oriented predictor | A SR, global | A SR, task cells | B−A gain, global |
|---|---|---|---|
'''
for k,label in [('loeo_d_pair','−LOEO distance/pair scale'),('loeo_err10','−LOEO action RMS'),('loeo_succ2','−LOEO successor RMS'),('loeo_q','LOEO Q'),('inf_d_pair','−Held-inf distance/pair scale'),('inf_d_self','−Held-inf distance/LOEO median'),('inf_covered','Held-inf C95'),('inf_q','Held-inf Q'),('inf_q_online','Held-inf online Q'),('cache_succ2','−Held-cache successor RMS'),('log_episodes','log episode count')]:
 text+=f"| {label} | {ci(C['A'][k]['global'])} | {ci(C['A'][k]['task'])} | {ci(C['B_gain'][k]['global'])} |\n"
text+='''
Negative correlations in the gain column have the expected direction: stronger banks need less rescue. B comprises the four π0.5 C10 arms and four GR00T periodic G10 arms, each paired to A on task/init. Gains are respectively +0.124, +0.038, +0.072, 0.000 for π0.5 l10-50/l10-500/sp-50/sp-500, and +0.110, −0.002, +0.052, −0.012 for the corresponding GR00T cells. These are two different B policies, not a common causal response to a Q intervention. The count baseline already correlates with gain at −0.809; this study does not demonstrate additional gain-prediction value from Q.

Subtracting pure inference changes the ranking, and its execution length matters. The gap is **A−inference**, so larger is better. L10 uses the two recorded π0.5 l10 references averaged within pair, one π0.5 spatial reference, and the GR00T L10 references. L5 uses the two seeded π0.5 references per suite and recorded inference outcomes for GR00T. This is explicit paired protocol matching, not substituting the brief’s averaged headline SR.

| Predictor | A−inference L10, global | A−inference L10, task | A−inference L5, global |
|---|---|---|---|
'''
for k,label in [('loeo_q','LOEO Q'),('inf_q','Held-inf Q'),('inf_d_pair','−Held-inf distance/pair'),('cache_succ2','−Held-cache successor RMS'),('log_episodes','log episode count')]:
 text+=f"| {label} | {ci(C['A_gap10'][k]['global'])} | {ci(C['A_gap10'][k]['task'])} | {ci(C['A_gap5'][k]['global'])} |\n"
text+='''
The relatively strong cache-successor association with the L5 gap is an exploratory finding, not evidence that the score knows when inference helps: those cache trajectories came from the historical recorded cache policy, not each evaluated A/B configuration. It also weakens and becomes uncertain for absolute A SR. A different controller can produce a different query distribution.

The all-controller global correlation for LOEO Q is '''+ci(C['all_requested']['loeo_q']['global'])+''', and for held-inf Q '''+ci(C['all_requested']['inf_q']['global'])+'''. In the same small π0.5 l10 bank, existing controllers range from 0.620 to 0.830 SR; a fixed library score cannot distinguish them. The five-step/growth global correlations are '''+ci(C['A5_growth']['loeo_q']['global'])+''' for LOEO Q and '''+ci(C['A5_growth']['inf_q']['global'])+''' for held-inf Q. Detailed coefficients for action head, full chunk, stitched executed actions, successors at one/two decisions, internal shift, and count/success baselines are in `correlations.csv`.

**Useful within-setting signal, without absolute transfer.** Across the six π0.5 l10 bank/fit choices, LOEO Q ranks SR at 1.000 [0.638, 1.000]; spatial gives 0.829 [0.600, 0.943]. Held-inf Q gives 0.829 [0.486, 0.943] and 0.600 [0.486, 0.886]. These are retrospective ranks over known variants, not six independent experimental replications. GR00T has only two bank choices per suite; a ±1 correlation from two points is uninformative and is not claimed as validation.

Within a fixed π0.5 model/suite/task, the mean rank across the six variants is '''+ci(C['A']['loeo_q']['within_task'])+''' for LOEO Q and '''+ci(C['A']['inf_q']['within_task'])+''' for held-inf Q, versus '''+ci(C['A']['log_episodes']['within_task'])+''' for count. The correlation intervals overlap; no incremental advantage over count is established. Mean within-configuration task ranking is only '''+ci(C['A']['inf_q']['within_config'])+''' for held-inf Q. Task-conditioned error and source provenance must accompany the scalar.

One concrete misranking survives all score definitions examined here: π0.5 spatial demo300 refit improves Q over demo200 (0.549 vs 0.540 library-only; 0.532 vs 0.509 held-inf), but recorded A SR falls 0.970→0.944. Existing paired R5 evidence calls this a noise-boundary disagreement, not a proven regression; it is a useful prospective falsification pair. At fixed demo200 contents, refit beats frozen Q and SR in both suites. In spatial, distance/pair alone very slightly prefers frozen (0.511 vs 0.518), despite SR 0.808 vs 0.970; action and successor errors detect that fit-quality difference. Coverage alone misses action relevance.

**Where coverage breaks.** Progress is the hindsight fraction of each recorded episode, not a semantic phase label or an online input. There are five equal progress bins. The complete task×bin results and first-break/longest-run diagnostics are in the per-fit JSONs and `phase_summary.json`.

| Cell / bank | C95 first fifth | C95 last fifth | Action RMS first fifth | Action RMS last fifth |
|---|---:|---:|---:|---:|
'''
for r in PH:
 if r['source']=='inf' and ('current_refit' in r['tag'] or 'bpool_cs' in r['tag'] or r['tag'].startswith('groot') and 'bpool_all' in r['tag']):
  b=r['bins'];text+=f"| {r['tag']} | {b[0]['covered']:.3f} | {b[-1]['covered']:.3f} | {b[0]['err10']:.3f} | {b[-1]['err10']:.3f} |\n"
text+='''
The small spatial banks break almost everywhere under their internal scale, not only near termination. The large banks show a clearer late-trajectory support/error deterioration. Query-success-conditioned summaries in `supplement.json` keep successful and failed recordings separate; the main score includes both. Longer failing episodes and unobserved task completion remain caveats for interpreting progress.

**Per-state usefulness is about error, not intervention value.** Across the 16 A fits on inference recordings, online Q’s mean within-task state rank correlation with negative action error ranges 0.469–0.662, and with negative successor error 0.449–0.618. At episode level, matching a recorded trajectory’s average Q to A’s same task/init success is much weaker: for GR00T l10-current it is 0.345 [0.079, 0.534], but for π0.5 l10-500 it is 0.084 [−0.038, 0.207]. These are correlations of a different recorded trajectory with a paired outcome, not A on-policy state-value estimates. `state_audit.json` gives every fit, both streams, and task-cluster intervals.

**A small held-out calibration sample helps predict error, not SR.** Using only the first five inference inits per task (50 episodes), fit two multiplicative factors for predicted action/successor error, then evaluate on the remaining 450 episodes. Growth uses calibration inits 25–29 and test 30–49, so no acquisition trajectory leaks. Equal-fit mean task MAE falls 0.04095→0.02212 for action and 0.02215→0.01432 for successor over the 22 fits. The factors range 1.0065–1.2487 and 1.0310–1.3032. No success labels enter this calibration. Calibrated Q still correlates with A SR only '''+ci(CA['calibrated_q']['global'])+''' globally and '''+ci(CA['calibrated_q']['task'])+''' by task on the held-out inits. That is the main limit on using this as a portable SR score.

![Library quality versus existing closed-loop outcomes](quality_vs_success.png)

Sources for all numbers in this section are generated by the commands in §1: `library_scores.csv`, `task_scores.csv`, `closed_loop_task_scores.csv`, `rank_correlations.json`, `rank_within_cell.json`, `state_audit.json`, `metric_refit_audit.json`, `supplement.json`, and `calibration_eval.json`. The reported prior R5 interpretation is read from `../../r05/ANALYSIS.md` §§6–7; no uncomputed new SR is reported.

**3. Exact portable algorithm and calibration recipe**

Deliver **(support, action reconstruction, successor reconstruction, Q)** as a diagnostic vector. Report Q globally and per task; expose online Q and its components at an anchor. Keep the components because a single scalar is demonstrably inadequate across the four cells. `quality.py` implements `Calibration.at_anchor(task, rows, weights, nearest_distance)` without camera counts, robot coordinates, gripper signs, benchmark names, policy calls or simulator access. `load_audit_calibration` is just the adapter to the delivered data.

1. **Declare the deployment contract from metadata.** Supply the deployed embedding/metric/kernel, task or instruction grouping, valid action/state coordinates, executed recording interval Δ, and cached commitment H. In this experiment Δ=5, H=10, two PCA-64 views plus 8 state coordinates; those are properties of A and the recorded robot interface, not constants embedded in the portable score. Use the same trained retrieval transform and kernel as the controller. Do not apply recorded inference’s fresh-continuity reranking to pure-cache A. Missing task support means unsupported, not cross-task nearest-neighbor substitution.

2. **Define normalization from the candidate library alone.** Action scale σa,d is population std over the library’s first Δ valid action controls. State scale σs,t,d is population std of valid state coordinates among task t’s library rows. Score only physically valid, nonconstant coordinates supplied by the robot metadata; if a coordinate is constant in the library but changes in a query, flag unsupported variation separately instead of inventing a physical-unit floor. The measured adapter uses a 1e-12 numerical guard (fallback divisor 1); none of the measured valid action or state scales needed that fallback. The deployed retrieval fit itself retains its historical current-library action units; only quality-error normalization uses the candidate bank’s own units. This distinction is recorded explicitly rather than silently refitting the deployed retriever.

3. **Library leave-one-episode-out.** For every library row i, retrieve only same-task rows in episodes other than episode(i), in the frozen deployed geometry. For step zero use its deployed early metric; otherwise use its main metric. This is *conditional LOEO*, not an independent generalization guarantee. Select the deployed k=16 rows and weights `w ∝ exp(-((d-d1)/max(d[kref]-d1,1e-6))²)`, with actual one-based kref=5 or 8; normalize weights to sum to one. The portable implementation accepts weights from the retriever, so it does not impose this kernel on another robot/controller.

4. **Residual labels.** Let â be the weighted stored chunk. Action error `ea = sqrt(mean(((â[:H]-a_recorded[:H])/σa)²))`. Also retain head-only error over Δ and a second H-control diagnostic that stitches the next actually executed head onto the current head. The latter has a different meaning from the recorded policy’s proposed tail; omit it if no next head exists. For successor error at lag h=H/Δ (here two consecutive decisions), predict `δhat = Σ wj (sj+h-sj)/σs`. Omit candidate rows lacking that observed successor, renormalize their remaining weights, and record retained edge mass. Compare to `(si+h-si)/σs` with RMS over valid state coordinates. Omit query rows without that successor, and never invent a zero-motion terminal. Record one-interval successor error as an alternative. These are observed displacement predictors, **not counterfactual dynamics under â**.

5. **Distance calibration.** Per task, take at most 512 deterministic evenly spaced candidate rows (indices `unique(linspace(0,n−1,min(512,n),dtype=int))`). Let `st` be the median Euclidean distance between sampled main-metric codes from different episodes. Define `D=d1/st`. This reference preserves a bank’s density relative to its overall feature extent better than dividing by its own nearest-neighbor median. If fewer than two episodes or st=0, return unsupported. Also report `d1/median(LOEO d1)` as a distribution-shift diagnostic and `C95 = 1[d1 ≤ quantile(LOEO d1,0.95)]`, using NumPy linear quantiles over task rows. The early branch shares this main-geometry reference in the measured baseline; a separately calibrated early regime is an untested extension. C95 and the upper-tail rank `(1 + #LOEO distances ≥ d1)/(1 + nLOEO)` are internal diagnostics, not conformal coverage guarantees: the fitted transform reused the calibration episodes and states within an episode are dependent.

6. **Portable offline score.** On rows with valid successor labels, define `Q = 1/(1 + D + ea + es)`. All three coefficients are exactly 1; no closed-loop outcome fitted them. Its range is (0,1], and **1 means zero measured discrepancy, not certain task success**. Compute row means within an episode, episode means within each task, then equal-task means globally. Export the components alongside Q. Report fixed recorded-inference and recorded-cache query scores separately; do not mix their different state distributions. Five progress bins, first C95 break, and longest uncovered-run/episode-length are reporting diagnostics only. No future progress or success flag enters online Q.

7. **Online anchor version.** For an actual retrieval, retain its rows, normalized weights and nearest distance. Predict `eahat=Σwj ea_LOEO[j]` and `eshat=Σwj es_LOEO[j]`; unavailable successor residual labels use the task’s mean valid residual, with imputed weight exposed as `successor_evidence_mass`. Compute `Qanchor = 1/(1 + D + eahat + eshat)`. A missing entire task’s successor calibration gives no composite score. This uses only the current retrieval plus library calibration arrays. It requires no new observation beyond the anchor and no future outcome. Q alone adds two residual scalars per library row and task summaries; the full diagnostic adapter also retains one distance scalar per row plus task/episode identifiers; search cost itself is unchanged and online latency was not benchmarked here.

8. **Optional independent error calibration.** Before evaluating outcomes, reserve a deterministic 10% of recorded inference episodes within each task; use ordered episode identities, and choose at least one episode per task if possible. The actual 500-episode streams use the first 5 of 50 inits/task, with the explicit growth split above. Compute each component’s ratio `factor = (task-macro observed mean residual)/(task-macro predicted mean residual)` on calibration episodes. If the denominator is zero but observed error is positive, mark calibration unsupported. Multiply eahat/eshat by these factors; keep all other constants fixed. The remaining episodes are the error-calibration test, not training data. This is a mean-error calibration recipe, not a distribution-free bound or a probability-of-success calibration. Set `action_factor` and `successor_factor` on `Calibration` to use it. Fifty recorded episodes is the measured sample size here, not a requirement that transfers to every benchmark; a smaller sample must have its uncertainty measured rather than assuming these results persist.

9. **What can be deployed from this evidence.** Use Q and its components to audit bank growth, identify poorly supported tasks/states, and nominate a short list of libraries for paired evaluation. Do not choose a universal Q threshold for suppressing inference or claiming an SR target. For small libraries, prefer independently recorded probes for calibration and report the metric-refit LOEO sensitivity. A fully inductive episode split would refit PCA, metric and normalizers excluding the episode; that is a different quantity from coverage of the deployed fixed fit, and the full-PCA version was not computed here. No closed-loop labels are needed to compute this diagnostic; a separate SR calibration would require labeled closed-loop validation under the target controller, which the present correlations do not justify replacing.

**4. Closed-loop test plan: eight arms, proposed only**

The exact reviewable plan is `proposed_arms.json`. All arms use pure A: top-16 retrieval, one cached blind block, 10 controls per anchor chunk, zero policy calls, `serving="anchor_tail", budget=1, gates="budget_only"`, kref=5, ridge=.1, three action-neighbor pairs, state scale=1, deployed step-zero metric. Each paired comparison uses identical task/init evaluation pairs, inits 5–49, seed 20260929. Recorded inits 0–4 are the fixed calibration/ranking sample. This holds bandwidth and commitment fixed. No arms were run.

| Arms | Exact library/fit setting | Paired question | Predicted outcome, explicitly a hypothesis |
|---|---|---|---|
| 1 / 2 | π0.5 spatial, DemoBlindAWM demo200/refit vs demo300/refit, kref=5 | Does the observed 200→300 inversion persist when Q predicts 300 is better? | Q predicts 300 ≥ 200; given the existing small-effect reversal, expect an unresolved SR difference. A repeat favoring 200 despite better Q would falsify monotone SR use. |
| 3 / 4 | π0.5 l10, DemoBlindAWM demo200/refit vs demo200/frozen50, kref=5 | Does quality add information beyond identical candidate count/content? | Refit should retain the better error score and higher SR; its prior SR is .788 vs .684, but that is not a forecast for the new seed. |
| 5 / 6 | GR00T l10, two disjoint 50-episode banks from the actual bpool_all, each fully refitted, standard BlindAWM(lib=current,kref=5) in isolated exports | Can Q rank equal-count banks without inherited current-library geometry? | The preregistered higher-Q bank is hypothesized to have higher SR; no bank ID or magnitude is predicted before these new fits are scored. Count alone is tied. Expect substantial uncertainty. |
| 7 / 8 | GR00T spatial, the analogous two disjoint 50-episode refitted banks, same exact A configuration | Does that equal-count ranking transfer across suite, where internal C95 failed most? | Same directional Q hypothesis; near-ceiling task outcomes may make the ordering unresolved. Failure here blocks universal threshold use. |

The four GR00T selection manifests are supplied as `<cell>_hash50_<0|1>_episodes.json`: for each task, sort episodes by SHA256 of `r6q1_equal50_v1|<cell>|<task>|<stem>`, tie by stem; bank 0 takes the first five, bank 1 the next five. Selection uses neither actions nor success. The coordinator must export/fit those banks, compute and freeze their Q ordering on the reserved recordings **before** either loop. Each new fit uses its bank alone, including action normalization, not a borrowed 500 metric. These are fully specified future banks, not silently missing measurements in the existing-library table.

Report paired SR differences and 95% task/init-cluster intervals, plus per-task results; use exact paired discordances as a secondary check. Predeclare the four directional comparisons and report all of them. Do not call non-significance equality. Treat replicated large differences as stronger evidence than a single small positive ordering, consistent with R4/R5 seed spread. These eight arms test library-quality ranking, leaving MISS-budget/placement optimization to Q2/Q3. Testing a different physical robot or benchmark remains necessary and is not represented by swapping π0.5 for GR00T on the same LIBERO robot.

**5. Risks and falsification**

- **Identifiability:** fixed recorded trajectories do not reveal the counterfactual states reached by a different action/controller. Two environments can agree on every observed transition and differ after an off-support action. Thus no universal SR guarantee follows from these inputs alone.
- **Representation leakage:** even after removing an episode from candidates, the deployed metric learned from it. The 14.7–18.1% metric-refit action-error increases quantify this risk at current size; no full-PCA inductive claim is made.
- **Covariate and source shifts:** current vs B-pool, successful-only growth, inference vs historical cache, early vs main metric, and different commitment lengths all change the estimand. Coverage’s internal threshold demonstrably fails as an absolute success threshold.
- **Loss mismatch:** standardized action imitation and proprioceptive successor error need not measure object/contact/task outcome error. There is no special gripper rule in Q; this improves portability but does not make contact consequences observable. Multimodal successful actions can also have high RMS discrepancy.
- **Uncertainty:** nested banks and repeated controllers are dependent; only four model/suite cells and two physical task suites exist. Task-cluster intervals do not integrate refitting or run-seed uncertainty. The small calibration sample improved error calibration in these stored streams, not out-of-robot success calibration.
- **What would falsify even the limited recommendation:** prospectively worse same-setting SR for reliably higher Q across the fixed-count pairs; near-zero/negative within-task error rank on new robots; error correction factors failing on independently held episodes; or size/source controls eliminating the apparent within-cell ranking. Each blocks using Q to select a library without paired SR tests.
- **What has already been falsified here:** interpreting C95 as an SR probability, assigning a universal Q→SR mapping across the four cells, and claiming lower conditional LOEO error alone solves controller selection. The existing R5 B1 six-cell negative test remains in force; this report does not overwrite it with retrospective correlations.

Deliverables: `REPORT.md`; `quality.py`; reproducible scoring/audit/statistics scripts; 66 state-array NPZs; 22 bank/fit JSON summaries; library/task/correlation CSVs; provenance/validation JSONs; PNG/PDF figures; and the eight-arm offline plan plus four exact proposed-bank episode manifests. All are confined to this directory, with temporary projections/logs only in `/tmp/r6_Q1`.
'''
wishlist=O/'DATA_WISHLIST.md'
if wishlist.exists():text+='\n'+wishlist.read_text()
(O/'REPORT.md').write_text(text)
print('wrote',len(text),'characters')
