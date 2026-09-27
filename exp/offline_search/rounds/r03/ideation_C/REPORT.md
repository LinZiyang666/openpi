# R3 ideation C: prevent uncertain commitments; measure the handoff frontier

My pure-cache ranking at the current library is **`awm_borrowed_prior` > `awm_spatial_ridge`**. For the 500-episode library, first run the existing full-library AWM: the first proposal becomes that reference, and the second defaults to no change. My mixed-system proposal is **`calibrated_event_handoff`**, evaluated as success rate versus actual policy-inference ratio. I recommend three proposals, not four: the additional synthesis ideas I screened lack enough supporting evidence.

The strongest independent finding is about **when** confidence becomes informative. AWM's low confidence almost perfectly identifies failed spatial episodes retrospectively, but a low-confidence gate at a nominal 10% decision budget reaches only **1/57 spatial failures and 6/100 l10 failures by the second decision of their first repeated-pick spell** on held-out initializations. The same gate eventually touches 45/57 and 70/100 failures. This is primarily a late detector. Gripper ambiguity and action disagreement provide earlier opportunities, with more interruptions of successful episodes. Neither failure detection nor a replayed trigger establishes that a policy call will rescue an episode.

All SR forecasts below are hypotheses, not new closed-loop results. No model, simulator, server, GPU, tmux session, or existing repository file was modified or operated.

## Scope, inputs, and reproducibility

Required material was read in the requested order, including A's notes. I independently processed all eight completed π0.5 arms, 500 matched task/init pairs per arm:

`/home/weiland/trace_runs/os_closed_loop/r02_g50/runs/oscl50_p_{sp,l10}_cl{0,1,2,3}/`

Inputs were `server_*/decisions_*.jsonl`, `client/journal.jsonl`, `client/per_step.jsonl`, launch configurations, and summaries. Accepted client decisions match the server counts exactly after excluding the **one `_kind=client_timing` summary per episode**. Every actual client decision is FULL_HIT. I did not analyze GR00T or 500-library closed-loop outcomes; conclusions about their SR remain untested here.

`/dev/shm/offline_search_store` is absent in this tool environment. All store reads used the documented disk store, **`/home/weiland/trace_runs/offline_search_store`**, including current/big actions and metadata, current PCA projections/bases, and recorded query keys, state, and `a_inf`. This is a fallback, not a claim that I inspected the RAM copy. R2 outputs came from `exp/offline_search/results/r02/<method>/<cell>.npz` and adjacent JSON files.

Every Python command was prefixed with `taskset -c 9-17,53-61`, using `/home/weiland/projects/openpi/.venv/bin/python`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=''`, and `PYTHONDONTWRITEBYTECODE=1`. Scripts are single-process; at most two diagnostic jobs overlapped. All generated files are in this directory.

Reproduce any script with:

```bash
taskset -c 9-17,53-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r03/ideation_C/diagnose.py
```

Replace the final filename for the other diagnostics:

| Script, relative to this report | Output | What was measured |
|---|---|---|
| [diagnose.py](diagnose.py) | [closed_loop_diagnostics.json](closed_loop_diagnostics.json), `cl2_{spatial,l10}.npz` | All eight arms, paired SR, neighborhood support, held-out trigger timing |
| [offline_synthesis.py](offline_synthesis.py) | [offline_synthesis.json](offline_synthesis.json) | 24 R2 method/cell combinations; episode reweighting and gripper-mode synthesis using saved top-10 |
| [metric_cv.py](metric_cv.py) | [metric_cv.json](metric_cv.json) | Current-library supervised leave-one-episode-out metric validation, four model/suite combinations; 3.86 s |
| [metric_trace_sample.py](metric_trace_sample.py) | [metric_trace_sample.json](metric_trace_sample.json), `*_metric_sample.npz` | 2,000 seeded non-step0 cache decisions per model/suite; exact full-16 kernel replay; 27.88 s |
| [exact_vote_audit.py](exact_vote_audit.py) | [exact_vote_audit.json](exact_vote_audit.json) | Exact full-kernel gripper outputs already saved by R2, including task 6 |

The sampled metric replay reconstructs saved AWM actions with mean sigma-normalized RMS **2.7e-7–7.5e-7**. It is a reliable action counterfactual at the recorded states, not a trajectory counterfactual. Library CV excludes the held-out episode from supervised metric fitting and candidates; PCA and action sigma remain fixed to the current library. It tests interpolation among library episodes, not off-library recovery.

The synthesis diagnostic is explicitly approximate: only ten of AWM's sixteen picks are saved. Its reconstructed action differs from the saved full mean by average RMS .0257/.0292 on current π0.5 cache cells and .0403/.0468 at 500 episodes. Comparisons below use the same truncated baseline; do not treat their small differences as exact full-method effects. Closed-loop gripper-vote and episode-support features also use top-10 reconstruction. Logged `conf`, `disp5`, `dst`, `still`, and top-1 spell timing are exact. This limitation does not affect the main confidence-timing result.

## Measured closed-loop evidence at the current library

| Arm | Spatial successes / SR / decisions | l10 successes / SR / decisions |
|---|---:|---:|
| CL0 native B0 | 334 / .668 / 14,653 | 220 / .440 / 40,283 |
| CL1 B0 + mean-5 | 382 / .764 / 13,320 | 214 / .428 / 41,376 |
| CL2 AWM kr5, current fit | 400 / .800 / 12,718 | 315 / .630 / 36,192 |
| CL3 AWM + recovery/blend | 399 / .798 / 12,762 | 321 / .642 / 35,002 |

The paired decomposition is important:

| Change at fixed current library | Spatial ΔSR; gains/losses | l10 ΔSR; gains/losses |
|---|---:|---:|
| Synthesis: CL0 → CL1 | +9.6 pp; 104/56 | −1.2 pp; 77/83 |
| Method: CL1 → CL2 | +3.6 pp; 68/50 | +20.2 pp; 141/40 |
| Recovery/blend: CL2 → CL3 | −0.2 pp; 24/25 | +1.2 pp; 39/33 |

Approximate paired 95% intervals for the last row are **[−2.95,+2.55] pp spatial** and **[−2.13,+4.53] pp l10**. There is no demonstrated aggregate recovery improvement. There are nevertheless individual gains and losses; these logs do not prove that every failed grasp is irrecoverable or that policy recovery cannot work.

**Correction to A's configuration interpretation:** CL3 launches `StuckRecovery(base=AWM, base_kwargs={lib:current,kref:5}, ot_still:true)`. AWM's gripper insurance is off. The wrapper defaults to both successor blending and recovery. Thus CL3 is evidence against that combination, not a controlled test of gripper hysteresis. This follows the launch script and `rounds/r02/g3_recovery/wrappers.py`, not the shorthand in A's notes.

For AWM, spell decisions occupy **1.57% versus 52.55%** of successful/failed spatial trajectories and **12.84% versus 47.89%** of successful/failed l10 trajectories. This strongly associates spells with failure but includes consequences of failure and differing trajectory lengths.

AWM already spreads weight across approximately **3.82/3.62 effective library episodes** in spatial successes/failures and **3.14/3.07** in l10. Effective episode count is `1/sum_e(mass_e²)`. Failed neighborhoods are not simply single-demonstration monopolies. That weakens the case for mandatory episode balancing.

## Early-warning audit: retrospective discrimination is not an intervention schedule

For each suite, thresholds were set using CL2 inits **0–24 per task**; timing was measured on **25–49**, 250 episodes. Spatial test split: 193 successes/57 failures; l10: 150/100. Thresholds at nominal 10% are empirical training quantiles, **not probability calibration**. Test flag rates differ. A spell is a run of at least three identical top-1 picks, retrospectively starting at its first pick; “early” below means trigger no later than start+1. Features at a trigger use only its current/past observations.

| Gate | Test decision flags, sp/l10 | Failed episodes eventually touched, sp/l10 | Failed episodes touched early, sp/l10 | Successful episodes ever touched, sp/l10 |
|---|---:|---:|---:|---:|
| Lowest raw AWM confidence, nominal 10% | 10.39% / 13.20% | 78.95% / 70% | **1.75% / 6%** | 0% / 2% |
| Highest action dispersion `disp5`, nominal 10% | 8.56% / 11.12% | 87.72% / 94% | **54.39% / 32%** | 18.13% / 54.67% |
| Lowest reconstructed `abs(grip_vote)`, nominal 10% | 9.39% / 9.43% | 89.47% / 99% | 77.19% / 49% | 61.66% / 81.33% |
| Proposed sign change with `abs(vote)<.8` | 2.55% / 4.93% | 56.14% / 93% | 35.09% / 33% | 15.03% / 58.67% |
| Repeated pick ≥2 AND `still>1.98` | 5.65% / 14.53% | 80.70% / 88% | 3.51% / 9% | 0.52% / 23.33% |

The low-confidence cutoffs are `conf < −32.931` spatial and `< −25.533` l10; dispersion cutoffs are .7335/.6326. These are diagnostic thresholds for this fit and library, not deployment constants.

Failure AUROC from the **whole-episode mean of negative confidence** is .9976/.9290. Restricting the same statistic to the first five decisions gives .5701/.6133. Mean stillness similarly drops from .9999/.9184 to .4011/.6693. High retrospective AUROC does not establish predictive lead time.

All these percentages describe shadow triggers on fixed pure-cache trajectories. After the first hypothetical MISS, subsequent observations, episode length, and triggers are counterfactual. **None is an estimate of mixed-policy SR, rescue probability, or actual inference ratio.** The early-warning window is also a spell proxy, not a verified instant of irreversible physical failure.

## Offline support and the limits of current-only shrinkage

On spatial task 6 the current π0.5 fit has **111 rows, five episodes, 136 features**, within-action covariance rank 109, effective trace rank 17.87, and regularized condition number 169.89. The three action-neighbor pairs all come from one other episode for only **1.8%** of rows. Pair duplication is not the main explanation here; covariance estimation with little independent data is more plausible.

Current-library supervised episode CV, sigma-normalized action error:

| Fit | π0.5 sp | π0.5 l10 | GR00T sp | GR00T l10 |
|---|---:|---:|---:|---:|
| AWM ridge .1 | .4328 | .3634 | .4452 | .4398 |
| Ridge 1.0 | .4209 | .3654 | .4348 | .4429 |
| Half task covariance + half other-task covariance | .4190 | .3632 | .4302 | .4398 |
| Entirely other-task covariance | .4465 | .3987 | .4471 | .4648 |
| One action neighbor per other episode | .4335 | .3618 | .4446 | .4391 |

The half-pool looked promising in CV but failed to generalize uniformly to recorded cache states. Exact full-16 replay on 2,000 sampled stale/non-step0 decisions per cell:

| Fit | π0.5 sp | π0.5 l10 | GR00T sp | GR00T l10 |
|---|---:|---:|---:|---:|
| Baseline | .5854 | .5252 | .5140 | .5198 |
| Ridge 1.0 | .5779 | .5168 | .5080 | .5245 |
| Half-pool | .5944 | .5195 | .5070 | .5288 |

Ridge 1.0 lowers sampled spatial medians .4905→.4750 for π0.5 and .4144→.4036 for GR00T. However, π0.5 sampled gripper mismatch rises .2245→.2290; task-6 ambiguity `abs(g)<.8` rises .3854→.3951. A lower mean error alone does not fix the chatter mechanism. This is why current-only ridge is a secondary candidate and blanket task pooling is not a main proposal.

By contrast, the exact saved R2 **500-episode fit with current candidates** reduces spatial task-6 severe ambiguity `abs(g)<.5` from **17.54% to 4.91%**, comparing **kref=8 to kref=8**; task-6 error .6148→.5859. Compared to deployed kref=5 the corresponding starting ambiguity is 17.41%, but that comparison also changes kernel bandwidth. Moderate ambiguity `<.8` barely improves at kref=8 (.2883→.2668). On **l10 task 6**, the borrowed fit raises severe ambiguity **22.88%→44.98%** even as error drops .6117→.5338. It is a promising prior, not a universal grasp fix.

Exact full-cell scoreboard for the matched-kref=8 comparison (four cells in π0.5-sp, π0.5-l10, GR00T-sp, GR00T-l10 order):

| Metric | Current candidates/current fit | Current candidates/500 fit | 500-episode candidates/500-episode fit |
|---|---|---|---|
| Cache mean err | .5882 .5112 .5180 .5106 | .5633 .4913 .4991 .4877 | .4992 .4317 .4307 .4424 |
| Cache median err | .4696 .4191 .4231 .4493 | .4639 .4126 .4156 .4325 | .3847 .3522 .3494 .3798 |
| Regret vs current-library oracle | .1937 .2343 .1572 .2494 | .1688 .2145 .1383 .2266 | .1047 .1549 .0700 .1813 |
| AURC | .3828 .3404 .3637 .3745 | .3638 .3302 .3526 .3674 | .2933 .2712 .2941 .3184 |
| Bad rate, harness definition | .7141 .5899 .9950 .9976 | .7206 .5855 .9970 .9988 | .5963 .4754 .9913 .9935 |

The borrowed prior reduces mean error without uniformly reducing bad rate. GR00T's teacher-noise floor makes the bad indicator almost saturated. Our top-10 synthesis screen's `bad_rate_global_p90` is a different coarse diagnostic; it must not be substituted for the harness's bin-specific bad metric.

## Library scale and deployable cost ledger

Counts below were read from the library arrays. “Current 50” is actually **49** episodes for π0.5 spatial. All proposals retain both camera keys in every regime, including after a MISS.

| Model/suite | Current episodes / entries | Big episodes / entries | Current / big compact key+action MB |
|---|---:|---:|---:|
| π0.5 spatial | 49 / 1,018 | 500 / 10,909 | .875 / 9.382 |
| π0.5 l10 | 50 / 2,640 | 500 / 29,472 | 2.270 / 25.346 |
| GR00T spatial | 50 / 1,063 | 500 / 11,751 | 1.093 / 12.080 |
| GR00T l10 | 50 / 2,645 | 500 / 29,631 | 2.719 / 30.461 |

The compact calculation uses **580 B retrieval representation/entry**, plus valid full-chunk actions **280 B π0.5 / 448 B GR00T**. Add **17.039 MB/suite PCA bases/means**, per-task metric matrices and auxiliary arrays, and implementation overhead; this is not the current pickle's actual footprint. Measured CL2 fit artifacts are **21.342 MB π0.5-sp / 24.631 MB π0.5-l10**. The R2 digest reports 47–117 MB for big fitted artifacts. Current native deployed pickles are approximately **431/1103 MB π0.5 and 429/1068 MB GR00T**, using roughly 262 KB keys/entry. A raw-key 10× library would occupy roughly 2.9–7.8 GB before other payloads.

The ideal compact totals before task/implementation overhead are therefore **17.9–19.8 MB current** and **26.4–47.5 MB big**. Retaining cached statistics for fitting is an offline cost; deployment need only retain the chosen metric and active candidate codes. A borrowed 500-episode metric with current candidates is **50-episode online storage but 500-episode training information**, which must be labeled explicitly.

## Proposal 1 — `awm_borrowed_prior`

**Pitch and hypothesis.** Keep AWM's action synthesis and compact 50-episode candidate library; use a covariance/PCA prior learned from the available 500 episodes. The current metric is underdetermined on spatial tasks. More fitting information can improve gripper neighborhoods without requiring a larger deployed action library. This overlaps A's leading proposal; my independent tests support ranking it above current-only covariance pooling, while exposing a l10 ambiguity risk.

**Algorithm (T1, Method API).** In `fit(lib,ctx)`, open `bpool_cs` for π0.5 or `bpool_all` for GR00T, fit/load its dual-camera PCA-64 basis, fit AWM's 136-D task metric and step0 metric on that library, and code only `current` candidate rows. Initial arm: the already implementable `AWM(lib='current', fit_data='big', kref=5, k=16)`; fit data changes, deployment candidates and synthesis match CL2. Preserve AWM's visual-plus-tail ranking after MISS and visual joint ranking after HIT. Confidence is recomputed using the active current candidates and the new metric, rather than reusing an old confidence threshold. No new reset state.

For partial shrinkage, project current and big rows into the **same big PCA basis**, normalize both with the **same big per-task mean/std**, estimate action-neighbor covariances, and use `S=(1-alpha)*S_current+alpha*S_big`, then the existing trace-scaled ridge inverse. Apply the same construction to early rows. Covariances expressed in different PCA or standardization coordinates must not be added. Log the fit source, alpha, and per-task severe gripper ambiguity.

**Predicted effect and decomposition.** Synthesis effect: zero. Fixed-candidate method/fit-data effect: the matched-kref table measures −.019 to −.025 mean error, smaller median improvements, lower regret/AURC, but not uniformly lower bad rate. Those are observed kr8 results, not a measured kr5 arm. Fresh mean errors in the existing borrowed-fit R2 results are .3556/.3227/.3618/.3387; continuity remains active and the borrowing benefit is mainly stale, not a guaranteed fresh improvement. With the 50-episode candidate library, my uncertain closed-loop expectation is **+2–5 pp spatial, 0–4 pp l10** versus CL2; l10 could regress because its task-6 ambiguity increases. These are prioritization estimates, not confidence intervals. With the 500-episode candidate library this converges to the existing `AWM_joint_big_fbig` reference; expected incremental method gain over that reference is zero. Its measured lower errors in the last table column are the **candidate-library effect at fixed 500 fit**, not extra evidence for the prior alone. No new 500-library SR is claimed.

**Cost and deployment.** T1; 580 B key/entry, unchanged action payload and fixed PCA/metric overhead from the ledger. Same approximate AWM query cost (~1.1 ms in isolated R2 measurements; actual loaded-server latency is larger). Loading a prior is cheap; fitting is seconds to tens of seconds with cached PCA. The 500-episode training pool need not be on the serving path.

**Kill criterion.** Against matched kr5 CL2 on the same 500 inits, reject as the spatial default if overall SR fails to improve and task-6 SR remains below .50. Reject as an l10 default if SR drops by >2 pp or the number of failed grasp-transition episodes rises. Do not accept an err-only win. Use paired gains/losses and intervals; a small pilot cannot certify a one-point change.

**Cheapest diagnostic first.** Already completed: exact vote audit and matched-kr8 full R2 comparison. Next run one matched-kr5 offline check, then a 100-init spatial pilot covering task 6 and other trap-heavy tasks plus an l10 transition-task pilot. Confirm actual commanded grip signs and first spell entry, not only action RMSE.

**Variants.** Alpha 1 first, .5 only if l10 harms appear; compare 50-episode candidates/500-episode fit against 500-episode candidates/500-episode fit to isolate library growth. Do not multiply bandwidth, insurance, and recovery changes into the same first arm.

## Proposal 2 — `awm_spatial_ridge`

**Pitch and hypothesis.** A conservative pure-cache option when all fitted information must come from the current episodes: reduce the amplification of poorly estimated spatial covariance directions. This is lower priority than proposal 1 because its measured error gain does not remove task-6 ambiguity.

**Algorithm.** Copy AWM's method into the R3 implementation area when selected. In `fit`, retain current dual-camera PCA, current candidates, kernel-16/kr5, and the existing early metric. For **spatial main metric only**, replace `.1*tr(S)/d*I` with `1.0*tr(S)/d*I`. Use the new main metric in fresh and stale branches, so both still use vision; keep the original early branch at step0. Recompute AWM pseudo-query confidence statistics. For l10, and for 500-episode libraries, default to existing ridge .1. `reset` and `query` otherwise match AWM. This is covariance regularization, not tuning vision/state fusion by offline L2.

**Predicted effect and decomposition.** Synthesis zero; library zero; method only. Current-library CV improved spatial error .4328→.4209 and .4452→.4348. The seeded stale trace sample improved mean/median by about .006–.008/.010–.015, with regret changing by exactly the same mean delta at fixed oracle. Predict full stale mean error down **0–.01**, AURC approximately unchanged to .01 better after recalibration; bad rate could move either way. Fresh err is unmeasured for this variant; expect changes within roughly .01 and verify it. Gripper mismatch can worsen, so expected π0.5 spatial SR gain is only **0–2 pp**, with real regression risk. L10 behavior is unchanged by default. At 500 episodes this defaults to ordinary AWM; no scale gain is attributed to ridge. GR00T has supporting spatial action diagnostics but no closed-loop SR evidence here.

**Cost and deployment.** T1, exactly the same 580 B/entry and action/PCA ledger. No extra query matrix or per-episode state. Our entire six-variant/four-cell supervised CV screen took 3.86 s using cached projections. Fit/query overhead relative to AWM is negligible.

**Kill criterion.** If the spatial pilot neither improves SR nor reduces severe gripper reversals, do not promote it despite lower sampled error. Kill before a full arm if fresh error worsens >.015 or exact transition gripper mismatch worsens >2 pp. At 500 episodes do not add this arm unless new evidence indicates an ill-conditioned metric problem.

**Cheapest diagnostic first.** Already completed: supervised CV and 2,000-decision full-16 replay with near-exact baseline reconstruction. Next: small spatial pure-cache paired pilot; prioritize reporting task 6, not a pooled mean.

**Variants.** Ridge .1 control versus 1.0 only. Half-pooled cross-task covariance is a diagnostic ablation, not recommended as the default: it raised sampled π0.5-sp and GR00T-l10 error by .009 each. Keeping early metric fixed isolates the measured main-metric change.

## Proposal 3 — `calibrated_event_handoff`

**Pitch and hypothesis.** Handoff uncertainty should account for an impending gripper commitment, not wait for a confidently recognizable deadlock. Use calibrated action-risk confidence for normal reuse, lower confidence around ambiguous proposed sign changes, and allow short policy-controlled bursts until observations support returning to cache. The deliverable is the measured **SR–inference-ratio curve**, not a risk-coverage curve substituted for it.

**Algorithm.** Wrap the selected vision-mandatory AWM base. In `fit`, generate grouped leave-one-library-episode-out pseudo-queries and candidates, using only real action dims 0:7/steps 0:5 and state dims 0:8. Features are AWM distance/support, `disp5`, `dst`, regime indicators, gripper vote margin, and whether the proposed sign differs from the previous executed sign. Fresh features additionally include the actual previous-policy-tail residual; both camera distance features remain present. Fit small ridge least-squares risk predictors, with held-out monotone calibration, for `P(err>epsilon)` and `P(gripper sign mismatch)`. Epsilon is fixed per model/suite using the existing teacher-floor calibration or an explicitly reported action tolerance. Keep regime calibration separate internally, but outputs are probabilities on the same event definitions.

Library pseudo-queries do **not** recreate cache-induced drift. Optionally recalibrate using recorded B0-cache/inf `a_inf` labels on training inits only; declare this additional supervised data source, hold out all matching task/init episodes across sources, and freeze it before closed-loop evaluation. Report reliability curves/Brier scores separately for step0/fresh/stale and transitions; do not call a library-only probability calibrated on mixed online states without checking it. The read-only quantile gates above are warning signals, not this proposed learned calibrator.

At query: let `p_err` and `p_grip` be the two calibrated risks. Use `risk=max(p_err, p_grip)` during a proposed sign change or a split-vote boundary, and `risk=p_err` otherwise; return `confidence=1-risk`. Thresholding either risk is well-defined, but **the maximum is not itself claimed to be a calibrated union probability**. As a minimal first implementation, compare the simpler dispersion-only gate against dispersion plus the sign-change/low-vote trigger measured above. Exact full-16 vote is available inside the method and should replace diagnostic top-10 reconstruction.

On a threshold rejection, let the server run the policy. If a boundary/no-progress event triggered rejection, maintain a two-decision minimum policy burst. After that, permit return only after two observations pass a stricter return threshold and the ambiguity has cleared; during a burst the method returns finite low confidence. Use `q.prev_hit`/actual executed chunks to advance state, never the proposed cache chunk. `reset` clears burst and return counters. After MISS use AWM's fresh visual-plus-tail branch. After HIT use its visual stale branch. No cached/policy action blending. A late stillness trigger may be retained as a backstop, not counted as evidence of prevention.

**Required plugin work, verified from the inspected source.** `STAGE1_ONLY=0` loads stage2/3, but that alone does not implement mixed control. `PluginJudge` in `closed_loop/plugin.py` always returns FULL_HIT; `PluginSession.on_executed` unconditionally appends `1` to hit history. The selected implementation must wire confidence into a real MISS judgment, record the actual HIT/MISS outcome, retain the actual policy output in action history, and log both proposal and served source. Audit cached-execution verification so a legitimate MISS is not called a pick mismatch. Test HIT→MISS→HIT and consecutive MISS sequences before evaluating. This report makes no such source changes and launches no servers.

**Predicted effect and three layers.** Synthesis unchanged. Base method/candidate library held fixed. Handoff is a separately reported **control effect**, with the policy's work counted rather than credited to retrieval. On fixed traces, dispersion gives much earlier warning than raw confidence; adding the boundary trigger has a plausible prevention mechanism. No measured rescue rate exists, so no defensible SR prediction at a specified inference ratio can be extracted from these logs. My acceptance target at 50 episodes is **at least +5 pp SR with ≤20% actual policy decisions** versus the same pure-cache base; sample targets 5/10/20/40% plus pure cache and full policy. At 500 episodes rerun calibration and the same frontier; do not assume the 50-library threshold transfers. Exact overall moderate vote ambiguity actually rises from .145/.269 at current kr5 to .163/.281 at big kr8 on π0.5 stale traces, so library growth does not guarantee fewer interventions.

For the offline scoreboard, leave candidate action/mean error/median/regret/bad rate unchanged when testing only confidence. The expected improvement is lower accepted-set risk and lower AURC/bad-AURC; it must be measured, especially across regimes. When the policy runs online, its output is outside the library-action-only Method contract and belongs to the server MISS branch. Do not represent that as a synthesized library hit.

**Cost and deployment.** T1 statistics/closed-form calibration; same library sizes and 580 B/entry as the base, plus a few KB of calibrator state and O(1) episode counters. Target <.1 ms gate overhead; unbenchmarked. It requires the full-model server. The documented π0.5 model footprint is roughly 7.6 GB/server versus 2.2 GB stage1-only; model residency and policy latency are distinct from library bytes. Vision stage1 runs even on cache hits.

**Kill criterion.** Reject if the measured frontier is dominated by a raw-confidence gate or a matched-inference-ratio periodic-policy control. Reject the prevention claim if the first handoff remains mostly after the first prolonged spell. A result with >20% policy decisions and <5 pp SR gain fails the initial efficiency target; compare with full policy to account for its own success ceiling. Calibration that fails on held-out mixed trajectories must be refit or labeled uncalibrated, not hidden by retuning thresholds on evaluation outcomes.

**Cheapest diagnostic first.** Completed: episode-held-out warning-time table. Next: CPU-only mixed-history/plugin selftest and held-out calibration diagnostics, then a small full-model closed-loop pilot. Report successful-episode interruption rates as well as failed-episode lead time. These diagnostics cannot establish the policy's rescue probability.

**Variants.** Error-risk only versus error-risk + transition risk; minimum policy burst 1 versus 2 decisions. Freeze the best pure-cache selector first. At each library size report actual MISS/all-decision ratio, success, decision count, latency, and paired task/init outcomes; exclude `client_timing` records from denominators.

## Rejected or deferred ideas, with measured reasons

1. **Mandatory episode-balanced kernel weights.** Top-10 `w_i/sqrt(episode_mass_i)` changed current π0.5 cache error .5779→.5839 spatial and .5150→.5147 l10; fresh error worsened .3348→.3433 and .3166→.3208. At 500 episodes it worsened cache error .5004→.5027 and .4320→.4343. Full episode equalization was worse. Combined with effective support of roughly three to four episodes already present in CL2, this does not justify a new arm. These are truncated-kernel screening results, not exact full-AWM or SR measurements.

2. **Always choose one gripper mode, then average only that mode.** Grouping top-10 by the full five-step gripper-sign pattern and choosing the largest-weight group raised current cache error .5779→.6029 and .5150→.5487; gripper mismatch barely changed (.2072→.2069 spatial, .1674→.1684 l10). At 500 episodes error rose .5004→.5345 and .4320→.4623, and gripper mismatch worsened. Restricting the intervention to vote ambiguity <.8 barely changed this result. This does not disprove a careful stateful commitment policy, but it does reject an assumed “mode-only averaging is free” improvement. Its offline cost is large enough to demand targeted closed-loop evidence before promotion.

3. **Another late exclusion/recovery arm, or raw low confidence as the whole mixed design.** CL3's paired SR differences include zero on both suites despite breaking many pick repeats. Raw-confidence flags reach failures mainly after spells begin, while a plain two-repeat trigger flags 25.3%/36.5% of held-out decisions and touches 74.6%/99.3% of successful episodes. Neither is a convincing low-inference prevention mechanism alone. Retain raw confidence as a comparison arm and late detection as a backstop; prioritize early boundary uncertainty and actual policy handoff.

## Recommended ordering of the real exam

At **50 episodes**, first compare matched-kr5 CL2 to proposal 1; proposal 2 is a low-cost spatial fallback if the prior cannot be used or harms other tasks. Then evaluate proposal 3 around the winning fixed base. Include a periodic-policy control at matched actual inference ratios; timing, not just the presence of policy calls, must earn its complexity.

At **500 episodes**, establish the existing AWM full-library SR first. Proposal 1 then adds no new metric change; proposal 2 is off by default. Spend the next arm budget on proposal 3 and a matched-inference control rather than stacking unproven pure-cache repairs. Report the candidate-library gain using the same 500 fit on current and big candidates, and the handoff gain using the same base and candidate library. The historical ~10× B0 SR (.810 spatial/.516 l10) is contextual, not a paired result for these proposals.

A 100-init pilot can screen gross regressions and event timing. Final SR claims and SR–inference frontiers require the common 500-initialization evaluation with paired uncertainty. No offline scalar in this report substitutes for that test.
