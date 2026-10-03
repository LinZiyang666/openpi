# R8 callback — E1 segmentation

**The data now support useful segmentation research, but they do not yet validate a hard/easy routing rule.** Retrospective change-point boundaries improve alignment with T2's physical-stage heuristics. Online waypoint scores sometimes identify more upcoming heuristic failure onsets than R7 event mass, but the improvement depends on the cell and continuation policy. Elapsed time remains a strong baseline. Independent event annotation and assignment-time stage logging are the main gaps.

## 1. What I ran

Run root: `/home/weiland/trace_runs/os_closed_loop/r08_main`. Analysis date: 2026-10-01. Sources read: the full ideation brief, my proposal, debug schema, both tool READMEs, reader, adapters, forensics, segmentation/benchmark, cards, decision divergence/common, and R7 stage definitions.

Scope: `r8_{groot,pi05}_{l10,spatial}_{50,500}_{A,CU,CT,FL}` plus `r8_{groot,pi05}_{l10,spatial}_P10`: **36 distinct arms**, 300 discovery episodes per arm. P10 is four physical runs shared across the two library sizes, not eight independent runs. All analyses use within-task **inits 0–29**. No holdout controls, outcomes, or tool metrics were examined; metadata returned by the reader was filtered immediately before analysis. No tool received an init ≥30.

Coverage: **10,800 episodes; 455,293 decisions; 2,365,580 controls, including 108,000 settle controls**. Required kinematic/predicate fields passed finite/contiguous-control checks; all 455,293 decision joins were `verified`; zero reported capture-error episodes, read issues, or analysis exceptions. For each of the **600 suite/task/init groups**, all 18 corresponding arm/model runs share one reset-state hash. This does not certify library/evaluation overlap or debug-on/off behavioral invariance.

I used the delivered functions through a discovery-only wrapper because the stock physical CLI loads all accepted episodes and emits holdout results. The wrapper adds the actual catalog event-mass comparator, matched exposure, and task-stratified uncertainty; it changes no production/tool code. Geometry scales were fitted on successful P10 discovery trajectories per model/suite. Physical radii were calibrated from the four P10 discovery cohorts and shared across arms/models within suite/task. R7 command centers came from the actual frozen library (`current`, π0.5 `bpool_cs`, GR00T `bpool_all`). P10's gripper baseline uses the 50-bank centers once. Fits and calibration hashes are saved.

Reproduction commands, from the repository root; the four `analyze` invocations were scheduled with at most three concurrent Python processes:

```bash
E1_DIR=exp/offline_search/rounds/r08/ideation/E1_segmentation
E1_OUT=/tmp/r8cb_E1_segmentation
E1_PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
       MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
       PYTHONPATH=.:src MPLCONFIGDIR=/tmp/r8cb_E1_segmentation/mpl .venv/bin/python)
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" prepare
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" analyze --model groot --suite l10
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" analyze --model groot --suite spatial
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" analyze --model pi05 --suite l10
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" analyze --model pi05 --suite spatial
"${E1_PY[@]}" "$E1_DIR/callback_checks.py" semantic
"${E1_PY[@]}" "$E1_DIR/callback_checks.py" augmentation
"${E1_PY[@]}" "$E1_DIR/callback_analysis.py" smoke --model groot --suite l10
"${E1_PY[@]}" "$E1_DIR/callback_summary.py"
"${E1_PY[@]}" "$E1_DIR/callback_extras.py"
"${E1_PY[@]}" "$E1_DIR/callback_plot.py"
```

The full grid includes 14 candidates, boundary tolerances ±1/5/10 controls, and waypoint reconstruction at 1/5/10-control resolution. Prefix checks cover init 0 of every task/arm: **360 episodes × 14 candidates = 5,040 checks**, each at three prefix cuts. The separate stock `segmentation_bench.build` smoke used GR00T L10-50 A, ten init-0 episodes: 3,696 segments, 840 boundary rows, 11,914 causal-decision rows, 6,132 risk rows, 140 passing prefix checks, and zero unavailable episodes. An initial wrapper-only Parquet serialization error for mixed-type fit tuples was resolved by saving fits as JSON; the benchmark itself completed.

The semantic audit uses init 0 of all tasks in A's eight arms and P10's four arms: **120 episodes**. Two delivered HTML/PNG cards were generated and inspected for GR00T L10 task 0/init 0, A50 and P10; both have captured camera images.

**Augmentation-dependent findings are partial:** only `r8_groot_l10_50_A` and `r8_groot_l10_50_FL`, ten init-0 episodes each, were passed to the delivered divergence tool after verifying their `AUG_DONE` markers. The primary 36-arm analysis needs no augmentation. No camera-shadow, full-grid divergence, or augmentation-completion claim is made.

All products are under `/tmp/r8cb_E1_segmentation/`: per-arm Parquet tables and `COMPLETE.json`, fits, `summary.json`, `report_numbers.txt`, risk/boundary/reconstruction/age CSVs, divergence tables, two episode cards, and `E1_discovery_summary.{png,pdf}`. Scripts remain in this ideation directory. Source captures were read-only, reader cache writes disabled; no GPU, simulator, server, worker, or git operation was used.

Review artifacts: [boundary/risk figure](/tmp/r8cb_E1_segmentation/E1_discovery_summary.png), [exportable PDF](/tmp/r8cb_E1_segmentation/E1_discovery_summary.pdf), [complete numeric tables](/tmp/r8cb_E1_segmentation/report_numbers.txt), [A episode card](/tmp/r8cb_E1_segmentation/episode_cards/episode_e82536074acc3eef38e4.html), [paired P10 card](/tmp/r8cb_E1_segmentation/episode_cards/episode_36f707ae632bd28bc165.html).

## 2. Do the tools and data answer the requested questions?

| Tool requested in my proposal | Verdict | What works; concrete remaining gap |
|---|---|---|
| `stage_data_audit` | Partial | Reader joins, manifests, actual controls, settled references, and required-field checks worked across all 10,800 episodes. This audit did not run the full contact integrity selfcheck, action/chunk parity, or debug-on/off invariance checks. |
| `physical_event_timeline` | Partial | Forensics T2 and cards expose per-control approach/grasp/carry/place/release/fixture intervals and poses. They lack validated contact-mode/acquisition labels, fine-alignment/insertion stages, and an independent annotation audit. “Sim truth” is a kinematic heuristic label. |
| `segmentation_bench` | Works for the delivered simple grid | Gripper/stop/waypoint/change candidates, matched-count boundaries, reconstruction and resolution curves all execute. Geometric knots use greedy RDP, not minimum-waypoint AWE; mean-change DP is retrospective. HSMM, contact-constrained segmentation and learned visual stages are absent. |
| `causal_stage_replay` | Partial | All 5,040 fixed-fit score prefix checks pass. Outputs are causal scalar scores, not semantic stage posteriors: posterior is null/unknown mass is 1. Confirmation delay, filtered-versus-smoothed stage agreement, and actual server availability of per-control proprioception remain untested. |
| `boundary_risk_audit` | Partial | Next-20-control windows, PR/lead metrics and candidate scores exist. I added actual catalog R7 mass, equal exposure per arm/task, distinct-onset recall, and paired task-stratified CIs. Onsets are unvalidated and outcome-dependent; stock unknown-onset negatives and in-sample calibration need correction. No nested discovery validation or holdout result is supplied. |
| `stage_divergence` | Partial | Served-versus-shadow coverage is **851/851 A and 611/611 FL decisions**; noise floors **28/851 and 19/611**. Native cache proposals cover **426/851 and 218/611**, chiefly fresh looks; deferred `shadow_look` is needed for fresh blind alternatives. These normalized-action comparisons do not independently validate physical risk. |
| `stage_call_value` | Partial | Delivered call/lottery tools enforce supported propensities and pre-assignment stages. In the checked A/FL subset, **0/1,462 decisions** have a certified `stage_pre`/`pre_stage`; all available stages are catalog-descriptive. E1 did not run call-effect estimation or impute causal moderators from post-choice labels. |
| `relative_bottleneck_audit` | Missing as the requested comparison | Captured object/EE poses and `grasp_audit` provide useful primitives. There is no demonstrated phase-aligned world-versus-object-frame dispersion/held-init validation; old library catalogs do not supply certified full object/contact trajectories. |
| `stage_atlas` | Works for static review | Both tested cards render physical/source strips, trajectories, aperture, onset and two cameras. A reviewer can inspect evidence; the requested double-annotation workflow, uncertainty adjudication, and a comparative atlas remain to be built. |

Data-field assessment: capture now supplies the core missing R7 observations—full-rate robot/object poses, aperture, actions, predicates, named contacts, images and stable IDs. However, raw pre-PCA keys are sampled **1/16**, rather than every decision as proposed, and extra policy draws are **1/32**, rather than 1/16. These are documented coverage limits for later visual/noise studies. Snapshots remain `restore_certified=false`; they do not authorize physical counterfactual claims.

## 3. Findings for segmentation and lazy-lever allocation

### Boundary quality and temporal resolution

Primary descriptive reference: T2 stage changes; tolerance ±5 controls. The table summarizes **2,400 A episodes with 16,516 reference transitions**. F1 is the equal-cell mean of each arm's micro-F1; predicted/matched counts are totals across the eight arms, so they do not directly recompute that mean.

| Candidate | Predicted boundaries | Matched reference boundaries | Mean F1 | Same-count uniform F1 |
|---|---:|---:|---:|---:|
| R7 gripper-command history | 6,664 | 2,129 | .183 | .185 |
| Stop q25, dwell 5 | 14,848 | 3,518 | .222 | .274 |
| Stop q25, dwell 10 | 11,762 | 3,531 | .259 | .234 |
| Waypoint ×0.5 | 137,261 | 15,004 | .220 | .231 |
| Waypoint ×1 | 95,385 | 13,323 | .267 | .283 |
| Waypoint ×2 | 64,783 | 11,485 | .312 | .328 |
| Change ×1 | 20,913 | 6,551 | .359 | .292 |
| Change ×2 | 16,122 | 5,353 | .340 | .277 |
| Change ×4 | 10,394 | 3,142 | .240 | .213 |

Change ×1 beats its equal-count uniform comparator in **36/36 arms**. Mean F1, candidate/uniform: A .359/.292; CU .382/.312; CT .377/.313; FL .350/.283; P10 .388/.324 (four arms). Waypoint ×1 loses in **36/36 arms**. Its A recall, 13,323/16,516 = **80.7%**, is largely density: equal-count uniform matches 14,120/16,516 = **85.5%**. Stop q25/dwell10 improves boundary placement in 34/36 arms but does not establish predictive risk value.

Waypoint ×1 is useful for trajectory compression: across 1,200 A episodes per suite, median knots at full control rate are **50 L10 / 21 Spatial**. The 95th percentile of episode maximum translation reconstruction error is **6.13 / 9.10 mm**, and rotation error **.00638 / .00775 rad**. At ten-control subsampling, median knots fall to **28 / 12**, while the corresponding errors rise to **36.56 / 22.34 mm** and **.08364 / .03566 rad**. Full-rate capture materially changes the reconstruction assessment. These are reconstruction numbers, not segmentation semantics or safety bounds.

### Does a candidate locate upcoming cache problems better than R7?

The benchmark's `gripper_r7` score is a confirmed flip in normalized **served-command history**. The actual R7 catalog comparator is weighted neighbor `event_near` mass; I reconstructed it and retained the last available retrieval on blind decisions. These are different baselines. The catalog score preserves R7's zero event contribution from unknown-stage members; it is not the entire CT event-plus-deviation policy.

Primary exploratory comparison: at-risk cache/cache-tail/follow decisions with a fully observed next-20-control window or absorbing success. Exclude failing episodes with no localized onset. End exposure at the first heuristic onset. Rank within each **arm/task**, selecting ceil(20% of windows), using identical deterministic tie coins for all candidates. Actual A exposure is **20.02–20.09%**. This is exact discovery exposure matching, not a frozen prospective threshold. CIs use 2,000 paired init resamples within fixed tasks, retaining repeated decisions; they are marginal discovery intervals without multiplicity correction or refitting.

The eight A arms supply **80,732 windows, 1,540 positive overlapping windows from 385 onset episodes**. Entries below are flagged positive-window counts; divide by the common positive denominator to obtain sensitivity. The flagged-window count is the common precision denominator for every score in that row.

| A cell | All windows / flagged | Positive windows | Catalog R7 | Gripper history | Waypoint ×1 | Change ×1 | Elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T L10-50 | 13,497 / 2,703 | 436 | 34 | 67 | 107 | 52 | 108 |
| GR00T L10-500 | 14,365 / 2,876 | 208 | 40 | 49 | 37 | 35 | 66 |
| GR00T Spatial-50 | 6,608 / 1,326 | 140 | 21 | 26 | 31 | 21 | 48 |
| GR00T Spatial-500 | 6,262 / 1,257 | 32 | 6 | 6 | 5 | 4 | 8 |
| π0.5 L10-50 | 13,954 / 2,794 | 316 | 40 | 47 | 84 | 31 | 68 |
| π0.5 L10-500 | 14,221 / 2,849 | 156 | 25 | 24 | 40 | 12 | 32 |
| π0.5 Spatial-50 | 5,674 / 1,139 | 216 | 43 | 33 | 45 | 26 | 31 |
| π0.5 Spatial-500 | 6,151 / 1,236 | 36 | 2 | 2 | 11 | 4 | 16 |

Waypoint ×1 improves over both R7 comparators in **6/8 A cells**; it reverses in both GR00T dense cells. Its sensitivity improvement versus catalog R7, in table order, is **+16.74, −1.44, +7.14, −3.13, +13.92, +9.62, +0.93, +25.00 percentage points**. Corresponding nominal 95% CIs: **[11.39,21.85], [−12.27,8.96], [−2.34,16.38], [−28.13,20.86], [6.34,21.10], [.49,18.55], [−6.47,8.18], [9.98,37.50]**. Dense Spatial has only **8 GR00T / 9 π0.5** contributing onset episodes; those estimates need particular restraint.

Even the promising sparse L10 scores have low precision: waypoint **107/2,703 = 3.96%** on GR00T and **84/2,794 = 3.01%** on π0.5, against event prevalences **436/13,497 = 3.23%** and **316/13,954 = 2.26%**. Their AUPRCs are **.0353/.0368**, versus catalog **.0278/.0219** and elapsed **.0523/.0631**. Across eight cells, mean window sensitivity is catalog **14.37%**, gripper history **15.92%**, waypoint **22.96%**, change **12.11%**, elapsed **27.08%**. Boundary-alignment improvement therefore does not imply anticipatory failure discrimination.

Distinct-event coverage differs from overlapping-window sensitivity. On GR00T L10-50, waypoint hits **79/109** onset episodes, versus catalog **21/109**; actionable hits with ≥5 controls lead are **71/109 versus 9/109**, with median flagged-positive lead **10 versus 4 controls**. On π0.5 L10-50 these are **54/79 versus 16/79**, actionable **46/79 versus 13/79**, median lead **9.5 versus 8**. These remain heuristic-onset leads.

As an exploratory time-confounding check, I repeated exposure matching within fixed **50-control elapsed bins and task**, using the same bins for all scores. Waypoint-minus-catalog deltas become positive in all eight A cells, but only three CIs exclude zero. Sparse GR00T L10 falls to **+9.40 pp [3.74,14.56]**, and sparse π0.5 L10 to **+4.75 pp [−3.71,12.66]**. Thus timing explains part of the apparent advantage; these post-analysis checks are not a replacement primary selection rule.

Continuation matters. The following equal-cell means summarize cache-origin windows only; the population and propensity to remain on cache differ across arms:

| Family | Eligible windows / positive windows / onset episodes | Catalog sensitivity | Waypoint sensitivity | Waypoint improvement cells |
|---|---:|---:|---:|---:|
| A | 80,732 / 1,540 / 385 | 14.37% | 22.96% | 6/8 |
| CU | 52,190 / 560 / 203 | 16.99% | 21.44% | 6/8 |
| CT | 52,541 / 569 / 202 | 27.77% | 21.75% | 1/8 |
| FL | 76,770 / 1,904 / 476 | 16.87% | 22.95% | 7/8 |

CT's reversal cannot establish that catalog segmentation is intrinsically better: CT has already changed actions, calls, and which decisions remain cache decisions. Similarly, FL survivor-age hazards cannot identify the effect of extending at an anchor. Use the randomized assignment tools with certified pre-choice stages for that question.

### Physical timelines and allocation context

T1 assigns onsets to **1,379/1,469 failed episodes**. Its labels include 660 grasp misses, 353 misplacements, 119 unknown releases, 107 drops, 81 undone predicates, 52 held-not-placed, seven drop/regrasp failures, and 90 failures without onsets. **712/1,379 localized failures** are grasp-miss or held-not-placed cases whose onset is backdated to first proximity or the start of the final carry interval. These are useful search locations but not independently established error times.

Among the 120 semantic-audit episodes, **99 succeeded**. Four successful episodes contain eight predicate losses; **three successes contain a loss lasting at least five controls**. Forensics assigns no onset to any success, so these observed recoveries never enter its failure-risk target. Of 195 `release` controls in that audit, **72 are all-goals-true controls with a closed command**, generated by the T2 terminal-state rule. The tag does not consistently denote an opening/release event.

The inspected GR00T L10 task0/init0 A card marks a grasp-miss onset at control **60**, when proximity is first detected; its command closes later and no sustained target lift follows. P10 on the same reset succeeds. The cards make such cases reviewable, but first proximity alone cannot distinguish a failed approach from an approach that later recovers.

Discovery package outcomes provide allocation context, not a newly selected stage policy:

| Cell | A successes | CU successes | CT successes | FL successes |
|---|---:|---:|---:|---:|
| GR00T L10-50 | 191 | 239 | 240 | 180 |
| GR00T L10-500 | 248 | 249 | 252 | 225 |
| GR00T Spatial-50 | 264 | 279 | 279 | 249 |
| GR00T Spatial-500 | 290 | 283 | 278 | 278 |
| π0.5 L10-50 | 215 | 264 | 267 | 194 |
| π0.5 L10-500 | 257 | 276 | 273 | 233 |
| π0.5 Spatial-50 | 240 | 278 | 282 | 233 |
| π0.5 Spatial-500 | 290 | 297 | 296 | 294 |

Every denominator is **300 discovery episodes**. Shared P10 references succeed **268/300 GR00T L10, 281/300 GR00T Spatial, 272/300 π0.5 L10, 297/300 π0.5 Spatial**. Pooled A/FL successes are **1,995/2,400 versus 1,886/2,400**; FL−A is **−4.54 pp [−6.13,−3.00]**. Recorded owner control-IR, `5×sum(owner_cost)/active_controls`, is **.07633 A versus .05614 FL**. CU/CT succeed **2,165/2,400 versus 2,167/2,400**, a **+0.08 pp [−1.21,1.38]** CT−CU difference at IR **.24676/.24770**. Intervals use 10,000 paired resamples within suite/task, keeping all model/library arms of each init together. These describe the collected packages and do not identify the value of a particular stage-triggered intervention.

## 4. Bugs and data-quality limitations

1. **No discovery-only CLI admission gate.** `physical/common.py:load_episodes` and `segmentation_bench.py:main/build` load and report holdout trajectories by default. Fitting excludes them, but exploratory execution can still expose their outcomes. The callback wrapper prevented this; a native `--split discovery` should enforce it before controls are loaded.
2. **Unknown-onset failures enter stock negative windows.** In `segmentation_bench.py:build`, a failing episode with `onset_control=None` supplies negative windows wherever 20 controls remain. Across the 32 cache arms, **5,543 such windows** were excluded from the primary comparison (1,420 in A). The original stock definition is preserved separately in the output. Endpoint observability and event absence need distinct fields.
3. **Outcome-conditioned onset and coarse stage semantics.** `forensics.py:physical` sets `carried = lifted & near & closed`; it does not use contact or measured aperture to certify acquisition. `grasp_miss` starts at `first(near)`; `held_not_placed` at the last carry's start. `analyse` sets successful episodes' onset to null. The all-goals-true T2 branch labels a closed command `release`. The 120-episode audit above demonstrates material consequences; this is the principal scientific limitation.
4. **Baseline ambiguity.** `segmentation.py:baseline_commands/r7_boundaries/causal_scores` replay command-history flips, whereas deployed R7 gating/tilting uses `StageTable.online` neighbor-stage/event mass and other conditions. Benchmark tables should name those separately. I supplied both comparisons, without treating either as the entire CT controller.
5. **Duplicate score rankings and differing tie lotteries.** Stop q10/q25 at a fixed dwell and change ×1/×2/×4 produce identical ranking-based flags: **zero differences in all 64 arm×endpoint-definition comparisons for each of five duplicate pairs**. The stock candidate-specific `tie_coin` can create apparent differences at tied thresholds. The 14 grid settings comprise nine distinct causal rankings; my comparison shares tie coins.
6. **Metric/uncertainty interpretation.** Stock `pr_metrics.onset_sensitivity` counts overlapping positive windows, not distinct onsets. Brier probabilities are fitted and scored on the same discovery observations. `physical/common.py:cluster_interval` pools task/init clusters rather than resampling in fixed tasks, and the benchmark interval is event prevalence rather than the paired candidate improvement. E1 reports both event/window denominators and paired fixed-task CIs; no calibration-validation claim is made.
7. **Catalog stage coverage in dense banks.** Positive unsuccessful-demo mass (>10⁻⁶) appears in **10,930/17,642 GR00T L10-500 A decisions**, **9,031/17,353 π0.5 L10-500**, **2,414/6,577 GR00T Spatial-500**, and **405/6,419 π0.5 Spatial-500**; all four sparse A banks have zero such decisions. R7 leaves these rows' stages unknown. Event-mass performance therefore also reflects library stage coverage; missing stage mass must not become certified easy exposure.
8. **Causal stage certification and feature availability.** `decision/common.py:stages` correctly distinguishes catalog-descriptive labels from `stage_pre/pre_stage`. The checked A/FL records lack the latter. Prefix-only physical scores can use client-side per-control measurements that are not necessarily delivered to the live server at that cadence. No posterior, live-observation parity, or intervention-stage certification should be inferred from a passing truncation check.

No corruption was observed in the selected required kinematic/predicate fields or decision joins. Contact-map correctness, snapshot restore, complete raw-key coverage, and all-arm augmentation quality remain outside this measured coverage.

## 5. What R9 should do

1. **Validate the event target first.** Use the proposed 200-window double annotation on discovery data, with uniform and boundary-centered sampling probabilities retained. Include successes and recoveries. Separate first observation, confirmation interval, persistent acquisition failure, slip, intended release, predicate loss, and unresolved timeout. Combine aperture, contact topology and object/EE co-motion; distinguish simulator observations from inferred intent. Keep uncertain cases unavailable.
2. **Retain change points as a descriptive candidate and waypoints as a reconstruction/risk hypothesis.** Keep exact catalog R7, causal gripper history, elapsed-time and equal-count controls. Test whether waypoint risk adds information after elapsed time, library support and stage coverage. Do not deploy “hard” stages from the present F1 or onset tables. Compare physical contact/object-relative interaction entries before adding HSMM/neural complexity.
3. **Use nested discovery evaluation before the locked holdout.** Group all arms/models of each task/init and reset alias; fit scales and risk thresholds only in training folds. Freeze one small candidate family, a primary 20-control target, 20% training exposure, actionable lead and multiplicity rules. Evaluate transfer by task/model/suite, then ask the coordinator to run the untouched 30–49 holdout once under the frozen protocol. Audit bank/init overlap and prior test-state reuse before calling that confirmatory; fresh R9 states may still be necessary.
4. **Log and consume pre-choice stages explicitly.** Save score/posterior or unknown, fit hash, feature timestamps, allowed sensors, stage source, and certified assignment-time stage alongside actual call/follow propensities. Replay portable features at their real decision/control availability, including blind checks. Stage risk, positive call value, and safe extra following remain separate estimands.
5. **Exploit existing randomized probes before adding many arms.** After augmentation and label validation, use IP/CU and FL assignment support to estimate call and extension effects in the frozen pre-choice strata. Preserve p=0/1 exclusions, original episode denominators, task/init clustering and small-support warnings. The current CT−CU and FL−A package results do not justify a new segmentation policy; a later equal-owner-IR prospective comparison should test one frozen rule across all eight cells.

The deliverable needed for R9 is a validated, available-before-choice event/risk specification with measured transfer and intervention support. R8 has supplied the raw evidence and most profiling machinery; the remaining work is chiefly label validity, evaluation discipline, and causal-stage integration.
