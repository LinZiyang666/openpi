# P5 handback: C stall component and Bmech

**Implemented, fitted, and checked.** `stall.py` exposes the exact `SELECTION.md §5` classes/methods. All eight library-only fits are under `/tmp/q3_stall_fits/<cell>/`; all eight independent refits match. Four Bmech specs and fitted artifacts are ready for the coordinator to copy. No server, worker, chain, simulator, remote host, or policy inference was started.

**Scientific caveat from the descriptive check:** at B’s no-progress-flagged anchors, `slow_confirmed` occurs at 169/361 (46.81%) on π0.5 L10-50, 378/802 (47.13%) on GR00T L10-50, and 25/40 (62.50%) on GR00T Spatial-50. This does not demonstrate selective suppression of the harmful cell. Constants were not changed in response. These are status overlaps on B trajectories, not retained causal benefits or predicted C SR. C’s shared budget/cooldown and changed state visitation still require closed-loop validation.

## 1. Integration and implemented recipe

Read [INTERFACE.md](INTERFACE.md) for the metric/key protocol. The preferred input reuses the A metric code already computed at retrieval, avoiding another visual projection:

```python
from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel, StallTracker
model = StallModel.load("/path/to/copied/cell/stall.pkl")
tracker = StallTracker(model, task_id)  # new at reset
tracker.observe({"metric_code": code, "metric": regime}, actual_executed_control_index)
diagnostics = tracker.status()
```

`observe` handles every fresh vision observation, including an extra LOOK. `status` returns cached diagnostics with state `inactive`, `ok`, `slow_confirmed`, or `slow_ambiguous`. It never requests inference, schedules a LOOK, samples a coin, or solves a budget. **C owns rho, its call priority/cooldown, and at most one ambiguous extra LOOK per W*L controls.** The previous Q3 step-5 q/B-budget rule is absent. B’s three other guards are not added to C by this component; they exist only in the separate Bmech control.

Implemented steps 1–4:

1. Read H/R from manifest `H`/`exec_steps`; require L=min(H,2R). Legacy library timestamps are `step*exec_steps`; a generic input may supply actual `control_index`. Phase uses the final **observed** library timestamp as phase 1, consistent with `step/(ep_len−1)` for complete regular episodes. T_med is the median final-observation duration/L for successful episodes of that task. W=max(2,ceil(.05*T_med)). Resample at L-control grid points using the latest observation at/before each point, remove repeated selections, and retain the final observation. Templates and LOEO windows use those sampled rows.
2. Align W+1 query observations independently against each successful template with unrestricted nondecreasing indices (repeats and skips allowed), minimizing summed A-metric Euclidean distances. A backward suffix DP followed by earliest optimal indices gives the lexicographically earliest complete path. Template ties use episode identifier order. Use K=min(5,E−1) both online and in LOEO; multiply advances/spread by W*L/actual span. Every quantile, including the median, uses the specified inverse ECDF (the lower middle value at even n), not an interpolated median.
3. On every eligible successful-library window, exclude its entire episode from templates. Keep known phase advance, estimated phase/distance/spread, and true-minus-estimated residual. Build episode-equal CDFs for distance/spread. At a current context, select one nearest L1 context per reference episode, with earliest window breaking ties; e90 is its residual q90 and a10 its true-advance q10. Require ≥4 contributing reference episodes. The fitted representation stays fixed; this is empirical conditional calibration, not a conformal theorem.
4. `delta_hat+e90<a10` gives confirmed slow; `delta_hat<a10<=delta_hat+e90` gives ambiguous; otherwise ok. Unknown task, insufficient references, warmup, or invalid observation yields inactive. Invalid observations clear the window. Duplicate/decreasing control timestamps raise an error rather than creating artificial progress. No robot displacement, gripper, contact, suite, or task-name threshold is used.

Library success metadata only selects successful demonstrations, as requested. No evaluation success label enters a fit, and no recorded calibration trajectory is required for this component. The B-val recordings and shared-budget solution belong to P4.

## 2. Artifacts and reproducibility

The eight fits contain **33,626 LOEO calibration windows**. Each directory contains `stall.pkl`, `stall.json` (payload hash, model fingerprint, actual bank-file hashes, source fit hash, task metadata), and `stall.loeo.csv`. Copy the `.pkl` and `.json` together; the CSV is the inspectable numerical table. The artifacts are self-contained and require no source-bank path at load time. Load verifies both the serialized payload hash and deterministic content fingerprint. Only trusted artifacts should be loaded (pickle is not a sandbox).

| Cell | Successful E/task | W | LOEO windows | Artifact MiB | Fingerprint prefix |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 5–5 | 2 | 1257 | 23.28 | e4ed6e3d0ed45115 |
| pi05_l10_500 | 25–50 | 2/3 | 10839 | 74.98 | 312754f81b3971dd |
| pi05_sp_50 | 4–5 | 2 | 443 | 19.73 | f93f6c830d4411b2 |
| pi05_sp_500 | 45–50 | 2 | 4554 | 41.43 | 4ca05097b54dfc17 |
| groot_l10_50 | 5–5 | 2 | 1262 | 23.30 | e962aa7d665a23a6 |
| groot_l10_500 | 29–48 | 2/3 | 10458 | 74.25 | f5426ac559b27059 |
| groot_sp_50 | 5–5 | 2 | 469 | 19.84 | 3def6cdcb5a8c12d |
| groot_sp_500 | 36–49 | 2 | 4344 | 41.69 | 1fb0d9754c2100d2 |

All 80 tasks meet the ≥4-reference rule. Only task 8 in the two L10-500 cells has W=3; all other tasks have W=2. Independent full refits, not merely load/save roundtrips, reproduce every fingerprint (which includes every LOEO record and reference array). See `fit_<cell>.json`, `refit_<cell>.json`, [source_manifest.json](source_manifest.json), and [delivery_manifest.json](delivery_manifest.json). Fit artifacts are intentionally outside the source tree.

### Deployed metric verification and numerical convention

[metric_audit.json](metric_audit.json) verifies that A/B fitted metric arrays and projections are identical in all eight cells, and the saved stall metric matches them. **1,920 raw-query checks** across both regimes reproduce A’s query metric codes exactly and return the same tracker status as passing those codes directly. The early observation remains in the early coordinate system even inside a mixed early/main window; candidate template codes use that observation’s regime.

Distances are evaluated as float64 Euclidean norms of these fixed metric codes. They are not bit-identical to A’s float32 squared-norm/dot-product evaluation (particularly near self matches), and action-tail continuity reranking is not included in the alignment metric. No PCA, learned metric, or kernel is refitted. The recorded audit’s largest absolute distance difference is 0.499870; when excluding the query’s entire source episode, the maximum is 0.034908, maximum relative difference 1.0458%. This numerical convention was fixed before the pilot replays; the same arithmetic builds LOEO and serves online. Do not label the distance evaluation bit-identical to the old A implementation.

## 3. Pilot status rates — descriptive / engineering only

Replayed all A/B cohorts: **960 episodes and 20,836 anchors**, 60 episodes per cohort/cell. Inputs are the official strict tables at `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/`. Timestamps are cumulative preceding `decisions.actual_controls`, joined to each anchor by (arm,uid,attempt,step); no nominal step count substitutes for actual execution. The replay reads no Y column. Every model was already fitted before these reads. Status counts were identical when the CPU timer was added. Warmup is included in the denominator; no interval or SR interpretation is claimed.

| Cell | Cohort | Anchors | Inactive % | Ok % | Confirmed % | Ambiguous % |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | A | 2055 | 5.84 | 60.44 | 18.69 | 15.04 |
| pi05_l10_50 | B | 1784 | 6.73 | 68.27 | 13.79 | 11.21 |
| pi05_l10_500 | A | 1657 | 7.60 | 84.13 | 5.37 | 2.90 |
| pi05_l10_500 | B | 1664 | 7.57 | 81.01 | 7.09 | 4.33 |
| pi05_sp_50 | A | 813 | 14.76 | 55.60 | 15.87 | 13.78 |
| pi05_sp_50 | B | 761 | 15.77 | 61.37 | 12.09 | 10.78 |
| pi05_sp_500 | A | 686 | 17.49 | 74.34 | 7.14 | 1.02 |
| pi05_sp_500 | B | 675 | 17.78 | 76.30 | 5.48 | 0.44 |
| groot_l10_50 | A | 2234 | 5.37 | 42.21 | 40.06 | 12.35 |
| groot_l10_50 | B | 2155 | 5.57 | 51.88 | 26.96 | 15.59 |
| groot_l10_500 | A | 1862 | 6.77 | 63.59 | 19.12 | 10.53 |
| groot_l10_500 | B | 1721 | 7.32 | 70.37 | 14.00 | 8.31 |
| groot_sp_50 | A | 706 | 17.00 | 66.29 | 8.36 | 8.36 |
| groot_sp_50 | B | 687 | 17.47 | 68.41 | 5.97 | 8.15 |
| groot_sp_500 | A | 678 | 17.70 | 77.29 | 3.10 | 1.92 |
| groot_sp_500 | B | 698 | 17.19 | 74.07 | 7.02 | 1.72 |

### Overlap with B’s no-progress flags

| Cell | NP-flagged anchors | Confirmed count (%) | Ambiguous count (%) | Ok count | Inactive count |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 361 | 169 (46.81%) | 63 (17.45%) | 123 | 6 |
| pi05_l10_500 | 285 | 72 (25.26%) | 27 (9.47%) | 186 | 0 |
| pi05_sp_50 | 127 | 68 (53.54%) | 37 (29.13%) | 21 | 1 |
| pi05_sp_500 | 46 | 24 (52.17%) | 1 (2.17%) | 21 | 0 |
| groot_l10_50 | 802 | 378 (47.13%) | 160 (19.95%) | 254 | 10 |
| groot_l10_500 | 446 | 137 (30.72%) | 60 (13.45%) | 243 | 6 |
| groot_sp_50 | 40 | 25 (62.50%) | 6 (15.00%) | 9 | 0 |
| groot_sp_500 | 81 | 34 (41.98%) | 2 (2.47%) | 42 | 3 |

These include co-fired guard bits; they are not isolated NP-only calls. A-cohort overlaps use the profiler’s shadow B diagnostic and are exported separately. Source files: `pilot_status_<cell>.csv`, `pilot_replay_<cell>.json`, and [status_rates.csv](status_rates.csv). Status eligibility does not equal actual C calls or LOOKs after its shared budget, and this recorded replay cannot estimate C’s future trajectory/SR.

## 4. Per-anchor CPU cost

The main table times only `observe()` on the actual pilot code streams. CPU time uses `process_time_ns`; wall latency uses `perf_counter_ns`. It includes code validation/copy, alignment, reference lookup, and status construction. It excludes CSV/JSON parsing, model loading, A’s existing key projection/retrieval, and the cheap subsequent `status()` copy. Warmup anchors are included. At most four analysis processes ran concurrently, each with one BLAS/OpenMP thread and affinity 22–25,66–69; these are host measurements, not a real-time guarantee on another robot.

| Cell | Anchors | CPU median ms | CPU p99 ms | Wall median ms | Wall p99 ms |
| --- | --- | --- | --- | --- | --- |
| pi05_l10_50 | 3839 | 0.816 | 1.069 | 0.813 | 1.069 |
| pi05_l10_500 | 3321 | 5.557 | 7.214 | 5.554 | 7.210 |
| pi05_sp_50 | 1574 | 0.571 | 0.784 | 0.568 | 0.781 |
| pi05_sp_500 | 1361 | 3.965 | 4.992 | 3.961 | 4.990 |
| groot_l10_50 | 4389 | 0.651 | 0.870 | 0.648 | 0.868 |
| groot_l10_500 | 3583 | 4.295 | 5.063 | 4.292 | 5.102 |
| groot_sp_50 | 1393 | 0.555 | 0.669 | 0.552 | 0.674 |
| groot_sp_500 | 1376 | 3.886 | 4.454 | 3.884 | 4.480 |

Full timing samples are in the status CSVs and [timing.csv](timing.csv). `metric_audit.json` also measures the raw-query adapter, including its additional projection, on 240 library queries per cell and both regimes. That synthetic warmup-heavy microbenchmark is explicitly separate from the pilot latency distribution. P4 should reuse the already available metric code.

## 5. Bmech and replay identity

[bmech.py](bmech.py) subclasses the deployed π0.5/GR00T B judges. The parent computes all diagnostics; only returned `os_flags`, `os_reason`, and `os_force_miss` are rederived with bit 8 masked. The raw diagnostic memo `_s["flag"]`, `_vision_progress`, `_noprog_span`, all other histories, cache action/score/confidence, blind LOOK veto, and policy-tail lifecycle are retained. It deliberately does **not** inherit the P2 ablation’s `blind_step` override. `mask_no_progress=False` is the exact B identity used in tests; emitted arms use its default True.

[emit_arms_bmech.json](emit_arms_bmech.json) contains the four 50-library arms. Deployed B kwargs, client overrides and serving flags are copied; only the name, subclass, fit destination and explicit `--os-root /home/weiland/trace_runs/offline_search_store` are changed as needed. The fit is transplanted without any retraining. Copy the following files to each emitted `<RUN>/fits/<name>.pkl`:

| Name | Source artifact to copy |
| --- | --- |
| r6p5_bmech_g_l10_50 | /tmp/q3_stall_fits/bmech/r6p5_bmech_g_l10_50.pkl |
| r6p5_bmech_g_sp_50 | /tmp/q3_stall_fits/bmech/r6p5_bmech_g_sp_50.pkl |
| r6p5_bmech_p_l10_50 | /tmp/q3_stall_fits/bmech/r6p5_bmech_p_l10_50.pkl |
| r6p5_bmech_p_sp_50 | /tmp/q3_stall_fits/bmech/r6p5_bmech_p_sp_50.pkl |

On 20 recorded trajectories per cell (ten tasks × inits 0/49), B, Bmech, and mask-disabled Bmech receive the **same history**, driven by Bmech decisions and same-observation recorded policy chunks. The unmasked result equals B bit-for-bit; the masked result differs only in the three verdict fields. All diagnostic memos, blind results, and policy tails are checked. Masked NP-only decisions actually exercise the still-active next-step LOOK veto in this replay.

| Cell arm | Decisions | Vision queries | NP-only MISSes masked | NP LOOK vetoes retained | Tail comparisons |
| --- | --- | --- | --- | --- | --- |
| r6p5_bmech_g_l10_50 | 1383 | 849 | 306 | 303 | 43 |
| r6p5_bmech_g_sp_50 | 529 | 307 | 75 | 74 | 25 |
| r6p5_bmech_p_l10_50 | 1615 | 953 | 285 | 281 | 174 |
| r6p5_bmech_p_sp_50 | 623 | 342 | 51 | 49 | 79 |

Total: **4,150 decisions**, 2,451 exact result comparisons per judge, 717 NP-only verdict changes, 707 retained NP LOOK vetoes; all checks pass. A preliminary B-driven-history coverage assertion failed because that history did not exercise NP LOOKs; the test driver was corrected to Bmech while preserving identical inputs across compared judges. No method/threshold was changed to obtain the pass. Different deployed controllers need not have identical future histories after a changed MISS. This test does not claim closed-loop SR parity.

## 6. Tests, commands, and limitations

**13 unit tests passed** via the command in [commands.txt](commands.txt): exhaustive nondecreasing-path oracle checks on 150 integer-cost matrices with ties, inverse-ECDF/episode weighting, successful-only LOEO, minimum reference support, model roundtrip/corruption/no-overwrite, mixed regimes, variable control span, warmup/missing data/timestamps, all status inequalities, every four-bit guard combination, and mask-disabled B identity. Non-default synthetic H=11, R=3 checks that the core does not assume LIBERO control lengths. All eight real-bank independent refits and the recorded Bmech/metric checks above also passed.

All commands were run from `/home/weiland/projects/openpi`; the complete expanded list is [commands.txt](commands.txt). Initial fitting/emission refuse to overwrite artifacts. To reproduce an existing fit, use `--verify-reproducible` (recomputes fully and checks the fingerprint):

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.fit_models --cell pi05_l10_50 --verify-reproducible
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.replay_pilot --cell pi05_l10_50
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.replay_bmech --name r6p5_bmech_p_l10_50
```

Only this `stall/` directory and `/tmp/q3_stall_fits/` were written. Source code and result hashes are in [delivery_manifest.json](delivery_manifest.json). The controller integration and budget replay are P4’s work; they were not executed here. The component neither establishes progress-calibration coverage on shifted cache states nor certifies beneficial calls. No LIBERO worker, closed-loop smoke, new benchmark, or robot validation was run. The overlap counterexample above must remain visible when interpreting C’s eventual results.

