# Exact online stall optimization: verification and timing

**All equivalence checks pass; performance targets are missed on this host.** The eight-cell pilot replay compares 20,836 anchors / 960 episodes and 427,679 per-template distance matrices and complete paths, with zero differences and maximum absolute float difference 0. All 1,920 raw-query audit checks pass. All eight production calibration JSONs and their cost-replay JSONs are exactly equal to production; 160 budget solutions and 5,052 cadence nodes are covered. Sixteen artifact copies load and verify with unchanged payload hashes, sidecars and content fingerprints.

Observe-only CPU medians are 0.340–0.386 ms for 50-episode cells and 0.569–0.888 ms for 500-episode cells. Median speedups are 1.89–8.42×. None meets the requested ≤0.1 / ≤0.3 ms target. No numeric tolerance or tie-breaking relaxation was used.

## 1. Implementation and files

- `stall.py`: `_template_cache` derives per-task/per-regime contiguous float64 codes stacked over valid points, padded phases, valid-point masks and episode identities. `_distances` retains the original subtraction / `einsum("ij,ij->i")` / square-root arithmetic. `_estimate_distances` and `_monotone_alignment_batch` batch the backward suffix DP and greedy earliest-path recovery over all templates. Stable sorting preserves (mean, original template order). Quantile indices are the same inverse ECDF, including the lower middle at even K. `_reference_cache` and `calibrated_status` batch nearest L1 contexts over all reference episodes; padding is masked and first argmin preserves earliest ties.
- `StallTracker.observe`: each tracker has a private bounded distance deque. The first full window computes its W+1 columns once; subsequent anchors compute only the newest observation. Invalid observations clear both deques. A distance overflow raises the same ValueError at the same first complete window as before, and clears the distance cache for correct recovery if the caller continues. Distance computation is deliberately deferred during warmup to preserve the original exception timing.
- Repeated alignment validation and five estimate-quantile conversions/checks per anchor are gone. Public `encode`, `monotone_alignment`, timestamp validation and `status` copies retain their original behavior. Encoded input snapshots remain owned/read-only, so caller mutation cannot corrupt a window. The derived caches are ordinary private model attributes outside `_payload`, never inserted into `tasks`, metric, provenance or serialized artifacts. The model remains immutable by contract.
- `stall_reference.py`: byte-for-byte snapshot of the original entire module, SHA256 `8d595cf8e0eafe45e94a59d93095400d35ed2b355f5c4d0ffc10a4620775c0e3`. `_estimate_reference` / `calibrated_status_reference` in the new model expose it; its own model/tracker classes also retain the original full boundary and fit behavior.
- `test_stall_fast.py`: seven permanent parity tests; `test_stall.py` retains all 13 existing assertions/tests, with its temporary-output setup using default-location `tempfile.mkdtemp(prefix="_test_model_")` and `self.addCleanup(shutil.rmtree, path)` for exactly that test-created directory (§8).
- `verify_fast.py`: separate artifacts, pilot, timing, audit, calibration, logs and component-breakdown stages, reading real inputs without changing consumers. `build_fast_handback.py` renders this report, `timing_fast.csv`, `timing_fast_samples.csv` and `verification_fast.json` from scratch evidence.

The initial tracker optimization changed no consumer file; the follow-up changes only `method_c/methods.py` to use the existing metric-code protocol (§8). Consumer interfaces, `StallModel.fit/load/save`, `_payload`, SCHEMA, projection, CDFs, serialized templates/calibration, and artifact overwrite refusal remain intact. No production artifact, arm, spec, run root, or fitted model was rewritten. No real bank was refitted.

## 2. Exact equivalence

The pilot joins/filter/order/timestamps are exactly those in `replay_pilot.py`: only A/B cohorts; strict decisions and anchors joined one-to-one by (arm, uid, attempt, step); control timestamp = cumulative preceding `actual_controls`. Inputs are the strict tables under `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/`. No Y column is read. Warmup anchors are compared and counted.

At every anchor the complete status dictionary is compared recursively: state/reason, `delta_hat`, `phase_hat`, `distance_hat`, `spread_hat`, `e90`, `a10`, `reference_windows`, `window_span`, `W`, `K`, `reference_episodes`. Floats are compared by packed float64 bytes, including sign bits; integers/list order/dictionary keys match. At every active anchor, independently recomputed reference distances and paths are compared byte-for-byte for every template, and the selected template identities/top-K order match. All fields and all cells have zero differences.

| Cell | A anchors | B anchors | Total | Active | Template distance/path pairs | Differences |
| --- | --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 2234 | 2155 | 4389 | 4149 | 20745 | 0 |
| groot_l10_500 | 1862 | 1721 | 3583 | 3331 | 138932 | 0 |
| groot_sp_50 | 706 | 687 | 1393 | 1153 | 5765 | 0 |
| groot_sp_500 | 678 | 698 | 1376 | 1136 | 52121 | 0 |
| pi05_l10_50 | 2055 | 1784 | 3839 | 3599 | 17995 | 0 |
| pi05_l10_500 | 1657 | 1664 | 3321 | 3069 | 130822 | 0 |
| pi05_sp_50 | 813 | 761 | 1574 | 1334 | 6575 | 0 |
| pi05_sp_500 | 686 | 675 | 1361 | 1121 | 54724 | 0 |

Synthetic parity covers 14 randomized complete fits (W=2 and W=3; dimensions 1, 2, 3, 7, 16, 17, 136 and different early dimensions), 630 observation events, integer costs/code ties plus noninteger queries, variable template/reference lengths, mixed early/main windows, exclusions, span scaling, invalid observations and warmup. Every synthetic fitted payload and fingerprint equals the frozen fit. There are also 120 batched DP cases checked against the frozen scalar DP and an exhaustive path oracle; an all-contexts-tied reference test; cache call-count / episode-isolation / caller-mutation tests; timestamp/type/shape/nonfinite rejection tests and overflowing-distance recovery; and real-bank tests on π0.5 L10-50/L10-500, tasks 0 and 8 (including real W=3). A temporary save test proves caches do not change serialized payload bytes.

Artifact evidence covers both `/tmp/q3_stall_fits/<cell>/` and `/home/weiland/trace_runs/os_closed_loop/r06_c_cal/stall/<cell>/` (all 16). Both the new and frozen loaders verify each payload SHA256/content fingerprint; loaded payloads are byte/field equal; fingerprints are rechecked after all task caches are populated; files/sidecars are hashed again and unchanged. See `verification_fast.json` for full fingerprints/hashes.

`audit_metrics.main()` ran verbatim against the new code, with only its output directory redirected to `/tmp/codex_stall_fast/metric_audit/`. Every cell has 240 raw-query checks: A/B saved metrics match, projected raw codes equal deployed A exactly in both regimes, and raw/code tracker statuses match. The original `metric_audit.json` was preserved.

## 3. Production calibration rerun

The unchanged `fit_calibration.fit` was rerun in production mode (not dry-run), using the original R bank, strict B-val tables, frozen `calibration_manifest_<cell>.json`, actual reset attestation client telemetry, original stall artifact, c1=0.152 (π0.5) / 0.148 (GR00T), targets 0.18/0.30/0.45 and stall cooldown. The inputs were traced from production `calibration.json`, manifests, `chain_calibration.sh` (read only), and `fit_*.log` / `read_v2_*.log`. Spatial production directories use `spatial`, while stall artifact/pilot cells use `sp`.

Outputs are `/tmp/codex_stall_fast/calibration_verified/<cell>/`. The consumer output allowlist only accepts `method_c/` or `/tmp/q1_method_c_fits/`; a runtime-only `unittest.mock.patch` replaces that single output-path function with a check permitting exactly the authorized scratch destination. No fitting, validation, reset attestation, budget calculation, hashing, or overwrite check is patched. An initial comparator checked the returned Python dict against parsed JSON and encountered int task keys versus serialized string keys. The verifier now compares emitted JSON to emitted JSON; the first scratch result is retained separately, and the full successful rerun used fresh outputs.

The **entire emitted calibration JSON** equals production, without discarding even timestamps/paths. This includes λ/d parameters, floors, ceilings, feasibility, coefficients, library budgets, every stall/no-stall and R/uniform solution, provenance/hash/reset fields and stall fingerprints. The entire emitted `cost_replay.json` also equals production (topology, compact stall states/diagnostics and cost inputs). Each cell covers 20 solutions; totals: 80 episodes, 1,817 labeled anchors, 160 solutions and 5,052 cadence nodes.

| Cell | Anchors | No-stall / stall nodes | Solutions | Stall floor | Stall ceiling | JSON / replay differences |
| --- | --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 365 | 365 / 851 | 20 | 0.153875017851 | 0.440704194944 | 0 / 0 |
| groot_l10_500 | 290 | 290 / 475 | 20 | 0.106482772266 | 0.476721493845 | 0 / 0 |
| groot_sp_50 | 133 | 133 / 208 | 20 | 0.119061009895 | 0.475642604002 | 0 / 0 |
| groot_sp_500 | 122 | 122 / 135 | 20 | 0.088943942339 | 0.497769634969 | 0 / 0 |
| pi05_l10_50 | 346 | 346 / 725 | 20 | 0.133324939828 | 0.461342451012 | 0 / 0 |
| pi05_l10_500 | 274 | 274 / 376 | 20 | 0.0917242546233 | 0.489185276568 | 0 / 0 |
| pi05_sp_50 | 152 | 152 / 255 | 20 | 0.125752969697 | 0.466940744449 | 0 / 0 |
| pi05_sp_500 | 135 | 135 / 210 | 20 | 0.109914921243 | 0.495200268348 | 0 / 0 |

Full expanded input paths, targets and fingerprints are in `verification_fast.json` and scratch `calibration_<cell>.json`; rerun products include the original bank copied into the scratch output directory by the unchanged production fitter.

## 4. Recorded C validation decision logs

Inspected all 12 recorded arms with a configured stall artifact: 242,482 decisions and 124,728 fresh anchors under `/home/weiland/trace_runs/os_closed_loop/r06_c_validation/runs/`. Fresh records log `os_c_stall_state`, control index and scalar diagnostics, but none contains a metric code, raw visual key, or input-archive pointer. Retrieval rows/weights/scores and `robot_state` cannot recover the visual query code. Therefore logged stall codes cannot be independently replayed from these logs; no agreement count is claimed. Counts describe the read snapshot. Per-arm counts are in `verification_fast.json` / `/tmp/codex_stall_fast/decision_log_check.json`.

## 5. Observe-only timing and remaining cost

Methodology reproduces HANDBACK §4: actual pilot code lists/regimes, every fresh anchor including warmup, only `observe()` timed with `process_time_ns` and `perf_counter_ns`. CSV/JSON parse, joins, model loading, A retrieval/projection and the following `status()` copy are outside the interval. Lazy per-task preparation is included when first needed. Fresh models/trackers are used in two standalone full passes per cell (reference then fast); equivalence/path instrumentation runs separately. Timing had one analysis Python process active, one BLAS/OpenMP thread and affinity 22–25,66–69. No hardware isolation or real-time guarantee is claimed. Percentiles use the same pandas median/quantile convention as `replay_pilot.py`; float computation itself uses inverse ECDF.

| Cell | CPU median ms ref → fast | CPU p99 ms ref → fast | Wall median ms ref → fast | Wall p99 ms ref → fast | CPU median speedup |
| --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 0.767 → 0.383 | 0.951 → 0.576 | 0.764 → 0.380 | 0.946 → 0.573 | 2.00× |
| groot_l10_500 | 5.373 → 0.877 | 6.362 → 1.896 | 5.370 → 0.874 | 6.359 → 1.888 | 6.12× |
| groot_sp_50 | 0.658 → 0.343 | 0.738 → 0.483 | 0.655 → 0.340 | 0.735 → 0.480 | 1.92× |
| groot_sp_500 | 4.795 → 0.569 | 5.313 → 1.123 | 4.792 → 0.566 | 5.310 → 1.120 | 8.42× |
| pi05_l10_50 | 0.781 → 0.386 | 0.931 → 0.584 | 0.778 → 0.382 | 0.931 → 0.584 | 2.02× |
| pi05_l10_500 | 5.335 → 0.888 | 6.828 → 1.977 | 5.333 → 0.885 | 6.825 → 1.974 | 6.00× |
| pi05_sp_50 | 0.643 → 0.340 | 0.745 → 0.494 | 0.641 → 0.337 | 0.745 → 0.491 | 1.89× |
| pi05_sp_500 | 5.050 → 0.625 | 5.485 → 1.328 | 5.047 → 0.621 | 5.492 → 1.325 | 8.08× |

All eight target medians are missed. The retained float64 norm and NumPy DP/reference-array operations remain material. Below are CPU component medians on a separate instrumented pilot pass (including first-use work); the method timers add overhead, and these medians are not additive. Distance call counts equal anchor counts: the initial complete window fills all columns, then exactly one new column is computed per fresh anchor.

| Cell | Encode ms | Newest-distance ms | DP + estimate ms | Calibrated status ms |
| --- | --- | --- | --- | --- |
| groot_l10_50 | 0.036 | 0.084 | 0.157 | 0.101 |
| groot_l10_500 | 0.042 | 0.398 | 0.234 | 0.179 |
| groot_sp_50 | 0.036 | 0.066 | 0.153 | 0.098 |
| groot_sp_500 | 0.037 | 0.193 | 0.199 | 0.142 |
| pi05_l10_50 | 0.036 | 0.091 | 0.158 | 0.102 |
| pi05_l10_500 | 0.038 | 0.434 | 0.240 | 0.186 |
| pi05_sp_50 | 0.036 | 0.062 | 0.150 | 0.096 |
| pi05_sp_500 | 0.038 | 0.235 | 0.215 | 0.153 |

`timing_fast.csv` contains 16 full-precision summary rows. `timing_fast_samples.csv` contains 41,672 per-anchor samples (20,836 each implementation). Scratch `timing_<implementation>_<cell>.csv` and `breakdown_<cell>.json` retain the separate inputs/results. Existing `timing.csv` and original pilot status CSVs were preserved.

## 6. Retrieval-distance reuse estimate (not implemented)

Reusing A distances could at most remove the remaining new-observation distance computation, roughly 0.062–0.434 ms per anchor in the component measurement, before row gathering, square roots and padding overhead. The per-cell upper-bound estimate is the newest-distance column above; it would not remove DP, reference lookup or raw-key projection. No retrieval-reuse path or experiment was implemented.

It is not equivalent: A computes float32 squared-norm/dot distances; stall uses float64 subtraction and `einsum` Euclidean norms, without a floor or action-tail continuity term. The rerun metric audit records a largest absolute distance discrepancy 0.499870; excluding the source episode the maximum is 0.034908. Such substitutions violate bitwise `distance_hat` and could alter alignment, context selection and state inequalities. Reusing the already computed metric code remains the exact supported boundary.

## 7. Commands, results, and limitations

All commands run from `/home/weiland/projects/openpi` (branch Ziyang). Every Python invocation used the prefix below, at most four Python processes at once; only one during timing/breakdown. No server, chain, worker, simulator, GPU, remote host, tmux session, port or unrelated process was started or touched. The initial optimization products are confined to this `stall/` directory and `/tmp/codex_stall_fast/`; §8 additionally changes the authorized consumer `methods.py`. Initial existing-test runs left new `_test_model_*` temporary directories directly under `/tmp`; they are retained. Follow-up tests create their own default-location temporary directory and remove exactly that directory through unittest cleanup. No existing file was deleted; the pre-existing untracked `_test_model_5beh_ykn/` directory remains untouched. Unrelated pre-existing repository changes remain untouched. `tests/review_tests/` was not read.

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m unittest exp.offline_search.rounds.r06.ideation_Q3.stall.test_stall -v
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m unittest exp.offline_search.rounds.r06.ideation_Q3.stall.test_stall_fast -v
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast artifacts
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast pilot
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast audit
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast calibration
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast logs
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast timing
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_fast breakdown
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.build_fast_handback
```

The two unittest commands pass: 13 existing tests and 7 new parity tests. Artifacts/pilot/audit/calibration/log-inspection/timing/breakdown stages exit 0; result counts are above. Scratch logs preserve complete output. Production calibration continues refusing existing outputs; a second calibration invocation must use a fresh scratch destination (or compare the existing scratch JSON read-only). Other verification products can be regenerated.

Limitations: latency targets remain unmet on this host; model caches use extra memory and assume the documented immutable model contract; NumPy/CPU floating-point portability beyond the tested environment is not asserted. Timing in §5 concerns metric-code input. The follow-up removes the consumer's duplicated raw projection and measures controller overhead in §8; the public raw-query projection remains available. Validation log replay lacks inputs. No closed-loop trajectory/SR claim follows from fixed-stream equivalence or unchanged recorded calibration; the original HANDBACK scientific limitations still apply.

Files written: `stall.py`, `stall_reference.py`, `test_stall.py`, `test_stall_fast.py`, `verify_fast.py`, `build_fast_handback.py`, `HANDBACK_FAST.md`, `timing_fast.csv`, `timing_fast_samples.csv`, `verification_fast.json`; follow-up files are listed in §8. Other outputs are under `/tmp/codex_stall_fast/`.

## 8. Controller path (follow-up)

**The deployed C consumer now supplies A's already computed metric code.** All 8,879 before/after controller anchor comparisons have zero differences (maximum absolute float difference 0), including 320 early and 8,559 main codes. All eight production calibration and cost-replay JSONs are byte-equal after another production rerun. The 22 unit tests, 16 existing plugin replays and existing result/packaging checks pass.

On real B-val inputs following C call/LOOK cadence, the **end-to-end incremental C/stall CPU cost per fresh observation** falls from 1.181–1.914 ms to 0.516–1.009 ms across cells (1.89–2.48× median speedup). This includes tracker inference, status, code handoff and unchanged C decision bookkeeping; the original A work is excluded as described below. The float64 distance/DP/reference path is unchanged.

**Consumer change.** Only `ideation_Q1/method_c/methods.py` changes outside `stall/`. `_query_with_metric_code` calls the original A `query`/`_dist` once and captures the local code at the matrix multiplication that consumes it. `_MetricCodeMatrix.__matmul__` forwards to the original ndarray with the same operand. `_metric_distance` uses shallow method/task facades so fitted matrices and the real task dictionary are unchanged; A query state updates remain on the real base. The temporary `_dist` hook is restored in `finally`, including instance overrides and failure paths. No projection or retrieval arithmetic is copied or recomputed.

The tracker receives `{"metric_code": code, "metric": regime}` after successful A retrieval, with `early` exactly when `step == 0 and base.early`, otherwise `main`. Main capture uses `Z`; early capture uses `Z0`, or `A0` for the unstored early transform, before its code is transformed into main coordinates. Every fresh `query`, including an extra LOOK, uses this path and the same `step * block_controls` timestamp. No-stall stubs retain their original raw-query protocol. A failed query still sends the raw observation to preserve invalid-window clearing before propagating the failure. `stall_bridge.py`, public interfaces, controller kwargs, log fields, budget rules, artifacts, specs, arms and fitted models are unchanged.

**Controller equivalence on real inputs.** The frozen before-consumer source is `/tmp/codex_stall_fast/controller_followup/methods_before.py`; both versions use the same current exact tracker and actual production controller/calibration arguments. Input vectors and executed chunks come from the tables' `absolute_input_archive` NPZs. Recorded A history and preceding executed action are supplied; no Y field is read. All 80 B-val episodes are checked twice: all 3,606 recorded/diagnostic observations (covering every one of the 1,817 actual calibration anchors), then 1,867 fresh observations at C cadence, advancing one request after an extra LOOK and two otherwise. Pilot sampling checks every actual anchor in the first episode of each task and A/B cohort: 160 episodes / 3,406 anchors.

| Cell | Actual B-val anchors covered | All B-val observation probes | C-cadence probes | Pilot anchors | Differences |
| --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 365 | 727 | 380 | 704 | 0 |
| groot_l10_500 | 290 | 577 | 295 | 564 | 0 |
| groot_sp_50 | 133 | 262 | 137 | 241 | 0 |
| groot_sp_500 | 122 | 241 | 123 | 239 | 0 |
| pi05_l10_50 | 346 | 688 | 357 | 609 | 0 |
| pi05_l10_500 | 274 | 546 | 279 | 586 | 0 |
| pi05_sp_50 | 152 | 301 | 159 | 244 | 0 |
| pi05_sp_500 | 135 | 264 | 137 | 219 | 0 |
| TOTAL | 1817 | 3606 | 1867 | 3406 | 0 |

At every probe, raw projection, recorded A code and code delivered through the actual new controller are equal by packed float bytes after the public encode boundary, with the same regime. The entire stall status matches, including all float diagnostics, reference windows, W/K/counts/span and state. All retrieval result arrays/scalars/extras and C internal/log fields match: Ehat, nominal p, p, coin, call, stall call, cooldown, scheduled extra LOOK, control/anchor bookkeeping and the policy-tail gate. These streams exercise 4,557 calls, 1,077 cooldown anchors and 337 scheduled LOOK flags. The unchanged plugin replay separately verifies 35 realized extra LOOKs and their next fresh observations on both models. No float tolerance is applied to before/after equality. Original §2 template/path/top-K verification remains valid.

**Production calibration rerun.** The unchanged production fitter is invoked again with the same original bank, B-val tables, manifest, reset-attestation telemetry, c1, targets and original stall artifacts described in §3. Fresh outputs are under `/tmp/codex_stall_fast/controller_followup/calibration_verified/<cell>/`. The output allowlist alone is redirected in memory; existing-output refusal remains in effect. Full bytes of both JSONs equal production, including all λ/d solutions, floors, ceilings, feasibility and stall fingerprints; even timestamps and paths require no exclusions.

| Cell | Solutions | calibration.json byte-equal | cost_replay.json byte-equal | Stall fingerprint |
| --- | --- | --- | --- | --- |
| groot_l10_50 | 20 | yes | yes | unchanged |
| groot_l10_500 | 20 | yes | yes | unchanged |
| groot_sp_50 | 20 | yes | yes | unchanged |
| groot_sp_500 | 20 | yes | yes | unchanged |
| pi05_l10_50 | 20 | yes | yes | unchanged |
| pi05_l10_500 | 20 | yes | yes | unchanged |
| pi05_sp_50 | 20 | yes | yes | unchanged |
| pi05_sp_500 | 20 | yes | yes | unchanged |

**Timing.** Both passes run in one CPU-only process, one BLAS/OpenMP thread, affinity 22–25,66–69, with no concurrent analysis Python job. They preload the same raw B-val NPZ vectors outside timers and follow the same C cadence, 80 episodes / 1,867 observations per pass, including warmup and first-use caches. Timers are `process_time_ns` and `perf_counter_ns`; medians/p99 use pandas as in §5. The before-consumer passes raw q, the after-consumer passes A code; both use the same exact fast tracker.

Observe-only timing (raw projection included before; code validation/copy included after):

| Cell | Anchors / pass | CPU median ms raw → code | CPU p99 ms raw → code | Wall median ms raw → code | Wall p99 ms raw → code |
| --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 380 | 1.285 → 0.423 | 1.912 → 0.813 | 1.281 → 0.420 | 1.909 → 0.810 |
| groot_l10_500 | 295 | 1.809 → 0.861 | 4.657 → 3.703 | 1.804 → 0.858 | 4.653 → 3.700 |
| groot_sp_50 | 137 | 1.104 → 0.383 | 1.648 → 0.763 | 1.101 → 0.380 | 1.645 → 0.760 |
| groot_sp_500 | 123 | 1.437 → 0.597 | 2.817 → 1.988 | 1.434 → 0.594 | 2.814 → 1.985 |
| pi05_l10_50 | 357 | 1.282 → 0.423 | 1.910 → 0.827 | 1.279 → 0.419 | 1.907 → 0.824 |
| pi05_l10_500 | 279 | 1.811 → 0.848 | 3.838 → 2.675 | 1.807 → 0.845 | 3.835 → 2.672 |
| pi05_sp_50 | 159 | 1.202 → 0.389 | 1.673 → 0.741 | 1.199 → 0.385 | 1.668 → 0.738 |
| pi05_sp_500 | 137 | 1.526 → 0.610 | 2.919 → 1.930 | 1.523 → 0.606 | 2.915 → 1.927 |

End-to-end incremental controller cost over the original A work:

| Cell | Anchors / pass | CPU median ms raw → code | CPU p99 ms raw → code | Wall median ms raw → code | Wall p99 ms raw → code |
| --- | --- | --- | --- | --- | --- |
| groot_l10_50 | 380 | 1.377 → 0.560 | 2.002 → 0.951 | 1.377 → 0.561 | 2.002 → 0.954 |
| groot_l10_500 | 295 | 1.907 → 1.009 | 4.801 → 3.840 | 1.908 → 1.010 | 4.801 → 3.841 |
| groot_sp_50 | 137 | 1.181 → 0.516 | 1.757 → 0.901 | 1.181 → 0.517 | 1.756 → 0.901 |
| groot_sp_500 | 123 | 1.524 → 0.735 | 2.922 → 2.131 | 1.525 → 0.736 | 2.923 → 2.129 |
| pi05_l10_50 | 357 | 1.371 → 0.555 | 1.995 → 0.955 | 1.374 → 0.556 | 1.996 → 0.956 |
| pi05_l10_500 | 279 | 1.914 → 0.986 | 3.952 → 2.824 | 1.914 → 0.986 | 3.952 → 2.829 |
| pi05_sp_50 | 159 | 1.299 → 0.524 | 1.772 → 0.895 | 1.299 → 0.527 | 1.772 → 0.902 |
| pi05_sp_500 | 137 | 1.623 → 0.745 | 3.030 → 2.061 | 1.624 → 0.746 | 3.030 → 2.062 |

The latter is measured as full C query minus its nested base query, adding back the code-capture facade preparation and operand interception that run inside the base query. Thus it includes the entire code handoff, tracker/status, Ehat/coin/cooldown/LOOK and result/log bookkeeping, while subtracting only original A projection/retrieval/synthesis. Nested timers add overhead, so this is a conservative instrumented host estimate. Per-cell full C query (including A) and capture-overhead statistics are retained in the CSV and evidence JSON. The all-observation diagnostic timing pass is retained separately under scratch `diagnostic_timing/`; headline numbers use the C-cadence pass above.

Afterward, observe-only CPU medians are 0.383–0.423 ms (50) and 0.597–0.861 ms (500). The original ≤0.1 / ≤0.3 ms targets remain missed. No precision, summation order, quantile or tie rule was relaxed. The remaining float64 norm/DP/reference work is described in §5. A's float32 retrieval distances are still not reused (§6). 

**Tests and cleanup.** All 22 unit tests pass: 13 existing stall tests, seven exact tracker tests and two consumer-capture tests covering early/main, absent Z0, original operand dtype/bytes, task identity and hook restoration on failure. `test_stall.py` now uses `tempfile.mkdtemp(prefix="_test_model_")` with no hard-coded directory, and `self.addCleanup(shutil.rmtree, path)` removes exactly its own directory. The final combined test run proves the default `/tmp/_test_model_*` names are identical before and after: `_test_model_1gm0whs2` and `_test_model_o17uckyv` remain untouched. The pre-existing source `_test_model_5beh_ykn/` remains untouched.

The existing method_c assertions run unchanged against new replay products: 16 cases on π0.5/GR00T L10-50, 6,208 decisions, 3,126 fresh anchors; 2,738 C anchors match the budget DAG. Served NPZ arrays (actions/source/vision/HIT) are byte-equal to the retained v2b products. `check_results` passes eight-cell 1,000-seed rate checks, 192 exhaustive budget values, loader gates and B-val exclusions. Its pre-existing independent DAG-versus-enumeration check reports max difference 3.197442310920451e-14; this is not a before/after comparison or a relaxed equivalence criterion. `check_packaging` passes 20 prefits, 32 validation specs and 44 C spec rows. Output paths alone are redirected to authorized scratch. For the existing replay RecordingTracker telemetry probe, a test-only dict exposes raw query attributes while retaining metric-code keys; production observe still takes the encoded path. Each replay uses a fresh interpreter because plugin installation is a singleton; the retained initial multi-replay attempt hit that guard after the first case.

Commands (from the repository root, with the required prefix on every Python command):

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m unittest exp.offline_search.rounds.r06.ideation_Q3.stall.test_stall exp.offline_search.rounds.r06.ideation_Q3.stall.test_stall_fast exp.offline_search.rounds.r06.ideation_Q3.stall.test_controller_fast -v
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_controller_fast parity
for cell in pi05_l10_50 groot_l10_50; do
  for case in A floor uniform R R_reverse R45 stall stall30; do
    taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_controller_fast replays --cell "$cell" --case "$case"
  done
done
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_controller_fast checks
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_controller_fast calibration
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.verify_controller_fast timing
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.build_fast_handback
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q3.stall.build_controller_handback
```

The final unit suite was run through unittest's loader/runner to also snapshot temporary directory names; `unit_final.log` and `test_cleanup.json` retain results. All successful stages exit 0. Existing replay/calibration outputs are never overwritten; reruns need fresh scratch destinations. Counts apply to fixed recorded observations and histories, not new closed-loop trajectories. C validation logs still lack replayable visual inputs (§4); no additional validation-log agreement claim is made.

Follow-up files changed: `ideation_Q1/method_c/methods.py`, `test_stall.py`, `build_fast_handback.py`, `HANDBACK_FAST.md`. New files: `test_controller_fast.py`, `verify_controller_fast.py`, `build_controller_handback.py`, `timing_controller_fast.csv`, `timing_controller_fast_samples.csv` (3,734 rows), `verification_controller_fast.json`. The two builders preserve §8 when regenerating the initial report. Other follow-up outputs and full source hashes are in `/tmp/codex_stall_fast/controller_followup/` and the evidence JSON.
