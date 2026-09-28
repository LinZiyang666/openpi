# K7 handback: vision-confirmed stuck guard

Completed verification: 2026-09-27T22:07:28.072143+00:00.

**Implementation, evidence, arm validation and final checks passed. Deployment placement is blocked:** the requested `/home/weiland/trace_runs/os_closed_loop/r04_k7/fits/` is mounted read-only in this session. The plugin completed the first fit but failed opening its `.pkl.tmp` with `OSError: [Errno 30] Read-only file system` (`results/prefit.log`). All five deployment artifacts were then fitted successfully with the installed plugin into `/tmp/k7_guard_fits/`, plus a spatial-50 verification fit. No artifact was installed at the requested destination. The coordinator must copy the verified files before launch.

Only the new K7 directory and K7-owned temporary artifacts were written. Shared K1, K6, R3, plugin, src, harness and profile files were not edited. No GPU, server, port, simulator/LIBERO worker, chain, remote host, git operation or review-test access was used. All Python processes used CPUs 26–29,70–73 with BLAS threads 1 and CUDA hidden; at most eight Python processes overlapped. The concurrency worker uses eight connections/threads.

## Rule and implementation

`judge.py:VisionConfirmedBlindMixedJudge` subclasses K1 `BlindMixedJudge`; `judge.py:BlindMixedJudge` is an alias. Enable `stuck_guard="vision_confirmed"` (the default). `stuck_guard="dense"` explicitly delegates to K1. `progress_guard="noprog_span"`, `memo_reset_after_miss=False`, `events="none"` are the deployment settings. GR00T is refused with `SkipCell` before fitting because stock and K1 have different terminal-closed gripper signs. A memo-reset option is refused with vision confirmation because it would violate stock no-progress parity.

For each pair of consecutive real vision anchors a < b, compute the minimum of the two task-centred camera cosines, using stock `core.centred_cos`, `M0`, `M1`, and the stock library 95th-percentile `c_thr`. Every dense transition j in (a,b] is confirmed only if that anchor cosine is ≥ c_thr AND its own float32 valid-state L2 motion `norm(rs[j,:8]-rs[j-1,:8])` is strictly below the stock pooled library 10th-percentile `m_thr`. `stuck_n` is the length of the trailing consecutive confirmed run at the current vision decision. A high-motion transition breaks the run; later low-motion transitions within the same confirmed anchor interval can form a new suffix. Prefix transitions without a left vision anchor are never counted. No keys are read from blind rows, copied forward or synthesized.

This is causal: confirmation arrives only at the closing anchor. No retrospective verdict is applied to already served blind HITs, and no right anchor is anticipated. Endpoint agreement is evidence of stillness, not proof that nothing moved out and back inside the gap. Requiring endpoint agreement preserves the stock visual veto and the raw-motion threshold while allowing dense state evidence to span a short gap.

The same corrected count, clipped at 5, feeds V7’s `stuck` feature. Its stock LOEO library calibration and fitted thresholds are unchanged and were compared bit for bit against K1’s stock-calibrated frozen fits. Overtime intentionally consumes the corrected count too, since stock guard 3 requires `stuck_n >= 1`. All other guard/event/burst code on gaps is copied unchanged from K1; the motion diagnostic uses raw L2 and `dense_motion_guard=0` identifies the change. K1’s `blind_step`, phase advancement, budget, gripper, terminal, dense-motion look gate, residual gate, lifecycle gate and span no-progress logic are inherited.

When all history decisions have vision, the implementation calls stock `MixedJudge.query` directly and synchronizes K1’s progress span afterward. All stock extras remain exact, including `noprog_n` (the all-vision branch does not add `noprog_span` to the result). Internally the span equals stock `noprog_n`, so subsequent blind look decisions retain K1 behavior. Step zero starts at count 0. After MISS, blind stepping requires a new real anchor; the stuck history itself is not reset, matching stock. Episode/task resets clear counters, progress and anchors; implicit query identity changes also reset.

## Final parity and offline evidence

Final rerun: **15,272 full scalar result comparisons, zero mismatches**, 20 evenly spaced episodes per cell/scale, in guard-only and events+burst configurations. The reference is the stock MixedJudge class with the stock AWM class and frozen stock calibration, not K7 in another mode. Compared complete top-k, scores, actions, library, confidence, every extras key/value and `_s` state. This includes all `os_*` verdict inputs, flags/reasons/phases, `stuck_n`, `motion`, `vself`, `pred_err`, `zsum` and raw/final V7 confidence. Fresh, stale and step-zero regimes are all covered. The independent count computation also matched stock on all 189,904 decision/library pairs.

| Cell | Library | Guard-only comparisons | Events+burst comparisons | Mismatches |
|---|---|---|---|---|
| pi05_l10_inf | 50 | 1081 | 1081 | 0 |
| pi05_l10_cache | 50 | 1706 | 1706 | 0 |
| pi05_l10_inf | 500 | 1081 | 1081 | 0 |
| pi05_l10_cache | 500 | 1706 | 1706 | 0 |
| pi05_spatial_inf | 50 | 453 | 453 | 0 |
| pi05_spatial_cache | 50 | 578 | 578 | 0 |
| pi05_spatial_inf | 500 | 453 | 453 | 0 |
| pi05_spatial_cache | 500 | 578 | 578 | 0 |

Source: `evidence.py`, `results/parity_*.json`, `results/final/parity_*.log`. Final command lists are `results/final/parity_commands.json`. All calibration arrays, task means and thresholds are identical at each deployed scale; no 50-library result borrows 500-library information.

Every all-vision row below uses all 500 recorded episodes of its query cell. These are **stuck flag counts / all decisions**, not total MISS rates or SR. Stock predicates and K1 dense predicates are evaluated on the exact same online QueryViews; K7’s independent history function is asserted equal to the stock scalar recurrence at every decision. Full retrieval/guard/V7 parity is the separate scalar check above.

| Cell | Library | N | Stock count (rate) | K1 dense count (rate) | K7 count (rate) |
|---|---|---|---|---|---|
| pi05_l10_inf | 50 | 29406 | 2137 (7.2672%) | 3715 (12.6335%) | 2137 (7.2672%) |
| pi05_l10_cache | 50 | 40127 | 7154 (17.8284%) | 8146 (20.3005%) | 7154 (17.8284%) |
| pi05_l10_inf | 500 | 29406 | 1060 (3.6047%) | 2178 (7.4067%) | 1060 (3.6047%) |
| pi05_l10_cache | 500 | 40127 | 4859 (12.1091%) | 6867 (17.1132%) | 4859 (12.1091%) |
| pi05_spatial_inf | 50 | 10798 | 76 (0.7038%) | 288 (2.6672%) | 76 (0.7038%) |
| pi05_spatial_cache | 50 | 14621 | 1521 (10.4028%) | 1955 (13.3712%) | 1521 (10.4028%) |
| pi05_spatial_inf | 500 | 10798 | 57 (0.5279%) | 235 (2.1763%) | 57 (0.5279%) |
| pi05_spatial_cache | 500 | 14621 | 1485 (10.1566%) | 1801 (12.3179%) | 1485 (10.1566%) |


For B=2, the frozen scheduler uses K1’s real blind-step/gate machinery and span progress, with ideation A’s saved 16-member anchors and K1 replay helpers. Robot observations, executed chunks and HIT/MISS history stay recorded; hypothetical judge verdicts do not alter the stream. Thus all comparators see the same vision mask. Stock is an **omnivision reference projected onto those vision decisions**, not a deployable stock judge with missing keys. K7 receives NaN key rows plus the vision mask for blind history; the K1 dense predicate reads only robot states. Inf cells have no blind decisions because every recorded previous action is a MISS; their B=2 rates equal B=0.

| Cache cell | Library | Vision / all | Stock fires | K1 dense fires | K7 fires | Stock / dense / K7 rates per vision |
|---|---|---|---|---|---|---|
| pi05_l10_cache | 50 | 28462/40127 | 7134 | 8146 | 7052 | 25.0650% / 28.6206% / 24.7769% |
| pi05_l10_cache | 500 | 28254/40127 | 4854 | 6867 | 4812 | 17.1799% / 24.3045% / 17.0312% |
| pi05_spatial_cache | 50 | 9849/14621 | 1504 | 1955 | 1486 | 15.2706% / 19.8497% / 15.0878% |
| pi05_spatial_cache | 500 | 9796/14621 | 1462 | 1801 | 1434 | 14.9245% / 18.3851% / 14.6386% |


Source: `results/rates_*.json` and `results/counts_*.npz` (full count arrays and B=2 mask). The B=2 schedule inherits ideation A’s documented batched-versus-scalar anchor limitation (maximum action difference 0.000298366 in its validation); no B=0 bit-parity claim uses those batched anchors. No counterfactual state evolution, live SR or IR improvement is inferred from these measurements.

## Historical closed-loop audit

Read all accepted attempts, including journal status `failed`, from the three existing 500-episode arms. The supplied reason counts are reproduced exactly. Stock R3 has no robot-state field here, but its logged raw `motion` and `vself` suffice for its stuck recurrence. K1 B=0 has full robot state and adjacent `vself`, allowing exact float32 raw-motion reconstruction. K1’s original dense count was also reconstructed with zero mismatches on both R4 paths. All three have no keys and `log_inputs=false`.

| Arm | N | Vision | Observed MISS | Observed stuck | Stock predicate on same path |
|---|---|---|---|---|---|
| r3mx_p_l10_g500 | 29340 | 29340 | 2983 | 699 | 699 |
| r4b3_p_l10_500_b0g | 29615 | 29615 | 4144 | 1952 | 670 |
| r4b3_p_l10_500_ph2g | 29902 | 17535 | 4304 | 1774 | Unavailable across gaps |


On K1 B=0, holding every observation, proposal and other logged flag fixed, corrected reason counts are **stuck 670, terminal 259, overtime 119, no-progress 2,088**, for **3,136 / 29,615 (10.5892%)** projected forced MISSes, versus 4,144 (13.9929%) observed. Terminal/no-progress reason totals may rise when the higher-priority stuck bit disappears; their underlying flag bits were preserved. This is not an executed new arm. R3’s 699 stuck counts reproduce with zero count mismatches over all 29,340 decisions. For B=2, `vself` is absent after blind gaps, and no anchor-pair cosine/keys are logged: **K7 gap guards and bitwise V7 confidence cannot be reconstructed from those logs**. Full V7 bit parity is established only with the offline keys above.

Source: `historical.py`, `results/historical.json`; it records all exact input file paths and sizes at read.

## Plugin and edge verification

Final rerun of installed plugin blind selftest: **5/5 PASS**, 240 decisions, 80 blind, 10 MISS, 160 stage-1 calls, 240 broadcasts. Each uses two interleaved connections, four episodes, duplicate rejection and offline exact log replay. The command’s `--blind --judge guard_only` is translated by the existing selftest into plugin `--os-blind --os-judge guard_only`.

| Arm | Decisions | Vision | Blind | MISS |
|---|---|---|---|---|
| r4k7_p_l10_500_b0g | 48 | 48 | 0 | 2 |
| r4k7_p_l10_500_ph2g | 48 | 26 | 22 | 2 |
| r4k7_p_l10_500_ph1g | 48 | 32 | 16 | 4 |
| r4k7_p_l10_50_ph2g | 48 | 28 | 20 | 2 |
| r4k7_p_sp_500_ph2g | 48 | 26 | 22 | 0 |


Eight-connection concurrent execution matched a fresh serialized process of the **same installed plugin** in observed reservation order. Comparisons include every non-timing decision field, wire action bytes, verdict, dense state/key/action/HIT/vision history, guard state, progress span and the complete blind anchor (rows, weights, phase, action, state and lifecycle). Native shadow search and the existing offline log verifier run too. Each configuration has 16 episodes and 183 decisions; all reached eight overlapping fake stage-1 calls. Delays are 60 ms for stage 1 and 25 ms on MISS. This is CPU fake-policy correctness, not GPU batching or live latency validation.

| Arm | Connections | Decisions | Vision | Blind | MISS | Exact match |
|---|---|---|---|---|---|---|
| r4k7_p_l10_500_b0g | 8 | 183 | 183 | 0 | 6 | True |
| r4k7_p_l10_500_ph2g | 8 | 183 | 91 | 92 | 2 | True |
| r4k7_p_l10_500_ph1g | 8 | 183 | 117 | 66 | 7 | True |
| r4k7_p_l10_50_ph2g | 8 | 183 | 98 | 85 | 6 | True |
| r4k7_p_sp_500_ph2g | 8 | 183 | 110 | 73 | 6 | True |


Evidence: `results/final/plugin_*/{selftest_report.json,verify_blind.json}`; `/tmp/k7_guard_concurrency_final/<arm>/{threaded,serialized}/{concurrency.json,verify.json}`; compact `results/concurrency_summary.json`. `concurrency_test.py` is an owned adaptation of K6’s driver. It adds guard/anchor state snapshots, uses the exact ncal=3000 arm prefits, and treats timing as informational under shared CPU load.

**8/8 unchanged harness smokes PASS**, 728 decisions, both suites/regimes/scales, fresh default-ncal fits and reversed-episode determinism. **13 focused edge checks**, 150 real gap queries, and four lifecycle configurations passed. These cover step zero, missing left anchor, blind-key sentinel independence, camera conjunction, strict/inclusive thresholds, visual/motion run breaks, MISS-to-new-anchor, task reset, unchanged flags other than stuck/overtime, unchanged proposals, repeated gap queries, V7’s corrected count, and explicit GR00T refusal. An initial edge fixture expected a blind HIT while K1’s no-progress gate was active; the lifecycle-only fixture was corrected to disable guards for that isolated check. The judge required no correction. Evidence: `results/edges.json`, `results/final/smoke_*.log`.

## Arms, fits and footprint

`arms_k7.json` is emit_arms format. All five rows specify full model, `cost_ledger: true`, `--os-blind`, `--os-judge guard_only`, `--os-no-shadow-native`, the explicit cold-store `--os-root`, and `<RUN>/fits/<arm>.pkl`. Base kwargs are copied from their exact K1 counterparts: phase_particles, gates all; current/kref 5 at 50, big/kref 8 at 500. B=0 is the adapter control, expected to match stock g500 on identical inputs; its rollout SR is unverified. Five emitted CacheConfigs and artifact metadata all validated through the unchanged emitter/parser (`results/arms_validation.json`).

| Arm | Suite | Library | B |
|---|---|---|---|
| r4k7_p_l10_500_b0g | l10 | big | 0 |
| r4k7_p_l10_500_ph2g | l10 | big | 2 |
| r4k7_p_l10_500_ph1g | l10 | big | 1 |
| r4k7_p_l10_50_ph2g | l10 | current | 2 |
| r4k7_p_sp_500_ph2g | spatial | big | 2 |


All listed artifacts were produced by the installed plugin `prefit_main` on CPU, protocol 4, default ncal=3000, loaded successfully in verification, and hashed after the final checks. Each artifact path currently starts with `/tmp/k7_guard_fits/`.

| Artifact basename | Bytes | SHA256 |
|---|---|---|
| r4k7_p_l10_500_b0g.pkl | 142348550 | `fa7ce005f00cfbc40f2bd47b6c710989c35b36cee6d6de70cb7ed67eb23a9d27` |
| r4k7_p_l10_500_ph2g.pkl | 142348550 | `aa633f2d5b54bfa051e4d1742c9fcd4f496aa6a704587776111c7e95c135022b` |
| r4k7_p_l10_500_ph1g.pkl | 142348550 | `37c934f9b6e4227a521aaa8fdbce7fc60f6ae73099154ddc04d1c55434c3f296` |
| r4k7_p_l10_50_ph2g.pkl | 32704515 | `302053ef5837f6f93dde89d36073a42bfd7d64e350463ab16df6ef088fbd601d` |
| r4k7_p_sp_500_ph2g.pkl | 66500540 | `52833d9daa18bd653731301a5cb0632ef210506e70e6e94ede3b5303e7c5e365` |
| verification_pi05_spatial_50.pkl | 26076510 | `ea57f32e003f0ef7b08a4dccf5cf8aead47454d12b2f34cda079db52e277cf57` |


The extra `verification_pi05_spatial_50.pkl` is evidence-only, not a sixth arm. Fits differ in their method/budget metadata even when byte sizes coincide. `results/fits.json` records every actual command and requested destination.

| Artifact | Library eps / rows | Representation bytes | Fit pickle MB | Fit s | Stock m_thr / c_thr |
|---|---|---|---|---|---|
| r4k7_p_l10_500_b0g | 500 / 29472 | 18626304 | 142.348550 | 43.670 | 0.0385605428368 / 0.98449782881 |
| r4k7_p_l10_500_ph2g | 500 / 29472 | 18626304 | 142.348550 | 43.670 | 0.0385605428368 / 0.98449782881 |
| r4k7_p_l10_500_ph1g | 500 / 29472 | 18626304 | 142.348550 | 41.347 | 0.0385605428368 / 0.98449782881 |
| r4k7_p_l10_50_ph2g | 50 / 2640 | 1668480 | 32.704515 | 9.541 | 0.0518229078501 / 0.957482802689 |
| r4k7_p_sp_500_ph2g | 500 / 10909 | 6894488 | 66.500540 | 19.338 | 0.0938192620873 / 0.905616642749 |
| verification_pi05_spatial_50 | 49 / 1018 | 643376 | 26.076510 | 3.475 | 0.0931858614087 / 0.889220086159 |


Representation is **632 bytes/row**, excluding shared actions and fixed calibration/projection arrays; valid π0.5 full-chunk action payload is another 280 bytes/row. The actual pickle sizes above include padded actions and auxiliary/fixed arrays. The supplied deployed-pickle comparison is 431 MB for π0.5 spatial and 1,103 MB for π0.5 l10. Spatial “50” actually contains 49 library episodes. `results/footprint.json` also records top-level stored NPY bytes (including full-resolution keys, excluding token subdirectories) to distinguish the cold store from the compact deployment fit. No borrowed big-library information or external models were used.

## Exact commands and coordinator next steps

All commands run from `/home/weiland/projects/openpi`. The exact per-arm prefit commands for the **requested final destination** are in `prefit.sh` and reproduced here. The first failed at the read-only destination; the remaining final-destination invocations were not run there. The successful commands differ only in artifact prefix, `/tmp/k7_guard_fits/`; `prefit_staged.sh` and `results/prefit_staged_commands.json` contain all six executed commands.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":0,"gates":"all"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k7_guard/results/prefit/r4k7_p_l10_500_b0g --os-tag r4k7_p_l10_500_b0g --os-no-shadow-native --os-blind --os-judge guard_only --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_b0g.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":2,"gates":"all"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k7_guard/results/prefit/r4k7_p_l10_500_ph2g --os-tag r4k7_p_l10_500_ph2g --os-no-shadow-native --os-blind --os-judge guard_only --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_ph2g.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":1,"gates":"all"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k7_guard/results/prefit/r4k7_p_l10_500_ph1g --os-tag r4k7_p_l10_500_ph1g --os-no-shadow-native --os-blind --os-judge guard_only --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_ph1g.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"phase_particles","budget":2,"gates":"all"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k7_guard/results/prefit/r4k7_p_l10_50_ph2g --os-tag r4k7_p_l10_50_ph2g --os-no-shadow-native --os-blind --os-judge guard_only --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_50_ph2g.pkl
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":2,"gates":"all"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r04/k7_guard/results/prefit/r4k7_p_sp_500_ph2g --os-tag r4k7_p_sp_500_ph2g --os-no-shadow-native --os-blind --os-judge guard_only --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_sp_500_ph2g.pkl
```

The checks actually run (initial and final labels use separate plugin log directories):

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
K=exp.offline_search.rounds.r04.k7_guard
"${PY[@]}" -m "$K.prepare"
bash exp/offline_search/rounds/r04/k7_guard/prefit.sh  # failed: read-only destination
"${PY[@]}" -m "$K.run_prefits"
"${PY[@]}" -m "$K.run_checks" parity --tag initial
"${PY[@]}" -m "$K.run_checks" rates --tag full
"${PY[@]}" -m "$K.edge_checks"
"${PY[@]}" -m "$K.historical"
"${PY[@]}" -m "$K.validate_arms"
"${PY[@]}" -m "$K.run_checks" plugin --tag initial
"${PY[@]}" -m "$K.run_checks" smoke --tag final
"${PY[@]}" -m "$K.run_checks" parity --tag final
"${PY[@]}" -m "$K.run_checks" plugin --tag final
"${PY[@]}" -m "$K.concurrency_test" --source installed --config all --out /tmp/k7_guard_concurrency_final
"${PY[@]}" -m "$K.edge_checks"  # final edge rerun
"${PY[@]}" -m "$K.write_handback"
```

Coordinator: copy the five staged arm pickles to the requested directory, check the hashes above, resolve `<RUN>` to `/home/weiland/trace_runs/os_closed_loop/r04_k7`, then use normal emit_arms and the normal next server start. No shared plugin change or refit is required. Run the B=0 control before interpreting B=1/B=2, then measure paired 500-init SR, vision share, MISS share and the cost ledger on the requested arms. Existing servers do not import the new class until their next startup.

```bash
cp --no-clobber /tmp/k7_guard_fits/r4k7_p_l10_500_b0g.pkl /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_b0g.pkl
cp --no-clobber /tmp/k7_guard_fits/r4k7_p_l10_500_ph2g.pkl /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_ph2g.pkl
cp --no-clobber /tmp/k7_guard_fits/r4k7_p_l10_500_ph1g.pkl /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_500_ph1g.pkl
cp --no-clobber /tmp/k7_guard_fits/r4k7_p_l10_50_ph2g.pkl /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_l10_50_ph2g.pkl
cp --no-clobber /tmp/k7_guard_fits/r4k7_p_sp_500_ph2g.pkl /home/weiland/trace_runs/os_closed_loop/r04_k7/fits/r4k7_p_sp_500_ph2g.pkl
```

Copying/launching was **not performed** here. No claim is made that the projected MISS reduction preserves success, that endpoint agreement observes hidden visual motion, or that CPU fake-policy concurrency predicts GPU/SR performance.

## Deliverables and SHA256

Modification times and exact source/script sizes are in `results/deliverable_hashes.json`. Read-only dependency hashes are in `results/dependency_hashes.json`. The handback itself is hashed separately in `HANDBACK.sha256` after generation.

| Owned file | SHA256 |
|---|---|
| README.md | `db7e7a275c787b3d93cd5aee954373ea45b56b0335da42dc64074d6b1c7a3808` |
| __init__.py | `b67dbe458069ab4ee60e9426bf4847eff579d6da73451df41ef5b7c863fad037` |
| arms_k7.json | `fcaa3852d5c63e9f93fffacf24ca8fe813112248bdb9065b5dda70bd60863908` |
| concurrency_test.py | `9423df8782829f183f8a7628cf6f15a66a21506309db3c13ccdc80666fb70fc3` |
| edge_checks.py | `b9480e6f1c95c98d5ae6b20599e577baff72a0d582ece62ed8b71a4473b0d55f` |
| evidence.py | `001f5b6d557223f3873627a5b959c684ae99c2cb92071a82542e5f0d03a724a7` |
| historical.py | `acfcd8f8801670181bb61c0a50085bcbc5374a4ffa30f74ea52d28b21fbaff36` |
| judge.py | `b812faaded941e4c34a3a3c96323ebecdeda4c3a6681fb5d1e58bb7ef07183a4` |
| prefit.sh | `96fee7bf76b30555a42e19a4fde38a8093d677f86c5541034e3c2a83d4b9eb70` |
| prefit_staged.sh | `53dc9485c0cbbc0a87e1a99af3332ada1405bafe492414d92e5a2ef305b4a839` |
| prepare.py | `d7c747f7945c5681ee0a406c4c4a0801688e8c2093b83b1e08af99aeb7803557` |
| run_checks.py | `c8fa96d2576acff03c85d9fe05ac79fb8bdfc4fb2598ebb019763c480e4663fa` |
| run_prefits.py | `f3e86aaa7c2a518d57b4c5aa7d5eda3a6024108d34cbd8ba8102c3754db181b2` |
| validate_arms.py | `99e0bbe5bc97ac40ee140640b4f483d646ffca45c10bfda9492cfe573b12c242` |
| write_handback.py | `77098cd9f05437c270ee02ff4cdca20d99bf63158b716be048fdcec037b1d0c3` |

