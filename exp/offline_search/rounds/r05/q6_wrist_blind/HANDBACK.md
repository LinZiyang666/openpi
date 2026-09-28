# Q6 wrist-only blind guard hand-back

Final audit: **PASS**, 2026-09-28T02:55:11.356025+00:00.


Only `exp/offline_search/rounds/r05/q6_wrist_blind/` and `/tmp/q6_*` were written. No shared plugin, stage override,
K1/K3/K7, harness or src file was edited. No GPU, server, port, LIBERO worker, chain, remote host, git command,
or review-test read was used. Python commands used CPUs **10–13,54–57**, BLAS/OMP threads 1 and CUDA hidden;
driver subprocesses are explicitly affinity-prefixed, with at most eight Python processes.

## Rule and justification

`judge.py:WristVisionConfirmedBlindJudge` subclasses K7 `VisionConfirmedBlindMixedJudge`, fixes its base to
K1 `BlindWristAWM` (K1 blind machinery + K3 `WristAWM`), and supplies K3 `WristView` during threshold fitting,
V7 calibration, queries and `confirmed_stuck`. Both legacy visual slots reference the wrist key, so
`min(c_wrist,c_wrist) = c_wrist`. The guard consumes no base-camera placeholder or blind-row visual key. The retrieval
representation contains wrist once: 64 wrist PCs + 8 state dimensions. This is a method composition, with no
plugin change required.

For consecutive real vision anchors a < b, compute `core.centred_cos(w[b], w[a], M_wrist[task])`. Confirm
transitions in `(a,b]` only when this is **>= the deployed library wrist p95**. Each dense transition must also
have **float32 valid-state L2 motion < the pooled deployed-library p10**. Count the trailing run exactly as K7:
high motion or failed visual confirmation breaks it; a prefix without a left visual anchor is unconfirmed.
The stuck bit fires at count >= 2; the corrected count also feeds overtime and V7 (clipped at 5).
The complete motion/gap implementation is inherited from K7, with no normalized K1 dense-motion substitution.
At all-vision/B=0 the stock MixedJudge path is retained and agrees bitwise with K3's wrist judge on the tested
streams, including all extras and confidence. Q6 does not claim two-camera stock retrieval or verdict parity.

The available wrist view supplies a real visual veto while fitting its own percentile keeps the same nominal
upper-tail calibration convention. Reusing the two-camera-min threshold would calibrate a different statistic.
Percentile calibration does not make guard firing probabilities equal: motion conjunction, serial runs, and
task distribution matter. Wrist-only confirmation can miss changes visible exclusively to the base camera;
endpoint agreement also cannot exclude out-and-back motion during a blind gap. These are limitations, not
offline evidence of an SR improvement.

Defaults: `base_kwargs={"serving":"anchor_tail","budget":1,"gates":"budget_only"}`, `events="none"`,
`progress_guard="noprog_span"`, `stuck_guard="vision_confirmed"`, `memo_reset_after_miss=False`.
`budget_only` is K1's base gate setting: K1's judge-level no-progress look rule and lifecycle checks remain active.
First decision, after MISS, exhausted budget, invalid anchor, reset and task change require vision. The tail
uses exactly steps 5–9 of the accepted vision chunk. The separate phase variant explicitly uses
`serving="phase_particles", budget=2, gates="all"`; it is not the default or a measured winner.
GR00T, dense-stuck substitution, legacy `noprog_n`, altered percentiles, and memo reset are refused.

## Calibration and library composition

Each fit uses its own deployed library (`current` at 50, `bpool_cs` at 500), including task means, wrist p95,
motion p10, AWM metric and V7 LOEO calibration. No 50-scale fit borrows 500-scale rows. The deployed 500 pools
are **not entirely successful**: l10 has 436 successful / 64 failed episodes; spatial has 487 / 13.
Calibration uses all deployed rows, as requested. Success labels are evaluator-only for the separate
successful-demo rates below. Spatial “50” has 49 actual episodes.


| Suite | Scale | Episodes / rows | Successful eps / rows | Raw motion p10 | Two-camera min p95 | Wrist p95 |
| --- | --- | --- | --- | --- | --- | --- |
| l10 | 50 | 50 / 2640 | 50 / 2640 | 0.0518229078501 | 0.957482802689 | 0.965840291539 |
| l10 | 500 | 500 / 29472 | 436 / 22816 | 0.0385605428368 | 0.98449782881 | 0.985327864465 |
| spatial | 50 | 49 / 1018 | 49 / 1018 | 0.0931858614087 | 0.889220086159 | 0.902513620976 |
| spatial | 500 | 500 / 10909 | 487 / 10337 | 0.0938192620873 | 0.905616642749 | 0.921815248717 |



## Final stuck firing rates

These are **stuck-bit counts / vision decisions**, with percentages, including episode-start decisions in the
denominator. They are not overall guard/MISS rates or SR. All-vision library/stream counts use the exact online
float32 cosine and motion predicates; independent K7/Q6 history computations are asserted equal to scalar
recurrences at every row. Library percentile fitting retains stock/K3 arithmetic (float64 reduction after
float32 centering), while online comparison uses stock `centred_cos`.
The K1 dense comparator retains its own per-task state normalization (std floor .05), RMS motion and
task-specific p10. Stock/K7/Q6 use the pooled raw-L2 p10. Every comparator uses stuck count >= 2.

### Successful demos and all deployed library rows


successful_library

| Suite | Scale | Episodes | N | Stock 2cam | K1 dense | K7 2cam | Q6 wrist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| l10 | 50 | 50 | 2640 | 62 (2.3485%) | 138 (5.2273%) | 62 (2.3485%) | 59 (2.2348%) |
| l10 | 500 | 436 | 22816 | 199 (0.8722%) | 781 (3.4230%) | 199 (0.8722%) | 196 (0.8590%) |
| spatial | 50 | 49 | 1018 | 0 (0.0000%) | 16 (1.5717%) | 0 (0.0000%) | 1 (0.0982%) |
| spatial | 500 | 487 | 10337 | 16 (0.1548%) | 185 (1.7897%) | 16 (0.1548%) | 11 (0.1064%) |


deployed_library_all

| Suite | Scale | Episodes | N | Stock 2cam | K1 dense | K7 2cam | Q6 wrist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| l10 | 50 | 50 | 2640 | 62 (2.3485%) | 138 (5.2273%) | 62 (2.3485%) | 59 (2.2348%) |
| l10 | 500 | 500 | 29472 | 868 (2.9452%) | 1,896 (6.4332%) | 868 (2.9452%) | 859 (2.9146%) |
| spatial | 50 | 49 | 1018 | 0 (0.0000%) | 16 (1.5717%) | 0 (0.0000%) | 1 (0.0982%) |
| spatial | 500 | 500 | 10909 | 45 (0.4125%) | 238 (2.1817%) | 45 (0.4125%) | 33 (0.3025%) |


### Replayed decision streams, all vision

All 500 recorded episodes in each cell.


| Cell | Scale | N | Stock 2cam | K1 dense | K7 2cam | Q6 wrist |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_inf | 50 | 29406 | 2,137 (7.2672%) | 3,715 (12.6335%) | 2,137 (7.2672%) | 1,995 (6.7843%) |
| pi05_l10_cache | 50 | 40127 | 7,154 (17.8284%) | 8,146 (20.3005%) | 7,154 (17.8284%) | 6,952 (17.3250%) |
| pi05_l10_inf | 500 | 29406 | 1,060 (3.6047%) | 2,178 (7.4067%) | 1,060 (3.6047%) | 1,024 (3.4823%) |
| pi05_l10_cache | 500 | 40127 | 4,859 (12.1091%) | 6,867 (17.1132%) | 4,859 (12.1091%) | 4,809 (11.9844%) |
| pi05_spatial_inf | 50 | 10798 | 76 (0.7038%) | 288 (2.6672%) | 76 (0.7038%) | 71 (0.6575%) |
| pi05_spatial_cache | 50 | 14621 | 1,521 (10.4028%) | 1,955 (13.3712%) | 1,521 (10.4028%) | 1,526 (10.4370%) |
| pi05_spatial_inf | 500 | 10798 | 57 (0.5279%) | 235 (2.1763%) | 57 (0.5279%) | 45 (0.4167%) |
| pi05_spatial_cache | 500 | 14621 | 1,485 (10.1566%) | 1,801 (12.3179%) | 1,485 (10.1566%) | 1,467 (10.0335%) |



### Same-mask replay with blind gaps

The actual Q6 `blind_step` and scalar `query` select each common vision mask using recorded states, actions,
HIT/MISS history and its real wrist retrieval/progress. Hypothetical guard verdicts do not change that recorded
path. Stock is an **omnivision reference projected onto the common vision mask**; it is not deployable with
missing keys. K7 and Q6 see only real anchor keys, with NaN blind history rows. K1 sees dense states. This isolates
the guard predicates, not four alternative closed-loop trajectories. Inf cells contain only recorded MISSes,
so lifecycle prohibits blindness and both gap schedules have exactly their all-vision rates above.


tail

| Cell | Scale | Vision / all | Stock 2cam | K1 dense | K7 2cam | Q6 wrist |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_cache | 50 | 27272 / 40127 | 6,806 (24.9560%) | 7,436 (27.2661%) | 6,586 (24.1493%) | 6,416 (23.5260%) |
| pi05_l10_cache | 500 | 27421 / 40127 | 4,637 (16.9104%) | 6,337 (23.1100%) | 4,428 (16.1482%) | 4,389 (16.0060%) |
| pi05_spatial_cache | 50 | 9324 / 14621 | 1,464 (15.7014%) | 1,817 (19.4873%) | 1,395 (14.9614%) | 1,392 (14.9292%) |
| pi05_spatial_cache | 500 | 9265 / 14621 | 1,385 (14.9487%) | 1,604 (17.3125%) | 1,309 (14.1284%) | 1,306 (14.0961%) |


phase2

| Cell | Scale | Vision / all | Stock 2cam | K1 dense | K7 2cam | Q6 wrist |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_cache | 50 | 28388 / 40127 | 7,132 (25.1233%) | 8,146 (28.6952%) | 7,054 (24.8485%) | 6,856 (24.1510%) |
| pi05_l10_cache | 500 | 28473 / 40127 | 4,855 (17.0512%) | 6,867 (24.1176%) | 4,816 (16.9143%) | 4,768 (16.7457%) |
| pi05_spatial_cache | 50 | 9875 / 14621 | 1,509 (15.2810%) | 1,955 (19.7975%) | 1,486 (15.0481%) | 1,495 (15.1392%) |
| pi05_spatial_cache | 500 | 9845 / 14621 | 1,465 (14.8807%) | 1,801 (18.2936%) | 1,430 (14.5251%) | 1,421 (14.4337%) |



Raw evidence: `results/evidence_{l10,spatial}_{50,500}.json`, per-row count/vision-mask arrays
`results/counts_*.npz`, exact commands `results/final/evidence_commands.json`. Inputs are the cold store's
`library/pi05_{l10,spatial}/{current,bpool_cs}/` and `queries/pi05_{l10,spatial}_{inf,cache}/`.
Historical closed-loop logs without endpoint keys cannot establish the wrist predicate across gaps; no
historical wrist SR or observed closed-loop guard reduction is claimed here.

## Tests rerun against final files


**20 synthetic/configuration checks**, **3,714 full bitwise K3 wrist comparisons**, **248 real gap queries**; four same-library threshold recalibration checks, exact tail slices, missing-camera access traps, NaN/sentinel independence, strict/inclusive boundaries, task centering, lifecycle/reset, V7 corrected count, unchanged proposal and non-stuck/non-overtime flag checks all passed. `results/unit.json`.


Eight installed-plugin blind selftests passed: **384 decisions, 186 blind, 0 MISS**. Each includes two interleaved connections, four episodes, duplicate rejection, once-only broadcast/commit and exact logged-input replay.


| Arm | Decisions | Vision | Blind | MISS |
| --- | --- | --- | --- | --- |
| r5q6_p_l10_500_phase2 | 48 | 24 | 24 | 0 |
| r5q6_p_l10_500_tail | 48 | 24 | 24 | 0 |
| r5q6_p_l10_50_phase2 | 48 | 28 | 20 | 0 |
| r5q6_p_l10_50_tail | 48 | 24 | 24 | 0 |
| r5q6_p_spatial_500_phase2 | 48 | 26 | 22 | 0 |
| r5q6_p_spatial_500_tail | 48 | 24 | 24 | 0 |
| r5q6_p_spatial_50_phase2 | 48 | 24 | 24 | 0 |
| r5q6_p_spatial_50_tail | 48 | 24 | 24 | 0 |



`plugin_selftest.py` invokes the **unchanged installed** selftest in its own process. It passes
`--os-blind --os-stage1-mode wrist_only --os-tokens off --os-judge guard_only` through the production stage
parser/validator and startup hook; its fake key builder emits zero camera 0 plus exact stored wrist/state,
and token access raises. Startup asserts wrist_only and K10. This is CPU serving/history verification, not
a GPU vision-tower/MISS-completion test; K3's existing GPU evidence is a dependency, not rerun or credited here.


Eight connections per configuration, **1464 total decisions**, including **667 blind and 49 MISS**, all eight configurations matched a fresh serialized process in reservation order. All non-timing decision fields, wire action bytes, verdicts, dense key/state/action/HIT/vision histories, guard/progress state and full blind anchors matched exactly; each threaded run reached eight overlapping fake stage calls.


| Arm | Decisions | Vision | Blind | MISS | Peak concurrent |
| --- | --- | --- | --- | --- | --- |
| r5q6_p_l10_50_tail | 183 | 96 | 87 | 6 | 8 |
| r5q6_p_l10_500_tail | 183 | 96 | 87 | 7 | 8 |
| r5q6_p_spatial_50_tail | 183 | 97 | 86 | 8 | 8 |
| r5q6_p_spatial_500_tail | 183 | 96 | 87 | 7 | 8 |
| r5q6_p_l10_50_phase2 | 183 | 96 | 87 | 6 | 8 |
| r5q6_p_l10_500_phase2 | 183 | 91 | 92 | 2 | 8 |
| r5q6_p_spatial_50_phase2 | 183 | 113 | 70 | 6 | 8 |
| r5q6_p_spatial_500_phase2 | 183 | 112 | 71 | 7 | 8 |


Raw concurrency evidence: `/tmp/q6_concurrency_final/<arm>/{threaded,serialized}/`; compact `results/concurrency_summary.json`. Sleeping fake-policy timing is not a throughput claim.


Eight existing selftest recipes passed (the standard test source was not edited):
| Recipe | Decisions | PASS |
| --- | --- | --- |
| k1_full | 48 | True |
| k1_wrist | 48 | True |
| k3_wrist | 41 | True |
| k7_full | 48 | True |
| legacy_groot_hit | 38 | True |
| legacy_groot_periodic:1 | 38 | True |
| legacy_pi05_hit | 41 | True |
| legacy_pi05_periodic:1 | 41 | True |


Eight fresh-fit harness smokes passed, **728 decisions**, both suites/regimes/scales, including reversed-episode determinism, valid online inputs and finite/shape checks. Default ncal=3000 for Q6 fits, smokes, plugin and concurrency tests; only the inherited K1/K3/K7 regression recipes use ncal=64. Full smoke metrics/timing are in `results/final/smoke_*/`; runtime timing was measured under shared CPU load.


| Smoke | Decisions | Mean action error | AURC | ms/query |
| --- | --- | --- | --- | --- |
| smoke_l10_500_cache | 159 | 0.535537 | 0.342095 | 1.920 |
| smoke_l10_500_inf | 104 | 0.255083 | 0.208149 | 2.038 |
| smoke_l10_50_cache | 159 | 0.663406 | 0.427487 | 1.439 |
| smoke_l10_50_inf | 104 | 0.364435 | 0.264034 | 1.568 |
| smoke_spatial_500_cache | 61 | 0.661371 | 0.384501 | 1.495 |
| smoke_spatial_500_inf | 40 | 0.286640 | 0.220717 | 1.680 |
| smoke_spatial_50_cache | 61 | 0.739333 | 0.461565 | 1.471 |
| smoke_spatial_50_inf | 40 | 0.351085 | 0.291696 | 1.991 |



Initial evidence-driver checks exposed two evaluator issues: its direct wrist `_self_change` call initially
omitted `WristView`, and its initial assertion incorrectly assumed the 500 pools were all-success. The driver
was corrected; no judge change was needed. Final evidence reran all four cells/scales after those corrections.
Initial failed logs are retained. Every final command group returned zero.

## Arms, prefits and bytes

`arms_q6.json` has exactly **four primary arms**, π0.5 `{l10,spatial}` × `{50,500}`, all wrist + anchor_tail B1 +
wrist-confirmed guard. `arms_phase2.json` is a separate four-arm prespecified variant, not part of the requested
four-arm primary set. Both are emit_arms format, with `<RUN>` fit/evidence placeholders, `full_model:true`,
`miss.num_steps:10`, `write_policy.type:never`, and `cost_ledger:true`. All eight emitted CacheConfigs, production
stage-method validation and exact artifact spec/kwargs/cell metadata passed. No policy-tail flag is enabled.

All eight prefits were actually executed, sequentially, using the installed CPU plugin into `/tmp/q6_fits/`.
`prefit.sh` is the exact complete command list; `results/prefit_commands.json` has argv arrays. Stage mode is a
server-wrapper flag and is therefore omitted from the plugin-only prefit command; the fitted method already
fixes the wrist metric. Serve with every flag from the arm JSON.


| Artifact | Bytes | Fit seconds | SHA256 |
| --- | --- | --- | --- |
| r5q6_p_l10_500_phase2.pkl | 117133047 | 37.399 | `a70dec58726749433e6780c34dd6af65854a0b672cd7349a1cd6d26ca318e28c` |
| r5q6_p_l10_500_tail.pkl | 117133059 | 36.631 | `a0ad6da613ded327b076b445dbe506fb9ccb34d29519dbb47da50e18c36d1bd4` |
| r5q6_p_l10_50_phase2.pkl | 21227208 | 7.781 | `07ecab4cb2fac0be64913ee70e81d6d012b7137c22467dba9c9c2c8de514e3d5` |
| r5q6_p_l10_50_tail.pkl | 21227220 | 7.711 | `cba87a0255fc02d63bf8de50eae4d3ee5e9fd0f2dc3021461d66601df304605b` |
| r5q6_p_spatial_500_phase2.pkl | 50789085 | 15.222 | `0924ca21a59dc86987632edcb8bce8bce6e658b08e6f294a861e2ae3dcfc2674` |
| r5q6_p_spatial_500_tail.pkl | 50789097 | 15.408 | `c907862eda5347412e8b67508bcea95ce4fca09ad2b99b962838d3488e44e980` |
| r5q6_p_spatial_50_phase2.pkl | 15429610 | 2.619 | `20bb712ccb6d311fe9b965fdbb582e0c87615a13965bf94bda19f85718af1870` |
| r5q6_p_spatial_50_tail.pkl | 15429622 | 2.648 | `462c69185ef44cfa7b8bb65f3b9a7376d48e34cd5e7c088d19b59471499054df` |


| Suite / scale | Stored library NPY bytes | Representation bytes | Tail artifact bytes |
| --- | --- | --- | --- |
| l10 / 50 | 695839536 | 992640 | 21227220 |
| l10 / 500 | 7768083936 | 11081472 | 117133059 |
| spatial / 50 | 268320886 | 382768 | 15429622 |
| spatial / 500 | 2875341211 | 4101784 | 50789097 |



Representation is **376 bytes/row**, excluding shared action payload and fixed PCA/calibration arrays.
Valid π0.5 full-chunk actions add 280 bytes/row. Actual fit pickle sizes above include padded action chunks and
auxiliary arrays. Stored library bytes count top-level NPY files (full-resolution keys included, token
subdirectories excluded). Owner-supplied deployed-pickle reference is **431 MB spatial / 1,103 MB l10**; those
reference figures were not remeasured here. Detailed fit kwargs, timestamps and thresholds: `results/fits.json`.

Exact prefit commands, already run (execute on a fresh destination; existing files are intentionally refused):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_l10_50_tail --os-tag r5q6_p_l10_50_tail --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_l10_50_tail.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_l10_500_tail --os-tag r5q6_p_l10_500_tail --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_l10_500_tail.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_spatial_50_tail --os-tag r5q6_p_spatial_50_tail --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_spatial_50_tail.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_spatial_500_tail --os-tag r5q6_p_spatial_500_tail --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_spatial_500_tail.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"phase_particles","budget":2,"gates":"all"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_l10_50_phase2 --os-tag r5q6_p_l10_50_phase2 --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_l10_50_phase2.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":2,"gates":"all"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_l10_500_phase2 --os-tag r5q6_p_l10_500_phase2 --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_l10_500_phase2.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"phase_particles","budget":2,"gates":"all"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_spatial_50_phase2 --os-tag r5q6_p_spatial_50_phase2 --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_spatial_50_phase2.pkl
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r05.q6_wrist_blind.judge:WristVisionConfirmedBlindJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"phase_particles","budget":2,"gates":"all"},"events":"none","progress_guard":"noprog_span","stuck_guard":"vision_confirmed"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q6_wrist_blind/results/prefit/r5q6_p_spatial_500_phase2 --os-tag r5q6_p_spatial_500_phase2 --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-blind --os-tokens off --os-judge guard_only --os-fit-artifact /tmp/q6_fits/r5q6_p_spatial_500_phase2.pkl
```



## Final reproduction and coordinator handoff

From `/home/weiland/projects/openpi`, `bash exp/offline_search/rounds/r05/q6_wrist_blind/verify_final.sh`
reruns unit/parity, the eight Q6 plugin selftests, eight existing recipes, eight harness smokes, arm validation,
eight-connection concurrency and full-library/full-stream evidence. It contains the exact CPU/thread prefix;
per-command argv and return codes are in `results/final/*_commands.json`. Use fresh output paths on a repeat:
change the test groups' `--tag final` to a new tag and `/tmp/q6_concurrency_final` to another `/tmp/q6_*` path
(plugin logs append and the concurrency driver deliberately refuses existing directories). The command to audit/write
this report is:

```bash
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q6_wrist_blind.write_handback
```



Coordinator: copy the four `*_tail.pkl` artifacts into `<RUN>/fits/`, check the SHA256s above, replace `<RUN>`
in `arms_q6.json`, and emit using the installed `closed_loop.ops.emit_arms`. Start with the normal coordinator
short smoke on each suite/scale: verify startup `stage1_mode=wrist_only`, `miss_steps=10`, first/after-MISS
vision, no more than one consecutive blind decision, and `src=cache_blind` action equality to anchor steps5–9.
Then run the paired 500-init evaluation if the smoke passes. The phase2 file is optional and needs its own
matching prefits. Copying into the run store, model serving and rollouts were not performed by Q6.

The exact CPU integration smoke can be repeated before coordinator serving with
`run_checks plugin --tag coordinator_smoke` using the module and CPU prefix in `verify_final.sh`;
`results/final/plugin_commands.json` lists each complete per-arm command and all required wrist/blind flags.

Report SR, realized vision share v and MISS share m. Primary owner cost remains `.152*v + .848*m` per five
controls, with MISS charged as full inference. If reporting the K3 measured wrist-cost basis separately,
include base-camera completion on MISS; do not mix those eager measurements with owner constants. No new
hardware-cost measurement or IR/SR improvement is claimed here.

Decomposition: Q6 preserves K3 wrist selection/synthesis at fixed library; the method change is the visual
confirmation used by blind-gap stuck detection. Blind control completes an accepted chunk or advances
phase particles in the separately named variant. Library size and kref differ by the established 50/500
presets. These offline rates establish predicate behavior, not causal success or error-based arm selection.

## Owned files and read-only dependencies

`results/deliverables.json` records source/script/arm sizes, modification timestamps (installation times for
these newly created, previously unused paths), and SHA256. Shared dependency hashes before/after final
verification match in `results/dependencies_{before,after}_final.json`. No shared installation step was needed.


| Owned file | Modified UTC | SHA256 |
| --- | --- | --- |
| __init__.py | 2026-09-28T02:23:52.356993+00:00 | `6361c0764a2f193d75a473eb844bdbe3f8dfa98a354491dfe54dd9c2d69ec098` |
| arms_phase2.json | 2026-09-28T02:23:59.709998+00:00 | `6d3aac56db60c4b57c41b6edc4c6f72878e44601ba23af83d4a6fa234342eb93` |
| arms_q6.json | 2026-09-28T02:23:59.708998+00:00 | `713ac8af3a58f0a76054389c33bdf1672b1b4c10e4196afc6fc62dbf481e226b` |
| common.py | 2026-09-28T02:23:52.484993+00:00 | `01a1ec0c886aecfa8e6e2f0a6a3176a0c26d7fe2c2e7ff0f9d1819cd3ce36c8d` |
| concurrency_test.py | 2026-09-28T02:24:49.407034+00:00 | `968f15f432e27d60c5fae7c1774d75f498c6ff53e4e8e02b9948fd304424c1d9` |
| evidence.py | 2026-09-28T02:31:37.988352+00:00 | `c1bfdbe281f418cc11c9f022945e5a72d07bd048fdd94ca2fb2f0dad00744267` |
| judge.py | 2026-09-28T02:23:52.421993+00:00 | `5cdec536abf9f358c44055e1f53225498f02df553f97ad8feee64b31e27c4406` |
| plugin_selftest.py | 2026-09-28T02:24:49.277034+00:00 | `41c335ee59566d574bea8b751af5fdce9fc4d443ce27975c5c3f4cbfa1dc371f` |
| prefit.sh | 2026-09-28T02:23:59.709998+00:00 | `d5453757b785909cad94206f64e9af72e15e69bf874e851ee65af8dd7aee3f94` |
| prepare.py | 2026-09-28T02:23:52.544993+00:00 | `0b4b2b8d2c9f2df3dfe3b1537d1acfb8c17ea4e6d9888aa2c9af9268647aeed9` |
| run_checks.py | 2026-09-28T02:29:36.796254+00:00 | `4d655b6c301724f3de8bd6fa27f5d609041348d6b87bda591f445be2e238b267` |
| tests.py | 2026-09-28T02:26:24.764105+00:00 | `38be3a5b3539e7524f99788b727f44090642fb407c6e40ffdab7965ae3200ac4` |
| validate_arms.py | 2026-09-28T02:29:36.858254+00:00 | `2e93eca130158e49a6f9d561d24e2db2e954e115f2eaa23a06b9b129bd5c4a8d` |
| verify_final.sh | 2026-09-28T02:32:12.853380+00:00 | `afe6e29c26071778c9340116a9a532347c7a9290d420ad3f1fd71d606fb6d737` |
| write_handback.py | 2026-09-28T02:55:09.255764+00:00 | `f9dca73a7a54e18473b473aee3af2ef70f9f29c086078073ae20102cff462280` |

