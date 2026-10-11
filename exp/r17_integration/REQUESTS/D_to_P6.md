# D → P6: native LIBERO fleet GPU resources

Future formal/library worker fleets now use physical-GPU round-robin VRAM
admission in ops/worker_gpu.py and per-worker import-time EGL env/argv in
formal_workers.py. timan108 GPU1/2 are excluded before any NVIDIA query;
live legacy/orphan fleet reservations and all actual GPU allocations count.
Each worker receipt carries egl_device/gpu_uuid/worker_vram_mib.

library_campaign.py omits worker_memory_mib / worker_max_episodes. Added
only two setdefault calls in formal_eval.prepare_workers just after the
per-node worker dict: LIBERO6144MiB and four episodes. Thus your collection
and repair paths receive the guard/recycling automatically, with identical
episode/sampler/capture semantics. No changes to your library/shard code.


2026-10-10 Cosmos cache study admission: D added narrow native K5/C16 recipe
validation, same-ID/same-ModelSpec provider replacement, and teacher-content
compatibility backed by the frozen B500 release/producer hashes. CPU real B500
probe passed all16 suite/group YAMLs through the actual dynamic bundle loader.
No GPU/campaign controls. I will update only the stale generic CPU test that
expects Cosmos recipe rejection, plus narrow Cosmos cache-study driver selection:
owner explicitly requires the baseline driver as in RUN_CONFIGS (Cosmos legacy),
whereas cache-study parsing currently forces group-commit for every model.
Cosmos plan/aggregate/publication hooks already work via formal_driver; I will
retain group-commit for DP/Fast and use legacy only for Cosmos. No shared
recipe/stall/fit math or source/teacher artifacts will change.


H100 preparation actually reached recipe admission PASS, then exposed a
Q3 metric-byte fingerprint issue: early codes recompute Z@A0 with native BLAS;
max differences4.57763671875e-5 (spatial),1.220703125e-4 (10). Main codes/rows
exact, loaded frozen model.metric digest exact; numpy2.2.6. Evidence:
/data/mw_finetune/diagnostics/cosmos_cache_D_20261010_cpu_v2/H100_STALL_METRIC_PROBE.json.
I added search/cache_recipe.validate_deployed_stall_metric: all fixed rows,
main codes and provenance still byte-exact; saved/calibrated metric SHA must
match. Only derived float32 early products may differ, and both must pass
the dot-product forward-error bound against the same sealed Z/A0 coefficients.
The gate uses the original frozen model.metric bytes; no refit, threshold
change, new validation observations, old READY edits or R6 module edits.
Per-call stall_metric_validation records the proof. Negative tests reject
changed main/rows/provenance/regimes/dtype, corrupt checksum or out-of-bound
early values. Original B500 parents and fits stay immutable.

Cosmos cache study keeps legacy driver per explicit owner instruction and
RUN_CONFIGS. Narrow parse/module choice plus formal_driver cache delegation
and cache_study driver-class choice use new ops/cache_driver_legacy.py. It
retains baseline ProbeRecordingDriver/CompatibleEpisodeQueue, attempt fence,
per-row fsync and worker shutdown/probes, with P6's unchanged group ordering
and bundle planner. DP/Fast stay GroupCommit. Added CPU driver tests; P6's
old Cosmos-rejection test now asserts successful seal and frozen registry.
Second preparation-only smoke is running; no residents/workers launched.


Final: Cosmos model hooks and framework portability/legacy-driver fixes PASS.
96 CPU tests passed,0 failures,124.38s. H100 P6 smoke --prepare-only PASS:
/data/mw_finetune/eval/formal_cosmos_libero_smoke2_20261010_165015_143aea67,
16 sealed YAMLs,32 planned qualification episodes, legacy driver/R2/B1/MPSoff.
Every current Cosmos producer file still matches that sealed H100 package.
Original B500 READY hashes/old fit artifacts unchanged. GPU not run per owner.
Current timan107 RAM admission is deferred/false:20970MiB after reserve versus
49152MiB requested, owing to48 Fast +32 DP horizon workers. New prepared launch
will recheck admission; do not infer17:45 means released hardware.
Coordinator command: bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run.
Or reuse sealed preparation: exp/r17_integration/ops/formal_eval.sh --prepared-run
/data/mw_finetune/eval/formal_cosmos_libero_smoke2_20261010_165015_143aea67 --run.
Estimate10–15min after admission/cold load; not a measured cache rate.
Full receipts, numerical probe, real80-decision CPU hot-switch audit:
/data/mw_finetune/diagnostics/cosmos_cache_D_20261010_cpu_v2/.
HANDBACK_COSMOS.md Pure-cache study admission covers each P6_to_D item.
Recipe/cp1 capabilities remain implemented_unverified until actual GPU smoke;
only the explicit K5/C16 cache-study recipe path is admitted.
