R13 implementation and CPU mechanics pass. All 127 frozen astra files are unchanged. No pilot/A rollout, server, GPU, remote sync/setup/chain, tether, or git operation was performed. Test-A remains staged and uninstalled. Live budget, engagement, outcome and replicate-dependence gates remain the coordinator's next work.

Files and behavior:

- `../knob/recipe.py`: `R13Knob.query`, `accepted_history`, `configure`, `reset`. Feedback uses geometric `_dist(q)[7].min()`, float64 frozen arithmetic, accepted N/V/M, strict `>`, unclipped debt/threshold, conservative invalid-distance calls, original guard reasons/memo, proposal retries and connection-local checkpoints. Audit scalars come first. Collision audit found inherited guard/lifecycle reasons below 61 and R11 reasons 61–68; new knob-only reasons are 69/70. Off/periodic dispatch unchanged to R11; C10 is exact periodic K=0.
- `prepare.py`: `build_fits`, `emit_root`, `prepare`, `install_test_a`. `fits.json` binds 92 entries: 32 feedback, 32 periodic (17 reused R11 bytes, 15 new exact-solver artifacts, including .45), 8 off, 8 static B, 8 C10, 4 standard C5. Frozen scalars and settings/source SHA are embedded in new fit pickles; no refitting or exploration import at serving load. C5 uses the existing `pure_inference` / `periodic:1` arm mode. Its legacy slotted block carrier was replaced with field-identical namespaces so the existing planner can inspect it; all 40 checked B proposals preserve bytes (`pure_c5_planner_carrier.json`).
- Shared default-off atomic edits: `closed_loop/plugin.py` adds `--os-rollout-seed`, `--os-record-look-raw-distance`, `--os-record-look-cache-proposal`; `look_recording.py` saves float64 geometry and the corrected first ten normalized cache controls before MISS replacement; `worker_pool.py:install_worker_options` wires the simulator seed; `ops/h100/control.py:execution_env` propagates it. New `closed_loop/rollout_seed.py:seed_policy_process` seeds Python/NumPy/Torch only on explicit opt-in. `shared_changes.diff` compares the four edited shared files to saved originals.

Verification (CPU affinity 22–37,66–81, CUDA hidden; logs/XML/JSON here):

- `test_compatibility.py` + `test_r13.py`: **71 passed**, zero failures/skips. Off ≡ R11 off ≡ R10 on 960 real-plugin decisions across all eight cells; zero differing served action bytes or serving decision payloads. All nine existing R11 methods also match the saved previous plugin on 432 decisions. Comparison normalizes run identity, method labels, timing and global reservation fields, retaining reasons/flags/HITs/scores/topk/extras/actions. Existing default-off recording arrays and guard metadata are byte-identical.
- Reference equality: 3,200 proposals + 3,200 exact retries across all 32 frozen feedback configs. Real PluginRuntime/CacheOrchestrator tests use recorded B camera keys/states and fake stages 2/3: accepted-ledger reconciliation, one inherited query/call, normal 5+5 policy tail then inherited stale retrieval, invalid/missing tail and partial-execution fallbacks, early termination, duplicate refusal, reset/clone isolation, reversed independent connections, strict ties/saturation/nonfinite cases, original guard reason/memo, all 32 periodic doses, C10 across sizes, C5 every five controls, eight static B arms, and contract errors without fabricated action/cost commits. Exact active-MISS cache proposal and raw recording are checked on both models. Selection/planning checks find zero unknown/missing identities.
- Existing regressions via `regression_runner.py`: **296 passed + 25 subtests**: 253 closed_loop, 16 R11 knob, 14 R12 lookhalf, 13 R12 guard. Unchanged assertions run with scratch redirected here; the old no-runs assumption uses isolated definition copies of already-active R12 roots. JUnit counts the 25 subtests in its 321 total.
- `periodic_solver.json`: unchanged R11 solver reproduces all 32 settings exactly. `plans.json`: real dependency planning, fit relocation, recording contracts, worker environment and serving argv pass for all 22 roots / 320 arms, with zero transports/launches. `runtime_audit.json`: all 92 fits unpickle without exploration imports. `benchmark.json`: incremental wrapper median 0.94–1.13 ms on recorded pi05/GR00T L10 size-50/500 CPU queries; includes geometry, ledger checks and proposal copying, excludes policy latency. `RELEASE.json` binds evidence SHA and frozen-input checks.

Reproduce the new tests and regressions from the repo (do not launch serving):

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/weiland/projects/openpi/exp/offline_search/rounds/r13/impl/tmp .venv/bin/python -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r13/impl/test_compatibility.py exp/offline_search/rounds/r13/impl/test_r13.py --basetemp=exp/offline_search/rounds/r13/impl/pytest_mechanics_final --junitxml=exp/offline_search/rounds/r13/impl/mechanics_tests.xml
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/weiland/projects/openpi/exp/offline_search/rounds/r13/impl/tmp .venv/bin/python -m exp.offline_search.rounds.r13.impl.regression_runner
```

Replicate finding: the CSV seed now reaches `WorkerSpec.seed` → conductor `--seed` → `worker_entry._build_episode_setup` → LIBERO `env.seed`. Startup logs `OSCL_SIMULATOR_SEED` and server `rollout_seed`. Both deployed policies are Torch backends (pi05 PyTorch checkpoint and GR00T); their ordinary stage-3 noise uses global Torch draws. CPU tests prove equal-seed draw reproducibility, changed draws across replicate seeds, and unchanged R11 keyed coins (seed 0). No simulator render or model inference was launched. These are process streams, dependent on warmup and concurrent request order; equal seeds across server processes can start equal streams. Do not claim independent episode or replicate noise. Identities, banks, scalar configs and knob coins stay fixed; rollout seeds, arm IDs and frozen batch order differ by block.

Pilot roots under `/home/weiland/trace_runs/os_closed_loop/`:

| Root | Fleet | Arms |
| --- | --- | ---: |
| `r13_pilot_groot_l10_50` | local 4090 / timan107 | 12 |
| `r13_pilot_groot_l10_200` | local 4090 / timan107 | 8 |
| `r13_pilot_groot_spatial_50` | local 4090 / timan107 | 12 |
| `r13_pilot_pi05_l10_50` | H100 / timan108 | 12 |
| `r13_pilot_pi05_l10_200` | H100 / timan108 | 8 |
| `r13_pilot_pi05_spatial_50` | H100 / timan108 | 12 |

`pilot_roots.json` and `queue_pilot.txt` bind six separate roots, 56 primary + 8 static arms, 100 exact fresh B pairs/arm, 6,400 episode runs. Normal SHA-bound dev contracts exclude the actual served subsets and the complete cross-model used-B union. Every pilot real look records raw distance; only the eight frozen size-50 primary .32/rep0 arms request shadows and corrected cache proposals. Intended hosts are metadata/queue pins, never hard-wired root server hosts.

`test_a_roots.json` and `queue_test_a.txt` bind 16 roots in frozen model/suite/replicate batch order, 256 arms / 128,000 runs, with exact R11 eval500 bytes and global_order. Definitions are in `test_a_staged/`; recording is off. Installer was verified on an isolated impl destination only. After pilot release gates, install definitions with:

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/weiland/projects/openpi/exp/offline_search/rounds/r13/impl/tmp .venv/bin/python -m exp.offline_search.rounds.r13.impl.prepare install-test-a
```

Deployment closure: `H100_SOURCES.sha256` lists all **57** existing/new serving and worker source/config entries; list SHA256 `0ad940acbd819011deac5bc22209854322c9ac878da46cc7ef1d8e8485b2f461`. Push/verify this closure in the H100 serving tree and selected timan107/timan108 worker trees through the established deployment process once their fleets are free. The seven added/changed runtime files are `rounds/r13/knob/{__init__,recipe}.py`, `closed_loop/{plugin,look_recording,worker_pool,rollout_seed}.py`, and `closed_loop/ops/h100/control.py`. H100 needs knob/plugin/recorder/seed code; both workers need the new worker_pool and existing run_gtp_subset/execution_worker dependency path. The local serving tree already has these files. Normal sync transfers exact per-root fit/payload/config assets and verifies destination relocation SHA; it does not deploy runtime source. Do not replace live server code trees or assume old loaded processes have the new hooks.

Coordinator commands below are provided and syntax-checked only. The script preserves queue/arm order, uses separate roots per cell, existing fleet locks, four serving ports and WPS=4, with local ports 23140–23143 / sync 23197 and H100 ports 23240–23243 / sync 23198. Check availability before use; capacity/ports can be overridden without moving a cell between hosts. From the repo, after source deployment and fleet availability:

```bash
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh plan local pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh sync local pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh chain local pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh collect local pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh plan h100 pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh sync h100 pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh chain h100 pilot
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh collect h100 pilot
```

For A after release/install, execute one block on both fleets. Plan can use staged roots before installation; sync/chain/collect use final installed roots. Start with these exact rep0 commands; only after **both** fleets complete and collect the block, repeat with rep1, then rep2, then rep3. No automatic advance, arm dropping, SR-based stopping or host migration:

```bash
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh plan local test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh sync local test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh chain local test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh collect local test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh plan h100 test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh sync h100 test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh chain h100 test-a rep0
bash exp/offline_search/rounds/r13/impl/coordinator_commands.sh collect h100 test-a rep0
```

Pending release gates follow SPEC/PLAN exactly: per-primary-arm realized aggregate IR error ≤.02 with episode-cluster intervals; paired IR difference ≤.01 for a matched-budget label; non-guard threshold-interior fraction ≥.40 and constant-score selected-position change ≥.20, plus shuffling/nonfinite/saturation diagnostics. Report static failures without retuning. Hold failing primary cells or label authorized measurements Q1 failures. Both size-500 cells and GR00T L10-200 use explicitly frozen library-only initialization; size-500 A remains a library-only transfer requiring smaller-cell pilot passes and its actual-fit mechanics (passed here), never B-validated. No validation/A adjustment was made.

Write-scope exception: the first regression run wrote two `.pytest_cache/v/cache` metadata files and 32 R12 `lookhalf/guard/scratch/contracts` files outside the allowed tree. `write_scope_exception.json` lists all 34 paths/timestamps. The runner was corrected to disable pytest caching and isolate all subsequent scratch output under impl; the complete passing regression rerun used that isolation. Earlier scratch outputs were left untouched because there is no before-snapshot for them. `RELEASE.json` explicitly records `write_scope_compliant=false` alongside the passing functional checks.
