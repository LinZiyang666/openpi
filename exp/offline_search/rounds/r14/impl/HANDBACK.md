R14 implementation handback — sol, 2026-10-04

The frozen layer-five implementation is complete on CPU. It provides independent task/pool scope, main and early metric, head, continuity and guard switches above R10Recipe, plus task_periodic above the unchanged R11 lower layers. Production heads use every selected row. Guard-only pooling is retained solely as a CPU equivalence artifact. Nothing under `rounds/r14/astra/`, `rounds/r13/` or the existing R13 roots was edited. No server, GPU, remote operation, sync or chain was run. The eight test-A roots remain staged.

The source of truth is `astra/SPEC.md`, with these binding SHA256 values:

| Input | SHA256 |
| --- | --- |
| SPEC.md | e7828b68b7b9071d59bb9f0c12495fbd349acbb1c2c7d04319a93585ba26fed6 |
| design.json | abb9000ccb9cfaaef7e4a1565aa91251daa291a954a85fbe681d07f1fe79a636 |
| allocation_fits.json | 1ac5676e59e7d703469e919a7918ec6aae7bcb5c743e1593cd7199536295bd74 |
| pilot_grid.csv | 298510ac4d4c5a94fee5d932989b049b2124f997183c7ce444eb31127ca976fe |
| pilot_selection.json | 91117ef67c583e465dc051e6cb9e7a5fc1bb040791e2dde981c71abd90783fd3 |
| test_grid.csv | 16f66711e7485c7e1385f9d25b9d95b461f9973709c32a5ed5283863c60f3fd6 |

Files relative to `exp/offline_search/rounds/r14/`:

| Files | Purpose |
| --- | --- |
| `knob/recipe.py`, `controller.py`, `__init__.py` | Serving adapter, query-task partitions, inherited periodic coins and reason 73, versioned diagnostics |
| `knob/fitting.py` | Full-row LOEO heads, pooled main/early geometry, five complete episode folds, exact pooled continuity, inactive guard auxiliaries |
| `knob/artifacts/*.pkl`, `impl/fits.json` | Final SHA-bound catalog and fitted artifacts |
| `impl/build.py`, `common.py`, `seal_fits.py`, `implementation_sources.json` | Verified original artifacts/PCA, numerical construction and source binding |
| `impl/allocation_audit.py`, `allocation_audit.json` | Independent exact recomputation of all 16 priority/reference-cap/bias/task-cap fits from permitted B inputs |
| `impl/pilot_identity_audit.py`, `pilot_identity_audit.json` | All six frozen selections already consumed in B and disjoint from the actual served subset; middle GR00T cross-model exclusion checked |
| `impl/mechanics.py`, `test_r14.py`, `mechanics.json`, `mechanics_tests.{log,xml}` | Real-plugin CPU mechanics and equivalence evidence |
| `impl/regression.py`, `r13_tests.{log,xml}`, `regression_tests.{log,xml}` | Original R13 suites and original regression-runner suite, with writes redirected to R14 or `/tmp` |
| `impl/prepare.py`, `pilot_roots.json`, `test_a_roots.json`, `build.json` | Six B roots, eight staged A batches, SHA-bound definitions/contracts and safe installer |
| `impl/queue_pilot.txt`, `queue_test_a.txt`, `test_queue_segments.json` | Frozen arm order and host assignment |
| `impl/plans.py`, `plans.json`, `plans/*/*/plan.json` | Fourteen standard asset plans, generated contracts/configs and H100 path-relocated fits; no transfer |
| `impl/runtime_audit.py`, `runtime_closure.json`, `*_SOURCES.sha256` | Clean serving closure, per-host source hashes, external action payloads and all fit hashes |
| `impl/shared_edits.py`, `shared_edits.json`, `before/*.py`, `protected_before.json` | Atomic shared edits, saved preimages and protected-file evidence |
| `impl/release_audit.py`, `release_audit.json`, `RELEASE.sha256` | Final integrity/count audit and hashes for release evidence |
| `impl/coordinator_commands.sh` | Exact fleet-specific plan/sync/chain/collect commands; syntax checked only |

Only `closed_loop/plugin.py` and `look_recording.py` changed outside R14. Their new `--os-r14-audit` option defaults off. It adds an accepted-buffer preflight, an actual served-artifact SHA, and a versioned `offline_search.r14.v1` side record that survives the 40-scalar limit. Recording adds that JSON to B look shards only when enabled. The preflight repairs malformed accepted HIT histories before blind-tail indexing; default R13 behavior remains covered by both required reruns. Original R10 recipe files were unchanged.

All arms log the actual combined retrieval/correction query time. Partition adapters additionally measure retrieval and correction separately. The unchanged original corrector uses a read-only donor/gate/norm reconstruction for diagnostics; that audit's extra time is explicitly recorded separately. Original artifacts leave separate retrieval/correction fields unset rather than assigning estimated timings. Accepted logs carry N/V/M, task/caps/coin/guard/knob, branch, parent rows/tasks, fallback counts and the actual fit hash.

The catalog contains 104 distinct configurations: 64 A variants, eight CPU guard-null artifacts, 16 task-periodic artifacts and 16 original periodic references. There are 95 R14-owned pickles: 88 new adapter fits and seven unchanged R11 controllers constructed on verified original off graphs because those doses were absent from the old artifact grid. Nine references reuse existing R11 files. Catalog bytes total 7,473,646,586. Each reference cap was independently recomputed and exactly matches the freeze. No bias, gain or cap used a pilot outcome, A result or reserved R13 identity.

| Cell | Production rows | Episodes | Recomputed pooled continuity |
| --- | ---: | ---: | ---: |
| pi05_l10_50 | 2925 | 50 | 0.2240506276234645 |
| pi05_l10_200 | 11437 | 200 | 0.15859028586618726 |
| pi05_l10_500 | 29472 | 500 | 0.13302612220558133 |
| pi05_spatial_50 | 1161 | 50 | 0.2854965560566725 |
| groot_l10_50 | 2955 | 50 | 0.2359510930544424 |
| groot_l10_200 | 11702 | 200 | 0.17335799238755728 |
| groot_l10_500 | 29631 | 500 | 0.14530788356417473 |
| groot_spatial_50 | 1201 | 50 | 0.290299529492649 |

Every cell shares its verified selected-library PCA, action sigma and kref with the original recipe. Scope changes embed every allowed donor in the query task's coordinates. Geometry changes regenerate every LOEO proposal, task/pool head and complete five-fold GC_dist fit. Pooled heads have 207 inputs, task heads 217; weights remain episode-balanced, with the inherited 384 RFFs, alpha 100 and seed 0. Parent donor rows and foreign donor tasks are preserved through full action synthesis and corrector anchoring. The corrector uses the actual donor minimum geometric distance, changes only the first 10 controls in six motion channels, and leaves the gripper intact. Step-zero correction is zero.

Validation completed with CUDA hidden and the prescribed CPU affinity:

| Suite | Result |
| --- | --- |
| R14 real-plugin mechanics/equivalence/root contracts | 78 passed |
| Original `rounds/r13/impl/test_r13.py` and `test_compatibility.py` | 71 passed |
| Original R13 regression-runner suite | 296 passed, 25 subtests passed |
| Independent frozen allocation audit | 16 exact fits |
| Frozen pilot identity audit | Six cells passed |
| Clean-interpreter serving/import audit | 104 artifacts passed; 55 source files; four parent action payloads |
| Standard dependency plans | 14 roots planned |

The R14 tests compare base/off actions byte-for-byte to both R10Recipe and R11 off over 120 decisions per cell, test guard-null over another 120, and compare periodic/zero-gain allocation at both doses. They check 1920 frozen reference proposals plus same-step retries, exact keyed coins, cost reconciliation, independent connection ordering, reset/clone, policy/cache tails, invalid results and wire tails, partial execution and early termination. All seven nonbase A variants run through the real plugin in every cell. Complete pooled parent-row support and query-task main/early arithmetic pass; foreign donor anchors exercise the correct gate. Continuity is identical in 30 nominal stale/early cases per cell and changes all 30 injected fresh cases. Side-record tests cover discarded corrected proposals on misses and the scalar-field limit. Pytest cache is disabled; scratch is outside the repository.

The CPU seed test verifies the WorkerSpec/execution seed route and repeatable/different CPU policy RNG streams for replicate seeds 20261401 and 20261402. It does not launch LIBERO or a real policy server. The coordinator must still verify that these seeds change simulator/policy randomness before interpreting the two replicates as independent. The knob coin seed remains zero.

Pilot roots are under `/home/weiland/trace_runs/os_closed_loop/`, one root per cell and exactly 50 frozen B pairs per arm:

| Root basename | Arms | Fleet |
| --- | ---: | --- |
| r14_pilot_pi05_l10_50 | 14 | H100 / timan108 |
| r14_pilot_pi05_l10_200 | 6 | H100 / timan108 |
| r14_pilot_pi05_spatial_50 | 14 | H100 / timan108 |
| r14_pilot_groot_l10_50 | 14 | local 4090 / timan107 |
| r14_pilot_groot_l10_200 | 6 | local 4090 / timan107 |
| r14_pilot_groot_spatial_50 | 14 | local 4090 / timan107 |

These are 68 arms / 3400 episode runs. Normal `dev_contracts.json` binds each manifest, served subset and actual fit SHA. All arms record real looks and raw distances; A-type pilot arms also record shadow chunks and corrected cache proposals on misses. Periodic/task-periodic arms add no shadow call. The identities are the exact pairs in `pilot_selection.json`; no new selection was made. The required old R13 tests may read their own reservation as part of their original checks; it was never a R14 fit or selection input.

Test-A batches are under `impl/test_a_staged/`, with byte-preserved R11 `r11_knob_{1,2,3,4}/eval500.json` templates and recording disabled. Install only after the coordinator's release decision:

| Root basename | Arms |
| --- | ---: |
| r14_test_pi05_l10_rep0 | 36 |
| r14_test_pi05_l10_rep1 | 12 |
| r14_test_pi05_spatial_rep0 | 12 |
| r14_test_pi05_spatial_rep1 | 4 |
| r14_test_groot_l10_rep0 | 36 |
| r14_test_groot_l10_rep1 | 12 |
| r14_test_groot_spatial_rep0 | 12 |
| r14_test_groot_spatial_rep1 | 4 |

These are 128 arms / 64,000 episode runs, with 64 A arms and 64 B-method arms. The installer checks staged definition and fit hashes, preserves YAML/eval500 bytes, rewrites owned JSON root paths, refuses existing destinations and recomputes root hashes. Its `/tmp` installation check passed; no final trace-root test batch was installed.

Both queues use `@local /abs/root|arm arm ...` or `@h100 /abs/root|arm arm ...`. The pilot queue has six lines. The test queue has 33 sequential segments but only eight unique roots: this preserves the exact globally frozen CSV order when a batch recurs. A root can be served by only one fleet at a time. Run at most one queue consumer per fleet; each supplied command processes its selected segments sequentially. GR00T stays on the local 4090 and pi05 stays on H100. No root contains `server_host`; the coordinator chooses its designated fleet.

Runtime closure for deployment is in `runtime_closure.json`. The identical `H100_SOURCES.sha256`, `TIMAN107_SOURCES.sha256` and `TIMAN108_SOURCES.sha256` list all 55 required experiment source files. The list itself has SHA256 `5e8ce0958b45a53203d0e582aee8b220118d636b455357ec2446068986547264`. It includes the inherited R11 serving closure, the shared edits, worker/execution/seed routing and these new serving modules:

| File | SHA256 |
| --- | --- |
| knob/__init__.py | 9f876a88f00588c1ef33302a9a75c472be06e8e6444a7bb5acad46633a83dbf9 |
| knob/controller.py | 9f2e53703b68c9b42ad8acd73a19c935c63cba215db26a7bad4d295593e5f50d |
| knob/recipe.py | bd8741b7564e4045c7640fe38576a794fe91b627ba6a578606c60eb0d808d696 |
| closed_loop/plugin.py | fac08f913b37fd401a32b9924551bb5508d7278718cf013c68d1ab97e0ff8872 |
| closed_loop/look_recording.py | 9a2070aa30608583b2d812f6fa8ee5a5e1300a075e798a4d33887f00939efc12 |

No Astra explorer or fitter is imported while loading the 104 serving artifacts. Their coefficients/tables are self-contained; the only external array opened by the fitted method is its lazy `IndexedRows` parent action payload. The four payload hashes are:

| Store-relative path | SHA256 |
| --- | --- |
| library/pi05_l10/bpool_cs/action.npy | 8a573c0606cc4de2345092baed20e97050de814d702e0fa484532a76f673b154 |
| library/pi05_spatial/bpool_cs/action.npy | b7ecd7725733df1e363d550f08c9be1601fdc8a835a620ff2f0b629980f5ed64 |
| library/groot_l10/bpool_all/action.npy | f4b6d4380f98e7a3766cf6a9484cf0559c75a1e14c3fde617e08da0c8ce0c3dc |
| library/groot_spatial/bpool_all/action.npy | 2bafed2b909d1723ff5ba764e7b7275b6ec844c8f5335d870087f82112c4c8ab |

Use the standard asset planner as provided. PluginRuntime/native initialization also discovers current metadata, native PKLs and other stored library action/task/key_v0 tables; those inherited dependencies are explicitly enumerated in each plan. Do not transfer just the four method payloads. `fits.json` records original pickle hashes; each plan records both source and served hashes after H100 path relocation. The plugin logs the actual served hash, which may differ from the original catalog hash. Local plans report zero server transfer; their file-byte totals describe dependencies, not bytes to copy. Model checkpoints and existing host environments remain prerequisites of the ordinary topology.

Run these local validation/reproduction commands from the repository root. All Python invocations use the required CPU settings:

```bash
cd /home/weiland/projects/openpi
CPU=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python)
IMPL=/home/weiland/projects/openpi/exp/offline_search/rounds/r14/impl
sha256sum -c "$IMPL/RUNTIME_SOURCES.sha256"
sha256sum -c "$IMPL/RELEASE.sha256"
"${CPU[@]}" -m pytest -p no:cacheprovider --basetemp=/tmp/r14_review_pytest --junitxml="$IMPL/mechanics_tests.xml" "$IMPL/test_r14.py"
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.regression r13
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.regression regression
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.allocation_audit
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.pilot_identity_audit
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.runtime_audit
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.release_audit
```

The completed fit construction used the commands below. Rebuilding changes artifact/contract hashes; reseal and refresh only owned unserved definitions afterward. `--resume` refuses roots with runs/state/sync receipts.

```bash
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.build --workers 4
# The completed numerical fits were retained when filling absent R11 reference doses:
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.build --recover
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.seal_fits
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.prepare prepare --resume
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.plans
```

The following commands are for the coordinator later; sol has not executed them. Source setup pushes the ordinary code snapshot, including the hashed runtime closure, to H100/timan108 and to the timan107 worker island. Source lists use repository-relative paths for checking the deployed tree. Setup must respect the coordinator's R13 scheduling decision and fleet availability.

```bash
SERVER_HOST=h100 WORKER_HOST=timan108 "${CPU[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup
SERVER_HOST=local WORKER_HOST=timan107 "${CPU[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only
SERVER_HOST=h100 WORKER_HOST=timan108 "${CPU[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
SERVER_HOST=local WORKER_HOST=timan107 "${CPU[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker

bash "$IMPL/coordinator_commands.sh" plan local pilot
bash "$IMPL/coordinator_commands.sh" plan h100 pilot
bash "$IMPL/coordinator_commands.sh" sync local pilot
bash "$IMPL/coordinator_commands.sh" sync h100 pilot
bash "$IMPL/coordinator_commands.sh" chain local pilot
bash "$IMPL/coordinator_commands.sh" chain h100 pilot
bash "$IMPL/coordinator_commands.sh" collect local pilot
bash "$IMPL/coordinator_commands.sh" collect h100 pilot
```

The local fleet uses timan107, four servers at ports 23140–23143, sync port 23197 and server CPU affinity 22–37,66–81. H100 uses timan108, ports 23240–23243 and sync port 23198. WPS defaults to four; the script clears inherited manifest/pool/seed/recording overrides. Do not run overlapping consumers of one fleet or serve one root through a second fleet.

Before test-A release, require the frozen B gates for every affected cell/dose: total realized IR within 0.02 of target; paired allocation/reference IR within 0.01; non-guard call-position symmetric difference/union at least 0.05; and at least 0.02 range across tasks of allocation-minus-reference knob-call fractions on the allocation arm's own states. Report every observed task, missing tasks, mandatory-guard floor, terminal N/V/M and episode-cluster bootstrap intervals. There is no pilot refit or arm substitution. Allocation failures stop that matched-budget claim; A-only ablations can be released separately. Size-500 allocation is library-only transfer and requires the smaller long-horizon controller gates for its model. Live donor/head changes and continuity fallback counts remain to be measured.

After those gates and the live rollout-seed check, install and plan/sync/chain the frozen test queue. A staged plan can be inspected before installation via the same plan command; sync/chain require installed roots. Running `test-a` without a replicate argument preserves the full frozen order. Optional `rep0`/`rep1` filters preserve only that replicate's relative order and require a coordinator scheduling choice.

```bash
"${CPU[@]}" -m exp.offline_search.rounds.r14.impl.prepare install-test-a
bash "$IMPL/coordinator_commands.sh" plan local test-a
bash "$IMPL/coordinator_commands.sh" plan h100 test-a
bash "$IMPL/coordinator_commands.sh" sync local test-a
bash "$IMPL/coordinator_commands.sh" sync h100 test-a
bash "$IMPL/coordinator_commands.sh" chain local test-a
bash "$IMPL/coordinator_commands.sh" chain h100 test-a
bash "$IMPL/coordinator_commands.sh" collect local test-a
bash "$IMPL/coordinator_commands.sh" collect h100 test-a
```

The release evidence establishes implementation mechanics, full-row construction and contract integrity. Pilot/live budget and engagement gates, simulator/policy replication and any R14 outcome analysis remain coordinator work after launch. All failures must be retained under the frozen design.
