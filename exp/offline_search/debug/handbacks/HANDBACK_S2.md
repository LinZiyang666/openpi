# R8 S2 handback

Implemented client capture, durable transport and verification, coordinator ops, diagnostic oracle payloads, and exact-source library backfill. All implementation changes are within S2 ownership. Shared schema, stock launchers, LIBERO runners, scientific journals, and other coders' files were reused read-only. No GPU, remote deployment, remote simulator/service/worker/chain, tmux operation, prohibited port, git operation, or review-test access was performed.

## Implementation and files

Paths below are relative to `exp/offline_search/debug/`.

| Files | Behavior |
|---|---|
| `client/{__init__,worker,driver,compat,preflight}.py` | Inject capture into the stock runner and worker specification. Stock seed is authoritative, including default 7 and explicit overrides. Persist attempt/generation fencing in both file and stream modes. Python 3.8 compatibility is process-local; no P3 marker is required. |
| `client/{capture,adapter,snapshots}.py` | Copy request envelopes, strip response echoes, capture every issued control including settling and partial final chunks, write contiguous 64-control blocks and events, and numeric reset/sampled/final snapshots. Capture errors become explicit telemetry errors without retrying or changing the real action/inference. |
| `client/parity.py` | Coordinator action-tape comparison of capture off/on, requiring at least three inits and two suites, checking every float64 qpos/qvel byte. No policy or network calls. |
| `transport/{__init__,protocol,sink,receiver,dispatch_fence}.py` | Attributed P3 adaptations with the R8 allowlist, durable offset ACKs, idempotent retries, atomic receipts, authenticated run/arm identity, cross-process receiver locks, default 8 MiB sender queue, and durable sticky spill. |
| `transport/{validation,collect,cleanup}.py` | Validate full client records, replay exact spill bytes, certify file mode through the same store, reconcile accepted journal outcomes and server echoes, and verify per-file/aggregate hashes. Cleanup removes only individually hash-certified spill files after verification. |
| `ops/{__init__,config,receiver,build_client_bundle,install_client_payload}.py` | Frozen manifest/config, structured launch arguments and protected client environment JSON, receiver ensure/reuse, deterministic minimal payload, and validate-all-before-writing installation. |
| `ops/{chain_debug,build_client_bundle,deploy_client}.sh`, `ops/remote/run_arm_debug.sh` | Coordinator-only stock-derived orchestration. Server drain precedes collection; DONE follows receipt/journal/server reconciliation and S3 capture-schema validation. Existing selection/resume/EV markers remain available; scientific journals are never purged by capture verification. |
| `backfill/{__init__,replay,trajectories,prepare_source}.py` | Inventory source feasibility; load exact B-pool saved trajectories or explicit source-map NPZs; replay through the same capture; compare wire robot state, saved simulator state, normalized library rs, decision coverage, terminal control count, and success before admission. Export exact controls/reset from prior client captures with declared library provenance. |
| `backfill/{feasibility,source_audit}.json` | Eight-library source inventory and six sampled exact robot-state comparisons, including source/stat hashes. |
| `tests/test_client_capture.py`, `tests/test_client_ops_backfill.py`, `tests/test_transport_debug.py` | Capture parity/failure isolation, entity/contact/oracle checks, contiguity, transport retry/spill/hash/fence checks, bundle/install/config, source validation, and backfill admission. |

Entity names come from every model id-to-name slot, preserving unnamed entries; contact geom IDs therefore index the actual model catalog. Movable object/articulation descendants are collected while robot/gripper descendants are excluded. Contact forces use `mj_contactForce` when available, with explicit capability status otherwise. A settled-reset table/free-object contact self-check and a post-settle object reference are recorded. Goal predicates are read-only evaluations of the environment's goal state.

Snapshots contain numeric simulation/controller arrays plus Unicode JSON RNG inventories; object arrays and pickle are absent. Unsupported controller attributes are inventoried, and every snapshot has `restore_certified=false`. Actuator substeps are copied from actual `sim.step` calls when exposed, with padding plus a per-control count. The temporary step hook restores the original instance/class lookup semantics.

Oracle payloads are enabled only by `OSDEBUG_ORACLE=1` or an oracle arm/config. They identify unsatisfied, unlifted placement-goal subjects from predicates, include distance/object IDs and availability status, and label privileged diagnostic use. Radius/lift defaults (.10 m/.03 m) are confined to the diagnostic adapter and configurable; the tight arm's .05 m rule and call cap remain in S5. `ops.config` adds server `--os-oracle` together with the client flag when the arm declares oracle use.

## Run and deployment interfaces

The normal coordinator path is `ops/chain_debug.sh RUN ARM [ARM ...]`, after building/deploying a bundle containing that run's YAMLs. Streaming is default. The receiver defaults to port 24080, configured with `OSDEBUG_RECEIVER_PORT`; `OSDEBUG_RECEIVER_HOST` defaults to `ziyanglin.com`. Receiver state/token live under `RUN/state/`; tokens and client environment JSON are mode 0600.

The driver is `python -m exp.offline_search.debug.client.driver` with stock run_gtp arguments. It supplies worker entry `exp.offline_search.debug.client.worker` and passes `OSDEBUG_*` variables alongside the existing worker environment. File mode requires `OSDEBUG_CLIENT_DIR`, `OSDEBUG_CAMPAIGN`, `OSDEBUG_ARM`, `OSDEBUG_MODE=file`. Stream mode additionally requires `OSDEBUG_STREAM=HOST:PORT`, `OSDEBUG_STREAM_TOKEN`, `OSDEBUG_STREAM_RUN`, and `OSDEBUG_STREAM_ARM`. Queue/retry knobs are `OSDEBUG_STREAM_QUEUE_BYTES`, `OSDEBUG_STREAM_FAIL_S`, `OSDEBUG_STREAM_TIMEOUT_S`, and `OSDEBUG_STREAM_CLOSE_S`.

Server additions are:

```text
--os-debug-dir RUN/runs/ARM/debug/server_PORT
--os-debug-config '{"campaign":"RUN_NAME","arm":"ARM","rawkeys_mod":16,"snapshot_mod":16,"draws_mod":32,"writer_queue_bytes":536870912}'
--os-oracle                       # only for diagnostic oracle arms
```

Client output and receipt paths are `RUN/runs/ARM/debug/client/EPISODE_KEY/` and `RUN/runs/ARM/debug/receipts/EPISODE_KEY/`. Fences are persisted beside remote capture directories in `.osdebug_dispatch` and retained on cleanup. Verification rejects incomplete/error receipts, missing accepted captures, outcome mismatches, control/decision gaps, missing declared snapshots, hash corruption, missing server echoes, and orphan server decisions. `--client-only` is a transport-development option and must not be used for campaign admission.

Final tested generic bundle: `/tmp/r8_S2/bundle_final/client_bundle.tar`, 26 payload files, SHA256:

```text
56b4a0b8f7f472ce6f86e16764e7e585e9f66dc5a360418d44b3cee877ef7798
```

The generic bundle has no run YAMLs. Build a new directory with `--run-root`/`--arms` for actual selected arms. It includes S1's shared schema read-only. Installation protects existing `os_cl/run_arm.sh`, `run_gtp_subset.py`, and `count.py`; namespace markers are installed only if absent. Deployment performs checksum/install/import preflight and does not launch a simulator or worker.

## Tests and local evidence

Executed from `/home/weiland/projects/openpi`:

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/r8_S2/pytest_final_2 -q exp/offline_search/debug/tests/test_client_capture.py exp/offline_search/debug/tests/test_transport_debug.py exp/offline_search/debug/tests/test_client_ops_backfill.py
```

Result: **23 passed, 2 skipped in 3.90 s**, exit 0. Both skipped cases are the parametrized real-loopback offset/lost-ACK tests: the sandbox denies socket creation. The no-socket wire test still exercises the actual packed protocol and receiver with an ACK lost after durable application, then exact retry. Spill/replay, file certification, server joins, corruption rejection, and certified cleanup ran successfully.

The fake-env parity test compares every qpos/qvel, issued action, and global Python/NumPy RNG state across capture off/on, including 136 controls, 27 decisions, a two-control final partial chunk, and blocks of 64/64/8. Other tests verify action copies before an environment mutates its input, explicit seed preservation, force/name alignment, gripper exclusion, oracle post-settle reference, and observation failure isolation.

Additional successful checks:

- `bash -n` on all four shell entrypoints; only syntax checking, no remote execution.
- Local `client.preflight`: syntax compilation and six imports pass; stock/simulator imports were not requested locally.
- Cached real Python **3.8.20** at `/home/weiland/miniconda3/pkgs/python-3.8.20-he870216_0/bin/python3.8`, launched with inherited pinned CPUs and `-I -S`: **27 Python files compiled**, six isolated worker/driver/compat/protocol/sink/fence imports pass, strict-zip compatibility and persisted-fence runtime pass. NumPy and the simulator are unavailable in this cached interpreter, so this does not certify their remote runtime. Results: `/tmp/r8_S2/python38_final.json`.
- Deterministic bundle/hash validation, protected installation, corrupt-payload prewrite rejection, and deployment `--dry-run` pass. No network deployment was performed.

Tiny CPU fixture measurement: 504 controls / 100 decisions produced 279,895 bytes. Capture plus fixture bookkeeping/serialization took 437.6 ms versus 8.6 ms for plain fake env steps; their difference is 0.851 ms/control. This is not an isolated real-environment overhead measurement or a production capacity forecast. Raw measurement: `/tmp/r8_S2/benchmark.json`.

## Library source feasibility

`backfill/feasibility.json` is the final inventory; `backfill/source_audit.json` records source hashes and exact comparisons. Certified simulator replays are **zero** for every library because no real simulator was run by S2.

| Cell | Library | Episodes | Known original init IDs | Exact executed-control source candidates |
|---|---|---:|---:|---:|
| pi05_l10 | current | 50 | 0 | 0 |
| pi05_l10 | bpool_cs | 500 | 500 | 500 |
| pi05_spatial | current | 49 | 49 | 0 |
| pi05_spatial | bpool_cs | 500 | 500 | 500 |
| groot_l10 | current | 50 | 50 | 0 |
| groot_l10 | bpool_all | 500 | 500 | 0 |
| groot_spatial | current | 50 | 3 | 0 |
| groot_spatial | bpool_all | 500 | 36 | 0 |

For the two pi05 `bpool_cs` libraries, exact sources exist at `/archive/openpi/ablation_study/cache_size/save_traj/{libero_10,libero_spatial}/task_N/episode_M.h5`. These contain actual executed actions, including the partial last chunk, wire robot state, full simulator state, seed/init identity, settling count, and terminal count/outcome. Initial states must come from the original B-pool `.init` files under `exp/common/data/db_init/libero/{suite}/`, bound by the recorded original init index and file SHA; matching numeric indices in pruned A-pool files are insufficient. Settling controls are the stock dummy action times the recorded settling count, with this derivation declared in provenance.

Six sampled episodes (library episodes 0, 1, 2 in each suite) match stored `rs` **bit-for-bit** after the original quantile normalization, float32 conversion, and 32-state padding. Maximum absolute rs error is 0.0 in all six; physics/success admission remains pending. Normalization artifact: `/data/openpi/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json`, SHA256 `c0ee3c1a6ca5f1414d31180af3bc333dca1d78cf912f341943191b393fe68707`.

The other six libraries currently lack an exact reset/control source. Library `action.npy` contains normalized proposals, not the issued action tape. Sampled ordinary source H5 files have normalized clean actions/robot state but no executed controls/reset. `dual_20260923` client folders contain evaluation journals/per-step logs from a different A-pool collection and do not establish library-demonstration replay provenance. The GR00T collection recipe did not request `--save-trajectory`. Additional exact records can enable these libraries through `--source-map`; current evidence does not support automatic replay admission.

Backfill CLI defaults to inventory only. With `--replay`, each admitted episode writes client-format records under `OUT/client/`, with `OUT/replay_report.json` containing PASS/REJECTED/UNSUPPORTED. A state-transform adapter is required for model-normalized stores unless `--norm-stats` supplies the pi05 transform. Do not consume rejected captures as library physics evidence. Use a fresh output root for each smoke/full pass; replay currently does not skip prior completed episodes.

## Coordinator validation commands (not executed by S2)

Run these only in the coordinator's authorized environment, with available non-prohibited ports and GPUs as appropriate. S2 did not start any of these services or simulators.

1. Re-run the test command above in an environment permitting ephemeral `127.0.0.1` sockets. Require **25 passed, zero skipped**. Compile/import on the actual Python 3.8 worker environment after deployment:

```bash
cd /scratch/zixuans8/openpi_trace
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 LIBERO_CONFIG_PATH=/home/zixuans8/.libero PYTHONPATH=packages/openpi-client/src:src:. /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.client.preflight --stock-imports
```

2. Build/deploy a new bundle on weilandserver; `R8_RUN_ROOT` must be the coordinator's prepared non-test smoke run. The first command is local packaging; the second is remote deployment/import preflight, not a chain launch:

```bash
cd /home/weiland/projects/openpi
exp/offline_search/debug/ops/build_client_bundle.sh --out /tmp/r8_S2/coordinator_bundle --run-root "$R8_RUN_ROOT" --arms "$R8_ARM"
exp/offline_search/debug/ops/deploy_client.sh /tmp/r8_S2/coordinator_bundle
```

3. On timan107, compare the same action tape with capture off/on at three non-test inits per suite. `MUJOCO_GL=egl` uses the coordinator's renderer environment; omit the CUDA-hiding setting if that renderer requires GPU visibility. Prefer a recorded contact-rich `--action-tape /absolute/tape.npz` when available (numeric `action` with shape N×7, including settling); otherwise the CLI supplies a deterministic test tape. Require all six PASS values in `parity.json` and inspect contact/self-check/capability metadata:

```bash
cd /scratch/zixuans8/openpi_trace
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LIBERO_CONFIG_PATH=/home/zixuans8/.libero MUJOCO_GL=egl PYTHONPATH=packages/openpi-client/src:src:. /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.client.parity --suites libero_10 libero_spatial --inits 50 51 52 --seed 7 --out /scratch/zixuans8/r8_capture_parity
```

4. Stage only the necessary library metadata/arrays (`manifest.json`, `episodes.json`, `rs.npy`, `episode.npy`, `step.npy`), exact saved trajectory H5s, original B-pool `.init` files, and normalization artifact to a coordinator-selected directory, assigned to `R8_BACKFILL_ASSETS`. The following commands run three-episode admission per dense pi05 library; require every `replay_report.json` row to be PASS before a full pass. The full pass uses a new output root and omits `--episodes`:

```bash
cd /scratch/zixuans8/openpi_trace
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LIBERO_CONFIG_PATH=/home/zixuans8/.libero MUJOCO_GL=egl PYTHONPATH=packages/openpi-client/src:src:. /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.backfill.replay --library "$R8_BACKFILL_ASSETS/library/pi05_l10/bpool_cs" --trajectory-root "$R8_BACKFILL_ASSETS/save_traj" --suite libero_10 --init-states-dir "$R8_BACKFILL_ASSETS/db_init/libero/libero_10" --norm-stats "$R8_BACKFILL_ASSETS/norm_stats.json" --episodes 0 1 2 --replay --out /scratch/zixuans8/r8_backfill_l10_smoke
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LIBERO_CONFIG_PATH=/home/zixuans8/.libero MUJOCO_GL=egl PYTHONPATH=packages/openpi-client/src:src:. /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.backfill.replay --library "$R8_BACKFILL_ASSETS/library/pi05_spatial/bpool_cs" --trajectory-root "$R8_BACKFILL_ASSETS/save_traj" --suite libero_spatial --init-states-dir "$R8_BACKFILL_ASSETS/db_init/libero/libero_spatial" --norm-stats "$R8_BACKFILL_ASSETS/norm_stats.json" --episodes 0 1 2 --replay --out /scratch/zixuans8/r8_backfill_spatial_smoke
```

5. Run the selected non-test smoke arm with receiver/network reachability and S1/S3 integration available. The coordinator must assign a free server port outside 23100–23199; this example uses 23310. `OSCL_MANIFEST` should enumerate the chosen non-test task/init pairs. The chain computes expected count from the manifest:

```bash
cd /home/weiland/projects/openpi
PORTS=23310 WPS=1 SERVER_CPUS=34-37,78-81 OPS_CPUS=18-21,62-65 OSDEBUG_RECEIVER_PORT=24080 OSCL_MANIFEST="$R8_SMOKE_MANIFEST" exp/offline_search/debug/ops/chain_debug.sh "$R8_RUN_ROOT" "$R8_ARM"
```

After server drain, manual re-verification on home storage is:

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.debug.transport.collect --run-root "$R8_RUN_ROOT" --arm "$R8_ARM" --verify --expected "$R8_EXPECTED"
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.debug.validate --run-root "$R8_RUN_ROOT" --arms "$R8_ARM" --capture-only --expected-pairs "$R8_EXPECTED" --out "$R8_RUN_ROOT/runs/$R8_ARM/debug/validation.json"
```

Exercise receiver outage/reconnection and collect a spill on a smoke run, verify duplicate replay is idempotent, and measure real client bytes/episode, queue high-water, and receiver drain with the intended worker concurrency. Collection preserves a home archive; default remote cleanup checks individual spill hashes before unlinking. `--keep-remote` retains verified spills for inspection.

## Known gaps and integration boundaries

- Actual NumPy/LIBERO Python 3.8 runtime, real-environment parity, contact self-checks, full backfill simulator replay, production sockets, and multi-lane throughput await coordinator execution. CPU tests and the six source rs matches do not certify these.
- Only the two dense pi05 libraries currently have exact replay candidates; six libraries require additional provenance. No fabricated actions or reset states are substituted.
- General collection code has an environment adapter boundary; the supplied adapter implements MuJoCo/LIBERO. Other environments need an adapter, not task-name branches.
- Snapshot restore is deliberately uncertified, and replay is not resumable by skipping completed output directories.
- Abrupt client termination can lose an unacknowledged RAM queue; such attempts have no completion certificate and cannot pass admission. Bounded RAM overflow/retry timeout spills durably instead of silently dropping.
- No edits are requested in another coder's ownership. Live orchestration depends on S1's flags/echo/schema and S3's `--capture-only` validator. Any server response lacking a matching debug echo is recorded as unverified and fails campaign admission.

## Fix round 1

Read `rounds/r08/REVIEW_1.md` fully. Preserved the coordinator's nested-environment unwrapping and raw MuJoCo struct arguments, public receiver-port support, and explicit non-test `OSDEBUG_APOOL_RECORD`/`OSDEBUG_APOOL_DIR` overrides. Server float64 wire capture and schema were reused read-only. Only S2 implementation/tests and this handback were edited. No GPU, remote service/worker/simulator/chain, tether invocation, tmux, prohibited-port connection, git command, or review-test access was performed. All remote calls in tests were replaced by CPU fakes.

**This section supersedes the original receiver-port, parity-init/pool, bundle, and validation instructions above.** In particular, the coordinator's reachable public port range is 23100–23199; the default receiver is now 23199. S2 did not bind or connect to that range.

Changes:

| Review item | Files and resulting behavior |
|---|---|
| B1 | `client/adapter.py` resolves each contact geom to its free-body ancestor, anchors contacts to any non-free body (including articulated fixtures), and propagates support through free-object contact chains. Isolated free-object cycles remain unsupported. It reports `diagnostic_only`, `passed`, supported/missing body IDs; legacy direct-table absence remains informational. `client/capture.py` isolates self-check exceptions; `transport/validation.py` and `backfill/replay.py` do not reject self-check findings, including legacy `status="error"` labels. |
| M1 | `ops/receiver.py` defaults both API and CLI to 23199 and adds coordinator-only `--probe`: a tether exec runs a TCP connect to `ziyanglin.com:<actual receiver port>` with a 3-second socket timeout and 15-second subprocess timeout. Host override remains supported. `ops/chain_debug.sh` probes after receiver ensure and before server startup, emitting `RECEIVER_UNREACHABLE` and stopping early on failure, or `RECEIVER_REACHABLE` on success. |
| M2 | New `transport/archive.py` stages only exact finalized spill filenames, `stream.osdebugspill` and `stream.osdebugspill.json`, and separately inventories unfinished spill files. `transport/collect.py` records them in `debug/stale_partial_spills.json` with episode key, size, reason, and accepted/unaccepted attempt classification. Partial files in older archives are reported without extraction/replay; a finalized spill lacking its metadata is also reported/skipped. These diagnostics never reject an otherwise complete accepted arm. Accepted receipt/count/hash verification still applies. |
| m3 | `ops/config.py` hashes only `debug/schema.py`, source files under `debug/{server,client,transport,ops}/`, and `closed_loop/{plugin,blind,stage_overrides,serve_*}.py`. Ops shell sources are included. Analysis, reader, augmentation, backfill, and test edits cannot change the capture manifest. Actual capture-code or config changes still reject a frozen-manifest mismatch. |
| m4 | After successful verification, collection removes the exact remote `/tmp/osdebug_<tag>_<arm>.tar`, checking its current SHA against the retained home archive first. This also applies with `--keep-remote` (which retains raw spills). Failed verification never deletes the tar; a changed tar is preserved. |
| m5 client | `client/adapter.py` emits a status/reason per goal subject, keeps resolved entries in `objects`, and separately lists `unresolved_objects`. Top-level status is `partial` when some subjects resolve, `unsupported` when none resolve, and `available` when all resolve. Relational goal subjects are extracted by predicate structure, including right/left relations, instead of restricting to on/in. S5's current consumer accepts `partial` and filters per-object status. |
| m6 | `client/parity.py` defaults to inits 0, 1, 2. Any non-default `--init-states-dir` must explicitly bind every requested suite as `suite=directory`; bare, incomplete, duplicate, or unknown-suite bindings fail before simulator imports. Out-of-range indices produce a clear pool/count argument error. |
| Shared layout | `transport/collect.py` reads every `server_*/decisions*.jsonl`, including old `decisions.jsonl`. It uses the shared newline-boundary JSONL reader: unterminated final lines, including partial UTF-8, are skipped and reported; malformed complete lines still fail. Reports are returned as `skipped_server_lines` and persisted in `debug/server_jsonl_skips.json`. Journal/client/P3-source JSONL consumers also skip/report final fragments. Missing accepted decisions or a missing done lifecycle still fail completeness. No S2 consumer assumes a single meta/stats filename; manifest provenance now references `meta*.json`. |

New tests are `tests/test_client_fix_round1.py` and `tests/test_transport_fix_round1.py`. They cover every fix, including the three coordinator regressions: nested outer/inner environments produce non-empty goals and available predicates; a fake native `mujoco.mj_contactForce` rejects wrapper structs; actual passive observer response capture preserves float64 actions through NPZ and issued-control validation, while deliberately restoring float32 capture causes `issued_action` rejection. B1 tests cover descendant geoms, multi-object chains to an articulated fixture, unanchored cycles, diagnostic exceptions/legacy error labels, complete receipts, and backfill admission. M2 tests include stale `.part`/`.json.tmp` files, old archives, and an orphan finalized prefix, with a complete accepted attempt remaining verifiable.

Executed final test command:

```bash
cd /home/weiland/projects/openpi
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/r8_S2/pytest_fix1_final -q exp/offline_search/debug/tests/test_client_capture.py exp/offline_search/debug/tests/test_client_ops_backfill.py exp/offline_search/debug/tests/test_transport_debug.py exp/offline_search/debug/tests/test_client_fix_round1.py exp/offline_search/debug/tests/test_transport_fix_round1.py
```

Result: **51 passed, 2 skipped in 11.52 s**, exit 0. The two skips remain the sandbox-denied ephemeral loopback socket cases. No real remote probe or cleanup was executed. `bash -n` passes on all four shell entrypoints. The actual local Python 3.8.20 environment at `/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python` passes client preflight: **23 files compile, six imports pass**, including capture/adapter with NumPy; `--stock-imports` was not run. Evidence: `/tmp/r8_S2/python38_fix1.json`.

Read-only real smoke audit: `/home/weiland/trace_runs/os_closed_loop/r08_smoke/runs/r8_pi05_l10_50_A_smoke` passes transport verification and S3 capture-only validation, with **20 accepted attempts, 1,380 decisions, 310 client files, 18,564,942 client bytes, zero skipped lines, and zero capture gaps**. Earlier `__capfail1/2` attempts were not used for admission. Audit used `verify_arm(..., write_report=False)` and `arm.cache_enabled=False` so no files or caches were written under trace_runs. Evidence is `/tmp/r8_S2/smoke_fix1_audit.json`.

Rebuilt generic payload: `/tmp/r8_S2/bundle_fix1/client_bundle.tar`, **27 payload files**, including the new remote archive module; SHA256:

```text
996eced0a7570dcfabcc619c097cf8bc924aee62337d53b54308ca53aeaeb1e6
```

`deploy_client.sh /tmp/r8_S2/bundle_fix1 --dry-run` passes checksum verification and prints deployment/import commands only. The coordinator must redeploy this payload before using the new remote staging/probe workflow; the previous 26-file bundle is superseded. Build a fresh run-specific bundle with the existing `--run-root`/`--arms` flags if YAMLs also need staging. Do not rewrite old frozen MANIFESTs: capture code and hash scope changed, so use a fresh smoke campaign/arm.

Coordinator commands, **not executed by S2**:

```bash
# Local build, then coordinator-authorized deployment/import preflight only.
cd /home/weiland/projects/openpi
exp/offline_search/debug/ops/build_client_bundle.sh --out /tmp/r8_S2/coordinator_fix1_bundle --run-root "$R8_RUN_ROOT" --arms "$R8_ARM"
exp/offline_search/debug/ops/deploy_client.sh /tmp/r8_S2/coordinator_fix1_bundle
```

For a non-test parity run on timan107, bind the actual staged B-val pool separately for both suites and use indices within each pool. The following example selects rows 0–2 of each explicitly bound non-test pool, rather than assuming benchmark indices 50–52 exist. `R8_L10_BVAL_POOL` and `R8_SPATIAL_BVAL_POOL` must be absolute remote paths to the intended non-test pools:

```bash
cd /scratch/zixuans8/openpi_trace
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LIBERO_CONFIG_PATH=/home/zixuans8/.libero MUJOCO_GL=egl PYTHONPATH=packages/openpi-client/src:src:. /scratch/zixuans8/libero_sim/bin/python -m exp.offline_search.debug.client.parity --suites libero_10 libero_spatial --inits 0 1 2 --init-states-dir "libero_10=$R8_L10_BVAL_POOL,libero_spatial=$R8_SPATIAL_BVAL_POOL" --seed 7 --out /scratch/zixuans8/r8_capture_parity_fix1
```

Require six PASS rows, then run a fresh L10 plus Spatial smoke with real contact/stacked starts and receiver throughput. For the coordinator chain, pick distinct free public server and receiver ports (receiver default 23199); pass the explicit non-test manifest and staged pool overrides. The chain now performs the bounded timan107 TCP probe itself before starting servers. Example only:

```bash
cd /home/weiland/projects/openpi
PORTS=23170 WPS=1 SERVER_CPUS=34-37,78-81 OPS_CPUS=18-21,62-65 OSDEBUG_RECEIVER_PORT=23199 OSCL_MANIFEST="$R8_SMOKE_MANIFEST" OSDEBUG_APOOL_RECORD="$R8_REMOTE_BVAL_RECORD" OSDEBUG_APOOL_DIR="$R8_REMOTE_BVAL_DIR" exp/offline_search/debug/ops/chain_debug.sh "$R8_RUN_ROOT" "$R8_ARM"
```

Re-run all five test modules where local ephemeral sockets are permitted; require **53 passed, zero skipped**. Confirm a stale killed-attempt spill is listed as unaccepted without blocking the smoke, that a deliberate receiver reachability failure stops before model startup, and that successful collection retains its home archive while removing the matching remote `/tmp` tar. Real capture off/on simulator parity, Spatial self-check diagnostics, public-network probe, remote cleanup, and production concurrency remain coordinator validation work. No additional edits are requested outside S2 ownership.
