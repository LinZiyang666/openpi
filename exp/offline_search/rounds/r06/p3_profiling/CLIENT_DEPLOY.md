# V2 client deployment handback

Prepared locally; **no deployment, tether invocation, remote access, server,
worker, chain or simulator was executed**. Shared `closed_loop/ops/chain.sh`,
`ops/remote/run_arm.sh`, `closed_loop/plugin.py` and all `src/` files are unchanged.
The coordinator runs the commands below. This supersedes the old v2 instruction
to distribute an unspecified package tree or use the shared launcher.

## Python 3.8 audit and fixes

Python 3.8 was absent from PATH, `/usr/bin`, and the local uv interpreter cache,
but the Conda package cache contains a working **Python 3.8.20** at
`/home/weiland/miniconda3/pkgs/python-3.8.20-he870216_0/bin/python3.8`.
`check_python38.py`, launched by repository Python, invoked that interpreter
with the required CPU affinity/thread limits and CUDA hidden: **65 files
compiled**, four dependency-free client modules imported from the isolated
bundle, the actual stock `cache/types.py` loaded with deferred annotations,
the import hook remained idempotent, and five strict-zip cases passed.
See `results/python38.json` for the exact command and version. This isolated
interpreter lacks the third-party simulator environment; it does not verify
NumPy/LIBERO runtime compatibility. The other CPU checks used repository Python
**3.11.15** and NumPy **1.26.4** under the same resource limits.

`audit_client_compat.py` parsed **65 repository files** with
`ast.parse(feature_version=(3,8))`: seven delivered client/installer Python files
plus the stock import closure. It inventories every import, including lazy
imports, in `results/client_compat.json` and `CLIENT_IMPORTS.md`. It recursively
follows module-level repository imports, includes known worker/driver runtime
entry dependencies, and excludes TYPE_CHECKING branches. Optional experiment
paths inside uncalled functions and third-party package implementations are
listed dependencies, not claims of Python 3.8 execution or island byte equality.

Two stock-source compatibility issues require a **process-local** fix:

- `openpi.cache.types` and `backend_base` evaluate builtin generic annotations
  such as `frozenset[str]` / `list[...]` without postponement. Before any stock
  import, the Python 3.8 client installs a restricted source loader that compiles
  the cache/conductor/LIBERO/driver modules with the postponed-annotations flag.
  No source file or builtins are changed, and no old pyc is reused.
- Driver sharding uses `zip(strict=True)`, unavailable before 3.10. Module-local
  `zip` bindings in sharding/config/orchestrator use a checked backport on older
  Python. Both unequal-length directions raise; ordinary zip still truncates.

`client_compat.py` is only activated by the explicitly selected client entry.
The owned client has no PEP 585/604 annotations, evaluated modern type aliases,
pattern matching, newer pathlib helpers, or newer dataclass arguments.
The NumPy calls are longstanding array/copy/dtype/tobytes/savez_compressed APIs;
RNG `bit_generator` is guarded with `hasattr`, with legacy `get_state` supported.
There are no deprecated `np.float`/`np.int`/`np.bool` aliases. The intended existing
Python 3.8 NumPy environment needs no new dependency installation; its actual
version and behavior remain subject to the remote preflight/smoke.

`run_gtp_v2` now uses `runpy.run_path` on the existing island
`os_cl/run_gtp_subset.py`, after installing its WorkerSpec wrapper. It handles
OSCL_EPISODES, OSCL_MANIFEST, `--manifest PATH`, and `--manifest=PATH`. It never
imports `exp.offline_search.closed_loop.ops.remote` on the client.

## Remote imports and API assumptions

The full source-file/import inventory is in `CLIENT_IMPORTS.md`; operational
assumptions are below. These dependencies already belong to the stock island
and are **not** overwritten by this bundle.

| Import / object | Required contract |
|---|---|
| `examples.libero.episode_runner` | `LiberoEpisodeRunner(..., client_factory=..., run_episode_fn=...)`; `default_client_factory(endpoint)`; identity has task_uid/attempt. Signature checked by remote preflight. |
| `examples.libero.worker_entry` | `main()` uses the runner imported inside main; accepts `--seed`. The wrapper appends the scheduled seed; no global RNG patch. |
| `examples.libero.main` | `_run_episode(env, client, initial_state, task_description, args, max_steps, ...)` returns five items; calls set_init_state, then infer before each action block. Existing transforms/replan=5 are preserved. |
| `exp.gate_threshold_pareto.run_gtp` | Mutable module `WorkerSpec` binding; spec accepts worker_module/env and preserves MUJOCO_EGL_DEVICE_ID; `SweepStrategy._episodes`, `main()`. No main loop is called during preflight. |
| `os_cl/run_gtp_subset.py` | The existing standalone script imports the same run_gtp module and selects exact manifest pairs. It and `os_cl/count.py` remain installed stock dependencies. |
| `openpi.conductor` / agent, worker, driver, task, protocol, scheduler, sharding, journal, health, monitor, strategy | Existing socket/dispatch/journal APIs; WorkerAgent merges spec.env into worker env and can spawn the specified worker_module. The bundle starts none of them during installation. |
| `exp.ablation_study.cache_size.run_size_eval`, `exp.gate_threshold_pareto.emit_gtp_yamls`, `libraries`; `openpi.cache.config` and its import closure | Existing driver matrix/A-pool/config utilities. The compatibility loader/backport handles the identified annotations/strict zip uses. Native cache/model construction is not performed by the P3 client. |
| `openpi_client.websocket_client_policy`, base_policy, msgpack_numpy, image_tools | Existing endpoint, lifecycle and NumPy wire protocol; websocket client uses `websockets.sync.client`, msgpack and `typing_extensions.override`. Versions must already support the stock Python 3.8 client. |
| `libero.libero` benchmark / OffScreenRenderEnv; robosuite; torch for init-state loading/probe | Existing simulator environment. No direct new mujoco import; both mujoco-py style and robosuite's sim object are accessed through env. Probe imports robosuite only when enabled by stock worker. |
| env `get_sim_state`, `set_init_state`, `step`; inner `timestep`, `cur_time`, `sim.data`, `robots[].controller` | Required snapshot/control interfaces; get_sim_state must yield the stock flat numeric state. Numeric controller attributes saved; object fields remain explicitly skipped. |
| sim `data.ctrl`, act/qpos/qvel, body poses, cvel, sensordata, contacts, forces; model entity names | Available numeric fields logged; contact slices and exposed attributes assumed read-only. `sim.step` must be writable for per-physics-substep actuator tracing; otherwise a missing-reason field is recorded. Actual availability is a live-smoke check. |
| inner `goal_state` or `parsed_problem['goal_state']`, `_eval_predicate` | Evaluator assumed read-only. Missing APIs/errors are recorded; no success/intent labels are inferred from contacts. |
| numpy, cv2, h5py, imageio, PIL, tqdm, tyro, PyYAML, torch, websockets, msgpack, typing_extensions | Existing stock third-party environment; not bundled/reinstalled. Remote preflight imports the real stock dependency path and fails on unavailable/incompatible versions. |
| standard library | argparse, pathlib, os, sys, runpy, copy, hashlib, json, random, time, tempfile, inspect, importlib, itertools, builtins, pkgutil; all used APIs exist in Python 3.8. |

The source-loader change does not prove third-party or remote-tree compatibility.
The deployment's last step runs a real interpreter compile/import/API preflight
under `/scratch/zixuans8/libero_sim/bin/python`. A failure stops the deployment
command; do not start the chain until it succeeds. Physical attributes still
need the actual smoke. Exact restoration is still **not certified**.

## Bundle, installer and owned chain

`client_bundle/client_manifest.json` maps every local file to its island path;
`client_bundle/client_bundle.tar` is the prepared minimal payload. Its twelve
installed files are six modules (`run_gtp_v2`, `worker_v2`, `telemetry`, `snapshots`,
`client_compat`, `client_preflight`), five `__init__.py` package markers for
exp/offline_search/rounds/r06/p3_profiling, and `os_cl/run_arm_v2.sh`.
Markers are only created when absent; existing markers are preserved and their
actual remote hashes printed. New markers extend the package namespace.

The tar also carries its manifest and temporary installer, which are not
installed as client modules. A run-specific bundle adds only the selected arm
YAML/matrix files under `os_cl/cfg` (16 more files for the eight-arm smoke).
No server method, fit, library, shared launcher, stock subset script or count
helper is included. Recipe: `build_client_bundle --out NEW_DIR [--run-root RUN]`.

`deploy_client.sh BUNDLE_DIR` verifies/prints local hashes, performs one
`tether push --force` and **one** `tether exec`. `/tmp/p3_stage` is intentionally
a **tar file**, avoiding a separate remote mkdir operation; it is unpacked in a
unique `/tmp/p3_unpack.*` directory. The installer validates every payload hash
and compiles Python files before copying. Each owned target is replaced via a
same-directory temporary file and os.replace. It copies the island's current
`os_cl/run_arm.sh` to `os_cl/run_arm.stock.sh`; a differing existing stock copy
causes rejection rather than replacement. The original is hash-checked before
and after. Payload and both launcher hashes are printed remotely, then the
preflight runs. `--dry-run` prints the prepared commands without calling tether.

Do **not** use shared `sync_remote.sh` for this delivery: it overwrites
`$ISL/run_arm.sh`. Arm config files are already included by our bundle builder.
The live island's stock script, including its actual interpreter selection, is
copied on the island; the local script's differing interpreter line is not
deployed or treated as authoritative.

`chain_p3.sh` is an owned copy of stock chain.sh. The complete unified diff is
`chain_p3.diff`. HERE is the absolute shared ops directory so its normal sibling
helpers still resolve. Only the P3 variant launches `$ISL/run_arm_v2.sh`.
`client_plan.py` resolves seed/stride/probability from explicit env, arm fields,
then schedule client_env (in that precedence); seed absence is an error, snapshot
defaults are 1/1. P3_PHASE selects smoke/pilot/continuation manifest; P3_SCHEDULE
can name another schedule. Known emit_arms drops arbitrary client_env fields,
so **the schedule is delivered separately**, not hidden in client_overrides.

Telemetry goes to `$ISL/p3_client/<run>/<arm>/p3_telemetry`, outside ordinary
client output to avoid collecting it twice. After ordinary collect, the owned
collector hashes/pulls an archive into `<RUN>/runs/<arm>/client_telemetry.tar`,
then extracts into sibling `client_telemetry/`. A collection failure prevents
DONE. On exhausted attempts it also attempts to collect error prefixes. The
collector validates archive paths/types and uses atomic file writes. Normal
DONE markers include the manifest hash; continuation manifests remain disjoint.

## Exact coordinator smoke commands — not executed by P3

Interpretation: **four cohorts per cell**, eight arms / 32 episodes total;
tasks0–1 × inits0–1 in π0.5 l10/50 and GR00T l10/500. Delivered placeholder specs:
`arms_smoke_v2.json`. The smoke intentionally uses K4 at every anchor and window
trigger p=1 to exercise resampling and delayed-call paths. A is p0, P10 p1;
factorial retains its state strata, durations/holds/cooldown. Production pilot
parameters in arms_v2.json are unchanged.

```bash
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_v2_client_smoke
P3PY=(taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)

"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.prepare_smoke --run-root "$RUN"
mkdir -p "$RUN/calibration"
cp "$D/calibration_v2/pi05_l10_50.json" "$D/calibration_v2/groot_l10_500.json" "$RUN/calibration/"
"${P3PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root "$RUN" --spec "$RUN/arms_in.json"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.prefit --run-root "$RUN" --spec "$RUN/arms_in.json"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.build_client_bundle --run-root "$RUN" --out "$RUN/client_bundle"
bash "$D/deploy_client.sh" "$RUN/client_bundle" 2>&1 | tee "$RUN/client_deploy.log"
test "${PIPESTATUS[0]}" -eq 0 || exit 1

mapfile -t ARMS < "$RUN/arm_names.txt"
PORTS=23164 WPS=2 SERVER_CPUS=2-5,46-49 STAGE1_ONLY=0 P3_PHASE=smoke \
  bash "$D/chain_p3.sh" "$RUN" "${ARMS[@]}"

"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 \
  --run-root "$RUN" --arms "${ARMS[@]}" --client-root "$RUN/runs" \
  --require-stage-counts --require-snapshots --out "$RUN/tables_v2_smoke"
```

Use a fresh bundle/table directory on rerun. The CPU array scopes its empty CUDA
mask to preparation/analysis; it does not export that mask to the coordinator's
server launch. No command above was run on port 23164 here. The owned chain
retains stock port/memory admission checks; it is not a second shared chain.
`read_v2` checks anchor coverage, assignment/compliance, exact selected chunks
and tails, measured stage counts, applied control offsets/terminal masks, and
selected snapshot identities. It does not establish exact restored physics or
make unavailable actuator/predicate fields become available; inspect those
missingness fields before accepting the physical data product.

## Server-only pilot fallback

Yes: run v2 servers with the standard client and use this **explicit** mode:

```bash
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 \
  --run-root <RUN> --arms <ARMS...> --server-only --require-stage-counts \
  --out <NEW_SERVER_ONLY_TABLE_DIRECTORY>
```

Accepted journal outcomes, full server decision skeleton, input archives,
shadow/cache pairs, every-decision raw keys, retrieval/guard/calibration,
randomization histories/propensities, durations/holds/delays/caps, private policy
seeds, K4 samples, stage timings/counts and request-based deployment IR remain.
No client telemetry is silently assumed. Rows say collection_mode=server_only,
controls_verified=false and environment_seed_verified=false. Actual-control IR,
completion/truncation masks, physical successors and applied-control counts are
null; no controls.csv is emitted. Server/journal outcomes must still agree.

Lost wishlist components: Q1.1 applied actions/transitions/masks/client clocks;
Q1.3 snapshots/paired-restore inputs; Q1.6 physical events; Q1.9 verified reset/RNG
and varied environment seeds; Q1.11 physical observer/latency checks; Q2.4
environment-seed variance, Q2.6 actual-control denominators, Q2.7 client deadline/
remaining-control detail; Q3.2–3 branch prerequisites, Q3.5 events, Q3.6 applied
stream/chronology, Q3.7 actual-control cost-to-go, Q3.12 reset/seed verification;
G.P6 physical traces, P7 snapshots/RNG, P8 predicates, P9 client timing and P10
actual completed-control/remaining-horizon detail. Other previously d/e limits
remain unchanged.

P3_ENV_SEED is unused by a standard client. Server replicate seeds still vary
policy/randomization, but these are **not** the planned varied-environment-seed
blocks. ICC can be estimated for the observed fixed-client-seed design; label
that estimand and do not pool it silently with the full pilot or use it as an
estimate of environment-seed variation. This fallback cannot yield every a–c
physical field promised for the full pilot.

## Local evidence

- Python 3.8.20: 65 files compiled, seven delivered Python files; four isolated
  imports, actual stock types annotation deferral and five zip cases pass.
  The separate 65-file AST/import audit passes. Remote dependency preflight
  remains unrun.
- `test_client_deploy`: 28 payload files including 16 smoke configs, six isolated
  module imports, five dispatch routes, worker factory/seed forwarding,
  annotation deferral, five zip cases, stock/marker preservation, corrupted
  payload and unsafe archive rejection. PASS; zero remote calls/simulator runs.
- Eight smoke specs emitted to 16 YAMLs and all eight prefits completed under
  `/tmp/p3_client_deploy_smoke` (about 1.3 GiB of fits). No model loaded or server
  started. The deploy dry-run is `/tmp/p3_client_deploy_dryrun.txt`.
- Reader mode fixtures: four episodes / 96 decisions / 48 anchors / 468 synthetic
  controls. Eight legacy corruption checks pass; server-only leaves controls
  unknown; absent measured counters and missing selected snapshots are rejected;
  48 synthetic snapshot joins pass. These are not real-physics evidence.
- V2 unit suite rerun: six staged configurations plus six legacy-after-v2
  cases, 12,000 assignment units, seven controls / 21 substeps / one snapshot.
  PASS. Local telemetry archive collection also passed without any network use.
- Bash syntax and unified diff checks are local only. Detailed JSON evidence is
  under `results/`; deployment publication hashes are in
  `results/install_manifest_client.json`. Historical serving matrix results in
  HANDBACK remain unchanged; no new shared-file regression/install was needed.
