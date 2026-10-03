# Local RTX 4090 policy serving — coordinator handback

Implemented `SERVER_HOST=h100|local`, default `h100`. The default h100 plan,
generated assets, server specs and legacy RPC behavior remain byte compatible.
The local branch uses `/home/weiland/projects/openpi`, its `.venv`, and
`/home/weiland/trace_runs/offline_search_store`. Workers remain on the selected
timan fleet. `local` endpoints are `ziyanglin.com:<port>`; h100 endpoints remain
`149.165.153.233:<port>`.

## Exact coordinator commands

Wait for the **existing timan107 pilot to finish normally** before worker setup,
sync, or chain launch. Its fleet lock intentionally excludes a second timan107
chain even if the new chain uses another server GPU. Leave the running timan108
`r11_knob_1` chain and its h100 servers alone. These commands do not need h100
setup, asset copies, a helper replacement, or h100 GPU capacity.

This example freezes one existing R11 arm into a fresh root, uses four local
servers and 16 workers per server (64 workers on timan107), and retains the arm's
exact B development manifest and external prefit. Change the explicit arm list
to queue additional arms from the source root. The destination must be fresh.

```bash
cd /home/weiland/projects/openpi
export SERVER_HOST=local WORKER_HOST=timan107
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/tmp
export LOCAL_SERVER_CPUS=22-37,66-81 OMP=1
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL
SOURCE=/home/weiland/trace_runs/os_closed_loop/r11_devknob_50
RUN=/home/weiland/trace_runs/os_closed_loop/r11_local_t107_1
ARMS=(r11_devknob_pi05_l10_50_off)
P=(taskset -c 22-37,66-81 /home/weiland/projects/openpi/.venv/bin/python -B)

"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.prepare_run \
  "$SOURCE" "$RUN" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"

# Automatic contiguous free-block selection; default count is four.
unset PORTS
LOCAL_SERVER_COUNT=4 WPS=16 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
```

Use an owned coordinator terminal/session for the foreground chain. Alternatively,
replace the final command with explicit ports (busy ports fail without signalling
their listeners):

```bash
PORTS=23150,23151,23152,23153 WPS=16 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
```

After this **new** chain has a launch receipt, collection/recovery uses its saved
worker/server topology, regardless of the invoking shell's current selector:

```bash
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
# Recovery for this new root only, if its own chain must be stopped:
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control abort "$RUN" "${ARMS[@]}"
```

`SERVER_HOST=h100` or an unset selector retains the historical h100 behavior and
requires `PORTS` in 23200–23299. Do not reuse either active chain's ports or root.

## Lifecycle, planning and limits

- Local servers use detached `setsid(2)` supervisor/child processes through
  `Popen(start_new_session=True)`, with no tmux sessions. Start/status/stop use an
  exact spec, launch ID, PID, starttime, and argv receipt. Stop sends signals only
  to recorded, authenticated PIDs. Repeated starts/stops are idempotent; stale
  stops cannot stop a later launch.
- Outputs, runtime caches, logs, and process receipts live at
  `<RUN>/runs/<arm>/server_<port>/`. Per-port and GPU launch locks live under
  `exp/offline_search/closed_loop/ops/h100/.local_locks/`; the lock files persist
  to preserve a common lock inode, with no holder left after cleanup.
- Port selection scans `ss -tlnpH` and bind-checks candidates. Automatic selection
  uses 23100–23196, leaving 23197–23199 for the existing sync/export namespace.
  Explicit `PORTS` may use any free port in 23100–23199. TIME_WAIT/bind conflicts
  are skipped automatically or refused explicitly. Launch rechecks the selected
  port under a local lock. The current serving commands use one replica, so there
  are no internal replica ports; the allocator also checks/reserves the full
  public/internal block when given a future replica count.
- Local CPU affinity defaults to 22–37,66–81. `LOCAL_SERVER_CPUS` must be a nonempty
  subset of those CPUs; reserved CPUs 38–43,82–87 are refused. OpenBLAS is always
  single threaded. Local `OMP` defaults to 1; explicit OMP/GPU_LOCK/denoising and
  checkpoint overrides retain their serving meanings.
- Local plans retain local library and fit paths. Generated YAML and any necessary
  local fit metadata live under the root's `h100_sync/generated/`. The local plan
  records `server_transfer_bytes=0`, worker island, public endpoint range and
  local server tree/store. Sync bundles worker configs/matrices and B pool files
  only; it does not start an rsync daemon or contact/copy assets to h100.
- Chains still authenticate legacy holders and share the maintenance gate, while
  owning one fleet lock and one root lock. A local timan107 chain can coexist with
  an h100 timan108 chain. Each server host checks its own GPU. H100 concurrent sync
  excludes local plans from the h100 asset namespace. Full maintenance remains
  excluded while either chain is active. Local collection pulls only worker logs;
  the local server logs are already in the collector's expected directory.
- Chain boot failures, polling failures, arm completion, interruptions, and abort
  all route server cleanup to the saved/selected backend. Direct local lifecycle
  CLI: `python -B -m exp.offline_search.closed_loop.ops.h100.local_server
  start|status|stop <spec.json>` (use the CPU/BLAS prefix above).

Local checkpoint/island mapping:

| Model | Interpreter | Checkpoint |
| --- | --- | --- |
| π0.5 | `/home/weiland/projects/openpi/.venv/bin/python` | `/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch` |
| GR00T | `/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python` | `/data/ckpt/n15_libero_10` or `/data/ckpt/n15_libero_spatial` |

GR00T's Python path includes
`/home/weiland/projects/openpi_ext/third_party/gr00t_n15` and its `examples/Libero`.
`SERVER_HOST=local ... control verify` selects these local checkpoints and both
local import probes; it does not contact h100. The live smoke exercised π0.5;
GR00T command mapping is unit tested but was not booted on the GPU in this task.

Expected memory: full π0.5 defaults to a 9,000 MiB per-server preflight budget;
the smoke's boot footprint was **7,574 MiB**. Four budgets total 36,000 MiB.
Full GR00T retains an 8,000 MiB budget; stage-1-only servers retain 3,000 MiB.
CLIP arms add the historical 1,000 MiB margin. `NEED_MB` remains configurable.
The GPU reports 49,140 MiB total and returned to 48,471 MiB free after smoke.

## Verification outputs

Tests were run on the specified CPUs with single-threaded BLAS, all test/cache/tmp
writes under `r11_localsmoke`. Command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/tmp \
taskset -c 22-37,66-81 .venv/bin/python -m pytest \
  exp/offline_search/closed_loop/ops/h100/test_h100.py \
  exp/offline_search/closed_loop/ops/h100/test_robustness.py \
  exp/offline_search/closed_loop/ops/h100/test_local.py \
  exp/offline_search/closed_loop/test_devset.py -q \
  --basetemp=/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/tests_final3 \
  -o cache_dir=/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/pytest_cache
```

```text
201 passed in 4.00s
```

Evidence: `/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/evidence/tests_final3.log`.
Coverage includes selector/endpoints, port/replica block collisions and bind races,
CPU constraints, local command mapping, detached fake listener lifecycle, PID
recycling and foreign command/receipt refusal, idempotency, local status ownership,
boot failure cleanup, authenticated local chain holders, saved topology for abort
and collection, a local chain under a real held h100 fleet lock, sync in both
directions across the two server namespaces, and default byte identity.

The existing root `r10_recipe_current`, arm `r10_recipe_pi05_l10_current`, was
planned using the pre-change frozen source and updated source. Both generated
assets and specs for ports 23210/23211 were compared byte for byte. All output was
written to smoke evidence, never the source root. Historical pickle first-load
string memoization changes bytes when method modules are lazily imported; the
comparison warms those same method imports before all three cases. SHA checks of
immutable dependencies are reused only while inode/size/mtime/ctime remain equal.
The historical h100 path remapper and fit walker are also asserted literally
unchanged against the frozen source in unit tests.

```text
DEFAULT_BYTE_IDENTITY_OK prechange plan_sha256=19c6aec3c5b1d6f3000997b3d7fcb6635bf68b435866e288ba56c14a0fb6a515 specs_sha256=cfa154250e5b79876ef1e8f90cb4829c1ed1d70c15e2e04d325629993d72b791
DEFAULT_BYTE_IDENTITY_OK default plan_sha256=19c6aec3c5b1d6f3000997b3d7fcb6635bf68b435866e288ba56c14a0fb6a515 specs_sha256=cfa154250e5b79876ef1e8f90cb4829c1ed1d70c15e2e04d325629993d72b791
DEFAULT_BYTE_IDENTITY_OK explicit_h100 plan_sha256=19c6aec3c5b1d6f3000997b3d7fcb6635bf68b435866e288ba56c14a0fb6a515 specs_sha256=cfa154250e5b79876ef1e8f90cb4829c1ed1d70c15e2e04d325629993d72b791
FROZEN_PRECHANGE_AND_BOTH_H100_DEFAULTS_IDENTICAL
```

Reproduce with the saved verifier (CPU/BLAS environment as above):

```bash
PYTHONPATH=/home/weiland/projects/openpi PYTHONHASHSEED=0 "${P[@]}" \
  /home/weiland/trace_runs/os_closed_loop/r11_localsmoke/evidence/verify_default.py
```

Full output: `r11_localsmoke/evidence/default_identity_verified.log`.

The one-shot bounded live smoke used the existing R11 arm without taking any
chain/fleet lock, syncing a worker island, or writing the source root. It replayed
one recorded observation from `pi05_l10_cache` through an inline, GPU-disabled
Python probe on timan107. No LIBERO simulators or episodes were run anywhere by
the smoke. Remote requests used finite connect/read deadlines and no retries.

```bash
SERVER_HOST=local WORKER_HOST=timan107 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/weiland/trace_runs/os_closed_loop/r11_localsmoke/tmp \
taskset -c 22-37,66-81 .venv/bin/python -u \
  -m exp.offline_search.closed_loop.ops.h100.local_smoke
```

Key output from `r11_localsmoke/evidence/live_smoke.log`:

```text
STARTED local supervisor_pid=1255097 port=23100
LOCAL_READY pid=1255706 port=23100 cpus=[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] gpu=NVIDIA GeForce RTX 4090, 49140 MiB, 40897 MiB free_delta_mb=7574
GPU_PROCESSES
1197742, 30 MiB
1255706, 7574 MiB
WORKER_CONNECTED {"host": "ziyanglin.com", "port": 23100, "metadata_keys": ["concurrent", "cuda_available", "gpu_name", "monitor_level", "policy_fingerprint", "stage_devices", "stage_probe_backends"]}
POLICY_ROUNDTRIP_OK {"shape": [10, 7], "finite": true, "queries": 1, "latency_s": 2.403, "actions_sha256": "60772c39e2d3bd9f3d5944eddc7bac754a7c199f00c2eb4056bad536e4cf0f42"}
STOPPED local
LOCAL_CLEANUP_OK {"running": false, "listening": false, "dead": true}
```

An independent post-smoke check found no listener on 23100, neither owned PID
present, and `NVIDIA GeForce RTX 4090, 48471 MiB` free. The pre-existing GPU process
1197742 was left alone. Logs/spec/ownership receipts are retained under
`r11_localsmoke/runs/r11_devknob_pi05_l10_50_off/server_23100/` and
`r11_localsmoke/smoke_spec.json`. The smoke refuses a second launch when its spec
already exists. To inspect its already stopped server:

```bash
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.local_server status \
  /home/weiland/trace_runs/os_closed_loop/r11_localsmoke/smoke_spec.json
```

Coordinator follow-up: no firewall change was needed for the proven public port
23100. Only that port was live tested. For a different explicit block, keep the
public forwarding/firewall range available and inspect worker connectivity if
the fleet cannot connect. No four-server load test or LIBERO evaluation was run;
those belong to the coordinator's subsequent chain.

## Test-isolation incident

An early test incorrectly expected worker setup to be excluded after releasing
its fake timan107 lock. It reached idempotent remote `mkdir -p` and a SHA-checked
push of the unchanged `node.py` helper into timan107's existing worker island.
The bundle function was mocked, so no application snapshot/config installation
or process action followed. The helper was independently verified afterwards:

```text
19ee875a9c0a4efea8971acba20ae2540f20787caaac41d4131a6ad1ac0c7995  exp/offline_search/closed_loop/ops/h100/node.py
19ee875a9c0a4efea8971acba20ae2540f20787caaac41d4131a6ad1ac0c7995  /scratch/zixuans8/openpi_trace_h100/os_cl/node.py
```

The test now takes its fake local fleet lock before asserting exclusion; all new
tests default to rejecting unmocked remote/push operations. No running chain,
server, worker, tmux session, active lock or run root was stopped/modified. This
unintended byte-identical helper push is recorded explicitly because the task
required strict write isolation. No git commands or pattern kills were used.

## Source SHA256 inventory

Changed: `control.py`, `assets.py`. New source/test/fixture files:
`server_host.py`, `local_server.py`, `local_smoke.py`, `test_local.py`,
`fixtures/assets_prelocal.txt`. New documentation: this handback. The self-referential
handback hash is excluded from its own list; generated logs, receipts, caches,
and persistent empty lock files are runtime evidence, not source changes.

Paths below are relative to `/home/weiland/projects/openpi`:

```text
afa7e4164837a5d9e588d3f780bb220aeba379c317fc30c48f154e8feda35723  exp/offline_search/closed_loop/ops/h100/control.py
089efa55a518fcfda747dbb2f84ce49310bbd1d9b380daa8a80fe57f2dfff413  exp/offline_search/closed_loop/ops/h100/assets.py
b45fd2d79c9cc1aff1bad8d915cdb624209334ef911ba13f01522873c73d5362  exp/offline_search/closed_loop/ops/h100/server_host.py
ea849b0431f84ba66de2e05a07d0295b006839ea92cbe5b574ebfcedf78c0075  exp/offline_search/closed_loop/ops/h100/local_server.py
42905d9d5d3b5f6006dedd43253921e157a57dd356d1d188a04d6c4ec384fd42  exp/offline_search/closed_loop/ops/h100/local_smoke.py
898136d7e4753be8d506431d9c5bb8e6545d0ea3469ec5e82c323ed39cadd497  exp/offline_search/closed_loop/ops/h100/test_local.py
fc1bbbc343e480ec0ed8f445be96b2b14640b558d2bbfed8545c09a5705df7de  exp/offline_search/closed_loop/ops/h100/fixtures/assets_prelocal.txt
```

## Port release fix

2026-10-03: fixed the consecutive-arm handoff that failed on port 23103 in
`r11_local_k3`. Local `stop()` now waits for the recorded child and supervisor
to exit and for the port to become bindable before marking the receipt stopped.
An empty exit-time command line no longer ends that wait. One shared deadline
covers termination, SIGKILL escalation and port release, capped at 60 seconds.
Local `start()` waits up to 60 seconds for previous launches recorded under the
same run root; receipt location, PID birth time and command identity are checked.
Foreign, unknown, mixed-owner and recycled-PID listeners remain refused.

Added `test_local_port_release.py`, using fake processes, signals, listeners,
bind checks, launches and time. Tests ran in one pytest process with
`taskset -c 22-37,66-81`, single-threaded BLAS, bytecode and pytest plugins/cache
disabled, and temporary fixtures under `/tmp`. Output:

```text
test_local_port_release.py: 24 passed in 0.21s
selected test_local.py/test_h100.py compatibility checks: 22 passed in 0.24s
```

Compatibility checks included frozen default-plan bytes, default server-spec
bytes, historical h100 helpers, port selection and duplicate-launch behavior.
`control.py`, `node.py`, `assets.py`, `server_host.py` and existing `test_local.py`
retain their SHA256 values above. No running chain processes, ports, locks or
roots were changed; no servers were launched or restarted.

New SHA256 values (supersede the earlier local-server entry):

```text
47551e8b857266f3c7958901f66bc8b33037c751d5846c8fb543b751747a60a1  exp/offline_search/closed_loop/ops/h100/local_server.py
f0ed53b1066fb25e482de6f16679509aadc777ea3f5681f50ded44cf70a862bc  exp/offline_search/closed_loop/ops/h100/test_local_port_release.py
```
