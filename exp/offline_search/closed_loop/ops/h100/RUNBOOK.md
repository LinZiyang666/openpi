# h100 policy servers → timan108 or timan107 LIBERO workers

Run these commands on weilandserver, from `/home/weiland/projects/openpi`.
The existing ops scripts are unchanged. This topology uses an isolated h100
tree at `/data/oscl_h100/openpi`, its existing pi0.5/GR00T interpreters, and a
worker island at `/scratch/zixuans8/openpi_trace`. Transfers to timan108 stage
on `/srv/local/zixuans8/oscl_sb_stage` (md0), because tether refuses `/scratch`.
`WORKER_HOST` defaults to `timan108`; `timan107` selects the separate
`/scratch/zixuans8/openpi_trace_h100` island and `/tmp/oscl_sb3_stage` staging.
The existing timan107 `openpi_trace` and `openpi` trees are never installed into.

## Setup once, or refresh after code changes

Stop the owned local chain before setup or asset sync. Both now hold
`/home/weiland/trace_runs/os_closed_loop/h100_chain.lock` and check existing
`*/state/h100_chain.lock` files, including the lock held by an older controller.
They refuse with `H100_CHAIN_BUSY` before any remote mutation while a chain is
active. A dry `plan` remains available. For the S-B2 patch, use the targeted
deployment below instead of refreshing the entire source snapshot.
This paragraph describes **full** setup/sync. The S-B3 scoped operations and
concurrent chain locks are described in **Second fleet** below; use that
procedure while the current chain A runs, without applying S-B2 deployment again.

```bash
cd /home/weiland/projects/openpi
OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
bash "$OPS/setup_once.sh"
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control render
```

`verify` checks safetensors rollups against `r08_fits/checkpoint_sha.json` and
imports both serving environments. All three checkpoints are currently present
at `/home/exouser/ckpt/{pi05_libero_pytorch,n15_libero_10,n15_libero_spatial}`.
No checkpoint copy is currently required. Optional `PI05_CKPT` / `GROOT_CKPT`
override the serving path; they must point to separately verified checkpoints.

## Sync selected arms per run

```bash
RUN=/home/weiland/trace_runs/os_closed_loop/r08_abl
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
# Or: ARMS=(r8abl_onlynp_p_l10_50 r8abl_onlynp_p_l10_500)
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
SYNC_PORT=23197 bash "$OPS/sync_assets.sh" "$RUN" "${ARMS[@]}"
```

Finish the selected arms' prefits first. Missing inputs stop planning before
copying. `plan` lists every file and total bytes; `h100_sync/plan.json` records
source SHA, installed SHA, size, destination and reasons. `sync` saves
`h100_sync/synced.json`. It rewrites paths in copied configs, manifests and
prefit metadata, and in launch arguments, without changing `arms.json`.
External run fits/calibration paths are supported. CLIP arms also select and
SHA check their online image-tower weights and set `R8_CLIP_WEIGHTS` remotely.

The temporary read-only rsync daemon serves hardlinks to the selected files;
no extra bulk staging copy is made. Source files and `h100_sync` must share a
filesystem. `SYNC_HOST` defaults to `ziyanglin.com`; `SYNC_PORT` must be a free
forwarded port in 23100–23197. Ports 23198/23199 are reserved. Pulls use
`--partial --inplace --checksum`, verify SHA, and refuse a plan that would leave
less than 30 GiB on h100. Transfers are split into ≤8 GiB RPC batches; a larger
single file is an explicit blocker. No automatic eviction occurs.

## Launch the chain

```bash
PORTS=23210,23211,23212,23213 WPS=8 MAX_ATTEMPTS=3 \
  bash "$OPS/chain_h100.sh" "$RUN" "${ARMS[@]}"
```

Use a coordinator-owned terminal/tmux session for the local chain. Each port
gets one single-replica server in h100 session `oscl<port>`. timan108 uses tmux
socket `oscl`, session `oscl_<arm>`. Workers connect directly to
`149.165.153.233:<port>` and cycle over GPUs 0–3 (timan108) or 0–7 (timan107).
`WPS × number of ports` must
be ≤40 for timan108 or ≤64 for timan107; a fleet lock inside `run_arm.sh` prevents concurrent fleets in this
island. Before launch, our UID's `worker_entry` processes whose argv references
the island are polled for up to 120 seconds. Other users and other trees do
not count. Driver stop waits for the fleet lock to become free (up to 180
seconds) and the island workers to exit before purge/count can proceed.

`STAGE1_ONLY`, `NEED_MB`, `OMP`, `GPU_LOCK`, `GROOT_DENOISING_STEPS`, `PORTS`,
`WPS`, `MAX_ATTEMPTS`, `OSCL_EPISODES`, `OSCL_TASKS`, `OSCL_MANIFEST`, per-arm
`full_model`, `server_seed`, `server_env` and client overrides are honored.
Defaults: WPS=8; stage-1 cache server=3000 MiB, full pi0.5=9000 MiB,
full GR00T=8000 MiB; CLIP adds 1000 MiB unless `NEED_MB` is explicit.
`SERVER_CPUS` is unnecessary on h100. `POLL_SECONDS` defaults to 60;
`STATUS_FAILURE_LIMIT` defaults to 10 consecutive failed polls. Failed status
reads are logged as `STATUS_UNKNOWN`; a wholly successful poll resets the
counter. A known dead server still triggers the existing restart path. Status
reads have one 60-second transport attempt; subsequent polls provide their
retry/backoff. Other idempotent tether calls retry up to four times, with
2/4/8-second backoff, for rc 69/70/75/255, transport timeout messages, and
`TimeoutExpired`. Nontransient failures still fail immediately. Uploaded RPC
specs are reused by digest (with a remote SHA check on first use), so steady
polls perform one exec per server plus one for the driver, without pushes.
An exact manifest takes precedence over Cartesian filters. Put smoke runs in
a separate run root so their DONE markers cannot satisfy a real run.

Before any arm launches, the chain reads and validates its plan while holding
the maintenance lock; every requested arm must occur in `synced.json` and
the source `arms.json`. At chain start, the remote node hashes the isolated
`/data/oscl_h100/openpi` tree once. Each `runs/<arm>/h100_launch.json` records
`code_tree.sha256`, root, file count, byte count, and the hash rule. The digest
covers sorted relative-path/file-SHA records for regular nonsymlink files;
`.git`, `.venv`, `__pycache__`, `.pyc`, and `.pyo` are excluded.

The chain resumes journals, purges client-exception outcomes, counts unique
accepted terminal UIDs without errors, emits `EV ` lines into `runs/chain.log`,
and writes the existing state markers. A failed collection emits
`EV ... COLLECT_PENDING arm=...`, leaves `<arm>.ERROR` containing
`COLLECT_PENDING`, omits the arm's DONE marker, and continues to the next arm.
The post-collection completion-count check remains required. If any collection
is pending, the chain leaves `CHAIN.ERROR` and omits `CHAIN.DONE`; the final
`CHAIN_DONE` event includes `collect_pending=...` to describe traversal of the
requested arms. Pulls retry `TimeoutExpired` within the existing eight-attempt
collection budget. Resume the same chain to retry a pending arm's collection;
the existing journal still controls which episode UIDs need work.
Completed launch requests remain idempotent after exit; the
chain assigns a fresh launch ID for intentional retries/restarts.

## Stop/abort and collect

From another terminal, with the same `OPS` and `RUN`:

```bash
ARM=$(cat "$RUN/state/current")
if [ "$ARM" != none ]; then bash "$OPS/abort_h100.sh" "$RUN" "$ARM"; fi
# Explicit recovery after a local chain crash:
bash "$OPS/abort_h100.sh" "$RUN" r8abl_onlynp_p_l10_50
bash "$OPS/collect_h100.sh" "$RUN" r8abl_onlynp_p_l10_50
```

Abort first signals the local chain using its PID/starttime/command receipt,
waits for its cleanup, then stops owned remote launches. It refuses recycled
PIDs, foreign sessions and mismatched launch receipts. It uses no pattern kill.
After the server's graceful stop timeout (120 seconds by default), stop checks
its recorded PID/starttime again and sends SIGKILL only to that owned PID,
then waits up to 20 seconds for exit. A driver/fleet that remains alive still
blocks purge/count and relaunch. Driver-stop RPCs use a 590-second deadline
to cover the process, fleet-lock, and worker-exit waits. Direct h100 wrappers
are `start_server.sh MODEL SUITE PORT YAML
LOGDIR TAG [plugin flags]` and `stop_server.sh LOGDIR TAG [timeout]`; for an
intentional subsequent direct start, set a fresh `OSCL_LAUNCH_ID` **before**
the tether call so duplicate executions share the same ID.

Collection SHA checks archives from **both** nodes and preserves
`runs/<arm>/client/`, `runs/<arm>/server_<port>/`, `summary.json`, all logs,
decisions, rotated logs and sidecars. Summaries use the unchanged existing
collector. To recompute already collected local results:

```bash
"${P[@]}" -m exp.offline_search.closed_loop.ops.collect --no-pull --run-root "$RUN" "${ARMS[@]}"
```

Full-round, full-model, stock and GR00T live execution remains for the
coordinator. The permitted pi0.5 smoke and all four EGL slots have been verified;
see `HANDBACK_SB.md` for commands, outputs, durations and remaining blockers.

## S-B2 targeted deployment by the coordinator

These commands have **not** been executed by the S-B2 task. The currently
running local process must be stopped and restarted to load `control.py`.
Choose the intended boundary between arms; stop the owned controller before
copying helpers, and require driver/fleet exit before copying application-tree
files or `run_arm.sh`. Never overwrite a running driver's script. The lock
guard below refuses an active legacy or patched controller.

Keep `OPS`, `P`, `RUN`, and the intended `ARMS` from the preceding sections:

```bash
SB2_ARM=$(cat "$RUN/state/current")
# No arm argument: stop the owned controller and let its active-launch cleanup finish.
bash "$OPS/abort_h100.sh" "$RUN"

SB2_ARM="$SB2_ARM" "${P[@]}" - "$RUN" <<'PY'
import json
import os
from pathlib import Path
import sys
from exp.offline_search.closed_loop.ops.h100 import control as c

root = Path(sys.argv[1]).resolve()
arm = os.environ["SB2_ARM"]
with c.chain_lock(root):
    helpers = {
        "h100": [(c.HERE / "node.py", c.BASE / "node.py")],
        "timan108": [(c.HERE / "node.py", c.ISLAND / "os_cl/node.py")],
    }
    # These standalone control helpers do not replace an executing driver script.
    for machine, pairs in helpers.items():
        for source, destination in pairs:
            c.push(machine, source, destination)
    # Repeat owned stops with the new helper, which also handles legacy outer-flock launches.
    if arm != "none":
        launch = json.loads((root / "runs" / arm / "h100_launch.json").read_text())
        print(c.rpc("timan108", "stop", launch["driver"], timeout=590))
        for spec in launch["servers"]:
            print(c.rpc("h100", "stop", spec))
    idle = ("import sys; sys.path.insert(0, " + repr(str(c.ISLAND / "os_cl")) + "); "
            "import node; node.wait_fleet(); node.wait_workers(); print('FLEET_IDLE')")
    print(c.remote("timan108", ["python3", "-c", idle], timeout=590))
    relative = c.HERE.relative_to(c.REPO)
    mirrors = {
        "h100": [(c.HERE / name, c.BASE / "openpi" / relative / name)
                 for name in ("control.py", "node.py")],
        "timan108": [(c.HERE / name, c.ISLAND / relative / name)
                     for name in ("control.py", "node.py")]
                    + [(c.HERE / "run_arm.sh", c.ISLAND / "os_cl/run_arm.sh")],
    }
    for machine, pairs in mirrors.items():
        for source, destination in pairs:
            c.push(machine, source, destination)
    for machine in helpers:
        pairs = helpers[machine] + mirrors[machine]
        output = c.remote(machine, ["sha256sum", *(dest for _, dest in pairs)])
        actual = {path: digest for digest, path in (line.split(maxsplit=1) for line in output.splitlines())}
        for source, destination in pairs:
            assert actual[str(destination)] == c.sha(source), destination
        print(f"SB2_SHA_OK node={machine} files={len(pairs)}")
    manifest = c.json_output(c.rpc("h100", "manifest", {"root": str(c.BASE / "openpi")}, timeout=590))
    print("SB2_CODE_MANIFEST " + json.dumps(manifest, sort_keys=True))
PY

# Use the same selected/synced arm list and the coordinator's prior run settings.
PORTS=23210,23211,23212,23213 WPS=8 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  bash "$OPS/chain_h100.sh" "$RUN" "${ARMS[@]}"
```

Expected deployment checks: `FLEET_IDLE`, `SB2_SHA_OK node=h100 files=3`,
`SB2_SHA_OK node=timan108 files=4`, and `SB2_CODE_MANIFEST` with the actual
remote digest. Stop if any command fails. No asset sync is needed for this
patch; unchanged synced arms and DONE markers are retained. Local changes are
`control.py`, `node.py`, `run_arm.sh`, `test_h100.py`, `test_robustness.py`,
`RUNBOOK.md`, `HANDBACK_SB.md`, and the new local evidence logs. Tests/docs do
not need remote copies. The helper/source copies are all inside the existing
owned topology trees; `/home/exouser/openpi` is unaffected.

## Second fleet (S-B3): concurrent chain B on timan107

This procedure supersedes the S-B2 deployment instructions **for adding B**.
Chain A stays running with its old controller, h100 helper, serving source tree,
ports 23210–23214 and timan108 island. Do not run full `setup` or ordinary `sync`
while either chain is active. No h100/timan108 deployment was done by S-B3.

New chains hold shared `h100_chain.lock.gate`, exclusive
`h100_fleet_<WORKER_HOST>.lock` and their own `state/h100_chain.lock`. Full
maintenance takes the gate exclusively, the original M5 global lock, both host
locks and the legacy root locks. Scoped worker setup/concurrent sync shares the
gate but takes its host/root exclusively. Linux `/proc/locks` plus matching
PID/starttime/argv receipts authenticate the current legacy A; unknown holders,
same worker host, same run tag or a busy root refuse with `H100_CHAIN_BUSY`.
The old controller's global lock stays held and untouched. Port/session ownership
checks remain active on h100, so overlapping ports are refused.

`sync --concurrent` reads all active chains' synced plans. It refuses conflicting
overlaps locally, then uses the **new** `pull-new` RPC through
`/data/oscl_h100/node_sb3.py`. That helper SHA-checks every existing target and
never includes an existing target in rsync's file list. It installs only absent,
unprotected paths. Missing active files, differing existing bytes, symlinks and
active-plan conflicts refuse before transfer. Even without an active chain this
mode refuses replacing existing assets. A partial newly transferred file from
an interrupted transfer can therefore require ordinary sync after both chains
stop. The existing 30 GiB reserve and 8 GiB batch/file bounds remain in force.

### Create B for a coordinator-selected arm list

Use a **fresh** destination, never copy A's runs/state/DONE markers. Keep external
fits/libraries as frozen source inputs; the helper copies the selected arm rows
and their YAML/matrix into B, and writes the 500-pair `eval500.json`. It reads A
only. `prepare_run` refuses an existing destination; for resume reuse that root.

```bash
cd /home/weiland/projects/openpi
OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
A=/home/weiland/trace_runs/os_closed_loop/r08_abl
B=/home/weiland/trace_runs/os_closed_loop/r08_abl_t107
# Replace this example with the coordinator's chosen exact arm list.
ARMS=(r8abl_onlynp_p_l10_50)
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.prepare_run "$A" "$B" "${ARMS[@]}"
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$B" "${ARMS[@]}"
```

Expected: `RUN_PREPARED ... arms=<N> pairs=500`, then `PLAN files=... bytes=...`.
Missing prefits are explicit blockers. The input fit paths still point to A or
their original external source; sync creates relocated copies, without fitting
or rewriting A. Full-run launches explicitly use B's `eval500.json` below.

### Deploy one new h100 helper; A does not pause

Install the local `node.py` as **node_sb3.py**, through a SHA-addressed temporary
file and atomic rename. Do not replace `/data/oscl_h100/node.py`, application
files under `/data/oscl_h100/openpi`, or any timan108 files. All old RPCs keep
using the original helper; only `pull-new` selects the new helper. Thus no A
restart or pause is needed, and B's recorded serving-tree digest still describes
the actual unchanged h100 application tree. Other local changes need no h100
copies: `control.py`, `fleet.py`, `prepare_run.py` run on weilandserver; worker
scripts/source go only to timan107's new island. Deploy before concurrent sync.

```bash
WORKER_HOST=timan107 "${P[@]}" - "$B" <<'PY'
from pathlib import Path
import shlex
import sys
from exp.offline_search.closed_loop.ops.h100 import control as c

with c.fleet_lock(Path(sys.argv[1])):
    source = c.HERE / "node.py"
    digest = c.sha(source)
    stage = c.BASE / ("node_sb3_" + digest + ".new")
    target = c.BASE / "node_sb3.py"
    c.push("h100", source, stage)
    qs, qt = shlex.quote(str(stage)), shlex.quote(str(target))
    check_stage = f'test "$(sha256sum {qs} | cut -d " " -f 1)" = {digest}'
    check_target = f'test "$(sha256sum {qt} | cut -d " " -f 1)" = {digest}'
    script = (f"if test -f {qs}; then {check_stage} && mv -T {qs} {qt}; fi && "
              f"{check_target} && echo SB3_H100_SHA_OK sha={digest}")
    print(c.remote("h100", ["bash", "-c", script]))
PY
```

Expected: `SB3_H100_SHA_OK sha=<local node.py SHA256>`. Duplicate tether exec
sees the final file after rename and re-verifies it. This task tested old CLI
payload/output compatibility against the frozen S-B2 helper in local fixtures;
the coordinator's deployment and B live serving are still required checks.

### Worker setup, verification and concurrent sync

The new timan107 island was already installed/verified by S-B3. The setup command
below can refresh it while A runs, provided B is idle; it makes **no h100 or
timan108 calls**. It copies code, client, init pools, relocated subset helper,
run_arm and LIBERO config. It reuses the existing interpreters and read-only
assets, without modifying timan107's old trees or environments. On timan107,
scratch and `/tmp` share the root filesystem (32 GiB free at verification);
the snapshot is about 18 MiB. Do not put bulk server assets there. The real
worker cap is 64 on 8 GPUs; default timan108 cap remains 40 on 4 GPUs.

```bash
WORKER_HOST=timan107 bash "$OPS/setup_once.sh" --worker-only
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control render
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$B" "${ARMS[@]}"
```

Expected: `INSTALL_OK ... dest=/scratch/zixuans8/openpi_trace_h100`,
`APOOL_OK libero_spatial total=500 sha=0eeece46a08b958efe7b7db4e6b13d3269b0433be4e20fbae3c0f352bc3aca9c`,
`APOOL_OK libero_10 total=500 sha=52457a37eb26f9511b708f2e2efb2c175d0a1f8665ba8d57180556693c1ee756`,
`WORKER_IMPORTS_OK`, and `RENDER_OK GPUs=0,1,2,3,4,5,6,7`. The render helper queries
GL through EGL's `eglGetProcAddress`, avoiding the old PyOpenGL query's segfault
on this installation. Sync prints `ASSETS_NEW_OK added=... reused=...`, worker
config `INSTALL_OK`, then `SYNC_OK`; `synced.json.worker_host` is `timan107`.
If 23197 is occupied, choose another free forwarded 23100–23197 port; never
stop an existing listener. Expected refusals include `H100_CHAIN_BUSY`,
`ACTIVE_ASSET_CONFLICT`, `ACTIVE_ASSET_MISSING`, `EXISTING_ASSET_CONFLICT`,
`DISK_BLOCKED`. Stop that substep on a refusal; never bypass the guard.

### Coordinator-only smoke, restore B config, and launch B

S-B3 ran **no policy/episode smoke or real chain**. The following smoke is for
the coordinator after deploying the new helper. It uses another fresh root,
four official pairs and two workers. Finish it before real B; then re-sync B's
worker config, which smoke sync temporarily replaces on the idle B fleet.

```bash
SMOKE=/home/weiland/trace_runs/os_closed_loop/r08_abl_t107_smoke
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.prepare_run "$B" "$SMOKE" "${ARMS[0]}"
"${P[@]}" - "$SMOKE/smoke4.json" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps([[0,0], [0,1], [1,0], [1,1]]) + "\n")
PY
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$SMOKE" "${ARMS[0]}"
WORKER_HOST=timan107 PORTS=23220 WPS=2 MAX_ATTEMPTS=1 POLL_SECONDS=10 \
OSCL_MANIFEST="$SMOKE/smoke4.json" bash "$OPS/chain_h100.sh" "$SMOKE" "${ARMS[0]}"
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$B" "${ARMS[@]}"

# Run in a NEW coordinator-owned terminal/tmux session. Do not touch r8abl_*.
# Ten ports × four workers = 40 workers, round-robin over all eight GPUs.
WORKER_HOST=timan107 PORTS=23220,23221,23222,23223,23224,23225,23226,23227,23228,23229 \
WPS=4 MAX_ATTEMPTS=3 POLL_SECONDS=60 OSCL_MANIFEST="$B/eval500.json" \
bash "$OPS/chain_h100.sh" "$B" "${ARMS[@]}"
```

Smoke should show `expect=4`, `SERVERS_READY`, `complete=4`, `COLLECT_OK`,
`ARM_DONE`, `CHAIN_DONE` and a `manifest_<sha>.DONE`. Success rate is measured,
not asserted. Real B should show `expect=500`, `workers=40`, ports 23220–23229
and driver outputs under `openpi_trace_h100/os_cl/runs/r08_abl_t107`; journals
and summaries remain under B's local `runs/<arm>/`. `h100_launch.json` records
`worker_host=timan107`. MPS already runs on h100 at `/tmp/nvidia-mps`; leave it
and A's ports untouched. Check H100 GPU/RAM capacity for the chosen arm; the
existing preflight can refuse `GPU_TIGHT`. Higher WPS must still total ≤64.

For B recovery, use the same B root and coordinator-selected manifest/settings.
Abort routes by the saved launch host (missing host in old receipts means
timan108); collection uses the same saved host. Explicit environment selection
is still advisable for manual collection before any launch receipt exists:

```bash
WORKER_HOST=timan107 bash "$OPS/abort_h100.sh" "$B"
WORKER_HOST=timan107 bash "$OPS/collect_h100.sh" "$B" "${ARMS[@]}"
```

The suite proves lock concurrency and safety using local mocks/real flock,
not live concurrent episodes. Throughput, scientific cross-hardware equivalence,
60/64-worker load, GR00T/CLIP/full-model B serving and long-run reliability
remain unverified. See the S-B3 evidence in `HANDBACK_SB.md`.
