# Central worker scheduler (replaces autoscale_1017)

Owner decisions (2026-10-19), binding:
- One central scheduler for every experiment. It only decides worker counts; episode traffic never passes through it.
- Allocate by what each experiment actually uses, not by declared maxima. All experiments are equal; spare capacity
  goes to whoever still wants more (4 slots / 4 experiments -> 1 each; 100 slots, 3 experiments using 1 each -> the
  4th gets the other 97).
- Two worker classes:
  - shared pool: `a100` (LIBERO only, no MetaWorld env), `timan1`, `timan107`. Any experiment may use them.
  - local: the H100 (`h100`) and the 4090 (`local`) workers serve only a server on that same machine, never an outside
    server, and are added only after the experiment's shared-pool allocation has been stable.
- LIBERO and MT50 (MetaWorld) workers are scheduled together.
- The scheduler runs on the A100 (public 149.165.152.232), not on the 4090, not on racknerd.
- Keep it simple and stateless (k8s style). No compatibility with autoscale_1017: rip it out. A batch whose
  runtime lacks this package must fail at launch, not silently run old code. No extra validation layers or gates.

## Components

```
driver process (one per batch, 4090 or timan107)        scheduler (A100:23500)
  BatchClient  --POST /v1/report every 15 s------------>  in-memory reports (TTL 60 s)
               <--- {targets: {node: n}} --------------   allocate() every 5 s, pure function
  publishes AUTOSCALE_CONTROL.json (unchanged format)
        |  existing channels, unchanged:
        |  (a) worker pull/progress replies carry _autoscale_control
        |  (b) launcher 30 s status call writes the node control file, reads AUTOSCALE_NODE.json back
        v
worker node (per batch per node)
  NodeExecutor: run target workers, memory safety line, GPU slots; writes AUTOSCALE_NODE.json
```

Package: `exp/offline_search/rounds/r17/scheduler/`. stdlib only on the scheduler host (python3, http.server).

| module | content |
|---|---|
| `allocate.py` | `allocate(reports, now) -> {batch: {node: int}}`, pure |
| `server.py` | HTTP service: `POST /v1/report`, `GET /v1/state`, `GET /healthz`; recompute every 5 s |
| `client.py` | `BatchClient`: driver-side hook (conductor `_autoscaler` interface), desire, reporting, control |
| `node.py` | `NodeExecutor`: node-side executor (R17 fleet loop and R15/R16 agent governor) |
| `fleet.py` | R17 fleet loop and GPU `Slots` (moved from autoscale_1017/fleet.py) |
| `worker.py` | worker `Observer` (episode timing, retire, recycle, control relay) |
| `channels.py` | `status_entries`, `status_result`, `control_entry`, `collect_status` (moved from autoscale_1017/driver.py) |
| `bootstrap.py`, `kernel_evidence.py` | moved unchanged |
| `defaults.py` | eligibility, caps, campaign fields |
| `__init__.py` | `VERSION = 'sched-1019b'` |

## Scheduler API

`POST /v1/report` body (one batch):

```json
{
  "batch": "fa_co_f_u_l",
  "worker_type": "libero",
  "server_host": "h100",
  "nodes": {"a100": {"class": "shared", "max": 80}, "timan1": {"class": "shared", "max": 166},
            "timan107": {"class": "shared", "max": 104}, "h100": {"class": "local", "max": 40}},
  "desired": 120,
  "active": {"a100": 40, "timan1": 60, "timan107": 0, "h100": 33},
  "shared_stable_s": 420.0,
  "done": false,
  "node_state": {
    "a100": {"time": 1792430000.0, "ram_available": 31000000000, "ram_floor": 17179869184,
             "workers_ram": 76000000000, "worker_ram": 1900000000,
             "vram_capacity": 38000, "workers_vram": 7400, "worker_vram": 180,
             "cpu_capacity": 30.0, "workers_cpu": 4.1, "worker_cpu": 0.1}
  }
}
```

- `nodes`: the batch's eligible nodes with class and hard cap (physical maximum).
- `active`: live, non-draining workers of this batch per node.
- `shared_stable_s`: seconds the batch's shared-class actual counts have equalled its shared targets.
- `node_state[n]` comes from this batch's NodeExecutor snapshot on node n (any batch's snapshot of a node describes
  the whole node; the scheduler keeps the freshest per node):
  - `ram_available` MemAvailable (cgroup-limited), `ram_floor` the node's floor,
  - `workers_ram` summed RSS of this batch's workers on n, `worker_ram` median RSS of its workers older than 5 min
    (null if none),
  - `vram_capacity` MiB the node's usable renderer VRAM for workers (sum over usable cards of total minus card
    reserve minus server reserve minus non-worker usage), `workers_vram` this batch's worker VRAM there,
    `worker_vram` median per worker (null if none),
  - `cpu_capacity` cores usable by workers (cores minus non-worker load), `workers_cpu`, `worker_cpu` likewise.
  - `gpu_util` percent, mean over 60 s of the busiest card on that node (used for the batch's server host).

Response: `{"targets": {"a100": 44, "timan1": 62, "timan107": 0, "h100": 33}, "time": 1792430001.2}`.
`GET /v1/state` returns all live reports, per-node capacity, the allocation and per-batch shares (for humans).
`GET /healthz` returns `{"ok": true, "version": "sched-1019b"}`.

Stateless: reports live in memory for 60 s; a restarted scheduler is correct again after one report period.

## Allocation (`allocate`)

Per node `n`, resources `r in {ram, vram, cpu}`:
- capacity: `ram = max(ram_available - ram_floor, 0) + sum(workers_ram over batches)`,
  `vram = vram_capacity` (already excludes non-worker use), `cpu = cpu_capacity`.
- per-worker cost of batch `b` on `n`: measured median (`worker_ram`, `worker_vram`, `worker_cpu`) if present, else
  the batch's median on any node, else the type default: LIBERO ram 2.2 GiB, vram 650 MiB, cpu 0.3;
  MetaWorld ram 1.0 GiB, vram 300 MiB, cpu 0.5.

Progressive filling, one worker at a time (deterministic):
1. Batches with `done` or `desired == 0` get 0 everywhere.
2. Shared phase. Repeat: among batches with `alloc_total < desired` that have a shared node where the cost fits the
   remaining capacity and `alloc[b][n] < max`, pick the batch with the smallest dominant share
   (`max_r used_r(b) / pool_r`, pool = sum of capacities of the shared nodes it can use; ties by batch name).
   Give it one worker on: a node where `alloc[b][n] < active[b][n]` (keep running workers where they are), else the
   eligible node with the largest remaining dominant headroom. Stop when nobody can be served.
3. Local phase, per local node: only batches whose `server_host` is that node, whose `shared_stable_s >= 600` or
   that have no shared node with remaining room, and that still have `alloc_total < desired`. Same filling.
4. Result: `targets[b][n] = alloc[b][n]`.

Downscaling falls out of the same loop: a batch over its fair share simply gets fewer than its `active` and its
NodeExecutor retires the excess at episode boundaries. Capacity counts actual workers, so freed slots appear only
after workers really exit.

## Desire (`BatchClient`, HPA style)

`server_timing.infer_ms` includes the server's batching queue, so client-side wait/service ratios cannot show
saturation. Desire uses two signals:
- `gpu_util`: mean GPU utilization of the batch's server host over the last 60 s, reported in that host's
  `node_state` by the batch's local-class NodeExecutor (present for servers on `h100` / `local`; absent for jupyter
  pods and h100-temp).
- accepted rate: accepted episodes per minute over a 5-minute window.

States (re-evaluated every 30 s; `active` = total live workers, `remaining` = pending + in-flight tasks):
- `done` -> 0.
- start: `desired = min(remaining, 16)` (or the last desire this batch directory recorded).
- grow: double (`active + max(8, active)`) until a step first fails to pay off, then `active + max(4, ceil(active / 4))`; wait until live workers reach >= 95% of that (or 2 min),
  then measure the rate for 3 min.
- after a grow step: if `rate_after - rate_before < (1/3) * rate_before * (active_after - active_before) / active_before`
  the step did not help -> hold; else grow again.
- `gpu_util >= 90%` at any time -> hold (skip probing).
- hold: `desired = active`; after 15 min try one probe step of `max(2, ceil(active / 10))` with the same test.
- clamp to `[1, min(remaining, sum of node max)]`.
This small per-batch state lives in the BatchClient (the batch's own controller); the scheduler stays stateless.
yaml hot switching (owner): one batch runs many yaml cohorts and the server hot-swaps bundles per task; workers
are not bound to a yaml and the scheduler allocates per batch, so nothing changes there. A rate comparison whose
two windows differ by more than 0.2 in any cohort's share is not comparable: re-baseline and keep growing.

## BatchClient (driver side)

Implements the conductor hook used by `src/openpi/conductor/driver.py` (`_autoscaler`): attributes `lock`,
methods `tick()`, `result(payload, accepted)`, `ingest(data)`, `pull(payload) -> (retire, control)`.
- `ingest`/`AUTOSCALE_NODES.json`: keeps the freshest NodeExecutor snapshot per node.
- `tick()` (rate-limited to 2 s): every 15 s POSTs the report, applies returned targets (retired nodes from
  `WORKER_NODE_STATE.json` forced to 0), publishes `AUTOSCALE_CONTROL.json` =
  `{campaign, epoch, sequence, time, targets, done}`; `time` strictly increases. On done: targets 0, done true.
- Scheduler unreachable: keep the last targets, log every failure; do not invent targets.
- Loaded via `OPENPI_AUTOSCALE_DRIVER_MODULE=exp.offline_search.rounds.r17.scheduler.client` (R17) and the legacy
  `OPENPI_WORKER_GOVERNOR` / `driver_from_environment` hook (R15/R16).
- Batch description comes from the campaign: `scheduler_url`, `worker_type` (= campaign `domain`), `server_host`,
  `scheduler_nodes` (`{node: {class, max}}`).

## NodeExecutor (node side)

One per batch per node. Each tick (2 s):
- `target` from the control file (fresh within 90 s; otherwise keep the previous target, never grow).
- Spawn while `live < target`, a GPU slot is free (existing `Slots` logic) and
  `ram_available - floor >= worker_ram` (measured median, else type default). One spawn per tick per node.
- `live > target`: mark the newest excess workers retired; they exit at the next episode boundary.
- Safety line: `ram_available < critical` -> retire newest workers to get back above `floor`;
  `ram_available < critical / 2` -> SIGKILL newest workers (episodes are re-dispatched by the driver).
- Writes `AUTOSCALE_NODE.json`: node, time, campaign, active, target, retired, done, the `node_state` fields above.
- Floors: `a100` 16 GiB (critical 6), `timan107` 24 GiB (critical 16), default 20 GiB (critical 10). Local-class VRAM:
  4090 keeps 5 GiB for its server (owner), H100 keeps 4 GiB.

No fairness, no projections, no peaks history, no external-legacy accounting, no spawn ramp beyond one per tick.

## Eligibility (`defaults.py`)

| node | class | worker types | hard cap |
|---|---|---|---|
| `a100` | shared | libero | 80 |
| `timan1` | shared | libero, metaworld | existing CAPS |
| `timan107` | shared | libero, metaworld | existing physical maximum |
| `h100` | local (server on h100 only) | libero | 40 |
| `local` (4090) | local (server on local only) | libero, metaworld | existing physical maximum |

Servers on jupyter pods and h100-temp have no local workers. A batch prepares a fleet on every eligible node (target
starts at 0). Campaign fields written at prepare: `scheduler_url` (default `http://149.165.152.232:23500`),
`scheduler_nodes`, `server_host`, `wire_zlib: true` (lossless image compression, already supported by receivers).
Removed fields: `autoscale_maximum`, `autoscale_node_maxima`, `autoscale_maxima_policy`, `autoscale_node_config`,
`autoscale_external_legacy_nodes`, `autoscale_reactive_nodes`, `autoscale_mature_s`, `autoscale_typical_rss`,
`autoscale_initial`, `autoscale_fairness_campaign`, `demand_targets`, the `ENABLED` switch.

## Launch requirement

Launch fails with a clear error if `GET {scheduler_url}/healthz` does not answer, or if the batch runtime / worker
payload does not contain `scheduler/__init__.py` with the current `VERSION` (re-prepare the batch). That is the only
check.

## Python interfaces between parts (fixed)

- `scheduler.client.from_environment(driver) -> BatchClient | None`: reads `OPENPI_AUTOSCALE_CONFIG` (path of the batch
  `campaign.json`); returns `BatchClient(driver, campaign, Path(path).parent)`. `BatchClient` exposes `lock`
  (RLock), `tick()`, `result(payload, accepted)`, `ingest(data)`, `pull(payload) -> (bool retire, dict control)`,
  exactly what `src/openpi/conductor/driver.py` calls today. Task counts come from `driver.scheduler`.
- `scheduler.node.NodeExecutor(campaign, run)`: same public surface the fleet loop and the legacy agent governor
  use on `LocalGovernor` today: `tick() -> snapshot`, `permit()` (context manager yielding bool, holds the node lock
  through Popen + register), `register(entry)`, `finished(worker_id, normal=False)`, `worker_env(worker_id) -> dict`,
  `entries`, `retired`, `lock()`, attribute `slots` (set by the fleet loop). Campaign keys it reads: `worker_node`,
  `node_workers` (hard cap), `domain`, `run_name`, `worker_max_episodes`, `worker_max_seconds`,
  `worker_vram_mib`, `worker_server_vram_headroom_mib`, `scheduler_node_config` (`{node: {floor, critical}}`, bytes).
- `scheduler.fleet.run_fleet(...)` and `scheduler.fleet.Slots`: same signatures as autoscale_1017/fleet.py.
- `scheduler.worker:Observer`: same constructor/methods as autoscale_1017/worker.py `Observer`; `NodeExecutor.worker_env`
  sets `OPENPI_WORKER_OBSERVER=exp.offline_search.rounds.r17.scheduler.worker:Observer`.
- `scheduler.channels`: `status_entries`, `status_result`, `control_entry`, `collect_status` with today's signatures.
- `scheduler.defaults.apply(campaign, directory=None, *, force=False) -> campaign`: writes `scheduler_url`,
  `scheduler_nodes`, `server_host`, `scheduler_node_config`, `wire_zlib`, `execution_worker_nodes`
  (= `{node: max}` of eligible nodes), `worker_nodes`, `workers`, `worker_vram_mib`; removes the old fields.
- `scheduler.defaults.require(campaign, runtime_dir)`: the one launch check (healthz + VERSION in the runtime).

## Ownership (parallel implementation)

| part | files |
|---|---|
| A scheduler | `scheduler/{allocate,server,client}.py`, `scheduler/__init__.py`, `scheduler/deploy_a100.sh` |
| B node | `scheduler/{node,fleet,worker,bootstrap,kernel_evidence,channels}.py` |
| C R17 integration | `scheduler/defaults.py`, `exp/r17_integration/ops/*`, `filter_failed/*`, `filter_failed_knob/{formal_a_queue,launch}.py`, `longseg_1017/*`, `src/openpi/conductor/*` only if unavoidable, packaging (`bundle`/`release`/`driver_release`) |
| D R15/R16 | new `scheduler/legacy.py` (driver hook + agent governor on top of BatchClient/NodeExecutor), `closed_loop/ops/h100/*`, `rounds/r16/conductor*.py`, `filter_failed_knob/{vla_workers,vla_shared,formal_a}.py`, `cache_study_1019/study.py`, `filter_failed/legacy.py`, retire `attach_local.sh` |

Interfaces between parts are exactly the API, the report/snapshot/control formats and the hook signatures above.

## Cutover

Work happens in the worktree `/home/weiland/projects/openpi_sched` (branch `sched-1019`). Running batches use frozen
copies of autoscale_1017 and are not touched. The autoscale_1017 directory stays on disk until the last batch
launched with it has finished (its launcher may still lazy-import it at finalization); a follow-up commit deletes it.
Prepared-but-unlaunched batches are re-prepared with the new code.
