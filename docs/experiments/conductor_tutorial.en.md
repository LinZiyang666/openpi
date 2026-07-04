# Concurrent Serving + Experiment Conductor Tutorial

> End-to-end how-to: **launch a concurrent inference server** + run large-scale evaluations using the new **experiment conductor framework** (`src/openpi/conductor/`). **Focus is on how to write the driver-side independent policy** (`ExperimentStrategy`).
> Design references: [`docs/architecture/experiment_conductor.md`](../architecture/experiment_conductor.md), [`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X, `logs/archive/client_conductor_two_layer_refactor.log.md`, `logs/archive/concurrent_serving_optimization_plan.log.md`.

---

## 0. What it replaces

- **Server side**: the post-Phase-5 server hosts a single base policy and uses `BatchingCoordinator` to batch stage1/2/3 forwards across connections, saturating the GPU; multiple cache YAMLs can be loaded side-by-side as "bundles", each connection bound to one bundle.
- **Client side (new)**: the old `examples/libero/main.py --num-workers N` (in-process multi-threading, single GPU ≤15) + `run_phase.py` hard-coded 7-step orchestration is replaced by the **experiment conductor framework** — you only write one `ExperimentStrategy`, and scheduling/resume/retry/monitoring/affinity are all handled by the generic engine.

```
driver (central)   ← you load one ExperimentStrategy
  └ EpisodeScheduler / Journal / Retry / Monitor / pool of ctl connections to each server
agent (per host)   ← forks & supervises local worker processes (pinned to GPU/EGL slot)
worker (process)   ← pull episode → EpisodeRunner.run → report; connects directly to server for infer
```

**Core mental model**: you **do not write** scheduling/pull/resume/retry/monitoring (generic mechanism); you **only write** (1) an `ExperimentStrategy` (the experiment script) and (2) reuse or implement an `EpisodeRunner` (how to run one episode). `src/openpi/conductor/` **does not depend on** `exp.*` or LIBERO.

---

## 1. Launching a concurrent server

### 1.1 `--concurrent` (default)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=<checkpoint dir> \
    --port 8000 --cache_config <some.yaml>
```

- One server process, multiple client connections sharing the same GPU weights.
- `BatchingCoordinator` spins up three stage worker threads, batching concurrent requests into single forwards. Defaults are `max_batch_size=8`, `max_wait_ms=10` (tuning in §8).
- `--cache_config` is loaded once at startup; subsequent bundles are injected via `__ctrl__: load_cache_config` (the strategy does this in `on_stage_begin`).
- **C2 write-frozen**: the backend is frozen after startup; `insert/delete/load_artifact` etc. raise `BackendFrozenError` at runtime; the YAML's `write_policy` **must be `"never"`**, otherwise startup raises `ConfigValidationError` immediately (fail-fast, no silent fix-up). Cache artifacts are built offline.

### 1.2 `--non-concurrent` (baseline / max speed)

```bash
python scripts/serve_policy.py ... --non-concurrent --cache_config <yaml>
```

- Single-connection server (subsequent connections are rejected with WS 1013). No `BatchingCoordinator` / lazy lifecycle / bundle indirection — preserves the raw single-connection structure for C1; numerically matches the current sdpa model and is **not** equivalent to the historical pre-Phase-5 eager baseline. Used to measure the single-connection latency upper bound. C2 still applies.

### 1.3 Multiple replicas: `--replicas N` shared public port, or multiple independent `--concurrent` endpoints

A single process is bottlenecked by **GIL-serialized CUDA kernel launches** (pi05_libero ~12 inf/s), so scaling is **process-level**. Both options are supported by the conductor:

> **A) `--replicas N` + shared public port (recommended)**: `serve_policy.py --replicas N` spawns N child inference processes behind one public port, scaled horizontally by the `replica_proxy` router. The router is **transparent** to the conductor: infer routes by least-connections sticky, bundle/`preload`/`unload` broadcast to every child, and **`fetch_dump` is aggregate** — it fans out to all children and **splices each replica's warmup dump slice** into a whole-machine dump (each child's DumpingJudge only writes the episodes routed to it; after the router merges via `merge_dump_replies` there are no partials). The driver registers a **single** `ServerEndpoint` (the public port).
>
> **B) Multiple independent `--concurrent` single-process endpoints**: each occupies one port (multi-process on the same GPU / multiple GPUs / multiple hosts), registered with the driver as **multiple independent `ServerEndpoint`s** and scheduled by the driver's worker→server allocation. Suitable when you want fine-grained per-server worker allocation (e.g. server1→48 workers, server2→48 workers).
>
> The two can be mixed (e.g. 2 hosts each `--replicas 3`, registering 2 public endpoints).
>
> **Uneven cross-server workers (e.g. 16/48)**: when the two servers have asymmetric compute/VRAM (e.g. one is being shared with another job), use `run_phase2 --server-workers "16,48"` to bind 16 workers to `servers[0]` and 48 to `servers[1]` (length must match the `--servers` endpoint count and overrides `--workers`). The same ratio is passed to `ConductorDriver` as `server_capacities`; `assign_servers` places yamls by `weight / capacity` (16:48 → ~1:3 of episodes land on each server) so that the two **finish at the same time** and the smaller server is not the bottleneck. Workers are **strictly affinity-bound** (a worker bound to one endpoint only runs yamls assigned to that endpoint), so worker allocation and yaml placement must follow the same ratio — this flag sets both. Empty = uniform round-robin + equal capacity (the old behavior, e.g. the 48/48 even example under B).

GPU VRAM ≈ `process_count × ~7.5 GB` (one copy of weights per process); on a 40 GB A100, ≤3 processes per GPU (more than that, per-GPU compute saturates, see §8.3).

---

## 2. Two interfaces you implement

| Interface | Lives at | Your responsibility |
|-----------|----------|---------------------|
| `ExperimentStrategy` | `exp/...` (policy layer) | The experiment script: `plan()` emits the stage graph + `on_stage_begin/complete/on_resume` orchestrates control frames |
| `EpisodeRunner` | `examples/...` (execution layer) | Run one episode (connect to server, infer, report). LIBERO ships a ready implementation you can reuse |

The order the driver core guarantees (you can rely on it): `upstream.on_stage_complete → downstream.on_stage_begin → downstream episodes ready → all done → downstream.on_stage_complete`. `on_stage_begin/complete` run on **separate threads** (they do not block worker pulls).

---

## 3. Writing `ExperimentStrategy` (core)

Interface (`src/openpi/conductor/strategy.py`):

```python
class ExperimentStrategy(abc.ABC):
    @abc.abstractmethod
    def plan(self, yamls: list[str], server_assignment: dict[str, ServerEndpoint]) -> TaskGraph: ...
    def on_stage_begin(self, stage, ctl, ctx): ...      # ordered setup before stage starts (default no-op)
    def on_stage_complete(self, stage, ctl, ctx): ...   # handoff/teardown after stage fully done
    def on_resume(self, stage, ctl, ctx): ...           # triggered server self-heal
```

- `ctl` is a `WebsocketClientPolicy` to the server owning this stage (provided by the driver's connection pool, **reused**).
- `ctx` is a `StageContext` (thread-safe blackboard) that hands the warmup-produced calibration buffer to the downstream eval stage.

### 3.1 `plan()`: build the stage graph

Emit a `TaskGraph`: a set of `Stage`s (each containing several `EpisodeTask`s) + inter-stage dependencies + `CalibrationArtifact`s.

```python
from openpi.conductor import task as T

def plan(self, yamls, server_assignment):
    g = T.TaskGraph()
    for yaml_id in yamls:
        server = server_assignment[yaml_id]            # driver-computed yaml→server assignment
        g.add_calibration(T.CalibrationArtifact(        # warmup→eval calib (decouples "data" from "the stage that produced it")
            calib_id=yaml_id, source="warmup_stage",
            warmup_stage_id=f"{yaml_id}:warmup", cleanup_id=yaml_id))
        g.add_stage(T.Stage(                            # eval stage (consumes calib)
            stage_id=f"{yaml_id}:eval", yaml_id=yaml_id, phase="eval", server=server,
            episodes=self._episodes(yaml_id, "eval", server),
            consumes_calib_id=yaml_id, setup={"eval_yaml": f"/cfg/{yaml_id}.yaml"}))
        g.add_stage(T.Stage(                            # warmup stage (produces calib)
            stage_id=f"{yaml_id}:warmup", yaml_id=yaml_id, phase="warmup", server=server,
            episodes=self._episodes(yaml_id, "warmup", server),
            produces_calib_id=yaml_id, setup={"warmup_yaml": f"/cfg/{yaml_id}__warmup.yaml"}))
        g.add_dependency(f"{yaml_id}:warmup", f"{yaml_id}:eval")  # warmup→eval barrier
    return g
```

Episode list (`task_uid` MUST be **deterministically derived** so resume can idempotently match the ledger):

```python
def _episodes(self, yaml_id, phase, server):
    return [
        T.EpisodeTask(
            task_uid=T.make_task_uid(yaml_id, phase, task_id, ep),  # deterministic
            yaml_id=yaml_id, phase=phase, experiment=self._task_suite,
            task_id=task_id, episode_idx=ep, orig_init_state_idx=ep,
            server_host=server.host, server_port=server.port, bundle_id=yaml_id)
        for task_id in self._task_ids for ep in range(self._trials[phase])
    ]
```

**Key points**: `phase` is an opaque label (workers do not interpret it); the core only distinguishes **warmup (server side-effects, atomic stage retry)** from **eval (idempotent, episode-level retry)**. The `setup` dict is for your own use (yaml paths etc.). The driver calls `TaskGraph.validate()` at startup (rejects dangling calib / dependency cycles).

### 3.2 `on_stage_begin()`: ordered setup before stage starts

```python
def on_stage_begin(self, stage, ctl, ctx):
    if stage.phase == "warmup":
        ctl.load_cache_config(yaml_content=_read(stage.setup["warmup_yaml"]),
                              yaml_id=f"{stage.yaml_id}__warmup")
    else:  # eval
        buf = ctx.get(stage.consumes_calib_id) or {}
        if buf:
            ctl.preload_normalizer_buffer(stage.yaml_id, buf)   # preload first
        ctl.load_cache_config(yaml_content=_read(stage.setup["eval_yaml"]),
                              yaml_id=stage.yaml_id)            # then load eval
```

> ⚠ **eval MUST call `preload_normalizer_buffer` BEFORE `load_cache_config`**: config loading fails fast on missing WarmupPool.

### 3.3 `on_stage_complete()`: handoff after stage completes

```python
def on_stage_complete(self, stage, ctl, ctx):
    if stage.phase == "warmup":
        content = ctl.fetch_dump(f"{stage.yaml_id}__warmup")   # retrieve the server-side dump
        buf = aggregate_dump(content)                          # aggregate into {factor_key: [floats]}
        ctx.publish(stage.produces_calib_id, buf)              # hand off to downstream eval's on_stage_begin
    else:
        ctl.unload_warmup_buffer(stage.yaml_id)                # clean up WarmupPool + dump
```

**This is what the old `run_phase.py` 7-step orchestration becomes in the new framework** — you no longer write spawn-worker, wait-for-process, switch-yaml, resume-from-failure. Full implementation at [`exp/verdict_factor_judge/strategies/warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py).

### 3.4 `on_resume()`: server self-heal (optional)

On resume (driver restart, journal non-empty), the core calls this before each eval setup so you can clear potentially-stale server-side WarmupPool:

```python
def on_resume(self, stage, ctl, ctx):
    if stage.phase == "eval":
        with contextlib.suppress(Exception):
            ctl.unload_warmup_buffer(stage.yaml_id)  # drop stale pool; on_stage_begin will re-preload
```

### 3.5 cleanup_id / dump naming constraint (must follow)

The existing server `unload_warmup_buffer(id)` accepts only one id and derives `<id>__warmup.jsonl`. So the warmup dump **must** be named `<cleanup_id>__warmup`, where `CalibrationArtifact.cleanup_id` is that id. In a 1:1 setup, `cleanup_id = eval_yaml_id`; in a shared setup, use a unified `cleanup_id`. Everything stays within the existing server protocol (**no server change**).

---

## 4. `EpisodeRunner`: reuse or write your own

LIBERO ships a ready implementation [`examples/libero/episode_runner.py`](../../examples/libero/episode_runner.py) (which reuses the validated `main._run_episode`); most experiments **use it as-is**. It handles: connect to `task.server`, `select_bundle(task.bundle_id)`, run one episode, periodically `report(step, actions_per_s, hit_type)`, return result + per-step `__hit_meta__` rows; the connection is reused across episodes.

Writing a runner for a new environment (interface in `src/openpi/conductor/worker.py`):

```python
class MyEpisodeRunner(EpisodeRunner):
    def run(self, task, report) -> EpisodeResult:
        client = self._ensure_client(task)          # connect task.server + select_bundle
        ...                                          # reset env, infer loop, report(...)
        return EpisodeResult(task.task_uid, success=done, n_steps=n, per_step_rows=rows)
    def close(self): ...
```

> Workers are "dumb": they do not interpret `phase` and hold no scheduling policy. The `attempt` field is auto-passed by the worker (stale-result fence after timeout-requeue); you do not need to handle it.

---

## 5. Launching the experiment: `ConductorDriver` + workers

### 5.1 Single host / in-process (dev / debug)

```python
from openpi.conductor import ConductorDriver, ServerEndpoint
from examples.libero.episode_runner import default_client_factory

driver = ConductorDriver(
    MyStrategy(task_ids=range(10), warmup_trials=2, eval_trials=10, ...),
    yaml_weights={"cell_a": 100, "cell_b": 100},    # per-yaml episode weight (for server balancing)
    servers=[ServerEndpoint("server1.host", 8001), ServerEndpoint("server1.host", 8002)],
    journal_path="run.jsonl",                        # resume ledger
    ctl_factory=default_client_factory,              # driver uses it to open control connections to each server
    episode_timeout_s=1800,                          # episode wall-clock timeout requeue
)
driver.run()   # returns when all stages are done
```

### 5.2 Cross-host / multi-server (production)

- **Per-server worker allocation** (48 → server1, 48 → server2): start a `WorkerAgent` on each client host; following `WorkerSpec(worker_id, server_key, gpu_id)`, fork worker processes locally (pinned to GPU/EGL slot, ≤15 per GPU); workers connect directly to the driver's pull port.
- **Resume**: restart the driver (same `journal_path`) → already-done episodes are skipped automatically, warmup stages re-run as a whole (stage-atomic).
- **Monitoring**: pass `Monitor(scheduler=...)`; `render()` produces an aggregated view (done/total/SR + per-server worker count + throughput).

```python
from openpi.conductor import WorkerAgent, WorkerSpec
specs = [WorkerSpec(f"w{i}", server_key="server1.host:8001", gpu_id=str(i % 8)) for i in range(48)]
WorkerAgent(specs, driver_host="driver.host", driver_port=9000).run()  # on each client host
```

---

## 6. Scheduling Semantics (the contract a strategy author should know)

The engine guarantees for you (you can rely on these when writing the strategy):

- **Never idle + yaml affinity**: a worker takes the next task immediately upon completion; same-server scheduling prefers "smallest active yaml set" (warmup and eval are **both ≤2 in parallel by default**, filling barrier/straggler gaps — the 2nd yaml is only activated when the 1st has exhausted ready episodes and a worker is idle; consecutive same-keybuilder yamls share a backend via BackendPool fingerprint, so the library is not reloaded and VRAM growth is negligible. Set `eval_concurrency=1` to trade min-VRAM for tail idle bubbles). **No wait bubble between subtasks/yamls.**
- **warmup→eval barrier**: an eval episode is not dispatched until its warmup stage is fully done AND your `on_stage_complete` has returned.
- **Warmup atomicity**: warmup failure/timeout → the entire stage is invalidated and rerun (no episode-level rerun, so no duplicate pollution of the server dump); eval failure → single episode requeued.
- **Retry classification**: network / timeout / crash → retriable; `ConfigValidationError` etc. → not retried, the stage is marked FAILED and cascades to downstream.
- **Stale-result fence**: after timeout-requeue + redispatch, a late result from the old worker (low `attempt`) is rejected.
- **Co-location / fan-out**: eval stages consuming the same `calib_id` are preferentially co-located on one server; for cross-server cases, the engine fans the buffer out and preloads it on every relevant server (you only `ctx.publish` once).

---

## 7. Common Patterns

- **Pure eval (no warmup)**: `plan()` does not create the warmup stage, and the eval stage has `consumes_calib_id=None` (see `WarmupEvalStrategy(skip_warmup=True)`).
- **Shared warmup (phase5 G3)**: multiple eval stages' `consumes_calib_id` point to the same `CalibrationArtifact` → warmup runs once, and the buffer is fanned out via `ctx`.
- **Historical warmup (phase5 G5)**: `CalibrationArtifact(source="historical_file", historical_path=...)` — no warmup stage; eval's `on_stage_begin` aggregates from the file and preloads directly.

---

## 8. Server Tuning

### 8.1 Batch parameters (`max_batch_size`, `max_wait_ms`)

The coordinator emits one stage batch when `max_batch_size` is reached or `max_wait_ms` expires. Set at startup via env vars (no code change) or hot-switch at runtime:

```bash
BATCHING_MAX_BATCH_SIZE=32 BATCHING_MAX_WAIT_MS=25 python scripts/serve_policy.py ...
# Hot-switch at runtime (for one endpoint):
python exp/serving_benchmark/dump_mem.py --host <ip> --port 8001 set-batch --max-batch-size 32 --max-wait-ms 25
```

In the LIBERO closed loop, batches rarely fill up (few requests per window), so a long `max_wait_ms` is mostly **pure latency tax**. Measured sweet spot is `max_wait_ms=25` (10 ms is too aggressive — batches never form; >50 ms only adds latency), `max_batch_size=32` is a safe upper bound.

### 8.2 CPU thread oversubscription

The cache search path (`InMemoryBackend.search`'s `cosine_similarity`) releases the GIL into BLAS; many concurrent workers easily oversubscribe. Before startup, set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to `cpu_count() // typical concurrent worker count`.

### 8.3 Recommended per-server configuration (measured by autotune)

`pi05_libero` + phase5 cache mix (FULL_HIT/WARM_START/MISS), **3 independent `--concurrent` endpoints per GPU**:

| Server | Endpoints | Client workers | `max_wait_ms` | Throughput |
|--------|:---------:|:--------------:|:-------------:|:----------:|
| a100 (A100-40GB) | 3 | 48 | 25 | ~24 inf/s |
| jupyter (H200) | 3 | 48 | 25 | ~31 inf/s |
| fleet (a100+jupyter) | 3+3 | 48+48 | 25 | ~48-51 inf/s |

Single-process baseline ≈12 inf/s → 3 endpoints ≈2.4×, fleet ≈4.3×. The throughput ceiling is the **closed-loop inference latency** (stage3 denoise ~1 s/call accounts for ~86% of client wall-clock), not batch/queue. Do not exceed 3 endpoints per GPU (more than that, per-GPU compute saturates).

To retune (different model / GPU / cache mix), use `exp/serving_benchmark/autotune_workers.py` (geometric bracket → golden-section → USL fit) to find the optimal worker count + `max_wait_ms` per endpoint. Full benchmark at [`serving_benchmark.md`](serving_benchmark.md).

---

## 9. Testing your strategy

- **`plan()` pure-logic test** (no GPU/server): construct strategy → `plan(yamls, assignment)` → assert stage/dependency/calib structure + `validate()` passes. See [`tests/exp/test_warmup_eval_strategy.py`](../../tests/exp/test_warmup_eval_strategy.py).
- **End-to-end integration test**: `FakeEpisodeRunner` + fake ctl (no real server) runs a mock experiment, verifying no-gaps / resume / retry / barrier. See `tests/conductor/test_integration.py` + `conftest.py`.

---

## 10. Hard Constraints and Deployment Constraints (do not violate)

- **C1 — non-concurrent raw single-connection structure**: the `--non-concurrent` path has no coordinator/bundle/lazy; numerics match the current sdpa model and are not bit-identical to the historical eager baseline (used to measure the latency upper bound).
- **C2 — runtime write-frozen**: backend frozen after startup, `write_policy` must be `"never"` (otherwise startup fails fast with `ConfigValidationError`); cache artifacts are built offline (`exp/common/factor_postprocess.py`).
- **Server endpoint = `--replicas` shared public port or independent `--concurrent` endpoints** (§1.3): `replica_proxy`'s `fetch_dump` already fans out + splices per-replica slices (`merge_dump_replies`), so warmup→eval dump through the router is complete — both are supported.
- **≤15 workers per GPU** (MuJoCo EGL context cap): `WorkerAgent` forks per (host, GPU) quota.

---

## 11. Troubleshooting

- **`BackendFrozenError: backend is frozen`**: someone wrote to the backend at runtime. Common causes: custom code calling `storage.batch_insert(...)` directly (move it offline); `write_policy` is not `never` and slipped past validation (check `ConfigValidationError` in startup / `load_cache_config` logs); a second `load_cache_config` of the same fingerprint (caught by the pool — it returns the frozen backend, no actual second load).
- **`select_bundle: unknown bundle_id`**: `select_bundle("foo")` was called without a prior `load_cache_config(..., bundle_id="foo")`. Load first, or use `"default"` (implicitly populated by the most recent `load_cache_config`).
- **`select_bundle or episode_start{bundle_id} required before infer`**: no bundle bound before infer and no `"default"` slot. Send `select_bundle` first, or include `bundle_id` in `episode_start`.
- **Throughput below Mode 0 baseline**: CPU thread oversubscription (§8.2); `--non-concurrent` passed by mistake; worker request rate too sparse (the coordinator can only batch what it sees within `max_wait_ms`); slow CP1 search on large pkl (use the M7 driver to inspect the `cp1_search` timing column).

---

## 12. Parameter Reference

> Parameters scattered across source docstrings but not previously surfaced in user docs. Cross-check this table during batch tuning / triage. Batch params are detailed in §8.1; scheduling semantics in §6.

### 12.1 Scheduler `EpisodeScheduler` (passed via `ConductorDriver(scheduler_kwargs={...})`)

| Param | Default | Meaning |
|---|---|---|
| `eval_concurrency` | `2` | Upper bound on the number of **active** eval yamls per server. `2` (default since 2026-05-26; was 1) activates the next yaml early during current yaml tail stragglers so idle workers pick up work immediately, **eliminating barrier bubbles** (measured: util goes from a 0–10% drop at the tail to a steady 98–100%) — consecutive same-keybuilder yamls share a backend (via BackendPool fingerprint), so the library is not reloaded and VRAM grows negligibly. Set `1` to trade min-VRAM (eval long-connection + KV is the main VRAM consumer) for a tail idle bubble. |
| `warmup_concurrency` | `2` | Upper bound on the number of active warmup yamls per server. Warmup episodes are far fewer than workers (e.g. 2 vs 48); relaxing it fills barrier gaps for utilization. |
| `max_episode_retries` | `3` | Retry count for a single eval episode (for retriable errors like network / timeout / crash). |
| `max_warmup_stage_retries` | `3` | Retry count for the warmup stage **as a whole** (warmup is atomic: on failure, first `unload_warmup_buffer` then rerun the whole stage; no episode-level retry). |
| `max_setup_retries` | `3` | Retry count for `on_stage_begin` / `on_stage_complete` hooks; on overflow or fatal (`ConfigValidationError`), the stage is marked FAILED and cascades to downstream. |

### 12.2 `ConductorDriver`

| Param | Default | Meaning |
|---|---|---|
| `episode_timeout_s` | `1800.0` | Episode wall-clock timeout; in-flight episodes from workers stuck in infer are reclaimed (eval requeued / warmup whole stage invalidated). |
| `bind_host` | `"127.0.0.1"` | Host the driver pull server binds to (workers connect here to pull tasks). |
| `bind_port` | `0` | `0` = random port. |
| `scheduler_kwargs` | `None` | Passed through to `EpisodeScheduler` (see §12.1). |
| `colocation` | `None` | `{yaml_id: server_key}` to force placement (co-location). |
| `poll_s` (`run()` arg) | `0.05` | Main loop poll interval (seconds). |

### 12.3 `WorkerAgent` / `WorkerSpec`

| Param | Default | Meaning |
|---|---|---|
| `poll_s` (WorkerAgent) | `1.0` | Worker-supervision poll interval (detect and restart dead workers). |
| `conda_env` (WorkerSpec) | `""` | If set, the worker is launched via `conda run -p <env>` (isolated interpreter, e.g. a LIBERO sim env that has libero+openpi_client); empty = use the driver's own Python. |
| `worker_module` | `examples.libero.worker_entry` | The target module for the worker's `python -m`. |
| `gpu_id` | — | The worker's `CUDA_VISIBLE_DEVICES` (EGL slot pinning). |

### 12.4 `run_phase2` CLI (the weighted_sum experiment entrypoint — used as a driver example)

| Flag | Default | Meaning |
|---|---|---|
| `--yaml-dir` | (required) | Directory of cache yamls to run. |
| `--init-map` | (required) | `libero_*_init_map.json` (held-out leakage guard). |
| `--journal` | (required) | Resume ledger jsonl. |
| `--servers` | (required) | Comma-separated `host:port` endpoints. |
| `--task-ids` | `0-9` | LIBERO task selection. |
| `--eval-trials` | `20` | Held-out trials per task. |
| `--task-suite` | `libero_spatial` | Task suite. |
| `--total-inits` | `50` | Full init count (for held-out computation). |
| `--episode-timeout-s` | `1800` | → driver `episode_timeout_s`. |
| `--workers` | `48` | Local worker process count. |
| `--gpus` | `8` | Number of GPUs to round-robin workers over (EGL slot; ≤15 workers per GPU, §10). |
| `--conda-env` | `""` | → `WorkerSpec.conda_env`. |
| `--bind-host` | `127.0.0.1` | → driver `bind_host`. |

> ⚠ `run_phase2` does not currently expose `eval_concurrency` etc. via `scheduler_kwargs` (uses default 1/2). To tune, pass `scheduler_kwargs={"eval_concurrency": 2}` to `ConductorDriver` in code, or add a CLI flag.

### 12.5 `serve_policy.py` CLI

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8000` | Listening port. |
| `--replicas` | `1` | Number of concurrent replica processes behind a single public port (`>1` launches the `replica_proxy` router with per-connection routing + broadcast bundle/preload + aggregate fetch_dump). |
| `--replica-spawn-batch` | `0` | When `replicas>1`, spawn in batches: spawn this many child processes concurrently per batch, wait for them to load+bind, then spawn the next batch (`0` = spawn all at once; batching prevents large-library simultaneous-load OOM). |
| `--concurrent` / `--non-concurrent` | `True` | Concurrent multi-client + dynamic bundle hot-swap (default); `--non-concurrent` = C1 raw single-connection max-speed baseline (no coordinator/bundle/lazy; current sdpa numerics). |
| `--cache-config` | `None` | Cache yaml loaded at startup. |
| `--record` / `--collect` / `--collect-dir` / `--collect-images` / `--cache` | … | Recording / collection-build flags (collection writes h5: `collection_policy` extracts vision/prompt embeddings stored as float16). |

### 12.6 Server environment variables

| Env | Default | Meaning |
|---|---|---|
| `OPENPI_SERVER_GPU_MEMORY_LOCK` | `1` | `1` = lock the reserved GPU memory block from being returned to the system (defends shared-host eviction / prevents fragmentation); `0` = old release-to-driver behavior. |
| `PYTORCH_CUDA_ALLOC_CONF` | — | Torch CUDA allocator config; commonly `expandable_segments:True` to reduce fragmentation. |
| `BATCHING_MAX_BATCH_SIZE` | `8` | Per-stage batch episode upper bound for the coordinator (§8.1). |
| `BATCHING_MAX_WAIT_MS` | `10` | Batch-formation timeout; rarely fills in the LIBERO closed loop, so long values are pure latency tax — measured sweet spot is `25` (§8.1). |
| `BATCHING_STAGE1/2/3_WORKERS` | `1` | Number of batching worker threads per stage (stage3 can use multi-thread + multi-CUDA-stream concurrent denoise for higher throughput). |
| `OPENPI_DISABLE_STAGE_STREAMS` | (empty) | `=1` disables per-stage CUDA streams (interceptor host-side CP logic between stages forces a sync, so when stream gains are limited it can be turned off). |
| `OPENPI_STAGE3_BUCKET_FIRST` | (empty) | `=1` uses stage3 bucket-first loop; default is generic pull-then-group (measured a100 throughput higher). |
| `OPENPI_MONITOR_LEVEL` | (empty) | Monitor instrumentation level (see `serving/monitor.py`). |
| `OPENPI_MONITOR_AUTOFLUSH_DIR` | `""` | Monitor metrics auto-flush directory. |
| `TORCHINDUCTOR_CACHE_DIR` | — | torch.compile inductor compilation cache directory. |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | — | BLAS thread count; under many concurrent workers, prevent CPU oversubscription by setting `cpu_count() // concurrent worker count` (§8.2). |

---

## See also

- [`docs/architecture/experiment_conductor.md`](../architecture/experiment_conductor.md) — conductor framework architecture
- [`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X — C1/C2, BackendPool, BundleDispatcher, BatchingCoordinator design
- [`docs/experiments/serving_benchmark.md`](serving_benchmark.md) — M7 throughput benchmark runbook
- Reference implementations: strategy [`warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py), runner [`episode_runner.py`](../../examples/libero/episode_runner.py), worker entrypoint [`worker_entry.py`](../../examples/libero/worker_entry.py)
