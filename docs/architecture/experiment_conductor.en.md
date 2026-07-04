# Experiment Conductor

> A generic, experiment-agnostic **episode-level orchestration engine** that supports large-scale evaluation across GPUs / machines / servers.
> Mechanism lives in `src/openpi/conductor/`, policy lives in `exp/`.
> Design decisions and review records: [`logs/archive/client_conductor_two_layer_refactor.log.md`](../../logs/archive/client_conductor_two_layer_refactor.log.md).

---

## 1. Motivation

The old client (`examples/libero/main.py` in-process threads + per-driver scripts under `exp/`) had four problems: dispatch granularity was yaml→subtask, so workers idled toward the tail and formed **wait bubbles**; resume / retry / monitoring were ad hoc and crude in each script; the in-process thread model could not span multiple GPUs or machines; and scheduling was coupled with experiment semantics and not reusable.

This framework cleanly separates the scheduling mechanism from experiment semantics and lowers dispatch granularity to the **episode level**.

## 2. Three-Layer Architecture

```
driver (1, central)         src/openpi/conductor/driver.py
  • EpisodeScheduler central episode queue + affinity greedy
  • TaskGraph execution engine (stage lifecycle begin/complete)
  • Journal ledger / resume          • Retry classification + HealthAggregator + Monitor
  • Pool of ctl connections to servers  • Loads one ExperimentStrategy
      ▲ pull EpisodeTask / ▼ report(progress|result)   (worker→driver: TCP, msgpack)
agent (per-host resident)    src/openpi/conductor/agent.py
  • Forks & supervises workers by GPU/EGL slot, restarts on death (lifecycle only, no message forwarding)
worker (process, pinned to 1 GPU)   src/openpi/conductor/worker.py
  • WorkerLoop: pull → EpisodeRunner.run → report  (connects directly to driver pull port)
  • EpisodeRunner interface; LiberoEpisodeRunner reuses main._run_episode
      ▲ existing WebSocket inference protocol (unchanged)
server1 / server2           (M2 multi-bundle + WarmupPool, already supported)
```

> **Implementation simplification (vs plan §3/§5.3)**: workers **connect directly to the driver**'s pull port; the agent only forks+supervises workers on its local host (it does **not** do the worker↔agent↔driver message forwarding from the original plan). When the agent itself crashes, the driver detects connection drops and re-queues that worker's in-flight episodes — no message loss. This is simpler than a two-hop chain and functionally equivalent.

- **Workers are dumb**: they only accept `EpisodeTask`, execute, and report; they hold no scheduling policy and do not interpret `phase`.
- **Server protocol is untouched**: reuses existing `infer` / `__ctrl__` (`load_cache_config` / `select_bundle` / `fetch_dump` / `preload_normalizer_buffer` / `unload_warmup_buffer`).

## 3. Mechanism / Policy Separation

| | Core mechanism (`src/openpi/conductor`) | Policy (`exp/`, `ExperimentStrategy`) |
|---|---|---|
| Responsibility | Scheduling / pull / affinity / journal resume / retry / monitoring / ctl pool | Per-yaml stage sequence, what to do at barriers, episode-task construction |
| Knows about warmup/eval? | **No** — only understands stage dependencies and lifecycle hooks | **Yes** — warmup/fetch/preload/eval are all in policy |

`src/openpi/conductor/` MUST NOT import `exp.*` or LIBERO; experiment semantics are injected via the `ExperimentStrategy` / `EpisodeRunner` interfaces.

## 4. Core Data Structures (`task.py`)

- `EpisodeTask`: dispatch unit. `task_uid` is deterministically derived as `f"{yaml_id}:{phase}:{task_id}:{episode_idx}"` (so resume can idempotently match the ledger).
- `Stage`: a group of episodes + `phase` + owning `server` + optional `produces_calib_id` / `consumes_calib_id`. The atomic unit of warmup retry/resume is the stage.
- `TaskGraph`: `Stage` + `StageDependency` + `CalibrationArtifact`; `validate()` rejects dangling calib references and dependency cycles.
- `StageContext` (`strategy.py`): thread-safe blackboard, carries the warmup→eval calibration-buffer handoff.
- `CalibrationArtifact`: decouples "calibration data" from "the warmup stage that produced it", supporting **sharing** (one producer → many consumers) and **historical sources** (`source=historical_file`). The `cleanup_id` constrains the warmup dump name to `<cleanup_id>__warmup`, so cleanup goes through the existing `unload_warmup_buffer(cleanup_id)` and **does not change the server protocol**.

## 5. Scheduling Algorithm (`scheduler.py`)

- **Affinity (static)**: `assign_servers` assigns each yaml as a whole to one server (all stages of a yaml on the same server, because `WarmupPool` is server-process-level state), balancing by total episode count and co-locating yamls that share calib.
- **Activation (dynamic) + affinity**: per server, the number of concurrently active yamls is bounded — **warmup is loose** (default ≤2, fills barrier gaps), **eval is tight** (default 1, saves VRAM).
- **Never idle**: as long as the server has any ready episode, a pull will return one.
- **Barrier gating**: a downstream stage stays blocked until all upstreams are done AND their `on_stage_complete` has returned.

## 6. Stage Lifecycle and Data Flow (`driver.py` + `strategy.py`)

For each stage, the core guarantees the order: `upstream.on_stage_complete → downstream.on_stage_begin → downstream episodes ready → done → downstream.on_stage_complete`. `begin/complete` run on **separate threads** so they do not block worker pulls (plan §6.4). Typical warmup→eval flow:

1. warmup `on_stage_begin`: `load_cache_config(warmup_yaml)` (the driver has already `unload_warmup_buffer(cleanup_id)`'ed the stale dump beforehand).
2. warmup episodes run → DumpingJudge writes to disk.
3. warmup `on_stage_complete`: `fetch_dump` + aggregate → `ctx.publish(calib_id, buffer)`.
4. eval `on_stage_begin`: `preload_normalizer_buffer` **first**, then `load_cache_config(eval_yaml)` (the config fails fast when WarmupPool is missing).
5. eval `on_stage_complete`: `unload_warmup_buffer`.

**Fan-out**: multiple eval stages consuming the same `calib_id` each preload from the shared `ctx` buffer in their own server's `on_stage_begin` — naturally fanning out to every relevant server.

## 7. Resume (`journal.py`, plan §8)

- **Ledger**: every terminal-state episode is appended as one JSONL line; on restart, replay the ledger — `done` is skipped, `failed` is re-enqueued per retry policy.
- **Warmup atomicity**: warmup only does **stage-level** resume/retry (the `<cleanup_id>__warmup` dump is cleared before re-running) to avoid duplicate appends polluting the buffer; only eval does episode-level resume.
- **Server self-heal** (plan §8.3): the existing protocol has no WarmupPool probe frame, so we take option **(B) unconditional rebuild**. On resume (driver restart), warmup stages re-run from scratch anyway (warmup is not episode-journaled), naturally rebuilding the pool; the driver additionally gates on `_resuming` (journal non-empty) and calls `strategy.on_resume` before each eval stage's `on_stage_begin` to clear the stale server pool. **As a first-order approximation, "journal non-empty == possibly stale" stands in for the §8.3 staleness check** (normal first runs do not trigger this).

## 8. Retry and Health

- **Retry classification** (`is_retriable_error`): network / timeout / crash → retriable; `ConfigValidationError` and similar fatal errors → not retried, the yaml is marked failed.
- **Disconnect requeue**: worker crash → driver connection drop → in-flight episodes on that connection are auto-requeued (eval requeued / warmup stage invalidated). This is the main lost-connection detector in this build.
- **Agent supervision**: a dead worker process is restarted on the same host (`poll()` detects process exit).
- **Episode wall-clock timeout**: `driver.requeue_timed_out` periodically reclaims episodes that have not reported within `episode_timeout_s` since dispatch (default 1800s) — **workers stuck in infer and not exiting are also reclaimed** (plan §9.2; eval requeued, warmup whole stage invalidated).
- **Stale-result fence**: `EpisodeTask` / `EpisodeResult` carry `attempt`; the scheduler bumps the generation of that uid on each dispatch. After timeout-requeue + redispatch, a late result from the old worker (lower attempt) is rejected by `mark_result` and does not pollute the current dispatch. Residual server-side dump leftovers (old worker appending after the rerun) are a known corner case (under the 1800s timeout, the old worker is in practice already dead and its result is fenced); full dump isolation is deferred.
- **Health observation**: `HealthAggregator` summarizes per-worker status / throughput, and `stale_workers` (based on `last_ts` of progress/result) gives a best-effort lost-connection hint. There is no separate heartbeat wire; explicit heartbeats are replaced by connection liveness + episode timeout.

## 9. Monitoring (`monitor.py`)

`Monitor` renders an **aggregated view** (global done/total/SR + running worker count + total throughput) rather than 96 lines of tqdm (headless-friendly); per-worker details are fetched on demand via `health.snapshot()`.

## 10. Deployment Constraints

- **Server endpoint may be a `--replicas` shared public port (router) or independent single-process endpoints**: `replica_proxy` has changed `fetch_dump` to **aggregate** (fan out to all children + splice each replica's warmup-dump slice; see `merge_dump_replies`), so the warmup→eval dump is complete through the router as well. `--replicas N` shared public port is transparent to the conductor (the driver registers one endpoint). Multiple independent `--concurrent` endpoints can still be registered for fine-grained per-server worker allocation.
- **EGL ceiling**: the agent forks per (host, GPU) quota, keeping the prior limit of ≤15 workers per GPU.

## 11. Modules and Tests

| Module | Responsibility |
|--------|----------------|
| `task.py` | Data structures + TaskGraph |
| `protocol.py` | msgpack-over-TCP wire (length-prefix framing + `protocol_version`) |
| `scheduler.py` | Scheduling state machine |
| `journal.py` | Journal resume |
| `strategy.py` | `ExperimentStrategy` ABC + `StageContext` |
| `worker.py` | `EpisodeRunner` ABC + `WorkerLoop` |
| `driver.py` | Engine main loop + pull service + ctl pool + ownership/retry |
| `agent.py` | Worker fork / supervision / restart |
| `health.py` / `monitor.py` | Health aggregation / aggregated rendering |

Tests live under `tests/conductor/` (CI is all-fake, no GPU) plus `tests/exp/test_warmup_eval_strategy.py`. End-to-end real-LIBERO + real-server runs are manual.

## 12. Extension

- **New experiment** → write an `ExperimentStrategy` (`plan` produces a TaskGraph + `on_stage_begin/complete` orchestrate control frames), placed under `exp/`. Example: [`exp/verdict_factor_judge/strategies/warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py).
- **New environment** → implement an `EpisodeRunner` (`run(task, report)`). Example: [`examples/libero/episode_runner.py`](../../examples/libero/episode_runner.py) (reuses `main._run_episode`).
- Scheduling / resume / retry / monitoring infrastructure requires zero rewrite.
