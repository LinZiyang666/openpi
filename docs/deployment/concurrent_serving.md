# Concurrent Serving Guide

How to run, drive, and tune the post-Phase-5 openpi inference server.

This server hosts one base policy (one set of GPU weights) and fans inference
requests out to a `BatchingCoordinator` that batches stage1/stage2/stage3
forwards across active client connections. Multiple cache YAMLs can be loaded
side-by-side as named "bundles"; each client connection binds to one bundle.

For throughput beyond one process's ceiling, `--replicas N` runs N such servers
behind a single public port (§1.3) — the single-process limit is the
GIL-serialized CUDA kernel-launch path, so scale-out is **process-level**. See
§3.3 for the auto-tuned recommended (replicas, workers, max_wait_ms) per GPU.

The design is described in [`logs/concurrent_serving_optimization_plan.log.md`](../../logs/concurrent_serving_optimization_plan.log.md) and
[`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X. This
document is the **operator** view: what to run, what flags exist, and how
existing experiments map onto the new wiring.

---

## 1. The two server modes

### 1.1 `--concurrent` (default)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=<checkpoint dir> \
    --port 8000 \
    --cache_config exp/verdict_factor_judge/config/.../some.yaml
```

* One server process; many client connections share the same GPU weights.
* The `BatchingCoordinator` spins up three worker threads (one per stage)
  and merges concurrent requests into batched forwards. Default
  `max_batch_size=8`, `max_wait_ms=10` (tune via M7 benchmark before
  changing).
* `--cache_config` is loaded once at startup. Additional bundles can be
  injected later via the `__ctrl__: load_cache_config` wire ctrl.
* Backends are **frozen** at server start (hard constraint C2). Any
  attempt to `insert / batch_insert / delete / upsert / load_artifact` at
  runtime raises `BackendFrozenError`. `write_policy` in your YAML is
  auto-overridden to `"never"` for the duration of the server run; a
  warning is logged so you know the override fired.

### 1.2 `--non-concurrent` (baseline / extreme-speed)

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=<checkpoint dir> \
    --port 8000 \
    --non-concurrent \
    --cache_config <yaml>
```

* Single-connection server. Subsequent clients are rejected with
  WebSocket close code 1013.
* No `BatchingCoordinator`. The original `Policy.infer` (or
  `InferenceInterceptor.infer` with `eager=False`) chain runs — this is
  the pre-Phase-5 path bit-identical, including `torch.compile`
  artifacts. Use this mode to measure the upper bound of a single
  client's request latency (hard constraint C1).
* C2 frozen-runtime still applies. Backends are still frozen after load
  and `write_policy` is still overridden to `"never"`. If you need to
  populate or rebuild artifacts, do it offline (`exp/common/factor_postprocess.py`).

### 1.3 `--replicas N` (multi-process scale-out, single public port)

A single concurrent server is capped by the **GIL-serialized CUDA
kernel-launch path** (~12 inf/s on an A100 for pi05_libero), even when the GPU
is far from compute-bound. Two independent server processes on the same GPU
roughly double throughput. `--replicas N` makes this a first-class mode instead
of hand-launching N servers:

**Step 1 — start the server.** The startup `--cache-config` is a lightweight
bootstrap bundle (`always_hit`); the real phase5 bundle is pushed in step 2 via
`load_cache_config` (the runtime is write-frozen, C2, so the eval bundle cannot
be the startup config — it depends on warmup buffers that only the preload
supplies). Use `uv run --no-sync python` so the launch does not re-sync deps:

```bash
OPENPI_MONITOR_LEVEL=basic TORCH_COMPILE_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
BATCHING_MAX_WAIT_MS=100 BATCHING_MAX_BATCH_SIZE=32 \
uv run --no-sync python scripts/serve_policy.py \
    --replicas 3 --port 8000 \
    --cache-config exp/serving_benchmark/config/cache/spatial16_w8_d4_microbench_always_hit.yaml \
    policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=<checkpoint dir>
```

**Step 2 — preload the phase5 bundle** (after the router logs `starting router
on public port`). This pushes phase5 `g5_g6__fh0.3_ws0.3` (the FULL_HIT /
WARM_START / MISS mix the §3.3 numbers were measured under) as the `default`
bundle to **all** replicas through the public port, so subsequent `main.py`
clients (which send no `select_bundle`) land on it automatically:

```bash
PYTHONPATH=. uv run --no-sync python exp/serving_benchmark/preload_phase5.py \
    --host <server-ip> --port 8000 \
    --cell-id phase5_g5_g6__fh0.3_ws0.3 \
    --eval-yaml exp/verdict_factor_judge/config/spatial16/phase5/eval/spatial16_w8_d4_phase5_g5_g6__fh0.3_ws0.3.yaml \
    --phase3-warmup-dir exp/verdict_factor_judge/data/phase3/warmup
```

What it does:

* The launched process becomes a **supervisor**: it spawns `N` child servers
  (ordinary concurrent servers, each with its own GPU weights + interpreter) on
  internal loopback ports `port+1 .. port+N`, and runs an in-process
  connection-sticky router (`openpi.serving.replica_proxy`) on the public
  `--port`. Only the public port is exposed; clients are unaware of the fan-out.
* **Routing is per connection, never per request** (correctness contract): each
  client WebSocket is pinned to one backend for its whole life, preserving that
  backend's per-connection `CacheOrchestrator` / `_state_history` / stage2→stage3
  KV state. The router never parses inference payloads.
* **Control-plane is broadcast**: `load_cache_config`, `preload_normalizer_buffer`,
  `unload_warmup_buffer`, `set_batch_params` go to *all* replicas — so
  `preload_phase5.py` and `dump_mem.py set-batch` work unchanged through the
  public port. **Metrics are aggregated**: `dump_metrics` / `throughput_summary`
  fan out and merge (each record tagged with its `replica` index).
* `--replica-spawn-batch B`: spawn `B` children concurrently, wait for them to
  load, then the next `B`. **Required on memory-constrained hosts** — on jupyter
  (32 GB host-RAM cgroup) three simultaneous model loads OOM; use
  `--replica-spawn-batch 2` so peak load RAM stays bounded.

```bash
# jupyter (H200, 32 GB host RAM): stagger the model loads
... BATCHING_MAX_WAIT_MS=25 BATCHING_MAX_BATCH_SIZE=32 \
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=<dir> --port 8000 --replicas 3 --replica-spawn-batch 2 \
    --cache-config <yaml>
```

GPU memory ≈ `N × ~7.5 GB` (each replica is a full weight copy). On a 40 GB
A100, `N=3` fits comfortably; do **not** raise N past 3 (per-GPU compute
saturates — see §3.3). To drive the public port, see §2.4.

---

## 2. Client usage

### 2.1 The simple case (LIBERO, existing experiments)

Use the existing `openpi_client.WebsocketClientPolicy` exactly as before:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="...", port=8000)
client.on_episode_start(experiment="phase5_libero10", task="task_3", ...)
action = client.infer(obs)
```

Old clients implicitly land on the `"default"` bundle (or the latest
loaded bundle when only one is loaded). The new lazy lifecycle on the
server side picks up the bundle on first `episode_start` and creates
the wrapper stack — invisible to the client.

### 2.2 Multi-bundle workflow

When you want to host several cache YAMLs at once (one server × N
bundles instead of N server processes × 1 bundle):

```python
client = websocket_client_policy.WebsocketClientPolicy(host="...", port=8000)

# Step 1 — load each bundle once. ``bundle_id`` is the slot name you
# will later select against; if you omit it the server falls back to
# yaml_id, then to "default".
client.load_cache_config(
    yaml_path="exp/.../bundle_a.yaml",
    yaml_id="bundle_a", bundle_id="bundle_a",
)
client.load_cache_config(
    yaml_path="exp/.../bundle_b.yaml",
    yaml_id="bundle_b", bundle_id="bundle_b",
)

# Step 2 — every worker connection binds to exactly one bundle.
client.select_bundle("bundle_a")
# ... episodes targeting bundle_a ...

client.select_bundle("bundle_b")   # switch this connection's bundle
# ... episodes targeting bundle_b ...
```

Notes:
* Bundle switching ends the current task (`on_task_end`) and starts a
  new one (`on_task_begin`) on the per-connection wrapper. Do it at
  episode boundaries, not mid-episode.
* The `BackendPool` shares the underlying in-memory backend across
  bundles that point at the same artifact pkl (identical fingerprint =
  `resolved_preload_path + vector_dims + index_type + backend_type`).
  So `K` bundles using the same pkl cost one 76 MB load, not K.

### 2.3 Existing sweep workflows — do I need to change anything?

| Workflow | Action |
|----------|--------|
| `examples/libero/main.py` (default LIBERO) | None. Lazy default bundle. |
| `exp/verdict_factor_judge/run_phase{3,4,5}.py` | None. They `load_cache_config` once per cell; server falls back to using `yaml_id` as the bundle_id, then `episode_start` (no bundle_id) → server binds to the latest bundle automatically. |
| Anything that depended on runtime `batch_insert` to cache | **Behaviour change**: writes are silently dropped (auto-`write_policy: never`). Rebuild artifacts offline. |
| Multi-server topologies (e.g. phase5 6-server) | Either keep starting 6 servers (still works), or migrate to 1 server × 6 bundles using `load_cache_config` + `select_bundle` for substantially less GPU memory. |

### 2.4 Worker count — how many clients to drive a server

LIBERO is a **closed loop**: each worker sends one `infer`, blocks until the
action returns, steps its MuJoCo sim, then sends the next. So a worker is *not*
continuously outstanding — server concurrency ≈ (workers) × (infer fraction of
the loop, ~0.86 here). Throughput vs worker count is concave: too few starve the
GPU; too many pile up in the stage3 queue (backlog, latency tail grows) without
adding throughput.

**One process = one WebSocket connection.** `examples/libero/main.py
--num-workers K` runs K threads in one process, but K is capped at 15 (MuJoCo
EGL context limit per GPU) and the threads are GIL-serialized for the Python
parts. To put **N concurrent connections** on a server, launch **N separate
processes** (`--num-workers 1` each), not one process with `--num-workers N`:

```bash
# Drive a server with N=48 connections from the sim host (one tmux per worker).
# The single public port fans connections across the server's replicas.
for i in $(seq 0 47); do
  GPU=$((i % 8))   # spread MuJoCo render across the sim host's GPUs
  tmux new -s drv_$i -d "python examples/libero/main.py \
    --host <server-ip> --port 8000 \
    --task-suite-name libero_spatial --num-workers 1 \
    --num-trials-per-task 2 --task-ids 0 \
    --cuda-visible-devices $GPU --seed $((42+i))"
done
```

Because a `--replicas N` server exposes one public port and the router
distributes whole connections (sticky, least-connections), the client does
**not** need to know about replicas or shard ports — just open the optimal
number of connections to `--port`.

**Auto-tuned optimal worker counts** (see §3.3) — start here:

| Server | optimal client connections | notes |
|--------|---------------------------:|-------|
| a100 (A100-40GB, 3 replicas)   | **48** | low-backlog operating point (q3≈2.2); pushing to 64 adds ~3 inf/s but builds queue (q3≈4.5) |
| jupyter (H200, 3 replicas)     | **48** | faster GPU → saturates with fewer workers, at higher throughput |
| a100 + jupyter together (fleet)| **~48 each (96 total)** from one sim host | ~2× the single-server total; sim host (timan107, 48 cores) is near its feed ceiling here |

To re-derive these for a different model / GPU / cache mix, use the autotuner in
§3.3 rather than guessing.

---

## 3. Tuning

### 3.1 Batch parameters (`max_batch_size`, `max_wait_ms`)

The coordinator fires a stage batch when it has `max_batch_size` requests OR
`max_wait_ms` has elapsed since the first one. Set them at startup via env vars
(no code edit needed):

```bash
BATCHING_MAX_BATCH_SIZE=32 BATCHING_MAX_WAIT_MS=25 python scripts/serve_policy.py ...
```

Or **hot-switch at runtime** (broadcast to all replicas through the public port —
no restart):

```bash
python exp/serving_benchmark/dump_mem.py \
    --host <server-ip> --port 8000 set-batch --max-batch-size 32 --max-wait-ms 25
```

Tuning intuition for the LIBERO closed loop: batches rarely fill (only a few
requests arrive per window at realistic concurrency), so a long `max_wait_ms` is
mostly a **pure latency tax** that slows the closed loop. Measured sweet spot is
`max_wait_ms=25` on both a100 and jupyter (10 ms is too aggressive — batches
don't form; >50 ms only adds latency). `max_batch_size=32` is a safe ceiling
that is essentially never the binding constraint here. See §3.3 for the
auto-tuned values.

Use the M7 benchmark suite to find the right values for your workload:

```bash
# Mode 0 — direct GPU sweep
python -m exp.serving_benchmark.gpu_microbench \
    --checkpoint <dir> --config-name pi05_libero \
    --batch-sizes 1,2,4,8,16,32 \
    --output exp/serving_benchmark/data/<run_id>/gpu_microbench.csv
python -m exp.serving_benchmark.plot \
    --run-id <run_id> --mode gpu_microbench

# Modes 1-4 — server-driven sweeps
python -m exp.serving_benchmark.sweep \
    --config exp/serving_benchmark/config/batch_window.yaml \
    --run-id <run_id>
```

See [`docs/experiments/serving_benchmark.md`](../experiments/serving_benchmark.md)
for the full runbook.

### 3.2 CPU thread oversubscription

The cache search path (`InMemoryBackend.search`) uses
`torch.nn.functional.cosine_similarity` which releases the GIL and falls
into BLAS. With many concurrent workers it is easy to oversubscribe CPU
cores. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `cpu_count() //
typical_concurrent_workers` before launching the server.

### 3.3 Recommended configuration per server (auto-tuned)

Measured for `pi05_libero` under the phase5 cache mix (FULL_HIT / WARM_START /
MISS), client on the timan107 sim host. Each server runs **3 replicas**
(`--replicas 3`); jupyter adds `--replica-spawn-batch 2`.

| Server | replicas | client workers (N\*) | `max_wait_ms` | throughput | USL N_opt |
|--------|:--------:|:--------------------:|:-------------:|:----------:|:---------:|
| **a100** (A100-40GB, exclusive) | 3 | **48** | **25** | ~24 inf/s | 60 |
| **jupyter** (H200, shared host)  | 3 | **48** | **25** | ~31 inf/s | 47 |
| **fleet** (a100 + jupyter)       | 3 + 3 | **48 + 48** | 25 | **~48-51 inf/s** | — |

(a100 peaks ~27 inf/s at N=64 but with queue build-up; **N=48 is the recommended
operating point** — ~24 inf/s with a clean queue / low tail latency.)

Context:
* Single-process baseline ≈ 12 inf/s → 3-replica ≈ 2.4×, fleet ≈ 4.3×.
* The throughput ceiling is **closed-loop inference latency** (stage3 denoise
  ~1 s/call dominates ~86 % of the client wall-clock), not batching or queueing.
* jupyter's H200 has spare server capacity (USL `alpha≈0`); the limiter above
  ~48 workers is the single sim host's feed rate, not the server.
* a100 is set to **N=48** (saturated, clean queue `q3≈2.2`, low tail latency).
  N=64 squeezes ~3 inf/s more but builds backlog (`q3≈4.5`) — not worth the
  latency hit, so 48 is the recommended operating point.

**Re-tune for a different model / GPU / cache mix** with the automated explorer
(geometric bracket → golden-section refine → Universal Scalability Law fit;
sweeps `max_wait_ms` at the found N\* via runtime hot-switch):

```bash
python exp/serving_benchmark/autotune_workers.py \
    --server-host <server-ip> --server-port 8000 --label a100 \
    --client-host timan107 --knee-gain 0.05 --max-workers 128 \
    --tune-wait --out /tmp/autotune_a100.json
```

It spawns LIBERO workers on `--client-host`, measures a steady-state window per
sample via the server's aggregated `dump_metrics`, and reports the recommended
`N*` (saturated without backlog) plus the best `max_wait_ms`.

---

## 4. Hard constraints (do not violate)

These are enforced in code; understand them before extending the server.

### C1 — non-concurrent mode preserved

The single-connection path (`--non-concurrent`) MUST remain
bit-identical to pre-Phase-5 behaviour. No `BatchingCoordinator`, no
`BundleDispatcher`, no lazy lifecycle. Use it to measure throughput /
latency ceilings without coordinator overhead.

### C2 — runtime write-frozen

`Backend.insert / batch_insert / delete / upsert / load_artifact` all
raise `BackendFrozenError` after the server's startup load completes.
Cache artifact creation MUST happen offline. The auto-override of
`write_policy → "never"` exists because most legacy sweep YAMLs ship
with `write_policy: on_any_miss` and would otherwise crash on the first
MISS episode.

---

## 5. Troubleshooting

### "BackendFrozenError: backend is frozen"

Something is trying to write to the backend at runtime. Common causes:
1. Custom code calling `storage.batch_insert(...)` directly — move that
   into an offline pipeline.
2. A `WritePolicy` that bypasses the auto-override (rare; check that
   `_enforce_runtime_write_policy` actually ran — it warns on override).
3. A second `load_cache_config` on the same fingerprint — that's
   blocked because `load_artifact` is also a mutation. The pool's
   `get_or_load` cache hit returns the existing frozen backend instead;
   no actual second load happens.

### "select_bundle: unknown bundle_id"

The client called `select_bundle("foo")` without a preceding
`load_cache_config(..., bundle_id="foo")`. Either:
1. Issue the `load_cache_config` first.
2. Or use `"default"` (always implicitly populated by the latest
   `load_cache_config`, even when no explicit `bundle_id` was given).

### "select_bundle or episode_start{bundle_id} required before infer"

The connection sent an `infer` request before binding a bundle. The
server's fallback path is supposed to kick in and auto-bind to the
default bundle — if you see this, it means there is no `"default"`
slot AND no `select_bundle` was sent. Send one before `infer`.

### Throughput is below `model.sample_actions` Mode 0 baseline

Likely causes:
1. CPU thread oversubscription — see §3.2.
2. `--non-concurrent` was passed — go back to concurrent mode.
3. Worker request rate is sparse — the coordinator can only batch what
   it sees within `max_wait_ms`. Either push more workers / higher rate,
   or accept the latency floor.
4. CP1 cache search is slow on a large pkl — profile with the M7 driver
   pulling SystemTimer CSV columns `cp1_search`.

---

## 6. Migration checklist (multi-server → 1 server × N bundles)

1. Start one server with `--concurrent` (default).
2. For each historical "server N hosts yaml N" mapping, push a
   `load_cache_config` via the client with `bundle_id=yamlN_stem`.
3. Replace your `host:port` rotation logic with a per-worker
   `select_bundle(bundle_id)` call right after the websocket connects.
4. Stop the extra server processes (you should observe ≈ (K − 1) × 4.6 GB
   freed on the GPU).
5. Re-run a smoke sweep cell. Result should match the multi-server run
   modulo timing variation.

---

## See also

- [`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X — design reference for C1/C2, BackendPool, BundleDispatcher, BatchingCoordinator
- [`docs/experiments/serving_benchmark.md`](../experiments/serving_benchmark.md) — M7 benchmark runbook
- [`docs/deployment/libero.md`](libero.md) — LIBERO-specific deployment notes
- [`docs/deployment/aloha_sim.md`](aloha_sim.md) — ALOHA Sim deployment notes
- [`logs/concurrent_serving_optimization_plan.log.md`](../../logs/concurrent_serving_optimization_plan.log.md) — full implementation plan + G1/G2 review history
