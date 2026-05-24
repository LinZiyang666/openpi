# Concurrent Serving Guide

How to run, drive, and tune the post-Phase-5 openpi inference server.

This server hosts one base policy (one set of GPU weights) and fans inference
requests out to a `BatchingCoordinator` that batches stage1/stage2/stage3
forwards across active client connections. Multiple cache YAMLs can be loaded
side-by-side as named "bundles"; each client connection binds to one bundle.

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

---

## 3. Tuning

### 3.1 Batch parameters

Default `max_batch_size=8`, `max_wait_ms=10`. To change them, edit
`scripts/serve_policy.py` where `BatchingCoordinator(...)` is constructed,
or expose a CLI flag (one-line addition).

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
    --config exp/serving_benchmark/configs/batch_window.yaml \
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
