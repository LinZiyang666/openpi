# Serving Throughput Problem — Facts for External Review

> This document states the problem and measured data only. It deliberately
> contains **no hypotheses about root cause** so as not to bias the reviewer.

## System under test

- **Model**: pi0.5 (`pi05_libero`): PaliGemma 2B (SigLIP vision + Gemma-2B LLM)
  + Gemma-300M action expert. bf16. `TORCH_COMPILE_DISABLE=1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Server**: `scripts/serve_policy.py` in `--concurrent` mode. One Python
  process. A WebSocket server (`websocket_policy_server.py`) accepts N
  simultaneous client connections; each connection's `infer()` runs via
  `asyncio.to_thread`. Inference is a 3-stage pipeline:
  - stage1 = SigLIP vision encode + prefix embed
  - stage2 = Gemma-2B LLM backbone forward (fills KV cache)
  - stage3 = Gemma-300M action expert flow-matching denoise (10 Euler steps)
- **BatchingCoordinator** (`batching_coordinator.py`): 3 background daemon
  threads, one per stage. Each connection submits a per-request payload to
  the stage queue and blocks on a reply event. Each stage worker pulls a
  dynamic batch (up to `max_batch_size`, or until `max_wait_ms` elapses),
  runs one batched forward, distributes results. Between stages, per-connection
  cache logic (CP1/CP2/CP3 gate/judge/search) runs on the connection's
  `to_thread` worker.
- **Cache config**: phase5 composite-judge yaml
  (`spatial16_w8_d4_phase5_g5_g6__fh0.3_ws0.3`), warmup buffer preloaded.
  Produces a mix of FULL_HIT (short-circuit), WARM_START (`run_stage3_from`,
  fewer denoise steps), MISS (`run_stage3`, 10 steps).

## Hardware

| Role | Host | Device | Notes |
|---|---|---|---|
| Inference server A | a100 | 1× A100-SXM4-40GB, 32 vCPU, 115 GB RAM | public 149.165.151.106:8000 |
| Inference server B | jupyter-ziyang10 | 1× H200-140GB, cgroup 10C/32G | via tether expose weiland.top:14000 |
| Sim clients | timan107 | 8× GTX 1080 (8 GB), 48 cores, 220 GB RAM | only host with LIBERO env |

- Client = N independent `examples/libero/main.py` processes, each
  `--num-workers 1` (1 WebSocket connection each), `--num-trials-per-task 2`,
  libero_spatial task 0. Each process pinned to GPU `N % 8` (MuJoCo EGL render).
- Each `policy.infer()` returns an action chunk of `action_horizon` actions;
  the client steps the sim through the chunk before the next inference.

## Goal

Maximize server throughput (inferences/sec) and GPU utilization on a single
server process, using the real phase5 workload. Constraints: do not break
inference correctness, do not break module decoupling, do not run multiple
server processes (single server must use the device fully); only the client
side may add workers.

## Metric definitions

- **throughput** = stage1 batched inferences per second = (Σ stage1 batch
  sizes) / (active window from first to last batch event). Every
  `policy.infer` call passes through stage1 exactly once.
- **GPU util** = mean of `nvmlDeviceGetUtilizationRates().gpu` sampled at 1 Hz
  over the run (NOT peak).
- **forward_ms** = wall-clock inside the coordinator worker for
  `model.run_stageN(...)` + result split + reply distribution (NOT a pure
  CUDA-event GPU timer; measured with `time.monotonic()`).
- **assemble_ms** = wall-clock for `stage_io` batch stacking before the forward.
- **wait_ms** = mean per-request time from enqueue to dispatch.
- LIBERO **success rate** = task success reported by the sim client (correctness check).

## Measured data

### a100 — N sweep at fixed params (batch=8, wait=10ms) unless noted

| N | (max_batch, max_wait_ms) | throughput inf/s | GPU util avg | queue depth peak |
|---|---|---|---|---|
| 4 | (8,10) | 2.16 | ~17% | 0 |
| 8 | (8,10) | 3.5–4.05 | ~10% | 5 |
| 16 | (8,10) | 3.25 | 15.4% | 11 |
| 16 | (8,50) | 7.05 | 27.0% | 11 |
| 16 | (8,200) | 6.69 | 21.3% | 11 |
| 16 | (16,50) | 6.82 | 19.4% | 10 |
| 24 | (8,50) | 7.40 | 22.5% | 19 |
| 48 | (8,50) | 8.60 | 32.5% | — |
| 48 | (32,100) | 10.79 | 31.3% | 25 |
| 64 | (32,100) | 11.08 | 31.8% | 29 |

### jupyter (H200) — same sweep

| N | (max_batch, max_wait_ms) | throughput inf/s | GPU util avg |
|---|---|---|---|
| 4 | (8,10) | 2.65 | (peak 93%) |
| 8 | (8,10) | 3.71 | 37.5% |
| 16 | (8,50) | 10.79 | 48.5% |
| 24 | (8,50) | 11.80 | 47.6% |

### Per-stage breakdown — a100 N=48 (32,100)

| stage | avg batch size | assemble_ms (CPU) | forward_ms | wait_ms |
|---|---|---|---|---|
| stage1 | 4.8 | 22 | 159 | 143 |
| stage2 | 3.5 | 7 | 111 | 121 |
| stage3 | 11.8 | 14 | 1219 | 614 |

### Per-stage forward at low contention — a100 N=4

| stage | avg batch size | forward_ms |
|---|---|---|
| stage1 | 1.3 | 74 |
| stage2 | 1.3 | 44 |
| stage3 | 1.4 | 297 |

### Resource saturation at a100 N=48

- timan107 (client) CPU: load average **15 / 48 cores**, **76–90 % idle**
  during the run.
- a100 GPU util: **33–41 %**.
- timan107 per-GPU memory: ~1 GB / 8 GB used; 48 sim procs on 48 cores.
- a100 server process CPU: 277–426 % (i.e. 2.8–4.3 cores).
- a100 GPU memory: ~7–11 GB / 40 GB.
- Per-worker throughput falls as N grows: N=8 → 0.44 inf/s/worker;
  N=48 → 0.21; N=64 → 0.17.

### Per-worker round trip

At a100 N=48, throughput ≈ 10 inf/s with 48 workers ⇒ ~4.8 s per inference
round trip per worker. Server-side stage waits+forwards sum to ≈ 2.2 s of
that.

### Correctness

LIBERO total success rate = 1.0 on all sampled workers across every
configuration tested.

## Interventions applied and their measured effect (a100 N=48)

| change | throughput before → after | GPU util before → after |
|---|---|---|
| denoise loop control: GPU-tensor `while timestep >= -dt/2` → Python `for _ in range(num_steps)` | 10.79 → 10.48 | 31.3% → 33.7% |
| per-stage CUDA streams (each stage worker on its own `torch.cuda.Stream`, sync before reply handoff) | 10.48 → 10.89 | 33.7% → 34.8% |
| disable per-connection SystemTimer CUDA-event measurement (its `stop()` calls `end_event.synchronize()` per `measure()` block) | 10.89 → 9.79 | 34.8% → 33.6% |

All three preserved LIBERO success rate = 1.0.

## The question

With a single server process, real phase5 workload, neither the client host
CPU nor the server GPU saturated, throughput plateaus at ~11 inf/s on a100
(~12 on H200) and GPU utilization plateaus at ~33% (a100) / ~48% (H200).
Adding client workers beyond N≈48 does not raise throughput or GPU
utilization. What is limiting throughput, and how can a single server process
be made to fully utilize the GPU?

## How to reproduce

1. Start server (a100): `OPENPI_MONITOR_LEVEL=basic BATCHING_MAX_WAIT_MS=100
   BATCHING_MAX_BATCH_SIZE=32 uv run --no-sync python scripts/serve_policy.py
   --port 8000 --cache-config <always_hit bootstrap yaml> policy:checkpoint
   --policy.config=pi05_libero --policy.dir=<ckpt>`
2. Preload phase5 bundle:
   `exp/serving_benchmark/preload_phase5.py --host <server> --port 8000
   --cell-id phase5_g5_g6__fh0.3_ws0.3 --eval-yaml <phase5 eval yaml>
   --phase3-warmup-dir <phase3 warmup dir>`
3. Spawn N client processes on timan107, each:
   `/scratch/zixuans8/libero_sim/bin/python examples/libero/main.py
   --host <server> --port 8000 --task-suite-name libero_spatial --num-workers 1
   --num-trials-per-task 2 --task-ids 0 --cuda-visible-devices <0..7>`
4. Read throughput/GPU/per-stage breakdown:
   `exp/serving_benchmark/dump_mem.py --host <server> --port 8000 summary`
   (the `throughput_summary` ctrl returns per-stage assemble/forward/wait +
   GPU util avg from in-memory buffers).

---

## External expert notes — 2026-05-24

The notes below are intentionally **not** part of the fact-only section above.
They are my external interpretation after reading `CLAUDE.md`, the concurrent
serving docs, the cache architecture docs, the current staged changes, and the
saved Phase-7 memory-attribution artifacts.

### Executive diagnosis

The immediate throughput limiter is not aggregate client CPU, aggregate server
CPU, network, or GPU memory. The limiter is the **single stage3 service lane**
inside the `BatchingCoordinator`.

The strongest numerical evidence is the a100 N=48 `(max_batch=32,
max_wait=100ms)` breakdown:

| stage | avg batch | assemble+forward | implied service capacity |
|---|---:|---:|---:|
| stage1 | 4.8 | 181 ms | ~26.5 inf/s |
| stage2 | 3.5 | 118 ms | ~29.7 inf/s |
| stage3 | 11.8 | 1233 ms | ~9.6 inf/s |

The observed plateau is ~10.8-11.1 inf/s. That is almost exactly the service
rate implied by stage3 alone. Queueing evidence agrees: stage3 wait is 614 ms,
far above stage1/stage2, and adding clients mostly raises queue depth / latency
instead of throughput.

The low NVML GPU-util reading is therefore not contradictory. A single Python
worker thread is issuing a long 10-step denoise loop, likely with many kernel
launch gaps, stream synchronizations, and sub-bucket serialization. The GPU is
under-filled even though the stage3 worker is the throughput bottleneck.

### Why H200 does not help much

Mode 0 direct model microbench shows H200 should be roughly 2x faster than a100
at larger batches. The real server only improves from ~11 inf/s to ~12 inf/s.
That is a major signal that the plateau is dominated by host-side scheduling /
stage dispatch / small-batch or fragmented-bucket execution, not raw GPU FLOPs.

The direct a100 batch=32 full-model p50 is ~1245 ms for 32 requests
(~25.7 rps). In the real server, stage3 alone takes ~1219 ms for an average
reported batch of only 11.8. That gap should be treated as suspicious. The most
likely explanation is that the nominal stage3 batch is internally split into
several `(mode, start_t, num_steps)` sub-buckets and executed serially, so the
actual GPU batch per denoise call is much smaller than 11.8.

### Memory finding

The earlier GPU memory blow-up was real, but it is a different problem from the
current throughput plateau.

Offline analysis of `logs/phase7_mem_attribution/stage2/history_ramp.pkl`
shows ~38.9 GB active allocation at `conn_close_n2_#1`, dominated by active
blocks from `torch.nn.Linear`, Gemma activations, eager attention softmax, and
layer norm. That points to autograd/eager-attention activation retention in the
background worker path.

After the current `torch.no_grad()` worker loop + SDPA changes,
`logs/phase7_mem_attribution/stage5_v3/no_grad_sdpa.pkl` stays around
~7.18 GB active through N=4 and cleanup snapshots. That supports the conclusion
that memory is no longer the main limiter for the N=48/N=64 throughput plateau.

### Current changes that look directionally right

- `torch.no_grad()` inside `BatchingCoordinator._stage_loop()` is mandatory.
  The interceptor's `torch.no_grad()` context is thread-local and does not
  cover coordinator daemon threads.
- Disabling per-connection `SystemTimer` CUDA-event probes for normal BASIC
  throughput runs is right. A CUDA-event `stop()` synchronizes the measured
  event and can defeat overlap when done on every stage/probe.
- Per-stage CUDA streams alone are not enough. They allow overlap, but with
  one stage3 worker there is still only one stage3 batch in service at a time.
- The new `BATCHING_STAGE3_WORKERS` direction is the first lever I would test.
  Use multiple independent stage3 workers/streams while keeping stage1/stage2
  at one worker initially.

### Risks in the current multi-worker direction

- The coordinator's `_last_assemble_ms` and `_last_forward_ms` fields are
  shared mutable state. With multiple workers, the reported per-batch metrics
  can race and become misleading even if inference correctness is fine.
- The model forward path mutates `_attn_implementation` during forward. With
  multiple worker threads using the same model instance, move that configuration
  to startup or otherwise make it immutable before relying on multi-threaded
  stage3 scaling.
- `torch.cuda.empty_cache()` every 32 batches is useful while chasing leaks, but
  it can introduce allocator locking / synchronization jitter. Once the memory
  issue is fixed, benchmark with it disabled or greatly relaxed.
- More stage3 workers will increase simultaneous in-flight activations. Current
  memory headroom looks sufficient on a100, but keep monitoring `torch_alloc_mb`
  and `reserved_mb`.

### Recommended next experiments

1. Sweep only stage3 worker count:
   `BATCHING_STAGE1_WORKERS=1`, `BATCHING_STAGE2_WORKERS=1`,
   `BATCHING_STAGE3_WORKERS in {1,2,4,8}` with `(max_batch,max_wait)` held
   fixed at `(32,100)`.
2. For each run, record throughput, GPU util, stage3 avg batch, stage3 wait,
   stage3 forward, and LIBERO success. If throughput scales with stage3 worker
   count and GPU util rises, the diagnosis is confirmed.
3. Add stage3 bucket metrics: per-bucket `(mode,start_t,num_steps)`, bucket
   size, and bucket forward_ms. The reported stage3 avg batch is not enough
   because sub-bucket fragmentation can hide the real GPU batch size.
4. Run `py-spy top` or equivalent per-thread sampling on the server during
   N=48. A process using 3-4 cores can still be bottlenecked by one hot Python
   launch thread.
5. Run a short Nsight Systems trace on a high-N plateau case. The trace should
   answer whether stage3 is launch-gap-bound, single-stream serialized,
   allocator-bound, or actually executing low-occupancy kernels.
6. Add lightweight CPU-only timing for CP1 build/search/judge at BASIC level.
   The server-side stage waits+forwards explain ~2.2 s of a ~4.8 s worker
   round trip; the remaining time needs to be split into cache CPU, websocket
   serialization, and LIBERO env stepping.

### Architectural follow-up if stage3 workers help but do not saturate

The current pipeline splits stage2 outputs into per-request `DynamicCache`
shards, returns them to connection threads, then stage3 re-stacks them. That is
correct and modular, but expensive. If the stage3-worker sweep improves
throughput but stops short of GPU saturation, the next architectural target is
to preserve batch groups across stage2 -> stage3 where possible, and only split
at true per-request decision boundaries.

Longer term, consider a continuation-style scheduler that owns the whole
request state machine instead of blocking each connection thread at every
stage. That would let the server batch, sub-batch, and advance requests without
repeated split/rejoin of large GPU objects.

### Bottom line

Treat stage3 as the bottleneck server. The single-process requirement is still
compatible with using multiple stage3 worker threads and CUDA streams inside
that process. The first credible path to higher GPU utilization is therefore:

1. keep stage1/stage2 single-lane,
2. run multiple stage3 lanes,
3. measure actual stage3 sub-bucket sizes,
4. remove remaining host synchronizations and metric races,
5. only then revisit broader architecture.

---

## Update 2 — stage3 worker sweep + seriality analysis (facts) — 2026-05-24

Applied the expert's recommended experiments. Two changes made first:
- Moved `config._attn_implementation = "sdpa"` from per-call (inside
  `_stage2_llm_backbone` / `denoise_step`) to `PI0Pytorch.__init__` (set once),
  removing the per-forward config mutation / multi-thread race.
- Added per-sub-bucket metrics in `_run_stage3_bucket`
  (`stage="3bucket"`, mode, start_t, num_steps, size, forward_ms).

### stage3 worker-count sweep — a100 N=48, (max_batch=32, max_wait=100ms)

| stage3 workers | throughput inf/s | GPU util avg | stage3 nominal batch | stage3 sub-bucket avg | stage3 forward_ms/batch | LIBERO success |
|---|---|---|---|---|---|---|
| 1 | 12.02 | 33.5% | 10.4 | 5.9 | 1034 | 1.0 |
| 2 | 9.77 | 30.5% | 7.7 | 4.6 | 1834 | 1.0 |
| 3 | 5.22 | 22.0% | 5.5 | — | 3347 | 1.0 |

Adding stage3 worker threads (each its own CUDA stream, pulling from the shared
stage3 queue) **lowers** throughput. As worker count rises: nominal batch per
worker shrinks (queue split), and forward_ms PER BATCH rises (1034 → 1834 →
3347 ms). Best is a single stage3 worker (12.02 inf/s — the highest a100 number
recorded, up from 10.89 before the attn-init change).

### stage3 sub-bucket fragmentation — a100 N=48 stage3=1

- 153 stage3 batches produced **273 sub-buckets** (≈1.78 buckets/batch).
- nominal stage3 batch avg = 10.4; **actual per-denoise-call (bucket) avg = 5.9**.
- MISS buckets: 150, avg size 7.0, forward 745 ms (10 steps).
- WARM_START buckets: 123, avg size 4.4, forward 378 ms (fewer steps).
- bucket size histogram: `{1:52, 2:30, 3:23, 4:21, 5:25, 6:26, 7:13, 8:17,
  9:12, 10:14, 11:4, 12:7, 13:9, 14:7, 15:3, 16:2, 17:3, 18:1, 20:2, 23:1, 25:1}`
  — i.e. 52 single-sample buckets each run a full denoise loop.

WARM_START is sub-bucketed by `(start_t, num_steps)`. phase5 caches at
start_t ∈ {0.3, 0.5, 0.7}, so warm requests in one batch split across up to 3
buckets; MISS is one bucket per batch.

### CPU observation (re: whether GIL is the binding constraint)

a100 server-process CPU during N=48 = 277–426 % (2.8–4.3 cores). If the Python
GIL were the binding constraint the process would sit near 100 % (one core).
The 3–4 cores in use indicate torch C-extension work runs GIL-released in
parallel; the GIL does not appear saturated.

### Seriality structure of stage3 (code-level)

stage3 is serial at three nested levels:
1. **Denoise steps within a bucket**: serial by construction — flow-matching
   Euler ODE, step t+1 consumes step t's `x_t`. Not parallelizable.
2. **Sub-buckets within one stage3 batch**: serial — `_run_stage3_bucket` is
   called in a Python `for bucket in buckets:` loop on one worker thread. The
   MISS bucket and each WARM_START bucket run one after another even though
   they are independent denoise loops.
3. **Batches**: serial per worker (1 worker ⇒ fully serial). Running >1 worker
   was measured to reduce throughput (rows above).

Level 1 is inherent. Levels 2 and 3 are implementation choices.

### Open question for the expert

If a single server process cannot be made to saturate the GPU on this workload
(stage3 = many small, fragmented, serially-executed denoise loops; per-denoise
GPU batch ≈ 5.9; GPU util plateaus ~33% on a100 / ~48% on H200; adding stage3
worker threads makes it worse), **is the right solution to scale stage3 out**
— e.g. dedicate stage3 to its own process / additional GPU(s), or shard the
stage3 service across replicas — rather than trying to saturate one GPU from
one process? If so, what scale-out topology preserves the cache framework's
per-connection correctness (the orchestrator's per-session `_state_history`,
`_step_counter`, search sessions, and the stage2→stage3 KV-cache handoff)?

---

## External expert response to Update 2 — 2026-05-24

Update 2 changes my recommendation. The worker sweep effectively falsifies the
"add more stage3 lanes on the same GPU" direction. The problem is not that the
single stage3 queue lacks enough Python worker threads; the problem is that the
GPU work reaching stage3 has the wrong shape.

### Revised diagnosis

The same-GPU multi-worker result is decisive:

- stage3 workers 1 -> 2 -> 3 lowers throughput: 12.02 -> 9.77 -> 5.22 inf/s.
- nominal stage3 batch shrinks: 10.4 -> 7.7 -> 5.5.
- actual bucket batch also shrinks: 5.9 -> 4.6.
- forward_ms per nominal batch grows sharply: 1034 -> 1834 -> 3347 ms.

That is the signature of queue fragmentation plus same-GPU contention, not
useful parallelism. Each worker drains the shared stage3 queue before a large
homogeneous bucket can form; the GPU then sees more small denoise calls, more
launch overhead, more allocator pressure, and worse GEMM efficiency.

So the current bottleneck is more specific than "stage3 is slow": **stage3 is a
bucket-shape bottleneck**. The important batch size is not the nominal stage3
batch of 10.4; it is the actual per-denoise bucket average of 5.9, plus 52
singleton buckets that each pay a full denoise loop.

### Should stage3 be scaled out?

For aggregate experiment throughput, yes, scale-out is a reasonable production
answer. But it should not be interpreted as solving single-GPU utilization; it
mostly side-steps that question by giving the workload more independent devices
and queues.

For max throughput per GPU, I would **not** add more same-GPU stage3 workers and
would not start with a remote stage3-only service. The first single-GPU target
should be larger actual denoise buckets:

1. reduce `(mode, start_t, num_steps)` key fragmentation,
2. batch by bucket key before dispatch,
3. profile stage3 with real bucket shapes.

If the objective is wall-clock LIBERO evaluation throughput across available
hardware, the pragmatic scale-out answer is full server replicas, one process
per GPU, with sticky connection or episode routing.

### Correctness-preserving scale-out topology

Preferred topology: **replicate the full policy + cache wrapper stack per
process/GPU**, behind a sticky router.

- Each replica owns its own `BatchingCoordinator`, per-connection
  `InferenceInterceptor`, `CacheOrchestrator`, `_state_history`,
  `_step_counter`, and search sessions.
- Route a WebSocket connection, or at minimum an entire episode, to exactly one
  replica from `select_bundle` / `episode_start` through `episode_end`.
- Broadcast `load_cache_config`, warmup preload, and bundle-selection control
  plane operations to all replicas, then expose a bundle only after all target
  replicas have acknowledged it.
- Load the same frozen cache bundle/backend into every replica. The runtime
  eval path already wants `write_policy=never`, so replicas do not need shared
  mutable online writes.

This topology preserves the current cache semantics because all stateful cache
correctness remains process-local. The router only distributes clients; it does
not split an orchestrator session, a trajectory history, or a stage2 -> stage3
KV object across processes.

I would avoid multiple full replicas on the **same** GPU as a first move. The
stage3 worker sweep already showed that splitting one GPU's request stream into
smaller service lanes hurts this workload. Use one full replica per GPU first.

### Stage3-only service: possible, but second choice

A remote stage3 service can be made correct only if it is stateless. The front
process must remain the owner of:

- CP1 search / judge / hit decision,
- `_state_history` and `_step_counter`,
- search session lifecycle,
- CP3 check / write / action broadcast,
- per-connection and per-episode lifecycle.

The stage3 service would be a pure function:

`(KV cache, state, prefix_pad_masks, mode, noise_or_start_x, start_t, num_steps)
-> Stage3Output`.

That design preserves correctness, but the handoff is likely expensive. The
payload contains a `DynamicCache` plus state/mask tensors. The current local
path already has to split and clone per-request KV shards for ownership; moving
that KV object across process or GPU boundaries adds serialization, lifetime,
failure, and transfer costs directly on the critical path. On a single GPU it
has no upside. Across GPUs, it is only worth considering after a same-node
CUDA-IPC/NVLink microbench with real `Stage2Output` payloads proves the
transfer cost is small relative to the saved stage3 time.

There is also a capacity ceiling: one stage1/stage2 frontend currently looks
like roughly 26-30 inf/s of service capacity, while one stage3 lane is ~12
inf/s. Two remote stage3 GPUs might be feedable; beyond that, stage1/stage2
become the new bottleneck unless they are also replicated. Full replicas avoid
that new split-brain bottleneck.

### In-process work before scale-out

1. **Bucket-first stage3 batching.** The current scheduler forms a nominal
   stage3 batch, then splits it into serial `(mode, start_t, num_steps)`
   buckets. Invert that: maintain queues by bucket key and dispatch each key at
   `target_bucket_size` or deadline. This attacks the measured avg bucket 5.9
   and singleton buckets directly.
2. **Reduce warm-start key cardinality.** Phase5 warm starts at `{0.3, 0.5,
   0.7}`, creating up to three warm buckets per nominal batch. A single
   canonical warm tier, if success holds, trades some cache granularity for
   larger buckets and better GPU efficiency.
3. **Run real stage3 microbench curves.** Measure MISS and each WARM_START
   start_t at bucket sizes `{1,2,4,8,16,32}` using real stage2 KV payloads. If
   isolated latency matches server bucket latency, workload shape is the limit.
   If server latency is much worse, chase Python staging, synchronization, or
   allocator effects.
4. **Replay bucket logs offline.** From recorded `3bucket` events, simulate
   bucket-first batching with deadlines such as 100/200/500 ms per key. This
   gives an expected throughput/latency tradeoff before changing scheduler
   code.
5. **Keep Nsight Systems on the critical path.** The remaining unknown is
   whether the low NVML util comes from launch gaps, CPU staging, allocator
   locking, or genuinely low-occupancy small kernels.

### Decision rule

- Need higher aggregate benchmark throughput now: run multiple full replicas,
  one per GPU, with sticky connection/episode routing.
- Need better single-GPU throughput: do not add same-GPU stage3 workers; improve
  actual bucket batch size and reduce warm-start fragmentation.
- Only build a stage3-only remote service after a transfer microbench with real
  `Stage2Output` objects proves that the KV handoff is cheap enough.

---

## Update 3 — decisive evidence: 2 processes on the SAME GPU = 2× throughput — 2026-05-24

New empirical fact (not yet considered above): running **two independent
server processes on the same single GPU** yields ~2× aggregate throughput
versus one process. One process alone cannot reach that throughput no matter
how it is tuned in-process.

This reframes the diagnosis:

- The GPU has ~2× FLOPS headroom available (two processes' work fits on one GPU
  and runs ~2× faster in aggregate).
- The per-process limiter is therefore **not** GPU compute, GPU memory, client
  CPU, or network. It is **how fast one process can issue CUDA kernel launches**.
- Two processes = two Python interpreters = two independent GILs = ~2× kernel
  launch rate ⇒ ~2× throughput. The CUDA-launch issuing path holds the GIL, so
  within ONE process all launch issuing is serialized regardless of how many
  worker threads exist.

This explains the earlier results coherently:
- Multiple stage3 **threads** in one process did NOT help (1 GIL ⇒ launch
  issuing still serial; queue also fragmented).
- Multiple **processes** DO help (N GILs ⇒ N× launch issuing).
- GPU util plateaus at ~33% (a100) because the single interpreter cannot issue
  the many small per-denoise-step kernels fast enough to keep the GPU fed
  (launch-latency / launch-throughput bound), not because the kernels are large.

### Implication for the single-process goal

To make ONE process match N processes without N interpreters, the launch count
issued per inference must drop by a large factor. Two in-process, correctness-
and decoupling-preserving levers, both of which reduce launches:

1. **CUDA Graph capture of the denoise step.** Capture ONE single denoise step
   (fixed batch shape) as a graph with static input/output buffers, then
   **replay it n times** per request. n is data-dependent (MISS=10,
   WARM_START=round(start_t*num_steps)=3/5/7) but the graph is identical — the
   variable step count is just a variable replay count, so warm-start is
   naturally supported by a single captured graph. This collapses ~30
   kernel launches/step to 1 replay/step (~30× fewer launches), letting one
   interpreter feed the GPU far faster. Batch size is the only shape that must
   be fixed → pad/bucket to a small set {1,2,4,8,16,32} of captured graphs.
2. **Bucket-first stage3 batching** (per external expert): queue by
   `(mode,start_t,num_steps)` key and dispatch each key at target size or
   deadline, so the actual per-denoise bucket grows beyond the measured avg 5.9
   and the 52 singleton buckets disappear. Larger buckets = larger GEMMs = fewer
   launches per inference + better GPU efficiency.

Both are being pursued. CUDA Graph attacks launch count per denoise loop;
bucket-first attacks denoise-loop count + per-call batch size.

---

## Update 4 — in-process levers exhausted; CUDA-graph infeasible — 2026-05-24

### CUDA Graph feasibility probe (decisive)

A standalone probe (`exp/serving_benchmark/cuda_graph_probe.py`) loaded
pi05_libero, built a real `Stage2Output`, and attempted to capture `run_stage3`
(MISS, 10 steps, batch 8) in a `torch.cuda.CUDAGraph`:

- `eager per-call: 244 ms`
- `CAPTURE: FAILED — RuntimeError: CUDA error: operation failed due to a previous error during capture`
- Same failure with `expandable_segments` off and `CUDA_LAUNCH_BLOCKING=1`.

The HF Gemma forward in the denoise path contains an op that is not capturable
by raw `torch.cuda.CUDAGraph`. (torch.compile `mode="reduce-overhead"` was the
next candidate but not run — compile latency/complexity vs. cache-interceptor
compatibility, deferred.)

### bucket-first stage3 scheduling — measured neutral

Implemented per-key (`mode,start_t,num_steps`) accumulation that dispatches a
bucket at `max_batch_size` or per-key deadline, instead of pull-then-split.

| config | throughput | GPU util | stage3 bucket avg |
|---|---|---|---|
| generic loop (baseline) | 12.02 | 33.5% | 5.9 |
| bucket-first, wait=100 | 11.50 | 32.7% | 6.0 |
| bucket-first, wait=300 | 10.53 | 28.5% | 5.7 |

Bucket size did **not** grow with longer wait. Reason: this is a **closed-loop**
workload — each client worker sends one request, waits for the action chunk,
then steps the LIBERO sim through the whole chunk before sending again. So the
number of in-flight requests ≈ number of client workers (N), spread across
stage1/stage2/stage3 + client sim. At stage3 at any instant there are only
~10–15 requests, split across 4 keys (miss + warm@{0.3,0.5,0.7}), so per-key
buckets stay ~5–7 regardless of the wait deadline. Longer wait only adds latency.
The arrival rate into stage3 ≈ the throughput, which is itself stage3-limited —
a closed feedback loop that keeps buckets small. (bucket-first left in code,
gated behind `OPENPI_STAGE3_BUCKET_FIRST=1`, default off.)

### Consolidated single-process result

All in-process levers tried, on a100 N=48, phase5 workload, all preserving
LIBERO success = 1.0:

| lever | throughput effect |
|---|---|
| `torch.no_grad()` in coordinator worker (memory fix; mandatory) | enables N=4 without OOM; throughput baseline |
| attn `_attn_implementation` set once in `__init__` (was per-call race) | 10.89 → 12.02 |
| denoise loop `while`(GPU tensor) → `for range` (removes per-step GPU sync) | marginal |
| per-stage / per-worker CUDA streams | +~4% |
| disable per-conn SystemTimer CUDA-event sync at < SNAPSHOT | neutral |
| ≥2 stage3 worker threads (same GPU) | WORSE (queue split + contention) |
| bucket-first stage3 scheduling | neutral / worse |
| raw CUDA graph of denoise | capture fails (HF op not graph-safe) |

Single-process a100 plateau: **~12 inf/s, GPU util ~33%**. The 2-process =
2× observation (same GPU) localises the ceiling to **per-process Python
kernel-launch throughput (one GIL issues CUDA launches serially)**, combined
with **closed-loop low concurrency** keeping per-denoise batches small. With
raw CUDA graph infeasible on this model, the remaining options are
torch.compile(reduce-overhead) (deferred) or multi-process replicas.

### Status / decision

Single-process optimization is paused here. Server code + config restored to
the best measured single-process state (all correctness/memory fixes kept;
bucket-first and multi-worker gated off by default). If single-process cannot
be pushed further, the planned path is **scale-out (multi-process replicas,
one per GPU, sticky per-connection routing)** as documented in the expert's
"correctness-preserving scale-out topology" section above.

---

## External expert response to Updates 3-4 — 2026-05-24

I would move the mainline effort to scale-out now.

Update 3 is the decisive result: two independent server processes on the same
GPU produce roughly 2x aggregate throughput, while extra stage3 threads inside
one process make throughput worse. That strongly localizes the remaining
single-process ceiling to the Python/CUDA launch issuing path, not to GPU
compute capacity, model memory, cache correctness, networking, or client CPU.

Update 4 then closes the two practical single-process escapes:

- raw CUDA Graph capture fails on a real `run_stage3` path,
- bucket-first scheduling does not grow bucket size in the real closed-loop
  workload and is neutral/worse.

At this point, continued single-process tuning is unlikely to produce a large
step-function improvement. There is one speculative leftover: `torch.compile`
with `mode="reduce-overhead"` on the stage3 path. It is worth at most a short
parallel spike because `reduce-overhead` can sometimes partition around
graph-unsafe ops that break raw `CUDAGraph`. But it should not block scale-out:
the concurrent coordinator path currently runs through raw stage calls, warm
start uses `run_stage3_from`, MISS may request intermediates, and any compiled
path has to be revalidated for output lifetime, cache key-builder references,
and all MISS/WARM_START variants. If it does not quickly demonstrate a large
speedup on the standalone probe and then in server mode, drop it.

### Final single-process assessment

I do not see another credible in-process optimization with the same upside as
replicas:

- More client workers only increases queueing because the workload is
  closed-loop.
- Longer batch wait only adds latency; it does not create more same-key stage3
  arrivals.
- More same-process stage3 threads do not bypass one interpreter's launch
  issuing limit and also fragment the queue.
- More CUDA streams have already shown only marginal benefit.
- A stage3-only remote service is scale-out with extra KV-transfer complexity,
  not a simpler single-process fix.

The best measured single-process state should remain the baseline: memory fixes
kept, SDPA configured once, tensor-while removed, per-stage streams enabled if
they help, bucket-first off by default, multi-worker off by default.

### Recommended scale-out plan

Use **full server replicas**, not a stage3-only service, as the first scale-out
architecture.

Because Update 3 shows two processes on the same GPU already help, make
`replicas_per_gpu` a measured parameter instead of assuming one process per GPU.
For a100, start with:

| replicas/GPU | expectation |
|---:|---|
| 1 | current ~12 inf/s baseline |
| 2 | expected near measured ~2x if memory and latency stay acceptable |
| 3 | maybe useful until GPU util or context overhead flattens |
| 4 | likely diminishing returns; measure before relying on it |

For multi-GPU hosts, run the same sweep per GPU, then choose the best
`replicas_per_gpu` value and multiply across GPUs. If available, NVIDIA MPS is
worth A/B testing because this is a multi-process, small-kernel launch workload;
but the already-measured two-process result means MPS is optional, not a
precondition.

### Routing and correctness contract

The routing rule should be strict: **sticky connection, not per-request
balancing**.

- A LIBERO worker opens one WebSocket connection to one replica and keeps it
  there through `select_bundle`, `episode_start`, all `infer` calls, and
  `episode_end`.
- If a connection dies mid-episode, reconnect to the same replica or restart
  the episode. Do not resume a live orchestrator session on another replica.
- A central proxy is optional. The simplest first implementation is client-side
  port sharding: start replicas on ports `8000..8000+R-1` and assign each
  worker to `ports[worker_id % R]`.
- If a proxy is used, it must be a WebSocket connection-level proxy with sticky
  assignment, not a per-message load balancer.

This preserves correctness because each replica owns its own per-connection
`InferenceInterceptor`, `CacheOrchestrator`, `_state_history`, `_step_counter`,
search sessions, and stage2 -> stage3 KV lifetime.

Control-plane operations must be broadcast to all replicas before evaluation:

- `load_cache_config`
- `preload_normalizer_buffer`
- `select_bundle` availability / bundle registration
- any warmup or unload operation that affects the selected bundle

The runtime cache path is frozen/read-only, so loading the same artifact into
each replica is acceptable. Do not share orchestrator state across replicas.

### What not to build first

Do not build a stage3-only service first. It would have to move `DynamicCache`
and stage1 tensors across process/GPU boundaries on the hot path, while the
front process still owns all orchestrator/session correctness. That is more
complex than full replicas and may simply move the bottleneck to KV transfer or
stage1/stage2 fan-out.

Do not invest further in same-process thread/stream variants unless a profiler
shows a new, specific synchronization problem. The current experiments already
show the structural limit.

### Go/no-go

My go/no-go recommendation is:

- **Go scale-out now** as the primary implementation path.
- Run `torch.compile(mode="reduce-overhead")` only as a bounded side experiment
  if it is cheap to run; do not make it a prerequisite.
- Validate scale-out with a sweep over `replicas_per_gpu`, reporting aggregate
  throughput, p50/p95 latency, GPU util, memory, and LIBERO success.
