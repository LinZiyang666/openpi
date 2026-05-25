# Mode 0 — GPU-direct microbench results (2026-05-24)

Plan: [`logs/concurrent_serving_optimization_plan.log.md`](../../../../logs/concurrent_serving_optimization_plan.log.md) §11 Phase 7.

Bypasses server / WebSocket / cache wrappers and calls `model.sample_actions`
directly on `pi05_libero_pytorch` checkpoint. Establishes the theoretical
throughput-latency upper bound any BatchingCoordinator can approach.

## Setup

- Model: `pi05_libero` (Pi0.5, paligemma_2b + gemma_300m action expert, bf16)
- Forward path: `TORCH_COMPILE_DISABLE=1` (skip max-autotune compile)
- Iters: 100 per batch_size, warmup 10
- Tools: `python -m exp.serving_benchmark.gpu_microbench`

## Hosts

| Host | GPU | Run dir |
|---|---|---|
| a100 | NVIDIA A100-SXM4-40GB | `data/full_a100_20260524_0241/` |
| jupyter-ziyang10 | NVIDIA H200 NVL (140 GiB) | `data/full_jupyter_20260524_0249/` |

## Throughput-latency table

| batch | a100 p50 ms | a100 rps | H200 p50 ms | H200 rps | H200/A100 rps |
|---|---|---|---|---|---|
| 1 | 325.81 | 3.1 | 113.06 | 8.8 | **2.9×** |
| 2 | 360.89 | 5.5 | 133.61 | 15.0 | 2.7× |
| 4 | 414.32 | 9.7 | 154.28 | 25.9 | 2.7× |
| 8 | 515.45 | 15.5 | 201.41 | 39.7 | 2.6× |
| 16 | 742.79 | 21.5 | 332.29 | 48.2 | 2.2× |
| 32 | 1245.13 | 25.7 | 593.24 | 53.9 | 2.1× |

## Observations

- **Saturation knee**: A100 → batch=8 (latency grows ~1.5× per batch doubling
  beyond 8 while rps grows <1.4×). H200 → batch=8~16.
- **batch=8 marginal**: A100 +29% latency / +57% rps vs batch=4. H200 +30% / +53%.
  Strongest cost/benefit point.
- **batch=16 marginal**: A100 +44% / +39%. H200 +65% / +21%. Beyond knee.
- **H200/A100 ratio shrinks with batch size** (2.9× → 2.1×): small batches are
  memory-bandwidth-bound (H200 4.8 TB/s vs A100 2.0 TB/s ≈ 2.4×), large batches
  approach compute-bound (FP16 989 vs 312 TFLOPS ≈ 3.2× theoretical, not
  realized due to kernel inefficiencies at non-tuned shapes).

## Conclusion for BatchingCoordinator defaults

- **`max_batch_size=8`** (plan default) is **confirmed appropriate** for both
  hosts: it sits exactly at A100's knee, and gives H200 most of its achievable
  throughput without inflating per-request latency past 200 ms.
- `max_wait_ms=10` (plan default) was not directly tested in Mode 0; defer to
  Mode 4 (`batch_window.yaml`) for that sweep.

## Open items deferred to Mode 1–4

- Real-cache search overhead (CP1/CP3 path through `InMemoryBackend`) not
  represented here — could shift the practical knee left.
- WebSocket + multi-worker contention not represented — Mode 1 / 2.
- BackendPool memory savings under K bundles — Mode 3.
- max_wait_ms / max_batch_size joint sweep — Mode 4.
