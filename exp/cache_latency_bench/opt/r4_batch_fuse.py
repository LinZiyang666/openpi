"""ROUND 4 build-segment — BATCH/FUSE lens (THROWAWAY research script).

Sister script to r4_build_micro.py. That script already covers per-field variants
(avg_pool2d vs adaptive, reshape-mean, channels-last) + _to_cpu_float32 cost. This one
isolates the specific batch/fuse question the lens owns:

  The active build does vision_0 AND vision_1 each as a SEPARATE spatial_pool
  ([256,2048]->[32768]). Lens: stack the two cameras into [2,256,2048] and do ONE
  adaptive_avg_pool2d / one reshape-mean, halving the Python+kernel dispatch count.

It answers, with statistically-robust median-of-runs timing at FIXED thread count:
  Q1  2x separate adaptive_avg_pool2d  vs  1x batched [2,2048,16,16] pool  (kernel fuse)
  Q2  cost of torch.stack([v0,v1]) itself (the price of batching)
  Q3  does batched adaptive_avg_pool2d stay BIT-EXACT vs 2x separate src calls?
  Q4  GPU path (production): is the batched pool win bigger when kernel launch
      latency dominates (GPU dispatch ~10us each), and is the split-back cheap?

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_batch_fuse.py
"""

from __future__ import annotations

import glob
import os
import statistics
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

GRID, POOL, EMB = 16, 4, 2048
H5_DIR = "exp/common/data/db/libero_cache/libero_10"

# --------------------------------------------------------------------------
# Real raw vision tokens (inputs to _reduce_vision) from H5 fake-stage1.
# --------------------------------------------------------------------------
from exp.common.build_in_memory_cache_artifact import _build_fake_stage1  # noqa: E402

paths = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
pairs = []  # list of (tok0[256,2048], tok1[256,2048])
with h5py.File(paths[0], "r") as f:
    names = sorted((k for k in f if k.startswith("step_")), key=lambda s: int(s.split("_")[-1]))
    for name in names[:64]:
        fs = _build_fake_stage1(f[name])
        prefix = fs.prefix_embs[0]
        pairs.append((prefix[0:256].contiguous(), prefix[256:512].contiguous()))
print(f"loaded {len(pairs)} real (v0,v1) token pairs from {os.path.basename(paths[0])}")


# --------------------------------------------------------------------------
# src per-field pool (verbatim) and the candidates.
# --------------------------------------------------------------------------
def src_pool(tokens):
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def two_separate_adaptive(v0, v1):
    return src_pool(v0), src_pool(v1)


def batched_adaptive(v0, v1):
    """ONE adaptive_avg_pool2d over [2,2048,16,16]."""
    x = torch.stack((v0, v1)).reshape(2, GRID, GRID, EMB).permute(0, 3, 1, 2)  # [2,2048,16,16]
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))  # [2,2048,4,4]
    out = pooled.permute(0, 2, 3, 1).reshape(2, -1)  # [2,32768], (b,oh,ow,emb) == src order
    return out[0], out[1]


def batched_avgpool(v0, v1):
    """ONE avg_pool2d(kernel=4) over [2,2048,16,16] (fixed pool, bit-exact-candidate)."""
    x = torch.stack((v0, v1)).reshape(2, GRID, GRID, EMB).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(x, kernel_size=POOL, stride=POOL)
    out = pooled.permute(0, 2, 3, 1).reshape(2, -1)
    return out[0], out[1]


def two_separate_avgpool(v0, v1):
    def one(t):
        x = t.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
        p = F.avg_pool2d(x, kernel_size=POOL, stride=POOL)
        return p.squeeze(0).permute(1, 2, 0).reshape(-1)
    return one(v0), one(v1)


# --------------------------------------------------------------------------
# Q3 equivalence: batched vs 2x separate src, over all pairs.
# --------------------------------------------------------------------------
print("\n=== Q3 EQUIVALENCE: batched vs 2x separate src (all pairs) ===")
for tag, fn in (("batched_adaptive", batched_adaptive), ("batched_avgpool", batched_avgpool),
                ("two_separate_avgpool", two_separate_avgpool)):
    max_abs, max_bit = 0.0, 0
    for v0, v1 in pairs:
        r0, r1 = src_pool(v0), src_pool(v1)
        c0, c1 = fn(v0, v1)
        for r, c in ((r0, c0), (r1, c1)):
            max_abs = max(max_abs, (r - c).abs().max().item())
            max_bit = max(max_bit, (r != c).sum().item())
    verdict = "BIT-EXACT" if max_bit == 0 else f"fp-diff ({max_bit} elems)"
    print(f"  {tag:22s} max|diff|={max_abs:.3e}  {verdict}")


# --------------------------------------------------------------------------
# Robust timing: median of R runs, each run = mean over `iters`, FIXED threads.
# Pinning threads avoids the adaptive-scheduler noise seen in the sister script.
# --------------------------------------------------------------------------
def robust(fn, iters=200, warmup=50, runs=15):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t0) / iters * 1000.0)
    return statistics.median(samples), min(samples)


v0, v1 = pairs[len(pairs) // 2]


def q_stack():
    return torch.stack((v0, v1))


cpu_cases = [
    ("2x separate adaptive [SRC]", lambda: two_separate_adaptive(v0, v1)),
    ("1x batched  adaptive", lambda: batched_adaptive(v0, v1)),
    ("2x separate avg_pool", lambda: two_separate_avgpool(v0, v1)),
    ("1x batched  avg_pool", lambda: batched_avgpool(v0, v1)),
    ("torch.stack(v0,v1) only", q_stack),
]

for nthreads in (1, 4):
    torch.set_num_threads(nthreads)
    print(f"\n=== CPU TIMING (2-vision pool algebra, no to_cpu)  threads={nthreads} ===")
    print(f"  {'impl':30s} {'median ms':>10s} {'min ms':>9s}")
    for name, fn in cpu_cases:
        med, mn = robust(fn)
        print(f"  {name:30s} {med:10.4f} {mn:9.4f}")


# --------------------------------------------------------------------------
# Q4 GPU path (production reality): vision encoder output is on GPU. Here the
# kernel-launch count matters most. Time 2x vs 1x batched on GPU with proper sync.
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    print("\n=== Q4 GPU TIMING (2-vision pool algebra, cuda sync) ===")
    g0, g1 = v0.cuda(), v1.cuda()

    def gpu_robust(fn, iters=200, warmup=50, runs=15):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / iters * 1000.0)
        return statistics.median(samples), min(samples)

    gpu_cases = [
        ("2x separate adaptive [SRC]", lambda: two_separate_adaptive(g0, g1)),
        ("1x batched  adaptive", lambda: batched_adaptive(g0, g1)),
        ("2x separate avg_pool", lambda: two_separate_avgpool(g0, g1)),
        ("1x batched  avg_pool", lambda: batched_avgpool(g0, g1)),
        ("torch.stack(g0,g1) only", lambda: torch.stack((g0, g1))),
    ]
    print(f"  {'impl':30s} {'median ms':>10s} {'min ms':>9s}")
    for name, fn in gpu_cases:
        med, mn = gpu_robust(fn)
        print(f"  {name:30s} {med:10.4f} {mn:9.4f}")

    # GPU bit-exactness of batched vs separate (kernel-level; may differ from CPU).
    r0 = src_pool(g0)
    c0, _ = batched_adaptive(g0, g1)
    ca0, _ = batched_avgpool(g0, g1)
    print(f"\n  GPU batched_adaptive vs src max|diff| = {(r0 - c0).abs().max().item():.3e}"
          f"  bit-exact={torch.equal(r0, c0)}")
    print(f"  GPU batched_avgpool  vs src max|diff| = {(r0 - ca0).abs().max().item():.3e}"
          f"  bit-exact={torch.equal(r0, ca0)}")
else:
    print("\n[Q4] no CUDA; GPU path skipped")
