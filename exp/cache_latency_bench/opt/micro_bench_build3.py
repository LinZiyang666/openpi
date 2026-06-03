"""ROUND 4 micro-bench part 3: robustness of matmul_pool bit-exactness across
many seeds + realistic value scales, and full-build latency with matmul_pool.

The win: average-pool over fixed non-overlapping blocks is LINEAR -> a sparse
pooling matrix P[16,256] @ tokens[256,2048]. GEMM is ~10x faster than the
pooling kernel on this tiny tensor and outputs contiguous [16,2048] (no permute,
no non-contiguous reshape copy). Need: is P@tokens bit-exact vs adaptive on
REAL-scale data, not just seed 0?

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/micro_bench_build3.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
GRID, POOL, EMB = 16, 4, 2048
NTOK = GRID * GRID


def stock(tokens):
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


P = torch.zeros(POOL * POOL, NTOK, dtype=torch.float32)
for r in range(GRID):
    for c in range(GRID):
        o = (r // 4) * POOL + (c // 4)
        P[o, r * GRID + c] = 1.0 / 16.0
P = P.contiguous()


def matmul_pool(tokens):
    return (P @ tokens).reshape(-1)


# Bit-exactness sweep across seeds, scales, and the LIBERO real pack if available.
print("BIT-EXACTNESS SWEEP (matmul_pool vs stock adaptive_avg_pool2d):")
worst = 0.0
n_exact = 0
n_total = 0
for scale in (0.01, 1.0, 10.0, 100.0):
    for seed in range(50):
        torch.manual_seed(seed)
        tok = (torch.randn(NTOK, EMB, dtype=torch.float32) * scale).contiguous()
        a, b = stock(tok), matmul_pool(tok)
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        n_exact += int(torch.equal(a, b))
        n_total += 1
print(f"  random sweep: {n_exact}/{n_total} bit-exact, worst max|diff|={worst:.3e}")

# Real LIBERO data path: rebuild raw vision tokens from one H5 step and compare.
import glob
import os

h5dir = "exp/common/data/db/libero_cache/libero_10"
paths = sorted(glob.glob(os.path.join(h5dir, "*.h5")))
if paths:
    import h5py
    import numpy as np
    with h5py.File(paths[0], "r") as f:
        step = sorted([k for k in f.keys() if k.startswith("step_")])[0]
        g = f[step]
        v0 = torch.from_numpy(np.array(g["vision_0"])).float().contiguous()  # [256,2048]
    a, b = stock(v0), matmul_pool(v0)
    print(f"  REAL libero_10 vision_0: bit-exact={torch.equal(a,b)} "
          f"max|diff|={(a-b).abs().max():.3e} "
          f"1-cos={1.0-float(F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0))):.2e}")
else:
    print(f"  (no H5 under {h5dir})")

# Full-build latency: 2 vision pools + robot_state, with _to_cpu_float32 semantics.
torch.manual_seed(0)
tok0 = torch.randn(NTOK, EMB).contiguous()
tok1 = torch.randn(NTOK, EMB).contiguous()
rs = torch.randn(32).contiguous()


def to_cpu(t):
    return t.cpu().float().contiguous()


def build_stock():
    return to_cpu(stock(tok0)), to_cpu(stock(tok1)), to_cpu(rs)


def build_matmul():
    return to_cpu(matmul_pool(tok0)), to_cpu(matmul_pool(tok1)), to_cpu(rs)


def bench(fn, iters=500, warmup=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


print("\nFULL BUILD latency (2 vision pool + 3 to_cpu), ms:")
for t in (1, 4, 8):
    torch.set_num_threads(t)
    bs, bm = bench(build_stock), bench(build_matmul)
    print(f"  threads={t}: stock={bs:.4f}  matmul={bm:.4f}  speedup={bs/bm:.2f}x")
