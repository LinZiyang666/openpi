"""ROUND 4 — WHY is the batched pool 4-6x faster on CPU? (THROWAWAY).

r4_batch_fuse.py showed 2x-separate-adaptive=5.1ms vs 1x-batched=1.1ms (t=1).
That is far more than the 2x dispatch saving alone predicts. This script decomposes
the SRC per-field path step-by-step to localize the cost, and forces materialization
(.sum() touch) so we can't be fooled by lazy/uncomputed outputs.

Hypotheses:
  H1 the second permute(1,2,0).reshape(-1) copy of [4,4,2048]->[32768] is cheap (small).
  H2 the INPUT permute(2,0,1) of [16,16,2048]->[2048,16,16] forces a 256*2048 copy that
     adaptive_avg_pool2d on a [1,2048,16,16] tensor then reads slowly (channels=2048 huge,
     spatial 16x16 tiny -> bad cache pattern); batching to [2,2048,16,16] changes nothing
     there, so the win must be elsewhere.
  H3 the win is the .reshape(GRID,GRID,EMB) + .permute being re-run twice + python overhead.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_batch_why.py
"""

from __future__ import annotations

import glob
import os
import statistics
import time

import h5py
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
GRID, POOL, EMB = 16, 4, 2048
H5_DIR = "exp/common/data/db/libero_cache/libero_10"

from exp.common.build_in_memory_cache_artifact import _build_fake_stage1  # noqa: E402

p = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))[0]
with h5py.File(p, "r") as f:
    names = sorted((k for k in f if k.startswith("step_")), key=lambda s: int(s.split("_")[-1]))
    fs = _build_fake_stage1(f[names[len(names) // 2]])
    prefix = fs.prefix_embs[0]
    v0 = prefix[0:256].contiguous()
    v1 = prefix[256:512].contiguous()


def robust(fn, iters=200, warmup=50, runs=15):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        s.append((time.perf_counter() - t0) / iters * 1000.0)
    return statistics.median(s)


# Force materialize with .sum() so nothing is skipped.
def m(t):
    return float(t.sum())


# Decompose src per-field [256,2048]->[32768].
def s1_reshape():
    return m(v0.reshape(GRID, GRID, EMB))


def s2_reshape_permute_in():
    return m(v0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0))


def s3_full_pool():
    x = v0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    return m(F.adaptive_avg_pool2d(x, (POOL, POOL)))


def s4_full_src():
    x = v0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return m(pooled.squeeze(0).permute(1, 2, 0).reshape(-1))


# The INPUT permute(2,0,1) builds a [2048,16,16] view; pool must read it. Is the cost
# the pool reading a transposed (non-contig) input, or the permute materialization?
def s5_pool_contig_in():
    x = v0.reshape(GRID, GRID, EMB).permute(2, 0, 1).contiguous().unsqueeze(0)  # force copy first
    return m(F.adaptive_avg_pool2d(x, (POOL, POOL)))


# Batched, force materialize both outputs.
def s6_batched():
    x = torch.stack((v0, v1)).reshape(2, GRID, GRID, EMB).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    out = pooled.permute(0, 2, 3, 1).reshape(2, -1)
    return m(out[0]) + m(out[1])


# 2x separate, force materialize both.
def s7_two_src():
    def one(t):
        x = t.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
        return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)
    return m(one(v0)) + m(one(v1))


for nthreads in (1, 4):
    torch.set_num_threads(nthreads)
    print(f"\n=== DECOMPOSE src per-field path (materialized)  threads={nthreads} ===")
    for name, fn in [
        ("reshape [16,16,2048]", s1_reshape),
        ("+ permute(2,0,1) in", s2_reshape_permute_in),
        ("+ adaptive_pool", s3_full_pool),
        ("+ out permute+reshape (FULL 1 field)", s4_full_src),
        ("pool with CONTIG input", s5_pool_contig_in),
        ("-- 2x separate FULL [SRC]", s7_two_src),
        ("-- 1x batched FULL", s6_batched),
    ]:
        print(f"  {name:40s} {robust(fn):8.4f} ms")
