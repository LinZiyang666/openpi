"""ROUND 4 micro-bench: decompose the cp1_build segment and bound pool optimizations.

build() on depth_1.yaml does: slice prefix_embs -> for vision_0 & vision_1 do
_spatial_pool_tokens([256,2048] -> [32768]) -> _to_cpu_float32, + robot_state[32] raw
-> _to_cpu_float32.

This script (a) decomposes build into its real sub-steps (slice / reshape+permute /
pool / second permute+reshape / _to_cpu_float32), (b) verifies bit/numeric equivalence
of each pool optimization candidate vs the stock _spatial_pool_tokens, and (c) measures
the speedup, across thread counts. Throwaway research script (delete after Round 4).

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/micro_bench_build.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

GRID = 16
POOL = 4
EMB = 2048
NTOK = GRID * GRID  # 256

# Synthetic SigLIP-like patch tokens, CPU float32 (matches bench fake-stage1 path).
torch.manual_seed(0)
tok0 = torch.randn(NTOK, EMB, dtype=torch.float32).contiguous()
tok1 = torch.randn(NTOK, EMB, dtype=torch.float32).contiguous()
rs = torch.randn(32, dtype=torch.float32).contiguous()


# ----------------------------------------------------------------------
# Stock implementation (verbatim from src key_builder.py)
# ----------------------------------------------------------------------
def stock_spatial_pool(tokens, grid_size=GRID, pool_size=POOL):
    emb_dim = tokens.shape[1]
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def stock_to_cpu_float32(t):
    return t.cpu().float().contiguous()


# ----------------------------------------------------------------------
# Candidate 1: adaptive_avg_pool2d -> avg_pool2d (fixed kernel/stride)
# ----------------------------------------------------------------------
def cand_avgpool2d(tokens, grid_size=GRID, pool_size=POOL):
    emb_dim = tokens.shape[1]
    k = grid_size // pool_size  # 4
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


# ----------------------------------------------------------------------
# Candidate 2: avoid the [emb,grid,grid] channels-first layout entirely.
# adaptive_avg_pool over a 16x16 grid into 4x4 with channels LAST.
# tokens[256,2048] -> [16,16,2048] -> view as [4,4,4,4,2048] -> mean over the
# two inner 4-blocks. This is exactly a non-overlapping 4x4 average pool with
# output order matching the stock permute(1,2,0): out[i,j,c] = mean over the
# block (i*4:(i+1)*4, j*4:(j+1)*4). reshape(-1) gives [i,j,c] row-major == stock.
# ----------------------------------------------------------------------
def cand_view_mean(tokens, grid_size=GRID, pool_size=POOL):
    emb_dim = tokens.shape[1]
    k = grid_size // pool_size
    x = tokens.reshape(pool_size, k, pool_size, k, emb_dim)  # [4,4,4,4,2048]
    # mean over the two block dims (1 and 3); keep [pool_i, pool_j, emb]
    return x.mean(dim=(1, 3)).reshape(-1)


# ----------------------------------------------------------------------
# Candidate 3: same idea but with explicit channels-first avg_pool2d on a
# pre-contiguous reshape (no permute), then no trailing permute. Sanity ref.
# Skipped — channels order differs; not equivalent. Keep view_mean as the win.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Candidate 4: batch vision_0 + vision_1 into ONE pool call.
# stack -> [2,256,2048] -> [2,16,16,2048] -> view [2,4,4,4,4,2048] -> mean
# -> [2,4,4,2048] -> reshape [2,32768]. One kernel for both cameras.
# ----------------------------------------------------------------------
def cand_batched_view_mean(t0, t1, grid_size=GRID, pool_size=POOL):
    emb_dim = t0.shape[1]
    k = grid_size // pool_size
    x = torch.stack((t0, t1), dim=0)  # [2,256,2048]
    x = x.reshape(2, pool_size, k, pool_size, k, emb_dim)
    out = x.mean(dim=(2, 4)).reshape(2, -1)  # [2,32768]
    return out[0].contiguous(), out[1].contiguous()


# ----------------------------------------------------------------------
# Equivalence checks
# ----------------------------------------------------------------------
print("=" * 72)
print("EQUIVALENCE (vs stock _spatial_pool_tokens, CPU float32)")
print("=" * 72)
ref0 = stock_spatial_pool(tok0)
for name, fn in [
    ("avg_pool2d (fixed k=4)", cand_avgpool2d),
    ("view+mean(1,3)", cand_view_mean),
]:
    out = fn(tok0)
    d = (out - ref0).abs().max().item()
    exact = torch.equal(out, ref0)
    print(f"  {name:28s} max|diff|={d:.3e}  bit-exact={exact}  shape={tuple(out.shape)}")

b0, b1 = cand_batched_view_mean(tok0, tok1)
ref1 = stock_spatial_pool(tok1)
d0 = (b0 - ref0).abs().max().item()
d1 = (b1 - ref1).abs().max().item()
print(f"  {'batched view+mean v0':28s} max|diff|={d0:.3e}  bit-exact={torch.equal(b0, ref0)}")
print(f"  {'batched view+mean v1':28s} max|diff|={d1:.3e}  bit-exact={torch.equal(b1, ref1)}")

# Cosine impact: the key is consumed as a normalized vector in cosine search.
# Even tiny abs diffs must not move cosine. Report cosine(stock, cand).
def cos(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

print("\ncosine(stock, cand) on vision_0 (search consumes normalized key):")
print(f"  avg_pool2d      : {1.0 - cos(ref0, cand_avgpool2d(tok0)):.3e} (1-cos)")
print(f"  view+mean       : {1.0 - cos(ref0, cand_view_mean(tok0)):.3e} (1-cos)")


# ----------------------------------------------------------------------
# Sub-step decomposition (where does build time go?)
# ----------------------------------------------------------------------
def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


# isolated sub-steps of stock pool
def s_reshape_permute():
    return tok0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)


_x = tok0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)


def s_adaptive_pool():
    return F.adaptive_avg_pool2d(_x, (POOL, POOL))


def s_avgpool():
    return F.avg_pool2d(_x, kernel_size=4, stride=4)


_pooled = F.adaptive_avg_pool2d(_x, (POOL, POOL))


def s_permute_reshape_out():
    return _pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def s_to_cpu():  # .cpu().float().contiguous() on already-cpu-f32 [32768]
    return ref0.cpu().float().contiguous()


def s_contiguous_only():
    return ref0.contiguous()


# full per-step build (2 vision pool + 2 to_cpu + robot_state to_cpu)
def build_stock():
    k0 = stock_to_cpu_float32(stock_spatial_pool(tok0))
    k1 = stock_to_cpu_float32(stock_spatial_pool(tok1))
    kr = stock_to_cpu_float32(rs)
    return k0, k1, kr


def build_viewmean():
    k0 = cand_view_mean(tok0).contiguous()
    k1 = cand_view_mean(tok1).contiguous()
    kr = rs.contiguous()
    return k0, k1, kr


def build_batched():
    bb0, bb1 = cand_batched_view_mean(tok0, tok1)
    kr = rs.contiguous()
    return bb0, bb1, kr


print("\n" + "=" * 72)
print("SUB-STEP & FULL-BUILD LATENCY (ms) across thread counts")
print("=" * 72)
cases = [
    ("sub: reshape+permute->[1,2048,16,16]", s_reshape_permute),
    ("sub: adaptive_avg_pool2d(16->4)", s_adaptive_pool),
    ("sub: avg_pool2d(k=4)", s_avgpool),
    ("sub: out permute+reshape->[32768]", s_permute_reshape_out),
    ("sub: _to_cpu_float32 (cpu f32 in)", s_to_cpu),
    ("sub: .contiguous() only", s_contiguous_only),
    ("FULL build stock (2pool+3cpu)", build_stock),
    ("FULL build view+mean", build_viewmean),
    ("FULL build batched", build_batched),
]
threads = (1, 4, 8)
print(f"\n{'impl':40s} " + " ".join(f"t={t:<7d}" for t in threads))
for name, fn in cases:
    row = []
    for t in threads:
        torch.set_num_threads(t)
        row.append(bench(fn))
    print(f"{name:40s} " + " ".join(f"{v:8.4f}" for v in row))

# the observed bench median build ~1.33ms was at threads=4 (round3 runner pins 4).
torch.set_num_threads(4)
bs = bench(build_stock)
bv = bench(build_viewmean)
bb = bench(build_batched)
print(f"\n@threads=4  stock={bs:.4f}ms  view+mean={bv:.4f}ms ({bs/bv:.2f}x)  batched={bb:.4f}ms ({bs/bb:.2f}x)")
