"""ROUND 4 micro-bench part 2: why is adaptive_avg_pool2d slow, and is the
non-contiguous permuted input the cause? Also pin down view+mean equivalence and
test a contiguous-first variant + einsum + reshape-mean orderings.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/micro_bench_build2.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
GRID, POOL, EMB = 16, 4, 2048
NTOK = GRID * GRID
torch.manual_seed(0)
tok0 = torch.randn(NTOK, EMB, dtype=torch.float32).contiguous()


def stock(tokens):
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


ref = stock(tok0)


def avgpool_noncontig(tokens):
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)  # non-contig
    p = F.avg_pool2d(x, 4, 4)
    return p.squeeze(0).permute(1, 2, 0).reshape(-1)


def avgpool_contig(tokens):
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).contiguous().unsqueeze(0)
    p = F.avg_pool2d(x, 4, 4)
    return p.squeeze(0).permute(1, 2, 0).reshape(-1)


def view_mean(tokens):
    x = tokens.reshape(POOL, 4, POOL, 4, EMB)
    return x.mean(dim=(1, 3)).reshape(-1)


def view_sum_div(tokens):
    # sum then divide: same reduction tree as mean but explicit
    x = tokens.reshape(POOL, 4, POOL, 4, EMB)
    return (x.sum(dim=(1, 3)) / 16.0).reshape(-1)


def einsum_mean(tokens):
    # average via einsum with a (4x4)->(4) block averaging? Express as a matmul:
    # reshape [256,2048] = [16block_rows? ...]. Use a pooling matrix P[16,256] where
    # each output cell averages its 16 source patches (in the [i,j] spatial order).
    pass  # built below as matmul


# Build pooling matrix once: out[o, :] = mean over the 16 patches in block o.
# patch index p = r*16 + c (r,c in 0..15). block index o = (r//4)*4 + (c//4).
P = torch.zeros(POOL * POOL, NTOK, dtype=torch.float32)
for r in range(GRID):
    for c in range(GRID):
        o = (r // 4) * POOL + (c // 4)
        P[o, r * GRID + c] = 1.0 / 16.0
P = P.contiguous()


def matmul_pool(tokens):
    # [16,256] @ [256,2048] -> [16,2048] -> reshape [32768]
    return (P @ tokens).reshape(-1)


print("EQUIVALENCE vs stock:")
for name, fn in [
    ("avgpool_noncontig", avgpool_noncontig),
    ("avgpool_contig", avgpool_contig),
    ("view_mean", view_mean),
    ("view_sum_div", view_sum_div),
    ("matmul_pool", matmul_pool),
]:
    out = fn(tok0)
    print(f"  {name:20s} max|diff|={(out-ref).abs().max():.3e} bit-exact={torch.equal(out,ref)} "
          f"1-cos={1.0-float(F.cosine_similarity(out.unsqueeze(0),ref.unsqueeze(0))):.2e}")


def bench(fn, iters=400, warmup=40):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


print("\nLATENCY (single vision field, ms):")
cases = [
    ("stock", lambda: stock(tok0)),
    ("avgpool_noncontig", lambda: avgpool_noncontig(tok0)),
    ("avgpool_contig", lambda: avgpool_contig(tok0)),
    ("view_mean", lambda: view_mean(tok0)),
    ("view_sum_div", lambda: view_sum_div(tok0)),
    ("matmul_pool", lambda: matmul_pool(tok0)),
]
for t in (1, 4, 8):
    torch.set_num_threads(t)
    print(f" threads={t}: " + "  ".join(f"{n}={bench(f):.4f}" for n, f in cases))
