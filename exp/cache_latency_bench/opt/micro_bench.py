"""Micro-benchmark: locate the cp1_search cost and bound the prenorm-dot target.

Round 1 left cp1_search at ~21ms median (F.cosine_similarity on the resident zero-copy
matrix). This bench times candidate field-scoring on the REAL largest task bucket
(N=399, D=32768) under several CPU fp32 implementations to (a) decompose where the time
goes (pure GEMV bandwidth floor vs row-norm recompute vs full cosine) and (b) measure how
low prenorm-dot can push it, across thread counts. Throwaway research script.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/micro_bench.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

PACK = "exp/cache_latency_bench/data/opt_bench/task_pack.pt"
torch.set_grad_enabled(False)

pack = torch.load(PACK, map_location="cpu", weights_only=False)
tk = max(pack, key=lambda k: pack[k]["vision_0"].shape[0])
M0 = pack[tk]["vision_0"].contiguous()
M1 = pack[tk]["vision_1"].contiguous()
RS = pack[tk]["robot_state"].contiguous()
N, D = M0.shape
q0 = M0[0].clone()
q1 = M1[0].clone()
qr = RS[0].clone()
print(f"bucket N={N} D={D}  vision read per query (2 fields) = {2 * N * D * 4 / 1e6:.1f} MB")

# Pre-normalized matrices — in the real prenorm-dot design these are built ONCE at load.
M0n = F.normalize(M0, dim=1).contiguous()
M1n = F.normalize(M1, dim=1).contiguous()


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


# Decomposition on a single vision field.
def mv_only():
    return M0n.mv(q0)  # pure GEMV, reads M once — the memory-bandwidth floor


def rownorm_only():
    return torch.norm(M0, dim=1)  # candidate row norms, reads M once


def cosine_one():
    return F.cosine_similarity(q0.unsqueeze(0), M0)  # dot + q-norm + row-norm + divide


# Full per-query field scoring (cp1_search does 2 vision cosine + 1 robot_state l2).
def cur_full():  # Round 1 path
    s0 = F.cosine_similarity(q0.unsqueeze(0), M0)
    s1 = F.cosine_similarity(q1.unsqueeze(0), M1)
    sr = torch.norm(qr.unsqueeze(0) - RS, p=2, dim=1)
    return s0, s1, sr


def prenorm_full():  # Round 2 candidate
    q0n = q0 / q0.norm()
    q1n = q1 / q1.norm()
    s0 = M0n.mv(q0n)
    s1 = M1n.mv(q1n)
    sr = torch.norm(qr.unsqueeze(0) - RS, p=2, dim=1)
    return s0, s1, sr


cases = [
    ("mv_only (1 field, bw floor)", mv_only),
    ("rownorm_only (1 field)", rownorm_only),
    ("cosine_one (1 field)", cosine_one),
    ("cur_full 2cos+l2 [ROUND1]", cur_full),
    ("prenorm_full 2mv+l2 [ROUND2]", prenorm_full),
]

threads = (1, 2, 4, 8)
print(f"\n{'impl':34s} " + " ".join(f"t={t:<6d}" for t in threads))
for name, fn in cases:
    row = []
    for t in threads:
        torch.set_num_threads(t)
        row.append(bench(fn))
    print(f"{name:34s} " + " ".join(f"{v:7.3f}" for v in row))

# Equivalence sanity: prenorm-dot vs cosine on one field (informs Round 2's fp32 check).
torch.set_num_threads(4)
c = F.cosine_similarity(q0.unsqueeze(0), M0)
p = M0n.mv(q0 / q0.norm())
print(f"\nprenorm-dot vs cosine max|diff| (1 field) = {float((c - p).abs().max()):.3e}")
print(f"(reference: cur_full @ t=4 should track the observed ~21ms cp1_search median)")
