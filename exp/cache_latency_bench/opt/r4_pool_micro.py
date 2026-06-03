"""ROUND 4 build-segment micro-bench: spatial_pool variants — equivalence + latency.

The cp1 build segment (orchestrator.py:425 `key_builder.build`) for depth_1.yaml does:
  vision_0 + vision_1: each `_spatial_pool_tokens([256,2048], grid=16, pool=4)` -> [32768]
  robot_state: raw [32]
Each output then `_to_cpu_float32 = .cpu().float().contiguous()`.

This script (a) proves pool-variant numerical equivalence vs the src
`_spatial_pool_tokens` (adaptive_avg_pool2d path), bit-level where claimed, and
(b) times each variant on CPU (the bench's fake-stage1 path) and CUDA (the
production vision-encoder path, incl. the real D2H in _to_cpu_float32).

Variants timed (all produce the SAME [pool*pool*emb] flat vector layout):
  V0_src      : src _spatial_pool_tokens (adaptive_avg_pool2d, 2x permute)
  V1_avgpool  : avg_pool2d(kernel=4,stride=4) replacing adaptive (same 2x permute)
  V2_reshape  : reshape-mean, NO permute-to-NCHW, NO adaptive — block-mean via view
  V3_batched  : V2 reshape-mean but vision_0+vision_1 stacked -> one [2,256,2048] call

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_pool_micro.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

PACK = "exp/cache_latency_bench/data/opt_bench/task_pack.pt"
GRID = 16
POOL = 4
EMB = 2048
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------
# Pool variants. Input: tokens [256, emb]. Output: [pool*pool*emb] flat.
# --------------------------------------------------------------------------
def v0_src(tokens):
    """Exact src _spatial_pool_tokens — adaptive_avg_pool2d + 2x permute."""
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def v1_avgpool(tokens):
    """avg_pool2d(kernel=4,stride=4) in place of adaptive — same NCHW + 2x permute."""
    x = tokens.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(x, kernel_size=GRID // POOL, stride=GRID // POOL)
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def v2_reshape_mean(tokens):
    """Block-mean via 6D view + mean over the two block axes. No NCHW, no adaptive.

    Layout proof: src output index order is (oh, ow, c) flattened. The pooled
    cell (oh,ow) averages the 4x4 input block [oh*4:oh*4+4, ow*4:ow*4+4] over
    channel c. We reshape the [16,16,emb] grid -> [4,4, 4,4, emb] =
    [oh, ih, ow, iw, c] then mean over (ih, iw) -> [oh, ow, emb] -> flatten.
    Output index order (oh, ow, c) == src. Same arithmetic mean -> bit-equal
    up to FP reduction order.
    """
    bs = GRID // POOL  # 4
    g = tokens.reshape(POOL, bs, POOL, bs, EMB)  # [oh, ih, ow, iw, c]
    pooled = g.mean(dim=(1, 3))                  # [oh, ow, c]
    return pooled.reshape(-1)


def v3_batched(tok_pair):
    """V2 but on a stacked [2,256,emb] pair -> one kernel, returns [2, 32768]."""
    bs = GRID // POOL
    g = tok_pair.reshape(2, POOL, bs, POOL, bs, EMB)
    pooled = g.mean(dim=(2, 4))                  # [2, oh, ow, c]
    return pooled.reshape(2, -1)


def to_cpu_float32(t):
    return t.cpu().float().contiguous()


# --------------------------------------------------------------------------
# Load a real vision_0/vision_1 token sequence. task_pack stores the POOLED
# [N,32768] keys, not raw [256,2048] tokens. Reconstruct realistic raw tokens
# from the H5 fake-stage1 path so the pool input distribution is real.
# --------------------------------------------------------------------------
def load_raw_tokens(n=64):
    """Return [n,256,2048] real raw SigLIP tokens for vision_0 and vision_1."""
    import glob

    from exp.common.build_in_memory_cache_artifact import _build_fake_stage1
    import h5py

    paths = sorted(glob.glob("exp/common/data/db/libero_cache/libero_10/*.h5"))
    v0, v1 = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            for k in sorted(x for x in f.keys() if x.startswith("step_")):
                s1 = _build_fake_stage1(f[k])
                prefix = s1.prefix_embs[0]  # [prefix_len, 2048]
                v0.append(prefix[0:256].clone())
                v1.append(prefix[256:512].clone())
                if len(v0) >= n:
                    return torch.stack(v0), torch.stack(v1)
    return torch.stack(v0), torch.stack(v1)


def bench(fn, iters=200, warmup=20, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    print("loading real raw tokens from H5 ...")
    V0, V1 = load_raw_tokens(64)
    print(f"raw tokens V0={tuple(V0.shape)} V1={tuple(V1.shape)} dtype={V0.dtype}")
    t0 = V0[0]
    t1 = V1[0]

    # ---- Equivalence (CPU fp32 first) ----
    print("\n=== EQUIVALENCE (CPU fp32) vs v0_src ===")
    ref = v0_src(t0)
    for name, fn in [("v1_avgpool", v1_avgpool), ("v2_reshape_mean", v2_reshape_mean)]:
        out = fn(t0)
        bit = torch.equal(out, ref)
        maxd = float((out - ref).abs().max())
        print(f"  {name:18s} shape={tuple(out.shape)} bit_equal={bit} max|diff|={maxd:.3e}")
    pair = torch.stack([t0, t1])
    b = v3_batched(pair)
    bit0 = torch.equal(b[0], ref)
    bit1 = torch.equal(b[1], v0_src(t1))
    print(f"  {'v3_batched[0]':18s} shape={tuple(b[0].shape)} bit_equal_vs_src={bit0}")
    print(f"  {'v3_batched[1]':18s} shape={tuple(b[1].shape)} bit_equal_vs_src={bit1}")

    # ---- Equivalence on CUDA fp32 (production device) ----
    if torch.cuda.is_available():
        print("\n=== EQUIVALENCE (CUDA fp32) vs v0_src ===")
        g0 = t0.cuda()
        refg = v0_src(g0)
        for name, fn in [("v1_avgpool", v1_avgpool), ("v2_reshape_mean", v2_reshape_mean)]:
            out = fn(g0)
            bit = torch.equal(out, refg)
            maxd = float((out - refg).abs().max())
            print(f"  {name:18s} bit_equal={bit} max|diff|={maxd:.3e}")
        # cross-device: does CUDA v0_src match CPU v0_src? (informs whether prod==bench keys)
        cross = float((refg.cpu() - ref).abs().max())
        print(f"  CUDA v0_src vs CPU v0_src max|diff| = {cross:.3e} (device numerics)")

    # ---- LATENCY: CPU (bench fake-stage1 path) ----
    print("\n=== LATENCY CPU (2 vision fields + to_cpu_float32), threads sweep ===")
    print("  per-call = full build vision work for depth_1 (vision_0 + vision_1)")
    threads = (1, 2, 4, 8)
    cpu_cases = {
        "V0_src 2x(pool+cpu)": lambda: (to_cpu_float32(v0_src(t0)), to_cpu_float32(v0_src(t1))),
        "V1_avgpool 2x": lambda: (to_cpu_float32(v1_avgpool(t0)), to_cpu_float32(v1_avgpool(t1))),
        "V2_reshape 2x": lambda: (to_cpu_float32(v2_reshape_mean(t0)), to_cpu_float32(v2_reshape_mean(t1))),
        "V3_batched 1x": lambda: to_cpu_float32(v3_batched(pair)),
    }
    print(f"\n  {'impl':24s} " + " ".join(f"t={t:<6d}" for t in threads))
    for name, fn in cpu_cases.items():
        row = []
        for t in threads:
            torch.set_num_threads(t)
            row.append(bench(fn))
        print(f"  {name:24s} " + " ".join(f"{v:7.4f}" for v in row))
    torch.set_num_threads(4)

    # pool-only (exclude cpu/contiguous, isolate pool kernel cost on CPU)
    print("\n  --- pool-only (no to_cpu_float32), t=4 ---")
    for name, fn in [("v0_src", v0_src), ("v1_avgpool", v1_avgpool), ("v2_reshape", v2_reshape_mean)]:
        ms = bench(lambda: (fn(t0), fn(t1)))
        print(f"    {name:14s} 2 fields = {ms:.4f} ms")

    # ---- LATENCY: CUDA (production path incl. real D2H) ----
    if torch.cuda.is_available():
        print("\n=== LATENCY CUDA (production: GPU pool + real D2H in to_cpu_float32) ===")
        g0, g1 = t0.cuda(), t1.cuda()
        gpair = torch.stack([g0, g1])
        torch.cuda.synchronize()
        cuda_cases = {
            "V0_src 2x(pool+D2H)": lambda: (to_cpu_float32(v0_src(g0)), to_cpu_float32(v0_src(g1))),
            "V1_avgpool 2x": lambda: (to_cpu_float32(v1_avgpool(g0)), to_cpu_float32(v1_avgpool(g1))),
            "V2_reshape 2x": lambda: (to_cpu_float32(v2_reshape_mean(g0)), to_cpu_float32(v2_reshape_mean(g1))),
            "V3_batched 1x(1 D2H)": lambda: to_cpu_float32(v3_batched(gpair)),
        }
        for name, fn in cuda_cases.items():
            ms = bench(fn, cuda=True)
            print(f"  {name:24s} {ms:.4f} ms")
        # GPU pool-only (no D2H) to expose what fraction is transfer
        print("\n  --- GPU pool-only (no D2H), keep on device ---")
        for name, fn in [("v0_src", v0_src), ("v1_avgpool", v1_avgpool), ("v2_reshape", v2_reshape_mean)]:
            ms = bench(lambda: (fn(g0), fn(g1)), cuda=True)
            print(f"    {name:14s} 2 fields = {ms:.4f} ms")
        # D2H-only floor: how long to move 2x[32768] fp32 GPU->CPU
        out_g = v2_reshape_mean(g0)
        ms_d2h = bench(lambda: (out_g.cpu(), out_g.cpu()), cuda=True)
        print(f"\n  D2H floor 2x[32768] fp32 = {ms_d2h:.4f} ms (irreducible production cost)")


if __name__ == "__main__":
    main()
