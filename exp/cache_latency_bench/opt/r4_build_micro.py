"""ROUND 4 build-segment micro-bench + equivalence probe (THROWAWAY research script).

Targets the build() segment (1.33ms median / 21% of cp1_total after R1-R3 squeezed
search to 3.7ms). The active build per step (depth_1.yaml) is:
  vision_0 spatial_pool([256,2048]->[32768]) + vision_1 spatial_pool(...) + robot_state raw[32]
then _to_cpu_float32 on each.

This script (a) times the original _spatial_pool_tokens vs four optimized variants and
(b) checks bit/numeric equivalence of every variant against the original, on REAL token
inputs reconstructed from the same HDF5 the library was built from. It runs BOTH:
  - CPU path (the bench's fake-stage1 reality: inputs already CPU float32)
  - GPU path (production reality: vision encoder output on GPU, _to_cpu_float32 is real D2H)

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_build_micro.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.manual_seed(0)

GRID = 16
POOL = 4
EMB = 2048
N_TOK = GRID * GRID  # 256

# --------------------------------------------------------------------------
# Variants. v_orig is a byte-for-byte copy of src _spatial_pool_tokens.
# --------------------------------------------------------------------------


def v_orig(tokens: torch.Tensor) -> torch.Tensor:
    """src/openpi/cache/components/key_builder.py:192 (verbatim)."""
    emb_dim = tokens.shape[1]
    x = tokens.reshape(GRID, GRID, emb_dim).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def v_avgpool(tokens: torch.Tensor) -> torch.Tensor:
    """Candidate 1: fixed avg_pool2d(kernel=4,stride=4) replaces adaptive (16//4=4 exact)."""
    emb_dim = tokens.shape[1]
    x = tokens.reshape(GRID, GRID, emb_dim).permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(x, kernel_size=POOL, stride=POOL)
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def v_reshape_mean(tokens: torch.Tensor) -> torch.Tensor:
    """Candidate 2: no permute/pool kernel. Block-average via reshape + mean.

    Output ordering MUST match v_orig: orig flattens [pool,pool,emb] (out-row, out-col, emb).
    Each output cell (po,pc) = mean over the 4x4 input block tokens[po*4:po*4+4, pc*4:pc*4+4].
    tokens is [256,2048] row-major over (gr, gc): index = gr*16+gc.
      reshape -> [4(po),4(ir),4(pc),4(ic),emb]  (gr=po*4+ir, gc=pc*4+ic)
      mean over (ir,ic) -> [po,pc,emb] -> reshape(-1) == orig's [pool,pool,emb] flatten.
    """
    emb_dim = tokens.shape[1]
    x = tokens.reshape(POOL, GRID // POOL, POOL, GRID // POOL, emb_dim)
    return x.mean(dim=(1, 3)).reshape(-1)


def v_avgpool_noperm_in(tokens: torch.Tensor) -> torch.Tensor:
    """Candidate 2b: keep avg_pool2d but feed channels-last via a single contiguous block.

    Treat emb as batch, (16,16) as a single HxW image: x=[emb,1,16,16]. Pool to [emb,1,4,4].
    Output flatten order must be [po,pc,emb]; pooled is [emb,po,pc] -> permute(1,2,0).reshape.
    Same number of permutes as orig but the input reshape is a view (no permute before pool).
    """
    emb_dim = tokens.shape[1]
    x = tokens.t().reshape(emb_dim, 1, GRID, GRID)  # [emb,1,16,16]
    pooled = F.avg_pool2d(x, kernel_size=POOL, stride=POOL)  # [emb,1,4,4]
    return pooled.reshape(emb_dim, POOL * POOL).t().reshape(-1)  # [po*pc, emb] flat == [po,pc,emb]


def make_batched(fn):
    """Candidate 4 wrapper: stack vision_0+vision_1 -> one call over [2,256,2048]."""

    def batched(tok0, tok1):
        x = torch.stack([tok0, tok1], dim=0)  # [2,256,2048]
        emb_dim = x.shape[2]
        # reshape-mean batched
        xr = x.reshape(2, POOL, GRID // POOL, POOL, GRID // POOL, emb_dim)
        out = xr.mean(dim=(2, 4))  # [2,po,pc,emb]
        return out[0].reshape(-1), out[1].reshape(-1)

    return batched


# --------------------------------------------------------------------------
# Equivalence: load real vision tokens from one HDF5 episode (bench reality).
# --------------------------------------------------------------------------


def load_real_tokens(n_steps: int = 64):
    """Reconstruct [256,2048] vision_0/vision_1 token blocks from real fake-stage1."""
    import glob
    import os

    import h5py

    from exp.common.build_in_memory_cache_artifact import _build_fake_stage1

    h5_dir = None
    for cand in (
        "exp/common/data/db/libero_cache/libero_10",
        "exp/common/data/db/libero_cache/libero_spatial",
    ):
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.h5")):
            h5_dir = cand
            break
    if h5_dir is None:
        return None
    path = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))[0]
    out = []
    with h5py.File(path, "r") as f:
        names = sorted((k for k in f.keys() if k.startswith("step_")),
                       key=lambda s: int(s.split("_")[-1]))
        for name in names[:n_steps]:
            fs = _build_fake_stage1(f[name])
            prefix = fs.prefix_embs[0]  # [prefix_len, 2048]
            tok0 = prefix[0:256].contiguous()
            tok1 = prefix[256:512].contiguous()
            out.append((tok0, tok1))
    return out, h5_dir, path


def equiv_report(variants, tokens_list, label):
    print(f"\n=== EQUIVALENCE [{label}] over {len(tokens_list)} real steps ===")
    ref_name = "v_orig"
    for name, fn in variants.items():
        if name == ref_name:
            continue
        max_abs = 0.0
        max_bits = 0  # count of elements that differ bitwise
        total = 0
        for tok0, tok1 in tokens_list:
            for tok in (tok0, tok1):
                r = v_orig(tok)
                c = fn(tok)
                d = (r - c).abs().max().item()
                max_abs = max(max_abs, d)
                bit_ne = (r != c).sum().item()
                max_bits = max(max_bits, bit_ne)
                total += r.numel()
        print(f"  {name:24s} max|diff|={max_abs:.3e}  max bit-differing elems={max_bits}/{r.numel()}"
              f"  {'BIT-IDENTICAL' if max_bits == 0 else 'fp-diff'}")


def equiv_batched(tokens_list, label):
    print(f"\n=== EQUIVALENCE [batched candidate 4, {label}] ===")
    bf = make_batched(None)
    max_abs = 0.0
    max_bits = 0
    for tok0, tok1 in tokens_list:
        r0, r1 = v_orig(tok0), v_orig(tok1)
        c0, c1 = bf(tok0, tok1)
        for r, c in ((r0, c0), (r1, c1)):
            max_abs = max(max_abs, (r - c).abs().max().item())
            max_bits = max(max_bits, (r != c).sum().item())
    print(f"  batched_reshape_mean     max|diff|={max_abs:.3e}  max bit-differing={max_bits}"
          f"  {'BIT-IDENTICAL' if max_bits == 0 else 'fp-diff'}")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def bench(fn, arg, iters=300, warmup=30):
    for _ in range(warmup):
        fn(arg)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(arg)
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_full_build_cpu(reduce_fn, to_cpu_fn, tok0, tok1, rs, iters=300, warmup=30):
    """Time the active build: 2 vision reduces + 3 _to_cpu_float32 (CPU bench reality)."""

    def step():
        k0 = to_cpu_fn(reduce_fn(tok0))
        k1 = to_cpu_fn(reduce_fn(tok1))
        kr = to_cpu_fn(rs)
        return k0, k1, kr

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - t0) / iters * 1000.0


def to_cpu_orig(t):
    """src _to_cpu_float32."""
    return t.cpu().float().contiguous()


def to_cpu_lean(t):
    """bench-only: inputs already CPU float32 contiguous; ensure contiguous cheaply."""
    return t.contiguous()


def main():
    loaded = load_real_tokens()
    if loaded is None:
        print("[r4] no real H5 found; using random tokens for timing only (equiv skipped)")
        tokens_list = [(torch.randn(N_TOK, EMB), torch.randn(N_TOK, EMB)) for _ in range(32)]
        have_real = False
    else:
        tokens_list, h5_dir, path = loaded
        print(f"[r4] loaded {len(tokens_list)} real steps from {path}")
        have_real = True

    variants = {
        "v_orig": v_orig,
        "v_avgpool": v_avgpool,
        "v_reshape_mean": v_reshape_mean,
        "v_avgpool_noperm_in": v_avgpool_noperm_in,
    }

    # ---- Equivalence (CPU) ----
    equiv_report(variants, tokens_list, "CPU fp32")
    equiv_batched(tokens_list, "CPU fp32")

    # ---- Equivalence (GPU) if available: does adaptive vs avg differ on GPU kernels? ----
    if torch.cuda.is_available() and have_real:
        gpu_list = [(t0.cuda(), t1.cuda()) for t0, t1 in tokens_list[:16]]
        equiv_report(variants, gpu_list, "GPU fp32")

    # ---- Timing: single reduce on the [256,2048] block ----
    tok = tokens_list[0][0]
    print(f"\n=== TIMING single _reduce_vision [256,2048]->[32768] (CPU) ===")
    threads = (1, 2, 4)
    print(f"{'variant':24s} " + " ".join(f"t={t:<6d}" for t in threads))
    for name, fn in variants.items():
        row = []
        for t in threads:
            torch.set_num_threads(t)
            row.append(bench(fn, tok))
        print(f"{name:24s} " + " ".join(f"{v:7.4f}" for v in row))

    # ---- Timing: full active build (2 vision + robot_state, with _to_cpu) ----
    tok0, tok1 = tokens_list[0]
    rs = torch.randn(32)
    torch.set_num_threads(4)
    print(f"\n=== TIMING full active build @ t=4 (2 vision reduce + 3 _to_cpu_float32) ===")
    configs = [
        ("orig reduce + orig _to_cpu", v_orig, to_cpu_orig),
        ("orig reduce + lean _to_cpu", v_orig, to_cpu_lean),
        ("avgpool + orig _to_cpu", v_avgpool, to_cpu_orig),
        ("avgpool + lean _to_cpu", v_avgpool, to_cpu_lean),
        ("reshape_mean + orig _to_cpu", v_reshape_mean, to_cpu_orig),
        ("reshape_mean + lean _to_cpu", v_reshape_mean, to_cpu_lean),
    ]
    for name, rfn, cfn in configs:
        ms = bench_full_build_cpu(rfn, cfn, tok0, tok1, rs)
        print(f"  {name:34s} {ms:7.4f} ms")

    # ---- Timing: batched candidate 4 ----
    bf = make_batched(None)

    def batched_build():
        k0r, k1r = bf(tok0, tok1)
        return to_cpu_lean(k0r), to_cpu_lean(k1r), to_cpu_lean(rs)

    for _ in range(30):
        batched_build()
    t0 = time.perf_counter()
    for _ in range(300):
        batched_build()
    bms = (time.perf_counter() - t0) / 300 * 1000.0
    print(f"  {'batched reshape_mean + lean':34s} {bms:7.4f} ms")

    # ---- _to_cpu_float32 component cost on CPU (is .cpu()/.float()/.contiguous() free?) ----
    print(f"\n=== _to_cpu_float32 component cost on a CPU [32768] tensor @ t=4 ===")
    big = v_orig(tok0)  # CPU float32 contiguous already
    for name, fn in [
        (".cpu().float().contiguous()", lambda t: t.cpu().float().contiguous()),
        (".float().contiguous()", lambda t: t.float().contiguous()),
        (".contiguous()", lambda t: t.contiguous()),
        ("identity (already ok)", lambda t: t),
    ]:
        print(f"  {name:30s} {bench(fn, big):7.4f} ms")


if __name__ == "__main__":
    main()
