"""Student-policy latency benchmark on weilandserver (RTX 4090 48G / Xeon E5-2696 v4).

One policy on one device per process invocation — serial by design.

Fidelity: the measured callable IS the production sidecar's policy_fn, imported
from exp/ablation_study/sidecar_server.py (make_act_policy / make_smolvla_policy).
The timed region therefore matches the sidecar's `forward_ms` field
(sidecar_server.py:92-96) line for line: obs dict -> lerobot batch ->
predict_action_chunk -> .cpu().numpy().

ACT is loaded through the production factory with a single-task manifest
(task_0); SmolVLA from the frozen step-020000 checkpoint. Observations are
synthetic — latency is a function of tensor shapes and graph structure, not of
pixel or weight values.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np
import torch

REPO = "/home/weiland/openpi"
sys.path.insert(0, REPO)

from exp.ablation_study.sidecar_server import (  # noqa: E402
    _obs_to_lerobot_batch,
    make_act_policy,
    make_smolvla_policy,
)

# Single-model manifest: one ACT is all a per-call latency measurement needs
# (the ensemble only adds load time and residency; routing is a dict lookup).
MANIFEST = "/home/weiland/bench_latency/act_manifest_task0.json"
SMOLVLA_CKPT = ("/data/openpi/ablation_study/executor_substitution/checkpoints/"
                "libero_spatial/smolvla/checkpoints/020000/pretrained_model")


def synthetic_obs(rng: np.random.Generator, prompt: str) -> dict:
    """One LIBERO-client-shaped observation element (resize_with_pad output)."""
    return {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": prompt,
    }


def summarize(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p10": q(0.10), "p90": q(0.90), "p99": q(0.99),
            "min": s[0], "max": s[-1],
            "std": statistics.pstdev(s) if len(s) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, choices=["act", "smolvla"])
    ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    prompt = next(iter(json.loads(pathlib.Path(MANIFEST).read_text())))

    t0 = time.perf_counter()
    if args.policy == "act":
        # cwd must be the repo: manifest paths are repo-relative.
        import os
        os.chdir(REPO)
        policy_fn = make_act_policy(MANIFEST, args.device)
    else:
        policy_fn = make_smolvla_policy(SMOLVLA_CKPT, args.device)
    load_s = time.perf_counter() - t0

    rng = np.random.default_rng(20260819)
    pool = [synthetic_obs(rng, prompt) for _ in range(max(args.iters, args.warmup))]
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    for i in range(args.warmup):
        policy_fn(pool[i % len(pool)])
    sync()

    call_ms, prep_ms = [], []
    for i in range(args.iters):
        obs = pool[i % len(pool)]
        sync()
        t_a = time.perf_counter()
        out = policy_fn(obs)                    # exactly what the sidecar times
        sync()
        t_b = time.perf_counter()
        assert np.asarray(out).shape == (10, 7), np.asarray(out).shape
        call_ms.append((t_b - t_a) * 1000.0)

        # Separate reading of the obs->batch preprocessing share (not additive
        # to the above; measured on its own so the model share is attributable).
        sync()
        t_c = time.perf_counter()
        _obs_to_lerobot_batch(obs, args.device)
        sync()
        prep_ms.append((time.perf_counter() - t_c) * 1000.0)

    rec = {
        "policy": args.policy,
        "device": args.device,
        "host": platform.node(),
        "weights": "real frozen ckpt @020000 (libero_spatial)",
        "act_models_resident": 1 if args.policy == "act" else None,
        "load_seconds": round(load_s, 2),
        "warmup": args.warmup,
        "call_ms": summarize(call_ms),
        "obs_prep_ms": summarize(prep_ms),
        "raw_call_ms": [round(x, 3) for x in call_ms],
        "env": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(),
            "python": platform.python_version(),
        },
    }
    if args.device == "cuda":
        rec["gpu_mem_alloc_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    pathlib.Path(args.out).write_text(json.dumps(rec, indent=2))
    c = rec["call_ms"]
    print(f"\n== {args.policy} @ {args.device} on {rec['host']} | n={c['n']} ==")
    print(f"  policy_fn  median {c['median']:9.2f} ms | p10 {c['p10']:8.2f} | p90 {c['p90']:8.2f} "
          f"| min {c['min']:8.2f} | max {c['max']:9.2f}")
    print(f"  obs prep   median {rec['obs_prep_ms']['median']:9.3f} ms")
    print(f"  load {rec['load_seconds']}s | out: {args.out}")


if __name__ == "__main__":
    main()
