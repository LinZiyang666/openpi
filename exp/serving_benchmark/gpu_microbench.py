"""Mode 0 — GPU-direct microbenchmark (audit report §B.3 #4 alignment).

Bypasses the server / WebSocket / wrapper / cache layers and calls
``policy._model.sample_actions(device, batched_obs, noise)`` directly,
sweeping ``batch_size ∈ {1, 2, 4, 8, 16, 32}`` to surface the theoretical
GPU throughput / latency curve. The result is the upper bound any
batching-coordinator design can approach.

Usage:
    python -m exp.serving_benchmark.gpu_microbench \
        --checkpoint <path> \
        --batch-sizes 1,2,4,8,16,32 \
        --iters 100 --warmup 10 \
        --output data/<run_id>/gpu_microbench.csv

Output CSV columns:
    batch_size, iteration, latency_ms, throughput_rps
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

logger = logging.getLogger("serving_benchmark.gpu_microbench")


def _build_dummy_obs(batch_size: int, device, model) -> tuple:
    """Build (batched_obs, noise) for the model's ``sample_actions`` signature.

    Shapes are derived from ``model.config`` so the same script works across
    Pi0 / Pi0.5 / future variants.
    """
    import torch

    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    state_dim = action_dim  # pi0_pytorch states are action_dim-sized
    prefix_len = 200  # typical short prefix; mock for shape only

    obs = {
        "state": torch.zeros(batch_size, state_dim, device=device),
        "image": torch.zeros(batch_size, 3, 224, 224, device=device),
        "tokenized_prompt": torch.zeros(batch_size, prefix_len, dtype=torch.int64, device=device),
        "tokenized_prompt_mask": torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device),
    }
    noise = torch.zeros(batch_size, action_horizon, action_dim, device=device)
    return obs, noise


def run_gpu_microbench(
    model, device, batch_sizes: list[int], iters: int = 100, warmup: int = 10,
    output_csv: str | Path | None = None,
) -> dict:
    """Run the microbench. Returns a per-batch-size summary dict."""
    import torch

    results: dict[int, dict] = {}
    rows = []

    for bs in batch_sizes:
        obs, noise = _build_dummy_obs(bs, device, model)
        latencies_ms = []

        # Warmup — ignore early measurements for compile / cache warm-up.
        for _ in range(warmup):
            _ = model.sample_actions(device, obs, noise)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for it in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.sample_actions(device, obs, noise)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)
            rows.append((bs, it, latency_ms, bs * 1000.0 / latency_ms))

        latencies_ms.sort()
        results[bs] = {
            "p50_ms": latencies_ms[len(latencies_ms) // 2],
            "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)],
            "min_ms": latencies_ms[0],
            "max_ms": latencies_ms[-1],
            "throughput_rps_p50": bs * 1000.0 / latencies_ms[len(latencies_ms) // 2],
        }
        logger.info("batch_size=%d  p50=%.2fms  p95=%.2fms  rps_p50=%.1f",
                    bs, results[bs]["p50_ms"], results[bs]["p95_ms"],
                    results[bs]["throughput_rps_p50"])

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch_size", "iteration", "latency_ms", "throughput_rps"])
            for r in rows:
                w.writerow(r)

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Policy checkpoint dir")
    p.add_argument("--config-name", default="pi05_libero",
                   help="Training config name (matches openpi.training.config)")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> None:
    import torch

    from openpi.policies.policy_config import create_trained_policy
    from openpi.training import config as _config

    args = _parse_args()
    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]
    cfg = _config.get_config(args.config_name)
    policy = create_trained_policy(cfg, args.checkpoint)
    model = policy._model
    device = next(model.parameters()).device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — microbench will measure CPU forward latency only.")
    results = run_gpu_microbench(
        model, device, batch_sizes,
        iters=args.iters, warmup=args.warmup,
        output_csv=args.output or None,
    )
    print("=== GPU microbench summary ===")
    for bs, r in results.items():
        print(f"  batch={bs:>3d}  p50={r['p50_ms']:.2f}ms  "
              f"rps_p50={r['throughput_rps_p50']:.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
