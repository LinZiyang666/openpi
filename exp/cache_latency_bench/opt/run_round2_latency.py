"""ROUND 2 latency runner — replay benchmark with PrenormDotBackend injected.

Clone of run_round1_latency.py but injects the ROUND-2 prenorm-dot backend via
``attach_prenorm_dot`` and pins ``torch.set_num_threads(args.threads)`` BEFORE the run, so
the shipped latency and the 0-flip equivalence claim share one thread count. Warm-up
touches every normed bucket once. Output is the same per_step.csv / summary.json shape as
Round 1, so cp1_search median compares directly to Round-1's 21ms.

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/run_round2_latency.py \\
        --cache-config exp/cache_latency_bench/config/round2/cp1_libero10.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round2/prenorm --repeats 1 --threads 4
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_prenorm_dot
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 2 latency runner."""

    cache_config: str
    h5_dir: str
    out_dir: str
    repeats: int = 1
    device: str = "cpu"
    # Pinned CPU thread count (shared with the equivalence harness).
    threads: int = 4
    # Optional rescore lever (0 = off, the default shipped path).
    rescore_top_k: int = 0


def main(args: Args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "per_step.csv")
    out_json = os.path.join(args.out_dir, "summary.json")
    torch.set_num_threads(args.threads)

    kw = {}
    if args.rescore_top_k > 0:
        kw = {"rescore_top_k": args.rescore_top_k, "keep_raw": True}

    def hook(components):
        new = attach_prenorm_dot(components["storage"], **kw)
        for mat in new._mat.values():  # warm-up: page-in every normed bucket
            mat.sum()

    harness = ReplayHarness(args.cache_config, device=args.device, components_hook=hook)
    source = H5EpisodeSource(args.h5_dir)
    run_summary = harness.run(source, repeats=args.repeats, out_csv=out_csv)

    summary = summarize(out_csv)
    summary["run"] = run_summary
    summary["meta"] = {
        "cache_config": args.cache_config,
        "h5_dir": args.h5_dir,
        "repeats": args.repeats,
        "device": args.device,
        "threads": args.threads,
        "rescore_top_k": args.rescore_top_k,
        "backend": "PrenormDotBackend(round2_prenorm_dot)",
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[round2_latency] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[round2_latency] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
