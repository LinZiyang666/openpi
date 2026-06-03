"""ROUND 4 latency runner — LeanSearch backend + batched-avgpool keybuilder.

Stacks the R1-R3 search optimizations (``attach_lean_search``) AND the R4 build optimization
(``attach_batched_pool_keybuilder``) in one replay, so the measured cp1_build segment is the
optimized build in the lean-search context. Same per_step.csv / summary.json shape as the
prior runners; cp1_build_ms compares directly to the round3_lean baseline (~1.28ms).

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/run_round4_pool_latency.py \\
        --cache-config exp/cache_latency_bench/config/depth_study/depth_1.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round4/batched --repeats 1 --threads 4
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_batched_pool_keybuilder, attach_lean_search
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 4 build+search latency runner."""

    cache_config: str
    h5_dir: str
    out_dir: str
    repeats: int = 1
    device: str = "cpu"
    threads: int = 4


def main(args: Args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "per_step.csv")
    out_json = os.path.join(args.out_dir, "summary.json")
    torch.set_num_threads(args.threads)

    captured = {}

    def hook(components):
        backend = attach_lean_search(components["storage"])  # R1-R3 search
        for mat in backend._mat.values():
            mat.sum()
        attach_batched_pool_keybuilder(components)            # R4 build
        captured["backend"] = backend

    harness = ReplayHarness(args.cache_config, device=args.device, components_hook=hook)
    source = H5EpisodeSource(args.h5_dir)
    run_summary = harness.run(source, repeats=args.repeats, out_csv=out_csv)

    backend = captured["backend"]
    summary = summarize(out_csv)
    summary["run"] = run_summary
    summary["meta"] = {
        "cache_config": args.cache_config,
        "h5_dir": args.h5_dir,
        "repeats": args.repeats,
        "device": args.device,
        "threads": args.threads,
        "backend": "LeanSearchBackend + CP1SpatialPool16BatchedKeyBuilder (round4)",
        "lean_hits": backend._lean_hits,
        "lean_fallbacks": backend._lean_fallbacks,
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[round4_pool] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[round4_pool] lean_hits={backend._lean_hits} fallbacks={backend._lean_fallbacks}")
    print(f"[round4_pool] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
