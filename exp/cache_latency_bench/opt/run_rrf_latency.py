"""ROUND-RRF latency runner — stock / prebuilt(R1) / prenorm(R2) on weighted_rrf.

One runner with ``--backend {stock,prebuilt,prenorm}`` measures the weighted_rrf cp1_search
segment under: stock src InMemoryBackend (baseline), PrebuiltMatrix (R1, bit-equal), and
PrenormDot (R2, GEMV). Same per_step.csv / summary.json shape as the weighted_sum runners,
on the SAME libero_10 library — so RRF numbers are directly comparable to weighted_sum's.

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/run_rrf_latency.py \\
        --cache-config exp/cache_latency_bench/config/round_rrf/cp1_libero10_rrf_kin.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round_rrf/prenorm --backend prenorm --threads 4
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import (
    attach_batched_pool_keybuilder,
    attach_lean_rrf,
    attach_prebuilt,
    attach_prenorm_dot,
    attach_rrf_release,
)
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the weighted_rrf latency runner."""

    cache_config: str
    h5_dir: str
    out_dir: str
    backend: str = "stock"   # "stock" | "prebuilt" | "prenorm" | "lean_rrf" | "rrf_release"
    repeats: int = 1
    device: str = "cpu"
    threads: int = 4
    batched: bool = False    # also attach the R4 batched-avgpool keybuilder (build optimization)


def main(args: Args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "per_step.csv")
    out_json = os.path.join(args.out_dir, "summary.json")
    torch.set_num_threads(args.threads)

    def hook(components):
        if args.backend == "prebuilt":
            attach_prebuilt(components["storage"])
        elif args.backend == "prenorm":
            b = attach_prenorm_dot(components["storage"])
            for m in b._mat.values():
                m.sum()
        elif args.backend == "lean_rrf":
            b = attach_lean_rrf(components["storage"])
            for m in b._mat.values():
                m.sum()
        elif args.backend == "rrf_release":
            b = attach_rrf_release(components["storage"])
            for m in b._mat.values():
                m.sum()
        if args.batched:
            attach_batched_pool_keybuilder(components)  # R4 build optimization (orthogonal)

    components_hook = None if (args.backend == "stock" and not args.batched) else hook
    harness = ReplayHarness(args.cache_config, device=args.device, components_hook=components_hook)
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
        "backend": f"rrf_{args.backend}",
        "fusion": "weighted_rrf",
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    s = summary["probes"]["cp1_search_ms"]["ALL"]
    print(f"[rrf_latency_{args.backend}] {run_summary['n_steps']} steps; "
          f"cp1_search median={s['median']:.3f} p95={s['p95']:.3f} -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
