"""CLI entry for the cache latency replay benchmark.

Example::

    uv run exp/cache_latency_bench/run.py \\
        --cache-config exp/.../some_eval.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/run0 \\
        --repeats 1

Outputs ``<out-dir>/per_step.csv`` (one row per request, per-segment latency)
and ``<out-dir>/summary.json`` (per-segment x hit_type aggregates). The cache
system is assembled from ``--cache-config`` exactly like the server; no model
is loaded. See ``logs/cache_latency_bench_plan.log.md`` for known deviations
(notably: CPU mode excludes the build-segment D2H; only CP1 is driven).
"""

from __future__ import annotations

import dataclasses
import json
import os

import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the cache latency replay benchmark."""

    # Real cache YAML — defines how the cache system is assembled and works.
    cache_config: str
    # Directory of real per-episode HDF5 trajectory files (one file = one episode).
    h5_dir: str
    # Output root directory (per_step.csv + summary.json written here).
    out_dir: str
    # How many times to replay the whole H5 directory (default 1).
    repeats: int = 1
    # Device for stage1 tensors. Only "cpu" is supported (plan D1).
    device: str = "cpu"


def main(args: Args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "per_step.csv")
    out_json = os.path.join(args.out_dir, "summary.json")

    harness = ReplayHarness(args.cache_config, device=args.device)
    source = H5EpisodeSource(args.h5_dir)
    run_summary = harness.run(source, repeats=args.repeats, out_csv=out_csv)

    summary = summarize(out_csv)
    summary["run"] = run_summary
    summary["meta"] = {
        "cache_config": args.cache_config,
        "h5_dir": args.h5_dir,
        "repeats": args.repeats,
        "device": args.device,
        # Known-deviation flags (plan §6): build segment lacks GPU->CPU D2H on CPU,
        # only CP1 is driven (CP3 segment not measured).
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[cache_latency_bench] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[cache_latency_bench] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
