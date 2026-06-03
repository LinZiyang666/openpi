"""ROUND 1 latency runner — replay benchmark with PrebuiltMatrixBackend injected.

Reuses the standard ``ReplayHarness`` / ``H5EpisodeSource`` pipeline but injects a
prebuilt backend via ``ReplayHarness(..., components_hook=...)``. Prebuild + an explicit
warm-up pass over every bucket matrix happen before timing. Output is the same
``per_step.csv`` / ``summary.json`` as ``run.py`` so cp1_search median is directly
comparable to a baseline (un-injected) run on the same library / keybuilder / repeats.

Usage::
    uv run python exp/cache_latency_bench/opt/run_round1_latency.py \\
        --cache-config exp/cache_latency_bench/config/round1/cp1_libero10.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round1/prebuilt --repeats 1
"""

from __future__ import annotations

import dataclasses
import json
import os

import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_prebuilt
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 1 latency runner."""

    cache_config: str
    h5_dir: str
    out_dir: str
    repeats: int = 1
    device: str = "cpu"


def main(args: Args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "per_step.csv")
    out_json = os.path.join(args.out_dir, "summary.json")

    def hook(components):
        new = attach_prebuilt(components["storage"])
        # Warm-up: touch every bucket matrix once so first-query page-in does not
        # pollute the timed run (median is already robust; this is belt-and-braces).
        for mat in new._mat.values():
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
        "backend": "PrebuiltMatrixBackend(round1_stack_elim)",
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[round1_latency] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[round1_latency] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
