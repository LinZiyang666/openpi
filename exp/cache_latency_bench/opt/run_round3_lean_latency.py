"""ROUND 3 latency runner — replay benchmark with LeanSearchBackend injected.

Clone of run_round2_latency.py but injects the ROUND-3 ``LeanSearchBackend`` (scatter/
wrapper-free steady-state search) via ``attach_lean_search``, with ``set_num_threads``
pinned. Output is the same per_step.csv / summary.json shape so cp1_search median compares
directly to Round-2's ~4.7ms.

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/run_round3_lean_latency.py \\
        --cache-config exp/cache_latency_bench/config/round2/cp1_libero10.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round3/lean --repeats 1 --threads 4
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_lean_search
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 3 lean-search latency runner."""

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
        new = attach_lean_search(components["storage"])
        for mat in new._mat.values():
            mat.sum()
        captured["backend"] = new

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
        "backend": "LeanSearchBackend(round3_lean)",
        "lean_hits": backend._lean_hits,
        "lean_fallbacks": backend._lean_fallbacks,
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[round3_lean] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[round3_lean] lean_hits={backend._lean_hits} lean_fallbacks={backend._lean_fallbacks}")
    print(f"[round3_lean] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
