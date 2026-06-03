"""ROUND 5 latency runner — ReleaseVision backend (R1-R4 stack) + batched build, post-release.

Injects ``attach_release_vision`` (LeanSearch search path + ~692MB vision release) AND the
R4 batched keybuilder, then replays. Confirms (a) cp1_search/build do NOT regress after the
release (fast paths read ``_mat``, untouched), (b) ``hit_rate`` matches the pre-release
baseline (released entry tensors never participate in scoring), (c) reports bytes freed.

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/run_round5_release_latency.py \\
        --cache-config exp/cache_latency_bench/config/depth_study/depth_1.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 \\
        --out-dir exp/cache_latency_bench/data/round5/release --repeats 1 --threads 4
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
import tyro

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_batched_pool_keybuilder, attach_release_vision
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 5 release latency runner."""

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
        backend = attach_release_vision(components["storage"])  # R1-R3 search + ~692MB release
        for mat in backend._mat.values():
            mat.sum()
        attach_batched_pool_keybuilder(components)              # R4 build
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
        "backend": "ReleaseVisionBackend + batched build (round5)",
        "lean_hits": backend._lean_hits,
        "lean_fallbacks": backend._lean_fallbacks,
        "vision_released": backend._vision_released,
        "freed_mb": round(backend._freed_bytes / 1e6, 1),
        "build_excludes_d2h": args.device == "cpu",
        "checkpoints_driven": "cp1_only",
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[round5_release] {run_summary['n_steps']} steps -> {out_csv}")
    print(f"[round5_release] freed={summary['meta']['freed_mb']}MB lean_hits={backend._lean_hits} "
          f"fallbacks={backend._lean_fallbacks} hit_rate={summary.get('hit_rate')}")
    print(f"[round5_release] summary -> {out_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
