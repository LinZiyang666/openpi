"""ROUND 4 end-to-end equivalence + latency for the matmul spatial-pool builder.

Drives the REAL ReplayHarness on a SUBSET of episodes twice (stock keybuilder
vs matmul keybuilder injected via components_hook) and checks, per step:
  (1) query_keys bit-identical (torch.equal on vision_0 / vision_1 / robot_state),
  (2) hit_type + top_score identical (verdict / warm-tier unchanged),
then prints the in-situ cp1_build median for both (build speedup in the real loop).

Subset (not full 2640 steps) per the round's equivalence-on-subset rule.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/verify_matmul_pool_e2e.py \\
        --cache-config exp/cache_latency_bench/config/depth_study/depth_1.yaml \\
        --h5-dir exp/common/data/db/libero_cache/libero_10 --max-episodes 4 --threads 4
"""
from __future__ import annotations

import dataclasses
import statistics as st

import torch
import tyro

from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.opt.inject import attach_matmul_pool_keybuilder
from exp.cache_latency_bench.replay import ReplayHarness


@dataclasses.dataclass
class Args:
    cache_config: str
    h5_dir: str
    max_episodes: int = 4
    threads: int = 4


def _drive_subset(harness: ReplayHarness, source: H5EpisodeSource, max_eps: int):
    """Replay a subset; return per-step (keys, hit_type, score, build_ms) records."""
    orch = harness._orchestrator
    timer = harness._timer
    import time

    recs = []
    orch.on_task_begin()
    try:
        for ei, ep in enumerate(source.iter_episodes()):
            if ei >= max_eps:
                break
            orch.on_episode_start(ep.task_key, str(ep.episode_id))
            for step in ep.steps():
                timer.on_task_begin()
                t0 = time.perf_counter()
                res = orch.check(CheckpointID.CP1, stage1=step.fake_stage1)
                _ = (time.perf_counter() - t0) * 1000.0
                stats = timer.summary(task_only=True)
                build_ms = stats["cp1_build"].mean_ms if "cp1_build" in stats else float("nan")
                # snapshot query keys (clone to survive clear())
                keys = {k: v.clone() for k, v in (res.query_keys or {}).items()}
                recs.append((keys, res.hit_type, res.score, build_ms))
                action = res.payload.action_chunk if (res.hit_type == HitType.FULL_HIT and res.payload) else step.clean_action
                orch.broadcast_action(action)
                if res.query_keys is not None:
                    orch.buffer_for_write(res.query_keys, action)
                orch.clear()
            orch.on_episode_end()
    finally:
        orch.on_task_end()
    return recs


def main(args: Args) -> None:
    torch.set_num_threads(args.threads)
    source = H5EpisodeSource(args.h5_dir)

    # Stock run.
    h_stock = ReplayHarness(args.cache_config, device="cpu")
    rec_stock = _drive_subset(h_stock, source, args.max_episodes)

    # Matmul run (inject keybuilder via hook).
    def hook(components):
        attach_matmul_pool_keybuilder(components)

    h_mm = ReplayHarness(args.cache_config, device="cpu", components_hook=hook)
    rec_mm = _drive_subset(h_mm, source, args.max_episodes)

    assert len(rec_stock) == len(rec_mm), f"step count mismatch {len(rec_stock)} vs {len(rec_mm)}"
    n = len(rec_stock)

    key_exact = 0
    key_fields_checked = 0
    worst_diff = 0.0
    verdict_match = 0
    score_max_diff = 0.0
    for (ks, ht_s, sc_s, _), (km, ht_m, sc_m, _) in zip(rec_stock, rec_mm):
        fields_ok = True
        for fld in ks:
            key_fields_checked += 1
            a, b = ks[fld], km[fld]
            if torch.equal(a, b):
                pass
            else:
                fields_ok = False
                worst_diff = max(worst_diff, (a - b).abs().max().item())
        key_exact += int(fields_ok)
        if ht_s == ht_m:
            verdict_match += 1
        if sc_s is not None and sc_m is not None:
            score_max_diff = max(score_max_diff, abs(sc_s - sc_m))

    bs = [r[3] for r in rec_stock if r[3] == r[3]]
    bm = [r[3] for r in rec_mm if r[3] == r[3]]

    print("=" * 70)
    print(f"E2E EQUIVALENCE on {n} steps ({args.max_episodes} episodes), threads={args.threads}")
    print("=" * 70)
    print(f"  query_keys bit-identical steps : {key_exact}/{n}  "
          f"(fields checked={key_fields_checked}, worst max|diff|={worst_diff:.3e})")
    print(f"  hit_type / verdict match       : {verdict_match}/{n}")
    print(f"  top_score max|diff|            : {score_max_diff:.3e}")
    print(f"\n  cp1_build median  stock={st.median(bs):.4f}ms  matmul={st.median(bm):.4f}ms  "
          f"speedup={st.median(bs)/st.median(bm):.2f}x")
    print(f"  cp1_build mean    stock={st.mean(bs):.4f}ms  matmul={st.mean(bm):.4f}ms")
    ok = (key_exact == n) and (verdict_match == n)
    print(f"\n  RESULT: {'PASS — bit-identical keys + verdicts' if ok else 'FAIL — divergence detected'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
