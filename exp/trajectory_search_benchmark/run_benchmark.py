"""Trajectory search rewrite benchmark (plan §7.4 / §10 #4).

Runs the matrix:

  entries ∈ {5_000, 20_000}
  depth   ∈ {3, 5}
  fusion  ∈ {weighted_rrf, weighted_score_sum}
  modes   = (legacy, new_memo_off, new_memo_on)

For each cell we report the wall-clock median over 10 single-step search
calls. Then we run the per-step score-memo curve: the same trajectory
issued for 5 successive steps with search_session_id+trajectory_query_ids
engaged, comparing each step's wall-clock against the memo-off and
legacy baselines.

Run locally:

    uv run python -m exp.trajectory_search_benchmark.run_benchmark

The script prints two markdown tables to stdout. CI does not invoke
this script — capture the output to `analysis/results.md` (per the
canonical experiment artifact layout, see
`docs/experiments/artifact_layout.md` §2) when shipping benchmark
evidence.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _build_backend(n_entries: int, dim: int = 32, seed: int = 0) -> InMemoryBackend:
    """Insert a single linked chain of n entries with random query vectors."""
    g = torch.Generator().manual_seed(seed)
    backend = InMemoryBackend({"robot_state": dim})
    for i in range(n_entries):
        vec = torch.randn(dim, generator=g).float()
        backend.insert(CacheEntry(
            id=f"e:{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": vec},
            payload=CachePayload(action_chunk=torch.zeros(50, 32)),
            prev_ids=[f"e:{i - 1}"] if i > 0 else [],
            trajectory_id="traj",
            step_idx=i,
        ))
    return backend


def _build_spec(
    *,
    dim: int,
    depth: int,
    fusion_method: str,
    search_session_id: Optional[str] = None,
    trajectory_query_ids: Optional[list[int]] = None,
    seed: int = 1,
) -> QuerySpec:
    g = torch.Generator().manual_seed(seed)
    qkey = torch.randn(dim, generator=g).float()
    history = [
        {"robot_state": torch.randn(dim, generator=g).float()}
        for _ in range(depth)
    ]
    weights = [1.0 / depth] * depth
    return QuerySpec(
        query_keys={"robot_state": qkey},
        top_k=10,
        checkpoint_id=CheckpointID.CP1,
        trajectory_history=history,
        trajectory_weights=weights,
        fusion_method=fusion_method,
        fusion_weights={"robot_state": 1.0},
        search_session_id=search_session_id,
        trajectory_query_ids=trajectory_query_ids,
    )


def _median_ms(fn, *, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_matrix(repeats: int) -> list[dict]:
    """Single-step median runtime across the matrix."""
    rows: list[dict] = []
    for entries in (5_000, 20_000):
        backend = _build_backend(entries)
        for depth in (3, 5):
            for fusion in ("weighted_rrf", "weighted_score_sum"):
                spec_off = _build_spec(
                    dim=32, depth=depth, fusion_method=fusion,
                )

                def call_legacy():
                    with backend.force_legacy_path():
                        backend.search(spec_off)

                def call_new_off():
                    backend.search(spec_off)

                # Memo-on: same trajectory, sid registered, qids fixed.
                sid = f"bench-{entries}-{depth}-{fusion}"
                spec_on = _build_spec(
                    dim=32, depth=depth, fusion_method=fusion,
                    search_session_id=sid,
                    trajectory_query_ids=list(range(depth)),
                )
                backend.open_search_session(sid)
                # Warm the memo once so memo-on rows reflect steady-state hits.
                backend.search(spec_on)

                def call_new_on():
                    backend.search(spec_on)

                t_legacy = _median_ms(call_legacy, repeats=repeats)
                t_off = _median_ms(call_new_off, repeats=repeats)
                t_on = _median_ms(call_new_on, repeats=repeats)
                backend.close_search_session(sid)

                rows.append({
                    "entries": entries,
                    "depth": depth,
                    "fusion": fusion,
                    "legacy_ms": t_legacy,
                    "new_memo_off_ms": t_off,
                    "new_memo_on_ms": t_on,
                })
    return rows


def bench_step_curve(entries: int = 5_000, depth: int = 3,
                     fusion: str = "weighted_score_sum",
                     n_steps: int = 5, repeats: int = 5,
                     dim: int = 32) -> list[dict]:
    """Per-step wall-clock for n_steps successive trajectory searches.

    Build a stable per-qid vector bank up front: `qid_to_vec[i]` is fixed
    for the whole loop. Step *t* slides the depth-sized window through
    the bank (`history_qids = [t-1, t-2, t-3]` newest-first), so the
    same qid maps to the same tensor on every step where it appears.
    This is the property the score memo is supposed to exploit; the
    earlier seed-by-step formulation broke qid-to-vector identity and
    measured cache hits against a different tensor than the one that
    populated the slot.
    """
    backend = _build_backend(entries)
    rows: list[dict] = []

    # Stable per-qid vector bank. The bank is used for both query and
    # history positions; over n_steps, qids 0..n_steps+depth-1 each map
    # to a distinct, fixed unit-randomised vector.
    bank_size = n_steps + depth
    g = torch.Generator().manual_seed(2026)
    qid_to_vec = [torch.randn(dim, generator=g).float() for _ in range(bank_size)]

    sid = "step-curve"
    backend.open_search_session(sid)
    try:
        for step in range(n_steps):
            # qid for "current" query at this step. Newest-first window
            # covers (step, step-1, ..., step-depth+1); negative qids in
            # the cold-start phase are clamped to qid 0.
            current_qid = step + depth - 1
            history_qids_newest_first = [
                max(0, current_qid - k) for k in range(depth)
            ]

            current_vec = qid_to_vec[current_qid]
            history_newest_first = [
                {"robot_state": qid_to_vec[q]} for q in history_qids_newest_first
            ]
            weights = [1.0 / depth] * depth

            spec_off = QuerySpec(
                query_keys={"robot_state": current_vec},
                top_k=10,
                checkpoint_id=CheckpointID.CP1,
                trajectory_history=history_newest_first,
                trajectory_weights=weights,
                fusion_method=fusion,
                fusion_weights={"robot_state": 1.0},
            )
            spec_on = QuerySpec(
                query_keys={"robot_state": current_vec},
                top_k=10,
                checkpoint_id=CheckpointID.CP1,
                trajectory_history=history_newest_first,
                trajectory_weights=weights,
                fusion_method=fusion,
                fusion_weights={"robot_state": 1.0},
                search_session_id=sid,
                trajectory_query_ids=history_qids_newest_first,
            )

            def call_legacy():
                with backend.force_legacy_path():
                    backend.search(spec_off)

            def call_off():
                backend.search(spec_off)

            def call_on():
                backend.search(spec_on)

            rows.append({
                "step": step,
                "legacy_ms": _median_ms(call_legacy, repeats=repeats),
                "new_memo_off_ms": _median_ms(call_off, repeats=repeats),
                "new_memo_on_ms": _median_ms(call_on, repeats=repeats),
            })
    finally:
        backend.close_search_session(sid)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_matrix(rows: list[dict]) -> None:
    print("| entries | depth | fusion | legacy ms | new memo OFF ms | new memo ON ms | speedup (legacy → memo ON) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        speedup = r["legacy_ms"] / max(r["new_memo_on_ms"], 1e-9)
        print(
            f"| {r['entries']} | {r['depth']} | {r['fusion']} | "
            f"{r['legacy_ms']:.2f} | {r['new_memo_off_ms']:.2f} | "
            f"{r['new_memo_on_ms']:.2f} | {speedup:.2f}x |"
        )


def _print_curve(rows: list[dict]) -> None:
    print("| step | legacy ms | new memo OFF ms | new memo ON ms |")
    print("|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['step']} | {r['legacy_ms']:.2f} | "
            f"{r['new_memo_off_ms']:.2f} | {r['new_memo_on_ms']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10,
                        help="Wall-clock repeats per cell (median reported).")
    args = parser.parse_args()

    print("# Trajectory Search Benchmark — plan §7.4 / §10 #4\n")
    print("## Matrix (single step, median over"
          f" {args.repeats} repeats)\n")
    matrix = bench_matrix(repeats=args.repeats)
    _print_matrix(matrix)
    print()
    print("## Score-memo hit curve (5 successive steps, depth=3, score_sum, 5_000 entries)\n")
    curve = bench_step_curve(repeats=5)
    _print_curve(curve)


if __name__ == "__main__":
    main()
