"""What a static episode-level shard would cost, measured on a finished arm.

The scheduler runs one arm on one server, so a phase holding a single arm uses
1/N of an N-slot pool. Splitting that arm's episodes across all N slots removes
the idle N-1 -- but a *static* split reintroduces a smaller version of the same
problem, because the slots cannot steal from each other and LIBERO episodes do
not all take the same time (a success stops early; a failure runs to the step
cap). ``orchestrate_search.py``'s docstring rejects static splits for exactly
this reason -- at *cell* granularity, where a slot holds two or three items and
one long item dominates its shard.

This probe answers whether that objection survives at *episode* granularity, by
replaying a completed arm's real per-episode cost against both partitions. Cost
is decisions per episode (rows in the per-step log): one decision is one
inference plus ``replan_steps`` environment steps, so it tracks wall clock far
more closely than an episode count does.

    python shard_imbalance_probe.py <per_step_dir> [--shards 6] [--arms a,b,c]

Reports, per arm, the makespan of the slowest shard against the ideal even
split. "Today" is the same number for a single-arm phase with one slot working:
``100*(N-1)``%.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics as st


def episode_costs(path: pathlib.Path) -> list[int]:
    """Decisions per episode, in the order the scheduler would dispatch them.

    Keyed by ``(task_id, orig_init_state_idx)`` -- the same pair the integrity
    gate uses -- so an episode retried under a second attempt is not counted as
    two, and the ordering matches the sorted episode list a strategy plans.
    """
    per_ep: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            per_ep[(row.get("task_id"), row.get("orig_init_state_idx"))] += 1
    return [per_ep[key] for key in sorted(per_ep)]


def makespan(costs: list[int], shards: list[list[int]]) -> tuple[int, float, float]:
    """Slowest shard, the ideal even split, and the excess as a percentage."""
    sums = [sum(s) for s in shards]
    ideal = sum(costs) / len(shards)
    worst = max(sums)
    return worst, ideal, 100 * (worst / ideal - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_step_dir")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--arms", default="", help="comma-separated stems; default: every .jsonl")
    args = ap.parse_args()

    root = pathlib.Path(args.per_step_dir)
    stems = args.arms.split(",") if args.arms else sorted(p.stem for p in root.glob("*.jsonl"))
    n = args.shards
    for stem in stems:
        path = root / f"{stem}.jsonl"
        if not path.is_file():
            print(f"{stem}: MISSING")
            continue
        costs = episode_costs(path)
        if not costs:
            print(f"{stem}: no decision rows")
            continue
        mean = st.mean(costs)
        print(
            f"\n{stem}: {len(costs)} episodes, {sum(costs)} decisions, "
            f"mean={mean:.1f} sd={st.pstdev(costs):.1f} "
            f"cv={st.pstdev(costs) / mean:.2f} min={min(costs)} max={max(costs)}"
        )
        stride = [costs[k::n] for k in range(n)]
        worst, ideal, excess = makespan(costs, stride)
        print(f"  stride/{n}:     makespan={worst} ideal={ideal:.0f} tail bubble={excess:.1f}%")
        size = math.ceil(len(costs) / n)
        block = [costs[i * size : (i + 1) * size] for i in range(n)]
        worst_b, _, excess_b = makespan(costs, block)
        print(f"  contiguous/{n}: makespan={worst_b} tail bubble={excess_b:.1f}%")
        print(f"  today (1 slot of {n}): {100 * (n - 1)}%")


if __name__ == "__main__":
    main()
