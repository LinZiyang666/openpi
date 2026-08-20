"""Build the deterministic, nested size grid for the cache-size ablation (X9b).

The independent variable is **successful trajectories per task**, not "inits
sampled per task". That distinction is load-bearing: retrieval is task-scoped
(``search_strategy`` builds a ``QueryFilter(task_key=...)``), so a task with zero
entries in a given tier returns no candidates, ``AlwaysHitJudge`` falls back to
MISS, and that task's evaluation episodes silently run the full teacher. Since
low-success tasks are exactly the ones prone to zero coverage, and coverage
improves with size, an init-based axis would bias the curve flat in a direction
that flatters the expected conclusion. Counting successes removes the failure
mode structurally.

Ordering rules (plan §3.2):
  priority 1  successful members of the historical 5-init cache library, sorted
              by subset ``init_idx`` -- makes tier S3 the "same inits, fresh
              trajectories" anchor comparable with prior experiments. Subset
              index is the right space: the split file's ``protected_in_train``
              and ``val`` entries are subset indices too, so the anchor set, the
              exclusion set and the ordering key all live in one coordinate
              system.
  priority 2  the remaining successful B-train inits, deterministically shuffled
  excluded    B-val, always -- it is the whole TIER line's only slice that sits
              in neither student training nor the library, and this experiment
              must not burn it

Take rules (plan §3.2 R1-R5):
  R1  per-task count is ``min(k, n_t)`` for every tier, not just the largest
  R2  ``n_t == 0`` for any task makes the suite unrunnable -- fail loud
  R3  tiers that end up identical are reported as degenerate (the caller marks
      the corresponding comparison ``not evaluable``); tiers are never dropped
  R4  the curve's x-axis is the realized mean count, not the nominal k
  R5  a failed historical anchor is topped up from priority 2, so S3 still holds
      5 trajectories; the per-task anchor hit count is recorded and disclosed
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import h5py
import yaml

# Nominal tiers, in ascending order. S6's nominal 45 is the B-train budget; most
# tasks are expected to top out below it (R1/R4).
DEFAULT_TIERS: tuple[int, ...] = (1, 2, 5, 10, 20, 45)
SHUFFLE_SEED = 0


@dataclass
class Episode:
    """One collected rollout, keyed by the identity we can join on.

    Deliberately no ``orig_init_state_idx``: the server-side ``--collect``
    producer (``EpisodeDataCollector``) drops the client's ``extra_metadata``
    and writes only ``experiment_name / task / episode_id / num_steps /
    timestamp / success``, so that attribute is simply absent from this corpus.
    Identity comes from the path (``task_N/episode_M.h5``), which is subset
    index space -- the same space the split file uses.
    """

    task_id: int
    init_idx: int
    rel_path: str
    success: bool


@dataclass
class TaskGrid:
    """Per-task ordering plus the realized counts for each tier."""

    task_id: int
    ordered: list[Episode]
    anchor_hits: int
    anchor_total: int
    counts: dict[int, int] = field(default_factory=dict)

    @property
    def n_success(self) -> int:
        """Trajectories this task contributes, under the active outcome filter.

        Named for the frozen default (success-only). Under ``outcome_filter="all"``
        it is simply the eligible B-train count, which is 45 for every task -- so
        the R1/R3/R4 topping-out machinery stays wired up but never binds.
        """
        return len(self.ordered)


def scan_collected(collect_root: pathlib.Path) -> list[Episode]:
    """Read every ``task_N/episode_M.h5`` under one suite's collect directory.

    The layout mirrors ``EpisodeDataCollector``'s naming when the client runs
    with ``--save_trajectory``: the path itself encodes (task, init), which is
    what makes the grid joinable without extra bookkeeping.
    """
    episodes: list[Episode] = []
    checked_schema = False
    for task_dir in sorted(collect_root.glob("task_*")):
        if not task_dir.is_dir():
            continue
        task_id = int(task_dir.name.split("_", 1)[1])
        for h5_path in sorted(task_dir.glob("episode_*.h5")):
            init_idx = int(h5_path.stem.split("_", 1)[1])
            with h5py.File(h5_path, "r") as f:
                if not checked_schema:
                    _assert_collect_schema(h5_path, f)
                    checked_schema = True
                success = bool(f.attrs.get("success", False))
            episodes.append(
                Episode(
                    task_id=task_id,
                    init_idx=init_idx,
                    rel_path=h5_path.relative_to(collect_root).as_posix(),
                    success=success,
                )
            )
    return episodes


# Fields the artifact builder needs; their presence distinguishes the
# server-side --collect corpus from the client-side --save_trajectory dump,
# which uses the *identical* task_N/episode_M.h5 layout but carries only
# observations and actions. Pointing --collect-root at the wrong one would
# otherwise produce a grid that silently indexes unbuildable episodes.
_REQUIRED_STEP_FIELDS = ("vision_0", "prompt_emb", "robot_state", "clean_action")


def _assert_collect_schema(h5_path: pathlib.Path, f) -> None:
    if "success" not in f.attrs:
        raise ValueError(f"{h5_path}: no 'success' attr; not a --collect episode")
    steps = [k for k in f if k.startswith("step_")]
    if not steps:
        raise ValueError(f"{h5_path}: no step_* groups")
    missing = [k for k in _REQUIRED_STEP_FIELDS if k not in f[steps[0]]]
    if missing:
        raise ValueError(
            f"{h5_path}: step group lacks {missing}. This looks like the client-side "
            "--save_trajectory dump (same layout, observations only), not the "
            "server-side --collect corpus the library is built from."
        )


def order_task(
    episodes: list[Episode],
    anchor_inits: set[int],
    val_inits: set[int],
    seed: int = SHUFFLE_SEED,
    outcome_filter: str = "success",
) -> TaskGrid:
    """Order one task's episodes: anchors first, then shuffled rest.

    ``val_inits`` (B-val) are dropped outright, before any counting.

    ``outcome_filter`` selects what the library is made of, and it changes what
    the size axis *means*:

    *   ``"success"`` (the frozen default) -- size is "successful trajectories
        per task". Hard tasks contribute fewer, so the top tiers top out and the
        R1/R3/R4 rules do real work.
    *   ``"all"`` -- size is "collected trajectories per task", failures included.
        Every task then contributes all 45 of its B-train inits, so no tier ever
        tops out; the library contains failed rollouts, and ``always_hit`` replays
        them like any other. Owner ruling 2026-08-17.

    Must match the ``--outcome-filter`` passed to the builder: this picks which
    episodes are *listed*, the builder picks which listed episodes it *keeps*,
    and a mismatch silently shrinks the library below the tier it claims.
    """
    if outcome_filter not in ("success", "all"):
        raise ValueError(f"outcome_filter must be 'success' or 'all', got {outcome_filter!r}")
    task_ids = {e.task_id for e in episodes}
    assert len(task_ids) == 1, f"order_task expects one task, got {task_ids}"
    task_id = task_ids.pop()

    eligible = [e for e in episodes if e.init_idx not in val_inits]
    if outcome_filter == "success":
        eligible = [e for e in eligible if e.success]

    anchors = sorted(
        (e for e in eligible if e.init_idx in anchor_inits),
        key=lambda e: e.init_idx,
    )
    rest = [e for e in eligible if e.init_idx not in anchor_inits]
    rng = random.Random(f"{seed}:{task_id}")
    rng.shuffle(rest)

    return TaskGrid(
        task_id=task_id,
        ordered=anchors + rest,
        anchor_hits=len(anchors),
        anchor_total=len(anchor_inits),
    )


def validate_collection(
    episodes: list[Episode],
    expected_tasks: set[int],
    *,
    expected_inits_per_task: int | None = None,
) -> None:
    """Check the collected corpus against the authoritative task roster.

    Zero coverage for a task is the load-bearing risk this experiment is built
    around, and "the task directory never appeared at all" is a *stronger*
    version of it than "the task appeared but every episode failed". Counting
    only the tasks present would quietly produce a nine-task grid, divide the
    realized mean by nine, and never trip the fail-loud path.
    """
    seen = {e.task_id for e in episodes}

    missing = sorted(expected_tasks - seen)
    if missing:
        raise ValueError(
            f"collection is missing task(s) {missing} entirely; expected the full roster "
            f"{sorted(expected_tasks)}. A grid built from a partial roster would silently "
            "average over fewer tasks (plan §3.2 R2)"
        )
    unexpected = sorted(seen - expected_tasks)
    if unexpected:
        raise ValueError(
            f"collection contains unexpected task(s) {unexpected}; roster is "
            f"{sorted(expected_tasks)}"
        )

    for task_id in sorted(expected_tasks):
        inits = [e.init_idx for e in episodes if e.task_id == task_id]
        dupes = sorted({i for i in inits if inits.count(i) > 1})
        if dupes:
            raise ValueError(
                f"task {task_id}: init index/indices {dupes} collected more than once; "
                "duplicate episodes would double-count in the library"
            )
        if expected_inits_per_task is not None and len(inits) != expected_inits_per_task:
            raise ValueError(
                f"task {task_id}: collected {len(inits)} episodes, expected "
                f"{expected_inits_per_task} (the formal collection ledger)"
            )


def build_grid(
    episodes: list[Episode],
    anchor_inits_by_task: dict[int, set[int]],
    val_inits_by_task: dict[int, set[int]],
    tiers: tuple[int, ...] = DEFAULT_TIERS,
    seed: int = SHUFFLE_SEED,
    expected_tasks: set[int] | None = None,
    expected_inits_per_task: int | None = None,
    outcome_filter: str = "success",
) -> dict[str, Any]:
    """Assemble the full nested grid and its realized-count bookkeeping.

    ``expected_tasks`` is the authoritative roster (normally the split file's
    keys). It is validated before anything is counted.
    """
    if expected_tasks is None:
        expected_tasks = set(anchor_inits_by_task) | set(val_inits_by_task)
    if not expected_tasks:
        raise ValueError("no authoritative task roster supplied; cannot validate collection")
    validate_collection(episodes, expected_tasks,
                        expected_inits_per_task=expected_inits_per_task)

    by_task: dict[int, list[Episode]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task_id, []).append(ep)

    grids: dict[int, TaskGrid] = {}
    for task_id, eps in sorted(by_task.items()):
        grid = order_task(
            eps,
            anchor_inits_by_task.get(task_id, set()),
            val_inits_by_task.get(task_id, set()),
            seed=seed,
            outcome_filter=outcome_filter,
        )
        # R2: a task with no eligible trajectory breaks the "every task has at
        # least one entry" premise in *every* tier, which would silently turn
        # part of the evaluation into a teacher run. Refuse to emit a grid.
        if grid.n_success == 0:
            what = "successful " if outcome_filter == "success" else ""
            raise ValueError(
                f"task {task_id} has zero {what}B-train trajectories; the suite is "
                "unrunnable because retrieval is task-scoped and this task would fall "
                "back to the teacher in every tier (plan §3.2 R2)"
            )
        grids[task_id] = grid

    tier_lists: dict[str, dict[str, list[str]]] = {}
    for k in tiers:
        name = f"S{tiers.index(k) + 1}"
        per_task: dict[str, list[str]] = {}
        for task_id, grid in grids.items():
            take = min(k, grid.n_success)  # R1
            grid.counts[k] = take
            per_task[f"task_{task_id}"] = [e.rel_path for e in grid.ordered[:take]]
        tier_lists[name] = per_task

    # R3: flag tiers whose selection is identical to the previous one.
    degenerate: list[str] = []
    names = list(tier_lists)
    for prev, cur in zip(names, names[1:]):
        if tier_lists[prev] == tier_lists[cur]:
            degenerate.append(f"{prev}-{cur}")

    return {
        "seed": seed,
        "tiers": {f"S{i + 1}": k for i, k in enumerate(tiers)},
        "episodes": tier_lists,
        "realized_counts": {
            f"task_{tid}": {f"S{tiers.index(k) + 1}": g.counts[k] for k in tiers}
            for tid, g in grids.items()
        },
        # R4: the plotted x-axis, and what slope_k actually compares.
        "mean_realized": {
            f"S{tiers.index(k) + 1}": sum(g.counts[k] for g in grids.values()) / len(grids)
            for k in tiers
        },
        # R5: how much of the historical anchor survived, per task.
        "anchor_hits": {
            f"task_{tid}": {"hit": g.anchor_hits, "of": g.anchor_total}
            for tid, g in grids.items()
        },
        # Recorded so the builder invocation can be checked against the grid that
        # produced the episode lists: the two filters must agree, and a mismatch
        # would shrink the library below the tier it claims without any error.
        "outcome_filter": outcome_filter,
        "n_success": {f"task_{tid}": g.n_success for tid, g in grids.items()},
        "degenerate_pairs": degenerate,
    }


def _load_init_index(path: pathlib.Path, key: str) -> dict[int, set[int]]:
    """Read one field (``protected_in_train`` / ``val``) out of a split yaml."""
    raw = yaml.safe_load(path.read_text())
    out: dict[int, set[int]] = {}
    for task_name, spec in raw.items():
        task_id = int(task_name.split("_", 1)[1])
        out[task_id] = set(spec.get(key, []) or [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-root", required=True,
                    help="<collect_dir>/<suite> holding task_N/episode_M.h5")
    ap.add_argument("--split", required=True,
                    help="config/common/split_<suite>.yaml (protected_in_train + val)")
    ap.add_argument("--output", required=True, help="destination size_grid_<suite>.yaml")
    ap.add_argument("--tiers", default=",".join(str(t) for t in DEFAULT_TIERS))
    ap.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    ap.add_argument("--outcome-filter", default="success", choices=["success", "all"],
                    help="what the library is made of. 'success' (frozen default) makes "
                         "size mean successful trajectories per task; 'all' includes "
                         "failed rollouts, so every task contributes all 45 B-train "
                         "inits and no tier tops out. MUST match the builder's "
                         "--outcome-filter, or the library silently shrinks below the "
                         "tier it claims.")
    ap.add_argument("--expect-inits-per-task", type=int, default=50,
                    help="formal collection ledger: episodes collected per task "
                         "(0 disables the check)")
    args = ap.parse_args()

    tiers = tuple(int(t) for t in args.tiers.split(","))
    split_path = pathlib.Path(args.split)

    episodes = scan_collected(pathlib.Path(args.collect_root))
    anchors = _load_init_index(split_path, "protected_in_train")
    vals = _load_init_index(split_path, "val")
    grid = build_grid(
        episodes,
        anchor_inits_by_task=anchors,
        val_inits_by_task=vals,
        tiers=tiers,
        seed=args.seed,
        # The split file is the authoritative task roster.
        expected_tasks=set(anchors) | set(vals),
        expected_inits_per_task=args.expect_inits_per_task or None,
        outcome_filter=args.outcome_filter,
    )

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(grid, sort_keys=False, allow_unicode=True))
    print(json.dumps({
        "output": str(out),
        "mean_realized": grid["mean_realized"],
        "degenerate_pairs": grid["degenerate_pairs"],
    }, indent=2))


if __name__ == "__main__":
    main()
