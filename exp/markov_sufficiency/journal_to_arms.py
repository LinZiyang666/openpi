"""Convert conductor journals into the per-arm episode tables E4/E5 expects.

``e45_rollout_analysis.load_arm`` reads ``[{task_id, init_state_idx, success}]``,
while the rollout driver writes an append-only ``journal.jsonl`` of
``{task_uid, yaml_id, phase, status, success, ts}``. This module bridges the two.

The part that is easy to get wrong is the episode identity. Each arm is
collected in **two batches**: the official LIBERO ``pruned_init`` pool and the
disjoint ``db_init`` pool (plan section 3.5.2). Within a batch an episode is
``(task_id, init_idx)``, but index 0 of one pool is a *different* initial state
from index 0 of the other -- the two pools were verified to have zero row
overlap. Merging them on a bare init index would silently collapse 950 paired
episodes into 500 and pair each arm against the wrong episode. So the pool is
folded into the key with a fixed offset.

A second subtlety: a resumed run re-appends nothing for episodes it skips, but
a batch that was relaunched can still contain the same ``task_uid`` twice (for
example a retry after a worker died). The last terminal row wins, which matches
the driver's own semantics.

Public interface: :func:`parse_task_uid`, :func:`load_journal`,
:func:`build_arms`, :func:`main`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any, Iterable, Optional

#: Init indices of the db_init pool are shifted by this much so they can never
#: collide with the official pool's indices in a paired table.
POOL_OFFSET = {"official": 0, "db_init": 10_000}

#: Rows in any other state are not finished episodes and carry no outcome.
TERMINAL = ("done", "failed")


def parse_task_uid(uid: str) -> tuple[str, str, int, int]:
    """Split ``"<yaml_id>:<phase>:<task_id>:<init_idx>"``.

    The yaml id itself may contain no colon (the emitters never produce one),
    so the split is from the right and stays unambiguous.
    """
    parts = uid.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unrecognised task_uid {uid!r}: expected yaml:phase:task:init")
    yaml_id, phase, task_s, init_s = parts
    return yaml_id, phase, int(task_s), int(init_s)


def load_journal(path: str | pathlib.Path, pool: str) -> dict[str, dict[tuple[int, int], bool]]:
    """Read one batch into ``{yaml_id: {(task_id, keyed_init): success}}``."""
    if pool not in POOL_OFFSET:
        raise ValueError(f"unknown pool {pool!r}; expected one of {sorted(POOL_OFFSET)}")
    offset = POOL_OFFSET[pool]

    latest: dict[tuple[str, int, int], tuple[float, bool]] = {}
    n_rows = n_terminal = 0
    with pathlib.Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            n_rows += 1
            row = json.loads(line)
            if row.get("status") not in TERMINAL:
                continue
            n_terminal += 1
            yaml_id, _phase, task_id, init_idx = parse_task_uid(row["task_uid"])
            key = (yaml_id, task_id, init_idx)
            ts = float(row.get("ts", 0.0))
            if key not in latest or ts >= latest[key][0]:
                latest[key] = (ts, bool(row.get("success")))

    out: dict[str, dict[tuple[int, int], bool]] = collections.defaultdict(dict)
    for (yaml_id, task_id, init_idx), (_ts, success) in latest.items():
        out[yaml_id][(task_id, init_idx + offset)] = success
    return dict(out)


def build_arms(batches: Iterable[tuple[str, str]]) -> dict[str, dict[tuple[int, int], bool]]:
    """Merge ``(journal_path, pool)`` batches into one table per arm.

    A key appearing in two pools would mean the offset failed to separate them;
    that is a bug, not a data condition, so it raises rather than overwriting.
    """
    merged: dict[str, dict[tuple[int, int], bool]] = collections.defaultdict(dict)
    for path, pool in batches:
        for yaml_id, table in load_journal(path, pool).items():
            target = merged[yaml_id]
            for key, success in table.items():
                if key in target:
                    raise ValueError(
                        f"episode {key} appears twice for arm {yaml_id}: the pool offset "
                        "failed to separate the two init pools"
                    )
                target[key] = success
    return dict(merged)


def to_records(table: dict[tuple[int, int], bool]) -> list[dict[str, Any]]:
    """Render one arm in the shape ``e45_rollout_analysis.load_arm`` reads."""
    return [
        {"task_id": task_id, "init_state_idx": init_idx, "success": success}
        for (task_id, init_idx), success in sorted(table.items())
    ]


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="conductor journal -> per-arm episode tables")
    ap.add_argument(
        "--batch", action="append", required=True, metavar="POOL=JOURNAL",
        help="repeatable, e.g. official=.../official/journal.jsonl db_init=.../db_init/journal.jsonl",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    batches = []
    for raw in args.batch:
        pool, _, path = raw.partition("=")
        if pool not in POOL_OFFSET or not path:
            ap.error(f"--batch must be POOL=JOURNAL with POOL in {sorted(POOL_OFFSET)}, got {raw!r}")
        batches.append((path, pool))

    arms = build_arms(batches)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for yaml_id, table in sorted(arms.items()):
        records = to_records(table)
        (out_dir / f"{yaml_id}.json").write_text(json.dumps(records, indent=2))
        n_ok = sum(1 for r in records if r["success"])
        summary[yaml_id] = {"n_episodes": len(records), "n_success": n_ok,
                            "sr": (n_ok / len(records)) if records else float("nan")}
    (out_dir / "arms_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
