"""Merge per-step shards from both init pools into one E2-ready JSONL.

Two things make a plain ``cat`` wrong here.

**Pool collision.** Each arm is collected over two disjoint init pools (plan
section 3.5.2): the official LIBERO ``pruned_init`` pool and ``db_init``. Both
index their inits from zero, so ``e4_spatial__A0:eval:0:0`` names a *different*
episode in each. Every grouping key in the E2 pipeline --
``_timeaxis._episode_key``, ``dedupe_attempts`` and ``episode_table`` -- keys on
``subset_init_state_idx``, so concatenating the shards silently merges 450
episode pairs: the sample collapses from 950 to 500, ``n_cycle`` becomes the max
of two unrelated episodes and their deviation counts are summed. This module
offsets the db_init pool's index (the same offset ``journal_to_arms`` uses on
the journal side) so the two stay distinct.

**Shard multiplicity.** ``run_phase2 --per-step-out`` writes once at process
exit, so a batch that was resumed leaves several shards. They are disjoint by
construction (a resumed run skips finished episodes), and are concatenated in
file order.

``orig_init_state_idx`` is left untouched: it is the provenance of the state
inside its own pool, and rewriting it would destroy the only link back to the
init files.

Public interface: :func:`merge`, :func:`main`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any, Iterable, Optional

from exp.markov_sufficiency.journal_to_arms import POOL_OFFSET

#: Field every grouping key in the E2 pipeline is built on.
GROUP_FIELD = "subset_init_state_idx"


def _shift(row: dict[str, Any], pool: str) -> dict[str, Any]:
    offset = POOL_OFFSET[pool]
    out = dict(row)
    out["_pool"] = pool
    if offset and GROUP_FIELD in out and out[GROUP_FIELD] is not None:
        out[GROUP_FIELD] = int(out[GROUP_FIELD]) + offset
    return out


def merge(
    shards: Iterable[tuple[str, str | pathlib.Path]],
    arm: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read ``(pool, path)`` shards into one row list, offsetting by pool.

    ``arm`` restricts the output to one ``yaml_id``. E2-primary's estimand is
    the A0 arm alone (depth 1, ``step_filter: all``, ``always_hit``); the other
    arms in the same shard are different configurations and pooling them would
    silently widen the estimand.

    Sidecar rows (``_kind``-tagged) are carried through untouched: the time-axis
    gate drops them by schema, and rewriting their fields would be meaningless.
    """
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"per_pool": collections.Counter(), "sidecar": collections.Counter(),
                             "shards": [], "arm_filter": arm, "dropped_other_arms": 0}
    for pool, path in shards:
        if pool not in POOL_OFFSET:
            raise ValueError(f"unknown pool {pool!r}; expected one of {sorted(POOL_OFFSET)}")
        path = pathlib.Path(path)
        n = 0
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("_kind") is not None:
                    if arm is not None and row.get("yaml_id") != arm:
                        continue
                    stats["sidecar"][pool] += 1
                    rows.append({**row, "_pool": pool})
                    continue
                if arm is not None and row.get("yaml_id") != arm:
                    stats["dropped_other_arms"] += 1
                    continue
                rows.append(_shift(row, pool))
                stats["per_pool"][pool] += 1
                n += 1
        stats["shards"].append({"pool": pool, "path": str(path), "step_rows": n})

    keys = collections.Counter(
        (r.get("yaml_id"), r.get("task_id"), r.get(GROUP_FIELD))
        for r in rows
        if r.get("_kind") is None
    )
    by_pool_keys: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        if r.get("_kind") is None:
            by_pool_keys[r["_pool"]].add((r.get("yaml_id"), r.get("task_id"), r.get(GROUP_FIELD)))
    pools = sorted(by_pool_keys)
    collisions = 0
    for i, a in enumerate(pools):
        for b in pools[i + 1:]:
            collisions += len(by_pool_keys[a] & by_pool_keys[b])
    if collisions:
        raise ValueError(
            f"{collisions} episode keys still collide across pools after offsetting; "
            "the merged file would silently fuse distinct episodes"
        )
    stats["distinct_episode_keys"] = len(keys)
    stats["per_pool"] = dict(stats["per_pool"])
    stats["sidecar"] = dict(stats["sidecar"])
    return rows, stats


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="merge per-step shards across init pools")
    ap.add_argument(
        "--shard", action="append", required=True, metavar="POOL=PATH",
        help="repeatable, e.g. official=.../official/per_step__X.jsonl db_init=.../db_init/per_step__Y.jsonl",
    )
    ap.add_argument(
        "--arm", default=None,
        help="keep only this yaml_id. E2-primary's estimand is the A0 arm alone; "
        "pooling the other arms would widen it without saying so.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    shards = []
    for raw in args.shard:
        pool, _, path = raw.partition("=")
        if pool not in POOL_OFFSET or not path:
            ap.error(f"--shard must be POOL=PATH with POOL in {sorted(POOL_OFFSET)}, got {raw!r}")
        shards.append((pool, path))

    rows, stats = merge(shards, arm=args.arm)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    stats["out"] = str(out)
    stats["total_rows"] = len(rows)
    with out.with_suffix(".manifest.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
