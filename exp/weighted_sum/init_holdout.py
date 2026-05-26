"""Held-out init-state selection — the Phase-2 anti-leak guard.

The library artifacts were built from a small subset of each task's LIBERO init
states. If Phase-2 evaluation rolled out from the *same* init states, the live
query at step 0 would find its near-identical twin in the library (trivial hit,
inflated success that does not generalize). This module derives, per task, the
init-state indices that were NOT used to build the library, so Phase-2 episodes
draw exclusively from held-out inits.

The init map (``libero_spatial_init_map.json``) is the source of truth for which
``orig_init_state_idx`` each library episode used. It is a hard input: if it is
missing this module fails fast rather than silently disabling the guard.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_used_inits(init_map_path: str | Path) -> dict[int, set[int]]:
    """Return ``{task_id: {orig_init_state_idx used to build the library}}``.

    Fails fast (``FileNotFoundError``) if the init map is absent — the anti-leak
    guard must never be silently skipped.
    """
    p = Path(init_map_path)
    if not p.exists():
        raise FileNotFoundError(
            f"init map not found: {p}. It is the source of truth for the Phase-2 "
            f"init-state leak guard and must be present (tracked via .gitignore "
            f"exception)."
        )
    records = json.loads(p.read_text(encoding="utf-8"))
    used: dict[int, set[int]] = defaultdict(set)
    for rec in records:
        used[int(rec["task_id"])].add(int(rec["orig_init_state_idx"]))
    return dict(used)


def held_out_inits(
    init_map_path: str | Path,
    task_ids: list[int],
    total_inits_per_task: int = 50,
) -> dict[int, list[int]]:
    """Per-task list of init-state indices disjoint from the library's used set."""
    used = load_used_inits(init_map_path)
    out: dict[int, list[int]] = {}
    for t in task_ids:
        used_t = used.get(t, set())
        out[t] = [i for i in range(total_inits_per_task) if i not in used_t]
    return out
