"""Read-side helpers for a frozen cost map (shared index reconstruction).

Kept outside ``analysis/cost_map.py`` so the outcome stage can rebuild the
paired bootstrap index from the frozen map without importing the cost-only
module's build path.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def shared_index(cells_by_task: dict, seed: int, reps: int) -> list:
    rng = np.random.Generator(np.random.PCG64(seed))
    picks = []
    for _ in range(reps):
        rep = []
        for task in sorted(cells_by_task):
            lst = cells_by_task[task]
            idx = rng.integers(0, len(lst), len(lst))
            rep.extend(lst[j] for j in idx)
        picks.append(rep)
    return picks


def shared_index_from_map(cost_map: dict, grid) -> list:
    by_task: dict = {}
    for t, i in sorted(grid):
        by_task.setdefault(t, []).append((t, i))
    picks = shared_index(by_task, int(cost_map["seed"]), int(cost_map["replicates"]))
    sha = hashlib.sha256(json.dumps(picks, separators=(",", ":")).encode()).hexdigest()
    if sha != cost_map.get("bootstrap_index_sha256"):
        raise SystemExit("reconstructed bootstrap index != the frozen cost map's index")
    return picks


def index_arrays(picks: list, grid) -> tuple[list, "np.ndarray"]:
    """Sorted cell list plus an (R, n_cells) int64 array of resampled cell indices."""
    cells = sorted(grid)
    if not picks or not cells:
        raise SystemExit("bootstrap index and grid must be non-empty")
    if any(len(rep) != len(cells) for rep in picks):
        raise SystemExit("each bootstrap replicate must contain exactly one full resampled grid")
    cidx = {c: j for j, c in enumerate(cells)}
    try:
        idx = np.fromiter((cidx[c] for rep in picks for c in rep), dtype=np.int64,
                          count=len(picks) * len(cells)).reshape(len(picks), len(cells))
    except KeyError as exc:
        raise SystemExit(f"bootstrap index contains a cell outside the frozen grid: {exc.args[0]!r}") from exc
    return cells, idx
