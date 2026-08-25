"""The warmup pool must be disjoint from both the test set and the library.

Disjointness from the A pool is structural (the warmup runs against the B-pool
directory), so what needs testing is the half that is computed: the complement
of the library's own inits. A warmup episode whose trajectory sits in the
library retrieves itself, and self-retrieval scores are not drawn from the
distribution the thresholds are meant to describe -- theta comes out biased
high and every operating point on the frontier moves with it.
"""

from __future__ import annotations

import dataclasses
import json
import pickle

import pytest

from exp.libero_groot import emit_warmup_pool as pool


@dataclasses.dataclass
class _Entry:
    trajectory_id: str


def _write_library(path, per_task: dict[int, list[int]], trials: int = 50):
    """A pkl whose entries name the (task, init) pairs that went into it."""
    entries = [
        _Entry(f"episode_{task_id * trials + init_idx}_groot")
        for task_id, inits in per_task.items()
        for init_idx in inits
        # Two entries per episode: the unit is a trajectory, not a step.
        for _ in range(2)
    ]
    with open(path, "wb") as handle:
        pickle.dump({"entries": entries}, handle)
    return path


def test_library_inits_inverts_the_global_episode_id(tmp_path):
    path = _write_library(tmp_path / "lib.pkl", {0: [3, 7], 2: [0]})
    assert pool.library_inits(path, trials=50) == {0: {3, 7}, 2: {0}}


def test_warmup_pool_never_overlaps_the_library():
    in_library = {task: {task, task + 1, task + 2, task + 3, task + 4} for task in range(10)}
    picked = pool.warmup_inits(in_library, tasks=10, trials=50, per_task=10)
    for task_id, inits in picked.items():
        assert not set(inits) & in_library[task_id]


def test_every_task_contributes_the_same_number_of_episodes():
    in_library = {0: {0}, 1: {1, 2, 3}}
    picked = pool.warmup_inits(in_library, tasks=3, trials=50, per_task=10)
    assert sorted(picked) == [0, 1, 2]
    assert all(len(v) == 10 for v in picked.values())
    # Task 2 has no library entries at all -- absence must not become an
    # exception or a short list.
    assert picked[2] == list(range(10))


def test_selection_is_the_lowest_free_indices_and_is_reproducible():
    in_library = {0: {0, 2, 4}}
    picked = pool.warmup_inits(in_library, tasks=1, trials=50, per_task=5)
    assert picked[0] == [1, 3, 5, 6, 7]
    assert picked == pool.warmup_inits(in_library, tasks=1, trials=50, per_task=5)


def test_a_task_that_cannot_supply_enough_fails_loudly():
    # Silently taking fewer would leave one task under-represented in a
    # distribution whose quantiles are the entire deliverable.
    in_library = {0: set(range(48))}
    with pytest.raises(SystemExit, match="task 0"):
        pool.warmup_inits(in_library, tasks=1, trials=50, per_task=10)


def test_shard_entries_match_the_client_filter_schema():
    rows = pool.shard_entries({1: [4, 5]})
    assert rows == [
        {"task_id": 1, "subset_init_state_idx": 4, "orig_init_state_idx": 4},
        {"task_id": 1, "subset_init_state_idx": 5, "orig_init_state_idx": 5},
    ]
    # No remap: the client only skips filtered episodes, it never re-indexes
    # them, so the subset and original indices must stay equal.
    assert all(r["subset_init_state_idx"] == r["orig_init_state_idx"] for r in rows)


@pytest.mark.parametrize("lanes", [1, 3, 8])
def test_shards_partition_the_selection_without_loss_or_overlap(tmp_path, lanes):
    rows = pool.shard_entries({t: list(range(10)) for t in range(10)})
    paths = pool.emit_shards(rows, lanes, tmp_path, "gpw_sp")
    assert len(paths) == lanes
    recovered = [r for p in paths for r in json.loads(p.read_text())]
    assert recovered == rows
    keys = [(r["task_id"], r["orig_init_state_idx"]) for r in recovered]
    assert len(set(keys)) == len(keys) == 100


def test_zero_lanes_is_refused(tmp_path):
    with pytest.raises(ValueError, match="lanes"):
        pool.emit_shards([], 0, tmp_path, "gpw_sp")
