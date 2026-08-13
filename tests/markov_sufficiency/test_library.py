"""Tests for artifact loading, ancestor walking and the output chain.

The output-chain test is the one that matters scientifically: a bare
``action_chunk[0][:7]`` is model-space, and unnormalisation is a per-dimension
affine map, so substituting one for the other silently rescales every residual
and distance threshold in E1/E3.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from exp.markov_sufficiency import _library

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl"
NORM_STATS = REPO / "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"

needs_library = pytest.mark.skipif(not LIBRARY.exists(), reason="requires the local cache artifact")
needs_norm_stats = pytest.mark.skipif(not NORM_STATS.exists(), reason="requires the pi05 libero norm stats")


@dataclasses.dataclass
class _Payload:
    action_chunk: np.ndarray
    task_key: str = "t"


@dataclasses.dataclass
class _Entry:
    id: str
    trajectory_id: str
    step_idx: int
    prev_ids: list
    query_keys: dict
    payload: _Payload
    outcome: object = None


def _toy_library(n: int = 4) -> _library.Library:
    entries = []
    for i in range(n):
        entries.append(
            _Entry(
                id=f"ep:{i}",
                trajectory_id="ep",
                step_idx=i,
                prev_ids=[f"ep:{i - 1}"] if i else [],
                query_keys={"vision_0": np.full(3, float(i))},
                payload=_Payload(action_chunk=np.zeros((10, 32), dtype=np.float32)),
            )
        )
    by_id = {e.id: e for e in entries}
    return _library.Library(
        entries=entries,
        by_id=by_id,
        by_traj={"ep": entries},
        vector_dims={"vision_0": 3},
        key_builder_type="toy",
        meta={},
    )


# ------------------------------------------------------------------
# Ancestor walking
# ------------------------------------------------------------------


def test_walk_ancestors_returns_exactly_depth_entries():
    lib = _toy_library()
    assert _library.walk_ancestors(lib, "ep:3", 2) == ["ep:2", "ep:1"]


def test_walk_ancestors_pads_with_none_at_chain_head_without_truncating():
    lib = _toy_library()
    # Production scores a missing ancestor as 0.0 at its level, so the list
    # length must stay == depth rather than being shortened.
    assert _library.walk_ancestors(lib, "ep:1", 3) == ["ep:0", None, None]
    assert _library.walk_ancestors(lib, "ep:0", 2) == [None, None]


def test_walk_ancestors_treats_dangling_ids_as_missing():
    lib = _toy_library()
    lib.by_id["ep:2"].prev_ids = ["nonexistent"]
    assert _library.walk_ancestors(lib, "ep:2", 1) == [None]


def test_walk_ancestors_rejects_multi_parent_chains():
    lib = _toy_library()
    lib.by_id["ep:2"].prev_ids = ["ep:1", "ep:0"]
    with pytest.raises(ValueError, match="parents"):
        _library.walk_ancestors(lib, "ep:2", 1)


def test_walk_ancestors_rejects_negative_depth():
    with pytest.raises(ValueError):
        _library.walk_ancestors(_toy_library(), "ep:0", -1)


# ------------------------------------------------------------------
# Output chain
# ------------------------------------------------------------------


@needs_norm_stats
def test_output_chain_matches_production_transforms():
    """Our chain must equal Unnormalize -> LiberoOutputs applied by openpi itself."""
    from openpi import transforms as _transforms
    from openpi.policies import libero_policy

    with NORM_STATS.open() as fh:
        stats = json.load(fh)["norm_stats"]["actions"]
    norm_stats = {
        "actions": _transforms.NormStats(
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
            q01=np.asarray(stats["q01"], dtype=np.float32),
            q99=np.asarray(stats["q99"], dtype=np.float32),
        )
    }
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-1.0, 1.0, size=(10, 32)).astype(np.float32)

    expected = libero_policy.LiberoOutputs()(
        _transforms.Unnormalize(norm_stats, use_quantiles=True)({"actions": chunk.copy()})
    )["actions"]
    got = _library.build_output_chain(str(NORM_STATS))(chunk)
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-6)


@needs_norm_stats
def test_executed_action_is_not_a_bare_slice():
    """Unnormalisation must actually change the values, not just slice them."""
    rng = np.random.default_rng(1)
    chunk = rng.uniform(-1.0, 1.0, size=(10, 32)).astype(np.float32)
    entry = _Entry("e", "ep", 0, [], {}, _Payload(action_chunk=chunk))
    chain = _library.build_output_chain(str(NORM_STATS))
    executed = _library.executed_action(entry, out_chain=chain)
    assert executed.shape == (7,)
    assert not np.allclose(executed, chunk[0, :7])
    np.testing.assert_allclose(_library.raw_action(entry), chunk[0])


def test_executed_action_rejects_non_2d_chunks():
    entry = _Entry("e", "ep", 0, [], {}, _Payload(action_chunk=np.zeros(32, dtype=np.float32)))
    with pytest.raises(ValueError, match="\\[T, D\\]"):
        _library.executed_action(entry, out_chain=lambda x: x)


# ------------------------------------------------------------------
# Real artifact
# ------------------------------------------------------------------


@needs_library
def test_load_library_indexes_and_flags_legacy_outcomes():
    lib = _library.load_library(LIBRARY)
    assert len(lib) == 1018
    assert lib.meta["n_trajectories"] == 49
    # Legacy artifacts carry no per-entry outcome tag; that is recorded, not an
    # error, and it bars any claim that depends on outcome labels.
    assert lib.meta["outcome_all_none"] is True
    entry = lib.entries[0]
    assert lib.by_id[entry.id] is entry
    assert entry.id == f"{entry.trajectory_id}:{entry.step_idx}"


@needs_library
def test_trajectories_are_sorted_by_step():
    lib = _library.load_library(LIBRARY)
    for _, items in lib.trajectories():
        steps = [e.step_idx for e in items]
        assert steps == sorted(steps)
