"""Tests for trajectory search: two-pass recursive algorithm in InMemoryBackend."""

import torch
import torch.nn.functional as F

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int, index: int) -> torch.Tensor:
    """Unit vector e_i of shape [dim]."""
    v = torch.zeros(dim)
    v[index] = 1.0
    return v


def _make_entry(
    entry_id: str,
    state: torch.Tensor,
    *,
    cp: CheckpointID = CheckpointID.CP1,
    prev_ids: list[str] | None = None,
    next_ids: list[str] | None = None,
    trajectory_id: str | None = None,
    step_idx: int | None = None,
) -> CacheEntry:
    return CacheEntry(
        id=entry_id,
        checkpoint_id=cp,
        query_keys={"robot_state": state.cpu().float().contiguous()},
        payload=CachePayload(action_chunk=torch.randn(50, 32)),
        prev_ids=prev_ids or [],
        next_ids=next_ids or [],
        trajectory_id=trajectory_id,
        step_idx=step_idx,
    )


def _trajectory_spec(
    query_keys: dict[str, torch.Tensor],
    history: list[dict[str, torch.Tensor]],
    weights: list[float],
    *,
    top_k: int = 5,
    cp: CheckpointID = CheckpointID.CP1,
) -> QuerySpec:
    return QuerySpec(
        query_keys=query_keys,
        top_k=top_k,
        checkpoint_id=cp,
        trajectory_history=history,
        trajectory_weights=weights,
    )


# ---------------------------------------------------------------------------
# Test: trajectory rerank prefers linked chain
# ---------------------------------------------------------------------------


class TestTrajectoryRerank:
    """With two equally similar current-step entries, the one with a linked
    predecessor matching the history should score higher."""

    def test_linked_entry_ranked_higher(self):
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        # Trajectory: prev_entry -> entry_A (linked)
        # entry_B has no prev link
        state_current = _unit(32, 0)
        state_prev = _unit(32, 1)

        prev_entry = _make_entry("prev_0", state_prev, trajectory_id="traj_a", step_idx=0)
        entry_a = _make_entry(
            "entry_a", state_current,
            prev_ids=["prev_0"], trajectory_id="traj_a", step_idx=1,
        )
        entry_b = _make_entry("entry_b", state_current, step_idx=1)

        for e in [prev_entry, entry_a, entry_b]:
            storage.insert(e)

        # Query: current=state_current, history[1]=state_prev
        history = [
            {"robot_state": state_current},  # newest (current)
            {"robot_state": state_prev},      # t-1
        ]
        weights = [0.6, 0.4]
        spec = _trajectory_spec(
            {"robot_state": state_current}, history, weights,
        )

        results = backend.search(spec)
        assert len(results) >= 2
        # entry_a should rank first (has matching prev chain)
        assert results[0].id == "entry_a"
        assert results[0].score > results[1].score

    def test_deeper_chain_adds_score(self):
        """3-level trajectory: A0 -> A1 -> A2, with depth=3."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        states = [_unit(32, i) for i in range(3)]
        entries = []
        for i in range(3):
            e = _make_entry(
                f"a:{i}", states[i],
                prev_ids=[f"a:{i-1}"] if i > 0 else [],
                trajectory_id="traj_a", step_idx=i,
            )
            entries.append(e)
            storage.insert(e)

        # Also insert an unlinked entry at the same position as a:2
        lone = _make_entry("lone", states[2], step_idx=2)
        storage.insert(lone)

        history = [
            {"robot_state": states[2]},  # current
            {"robot_state": states[1]},  # t-1
            {"robot_state": states[0]},  # t-2
        ]
        weights = [0.5, 0.3, 0.2]
        spec = _trajectory_spec(
            {"robot_state": states[2]}, history, weights,
        )

        results = backend.search(spec)
        # a:2 should rank above lone (it has full 3-step chain)
        ids = [r.id for r in results]
        assert ids.index("a:2") < ids.index("lone")


# ---------------------------------------------------------------------------
# Test: depth=1 same as single step
# ---------------------------------------------------------------------------


class TestDepth1Fallback:
    def test_depth_1_no_trajectory(self):
        """With trajectory_weights=[1.0] (depth=1), should not trigger trajectory search."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        state = _unit(32, 0)
        e = _make_entry("e0", state)
        storage.insert(e)

        # Single weight → spec has len(weights)=1, so trajectory path NOT taken
        spec = QuerySpec(
            query_keys={"robot_state": state},
            top_k=1,
            checkpoint_id=CheckpointID.CP1,
            trajectory_history=[{"robot_state": state}],
            trajectory_weights=[1.0],  # only 1 weight → not > 1
        )
        results = backend.search(spec)
        # Should still find the entry via single-field fallback
        assert len(results) == 1
        assert results[0].id == "e0"


# ---------------------------------------------------------------------------
# Test: broken prev_ids graceful fallback
# ---------------------------------------------------------------------------


class TestBrokenChain:
    def test_missing_prev_entry(self):
        """Entry with prev_ids pointing to non-existent entry still scores."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        state = _unit(32, 0)
        e = _make_entry("e0", state, prev_ids=["ghost_entry"], step_idx=1)
        storage.insert(e)

        history = [
            {"robot_state": state},
            {"robot_state": _unit(32, 1)},  # won't find predecessor
        ]
        weights = [0.7, 0.3]
        spec = _trajectory_spec({"robot_state": state}, history, weights)

        results = backend.search(spec)
        assert len(results) == 1
        # Should still have a score (just from current level)
        assert results[0].score > 0


# ---------------------------------------------------------------------------
# Test: checkpoint guard
# ---------------------------------------------------------------------------


class TestCheckpointGuard:
    def test_cp3_in_chain_skipped(self):
        """Prev chain containing a CP3 entry: that entry is skipped in scoring."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        state_0 = _unit(32, 0)
        state_1 = _unit(32, 1)

        # CP3 entry (wrong checkpoint)
        cp3_entry = _make_entry("cp3_0", state_0, cp=CheckpointID.CP3, step_idx=0)
        # CP1 entry pointing to the CP3 entry
        cp1_entry = _make_entry(
            "cp1_1", state_1,
            prev_ids=["cp3_0"], step_idx=1,
        )
        storage.insert(cp3_entry)
        storage.insert(cp1_entry)

        history = [
            {"robot_state": state_1},
            {"robot_state": state_0},
        ]
        weights = [0.6, 0.4]
        spec = _trajectory_spec({"robot_state": state_1}, history, weights)

        results = backend.search(spec)
        assert len(results) >= 1
        # The CP3 entry should NOT contribute to the CP1 search
        # cp1_1 should still appear, but its trajectory score should come only
        # from level 0 (current step), not from the CP3 predecessor
        cp1_result = next(r for r in results if r.id == "cp1_1")
        assert cp1_result.score > 0


# ---------------------------------------------------------------------------
# Test: history shorter than depth
# ---------------------------------------------------------------------------


class TestPartialHistory:
    def test_short_history(self):
        """trajectory_depth=3 but only 2 history entries → partial matching."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        states = [_unit(32, i) for i in range(3)]
        for i in range(3):
            e = _make_entry(
                f"a:{i}", states[i],
                prev_ids=[f"a:{i-1}"] if i > 0 else [],
                trajectory_id="traj_a", step_idx=i,
            )
            storage.insert(e)

        # Only 2 history entries, but weights for 3
        history = [
            {"robot_state": states[2]},
            {"robot_state": states[1]},
        ]
        weights = [0.5, 0.3, 0.2]
        spec = _trajectory_spec(
            {"robot_state": states[2]}, history, weights,
        )

        results = backend.search(spec)
        # Should still return results (partial matching up to 2 levels)
        assert len(results) >= 1
        a2 = next(r for r in results if r.id == "a:2")
        assert a2.score > 0


# ---------------------------------------------------------------------------
# Test: verify weighted score is correct
# ---------------------------------------------------------------------------


class TestWeightedScoreCorrectness:
    def test_known_score_computation(self):
        """Manually verify trajectory score = sum(w_i * step_score_i)."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        # Use identical vectors so cosine similarity = 1.0 at each level
        state = _unit(32, 0)
        e0 = _make_entry("e0", state, trajectory_id="t", step_idx=0)
        e1 = _make_entry("e1", state, prev_ids=["e0"], trajectory_id="t", step_idx=1)
        storage.insert(e0)
        storage.insert(e1)

        history = [
            {"robot_state": state},  # current (matches e1)
            {"robot_state": state},  # t-1 (matches e0)
        ]
        weights = [0.6, 0.4]
        spec = _trajectory_spec({"robot_state": state}, history, weights)

        results = backend.search(spec)
        e1_result = next(r for r in results if r.id == "e1")
        # Each step has cosine=1.0, single-field search returns score=1.0
        # trajectory_score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert abs(e1_result.score - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Test: trajectory with weighted_rrf fusion
# ---------------------------------------------------------------------------


class TestTrajectoryWithRrf:
    def test_rrf_trajectory_search(self):
        """Multi-field trajectory search with RRF fusion method."""
        dims = {"robot_state": 32, "vision_0": 64}
        backend = InMemoryBackend(dims)
        storage = CacheStorage(backend)

        rs0 = _unit(32, 0)
        rs1 = _unit(32, 1)
        vis0 = torch.randn(64)
        vis0 = vis0 / vis0.norm()
        vis1 = torch.randn(64)
        vis1 = vis1 / vis1.norm()

        e0 = CacheEntry(
            id="e0", checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": rs0, "vision_0": vis0},
            payload=CachePayload(action_chunk=torch.randn(50, 32)),
            prev_ids=[], trajectory_id="t", step_idx=0,
        )
        e1 = CacheEntry(
            id="e1", checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": rs1, "vision_0": vis1},
            payload=CachePayload(action_chunk=torch.randn(50, 32)),
            prev_ids=["e0"], trajectory_id="t", step_idx=1,
        )
        storage.insert(e0)
        storage.insert(e1)

        history = [
            {"robot_state": rs1, "vision_0": vis1},
            {"robot_state": rs0, "vision_0": vis0},
        ]
        weights = [0.6, 0.4]
        spec = QuerySpec(
            query_keys={"robot_state": rs1, "vision_0": vis1},
            top_k=5,
            checkpoint_id=CheckpointID.CP1,
            fusion_method="weighted_rrf",
            fusion_weights={"robot_state": 1.0, "vision_0": 1.0},
            trajectory_history=history,
            trajectory_weights=weights,
        )

        results = backend.search(spec)
        assert len(results) >= 1
        # e1 should be top result (exact match at current + linked predecessor match)
        assert results[0].id == "e1"


# ---------------------------------------------------------------------------
# Test: empty backend returns empty
# ---------------------------------------------------------------------------


class TestTrajectoryEmpty:
    def test_empty_backend(self):
        backend = InMemoryBackend({"robot_state": 32})
        state = _unit(32, 0)
        spec = _trajectory_spec(
            {"robot_state": state},
            [{"robot_state": state}, {"robot_state": state}],
            [0.6, 0.4],
        )
        results = backend.search(spec)
        assert results == []


# ---------------------------------------------------------------------------
# Test: branching paths (multiple prev_ids)
# ---------------------------------------------------------------------------


class TestBranchingPaths:
    def test_multiple_prev_ids_takes_best(self):
        """Entry with two prev_ids: trajectory score uses max over branches."""
        backend = InMemoryBackend({"robot_state": 32})
        storage = CacheStorage(backend)

        state_current = _unit(32, 0)
        state_good_prev = _unit(32, 0)   # same as current → cosine=1.0
        state_bad_prev = -_unit(32, 0)   # opposite → cosine=-1.0

        good_prev = _make_entry("good_prev", state_good_prev, step_idx=0)
        bad_prev = _make_entry("bad_prev", state_bad_prev, step_idx=0)
        branching = _make_entry(
            "branching", state_current,
            prev_ids=["good_prev", "bad_prev"], step_idx=1,
        )
        storage.insert(good_prev)
        storage.insert(bad_prev)
        storage.insert(branching)

        history = [
            {"robot_state": state_current},
            {"robot_state": state_good_prev},  # matches good_prev well
        ]
        weights = [0.6, 0.4]
        spec = _trajectory_spec({"robot_state": state_current}, history, weights)

        results = backend.search(spec)
        br = next(r for r in results if r.id == "branching")
        # Score should use the better branch (good_prev)
        # good_prev: cosine=1.0 with history[1], bad_prev: cosine=-1.0
        assert br.score > 0.5  # must have taken the good branch


# ---------------------------------------------------------------------------
# Test: old artifact backward compat (backfill)
# ---------------------------------------------------------------------------


class TestArtifactBackfill:
    def test_old_entry_without_trajectory_fields(self, tmp_path):
        """load_artifact backfills missing trajectory fields on old entries."""
        import pickle

        # Simulate old CacheEntry without prev_ids/next_ids/trajectory_id
        state = _unit(32, 0)
        old_entry = CacheEntry(
            id="old_0",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": state},
            payload=CachePayload(action_chunk=torch.randn(50, 32)),
        )
        # Manually delete the fields to simulate old pickle
        del old_entry.prev_ids
        del old_entry.next_ids
        del old_entry.trajectory_id

        artifact = {
            "key_builder_type": "test",
            "checkpoint_id": "CP1",
            "vector_dims": {"robot_state": 32},
            "entries": [old_entry],
        }
        pkl_path = tmp_path / "old_artifact.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artifact, f)

        backend = InMemoryBackend({"robot_state": 32})
        backend.load_artifact(str(pkl_path))

        loaded = backend._entries["old_0"]
        assert loaded.prev_ids == []
        assert loaded.next_ids == []
        assert loaded.trajectory_id is None
