"""Tests for InMemoryBackend multi-field search and fusion."""

import math
import pickle

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QueryFilter,
    QuerySpec,
    SearchResultLite,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    eid: str,
    query_keys: dict[str, torch.Tensor],
    task_key: str = "",
    cp: CheckpointID = CheckpointID.CP1,
) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=cp,
        query_keys=query_keys,
        payload=CachePayload(action_chunk=torch.zeros(10, 32), task_key=task_key),
    )


def _make_spec(
    query_keys: dict[str, torch.Tensor],
    fusion_method: str | None = None,
    fusion_weights: dict[str, float] | None = None,
    field_similarity: dict | None = None,
    score_normalization: dict | None = None,
    backend_hints: dict | None = None,
    top_k: int = 10,
    cp: CheckpointID | None = CheckpointID.CP1,
    filters: QueryFilter | None = None,
) -> QuerySpec:
    return QuerySpec(
        query_keys=query_keys,
        top_k=top_k,
        checkpoint_id=cp,
        fusion_method=fusion_method,
        fusion_weights=fusion_weights,
        field_similarity=field_similarity,
        score_normalization=score_normalization,
        backend_hints=backend_hints,
        filters=filters,
    )


FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
}


# ---------------------------------------------------------------------------
# TestWeightedRrf
# ---------------------------------------------------------------------------


class TestWeightedRrf:
    def test_rrf_basic_ranking(self):
        """Two fields agree on ranking -> RRF preserves that order."""
        backend = InMemoryBackend({"vision_0": 4, "robot_state": 2})
        # Entry A: closer in both fields
        backend.insert(_make_entry("A", {
            "vision_0": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        }))
        # Entry B: farther in both fields
        backend.insert(_make_entry("B", {
            "vision_0": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "robot_state": torch.tensor([5.0, 5.0]),
        }))

        query = {
            "vision_0": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        }
        spec = _make_spec(
            query, fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 0.5, "robot_state": 0.5},
            field_similarity=FIELD_SIM,
            backend_hints={"rrf_k": 60},
        )
        results = backend.search(spec)
        assert len(results) == 2
        assert results[0].id == "A"

    def test_rrf_conflicting_fields(self):
        """Fields disagree; higher-weight field dominates."""
        backend = InMemoryBackend({"vision_0": 2, "robot_state": 2})
        # A: good vision, bad robot_state
        backend.insert(_make_entry("A", {
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([10.0, 10.0]),
        }))
        # B: bad vision, good robot_state
        backend.insert(_make_entry("B", {
            "vision_0": torch.tensor([0.0, 1.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        }))

        query = {"vision_0": torch.tensor([1.0, 0.0]), "robot_state": torch.tensor([0.0, 0.0])}

        # High vision weight -> A wins
        spec = _make_spec(
            query, fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 0.9, "robot_state": 0.1},
            field_similarity=FIELD_SIM,
            backend_hints={"rrf_k": 60},
        )
        results = backend.search(spec)
        assert results[0].id == "A"

        # High robot_state weight -> B wins
        spec2 = _make_spec(
            query, fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 0.1, "robot_state": 0.9},
            field_similarity=FIELD_SIM,
            backend_hints={"rrf_k": 60},
        )
        results2 = backend.search(spec2)
        assert results2[0].id == "B"

    def test_zero_weight_field_skipped(self):
        """weight=0 field should not affect results."""
        backend = InMemoryBackend({"vision_0": 2, "robot_state": 2})
        backend.insert(_make_entry("A", {
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([100.0, 100.0]),  # very far
        }))
        backend.insert(_make_entry("B", {
            "vision_0": torch.tensor([0.0, 1.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        }))

        query = {"vision_0": torch.tensor([1.0, 0.0]), "robot_state": torch.tensor([0.0, 0.0])}
        # robot_state weight=0 -> only vision matters -> A should win
        spec = _make_spec(
            query, fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 1.0, "robot_state": 0.0},
            field_similarity=FIELD_SIM,
            backend_hints={"rrf_k": 60},
        )
        results = backend.search(spec)
        assert results[0].id == "A"


# ---------------------------------------------------------------------------
# TestWeightedScoreSum
# ---------------------------------------------------------------------------


class TestWeightedScoreSum:
    def test_sum_basic(self):
        """Manually verify score sum ranking."""
        backend = InMemoryBackend({"vision_0": 2, "robot_state": 2})
        # A: identical vision, close robot_state
        backend.insert(_make_entry("A", {
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([0.1, 0.0]),
        }))
        # B: different vision, identical robot_state
        backend.insert(_make_entry("B", {
            "vision_0": torch.tensor([0.0, 1.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        }))

        query = {"vision_0": torch.tensor([1.0, 0.0]), "robot_state": torch.tensor([0.0, 0.0])}
        spec = _make_spec(
            query, fusion_method="weighted_score_sum",
            fusion_weights={"vision_0": 0.7, "robot_state": 0.3},
            field_similarity=FIELD_SIM,
            score_normalization={"type": "none"},
        )
        results = backend.search(spec)
        assert results[0].id == "A"  # High cosine for vision dominates

    def test_tau_effect(self):
        """Verify exp(-d/tau) conversion for robot_state."""
        backend = InMemoryBackend({"robot_state": 2})
        backend.insert(_make_entry("near", {"robot_state": torch.tensor([0.1, 0.0])}))
        backend.insert(_make_entry("far", {"robot_state": torch.tensor([5.0, 5.0])}))

        query = {"robot_state": torch.tensor([0.0, 0.0])}
        spec = _make_spec(
            query, fusion_method="weighted_score_sum",
            fusion_weights={"robot_state": 1.0},
            field_similarity={"robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}}},
            score_normalization={"type": "none"},
        )
        results = backend.search(spec)
        assert results[0].id == "near"
        # Verify approximate score: exp(-0.1/0.334717) ~ 0.741
        near_dist = 0.1
        expected = math.exp(-near_dist / 0.334717)
        assert abs(results[0].score - expected) < 0.05

    def test_percentile_normalization(self):
        """Verify p5/p95 clipping."""
        backend = InMemoryBackend({"vision_0": 2})
        # Create entries spanning cosine range
        backend.insert(_make_entry("A", {"vision_0": torch.tensor([1.0, 0.0])}))
        backend.insert(_make_entry("B", {"vision_0": torch.tensor([0.5, 0.5])}))

        query = {"vision_0": torch.tensor([1.0, 0.0])}
        spec = _make_spec(
            query, fusion_method="weighted_score_sum",
            fusion_weights={"vision_0": 1.0},
            field_similarity={"vision_0": {"type": "cosine"}},
            score_normalization={
                "type": "percentile",
                "fields": {"vision_0": {"p5": 0.5, "p95": 1.0}},
            },
        )
        results = backend.search(spec)
        # A has cosine=1.0 -> (1+1)/2=1.0 -> (1.0-0.5)/(1.0-0.5)=1.0
        # B has cosine~0.707 -> (0.707+1)/2=0.854 -> (0.854-0.5)/0.5=0.707
        assert results[0].id == "A"
        assert results[0].score > results[1].score


# ---------------------------------------------------------------------------
# TestFiltering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_task_key_filter(self):
        backend = InMemoryBackend({"vision_0": 2})
        backend.insert(_make_entry("A", {"vision_0": torch.tensor([1.0, 0.0])}, task_key="task1"))
        backend.insert(_make_entry("B", {"vision_0": torch.tensor([1.0, 0.0])}, task_key="task2"))

        query = {"vision_0": torch.tensor([1.0, 0.0])}
        spec = _make_spec(
            query, fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 1.0},
            field_similarity={"vision_0": {"type": "cosine"}},
            filters=QueryFilter(task_key="task1"),
        )
        results = backend.search(spec)
        assert len(results) == 1
        assert results[0].id == "A"

    def test_checkpoint_id_filter(self):
        backend = InMemoryBackend({"vision_0": 2})
        backend.insert(_make_entry("A", {"vision_0": torch.tensor([1.0, 0.0])}, cp=CheckpointID.CP1))
        backend.insert(_make_entry("B", {"vision_0": torch.tensor([1.0, 0.0])}, cp=CheckpointID.CP3))

        query = {"vision_0": torch.tensor([1.0, 0.0])}
        spec = _make_spec(query, cp=CheckpointID.CP1)
        results = backend.search(spec)
        assert all(r.checkpoint_id == CheckpointID.CP1 for r in results)


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_fusion_method_uses_single_field(self):
        """fusion_method=None -> original single-field cosine behavior."""
        backend = InMemoryBackend({"robot_state": 32})
        v1 = torch.randn(32)
        v2 = torch.randn(32)
        backend.insert(_make_entry("A", {"robot_state": v1}))
        backend.insert(_make_entry("B", {"robot_state": v2}))

        spec = _make_spec({"robot_state": v1}, fusion_method=None, top_k=1)
        results = backend.search(spec)
        assert len(results) == 1
        assert results[0].id == "A"  # Exact match


# ---------------------------------------------------------------------------
# TestArtifactLoading
# ---------------------------------------------------------------------------


class TestArtifactLoading:
    def test_load_artifact(self, tmp_path):
        dims = {"vision_0": 2, "robot_state": 2}
        entries = [
            _make_entry("A", {"vision_0": torch.tensor([1.0, 0.0]), "robot_state": torch.tensor([0.0, 0.0])}),
        ]
        artifact = {
            "key_builder_type": "cp1_mean_pool",
            "checkpoint_id": "CP1",
            "vector_dims": dims,
            "entries": entries,
        }
        pkl_path = tmp_path / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artifact, f)

        backend = InMemoryBackend(dims)
        backend.load_artifact(str(pkl_path))
        assert backend.count() == 1

    def test_load_artifact_dims_mismatch(self, tmp_path):
        artifact = {
            "key_builder_type": "cp1_mean_pool",
            "checkpoint_id": "CP1",
            "vector_dims": {"vision_0": 999},
            "entries": [],
        }
        pkl_path = tmp_path / "bad.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artifact, f)

        backend = InMemoryBackend({"vision_0": 2})
        with pytest.raises(ValueError, match="mismatch"):
            backend.load_artifact(str(pkl_path))


# ---------------------------------------------------------------------------
# TestEmptyBackend
# ---------------------------------------------------------------------------


class TestEmptyBackend:
    def test_search_empty(self):
        backend = InMemoryBackend({"vision_0": 2})
        spec = _make_spec(
            {"vision_0": torch.tensor([1.0, 0.0])},
            fusion_method="weighted_rrf",
            fusion_weights={"vision_0": 1.0},
            field_similarity={"vision_0": {"type": "cosine"}},
        )
        assert backend.search(spec) == []
