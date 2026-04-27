"""End-to-end artifact round-trip with B2 verdict-factor enrichment.

Covers the full lifecycle: build minimal artifact dict → run
`enrich_artifact_with_factors` → pickle → load via
`InMemoryBackend.load_artifact` → verify `payload.factors` + facade
`library_stats` are intact.

Also exercises the legacy fallback (no `library_stats` field) and the
F1b OnlineExtractor read-after-load path.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch

from exp.common.factor_postprocess import enrich_artifact_with_factors
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.factors.source_window import (
    SourceWindowSmoothnessAction,
)
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Fixture: a tiny in-memory chain
# ------------------------------------------------------------------


def _build_chain(n: int = 5, action_dim: int = 2) -> list[CacheEntry]:
    """Linear-motion chain. Action chunks are torch tensors here; the
    detach-then-enrich tests write numpy variants."""
    traj = "traj-0"
    entries = []
    for t in range(n):
        action_chunk = torch.tensor([[float(t), 0.0] + [0.0] * (action_dim - 2)])
        entries.append(
            CacheEntry(
                id=f"{traj}:{t}",
                checkpoint_id=CheckpointID.CP1,
                query_keys={"robot_state": torch.zeros(action_dim)},
                payload=CachePayload(action_chunk=action_chunk),
                step_idx=t, trajectory_id=traj,
            )
        )
    # Wire prev/next ids
    for i in range(n):
        if i > 0:
            entries[i].prev_ids = [entries[i - 1].id]
        if i < n - 1:
            entries[i].next_ids = [entries[i + 1].id]
    return entries


# ------------------------------------------------------------------
# Round-trip: enrich → pickle → load → verify
# ------------------------------------------------------------------


def test_round_trip_with_f1b_writer_writes_and_loads_factors(tmp_path):
    entries = _build_chain(n=5, action_dim=2)
    writer = SourceWindowSmoothnessAction(
        windows=[(1, 1)], descriptors=["jerk"], active_eps=0.01,
    )

    library_stats = enrich_artifact_with_factors(entries, [writer])

    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
        "library_stats": library_stats,
    }
    p = tmp_path / "artifact.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend = InMemoryBackend({"robot_state": 2})
    backend.load_artifact(str(p))
    storage = CacheStorage(backend)

    # Backend exposes library_stats; facade duck-types through.
    assert storage.library_stats is not None
    assert storage.library_stats.action_sigma.shape == (2,)

    # Interior entries carry finite jerk; boundary entries carry NaN.
    interior = backend.fetch_entry("traj-0:2")
    assert "f1b_a_jerk__p1_f1" in interior.payload.factors
    assert not math.isnan(interior.payload.factors["f1b_a_jerk__p1_f1"])
    boundary = backend.fetch_entry("traj-0:0")
    assert math.isnan(boundary.payload.factors["f1b_a_jerk__p1_f1"])


def test_round_trip_handles_numpy_payload_post_detach(tmp_path):
    """Simulate the primary builder path: subprocess emits numpy entries.
    Helper bridges via torch.as_tensor; round-trip succeeds."""
    entries = _build_chain(n=5, action_dim=2)
    # Detach to numpy (mimic _detach_entries done inside subprocess)
    for e in entries:
        e.payload.action_chunk = e.payload.action_chunk.numpy()
        e.query_keys = {k: v.numpy() for k, v in e.query_keys.items()}

    writer = SourceWindowSmoothnessAction(
        windows=[(1, 1)], descriptors=["jerk"], active_eps=0.01,
    )

    library_stats = enrich_artifact_with_factors(entries, [writer])
    assert library_stats.action_sigma.shape == (2,)

    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
        "library_stats": library_stats,
    }
    p = tmp_path / "artifact.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend = InMemoryBackend({"robot_state": 2})
    backend.load_artifact(str(p))

    # load_artifact converts numpy → torch on read; payload.factors are
    # plain Python floats so they survive pickle round-trip.
    interior = backend.fetch_entry("traj-0:2")
    assert isinstance(interior.payload.action_chunk, torch.Tensor)
    assert isinstance(interior.payload.factors["f1b_a_jerk__p1_f1"], float)


# ------------------------------------------------------------------
# F1b OnlineExtractor reads payload.factors back out
# ------------------------------------------------------------------


def test_round_trip_online_extractor_reads_offline_written_factors(tmp_path):
    entries = _build_chain(n=4, action_dim=2)
    writer = SourceWindowSmoothnessAction(
        windows=[(1, 1)], descriptors=["jerk"], active_eps=0.01,
    )
    library_stats = enrich_artifact_with_factors(entries, [writer])
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
        "library_stats": library_stats,
    }
    p = tmp_path / "artifact.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend = InMemoryBackend({"robot_state": 2})
    backend.load_artifact(str(p))
    storage = CacheStorage(backend)

    # Online F1b reads pre-computed factors via PayloadView.
    from openpi.cache.components.payload_view import StoragePayloadView

    online = SourceWindowSmoothnessAction(
        windows=[(1, 1)], descriptors=["jerk"], active_eps=0.01,
        library_stats=library_stats,
    )
    view = StoragePayloadView(storage)
    out = online.extract(
        [SearchResultLite(id="traj-0:1", score=1.0, checkpoint_id=CheckpointID.CP1)],
        view, None, {},
    )
    # Whatever offline computed for entry 1, online sees byte-identical
    expected = entries[1].payload.factors["f1b_a_jerk__p1_f1"]
    assert out["f1b_a_jerk__p1_f1"] == expected


# ------------------------------------------------------------------
# Legacy artifact: no library_stats field → fallback compute
# ------------------------------------------------------------------


def test_legacy_artifact_without_library_stats_falls_back(tmp_path):
    entries = _build_chain(n=3, action_dim=2)
    # Simulate legacy artifact: NO `library_stats` key, no payload.factors
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
    }
    p = tmp_path / "legacy.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend = InMemoryBackend({"robot_state": 2})
    backend.load_artifact(str(p))

    # Fallback ran — non-None stats now
    assert backend.library_stats is not None
    # F1b OnlineExtractor reading from legacy entries must see NaN
    from openpi.cache.components.payload_view import StoragePayloadView

    online = SourceWindowSmoothnessAction(
        windows=[(1, 1)], descriptors=["jerk"], active_eps=0.01,
        library_stats=backend.library_stats,
    )
    view = StoragePayloadView(CacheStorage(backend))
    out = online.extract(
        [SearchResultLite(id="traj-0:1", score=1.0, checkpoint_id=CheckpointID.CP1)],
        view, None, {},
    )
    assert math.isnan(out["f1b_a_jerk__p1_f1"])
