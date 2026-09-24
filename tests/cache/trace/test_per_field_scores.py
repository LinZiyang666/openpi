"""Backend ``per_field_scores`` capability (plan §4.4, §12-D).

InMemory: the matrix is ``[k, F]`` in the order of the given ids, WSS rows
reproduce the Layer-1 normalized scores of the search (the winner row equals
``winner_per_field``), the fused value carries the fusion weights, L2 goes
through the normalizer as a positive distance, RRF reports raw similarities
and no fused value, absent fields are masked, unknown ids raise, zero ids
give an empty matrix. Qdrant: the native score is read per field chunk with
an id filter, chunks are averaged, and no fused value is reported (mocked
client; the integration gate against a live instance is a manual step).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
from openpi.cache.types import CheckpointID

DIMS = {"vision_0": 8, "robot_state": 4}
FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}
SCORE_NORM = {
    "type": "per_field",
    "fields": {
        "vision_0": {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}},
        "robot_state": {"method": "exp_l2", "params": {"tau": 1.0}},
    },
}
WEIGHTS = {"vision_0": 0.7, "robot_state": 0.3}


def _backend(n=5, *, drop_state_for=()):
    b = InMemoryBackend(DIMS)
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        keys = {"vision_0": torch.randn(8, generator=g)}
        if i not in drop_state_for:
            keys["robot_state"] = torch.randn(4, generator=g)
        b.insert(CacheEntry(id=f"e{i}", checkpoint_id=CheckpointID.CP1, query_keys=keys,
                            payload=CachePayload(action_chunk=torch.zeros(2, 2))))
    return b


def _query(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"vision_0": torch.randn(8, generator=g), "robot_state": torch.randn(4, generator=g)}


def test_wss_matrix_matches_search_diagnostics():
    b = _backend()
    storage = CacheStorage(b)
    q = _query()
    spec = QuerySpec(query_keys=q, top_k=3, fusion_weights=WEIGHTS, fusion_method="weighted_score_sum",
                     field_similarity=FIELD_SIM, score_normalization=SCORE_NORM)
    results, feats = b.search_with_diagnostics(spec)
    ids = [r.id for r in results]
    pf = storage.per_field_scores(q, ids, field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                                  fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    assert pf.ids == tuple(ids) and pf.fields == ("vision_0", "robot_state")
    assert pf.scores.shape == (3, 2) and pf.present.all()
    assert pf.kind_by_field == {"vision_0": "normalized_layer1", "robot_state": "normalized_layer1"}
    # Row 0 is the fused winner: equals the search's own winner diagnostics.
    for f_idx, f in enumerate(pf.fields):
        assert pf.scores[0, f_idx] == pytest.approx(feats.winner_per_field[f], abs=1e-6)
    # Fused value carries the weights and reproduces the search score order.
    fused = pf.scores @ np.array([WEIGHTS[f] for f in pf.fields], dtype=np.float32)
    np.testing.assert_allclose(pf.current_step_wss, fused, atol=1e-6)
    np.testing.assert_allclose(pf.current_step_wss, [r.score for r in results], atol=1e-5)
    assert pf.not_applicable_reason is None


def test_l2_field_goes_through_normalizer_as_positive_distance():
    b = _backend()
    q = _query()
    pf = b.per_field_scores(q, ["e0"], field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                            fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    d = float(torch.norm(q["robot_state"] - b._entries["e0"].query_keys["robot_state"]))
    assert pf.scores[0, 1] == pytest.approx(np.exp(-d), abs=1e-6)


def test_rrf_reports_raw_similarities_and_no_fused_value():
    b = _backend()
    q = _query()
    pf = b.per_field_scores(q, ["e1", "e0"], field_similarity=FIELD_SIM, score_normalization=None,
                            fusion_weights=WEIGHTS, fusion_method="weighted_rrf")
    assert pf.ids == ("e1", "e0")
    assert pf.kind_by_field == {"vision_0": "raw_cosine", "robot_state": "raw_neg_l2"}
    assert pf.current_step_wss is None and "weighted_rrf" in pf.not_applicable_reason
    e1 = b._entries["e1"].query_keys
    cos = torch.nn.functional.cosine_similarity(q["vision_0"], e1["vision_0"], dim=0).item()
    assert pf.scores[0, 0] == pytest.approx(cos, abs=1e-5)
    assert pf.scores[0, 1] == pytest.approx(-float(torch.norm(q["robot_state"] - e1["robot_state"])), abs=1e-5)


def test_missing_field_is_masked_and_unknown_id_raises():
    b = _backend(drop_state_for=(2,))
    q = _query()
    pf = b.per_field_scores(q, ["e2", "e0"], field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                            fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    assert pf.present[0].tolist() == [True, False] and pf.present[1].tolist() == [True, True]
    assert pf.scores[0, 1] == 0.0
    with pytest.raises(KeyError):
        b.per_field_scores(q, ["nope"], field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                           fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    empty = b.per_field_scores(q, [], field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                               fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    assert empty.scores.shape == (0, 2)


def test_per_field_scores_does_not_touch_session_memo():
    b = _backend()
    b.open_search_session("sid")
    before = {k: dict(v) for k, v in b._score_memo.get("sid", {}).items()}
    b.per_field_scores(_query(), ["e0", "e1"], field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
                       fusion_weights=WEIGHTS, fusion_method="weighted_score_sum")
    after = {k: dict(v) for k, v in b._score_memo.get("sid", {}).items()}
    assert before == after == {}


def test_facade_raises_without_capability():
    class NoCap:
        vector_dims = DIMS
        def supported_filters(self): return frozenset()
        def insert(self, e): pass
        def search(self, spec): return []
        def fetch_payload(self, i): raise KeyError(i)
        def delete(self, ids): pass
        def count(self): return 0

    storage = CacheStorage(NoCap())
    with pytest.raises(NotImplementedError):
        storage.per_field_scores({}, [])


def test_qdrant_native_scores_mocked():
    from openpi.cache.backends.qdrant_backend import _MAX_CHUNK_DIM, QdrantVectorStore

    class _Client:
        def __init__(self):
            self.calls = []

        def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
            return [SimpleNamespace(id=i) for i in ids if i != 999]

        def get_collection(self, collection_name):
            from qdrant_client.models import Distance
            vectors = {k: SimpleNamespace(distance=Distance.COSINE) for k in
                       ("vision_0__chunk_000", "vision_0__chunk_001", "robot_state")}
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

        def query_points(self, *, collection_name, query, using, query_filter, limit, with_payload, with_vectors):
            self.calls.append((using, limit))
            # Two chunks of the vision field score 0.8 / 0.6 for point 1, one
            # chunk missing for point 2; robot_state scores 0.5 / 0.4.
            table = {
                "vision_0__chunk_000": {1: 0.8, 2: 0.7},
                "vision_0__chunk_001": {1: 0.6},
                "robot_state": {1: 0.5, 2: 0.4},
            }
            return SimpleNamespace(points=[SimpleNamespace(id=i, score=s) for i, s in table[using].items()])

    backend = QdrantVectorStore.__new__(QdrantVectorStore)
    backend._client = _Client()
    dim = 2 * _MAX_CHUNK_DIM  # exactly two chunks
    backend._config = SimpleNamespace(collection_name="c", vector_dims={"vision_0": dim, "robot_state": 4})
    q = {"vision_0": torch.zeros(dim), "robot_state": torch.zeros(4)}
    pf = backend.per_field_scores(q, ["1", "2"])
    assert pf.fields == ("robot_state", "vision_0")
    assert pf.kind_by_field == {"vision_0": "qdrant_native_chunk_mean", "robot_state": "qdrant_native"}
    assert pf.scores[0].tolist() == pytest.approx([0.5, 0.7])
    assert pf.present[0].tolist() == [True, True]
    assert pf.present[1].tolist() == [True, False]
    assert pf.current_step_wss is None
    assert pf.field_metadata["vision_0"]["chunk_count"] == 2
    assert pf.field_metadata["robot_state"]["distance_by_chunk"] == {"robot_state": "Cosine"}
    with pytest.raises(KeyError):
        backend.per_field_scores(q, ["999"])


# ---------------------------------------------------------------------------
# Qdrant integration gate (plan §12-D): a real qdrant-client local instance
# (":memory:"), the production collection schema, known vectors.
# ---------------------------------------------------------------------------


def test_qdrant_native_scores_against_known_vectors_local_instance():
    import uuid

    import numpy as np
    import torch
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from openpi.cache.backends.qdrant_backend import QdrantBackendConfig, QdrantVectorStore
    from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
    from openpi.cache.types import CheckpointID

    dims = {"vision_0": 6, "robot_state": 4}
    config = QdrantBackendConfig(collection_name="trace_gate", vector_dims=dims)
    backend = QdrantVectorStore.__new__(QdrantVectorStore)
    backend._config = config  # noqa: SLF001 - bypass the URL constructor
    backend._client = QdrantClient(":memory:")  # noqa: SLF001
    backend._client.create_collection(  # noqa: SLF001
        collection_name="trace_gate",
        vectors_config={name: VectorParams(size=d, distance=Distance.COSINE) for name, d in dims.items()},
    )
    rng = np.random.default_rng(0)
    keys = {}
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"e{i}")) for i in range(3)]
    for i, eid in enumerate(ids):
        v = rng.standard_normal(6).astype(np.float32)
        s = rng.standard_normal(4).astype(np.float32)
        keys[eid] = (v, s)
        backend.insert(
            CacheEntry(
                id=eid,
                checkpoint_id=CheckpointID.CP1,
                query_keys={"vision_0": torch.from_numpy(v), "robot_state": torch.from_numpy(s)},
                payload=CachePayload(action_chunk=torch.zeros(2, 3)),
                trajectory_id=f"t{i}",
            )
        )
    qv = rng.standard_normal(6).astype(np.float32)
    qs = rng.standard_normal(4).astype(np.float32)
    query = {"vision_0": torch.from_numpy(qv), "robot_state": torch.from_numpy(qs)}
    # the production search path runs on the same instance (RRF over two prefetches)
    results = backend.search(QuerySpec(query_keys=query, top_k=3, checkpoint_id=CheckpointID.CP1))
    assert {r.id for r in results} == set(ids)

    pf = backend.per_field_scores(query, ids)
    assert pf.fields == ("robot_state", "vision_0")
    assert pf.ids == tuple(ids) and pf.present.all()
    assert pf.kind_by_field == {"vision_0": "qdrant_native", "robot_state": "qdrant_native"}
    assert pf.current_step_wss is None and pf.not_applicable_reason

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    for r, eid in enumerate(ids):
        v, s = keys[eid]
        assert abs(pf.scores[r, pf.fields.index("vision_0")] - cos(qv, v)) < 1e-5
        assert abs(pf.scores[r, pf.fields.index("robot_state")] - cos(qs, s)) < 1e-5
    with pytest.raises(KeyError):
        backend.per_field_scores(query, [str(uuid.uuid5(uuid.NAMESPACE_URL, "nope"))])
