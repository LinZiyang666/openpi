"""Text-IVF bucket index tests (InMemoryBackend).

Covers: bucket build correctness, exact / nearest probe, deterministic
tie-break, bucket-count cap, missing-field fail-fast, empty library, mutation
invalidation, lazy-build idempotency + atomic publish, frozen lazy build,
in-bucket filter stacking, and BackendPool fingerprint isolation.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from openpi.cache.backend_pool import BackendFingerprint
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.config import BackendConfig, InMemoryConfig, TextIvfIndexConfig
from openpi.cache.storage_types import CacheEntry, CachePayload, QueryFilter, QuerySpec
from openpi.cache.types import CheckpointID

DIMS = {"prompt_emb": 4, "robot_state": 2}

PROMPT_A = [1.0, 0.0, 0.0, 0.0]
PROMPT_B = [0.0, 1.0, 0.0, 0.0]


def _entry(eid: str, prompt: list[float], step_idx: int, task: str = "t") -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={
            "prompt_emb": torch.tensor(prompt, dtype=torch.float32),
            "robot_state": torch.tensor([0.1 * step_idx, 0.2], dtype=torch.float32),
        },
        payload=CachePayload(action_chunk=torch.zeros(5, 3), task_key=task),
        step_idx=step_idx,
    )


def _backend(max_buckets: int = 8) -> InMemoryBackend:
    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig(max_buckets=max_buckets))
    for i in range(3):
        be.insert(_entry(f"a{i}", PROMPT_A, i, task="taskA"))
    for i in range(3):
        be.insert(_entry(f"b{i}", PROMPT_B, i, task="taskB"))
    return be


def _spec(prompt: list[float] | torch.Tensor, **kwargs) -> QuerySpec:
    defaults = dict(
        query_keys={
            "prompt_emb": torch.as_tensor(prompt, dtype=torch.float32),
            "robot_state": torch.tensor([0.05, 0.2]),
        },
        top_k=6,
        checkpoint_id=CheckpointID.CP1,
        fusion_method="weighted_score_sum",
        fusion_weights={"robot_state": 1.0, "prompt_emb": 0.0},
        field_similarity={
            "robot_state": {"type": "l2"},
            "prompt_emb": {"type": "cosine"},
        },
        score_normalization={
            "type": "per_field",
            "fields": {
                "robot_state": {"method": "exp_l2", "params": {"tau": 1.0}},
                "prompt_emb": {"method": "affine_clip", "params": {"lo": 0.0, "hi": 1.0}},
            },
        },
        backend_hints={"text_ivf": True},
    )
    defaults.update(kwargs)
    return QuerySpec(**defaults)


def test_bucket_build_groups_by_bytes():
    be = _backend()
    state = be._build_text_ivf_index()
    buckets = state[0]
    assert len(buckets) == 2
    assert sorted(len(v) for v in buckets.values()) == [3, 3]


def test_exact_probe_returns_own_bucket_only():
    be = _backend()
    ids = {r.id for r in be.search(_spec(PROMPT_A))}
    assert ids == {"a0", "a1", "a2"}


def test_nearest_probe_routes_to_closest_bucket():
    be = _backend()
    ids = {r.id for r in be.search(_spec([0.9, 0.1, 0.0, 0.0]))}
    assert ids == {"a0", "a1", "a2"}


def test_nearest_probe_tie_breaks_by_smallest_bucket_key():
    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig(max_buckets=8))
    be.insert(_entry("x", [1.0, 0.0, 0.0, 0.0], 0))
    be.insert(_entry("y", [0.0, 1.0, 0.0, 0.0], 0))
    # Equidistant query: cosine to both buckets identical.
    res = be.search(_spec([1.0, 1.0, 0.0, 0.0]))
    assert len(res) == 1
    state = be._text_ivf_state
    expected_bucket = state[0][state[1][0]]  # smallest byte key
    assert res[0].id == expected_bucket[0]
    # Determinism across rebuilds.
    be._invalidate_frozen_search_caches()
    res2 = be.search(_spec([1.0, 1.0, 0.0, 0.0]))
    assert res2[0].id == res[0].id


def test_bucket_cap_exceeded_raises():
    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig(max_buckets=1))
    be.insert(_entry("x", PROMPT_A, 0))
    be.insert(_entry("y", PROMPT_B, 0))
    with pytest.raises(ValueError, match="max_buckets"):
        be.search(_spec(PROMPT_A))


def test_missing_screening_field_raises():
    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig())
    e = _entry("x", PROMPT_A, 0)
    del e.query_keys["prompt_emb"]
    be.insert(e)
    with pytest.raises(ValueError, match="lacks screening field"):
        be.search(_spec(PROMPT_A))


def test_empty_library_probe_returns_empty():
    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig())
    assert be.search(_spec(PROMPT_A)) == []


def test_hint_without_index_config_raises():
    be = InMemoryBackend(DIMS)  # no text_ivf config
    be.insert(_entry("x", PROMPT_A, 0))
    with pytest.raises(RuntimeError, match="no.*text_ivf index"):
        be.search(_spec(PROMPT_A))


def test_mutation_invalidates_index():
    be = _backend()
    be.search(_spec(PROMPT_A))
    assert be._text_ivf_state is not None
    be.insert(_entry("a9", PROMPT_A, 9, task="taskA"))
    assert be._text_ivf_state is None  # invalidated
    ids = {r.id for r in be.search(_spec(PROMPT_A))}
    assert "a9" in ids  # rebuilt with the new entry


def test_lazy_build_idempotent():
    be = _backend()
    s1 = be._build_text_ivf_index()
    s2 = be._build_text_ivf_index()
    assert s1[0] == s2[0]
    assert s1[1] == s2[1]
    assert torch.equal(s1[2], s2[2])


def test_lazy_build_publishes_atomically(monkeypatch):
    """During a build, _text_ivf_state is either None or a complete state."""
    be = _backend()
    observed: list = []
    orig_sorted = sorted

    def spy_sorted(x, **kw):
        # Called mid-build (keys_sorted): the public slot must still be unset.
        observed.append(be._text_ivf_state)
        return orig_sorted(x, **kw)

    import openpi.cache.backends.in_memory_backend as mod
    monkeypatch.setattr(mod, "sorted", spy_sorted, raising=False)
    be._build_text_ivf_index()
    assert observed and all(v is None for v in observed)
    assert be._text_ivf_state is not None


def test_frozen_backend_lazy_build_allowed():
    be = _backend()
    be.freeze()
    ids = {r.id for r in be.search(_spec(PROMPT_A))}
    assert ids == {"a0", "a1", "a2"}


def test_step_range_filter_applies_inside_bucket():
    be = _backend()
    res = be.search(_spec(PROMPT_A, filters=QueryFilter(step_range=(1, 2))))
    assert {r.id for r in res} == {"a1", "a2"}


def test_load_artifact_eager_build_and_cap(tmp_path):
    entries = [_entry("x", PROMPT_A, 0), _entry("y", PROMPT_B, 0)]
    art = {
        "key_builder_type": "cp1_mean_pool",
        "checkpoint_id": "CP1",
        "vector_dims": DIMS,
        "entries": entries,
        "prompt_pool": {"masked": True, "instruction_span": False},
    }
    path = tmp_path / "a.pkl"
    with open(path, "wb") as fh:
        pickle.dump(art, fh)

    be = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig(max_buckets=8))
    be.load_artifact(str(path))
    assert be._text_ivf_state is not None  # eager build
    assert be.artifact_meta["prompt_pool"] == {"masked": True, "instruction_span": False}

    be_cap = InMemoryBackend(DIMS, text_ivf=TextIvfIndexConfig(max_buckets=1))
    with pytest.raises(ValueError, match="max_buckets"):
        be_cap.load_artifact(str(path))


def test_fingerprint_isolates_text_ivf_params(tmp_path):
    path = tmp_path / "a.pkl"
    path.write_bytes(b"x")  # fingerprint only resolves the path
    base = dict(preload_path=str(path))
    cfg1 = BackendConfig(type="in_memory", vector_dims=DIMS,
                         in_memory=InMemoryConfig(**base, index_type="text_ivf",
                                                  text_ivf=TextIvfIndexConfig(max_buckets=8)))
    cfg2 = BackendConfig(type="in_memory", vector_dims=DIMS,
                         in_memory=InMemoryConfig(**base, index_type="text_ivf",
                                                  text_ivf=TextIvfIndexConfig(max_buckets=16)))
    cfg3 = BackendConfig(type="in_memory", vector_dims=DIMS,
                         in_memory=InMemoryConfig(**base))
    fp1 = BackendFingerprint.from_config(cfg1)
    fp2 = BackendFingerprint.from_config(cfg2)
    fp3 = BackendFingerprint.from_config(cfg3)
    assert fp1 != fp2  # different max_buckets must not share an instance
    assert fp1 != fp3 and fp2 != fp3
    assert fp3.text_ivf_params is None  # legacy fingerprints unchanged
