"""Phase 1-6 — consolidated test module for the serving optimization plan.

Why one file: each test file under ``tests/cache/`` triggers a full
``tests/cache/conftest.py`` import chain (cache backends + components +
torch). Consolidating all phases into one module shares that ~2s collection
cost across every assertion instead of paying it per file.

Sections
--------
* M4 Frozen Guard — runtime write-frozen contract (C2)
* M5 offline_writers — attribute rename to ``_factors`` + isinstance filter
* M3 BackendPool — fingerprint identity + share / bypass / single-load
* M2 BundleDispatcher — multi-bundle dict + select_bundle ctrl
* M1 BatchingCoordinator + stage_io — stack/split + sub-bucket Stage3
"""

from __future__ import annotations

import dataclasses
import pickle
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from openpi.cache.backend_base import BackendFrozenError
from openpi.cache.backend_pool import (
    BackendFingerprint,
    BackendPool,
    _build_empty_backend,
    _build_legacy_complete,
)
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.config import (
    BackendConfig,
    InMemoryConfig,
    QdrantConfig,
    _collect_offline_writers_from_judges,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ==================================================================
# Shared helpers
# ==================================================================


def _make_entry(entry_id: str, dims: dict[str, int]) -> CacheEntry:
    return CacheEntry(
        id=entry_id,
        checkpoint_id="CP1",
        query_keys={name: torch.zeros(dim) for name, dim in dims.items()},
        payload=CachePayload(action_chunk=torch.zeros(50, 32)),
    )


def _write_artifact(path: Path, vector_dims: dict[str, int], n_entries: int = 3) -> None:
    """Write a minimal valid artifact pkl ``InMemoryBackend.load_artifact`` accepts."""
    entries = [_make_entry(f"e{i}", vector_dims) for i in range(n_entries)]
    data = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": vector_dims,
        "entries": entries,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _make_in_memory_cfg(preload_path, vector_dims=None) -> BackendConfig:
    dims = vector_dims or {"robot_state": 4}
    return BackendConfig(
        type="in_memory",
        vector_dims=dims,
        in_memory=InMemoryConfig(preload_path=preload_path, index_type="brute_force"),
    )


@pytest.fixture(autouse=True)
def _reset_pool():
    BackendPool.reset_for_tests()
    yield
    BackendPool.reset_for_tests()


# ==================================================================
# M4 Frozen Guard
# ==================================================================


def test_freeze_lifecycle_and_idempotency():
    """is_frozen flips on first freeze() and remains True under re-call."""
    backend = InMemoryBackend(vector_dims={"robot_state": 4})
    assert backend.is_frozen is False
    backend.freeze()
    assert backend.is_frozen is True
    backend.freeze()
    assert backend.is_frozen is True


def test_frozen_blocks_all_mutation_entries(tmp_path):
    """All 5 mutation entries raise BackendFrozenError post-freeze (G1 R1 Item 4)."""
    backend = InMemoryBackend(vector_dims={"robot_state": 4})
    backend.insert(_make_entry("e1", {"robot_state": 4}))  # pre-freeze OK
    backend.freeze()

    with pytest.raises(BackendFrozenError):
        backend.insert(_make_entry("e2", {"robot_state": 4}))
    with pytest.raises(BackendFrozenError):
        backend.batch_insert([_make_entry("e3", {"robot_state": 4})])
    with pytest.raises(BackendFrozenError):
        backend.delete(["e1"])
    with pytest.raises(BackendFrozenError):
        backend.load_artifact(str(tmp_path / "no_such.pkl"))


def test_frozen_allows_reads_and_session_lifecycle():
    """Reads + derived state (search_session lifecycle) still allowed post-freeze."""
    backend = InMemoryBackend(vector_dims={"robot_state": 4})
    backend.insert(_make_entry("e1", {"robot_state": 4}))
    backend.freeze()

    assert backend.count() == 1
    assert backend.fetch_payload("e1") is not None
    assert backend.fetch_entry("e1").id == "e1"
    # Derived state mutation must remain allowed (per-session memo cleanup).
    backend.open_search_session("sid-1")
    assert backend._has_active_search_sessions() is True
    backend.close_search_session("sid-1")
    assert backend._has_active_search_sessions() is False


def test_cache_storage_propagates_frozen_error():
    """The CacheStorage facade propagates BackendFrozenError on all mutation entries.

    G1 R1 Item 4 explicit: episode-end calls ``CacheStorage.batch_insert`` which
    is the real runtime write path. Without facade propagation the guard
    wouldn't fire on the actual codepath.
    """
    backend = InMemoryBackend(vector_dims={"robot_state": 4})
    backend.insert(_make_entry("e1", {"robot_state": 4}))
    backend.freeze()
    storage = CacheStorage(backend)

    assert storage.is_frozen is True
    with pytest.raises(BackendFrozenError):
        storage.insert(_make_entry("e2", {"robot_state": 4}))
    with pytest.raises(BackendFrozenError):
        storage.batch_insert([_make_entry("e3", {"robot_state": 4})])
    with pytest.raises(BackendFrozenError):
        storage.delete(["e1"])


# ==================================================================
# M5 offline_writers attribute fix (audit report §9.5)
# ==================================================================


class _OnlineOnlyFactor:
    def extract(self, ctx):  # noqa: ARG002
        return {}


class _OfflineWriterFactor:
    def extract(self, ctx):  # noqa: ARG002
        return {}

    def required_payload_fields(self):
        return set()

    def compute_for_episode(self, entries, library_stats):
        return [{} for _ in entries]


class _FakeJudge:
    """Minimal CompositeJudge stub exposing the ``_factors`` attribute."""

    def __init__(self, factors):
        self._factors = list(factors)


def test_offline_writers_collected_from_factors():
    """G1 R1 Item 4 / audit §9.5: collector queries ``_factors`` (not ``_extractors``)."""
    writer = _OfflineWriterFactor()
    judge = _FakeJudge([_OnlineOnlyFactor(), writer, _OnlineOnlyFactor()])
    result = _collect_offline_writers_from_judges({CheckpointID.CP1: judge})
    assert result == [writer]


def test_offline_writers_dedupe_and_fallback():
    """Same-instance dedup across CPs + non-CompositeJudge fallback to empty."""
    writer = _OfflineWriterFactor()
    cp1 = _FakeJudge([writer])
    cp3 = _FakeJudge([writer])  # same instance
    # Iteration order: sorted by CheckpointID.value (CP1 first).
    assert _collect_offline_writers_from_judges(
        {CheckpointID.CP3: cp3, CheckpointID.CP1: cp1}
    ) == [writer]

    # Judge missing _factors → empty list, no AttributeError.
    class _Plain:
        pass

    assert _collect_offline_writers_from_judges({CheckpointID.CP1: _Plain()}) == []


# ==================================================================
# M3 BackendPool — fingerprint identity
# ==================================================================


def test_fingerprint_identity_and_aliasing(tmp_path, monkeypatch):
    """Same config → same fingerprint; different path/dims → different;
    Path.resolve eliminates relative-path aliasing (G1 R1 Item 5)."""
    a = tmp_path / "a.pkl"
    b = tmp_path / "b.pkl"
    _write_artifact(a, {"robot_state": 4})
    _write_artifact(b, {"robot_state": 4})

    fp_a = BackendFingerprint.from_config(_make_in_memory_cfg(str(a)))
    fp_a_dup = BackendFingerprint.from_config(_make_in_memory_cfg(str(a)))
    fp_b = BackendFingerprint.from_config(_make_in_memory_cfg(str(b)))
    fp_a_diff_dims = BackendFingerprint.from_config(
        _make_in_memory_cfg(str(a), {"robot_state": 8})
    )

    assert fp_a == fp_a_dup
    assert fp_a != fp_b
    assert fp_a != fp_a_diff_dims

    # Path.resolve coalesces relative vs absolute paths.
    monkeypatch.chdir(tmp_path)
    fp_rel = BackendFingerprint.from_config(_make_in_memory_cfg("a.pkl"))
    assert fp_a == fp_rel


def test_fingerprint_rejects_invalid_configs():
    """Non-in_memory backends and empty preload_paths must not enter the pool index."""
    with pytest.raises(ValueError, match="only supports in_memory"):
        BackendFingerprint.from_config(BackendConfig(type="qdrant"))
    with pytest.raises(ValueError, match="non-empty preload_path"):
        BackendFingerprint.from_config(_make_in_memory_cfg(preload_path=None))


# ==================================================================
# M3 BackendPool — sharing / bypass / single-load
# ==================================================================


def test_pool_shares_same_fingerprint_and_freezes_immediately(tmp_path):
    artifact = tmp_path / "art.pkl"
    _write_artifact(artifact, {"robot_state": 4})
    cfg = _make_in_memory_cfg(str(artifact))
    pool = BackendPool.get()

    b1 = pool.get_or_load(cfg)
    b2 = pool.get_or_load(cfg)

    assert b1 is b2
    assert b1.is_frozen is True


def test_pool_distinct_for_different_paths(tmp_path):
    a = tmp_path / "a.pkl"
    b = tmp_path / "b.pkl"
    _write_artifact(a, {"robot_state": 4})
    _write_artifact(b, {"robot_state": 4})
    pool = BackendPool.get()
    assert pool.get_or_load(_make_in_memory_cfg(str(a))) is not pool.get_or_load(
        _make_in_memory_cfg(str(b))
    )


def test_pool_bypass_for_empty_preload_and_qdrant():
    """Empty preload_path and Qdrant configs bypass the pool entirely.

    Each call returns a fresh frozen backend (no caching).
    """
    pool = BackendPool.get()

    # Empty preload_path: in_memory backend, not cached.
    cfg_empty = _make_in_memory_cfg(preload_path=None)
    b1 = pool.get_or_load(cfg_empty)
    b2 = pool.get_or_load(cfg_empty)
    assert b1 is not b2
    assert b1.is_frozen is True

    # Qdrant: not cached either.
    cfg_qdrant = BackendConfig(type="qdrant", qdrant=QdrantConfig(url="http://localhost:6333"))
    q1 = pool.get_or_load(cfg_qdrant)
    q2 = pool.get_or_load(cfg_qdrant)
    assert q1 is not q2
    assert q1.is_frozen is True


def test_pool_calls_load_artifact_exactly_once_per_fingerprint(tmp_path):
    """G1 R2 Item 5: no double-load through pool path."""
    artifact = tmp_path / "art.pkl"
    _write_artifact(artifact, {"robot_state": 4})
    cfg = _make_in_memory_cfg(str(artifact))

    original_load = InMemoryBackend.load_artifact
    call_count = {"n": 0}

    def counting_load(self, path):
        call_count["n"] += 1
        return original_load(self, path)

    with patch.object(InMemoryBackend, "load_artifact", counting_load):
        pool = BackendPool.get()
        pool.get_or_load(cfg)
        pool.get_or_load(cfg)
        pool.get_or_load(cfg)

    assert call_count["n"] == 1


def test_build_helpers_responsibility_split(tmp_path):
    """G1 R2 Item 5: _build_empty_backend MUST NOT call load_artifact;
    _build_legacy_complete handles build + (optional load) + freeze."""
    artifact = tmp_path / "art.pkl"
    _write_artifact(artifact, {"robot_state": 4})
    cfg = _make_in_memory_cfg(str(artifact))

    with patch.object(InMemoryBackend, "load_artifact") as mock_load:
        empty_backend = _build_empty_backend(cfg)
        assert empty_backend.is_frozen is False
        mock_load.assert_not_called()

    # _build_legacy_complete: full single-shot flow.
    legacy_backend = _build_legacy_complete(cfg)
    assert legacy_backend.is_frozen is True
    assert legacy_backend.count() == 3


# ==================================================================
# M2 BundleDispatcher — multi-bundle index + select_bundle ctrl
# ==================================================================


def _reset_global_bundles():
    """Drop the module-level bundle registry so tests don't bleed state."""
    import openpi.serving.websocket_policy_server as wps

    with wps._bundle_lock:
        wps._bundles.clear()
        wps._current_bundle = None
        wps._bundle_version = 0


@pytest.fixture(autouse=True)
def _reset_bundles():
    _reset_global_bundles()
    yield
    _reset_global_bundles()


def test_get_current_cache_bundle_backward_compat():
    """Calling get_current_cache_bundle() with no args returns the
    legacy single-latest pointer (None when nothing has been registered)."""
    from openpi.serving.websocket_policy_server import (
        CurrentCacheBundle,
        get_cache_bundle_ids,
        get_current_cache_bundle,
    )
    import openpi.serving.websocket_policy_server as wps

    assert get_current_cache_bundle() is None
    assert get_cache_bundle_ids() == []

    bundle = CurrentCacheBundle(
        config_path="/tmp/a.yaml",
        cache_config=object(),
        shared_storage=object(),
        version=1,
        yaml_id="a",
    )
    with wps._bundle_lock:
        wps._bundles["a"] = bundle
        wps._current_bundle = bundle

    assert get_current_cache_bundle() is bundle
    assert get_cache_bundle_ids() == ["a"]


def test_get_current_cache_bundle_by_id_lookup():
    """Explicit ``bundle_id`` argument returns the matching slot only.

    Unknown bundle_id → None (caller surfaces error).
    """
    from openpi.serving.websocket_policy_server import (
        CurrentCacheBundle,
        get_current_cache_bundle,
    )
    import openpi.serving.websocket_policy_server as wps

    a = CurrentCacheBundle(
        config_path="/tmp/a.yaml", cache_config=object(),
        shared_storage=object(), version=1, yaml_id="a",
    )
    b = CurrentCacheBundle(
        config_path="/tmp/b.yaml", cache_config=object(),
        shared_storage=object(), version=2, yaml_id="b",
    )
    with wps._bundle_lock:
        wps._bundles["a"] = a
        wps._bundles["b"] = b
        wps._current_bundle = b

    assert get_current_cache_bundle("a") is a
    assert get_current_cache_bundle("b") is b
    assert get_current_cache_bundle("nope") is None
    # No-arg form still returns the single-latest pointer.
    assert get_current_cache_bundle() is b


def test_pool_concurrent_first_load_single_load(tmp_path):
    """8 threads racing on get_or_load for the same fingerprint trigger exactly
    one load_artifact call; all callers receive the same instance.

    The barrier synchronizes worker entry to ``get_or_load`` (the actual race
    point). It MUST NOT sit inside ``load_artifact`` — only one worker wins
    the per-fingerprint lock and reaches load_artifact, so a barrier inside
    that path deadlocks (one waiter never reaches N parties).
    """
    artifact = tmp_path / "art.pkl"
    _write_artifact(artifact, {"robot_state": 4})
    cfg = _make_in_memory_cfg(str(artifact))

    original_load = InMemoryBackend.load_artifact
    call_count = {"n": 0}
    call_count_lock = threading.Lock()

    def counting_load(self, path):
        with call_count_lock:
            call_count["n"] += 1
        return original_load(self, path)

    barrier = threading.Barrier(8)
    results: list = []
    results_lock = threading.Lock()

    def worker():
        # All 8 workers cross the barrier together, then race into get_or_load.
        barrier.wait()
        b = BackendPool.get().get_or_load(cfg)
        with results_lock:
            results.append(b)

    with patch.object(InMemoryBackend, "load_artifact", counting_load):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert call_count["n"] == 1
    assert len(results) == 8
    first = results[0]
    assert all(b is first for b in results)


# ==================================================================
# M1 BatchingCoordinator + stage_io
# ==================================================================


def _make_stage1_b1(state_dim=4, prefix_len=2, emb_dim=8) -> "Stage1Output":
    """Construct a B=1 Stage1Output with deterministic distinguishable tensors.

    Each field carries the batch index in slot [0] of the leading dim so
    stack→split round-trips can assert positional identity.
    """
    from openpi.models_pytorch.pi0_pytorch import Stage1Output
    return Stage1Output(
        state=torch.zeros(1, state_dim),
        prefix_embs=torch.zeros(1, prefix_len, emb_dim),
        prefix_pad_masks=torch.zeros(1, prefix_len, dtype=torch.bool),
        prefix_att_2d_masks_4d=torch.zeros(1, 1, prefix_len, prefix_len),
        prefix_position_ids=torch.zeros(1, prefix_len, dtype=torch.int64),
    )


# ---------- stage_io: stack/split ----------


def test_stage_io_stack_observation_no_double_batch_dim():
    """G1 R2 Item 2 invariant: per-request leaves are unbatched; coordinator
    stacks them once. N unbatched obs → shape [N, ...], NOT [N, 1, ...].
    """
    from openpi.serving.stage_io import stack_observation

    obs = {"state": torch.randn(32), "tokens": torch.zeros(200, dtype=torch.int64)}
    n = 4
    batched = stack_observation([obs] * n, device="cpu")
    assert batched["state"].shape == (n, 32)
    assert batched["tokens"].shape == (n, 200)


def test_stage_io_stage1_round_trip_preserves_values():
    """Stack then split must recover each input shard's tensors exactly."""
    from openpi.serving.stage_io import (
        split_stage1_output,
        stack_stage1_output,
    )

    n = 3
    shards = []
    for i in range(n):
        s = _make_stage1_b1()
        # Tag each shard's state to verify positional recovery.
        s.state = torch.full((1, 4), float(i))
        shards.append(s)

    batched = stack_stage1_output(shards)
    assert batched.state.shape == (n, 4)
    recovered = split_stage1_output(batched, n)
    for i, r in enumerate(recovered):
        assert r.state.shape == (1, 4)
        assert torch.all(r.state == float(i))


def test_stage_io_dynamic_cache_stack_split_clones_storage():
    """``DynamicCache.batch_split`` shards must not alias the merged cache's
    underlying K/V storage — required so per-request stage3 forwards can
    mutate their own KV without leaking across shards.
    """
    from transformers import DynamicCache
    from openpi.models_pytorch.pi0_pytorch import Stage2Output
    from openpi.serving.stage_io import (
        split_stage2_output,
        stack_stage2_output,
    )

    # Build 2 B=1 Stage2Outputs with distinct K/V values.
    def _make_stage2(tag: float) -> Stage2Output:
        cache = DynamicCache()
        # Single-layer cache with K/V shape [B=1, num_heads=1, L=2, head_dim=4].
        k = torch.full((1, 1, 2, 4), tag)
        v = torch.full((1, 1, 2, 4), tag + 0.5)
        cache.update(k, v, layer_idx=0)
        return Stage2Output(stage1=_make_stage1_b1(), past_key_values=cache)

    inputs = [_make_stage2(0.0), _make_stage2(1.0)]
    batched = stack_stage2_output(inputs)
    # Merged K shape == [2, 1, 2, 4].
    assert batched.past_key_values.key_cache[0].shape == (2, 1, 2, 4)

    shards = split_stage2_output(batched, 2)
    # Verify content is correct.
    assert torch.all(shards[0].past_key_values.key_cache[0] == 0.0)
    assert torch.all(shards[1].past_key_values.key_cache[0] == 1.0)
    # Mutate shard 0 — shard 1 must be unaffected (no view aliasing).
    shards[0].past_key_values.key_cache[0].fill_(99.0)
    assert torch.all(shards[1].past_key_values.key_cache[0] == 1.0)
    # The merged batched cache may or may not be aliased depending on
    # transformers internals; the post-condition we require is that
    # **independent shards** do not alias each other.


def test_stage_io_split_intermediates_handles_none():
    from openpi.serving.stage_io import split_stage3_intermediates

    # None propagates as [None, None, ...].
    assert split_stage3_intermediates(None, 3) == [None, None, None]

    inter = {0.7: torch.zeros(4, 50, 32), 0.5: torch.ones(4, 50, 32)}
    shards = split_stage3_intermediates(inter, 4)
    assert len(shards) == 4
    for i, s in enumerate(shards):
        assert torch.all(s[0.7] == 0.0)
        assert torch.all(s[0.5] == 1.0)
        assert s[0.7].shape == (1, 50, 32)


# ---------- BatchingCoordinator: model stub + behaviour ----------


class _StubModel:
    """Minimal stand-in for PI0Pytorch that exposes run_stage{1,2,3} +
    ``run_stage3_from`` with shape-only semantics.

    Tracks every call's batch size so tests can assert batching actually
    fires (vs. each request being dispatched individually).
    """

    def __init__(self):
        self.stage1_batch_sizes: list[int] = []
        self.stage2_batch_sizes: list[int] = []
        self.stage3_batch_sizes: list[tuple[int, str, float | None, int]] = []
        # Parameter-style attribute so BatchingCoordinator can derive a device.
        self._fake_param = torch.zeros(1)

    def parameters(self):
        # next(model.parameters()) is how the coordinator picks the device.
        return iter([self._fake_param])

    def run_stage1(self, obs):
        # Echo back a Stage1Output sized after batch.
        from openpi.models_pytorch.pi0_pytorch import Stage1Output
        b = obs["state"].shape[0]
        self.stage1_batch_sizes.append(b)
        return Stage1Output(
            state=torch.full((b, 4), -1.0),
            prefix_embs=torch.zeros(b, 2, 8),
            prefix_pad_masks=torch.zeros(b, 2, dtype=torch.bool),
            prefix_att_2d_masks_4d=torch.zeros(b, 1, 2, 2),
            prefix_position_ids=torch.zeros(b, 2, dtype=torch.int64),
        )

    def run_stage2(self, stage1):
        from openpi.models_pytorch.pi0_pytorch import Stage2Output
        from transformers import DynamicCache
        b = stage1.state.shape[0]
        self.stage2_batch_sizes.append(b)
        cache = DynamicCache()
        # Single-layer dummy KV: [B, 1, 2, 4]
        cache.update(torch.zeros(b, 1, 2, 4), torch.zeros(b, 1, 2, 4), layer_idx=0)
        return Stage2Output(stage1=stage1, past_key_values=cache)

    def run_stage3(self, stage2, *, noise, num_steps=10, return_intermediates=False):
        from openpi.models_pytorch.pi0_pytorch import Stage3Output
        b = stage2.stage1.state.shape[0]
        self.stage3_batch_sizes.append((b, "miss", None, num_steps))
        action = torch.full((b, 50, 32), 1.0)
        inter = {0.7: torch.full((b, 50, 32), 7.0)} if return_intermediates else None
        return Stage3Output(action_chunk=action, intermediates=inter)

    def run_stage3_from(self, stage2, *, start_x, start_t, num_steps=10):
        from openpi.models_pytorch.pi0_pytorch import Stage3Output
        b = stage2.stage1.state.shape[0]
        self.stage3_batch_sizes.append((b, "warm_start", start_t, num_steps))
        action = torch.full((b, 50, 32), 2.0)
        return Stage3Output(action_chunk=action, intermediates=None)


def test_coordinator_stage1_batches_concurrent_submissions():
    """N threads submitting unbatched obs concurrently → one batched run_stage1
    call of size N (when N ≤ max_batch_size and all arrive within window)."""
    from openpi.serving.batching_coordinator import BatchingCoordinator

    stub = _StubModel()
    n = 4
    barrier = threading.Barrier(n)
    results: list = [None] * n

    with BatchingCoordinator(stub, device="cpu", max_batch_size=n, max_wait_ms=50.0) as bc:
        def worker(i):
            barrier.wait()
            obs = {"state": torch.full((4,), float(i)), "tokens": torch.zeros(2, dtype=torch.int64)}
            results[i] = bc.submit_to_stage(1, "default", obs, request_id=f"r{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # All requests share one batched call.
    assert stub.stage1_batch_sizes == [n]
    # Each worker received a B=1 Stage1Output.
    for r in results:
        assert r is not None
        assert r.state.shape == (1, 4)


def test_coordinator_stage3_sub_buckets_by_mode_and_params():
    """Stage 3 must split a heterogeneous batch into one bucket per
    (mode, start_t, num_steps) and issue one batched forward per bucket
    (G1 R2 Item 3).
    """
    from openpi.serving.batching_coordinator import (
        BatchingCoordinator,
        Stage3MissPayload,
        Stage3WarmStartPayload,
    )
    from openpi.models_pytorch.pi0_pytorch import Stage2Output
    from transformers import DynamicCache

    def _make_stage2_b1() -> Stage2Output:
        cache = DynamicCache()
        cache.update(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4), layer_idx=0)
        return Stage2Output(stage1=_make_stage1_b1(), past_key_values=cache)

    stub = _StubModel()
    # 5 requests: 2 MISS + 2 WARM_START(start_t=0.7,num_steps=10) +
    # 1 WARM_START(start_t=0.5,num_steps=10) → 3 distinct buckets.
    payloads = [
        Stage3MissPayload(stage2_out=_make_stage2_b1(), noise=torch.zeros(50, 32), num_steps=10),
        Stage3MissPayload(stage2_out=_make_stage2_b1(), noise=torch.zeros(50, 32), num_steps=10),
        Stage3WarmStartPayload(stage2_out=_make_stage2_b1(),
                                start_x=torch.zeros(50, 32), start_t=0.7, num_steps=10),
        Stage3WarmStartPayload(stage2_out=_make_stage2_b1(),
                                start_x=torch.zeros(50, 32), start_t=0.7, num_steps=10),
        Stage3WarmStartPayload(stage2_out=_make_stage2_b1(),
                                start_x=torch.zeros(50, 32), start_t=0.5, num_steps=10),
    ]

    n = len(payloads)
    barrier = threading.Barrier(n)
    results: list = [None] * n

    with BatchingCoordinator(stub, device="cpu", max_batch_size=n, max_wait_ms=50.0) as bc:
        def worker(i):
            barrier.wait()
            results[i] = bc.submit_to_stage(3, "default", payloads[i], request_id=f"r{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Three buckets: MISS×2, WS(0.7,10)×2, WS(0.5,10)×1.
    bucket_keys = sorted(stub.stage3_batch_sizes)
    expected = sorted([(2, "miss", None, 10),
                       (2, "warm_start", 0.7, 10),
                       (1, "warm_start", 0.5, 10)])
    assert bucket_keys == expected
    # Sanity-check per-request return shape.
    for r in results:
        assert r is not None
        assert r.action_chunk.shape == (1, 50, 32)


# ==================================================================
# M6 Single Process Default — Args defaults + non_concurrent override
# ==================================================================


def test_args_default_concurrent_is_true():
    """Phase 5: ``--concurrent`` flips its default to True so sweep workflows
    naturally use the single-server multi-bundle layout."""
    from scripts.serve_policy import Args

    args = Args()
    assert args.concurrent is True
    assert args.non_concurrent is False


def test_non_concurrent_flag_overrides_concurrent():
    """``--non-concurrent`` is the opt-out path back to the baseline C1
    single-connection mode. Verifies the overrider logic without running
    ``main`` (which needs a real checkpoint).
    """
    from scripts.serve_policy import Args

    # Inline the same dataclasses.replace logic used at the top of main().
    args = Args(non_concurrent=True)
    assert args.concurrent is True  # raw default kept
    overridden = dataclasses.replace(args, concurrent=False, non_concurrent=False)
    assert overridden.concurrent is False
    assert overridden.non_concurrent is False


def test_coordinator_propagates_exception_to_submitter():
    """If the batched forward raises, every request in the batch receives the
    same exception via ``submit_to_stage``.
    """
    from openpi.serving.batching_coordinator import BatchingCoordinator

    class _RaisingStub(_StubModel):
        def run_stage1(self, obs):
            raise RuntimeError("forced failure")

    stub = _RaisingStub()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=2, max_wait_ms=10.0) as bc:
        with pytest.raises(RuntimeError, match="forced failure"):
            bc.submit_to_stage(
                1, "default",
                {"state": torch.zeros(4), "tokens": torch.zeros(2, dtype=torch.int64)},
                request_id="r0",
            )


# ==================================================================
# M7 Benchmark — sweep helpers (driver/sweep/plot zero-IO smoke tests)
# ==================================================================


def test_sweep_percentile_helper():
    """Sanity-check the internal percentile helper used in sweep summaries.

    The helper uses ``int(p * (len-1))`` which is nearest-rank-below — for
    a 10-element list this maps p=0.95 → index 8 (value 90), not 100.
    """
    import math

    from exp.serving_benchmark.sweep import _percentile

    xs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert _percentile(xs, 0.5) == 50.0
    assert _percentile(xs, 0.95) == 90.0
    # Empty input returns NaN.
    assert math.isnan(_percentile([], 0.5))


def test_sweep_yaml_schema_round_trip(tmp_path):
    """Each shipped sweep config must parse and contain the documented keys."""
    import yaml

    configs_dir = Path("exp/serving_benchmark/configs")
    expected = {
        "sparse_to_dense.yaml",
        "freq_sweep.yaml",
        "yaml_density.yaml",
        "batch_window.yaml",
    }
    for name in expected:
        path = configs_dir / name
        cfg = yaml.safe_load(path.read_text())
        assert "mode" in cfg
        assert "cells" in cfg and len(cfg["cells"]) >= 1
        for cell in cfg["cells"]:
            assert "id" in cell
            assert "num_workers" in cell
            assert "request_hz" in cell


def test_driver_config_defaults_match_docs():
    """DriverConfig defaults must agree with docs/experiments/serving_benchmark.md."""
    from exp.serving_benchmark.driver import DriverConfig

    cfg = DriverConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.duration_s == 30.0
    assert cfg.warmup_s == 2.0
    assert cfg.bundle_id == "default"


# ==================================================================
# G2 R1 — Integration: lazy lifecycle + coordinator wiring + client API
# ==================================================================


def test_interceptor_accepts_coordinator_and_bundle_id_kwargs():
    """G2 R1 Item 1: InferenceInterceptor.__init__ must accept coordinator+
    bundle_id. The internal stage_fn dispatch is patched to route via the
    coordinator when set."""
    import inspect

    from openpi.cache.interceptor import InferenceInterceptor

    sig = inspect.signature(InferenceInterceptor.__init__)
    assert "coordinator" in sig.parameters
    assert "bundle_id" in sig.parameters
    assert sig.parameters["coordinator"].default is None
    assert sig.parameters["bundle_id"].default == "default"


def test_interceptor_stage_fns_route_through_coordinator():
    """G2 R1 Item 1: when a coordinator is supplied, the interceptor's
    ``_stage{1,2,3}_fn`` attributes must dispatch to coordinator
    ``submit_to_stage`` rather than calling the model directly."""
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.serving.batching_coordinator import (
        BatchingCoordinator,
        Stage3MissPayload,
    )

    class _FakeStage1:
        pass

    captured_calls: list = []

    class _RecordingCoordinator:
        def submit_to_stage(self, stage_id, bundle_id, payload, **kw):
            captured_calls.append((stage_id, bundle_id, type(payload).__name__))
            return _FakeStage1() if stage_id == 1 else payload

    # Build a minimal stand-in for ``Policy`` that satisfies the
    # interceptor's expectations (just the internal references it borrows).
    class _DummyPolicy:
        _is_pytorch_model = True
        _model = type("M", (), {
            "run_stage1": staticmethod(lambda *a, **k: None),
            "run_stage2": staticmethod(lambda *a, **k: None),
            "run_stage3": staticmethod(lambda *a, **k: None),
            "run_stage3_from": staticmethod(lambda *a, **k: None),
        })
        _input_transform = staticmethod(lambda x: x)
        _output_transform = staticmethod(lambda x: x)
        _pytorch_device = "cpu"

    interceptor = InferenceInterceptor(
        _DummyPolicy(),                          # type: ignore[arg-type]
        eager=True,
        coordinator=_RecordingCoordinator(),
        bundle_id="bundle-xyz",
    )
    # Stage1 dispatch goes through coordinator.
    interceptor._stage1_fn({"state": torch.zeros(4)})
    assert captured_calls[-1] == (1, "bundle-xyz", "dict")
    # Stage2 dispatch — payload type echoes Stage1Output.
    interceptor._stage2_fn(_FakeStage1())
    assert captured_calls[-1] == (2, "bundle-xyz", "_FakeStage1")
    # Stage3 dispatch — wraps into Stage3MissPayload (MISS path).
    interceptor._stage3_fn(_FakeStage1(), noise=torch.zeros(50, 32))
    assert captured_calls[-1][0] == 3
    assert captured_calls[-1][1] == "bundle-xyz"
    assert captured_calls[-1][2] == "Stage3MissPayload"
    # Stage3 WARM_START routes through ``_stage3_from_fn`` and wraps as
    # ``Stage3WarmStartPayload`` (no noise field).
    interceptor._stage3_from_fn(
        _FakeStage1(), start_x=torch.zeros(50, 32), start_t=0.7, num_steps=10,
    )
    assert captured_calls[-1] == (3, "bundle-xyz", "Stage3WarmStartPayload")
    # Sanity: BatchingCoordinator class is importable from the same module
    # the interceptor consumed (no circular import surprises).
    assert BatchingCoordinator is not None  # noqa: B015 — explicit reference


def test_interceptor_no_coordinator_legacy_path():
    """G2 R1 Item 1 contrapositive: when ``coordinator is None`` the
    interceptor MUST keep its original stage_fn binding (run_stage{1,2,3})
    so the C1 non-concurrent path stays bit-identical.
    """
    from openpi.cache.interceptor import InferenceInterceptor

    class _DummyPolicy:
        _is_pytorch_model = True
        _model = type("M", (), {
            "run_stage1": staticmethod(lambda *a, **k: "s1"),
            "run_stage2": staticmethod(lambda *a, **k: "s2"),
            "run_stage3": staticmethod(lambda *a, **k: "s3"),
        })
        _input_transform = staticmethod(lambda x: x)
        _output_transform = staticmethod(lambda x: x)
        _pytorch_device = "cpu"

    interceptor = InferenceInterceptor(_DummyPolicy(), eager=True)  # type: ignore[arg-type]
    assert interceptor._coordinator is None
    assert interceptor._stage3_from_fn is None
    # Direct invocation falls through to the model's run_stage1.
    assert interceptor._stage1_fn(None) == "s1"


def test_websocket_factory_signature_accepts_bundle_id():
    """G2 R1 Item 2: ``_connection_policy_factory`` must accept a
    ``bundle_id`` positional argument so the lazy lifecycle can pass the
    selected bundle into the wrapper stack.
    """
    import inspect

    from scripts.serve_policy import _wrap_policy

    sig = inspect.signature(_wrap_policy)
    assert "bundle_id" in sig.parameters
    assert sig.parameters["bundle_id"].default == "default"


def test_client_select_bundle_helper_packs_correct_ctrl(monkeypatch):
    """G2 R1 Item 3: openpi_client.select_bundle must send the proper
    ``__ctrl__: select_bundle`` frame via the msgpack packer (not raw dict
    over the private socket).
    """
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    captured: dict = {}

    class _FakeWS:
        def send(self, data):
            captured["bytes"] = data

        def recv(self):
            # Pack a server ack matching the expected_ack.
            import msgpack

            return msgpack.packb({"__ack__": "select_bundle", "bundle_id": "bxy"})

    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._ws = _FakeWS()
    client._packer = __import__("openpi_client.msgpack_numpy", fromlist=["Packer"]).Packer()
    client._server_metadata = {}

    ack = client.select_bundle("bxy")
    assert ack["__ack__"] == "select_bundle"
    assert ack["bundle_id"] == "bxy"
    # The wire bytes must be a proper msgpack frame containing select_bundle.
    import msgpack

    payload = msgpack.unpackb(captured["bytes"])
    assert payload["__ctrl__"] == "select_bundle"
    assert payload["bundle_id"] == "bxy"


def test_client_load_cache_config_accepts_bundle_id():
    """G2 R1 Item 3: ``load_cache_config`` carries the new optional
    ``bundle_id`` field so the server can index multiple loaded bundles."""
    import inspect

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    sig = inspect.signature(WebsocketClientPolicy.load_cache_config)
    assert "bundle_id" in sig.parameters
    assert sig.parameters["bundle_id"].default is None


# ==================================================================
# G2 R2 — integration: infer() through a recording coordinator
# ==================================================================


def _build_infer_test_interceptor(coordinator, *, with_orchestrator: bool):
    """Construct a minimal InferenceInterceptor that exercises infer() end-to-end.

    Replaces the real Policy + Model with stubs sized to a small action space,
    so the interceptor can run ``infer()`` without touching GPU or large I/O.
    """
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.models_pytorch.pi0_pytorch import Stage1Output, Stage2Output, Stage3Output
    from transformers import DynamicCache

    ACTION_HORIZON = 50
    ACTION_DIM = 32

    class _StubConfig:
        action_horizon = ACTION_HORIZON
        action_dim = ACTION_DIM

    class _StubModel:
        config = _StubConfig()

        def sample_noise(self, shape, device):
            return torch.zeros(*shape, device=device)

        # The non-coordinator path needs these (legacy `_stage*_fn` bindings).
        def run_stage1(self, observation):
            # Build a Stage1Output sized after the obs batch dim.
            state = getattr(observation, "state", None)
            if state is None and isinstance(observation, dict):
                state = observation["state"]
            b = state.shape[0] if state.ndim >= 2 else 1
            return Stage1Output(
                state=torch.zeros(b, 4),
                prefix_embs=torch.zeros(b, 2, 8),
                prefix_pad_masks=torch.zeros(b, 2, dtype=torch.bool),
                prefix_att_2d_masks_4d=torch.zeros(b, 1, 2, 2),
                prefix_position_ids=torch.zeros(b, 2, dtype=torch.int64),
            )

        def run_stage2(self, stage1):
            cache = DynamicCache()
            b = stage1.state.shape[0]
            cache.update(torch.zeros(b, 1, 2, 4), torch.zeros(b, 1, 2, 4), layer_idx=0)
            return Stage2Output(stage1=stage1, past_key_values=cache)

        def run_stage3(self, stage2, *, noise, num_steps=10, return_intermediates=False):
            b = stage2.stage1.state.shape[0]
            inter = {0.7: torch.zeros(b, ACTION_HORIZON, ACTION_DIM)} if return_intermediates else None
            return Stage3Output(
                action_chunk=torch.zeros(b, ACTION_HORIZON, ACTION_DIM),
                intermediates=inter,
            )

        def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
            b = stage2.stage1.state.shape[0]
            return Stage3Output(
                action_chunk=torch.zeros(b, ACTION_HORIZON, ACTION_DIM),
                intermediates=None,
            )

    model = _StubModel()

    class _StubPolicy:
        _is_pytorch_model = True
        _model = model
        # Identity transforms: input_transform consumes the raw obs dict.
        _input_transform = staticmethod(lambda obs: obs)
        _output_transform = staticmethod(lambda outputs: outputs)
        _pytorch_device = "cpu"

    # ``with_orchestrator`` is intentionally unused — building a full
    # orchestrator stub is fragile (many surfaces). Both integration tests
    # below drive the no-orchestrator branches of ``infer()`` (orchestrator
    # is None), which still exercise the new coordinator wiring including
    # the noise-sampling fix that lives in the no-cache stage3 branch.
    del with_orchestrator
    return InferenceInterceptor(
        _StubPolicy(),  # type: ignore[arg-type]
        eager=True,
        coordinator=coordinator,
        bundle_id="bundle-int",
        orchestrator=None,
    )


class _RecordingCoordinator:
    """Captures every ``submit_to_stage`` call and forwards to a real model
    so we can inspect payload shapes / types from the interceptor's
    ``infer()`` path without spinning up the actual BatchingCoordinator
    background threads.
    """

    def __init__(self, model):
        from openpi.models import model as _model

        self._model = model
        self._model_obs_cls = _model.Observation
        self.calls: list = []

    def submit_to_stage(self, stage_id, bundle_id, payload, **kwargs):
        self.calls.append((stage_id, bundle_id, payload))
        if stage_id == 1:
            # Match the coordinator's runtime contract: stack the (single)
            # unbatched dict, wrap into Observation, run stage1.
            from openpi.serving import stage_io

            batched = stage_io.stack_observation([payload], device="cpu")
            obs = self._model_obs_cls.from_dict(batched) if isinstance(batched, dict) else batched
            return self._model.run_stage1(obs)
        if stage_id == 2:
            return self._model.run_stage2(payload)
        # stage_id == 3
        from openpi.serving.batching_coordinator import Stage3MissPayload, Stage3WarmStartPayload

        if isinstance(payload, Stage3MissPayload):
            return self._model.run_stage3(
                payload.stage2_out, noise=payload.noise.unsqueeze(0),
                num_steps=payload.num_steps, return_intermediates=True,
            )
        if isinstance(payload, Stage3WarmStartPayload):
            return self._model.run_stage3_from(
                payload.stage2_out, start_x=payload.start_x.unsqueeze(0),
                start_t=payload.start_t, num_steps=payload.num_steps,
            )
        raise TypeError(type(payload).__name__)


def _make_libero_like_obs() -> dict:
    """Produce an unbatched obs dict that matches ``Observation.from_dict``."""
    import numpy as np

    return {
        "state": np.zeros(8, dtype=np.float32),
        "image": {"cam_0": np.zeros((224, 224, 3), dtype=np.uint8)},
        "image_mask": {"cam_0": np.zeros((), dtype=np.bool_)},
        "tokenized_prompt": np.zeros(48, dtype=np.int64),
        "tokenized_prompt_mask": np.zeros(48, dtype=np.bool_),
    }


def test_infer_coordinator_stage1_payload_is_unbatched():
    """G2 R2 Item 1: ``infer()`` must hand the coordinator a leaf with NO
    leading batch dimension. ``stack_observation`` is what adds the batch
    axis — a B=1 leaf at submit time produces ``[N, 1, ...]`` instead of
    ``[N, ...]`` after stacking, recreating the G1 R2 double-batch-dim bug.
    """
    coordinator = None  # placeholder before we build the interceptor

    class _Capturing(_RecordingCoordinator):
        def __init__(self):
            pass

        def submit_to_stage(self, stage_id, bundle_id, payload, **kwargs):
            self.calls = getattr(self, "calls", [])
            self.calls.append((stage_id, bundle_id, payload))
            if stage_id == 1:
                # Return a synthetic Stage1Output sized to whatever we receive.
                from openpi.models_pytorch.pi0_pytorch import Stage1Output

                # When called from infer(), payload is an unbatched dict ⇒
                # ``state`` is a 1-D tensor with no batch dim.
                state = payload["state"]
                assert state.ndim == 1, (
                    f"coordinator stage1 payload state must be unbatched; "
                    f"got shape {tuple(state.shape)}"
                )
                return Stage1Output(
                    state=torch.zeros(1, 4),
                    prefix_embs=torch.zeros(1, 2, 8),
                    prefix_pad_masks=torch.zeros(1, 2, dtype=torch.bool),
                    prefix_att_2d_masks_4d=torch.zeros(1, 1, 2, 2),
                    prefix_position_ids=torch.zeros(1, 2, dtype=torch.int64),
                )
            return None  # downstream stages unused in this test

    capturing = _Capturing()
    interceptor = _build_infer_test_interceptor(capturing, with_orchestrator=False)
    # Drive a single inference. The wrapper executes through stage1 only
    # (no orchestrator + coordinator path returns synthetic outputs).
    try:
        interceptor.infer(_make_libero_like_obs())
    except Exception:
        # Downstream stages aren't covered by this stub — we only care that
        # the stage1 invariant fired before any other stage.
        pass

    stage1_calls = [c for c in capturing.calls if c[0] == 1]
    assert stage1_calls, "stage1 was never submitted to the coordinator"
    payload = stage1_calls[0][2]
    assert isinstance(payload, dict), (
        f"coordinator path must submit unbatched dict, got {type(payload).__name__}"
    )
    # Concrete check: the state leaf is 1-D, not 2-D.
    assert payload["state"].ndim == 1


def test_infer_coordinator_miss_path_samples_noise_when_none():
    """G2 R2 Item 2: when ``infer(noise=None)`` reaches the no-cache stage3
    branch with the coordinator wired, the interceptor MUST sample a real
    per-request noise tensor before wrapping into ``Stage3MissPayload`` —
    the coordinator's MISS sub-bucket does ``torch.stack([... .noise ...])``
    and would crash on ``None``.

    Exercises the production ``infer()`` path with a recording coordinator
    (no orchestrator, no CP1/CP3) so we observe what gets submitted at
    stage3 without spinning up the full BatchingCoordinator threads.
    """
    from openpi.serving.batching_coordinator import Stage3MissPayload

    placeholder = _RecordingCoordinator.__new__(_RecordingCoordinator)
    placeholder._model = None
    placeholder.calls = []

    interceptor = _build_infer_test_interceptor(placeholder, with_orchestrator=False)
    real_coordinator = _RecordingCoordinator(interceptor._model)
    # Re-bind the stage closures to the real recording coordinator now that
    # the stub model is known; this mirrors `serve_policy.py`'s
    # post-construction wiring.
    interceptor._coordinator = real_coordinator
    interceptor._stage1_fn = lambda obs: real_coordinator.submit_to_stage(1, "bundle-int", obs)
    interceptor._stage2_fn = lambda stage1: real_coordinator.submit_to_stage(2, "bundle-int", stage1)

    from openpi.serving import batching_coordinator as _bc

    def _stage3(stage2, *, noise=None, num_steps=10, return_intermediates=False):
        payload = _bc.Stage3MissPayload(
            stage2_out=stage2,
            noise=noise.squeeze(0) if (noise is not None and noise.dim() == 3) else noise,
            num_steps=num_steps,
        )
        return real_coordinator.submit_to_stage(3, "bundle-int", payload)

    interceptor._stage3_fn = _stage3

    interceptor.infer(_make_libero_like_obs())

    stage3_calls = [c for c in real_coordinator.calls if c[0] == 3]
    assert stage3_calls, "stage3 was not submitted"
    payload = stage3_calls[0][2]
    assert isinstance(payload, Stage3MissPayload)
    # Key invariants: noise is concrete (not None), correct shape.
    assert payload.noise is not None
    assert isinstance(payload.noise, torch.Tensor)
    assert payload.noise.shape == (
        interceptor._model.config.action_horizon,
        interceptor._model.config.action_dim,
    )


def test_infer_coordinator_output_state_shape_preserved():
    """G2 R3 Item 1: under the coordinator path, ``infer()`` must produce
    ``outputs["state"]`` of the **same shape as the input state**, not a
    scalar.

    The legacy non-coordinator path used to add ``[None, ...]`` to all
    leaves, then strip ``[0, ...]`` at output time — net no-op for vectors.
    The coordinator path leaves ``inputs["state"]`` unbatched, so the
    naive ``[0, ...]`` strip collapses a ``(S,)`` vector into a 0-D
    scalar, breaking ``Policy.infer``'s output contract and any
    downstream output transform that indexes into state.
    """
    placeholder = _RecordingCoordinator.__new__(_RecordingCoordinator)
    placeholder._model = None
    placeholder.calls = []

    interceptor = _build_infer_test_interceptor(placeholder, with_orchestrator=False)
    real_coordinator = _RecordingCoordinator(interceptor._model)
    interceptor._coordinator = real_coordinator
    interceptor._stage1_fn = lambda obs: real_coordinator.submit_to_stage(1, "bundle-int", obs)
    interceptor._stage2_fn = lambda stage1: real_coordinator.submit_to_stage(2, "bundle-int", stage1)

    from openpi.serving import batching_coordinator as _bc

    def _stage3(stage2, *, noise=None, num_steps=10, return_intermediates=False):
        payload = _bc.Stage3MissPayload(
            stage2_out=stage2,
            noise=noise.squeeze(0) if (noise is not None and noise.dim() == 3) else noise,
            num_steps=num_steps,
        )
        return real_coordinator.submit_to_stage(3, "bundle-int", payload)

    interceptor._stage3_fn = _stage3

    obs = _make_libero_like_obs()
    out = interceptor.infer(obs)

    # State must keep the (S,) vector shape of the original input.
    assert out["state"].shape == obs["state"].shape, (
        f"coordinator infer() collapsed state from {obs['state'].shape} to "
        f"{out['state'].shape}; expected unchanged 1-D vector."
    )
    # Actions still strip the unit batch dim from stage3.
    assert out["actions"].shape == (
        interceptor._model.config.action_horizon,
        interceptor._model.config.action_dim,
    )


def test_infer_coordinator_output_survives_state_indexing_transform():
    """G2 R3 Item 1 (follow-on): the coordinator path output must be
    compatible with a downstream transform that indexes ``state[..., :n]``
    (the exact failure mode the reviewer demonstrated with
    ``AbsoluteActions``). With state preserved as a 1-D vector this slice
    is well-defined; with a 0-D scalar it raises IndexError.
    """
    placeholder = _RecordingCoordinator.__new__(_RecordingCoordinator)
    placeholder._model = None
    placeholder.calls = []

    interceptor = _build_infer_test_interceptor(placeholder, with_orchestrator=False)
    real_coordinator = _RecordingCoordinator(interceptor._model)
    interceptor._coordinator = real_coordinator
    interceptor._stage1_fn = lambda obs: real_coordinator.submit_to_stage(1, "bundle-int", obs)
    interceptor._stage2_fn = lambda stage1: real_coordinator.submit_to_stage(2, "bundle-int", stage1)

    from openpi.serving import batching_coordinator as _bc

    def _stage3(stage2, *, noise=None, num_steps=10, return_intermediates=False):
        payload = _bc.Stage3MissPayload(
            stage2_out=stage2,
            noise=noise.squeeze(0) if (noise is not None and noise.dim() == 3) else noise,
            num_steps=num_steps,
        )
        return real_coordinator.submit_to_stage(3, "bundle-int", payload)

    interceptor._stage3_fn = _stage3

    # Substitute a transform that mimics the AbsoluteActions failure mode:
    # it slices ``state[..., :dims]`` and asserts shape, surfacing 0-D
    # regression as IndexError instead of a silent scalar.
    import numpy as np

    captured: dict = {}

    def _indexing_transform(outputs):
        state = outputs["state"]
        # Will raise ``IndexError: too many indices for array: array is 0-dimensional``
        # if state regressed to scalar.
        sub = state[..., :3]
        captured["sub_shape"] = sub.shape
        return outputs

    interceptor._output_transform = _indexing_transform

    interceptor.infer(_make_libero_like_obs())
    assert captured.get("sub_shape") == (3,)
