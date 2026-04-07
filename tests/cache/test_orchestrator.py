"""Tests for CacheOrchestrator."""

import math

import torch
import torch.nn.functional as F

from openpi.cache.components.judge import HitType, ThresholdJudge
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import ROBOT_STATE, CheckpointID

from openpi.cache.backends.in_memory_backend import InMemoryBackend

from tests.cache.conftest import (
    TestStorageSearchStrategy,
    _wrap_per_checkpoint,
    insert_entry,
    make_counting_orchestrator,
    make_orchestrator,
    make_stage1,
    stable_hash,
)


# ---------------------------------------------------------------------------
# Deterministic vector helpers
# ---------------------------------------------------------------------------


def _unit_vector(dim: int = 32, index: int = 0) -> torch.Tensor:
    """Standard basis vector e_i as [1, dim] (batched)."""
    v = torch.zeros(1, dim)
    v[0, index] = 1.0
    return v


def _vector_with_known_cosine(base: torch.Tensor, target_cos: float) -> torch.Tensor:
    """Construct a unit vector with cosine similarity = target_cos to base."""
    dim = base.shape[1]
    ortho = torch.zeros_like(base)
    for i in range(dim):
        if abs(base[0, i].item()) < 1e-6:
            ortho[0, i] = 1.0
            break
    else:
        ortho[0, 0] = -base[0, 1]
        ortho[0, 1] = base[0, 0]
        ortho = F.normalize(ortho, dim=1)

    sin_val = math.sqrt(1.0 - target_cos ** 2)
    v = target_cos * base + sin_val * ortho
    v = F.normalize(v, dim=1)
    return v


# ---------------------------------------------------------------------------
# T4.1: check miss on empty store
# ---------------------------------------------------------------------------


def test_check_miss_on_empty_store():
    orch, _, _ = make_orchestrator()
    state = torch.randn(1, 32)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert result.payload is None
    orch.clear()


# ---------------------------------------------------------------------------
# T4.2: insert then check exact match -> FULL_HIT
# ---------------------------------------------------------------------------


def test_insert_then_check_exact_match_hits():
    orch, _, storage = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.FULL_HIT
    assert result.payload is not None
    assert torch.allclose(result.payload.action_chunk, payload.action_chunk)
    assert result.score is not None
    assert abs(result.score - 1.0) < 1e-5
    orch.clear()


# ---------------------------------------------------------------------------
# T4.3: different state -> MISS
# ---------------------------------------------------------------------------


def test_insert_then_check_different_state_misses():
    orch, _, storage = make_orchestrator()
    state_a = _unit_vector(32, 0)
    state_b = -_unit_vector(32, 0)  # cosine = -1

    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state_a, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state_b))
    assert result.hit_type == HitType.MISS
    orch.clear()


# ---------------------------------------------------------------------------
# T4.4: similar state near threshold -> FULL_HIT
# ---------------------------------------------------------------------------


def test_insert_then_check_similar_state_near_threshold():
    orch, _, storage = make_orchestrator()
    base = _unit_vector(32, 0)
    similar = _vector_with_known_cosine(base, target_cos=0.99)

    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, base, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(similar))
    assert result.hit_type == HitType.FULL_HIT
    orch.clear()


# ---------------------------------------------------------------------------
# T4.5: judge MISS does not call fetch_payload
# ---------------------------------------------------------------------------


def test_judge_miss_does_not_fetch_payload():
    judge = ThresholdJudge(cp1_threshold=1.01)  # impossible to reach
    orch, backend, storage = make_counting_orchestrator(judge=judge)

    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    storage.fetch_payload_call_count = 0

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert result.payload is None
    assert storage.fetch_payload_call_count == 0
    orch.clear()


# ---------------------------------------------------------------------------
# T4.6: gate False skips search
# ---------------------------------------------------------------------------


class NeverSearchGate:
    def __call__(self, checkpoint_id, cached_data):
        return False


def test_gate_false_skips_search():
    dims = {"robot_state": 32}
    backend = InMemoryBackend(dims)
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.key_builder import PlaceholderKeyBuilder
    from openpi.cache.timing import SystemTimer

    storage = CacheStorage(backend)
    strategy = TestStorageSearchStrategy(storage, top_k=1)
    orch = CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates=_wrap_per_checkpoint(NeverSearchGate()),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )

    state = torch.randn(1, 32)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert backend.search_call_count == 0
    orch.clear()


# ---------------------------------------------------------------------------
# T4.7: idempotent upsert (same id -> count=1)
# ---------------------------------------------------------------------------


def test_insert_idempotent_upsert():
    orch, backend, storage = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    insert_entry(storage, CheckpointID.CP1, state, payload)
    insert_entry(storage, CheckpointID.CP1, state, payload)

    assert backend.count() == 1


# ---------------------------------------------------------------------------
# T4.8: different states create separate entries
# ---------------------------------------------------------------------------


def test_insert_different_states_creates_separate_entries():
    orch, backend, storage = make_orchestrator()
    state_a = _unit_vector(32, 0)
    state_b = _unit_vector(32, 1)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    insert_entry(storage, CheckpointID.CP1, state_a, payload)
    insert_entry(storage, CheckpointID.CP1, state_b, payload)

    assert backend.count() == 2


# ---------------------------------------------------------------------------
# T4.9: CP3 check misses when only CP1 entries exist
# ---------------------------------------------------------------------------


def test_cp3_check_misses_with_cp1_entries():
    orch, _, storage = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP3, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    orch.clear()


# ---------------------------------------------------------------------------
# T4.12: clear resets key_builder
# ---------------------------------------------------------------------------


def test_clear_resets_key_builder():
    orch, _, _ = make_orchestrator()
    state = torch.randn(1, 32)
    orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    orch.clear()
    assert orch._key_builder.cached_data == {}


# ---------------------------------------------------------------------------
# T4.13: check returns score and entry_id on hit
# ---------------------------------------------------------------------------


def test_check_returns_score_and_entry_id_on_hit():
    orch, _, storage = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    entry = insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.score is not None
    assert result.entry_id is not None
    assert result.entry_id == entry.id
    orch.clear()


# ---------------------------------------------------------------------------
# T4.14: check returns query_keys on MISS
# ---------------------------------------------------------------------------


def test_check_returns_query_keys_on_miss():
    orch, _, _ = make_orchestrator()
    state = torch.randn(1, 32)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert result.query_keys is not None
    assert "robot_state" in result.query_keys
    orch.clear()


# ---------------------------------------------------------------------------
# T4.15: check returns query_keys on gate skip
# ---------------------------------------------------------------------------


def test_check_returns_query_keys_on_gate_skip():
    dims = {"robot_state": 32}
    backend = InMemoryBackend(dims)
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.key_builder import PlaceholderKeyBuilder
    from openpi.cache.timing import SystemTimer

    storage = CacheStorage(backend)
    strategy = TestStorageSearchStrategy(storage, top_k=1)
    orch = CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates=_wrap_per_checkpoint(NeverSearchGate()),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )

    state = torch.randn(1, 32)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert result.query_keys is not None
    assert "robot_state" in result.query_keys
    orch.clear()


# ---------------------------------------------------------------------------
# T4.16: check returns query_keys on HIT
# ---------------------------------------------------------------------------


def test_check_returns_query_keys_on_hit():
    orch, _, storage = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.FULL_HIT
    assert result.query_keys is not None
    assert "robot_state" in result.query_keys
    orch.clear()


# ---------------------------------------------------------------------------
# T4.17: broadcast_action propagates to components
# ---------------------------------------------------------------------------


def test_broadcast_action_propagates():
    """broadcast_action should call record_action on strategies/gates/judges."""

    class RecordingStrategy:
        def __init__(self):
            self.actions = []

        def search(self, ctx):
            return []

        def record_action(self, action):
            self.actions.append(action)

        def on_episode_start(self):
            self.actions.clear()

    strategy = RecordingStrategy()
    dims = {"robot_state": 32}
    backend = InMemoryBackend(dims)
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.key_builder import PlaceholderKeyBuilder
    from openpi.cache.timing import SystemTimer

    storage = CacheStorage(backend)
    orch = CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates=_wrap_per_checkpoint(NeverSearchGate()),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies={CheckpointID.CP1: strategy, CheckpointID.CP3: strategy},
        timer=SystemTimer(enabled=False),
    )

    action = torch.randn(50, 32)
    orch.broadcast_action(action)
    # Same instance registered for both CP1 and CP3 → called twice (once per dict value)
    assert len(strategy.actions) == 2
    assert torch.equal(strategy.actions[0], action)
    assert torch.equal(strategy.actions[1], action)


# ---------------------------------------------------------------------------
# T4.18: step counter increments on CP1 check
# ---------------------------------------------------------------------------


def test_step_counter_increments_on_cp1():
    orch, _, _ = make_orchestrator()
    assert orch._step_counter == 0

    state = torch.randn(1, 32)
    orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    orch.clear()
    assert orch._step_counter == 1

    orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    orch.clear()
    assert orch._step_counter == 2


# ---------------------------------------------------------------------------
# T4.19: on_task_begin resets step counter
# ---------------------------------------------------------------------------


def test_on_task_begin_resets_step_counter():
    orch, _, _ = make_orchestrator()
    state = torch.randn(1, 32)
    orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    orch.clear()
    assert orch._step_counter == 1

    orch.on_task_begin("new_task")
    assert orch._step_counter == 0
