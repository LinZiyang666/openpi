"""Tests for CacheOrchestrator (T4.1–T4.15)."""

import math

import torch
import torch.nn.functional as F

from openpi.cache.components.judge import HitType, ThresholdJudge
from openpi.cache.orchestrator import CacheOrchestrator, _stable_hash
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import ROBOT_STATE, CheckpointID

from openpi.cache.backends.in_memory_backend import InMemoryBackend

from tests.cache.conftest import (
    TestStorageSearchStrategy,
    _wrap_per_checkpoint,
    make_counting_orchestrator,
    make_orchestrator,
    make_stage1,
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
    """Construct a unit vector with cosine similarity = target_cos to base.

    base: [1, dim] unit vector.
    Returns: [1, dim] unit vector.

    Method: v = target_cos * base + sqrt(1 - target_cos^2) * orthogonal
    """
    dim = base.shape[1]
    # Find an orthogonal direction.
    ortho = torch.zeros_like(base)
    # Pick a dimension where base is zero (or near-zero) for orthogonality.
    # For standard basis vectors this is trivial.
    for i in range(dim):
        if abs(base[0, i].item()) < 1e-6:
            ortho[0, i] = 1.0
            break
    else:
        # General case: Gram-Schmidt.
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
# T4.2: write then check exact match -> FULL_HIT
# ---------------------------------------------------------------------------


def test_write_then_check_exact_match_hits():
    orch, backend, _ = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.FULL_HIT
    assert result.payload is not None
    assert torch.allclose(result.payload.action_chunk, payload.action_chunk)
    assert result.score is not None
    assert abs(result.score - 1.0) < 1e-5
    orch.clear()


# ---------------------------------------------------------------------------
# T4.3: different state -> MISS (deterministic vectors)
# ---------------------------------------------------------------------------


def test_write_then_check_different_state_misses():
    orch, _, _ = make_orchestrator()
    state_a = _unit_vector(32, 0)   # e0
    state_b = -_unit_vector(32, 0)  # -e0, cosine = -1

    payload = CachePayload(action_chunk=torch.randn(50, 32))
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state_a))
    orch.clear()

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state_b))
    assert result.hit_type == HitType.MISS
    orch.clear()


# ---------------------------------------------------------------------------
# T4.4: similar state near threshold -> FULL_HIT (deterministic)
# ---------------------------------------------------------------------------


def test_write_then_check_similar_state_near_threshold():
    orch, _, _ = make_orchestrator()
    base = _unit_vector(32, 0)
    similar = _vector_with_known_cosine(base, target_cos=0.99)  # > 0.98 threshold

    payload = CachePayload(action_chunk=torch.randn(50, 32))
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(base))
    orch.clear()

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(similar))
    assert result.hit_type == HitType.FULL_HIT
    orch.clear()


# ---------------------------------------------------------------------------
# T4.5: judge MISS does not call fetch_payload (two-phase semantics)
# ---------------------------------------------------------------------------


def test_judge_miss_does_not_fetch_payload():
    # Use a very high threshold so that search returns results but judge says MISS.
    judge = ThresholdJudge(cp1_threshold=1.01)  # impossible to reach
    orch, backend, storage = make_counting_orchestrator(judge=judge)

    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()

    # Reset counters after write.
    backend.fetch_payload_call_count = 0
    storage.fetch_payload_call_count = 0

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    assert result.payload is None
    # fetch_payload must NOT have been called.
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
    # search() should not have been called.
    assert backend.search_call_count == 0
    orch.clear()


# ---------------------------------------------------------------------------
# T4.7: write idempotent upsert (same state -> count=1)
# ---------------------------------------------------------------------------


def test_write_idempotent_upsert():
    orch, backend, _ = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()

    assert backend.count() == 1


# ---------------------------------------------------------------------------
# T4.8: different states create separate entries
# ---------------------------------------------------------------------------


def test_write_different_states_creates_separate_entries():
    orch, backend, _ = make_orchestrator()
    state_a = _unit_vector(32, 0)
    state_b = _unit_vector(32, 1)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state_a))
    orch.clear()
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state_b))
    orch.clear()

    assert backend.count() == 2


# ---------------------------------------------------------------------------
# T4.9: CP3 check always misses in Step 4
# ---------------------------------------------------------------------------


def test_cp3_check_always_misses_in_step4():
    orch, _, _ = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    # Write a CP1 entry.
    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()

    # CP3 check should miss (no CP3 entries, checkpoint_id filter).
    result = orch.check(CheckpointID.CP3, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
    orch.clear()


# ---------------------------------------------------------------------------
# T4.10: should_skip_inference returns None (stub)
# ---------------------------------------------------------------------------


def test_should_skip_inference_returns_none():
    orch, _, _ = make_orchestrator()
    assert orch.should_skip_inference() is None


# ---------------------------------------------------------------------------
# T4.11: schedule_next_action is no-op (stub)
# ---------------------------------------------------------------------------


def test_schedule_next_action_is_noop():
    orch, _, _ = make_orchestrator()
    orch.schedule_next_action(torch.randn(50, 32))
    assert orch.should_skip_inference() is None


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
    orch, _, _ = make_orchestrator()
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))

    orch.write(CheckpointID.CP1, payload, stage1=make_stage1(state))
    orch.clear()

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.score is not None
    assert result.entry_id is not None
    assert isinstance(result.entry_id, str)
    assert len(result.entry_id) == 32  # sha256[:32]
    orch.clear()


# ---------------------------------------------------------------------------
# T4.14: _stable_hash is deterministic
# ---------------------------------------------------------------------------


def test_stable_hash_deterministic():
    keys = {ROBOT_STATE: torch.randn(32)}
    h1 = _stable_hash(CheckpointID.CP1, keys)
    h2 = _stable_hash(CheckpointID.CP1, keys)
    assert h1 == h2


# ---------------------------------------------------------------------------
# T4.15: _stable_hash differs across checkpoints
# ---------------------------------------------------------------------------


def test_stable_hash_different_checkpoints_differ():
    keys = {ROBOT_STATE: torch.randn(32)}
    h1 = _stable_hash(CheckpointID.CP1, keys)
    h3 = _stable_hash(CheckpointID.CP3, keys)
    assert h1 != h3
