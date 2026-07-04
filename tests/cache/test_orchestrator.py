"""Tests for CacheOrchestrator."""

import math

import pytest
import torch
import torch.nn.functional as F

from openpi.cache.components.gate import (
    AlwaysSearchGate,
    ClientControlledGate,
    ScoreHysteresisGate,
)
from openpi.cache.components.judge import AlwaysWarmStartJudge, HitType, ThresholdJudge
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import CheckpointID

from openpi.cache.backends.in_memory_backend import InMemoryBackend

from tests.cache.conftest import (
    TestStorageSearchStrategy,
    _wrap_per_checkpoint,
    insert_entry,
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
    def __call__(self, checkpoint_id, cached_data, request_context=None):
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


def test_key_builder_property_returns_constructor_arg():
    """Public `key_builder` accessor (added for cp1_llm_layer_extract hook)
    must return the exact instance passed at construction so the
    InferenceInterceptor can call type-specific hooks like attach_model."""
    orch, _, _ = make_orchestrator()
    assert orch.key_builder is orch._key_builder


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


# ---------------------------------------------------------------------------
# WARM_START tests
# ---------------------------------------------------------------------------

_WARM_TIERS = [
    {"threshold": 0.95, "start_t": 0.3},
    {"threshold": 0.90, "start_t": 0.5},
]


def _warm_judge():
    return ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95, warm_tiers=_WARM_TIERS)


def test_warm_start_complete_payload():
    """WARM_START with complete intermediates returns payload and start_t."""
    orch, _, storage = make_orchestrator(judge=_warm_judge())

    state = _unit_vector(32, 0)
    payload = CachePayload(
        action_chunk=torch.randn(50, 32),
        intermediates={0.3: torch.randn(50, 32), 0.5: torch.randn(50, 32)},
        denoising_num_steps=10,
    )
    insert_entry(storage, CheckpointID.CP1, state, payload)

    query = _vector_with_known_cosine(state, 0.96)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(query))
    assert result.hit_type == HitType.WARM_START
    assert result.payload is not None
    assert result.start_t == 0.3
    assert result.payload.intermediates is not None
    assert 0.3 in result.payload.intermediates
    orch.clear()


def test_warm_start_missing_intermediates_downgrade():
    """WARM_START with no intermediates downgrades to MISS."""
    orch, _, storage = make_orchestrator(judge=_warm_judge())

    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    query = _vector_with_known_cosine(state, 0.96)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(query))
    assert result.hit_type == HitType.MISS
    assert result.score is not None
    assert result.entry_id is not None
    orch.clear()


def test_warm_start_missing_start_t_key_downgrade():
    """WARM_START with intermediates that lack the required start_t key downgrades to MISS."""
    orch, _, storage = make_orchestrator(judge=_warm_judge())

    state = _unit_vector(32, 0)
    payload = CachePayload(
        action_chunk=torch.randn(50, 32),
        intermediates={0.5: torch.randn(50, 32)},
        denoising_num_steps=10,
    )
    insert_entry(storage, CheckpointID.CP1, state, payload)

    query = _vector_with_known_cosine(state, 0.96)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(query))
    # Judge wants start_t=0.3 but payload only has 0.5 → downgrade
    assert result.hit_type == HitType.MISS
    orch.clear()


def test_warm_start_miss_count_incremented_on_downgrade():
    """Downgraded WARM_START increments miss counter."""
    orch, _, storage = make_orchestrator(judge=_warm_judge())

    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    query = _vector_with_known_cosine(state, 0.96)
    orch.check(CheckpointID.CP1, stage1=make_stage1(query))
    assert orch._miss_by_checkpoint.get(CheckpointID.CP1, 0) >= 1
    orch.clear()


# ---------------------------------------------------------------------------
# request_context passthrough + accepts_client_signal flag
# ---------------------------------------------------------------------------


class _RecordingGate:
    """Gate spy: records every (checkpoint_id, cached_data, request_context)
    triple it receives and returns a caller-configurable decision."""

    def __init__(self, decision: bool = True) -> None:
        self.decision = decision
        self.calls: list[tuple] = []

    def __call__(self, checkpoint_id, cached_data, request_context=None):
        self.calls.append((checkpoint_id, cached_data, request_context))
        return self.decision


def _make_orch_with_gate(gate):
    dims = {"robot_state": 32}
    backend = InMemoryBackend(dims)
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.key_builder import PlaceholderKeyBuilder
    from openpi.cache.timing import SystemTimer

    storage = CacheStorage(backend)
    strategy = TestStorageSearchStrategy(storage, top_k=1)
    return CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates=_wrap_per_checkpoint(gate),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )


def test_check_forwards_request_context_to_gate():
    spy = _RecordingGate(decision=False)
    orch = _make_orch_with_gate(spy)
    state = torch.randn(1, 32)
    ctx = {"gate_decision": "search", "extra": 123}

    orch.check(CheckpointID.CP1, request_context=ctx, stage1=make_stage1(state))

    assert len(spy.calls) == 1
    _, _, received_ctx = spy.calls[0]
    assert received_ctx is ctx
    orch.clear()


def test_check_defaults_request_context_to_none():
    spy = _RecordingGate(decision=False)
    orch = _make_orch_with_gate(spy)
    state = torch.randn(1, 32)

    orch.check(CheckpointID.CP1, stage1=make_stage1(state))

    assert len(spy.calls) == 1
    assert spy.calls[0][2] is None
    orch.clear()


def test_check_request_context_is_kwarg_only():
    """``request_context`` must not appear in ``**stage_outputs`` passthrough.

    The orchestrator declares it before ``**stage_outputs``, so passing it
    positionally raises TypeError instead of silently leaking into
    ``key_builder.collect`` stage kwargs.
    """
    spy = _RecordingGate(decision=False)
    orch = _make_orch_with_gate(spy)
    state = torch.randn(1, 32)

    with pytest.raises(TypeError):
        orch.check(CheckpointID.CP1, {"gate_decision": "skip"}, stage1=make_stage1(state))  # type: ignore[misc]
    orch.clear()


def test_accepts_client_signal_false_for_default_gates():
    orch, _, _ = make_orchestrator()
    assert orch.accepts_client_signal is False
    orch.clear()


def test_accepts_client_signal_true_when_any_checkpoint_uses_client_gate():
    """Mirror the YAML case: CP1 uses ClientControlledGate, CP3 is unset."""
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
        gates={
            CheckpointID.CP1: ClientControlledGate(),
            CheckpointID.CP3: AlwaysSearchGate(),
        },
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )
    assert orch.accepts_client_signal is True


def test_client_controlled_gate_skip_through_orchestrator():
    """End-to-end: request_context='skip' causes orchestrator to report MISS
    and avoid dispatching a search."""
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
        gates=_wrap_per_checkpoint(ClientControlledGate()),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )

    # Pre-populate so a non-skip path would produce a hit — proving the skip
    # short-circuits at the gate, not at search.
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(
        CheckpointID.CP1,
        request_context={"gate_decision": "skip"},
        stage1=make_stage1(state),
    )
    assert result.hit_type == HitType.MISS
    assert backend.search_call_count == 0
    orch.clear()


def test_client_controlled_gate_search_through_orchestrator():
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
        gates=_wrap_per_checkpoint(ClientControlledGate()),
        judges=_wrap_per_checkpoint(ThresholdJudge()),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )

    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(
        CheckpointID.CP1,
        request_context={"gate_decision": "search"},
        stage1=make_stage1(state),
    )
    assert result.hit_type == HitType.FULL_HIT
    orch.clear()


def test_prefill_mode_context_manager_enters_and_exits():
    orchestrator, _backend, storage = make_orchestrator()
    payload = CachePayload(action_chunk=torch.zeros(50, 32))

    assert storage._prefill_mode is False

    with orchestrator.prefill_mode(payload):
        assert storage._prefill_mode is True
        assert storage._prefill_payload is payload

    assert storage._prefill_mode is False
    assert storage._prefill_payload is None


def test_prefill_mode_context_manager_exits_on_exception():
    orchestrator, _backend, storage = make_orchestrator()
    payload = CachePayload(action_chunk=torch.zeros(50, 32))

    with pytest.raises(RuntimeError, match="boom"):
        with orchestrator.prefill_mode(payload):
            raise RuntimeError("boom")

    # finally branch ran — storage must be back to normal mode.
    assert storage._prefill_mode is False
    assert storage._prefill_payload is None


# ---------------------------------------------------------------------------
# G0a: task_key broadcast + verdict feedback to the gate (Stage 1c)
# ---------------------------------------------------------------------------


class _TaskKeySpyGate:
    """Gate that records the task_key broadcast to on_episode_start."""

    def __init__(self) -> None:
        self.received_task_key = None

    def __call__(self, checkpoint_id, cached_data, request_context=None) -> bool:
        return True

    def on_episode_start(self, task_key: str = "") -> None:
        self.received_task_key = task_key

    def record_action(self, action_chunk) -> None:
        pass


class _VerdictSpyGate:
    """Gate with a settable decision that records every record_verdict call."""

    def __init__(self, decision: bool = True) -> None:
        self.decision = decision
        self.verdicts: list[dict] = []

    def __call__(self, checkpoint_id, cached_data, request_context=None) -> bool:
        return self.decision

    def record_verdict(
        self, checkpoint_id, *, hit_type, cp1_score, winner_id, start_t, searched
    ) -> None:
        self.verdicts.append(
            {
                "checkpoint_id": checkpoint_id,
                "hit_type": hit_type,
                "cp1_score": cp1_score,
                "winner_id": winner_id,
                "start_t": start_t,
                "searched": searched,
            }
        )

    def on_episode_start(self, task_key: str = "") -> None:
        pass

    def record_action(self, action_chunk) -> None:
        pass


def test_task_key_broadcast_to_gate_on_episode_start():
    spy = _TaskKeySpyGate()
    orch, _, _ = make_orchestrator(gate=spy)
    orch.on_episode_start(task_key="pick_up_cup", episode_id="ep0")
    assert spy.received_task_key == "pick_up_cup"
    orch.clear()


def test_task_key_broadcast_to_gate_on_task_begin():
    spy = _TaskKeySpyGate()
    orch, _, _ = make_orchestrator(gate=spy)
    orch.on_task_begin("stack_blocks")
    assert spy.received_task_key == "stack_blocks"
    orch.clear()


def test_legacy_gate_without_task_key_param_unaffected():
    # AlwaysSearchGate.on_episode_start(self) has no task_key param; the
    # signature-filtered broadcast must not raise.
    orch, _, _ = make_orchestrator(gate=AlwaysSearchGate())
    orch.on_episode_start(task_key="anything", episode_id="ep0")  # must not raise
    orch.clear()


def test_verdict_fed_to_gate_on_full_hit():
    spy = _VerdictSpyGate(decision=True)
    orch, _, storage = make_orchestrator(gate=spy)
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    entry = insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.FULL_HIT

    assert len(spy.verdicts) == 1
    v = spy.verdicts[0]
    assert v["searched"] is True
    assert v["hit_type"] == HitType.FULL_HIT
    assert v["winner_id"] == entry.id
    assert v["cp1_score"] is not None and abs(v["cp1_score"] - 1.0) < 1e-5
    orch.clear()


def test_verdict_fed_to_gate_on_empty_store_miss():
    # Always-search MISS with an empty result set: searched=True but
    # cp1_score is None (no results). The gate must be told searched=True.
    spy = _VerdictSpyGate(decision=True)
    orch, _, _ = make_orchestrator(gate=spy)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(torch.randn(1, 32)))
    assert result.hit_type == HitType.MISS

    assert len(spy.verdicts) == 1
    v = spy.verdicts[0]
    assert v["searched"] is True
    assert v["hit_type"] == HitType.MISS
    assert v["cp1_score"] is None
    orch.clear()


def test_verdict_fed_to_gate_on_skip():
    # Gate returns skip -> no search -> record_verdict(searched=False, None).
    spy = _VerdictSpyGate(decision=False)
    orch, _, _ = make_orchestrator(gate=spy)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(torch.randn(1, 32)))
    assert result.hit_type == HitType.MISS
    assert result.searched is False

    assert len(spy.verdicts) == 1
    v = spy.verdicts[0]
    assert v["searched"] is False
    assert v["cp1_score"] is None
    assert v["hit_type"] == HitType.MISS
    orch.clear()


def test_verdict_fed_to_gate_on_warm_start_downgrade():
    # WARM_START whose payload lacks intermediates downgrades to MISS. The gate
    # must be fed the FINAL verdict (MISS) with searched=True, cp1_score = top
    # score, and winner_id / start_t passed from the judge result.
    spy = _VerdictSpyGate(decision=True)
    orch, _, storage = make_orchestrator(
        gate=spy, judge=AlwaysWarmStartJudge(start_t=0.5)
    )
    state = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))  # no intermediates
    entry = insert_entry(storage, CheckpointID.CP1, state, payload)

    result = orch.check(CheckpointID.CP1, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS  # WARM_START downgraded to MISS

    assert len(spy.verdicts) == 1
    v = spy.verdicts[0]
    assert v["searched"] is True
    assert v["hit_type"] == HitType.MISS
    assert v["cp1_score"] is not None and abs(v["cp1_score"] - 1.0) < 1e-5
    assert v["winner_id"] == entry.id
    assert v["start_t"] == 0.5
    orch.clear()


def test_check_no_crash_when_gate_lacks_record_verdict():
    # AlwaysSearchGate has no record_verdict; the hasattr guard must skip it.
    gate = AlwaysSearchGate()
    assert not hasattr(gate, "record_verdict")
    orch, _, _ = make_orchestrator(gate=gate)
    result = orch.check(CheckpointID.CP1, stage1=make_stage1(torch.randn(1, 32)))
    assert result.hit_type == HitType.MISS  # empty store; no exception raised
    orch.clear()


def test_score_hysteresis_gate_closes_loop_through_orchestrator():
    # End-to-end: the server-side N1 gate drives skip purely from verdict
    # feedback inside check() — no client, no __gate_decision__.
    gate = ScoreHysteresisGate(theta_low=0.5, theta_high=0.9, j=2, probe_interval=3)
    orch, _, storage = make_orchestrator(
        gate=gate,
        judge=__import__(
            "openpi.cache.components.judge", fromlist=["ThresholdJudge"]
        ).ThresholdJudge(cp1_threshold=0.9, cp3_threshold=0.95),
    )
    base = _unit_vector(32, 0)
    payload = CachePayload(action_chunk=torch.randn(50, 32))
    insert_entry(storage, CheckpointID.CP1, base, payload)

    low = _vector_with_known_cosine(base, 0.1)  # score 0.1 < theta_low
    # Query order: high, low, low, low.
    queries = [base, low, low, low]
    searched_flags = []
    for q in queries:
        r = orch.check(CheckpointID.CP1, stage1=make_stage1(q))
        searched_flags.append(r.searched)

    # First 3 searched (gate still searching / draining low_run); after 2 low
    # scores hit j=2 the gate stops -> the 4th step is a gate-skip.
    assert searched_flags == [True, True, True, False]
    orch.clear()
