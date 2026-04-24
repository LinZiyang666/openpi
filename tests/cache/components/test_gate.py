"""Tests for gate implementations."""

import pytest
import torch

from openpi.cache.components.gate import (
    AlwaysSearchGate,
    AlwaysSkipGate,
    ClientControlledGate,
    GateFunction,
    PeriodicGate,
    RandomGate,
)
from openpi.cache.types import CheckpointID


def test_always_search_gate_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}) is True


def test_always_search_gate_cp3_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is True


def test_always_search_gate_conforms_to_protocol():
    assert isinstance(AlwaysSearchGate(), GateFunction)


def test_always_search_gate_ignores_request_context():
    # Default gates accept request_context for protocol conformance but
    # never consume it; presence must not alter the decision.
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) is True
    assert gate(CheckpointID.CP1, {}, None) is True


def test_always_skip_gate_returns_false():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP1, {}) is False


def test_always_skip_gate_cp3_returns_false():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is False


def test_always_skip_gate_conforms_to_protocol():
    assert isinstance(AlwaysSkipGate(), GateFunction)


def test_always_skip_gate_lifecycle_hooks_are_noop():
    # Both hooks should be callable and return None without touching state.
    gate = AlwaysSkipGate()
    assert gate.on_episode_start() is None
    assert gate.record_action(torch.zeros(1)) is None


def test_always_skip_gate_ignores_request_context():
    gate = AlwaysSkipGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "search"}) is False
    assert gate(CheckpointID.CP1, {}, None) is False


# ---------------------------------------------------------------------------
# ClientControlledGate
# ---------------------------------------------------------------------------


def test_client_controlled_gate_skip_returns_false():
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) is False


def test_client_controlled_gate_search_returns_true():
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP1, {}, {"gate_decision": "search"}) is True


def test_client_controlled_gate_ignores_cached_data():
    # Decision depends only on the request signal. Different cached_data
    # must not flip the outcome.
    gate = ClientControlledGate()
    assert gate(CheckpointID.CP3, {}, {"gate_decision": "skip"}) is False
    assert (
        gate(CheckpointID.CP3, {"state": torch.randn(32)}, {"gate_decision": "skip"})
        is False
    )


def test_client_controlled_gate_missing_request_context_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="requires request_context"):
        gate(CheckpointID.CP1, {}, None)


def test_client_controlled_gate_missing_gate_decision_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="requires request_context"):
        gate(CheckpointID.CP1, {}, {"something_else": "skip"})


def test_client_controlled_gate_unknown_value_raises():
    gate = ClientControlledGate()
    with pytest.raises(ValueError, match="unknown gate_decision"):
        gate(CheckpointID.CP1, {}, {"gate_decision": "maybe"})


def test_client_controlled_gate_conforms_to_protocol():
    assert isinstance(ClientControlledGate(), GateFunction)


def test_client_controlled_gate_lifecycle_hooks_are_noop():
    gate = ClientControlledGate()
    assert gate.on_episode_start() is None
    assert gate.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# RandomGate
# ---------------------------------------------------------------------------


def test_random_gate_rejects_invalid_p_inference():
    with pytest.raises(ValueError, match="p_inference"):
        RandomGate(p_inference=-0.1, seed=0)
    with pytest.raises(ValueError, match="p_inference"):
        RandomGate(p_inference=1.5, seed=0)


def test_random_gate_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        RandomGate(p_inference=0.3, seed=-1)


def test_random_gate_p_zero_always_searches():
    gate = RandomGate(p_inference=0.0, seed=0)
    for _ in range(200):
        assert gate(CheckpointID.CP1, {}) is True


def test_random_gate_p_one_always_skips():
    gate = RandomGate(p_inference=1.0, seed=0)
    for _ in range(200):
        assert gate(CheckpointID.CP1, {}) is False


def test_random_gate_per_connection_determinism():
    # Fixed seed + identical on_episode_start count => byte-identical stream.
    def draw(seed: int, ep_count: int, n: int) -> list[bool]:
        g = RandomGate(p_inference=0.5, seed=seed)
        for _ in range(ep_count):
            g.on_episode_start()
        return [g(CheckpointID.CP1, {}) for _ in range(n)]

    assert draw(seed=7, ep_count=3, n=32) == draw(seed=7, ep_count=3, n=32)


def test_random_gate_consecutive_episodes_have_distinct_streams():
    g = RandomGate(p_inference=0.5, seed=7)
    g.on_episode_start()
    ep1 = [g(CheckpointID.CP1, {}) for _ in range(64)]
    g.on_episode_start()
    ep2 = [g(CheckpointID.CP1, {}) for _ in range(64)]
    assert ep1 != ep2


def test_random_gate_ignores_request_context_and_cached_data():
    # Decision depends only on internal RNG; extra inputs must not alter
    # the sequence.
    g1 = RandomGate(p_inference=0.5, seed=0)
    g1.on_episode_start()
    seq1 = [g1(CheckpointID.CP1, {}, None) for _ in range(32)]

    g2 = RandomGate(p_inference=0.5, seed=0)
    g2.on_episode_start()
    seq2 = [
        g2(CheckpointID.CP1, {"state": torch.randn(32)}, {"gate_decision": "skip"})
        for _ in range(32)
    ]
    assert seq1 == seq2


def test_random_gate_conforms_to_protocol():
    assert isinstance(RandomGate(p_inference=0.3, seed=0), GateFunction)


def test_random_gate_record_action_is_noop():
    g = RandomGate(p_inference=0.3, seed=0)
    assert g.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# PeriodicGate
# ---------------------------------------------------------------------------


def test_periodic_gate_rejects_invalid_lengths():
    with pytest.raises(ValueError, match="cache_len"):
        PeriodicGate(cache_len=0, inference_len=1)
    with pytest.raises(ValueError, match="inference_len"):
        PeriodicGate(cache_len=1, inference_len=0)


def test_periodic_gate_duty_cycle():
    gate = PeriodicGate(cache_len=3, inference_len=2)
    # One full period: TTTFF.
    expected = [True, True, True, False, False]
    got = [gate(CheckpointID.CP1, {}) for _ in range(5)]
    assert got == expected
    # Second period replays identical pattern.
    got2 = [gate(CheckpointID.CP1, {}) for _ in range(5)]
    assert got2 == expected


def test_periodic_gate_k1_n1_alternates():
    gate = PeriodicGate(cache_len=1, inference_len=1)
    seq = [gate(CheckpointID.CP1, {}) for _ in range(6)]
    assert seq == [True, False, True, False, True, False]


def test_periodic_gate_on_episode_start_resets():
    gate = PeriodicGate(cache_len=2, inference_len=3)
    # Advance mid-period.
    for _ in range(3):
        gate(CheckpointID.CP1, {})
    # Reset restores cache-first ordering.
    gate.on_episode_start()
    assert gate(CheckpointID.CP1, {}) is True   # first of new cache block
    assert gate(CheckpointID.CP1, {}) is True


def test_periodic_gate_ignores_request_context_and_cached_data():
    gate = PeriodicGate(cache_len=1, inference_len=1)
    # Same sequence regardless of extra inputs.
    seq = [gate(CheckpointID.CP1, {}, {"gate_decision": "skip"}) for _ in range(4)]
    assert seq == [True, False, True, False]


def test_periodic_gate_conforms_to_protocol():
    assert isinstance(PeriodicGate(cache_len=2, inference_len=1), GateFunction)


def test_periodic_gate_record_action_is_noop():
    gate = PeriodicGate(cache_len=2, inference_len=1)
    assert gate.record_action(torch.zeros(1)) is None


# ---------------------------------------------------------------------------
# Constructor-level int regressions (G2 Round 1 Blocking #2 defense in depth)
# ---------------------------------------------------------------------------


def test_random_gate_ctor_rejects_fractional_seed():
    with pytest.raises(TypeError, match="seed"):
        RandomGate(p_inference=0.3, seed=0.5)  # type: ignore[arg-type]


def test_random_gate_ctor_rejects_bool_seed():
    with pytest.raises(TypeError, match="seed"):
        RandomGate(p_inference=0.3, seed=True)  # type: ignore[arg-type]


def test_random_gate_ctor_rejects_non_numeric_p():
    with pytest.raises(TypeError, match="p_inference"):
        RandomGate(p_inference="0.3", seed=0)  # type: ignore[arg-type]


def test_periodic_gate_ctor_rejects_fractional_cache_len():
    with pytest.raises(TypeError, match="cache_len"):
        PeriodicGate(cache_len=1.5, inference_len=1)  # type: ignore[arg-type]


def test_periodic_gate_ctor_rejects_bool_inference_len():
    with pytest.raises(TypeError, match="inference_len"):
        PeriodicGate(cache_len=1, inference_len=True)  # type: ignore[arg-type]
