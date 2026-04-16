"""Tests for gate implementations."""

import pytest
import torch

from openpi.cache.components.gate import (
    AlwaysSearchGate,
    AlwaysSkipGate,
    ClientControlledGate,
    GateFunction,
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
