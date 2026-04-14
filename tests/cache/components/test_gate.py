"""Tests for gate implementations."""

import torch

from openpi.cache.components.gate import AlwaysSearchGate, AlwaysSkipGate, GateFunction
from openpi.cache.types import CheckpointID


def test_always_search_gate_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}) is True


def test_always_search_gate_cp3_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is True


def test_always_search_gate_conforms_to_protocol():
    assert isinstance(AlwaysSearchGate(), GateFunction)


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
