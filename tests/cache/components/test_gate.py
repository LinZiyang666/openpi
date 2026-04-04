"""Tests for AlwaysSearchGate (T2.1–T2.3)."""

import torch

from openpi.cache.components.gate import AlwaysSearchGate, GateFunction
from openpi.cache.types import CheckpointID


def test_always_search_gate_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP1, {}) is True


def test_always_search_gate_cp3_returns_true():
    gate = AlwaysSearchGate()
    assert gate(CheckpointID.CP3, {"state": torch.randn(32)}) is True


def test_always_search_gate_conforms_to_protocol():
    assert isinstance(AlwaysSearchGate(), GateFunction)
