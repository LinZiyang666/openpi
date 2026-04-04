"""Tests for PlaceholderKeyBuilder (T1.1–T1.9)."""

from types import SimpleNamespace

import pytest
import torch

from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.types import ROBOT_STATE, CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage1(state: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _stage3(action_chunk: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(action_chunk=action_chunk)


# ---------------------------------------------------------------------------
# T1.1: collect stores GPU refs (zero-copy)
# ---------------------------------------------------------------------------


def test_collect_stores_refs():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    assert "state" in kb.cached_data
    # Same object — zero copy.
    assert kb.cached_data["state"] is state
    assert kb.cached_data["state"].shape == (1, 32)


# ---------------------------------------------------------------------------
# T1.2: collect stage3 stores action_chunk
# ---------------------------------------------------------------------------


def test_collect_stage3_stores_action_chunk():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    action = torch.randn(1, 50, 32)
    kb.collect(
        CheckpointID.CP3,
        stage1=_stage1(state),
        stage3=_stage3(action),
    )
    assert "state" in kb.cached_data
    assert "action_chunk" in kb.cached_data


# ---------------------------------------------------------------------------
# T1.3: second collect clears previous
# ---------------------------------------------------------------------------


def test_collect_clears_previous():
    kb = PlaceholderKeyBuilder()
    state_a = torch.randn(1, 32)
    state_b = torch.randn(1, 32)
    kb.collect(CheckpointID.CP1, stage1=_stage1(state_a))
    kb.collect(CheckpointID.CP1, stage1=_stage1(state_b))
    assert kb.cached_data["state"] is state_b


# ---------------------------------------------------------------------------
# T1.4: build CP1 returns normalized CPU float32 contiguous
# ---------------------------------------------------------------------------


def test_build_cp1_returns_normalized_cpu_float32():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    result = kb.build(CheckpointID.CP1)

    assert ROBOT_STATE in result
    t = result[ROBOT_STATE]
    assert t.device == torch.device("cpu")
    assert t.dtype == torch.float32
    assert t.is_contiguous()
    assert t.shape == (32,)
    # L2 norm should be ~1.0.
    norm = torch.linalg.norm(t).item()
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# T1.5: build CP3 same as CP1
# ---------------------------------------------------------------------------


def test_build_cp3_same_as_cp1():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    cp1 = kb.build(CheckpointID.CP1)

    kb.collect(CheckpointID.CP3, stage1=_stage1(state))
    cp3 = kb.build(CheckpointID.CP3)

    assert torch.allclose(cp1[ROBOT_STATE], cp3[ROBOT_STATE])


# ---------------------------------------------------------------------------
# T1.6: build unsupported checkpoint raises
# ---------------------------------------------------------------------------


def test_build_unsupported_checkpoint_raises():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    kb.collect(CheckpointID.CP2, stage1=_stage1(state))
    with pytest.raises(ValueError, match="Unsupported"):
        kb.build(CheckpointID.CP2)


# ---------------------------------------------------------------------------
# T1.7: build is deterministic
# ---------------------------------------------------------------------------


def test_build_deterministic():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)

    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    r1 = kb.build(CheckpointID.CP1)

    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    r2 = kb.build(CheckpointID.CP1)

    assert torch.allclose(r1[ROBOT_STATE], r2[ROBOT_STATE])


# ---------------------------------------------------------------------------
# T1.8: clear empties cache
# ---------------------------------------------------------------------------


def test_clear_empties_cache():
    kb = PlaceholderKeyBuilder()
    state = torch.randn(1, 32)
    kb.collect(CheckpointID.CP1, stage1=_stage1(state))
    kb.clear()
    assert kb.cached_data == {}


# ---------------------------------------------------------------------------
# T1.9: build without collect raises KeyError
# ---------------------------------------------------------------------------


def test_build_without_collect_raises():
    kb = PlaceholderKeyBuilder()
    with pytest.raises(KeyError):
        kb.build(CheckpointID.CP1)
