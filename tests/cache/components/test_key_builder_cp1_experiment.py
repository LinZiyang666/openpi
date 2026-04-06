"""Tests for CP1 experiment key builders."""

from types import SimpleNamespace

import pytest
import torch

from openpi.cache.components.key_builder import (
    CP1MaxPoolKeyBuilder,
    CP1MeanPoolKeyBuilder,
    CP1SpatialPool16KeyBuilder,
    CP1SpatialPool64KeyBuilder,
    _max_pool_tokens,
    _mean_pool_tokens,
    _spatial_pool_tokens,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMB_DIM = 2048
STATE_DIM = 32
NUM_PROMPT_TOKENS = 20


class _FakeStage1:
    def __init__(self, prefix_embs, state):
        self.prefix_embs = prefix_embs
        self.state = state


class _FakeStage3:
    def __init__(self, action_chunk):
        self.action_chunk = action_chunk


def _make_fake_stage1(
    emb_dim: int = EMB_DIM,
    state_dim: int = STATE_DIM,
    num_prompt_tokens: int = NUM_PROMPT_TOKENS,
) -> _FakeStage1:
    prefix_len = 256 * 3 + num_prompt_tokens
    prefix_embs = torch.randn(1, prefix_len, emb_dim)
    state = torch.randn(1, state_dim)
    return _FakeStage1(prefix_embs, state)


ALL_FIELDS = [VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE]
LIBERO_FIELDS = [VISION_0, VISION_1, PROMPT_EMB, ROBOT_STATE]  # no vision_2


# ---------------------------------------------------------------------------
# TestCP1MeanPoolKeyBuilder
# ---------------------------------------------------------------------------


class TestCP1MeanPoolKeyBuilder:
    def test_output_shapes(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys[VISION_0].shape == (EMB_DIM,)
        assert keys[VISION_1].shape == (EMB_DIM,)
        assert keys[VISION_2].shape == (EMB_DIM,)
        assert keys[PROMPT_EMB].shape == (EMB_DIM,)
        assert keys[ROBOT_STATE].shape == (STATE_DIM,)

    def test_output_cpu_float32(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        for k, v in keys.items():
            assert v.device.type == "cpu", f"{k} not on CPU"
            assert v.dtype == torch.float32, f"{k} not float32"
            assert v.is_contiguous(), f"{k} not contiguous"

    def test_cp3_same_as_cp1(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=LIBERO_FIELDS)
        s1 = _make_fake_stage1()
        builder.collect(CheckpointID.CP1, stage1=s1)
        keys_cp1 = builder.build(CheckpointID.CP1)
        builder.collect(CheckpointID.CP3, stage1=s1)
        keys_cp3 = builder.build(CheckpointID.CP3)
        for field in keys_cp1:
            assert torch.allclose(keys_cp1[field], keys_cp3[field])

    def test_vision_2_excluded_when_disabled(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=LIBERO_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert VISION_2 not in keys
        assert VISION_0 in keys

    def test_vision_2_included_when_enabled(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert VISION_2 in keys
        assert keys[VISION_2].shape == (EMB_DIM,)

    def test_action_chunk_cached(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=LIBERO_FIELDS)
        action = torch.randn(1, 50, 32)
        builder.collect(
            CheckpointID.CP3,
            stage1=_make_fake_stage1(),
            stage3=_FakeStage3(action),
        )
        assert "action_chunk" in builder.cached_data
        assert builder.cached_data["action_chunk"] is action

    def test_clear(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=LIBERO_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        builder.clear()
        assert len(builder.cached_data) == 0

    def test_unsupported_checkpoint(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=LIBERO_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        with pytest.raises(ValueError, match="Unsupported"):
            builder.build(CheckpointID.CP2)

    def test_robot_state_not_normalized(self):
        """robot_state must be raw vector, not L2 normalized."""
        builder = CP1MeanPoolKeyBuilder(enabled_fields=[ROBOT_STATE])
        state = torch.tensor([[3.0, 4.0] + [0.0] * 30])
        s1 = _FakeStage1(torch.randn(1, 256 * 3 + 20, EMB_DIM), state)
        builder.collect(CheckpointID.CP1, stage1=s1)
        keys = builder.build(CheckpointID.CP1)
        rs = keys[ROBOT_STATE]
        assert abs(rs[0].item() - 3.0) < 1e-5
        assert abs(rs[1].item() - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# TestCP1SpatialPool16KeyBuilder
# ---------------------------------------------------------------------------


class TestCP1SpatialPool16KeyBuilder:
    def test_output_shapes(self):
        builder = CP1SpatialPool16KeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys[VISION_0].shape == (16 * EMB_DIM,)  # 4x4 * 2048 = 32768
        assert keys[VISION_1].shape == (16 * EMB_DIM,)
        assert keys[PROMPT_EMB].shape == (EMB_DIM,)  # mean pooled
        assert keys[ROBOT_STATE].shape == (STATE_DIM,)

    def test_spatial_pool_correctness(self):
        """When each 4x4 block in 16x16 grid has uniform value, pooling should preserve it."""
        emb_dim = 4
        tokens = torch.zeros(256, emb_dim)
        # Fill top-left 4x4 block (rows 0-3, cols 0-3 in 16x16 grid)
        for r in range(4):
            for c in range(4):
                tokens[r * 16 + c] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _spatial_pool_tokens(tokens, grid_size=16, pool_size=4)
        # After pooling to 4x4, the top-left cell should be [1,2,3,4]
        assert result.shape == (16 * emb_dim,)
        assert torch.allclose(result[:emb_dim], torch.tensor([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# TestCP1SpatialPool64KeyBuilder
# ---------------------------------------------------------------------------


class TestCP1SpatialPool64KeyBuilder:
    def test_output_shapes(self):
        builder = CP1SpatialPool64KeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys[VISION_0].shape == (4 * EMB_DIM,)  # 2x2 * 2048 = 8192
        assert keys[VISION_1].shape == (4 * EMB_DIM,)
        assert keys[PROMPT_EMB].shape == (EMB_DIM,)
        assert keys[ROBOT_STATE].shape == (STATE_DIM,)


# ---------------------------------------------------------------------------
# TestCP1MaxPoolKeyBuilder
# ---------------------------------------------------------------------------


class TestCP1MaxPoolKeyBuilder:
    def test_output_shapes(self):
        builder = CP1MaxPoolKeyBuilder(enabled_fields=ALL_FIELDS)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        assert keys[VISION_0].shape == (EMB_DIM,)
        assert keys[PROMPT_EMB].shape == (EMB_DIM,)
        assert keys[ROBOT_STATE].shape == (STATE_DIM,)

    def test_max_pool_picks_maximum(self):
        """Verify max pooling takes per-dim max."""
        tokens = torch.tensor([
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 6.0],
            [7.0, 1.0, 2.0],
        ])
        result = _max_pool_tokens(tokens)
        assert torch.allclose(result, torch.tensor([7.0, 5.0, 6.0]))

    def test_mean_pool_correctness(self):
        tokens = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        result = _mean_pool_tokens(tokens)
        assert torch.allclose(result, torch.tensor([2.0, 3.0]))


# ---------------------------------------------------------------------------
# TestEnabledFieldsNone (all fields enabled by default)
# ---------------------------------------------------------------------------


class TestEnabledFieldsNone:
    def test_none_enables_all(self):
        builder = CP1MeanPoolKeyBuilder(enabled_fields=None)
        builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
        keys = builder.build(CheckpointID.CP1)
        for field in ALL_FIELDS:
            assert field in keys, f"{field} missing when enabled_fields=None"
