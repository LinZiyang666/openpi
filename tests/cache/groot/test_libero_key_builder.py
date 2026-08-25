"""The LIBERO GR00T checkpoints feed two cameras, not RoboCasa365's three.

The camera-to-field mapping is defined by image-token run order, so the run
count is load-bearing: a two-camera observation sliced with the three-camera
expectation would either raise or silently mislabel wrist tokens as vision_1.
These tests pin both directions.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.groot.key_builder import (
    GrootLiberoCP1SpatialPool16KeyBuilder,
    slice_groot_cp1_fields,
)
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, VISION_0, VISION_1, VISION_2

EMB = 8
TOKENS = 256


def _stage1(n_cameras: int, prompt_tokens: int = 5):
    """One synthetic stage-1 sequence: n_cameras image runs then the prompt.

    Upstream separates the runs with template tokens, so the runs must not be
    adjacent -- glued together they would read as a single longer run.
    """
    total = n_cameras * (TOKENS + 1) + prompt_tokens
    embeds = torch.arange(total * EMB, dtype=torch.float32).reshape(1, total, EMB)
    mask = torch.zeros(1, total, dtype=torch.bool)
    for c in range(n_cameras):
        start = c * (TOKENS + 1)
        mask[0, start : start + TOKENS] = True
    # state/state_mask are [B, horizon, dim]; the slicer reads the last step
    state = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    state_mask = torch.ones(1, 1, 8, dtype=torch.bool)
    return embeds, mask, state, state_mask


class TestTwoCameraSlicing:
    def test_two_runs_map_to_vision_0_and_1(self):
        embeds, mask, state, state_mask = _stage1(2)
        out = slice_groot_cp1_fields(
            embeds, mask, state, state_mask, None,
            vision_fields=(VISION_0, VISION_1),
        )
        assert set(out) == {VISION_0, VISION_1, PROMPT_EMB, ROBOT_STATE}
        assert VISION_2 not in out
        assert out[VISION_0].shape == (TOKENS, EMB)
        assert out[VISION_1].shape == (TOKENS, EMB)
        # run order is camera order: vision_1 starts where vision_0 ends
        assert torch.equal(out[VISION_0][0], embeds[0, 0])
        assert torch.equal(out[VISION_1][0], embeds[0, TOKENS + 1])

    def test_three_camera_default_rejects_a_two_camera_sequence(self):
        embeds, mask, state, state_mask = _stage1(2)
        with pytest.raises(RuntimeError, match="expected 3 contiguous image-token runs"):
            slice_groot_cp1_fields(embeds, mask, state, state_mask, None)

    def test_two_camera_fields_reject_a_three_camera_sequence(self):
        embeds, mask, state, state_mask = _stage1(3)
        with pytest.raises(RuntimeError, match="expected 2 contiguous image-token runs"):
            slice_groot_cp1_fields(
                embeds, mask, state, state_mask, None,
                vision_fields=(VISION_0, VISION_1),
            )


class TestLiberoBuilder:
    def test_builder_narrows_the_camera_list(self):
        assert GrootLiberoCP1SpatialPool16KeyBuilder.vision_fields == (VISION_0, VISION_1)

    def test_registry_resolves_the_libero_types(self):
        from openpi.cache.config import KeyBuilderConfig, _build_key_builder

        kb = _build_key_builder(
            KeyBuilderConfig(type="cp1_groot_libero_spatial_pool_16"),
            enabled_fields=[VISION_0, VISION_1, ROBOT_STATE],
            vector_dims={},
        )
        assert isinstance(kb, GrootLiberoCP1SpatialPool16KeyBuilder)
