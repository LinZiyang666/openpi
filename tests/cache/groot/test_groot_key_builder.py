"""Mask-driven key slicing: the failure this pins is the one that does not raise."""

from __future__ import annotations

import pytest
import torch

from openpi.cache.groot.key_builder import (
    GrootCP1MeanPoolKeyBuilder,
    GrootCP1SpatialPool16KeyBuilder,
    slice_groot_cp1_fields,
)
from openpi.cache.types import CheckpointID

from .conftest import EMB_DIM, N_CAMERAS, STATE_VALID, TOKENS_PER_IMAGE


def _stage1(runner, model, prompt_tokens):
    with runner.session():
        return runner.run_stage1(model.build_inputs(prompt_tokens=prompt_tokens))


def test_vision_slices_are_identical_across_prompt_lengths(runner, stub_model):
    """The whole reason the slicing is mask-driven rather than offset-driven.

    Same images, different instruction length: a fixed offset table would cut
    at the wrong positions and still return correctly-shaped vectors.
    """
    short = _stage1(runner, stub_model, prompt_tokens=2)
    long = _stage1(runner, stub_model, prompt_tokens=17)

    a = slice_groot_cp1_fields(
        short.input_embeds, short.image_token_mask, short.state, short.state_mask, None
    )
    b = slice_groot_cp1_fields(
        long.input_embeds, long.image_token_mask, long.state, long.state_mask, None
    )

    assert short.input_embeds.shape[1] != long.input_embeds.shape[1]
    for field in ("vision_0", "vision_1", "vision_2"):
        assert torch.equal(a[field], b[field]), field


def test_each_camera_gets_its_own_run_in_order(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    for index, field in enumerate(("vision_0", "vision_1", "vision_2")):
        assert raw[field].shape == (TOKENS_PER_IMAGE, EMB_DIM)
        # The stub's extract_feature stamps camera i with the constant i + 1.
        assert torch.allclose(raw[field], torch.full_like(raw[field], index + 1.0))


def test_prompt_is_everything_the_vision_did_not_overwrite(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=4)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    expected = stage1.input_embeds.shape[1] - N_CAMERAS * TOKENS_PER_IMAGE
    assert raw["prompt_emb"].shape == (expected, EMB_DIM)


def test_slices_are_float32_even_though_stage1_is_not(runner, stub_model):
    """Pooling dtype has to match the offline path, which reads fp16 and upcasts."""
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    for value in raw.values():
        assert value.dtype is torch.float32


def test_robot_state_keeps_only_the_valid_dimensions(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    assert raw["robot_state"].shape == (STATE_VALID,)


def test_wrong_number_of_image_runs_is_refused(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    broken = stage1.image_token_mask.clone()
    broken[0, :] = False
    with pytest.raises(RuntimeError, match="contiguous image-token runs"):
        slice_groot_cp1_fields(
            stage1.input_embeds, broken, stage1.state, stage1.state_mask, None
        )


def test_short_image_run_is_refused(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    broken = stage1.image_token_mask.clone()
    first = int(torch.nonzero(broken[0]).flatten()[0])
    broken[0, first] = False  # leaves a 255-token run
    with pytest.raises(RuntimeError, match="must be 256 tokens"):
        slice_groot_cp1_fields(
            stage1.input_embeds, broken, stage1.state, stage1.state_mask, None
        )


def test_state_mask_change_mid_episode_is_refused(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    other = torch.zeros_like(stage1.state_mask)
    other[0, 0, :2] = True
    with pytest.raises(RuntimeError, match="state_mask changed"):
        slice_groot_cp1_fields(
            stage1.input_embeds,
            stage1.image_token_mask,
            stage1.state,
            other,
            None,
            expected_state_index=stage1.state_mask[0, -1],
        )


def test_enabled_fields_filter_is_honoured(runner, stub_model):
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds,
        stage1.image_token_mask,
        stage1.state,
        stage1.state_mask,
        {"vision_0", "robot_state"},
    )
    assert set(raw) == {"vision_0", "robot_state"}


def test_builder_output_matches_a_hand_computed_pool(runner, stub_model):
    """A value assertion that goes through build(), not just the pooling helper.

    Shape-only assertions cannot tell a correct slice from one that cut into
    the prompt, which is exactly the mistake worth catching.
    """
    builder = GrootCP1MeanPoolKeyBuilder(enabled_fields=["vision_1", "robot_state"])
    stage1 = _stage1(runner, stub_model, prompt_tokens=6)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = builder.build(CheckpointID.CP1)

    assert set(keys) == {"vision_1", "robot_state"}
    # Camera 1's tokens are all 2.0 in the stub, so the mean is 2.0 everywhere.
    assert torch.allclose(keys["vision_1"], torch.full((EMB_DIM,), 2.0))


def test_spatial_pool_dimension_is_grid_squared_times_embedding(runner, stub_model):
    builder = GrootCP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "prompt_emb"])
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = builder.build(CheckpointID.CP1)
    assert keys["vision_0"].shape == (16 * EMB_DIM,)
    assert keys["prompt_emb"].shape == (EMB_DIM,)


def test_builder_latches_the_state_index_and_resets_per_episode(runner, stub_model):
    builder = GrootCP1MeanPoolKeyBuilder()
    stage1 = _stage1(runner, stub_model, prompt_tokens=3)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    builder.build(CheckpointID.CP1)

    narrowed = torch.zeros_like(stage1.state_mask)
    narrowed[0, 0, :2] = True
    stage1.action_inputs["state_mask"] = narrowed
    builder.collect(CheckpointID.CP1, stage1=stage1)
    with pytest.raises(RuntimeError, match="state_mask changed"):
        builder.build(CheckpointID.CP1)

    builder.on_episode_start()
    builder.collect(CheckpointID.CP1, stage1=stage1)
    builder.build(CheckpointID.CP1)  # new episode re-latches, no raise
