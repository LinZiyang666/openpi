"""Online mask-slicing and offline offset-slicing must agree.

The online path locates image runs by token id; the offline artifact builder
reassembles a prefix from separately-stored fields and slices it at fixed
offsets. They are different code, so agreement is not free — and disagreement
would be invisible, because both produce correctly-shaped vectors either way.

This is the structural half of the gate: fed the same fp32 content, the two
must be bit-identical. The end-to-end half — real bf16 activations round-
tripped through fp16 storage — is a tolerance check and needs the real model,
so it lives in the manual island-B suite.
"""

from __future__ import annotations

import torch

from exp.common.build_in_memory_cache_artifact import _build_fake_stage1
from openpi.cache.components.key_builder import CP1MeanPoolKeyBuilder
from openpi.cache.groot.key_builder import (
    GrootCP1MeanPoolKeyBuilder,
    slice_groot_cp1_fields,
)
from openpi.cache.types import CheckpointID

from .conftest import EMB_DIM, TOKENS_PER_IMAGE


def _online_stage1(runner, model, prompt_tokens):
    with runner.session():
        return runner.run_stage1(model.build_inputs(prompt_tokens=prompt_tokens))


def _hdf5_like(raw: dict[str, torch.Tensor]) -> dict:
    """What the collector writes, as the offline builder would read it back."""
    return {
        "vision_0": raw["vision_0"].numpy(),
        "vision_1": raw["vision_1"].numpy(),
        "vision_2": raw["vision_2"].numpy(),
        "prompt_emb": raw["prompt_emb"].numpy(),
        "robot_state": raw["robot_state"].numpy(),
    }


def test_online_and_offline_builders_agree_bit_for_bit(runner, stub_model):
    stage1 = _online_stage1(runner, stub_model, prompt_tokens=7)

    online = GrootCP1MeanPoolKeyBuilder()
    online.collect(CheckpointID.CP1, stage1=stage1)
    online_keys = online.build(CheckpointID.CP1)

    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    fake = _build_fake_stage1(_hdf5_like(raw))
    offline = CP1MeanPoolKeyBuilder()
    offline.collect(CheckpointID.CP1, stage1=fake)
    offline_keys = offline.build(CheckpointID.CP1)

    assert set(online_keys) == set(offline_keys)
    for field in online_keys:
        assert torch.equal(online_keys[field], offline_keys[field]), field


def test_all_three_cameras_reach_the_key(runner, stub_model):
    """Two-camera assumptions elsewhere would silently drop the wrist view."""
    stage1 = _online_stage1(runner, stub_model, prompt_tokens=4)
    builder = GrootCP1MeanPoolKeyBuilder()
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = builder.build(CheckpointID.CP1)
    assert {"vision_0", "vision_1", "vision_2"} <= set(keys)
    assert not torch.equal(keys["vision_0"], keys["vision_2"])


def test_offline_reassembly_preserves_the_run_boundaries(runner, stub_model):
    stage1 = _online_stage1(runner, stub_model, prompt_tokens=11)
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
    )
    fake = _build_fake_stage1(_hdf5_like(raw))
    prefix = fake.prefix_embs[0]
    for index in range(3):
        block = prefix[index * TOKENS_PER_IMAGE : (index + 1) * TOKENS_PER_IMAGE]
        assert torch.equal(block, raw[f"vision_{index}"])
    assert prefix.shape[1] == EMB_DIM
