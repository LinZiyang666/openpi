"""``exp/libero_groot/cp2_reconstruct``: template + H5 step -> stage-1 output, fail-closed.

Off-island with a two-camera stub GR00T model (the LIBERO shape: two image
runs of 256 tokens after / between the chat-template text). The real-model
version of the same round trip is the island parity gate.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from exp.libero_groot import libero_keys as K
from exp.libero_groot.cp2_reconstruct import (
    LIBERO_VISION_FIELDS,
    REQUIRED_STEP_FIELDS,
    STAGE1_PATH,
    TemplateCache,
    build_template,
    dummy_wire_observation,
    h5_task,
    normalized_input_for,
    reconstruct_stage1,
)
from openpi.cache.groot.key_builder import slice_groot_cp1_fields
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE
from tests.cache.groot.conftest import EMB_DIM, STATE_VALID, TOKENS_PER_IMAGE, StubGrootModel
from tests.libero_groot.conftest import N_CAMERAS

def _group(h5, name, template, *, vision=None, prompt=None, state=None, drop=()):
    g = h5.create_group(name)
    embeds = template.input_embeds
    for i, field in enumerate(LIBERO_VISION_FIELDS):
        if field in drop:
            continue
        v = vision[i] if vision is not None else torch.randn(TOKENS_PER_IMAGE, EMB_DIM)
        g.create_dataset(field, data=v.to(torch.float16).numpy())
    if PROMPT_EMB not in drop:
        text = embeds[0][~template.image_token_mask[0]].float().to(torch.float16).numpy() if prompt is None else prompt
        g.create_dataset(PROMPT_EMB, data=text)
    if ROBOT_STATE not in drop:
        g.create_dataset(ROBOT_STATE, data=(np.linspace(0.1, 0.5, STATE_VALID).astype(np.float32) if state is None else state))
    return g


def test_dummy_observation_and_normalisation_go_through_the_serving_shaping(groot_harness):
    _, policy, _ = groot_harness
    obs = dummy_wire_observation("pick up the bowl")
    assert set(obs) == set(K.wire_observation_keys())
    assert obs[K.WIRE_IMAGE].shape == (K.WIRE_IMAGE_RESOLUTION, K.WIRE_IMAGE_RESOLUTION, 3)
    normalized_input_for(policy, obs)
    seen = policy.calls[-1]
    assert seen["video.image"].shape == (1, 1, K.WIRE_IMAGE_RESOLUTION, K.WIRE_IMAGE_RESOLUTION, 3)  # batch + time
    assert str(np.asarray(seen[K.LANGUAGE_KEY]).reshape(-1)[0]) == "pick up the bowl"


def test_template_has_two_full_image_runs_and_is_built_once_per_task(groot_harness):
    _, policy, runner = groot_harness
    with runner.session():
        cache = TemplateCache(policy, runner)
        t1 = cache.get("open the drawer")
        t2 = cache.get("open the drawer")
        t3 = cache.get("put the bowl on the plate please")
    assert t1 is t2 and len(cache) == 2 and len(policy.calls) == 2
    assert [n for _, n in t1.runs] == [TOKENS_PER_IMAGE] * N_CAMERAS
    assert t1.n_tokens != t3.n_tokens  # prompt length follows the instruction
    n_prompt = 3 + 3
    assert t1.runs == ((n_prompt, TOKENS_PER_IMAGE), (n_prompt + TOKENS_PER_IMAGE + n_prompt, TOKENS_PER_IMAGE))


def test_template_refuses_a_three_camera_sequence(groot_harness):
    model, policy, runner = groot_harness
    model.build_inputs = lambda prompt_tokens=None: StubGrootModel.build_inputs(model, prompt_tokens)  # 3 cameras
    with runner.session(), pytest.raises(RuntimeError, match="expected 2 image runs"):
        build_template(policy, runner, "task")


def test_reconstruction_overwrites_runs_and_state_and_leaves_the_template_alone(groot_harness):
    _, policy, runner = groot_harness
    with runner.session():
        template = build_template(policy, runner, "open the drawer")
        before = template.input_embeds.clone()
        state_before = template.action_inputs["state"].clone()
        vision = [torch.randn(TOKENS_PER_IMAGE, EMB_DIM), torch.randn(TOKENS_PER_IMAGE, EMB_DIM)]
        with h5py.File("m.h5", "w", driver="core", backing_store=False) as h5:
            g = _group(h5, "step_0000", template, vision=vision)
            out = reconstruct_stage1(template, g)
            stored_state = np.asarray(g[ROBOT_STATE])
        for (start, length), v in zip(template.runs, vision, strict=True):
            assert torch.equal(out.input_embeds[0, start : start + length].float(), v.to(torch.float16).float())
        text = ~template.image_token_mask[0]
        assert torch.equal(out.input_embeds[0][text], template.input_embeds[0][text])
        valid = out.state_mask[0, -1]
        assert np.array_equal(out.state[0, -1][valid].float().numpy(), stored_state)
        assert not out.state[0, -1][~valid].any()
        assert out.attention_mask is template.attention_mask and out.image_token_mask is template.image_token_mask
        # The template is a skeleton for every step: untouched.
        assert torch.equal(template.input_embeds, before)
        assert torch.equal(template.action_inputs["state"], state_before)
        assert out.action_inputs is not template.action_inputs


def test_collector_slicing_round_trips_bit_exactly(groot_harness):
    """Path B of the parity gate, offline: slice a real stage-1 output the way the
    collector does, store it fp16, rebuild -> identical sequence and state."""
    _, policy, runner = groot_harness
    with runner.session():
        inputs = policy.model.build_inputs(prompt_tokens=3 + 3)
        inputs["state"][0, 0, :STATE_VALID] = torch.tensor([0.25, -0.5, 0.125, 1.0, -2.0])
        stage1 = runner.run_stage1(inputs)
        raw = slice_groot_cp1_fields(stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask,
                                     None, vision_fields=LIBERO_VISION_FIELDS)
        template = build_template(policy, runner, "open the drawer")
        with h5py.File("m.h5", "w", driver="core", backing_store=False) as h5:
            g = h5.create_group("step_0000")
            for field in (*LIBERO_VISION_FIELDS, PROMPT_EMB):
                g.create_dataset(field, data=raw[field].to(torch.float16).numpy())
            g.create_dataset(ROBOT_STATE, data=raw[ROBOT_STATE].float().numpy())
            out = reconstruct_stage1(template, g)
        assert torch.equal(out.input_embeds.to(torch.float16), stage1.input_embeds.to(torch.float16))
        assert torch.equal(out.state, stage1.state)
        assert torch.equal(out.state_mask, stage1.state_mask)


@pytest.mark.parametrize("missing", REQUIRED_STEP_FIELDS)
def test_missing_fields_raise(groot_harness, missing):
    _, policy, runner = groot_harness
    with runner.session():
        template = build_template(policy, runner, "open the drawer")
        with h5py.File("m.h5", "w", driver="core", backing_store=False) as h5:
            g = _group(h5, "step_0000", template, drop=(missing,))
            with pytest.raises(RuntimeError, match=f"lacks '{missing}'"):
                reconstruct_stage1(template, g)


def test_text_drift_wrong_run_shape_and_state_count_are_refused(groot_harness):
    _, policy, runner = groot_harness
    with runner.session():
        template = build_template(policy, runner, "open the drawer")
        good_text = template.input_embeds[0][~template.image_token_mask[0]].float().to(torch.float16).numpy()
        with h5py.File("m.h5", "w", driver="core", backing_store=False) as h5:
            drift = good_text.copy()
            drift[0, 0] += 1.0
            g = _group(h5, "drift", template, prompt=drift)
            with pytest.raises(RuntimeError, match="differ from stored prompt_emb"):
                reconstruct_stage1(template, g)
            g = _group(h5, "longer", template, prompt=np.concatenate([good_text, good_text[:1]]))
            with pytest.raises(RuntimeError, match="differ from stored prompt_emb"):
                reconstruct_stage1(template, g)
            g = _group(h5, "shape", template, vision=[torch.randn(TOKENS_PER_IMAGE - 1, EMB_DIM), torch.randn(TOKENS_PER_IMAGE, EMB_DIM)])
            with pytest.raises(RuntimeError, match="stored shape"):
                reconstruct_stage1(template, g)
            g = _group(h5, "state", template, state=np.zeros(STATE_VALID - 1, dtype=np.float32))
            with pytest.raises(RuntimeError, match="valid slots"):
                reconstruct_stage1(template, g)


def test_h5_task_and_stage1_path_identity():
    with h5py.File("m.h5", "w", driver="core", backing_store=False) as h5:
        with pytest.raises(RuntimeError, match="no 'task' attribute"):
            h5_task(h5)
        h5.attrs["task"] = "open the drawer"
        assert h5_task(h5) == "open the drawer"
    assert STAGE1_PATH == "groot_reconstructed_template"
