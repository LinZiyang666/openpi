"""Real-model checks for the two-stage split. Island B only.

Everything else in this suite runs against a stub, which can show that the
split calls the right things in the right order but cannot show that the six
statements copied out of upstream's forward still reproduce it. That is what
these tests are for, and they need the actual 7.2 GB checkpoint.

Run (the PYTHONPATH entry is not optional -- `gr00t` is a worktree, not an
installed package, and without it `importorskip` turns this file into a silent
skip that reads as a pass)::

    cd /home/weiland/openpi
    PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python -m pytest \\
      tests/robocasa365/test_groot_cache_manual.py --run-manual -v

Negative controls are first-class here. `max|delta| == 0` between two paths
proves nothing on its own: if the flow-matching noise were accidentally pinned,
or if both paths shared a cached result, the equality would hold for the wrong
reason. So each equality assertion is paired with a condition under which the
difference must be non-zero.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual

CHECKPOINT = pathlib.Path(
    "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
EMBODIMENT_TAG = "new_embodiment"
SEED = 12345


@pytest.fixture(scope="module")
def policy():
    pytest.importorskip("gr00t", reason="requires the GR00T island")
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present: {CHECKPOINT}")

    from gr00t.model.policy import Gr00tPolicy

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    data_config = RoboCasa365DataConfig()
    return Gr00tPolicy(
        model_path=str(CHECKPOINT),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device="cuda",
    )


@pytest.fixture(scope="module")
def runner(policy):
    from openpi.cache.groot.staged import GrootStagedRunner

    # verify_upstream left on: the pinned hash is part of what is under test.
    return GrootStagedRunner(policy.model)


def _observation(step: int = 0) -> dict:
    """A legal observation; `step` varies it so successive keys differ."""
    from exp.robocasa365 import groot_keys

    import json

    stats = json.loads((CHECKPOINT / "experiment_cfg" / "metadata.json").read_text())
    state_stats = stats[EMBODIMENT_TAG]["statistics"]["state"]

    obs: dict = {}
    rng = np.random.default_rng(1000 + step)
    resolution = groot_keys.MODEL_IMAGE_RESOLUTION
    for key in groot_keys.VIDEO_KEYS:
        obs[key] = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)

    for key in groot_keys.STATE_KEYS:
        vector = np.asarray(
            state_stats[key.removeprefix("state.")]["mean"], dtype=np.float64
        )
        if key in groot_keys.QUATERNION_STATE_KEYS:
            norm = float(np.linalg.norm(vector))
            vector = (
                np.asarray(groot_keys.IDENTITY_QUATERNION_WXYZ, dtype=np.float64)
                if norm < 1e-8
                else vector / norm
            )
        else:
            vector = vector + 0.01 * step
        obs[key] = vector

    for key in groot_keys.LANGUAGE_KEYS:
        obs[key] = "pick up the object"
    return obs


def _normalized(policy, obs):
    """Wire-format obs -> normalized model input, via the PRODUCTION reshaping.

    ``_observation`` builds wire-format values (video ``(H, W, 3)``, state
    ``(D,)``); upstream ``apply_transforms`` consumes batched ``(B, T, ...)``.
    The T axis is added by the adapter's ``build_groot_observation`` and the B
    axis by the interceptor's unsqueeze — using both here means the test eats
    the same reshaping code the server runs, instead of a hand-rolled copy
    that can drift (the first real-machine run caught exactly that: a missing
    T axis sent PIL frames into ``prepare_input``).
    """
    from exp.robocasa365.groot_policy_adapter import build_groot_observation
    from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values

    shaped = build_groot_observation(obs)  # wire -> [T=1, ...] (production contract)
    if not _is_batched(shaped):
        shaped = _unsqueeze_values(shaped)  # add the batch axis, as the interceptor does
    for key, value in shaped.items():
        if not isinstance(value, np.ndarray):
            shaped[key] = np.array(value)
    return policy.apply_transforms(shaped)


def _warm_up(policy, inputs):
    """Discard one full call so cudnn's algorithm choice is already cached."""
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        policy.model.get_action(inputs)


# ------------------------------------------------------------------
# G0-C: the split reproduces the unsplit forward
# ------------------------------------------------------------------


def test_two_stage_split_is_bit_exact(policy, runner):
    inputs = _normalized(policy, _observation())
    _warm_up(policy, inputs)

    torch.manual_seed(SEED)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        reference = policy.model.get_action(inputs)["action_pred"].float().cpu()

    torch.manual_seed(SEED)
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        split = runner.run_stage2(stage1).action_pred.float().cpu()

    assert torch.equal(split, reference), (
        f"max|delta| = {(split - reference).abs().max().item()}"
    )


def test_unseeded_calls_differ(policy):
    """The control for the test above: without it, equality could be an artefact."""
    inputs = _normalized(policy, _observation())
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        first = policy.model.get_action(inputs)["action_pred"].float().cpu()
        second = policy.model.get_action(inputs)["action_pred"].float().cpu()
    assert not torch.equal(first, second), (
        "two unseeded flow-matching calls produced identical actions; the noise "
        "is pinned somewhere and the equality test above proves nothing"
    )


def test_backbone_features_match_the_upstream_backbone(policy, runner):
    """Our language-model call must land on the same tensor upstream selects."""
    inputs = _normalized(policy, _observation())
    captured = {}
    original = policy.model.action_head.get_action

    def _capture(backbone_outputs, action_inputs):
        captured["features"] = backbone_outputs["backbone_features"].float().cpu().clone()
        return original(backbone_outputs, action_inputs)

    policy.model.action_head.get_action = _capture
    try:
        with runner.session():
            runner.run_stage2(runner.run_stage1(inputs))
    finally:
        policy.model.action_head.get_action = original

    backbone_inputs, _ = policy.model.prepare_input(inputs)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        reference = policy.model.backbone(backbone_inputs)["backbone_features"]
    reference = reference.float().cpu()

    assert torch.equal(captured["features"], reference), (
        f"max|delta| = {(captured['features'] - reference).abs().max().item()}"
    )


def test_running_stage2_twice_on_one_stage1_is_reproducible(policy, runner):
    """Pins that the action head's in-place vlln is not applied to a reused mapping."""
    inputs = _normalized(policy, _observation())
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        torch.manual_seed(SEED)
        first = runner.run_stage2(stage1).action_pred.float().cpu()
        torch.manual_seed(SEED)
        second = runner.run_stage2(stage1).action_pred.float().cpu()
    assert torch.equal(first, second)


def test_running_outside_autocast_changes_the_numbers(policy, runner):
    """The control for the session contract: if this passed, the guard would be pointless."""
    inputs = _normalized(policy, _observation())

    torch.manual_seed(SEED)
    with runner.session():
        inside = runner.run_stage2(runner.run_stage1(inputs)).action_pred.float().cpu()

    # Bypass the guard deliberately to measure what it is protecting against.
    torch.manual_seed(SEED)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        stage1 = runner.run_stage1(inputs)
    torch.manual_seed(SEED)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(inputs)
        del backbone_inputs
        outputs = policy.model.backbone.eagle_model.language_model(
            inputs_embeds=stage1.input_embeds,
            attention_mask=stage1.attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
        )
        features = outputs.hidden_states[policy.model.backbone.select_layer]
        features = policy.model.backbone.eagle_linear(features)
        from transformers.feature_extraction_utils import BatchFeature

        no_autocast = policy.model.action_head.get_action(
            BatchFeature(
                data={
                    "backbone_features": features,
                    "backbone_attention_mask": stage1.attention_mask,
                }
            ),
            action_inputs,
        )["action_pred"].float().cpu()

    assert not torch.equal(inside, no_autocast), (
        "autocast made no difference; LayerNorm is no longer being promoted to "
        "fp32 and the session contract has lost its reason to exist"
    )


def test_stage1_guard_refuses_to_run_without_a_session(policy, runner):
    inputs = _normalized(policy, _observation())
    with pytest.raises(RuntimeError, match="must run inside"):
        runner.run_stage1(inputs)


# ------------------------------------------------------------------
# Real token layout
# ------------------------------------------------------------------


def test_image_runs_are_three_blocks_of_256_at_moving_offsets(policy, runner):
    from openpi.cache.groot.key_builder import _contiguous_runs

    short = _normalized(policy, _observation())
    obs_long = _observation()
    for key in ("annotation.human.task_description",):
        obs_long[key] = "pick up the object and put it on the counter please"
    long = _normalized(policy, obs_long)

    with runner.session():
        s_short = runner.run_stage1(short)
        s_long = runner.run_stage1(long)

    for stage1 in (s_short, s_long):
        runs = _contiguous_runs(stage1.image_token_mask[0])
        assert len(runs) == 3
        assert all(length == 256 for _, length in runs)

    short_starts = [start for start, _ in _contiguous_runs(s_short.image_token_mask[0])]
    long_starts = [start for start, _ in _contiguous_runs(s_long.image_token_mask[0])]
    assert short_starts != long_starts, (
        "image-token offsets did not move with prompt length; the mask-driven "
        "slicing would be indistinguishable from a fixed offset table here, so "
        "this test could not detect the bug it exists for"
    )


def test_scattered_positions_hold_the_vision_output(policy, runner):
    inputs = _normalized(policy, _observation())
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        backbone_inputs, _ = policy.model.prepare_input(inputs)
        vit = policy.model.backbone.eagle_model.extract_feature(
            backbone_inputs["eagle_pixel_values"]
        )
    scattered = stage1.input_embeds[0][stage1.image_token_mask[0]]
    assert torch.equal(scattered, vit.reshape(-1, vit.shape[-1]))


# ------------------------------------------------------------------
# G0-D2: online keys still retrieve their own step from the offline library
# ------------------------------------------------------------------


def test_online_and_offline_keys_retrieve_the_same_entry(policy, runner, tmp_path):
    from exp.common.build_in_memory_cache_artifact import build_artifact
    from exp.robocasa365.groot_cache_collector import GrootCacheCollector
    from exp.robocasa365.groot_key_parity import check_key_parity
    from openpi.cache.groot.key_builder import GrootCP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID

    n_steps = 6
    data_dir = tmp_path / "episodes"
    collector = GrootCacheCollector(policy, runner, out_dir=str(data_dir))
    collector.on_episode_start(task="ManualParity", episode_id=0)

    online_keys = []
    builder = GrootCP1SpatialPool16KeyBuilder()
    for step in range(n_steps):
        obs = _observation(step)
        collector.get_action(obs)
        inputs = _normalized(policy, obs)
        with runner.session():
            stage1 = runner.run_stage1(inputs)
        builder.collect(CheckpointID.CP1, stage1=stage1)
        online_keys.append(builder.build(CheckpointID.CP1))
    collector.on_episode_end(success=True)

    artifact = build_artifact(
        str(data_dir), "cp1_groot_spatial_pool_16", "CP1", workers=-1
    )
    entries = sorted(artifact["entries"], key=lambda e: e.step_idx)
    assert len(entries) == n_steps
    offline_keys = [
        {k: torch.as_tensor(v).float() for k, v in entry.query_keys.items()}
        for entry in entries
    ]

    metrics = {
        "vision_0": "cosine",
        "vision_1": "cosine",
        "vision_2": "cosine",
        "prompt_emb": "cosine",
        "robot_state": "l2",
    }
    report = check_key_parity(online_keys, offline_keys, metrics)
    print("\n" + report.summary())
    assert report.passed, report.summary()
