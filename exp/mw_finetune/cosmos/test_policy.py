"""CPU regression checks for the robot mapping and latent-conditioning layout."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.mw_finetune.cosmos.policy import (
    Options, expand_proprio, latent_layout, load_dit_weights, make_data_batch, project_actions,
    select_prompt, unpack_observation, validate_stats,
)


def test_opposed_gripper_and_identity_orientation():
    assert np.allclose(expand_proprio([.1, .6, .2, .7]), [.7, -.7, .1, .6, .2, 0, 0, 0, 1])
    with pytest.raises(ValueError):
        expand_proprio([.1, .6, np.nan, .7])


def test_action_projection_preserves_native_commands():
    source = np.asarray([[12, -15, 23, .3, .4, .5, -.2]], np.float32)
    output = project_actions(source)
    np.testing.assert_array_equal(output, np.asarray([[12, -15, 23, -.2]], np.float32))
    assert output.dtype == np.float32 and output.flags.c_contiguous
    with pytest.raises(ValueError):
        project_actions([[1, 2, 3, 4]])


def test_canonical_and_wire_images_have_same_mapping():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    state = [.1, .6, .2, .7]
    a = unpack_observation({"image": image, "state": state}, Options(image_rotation=180))
    b = unpack_observation({"observation/image": image, "observation/state": state}, Options(image_rotation=180))
    np.testing.assert_array_equal(a[0], image[::-1, ::-1])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_task_specific_push_back_cache_label_and_unknown_prompt():
    prompts = {"Push a puck back to a target behind it": None, "Push the puck to a goal": None}
    assert select_prompt({"task_name": "push-back-v3", "prompt": "Push the puck to a goal"}, prompts).endswith("behind it")
    with pytest.raises(ValueError, match="absent"):
        select_prompt({"prompt": "invented prompt"}, prompts)


@pytest.mark.parametrize("layout,n", [("single", 1), ("libero_duplicate", 2), ("robocasa_duplicate", 3)])
def test_batch_camera_order_raw_frames_and_normalization(layout, n):
    options = Options(camera_layout=layout, trained_with_image_aug=False, use_jpeg_compression=False)
    # Give each camera a different color to verify slot ordering independently
    # of the deliberate image duplication used at the wrapper boundary.
    cameras = [np.full((224, 224, 3), i + 10, np.uint8) for i in range(n)]
    stats = {"proprio_min": np.full(9, -1), "proprio_max": np.full(9, 1)}
    helpers = SimpleNamespace(prepare_images_for_model=lambda images, _: np.stack(cameras),
                             rescale_proprio=lambda p, s, **kw: 2 * (p - s["proprio_min"]) / (s["proprio_max"] - s["proprio_min"]) - 1)
    batch, action = make_data_batch(cameras[0], expand_proprio([.1, .6, .2, .7]),
                                    torch.zeros(1, 512, 1024), options, stats, "cpu", helpers, torch)
    idx = latent_layout(n)
    assert batch["video"].shape == (1, 3, 4 * (2*n+4) + 1, 224, 224)
    assert batch["num_conditional_frames"] == n + 2
    assert action == n + 2 and batch["action_latent_idx"].item() == action
    assert batch["value_latent_idx"].item() == 2*n + 4
    for phase in ("current", "future"):
        for i in range(n):
            k = idx[f"{phase}_camera_{i}"]
            assert torch.all(batch["video"][:, :, 1 + 4*(k-1):1 + 4*k] == i + 10)
    # Identity quaternion w stays 1 in raw proprio and is not clipped by mapping.
    assert batch["proprio"][0, -1].item() == 1
    assert batch["current_wrist_image_latent_idx"].item() == (2 if n > 1 else -1)
    assert batch["current_image_latent_idx"].item() == (2 if n == 1 else 3)
    assert batch["current_wrist_image2_latent_idx"].item() == -1


def test_stats_reject_degenerate_ranges_and_wrong_dimensions():
    s = {"actions_min": [-1]*7, "actions_max": [1]*7, "proprio_min": [-1]*9, "proprio_max": [1]*9}
    validate_stats(s)
    s["proprio_max"][0] = -1
    with pytest.raises(ValueError):
        validate_stats(s)


def test_dit_load_checks_parameters_and_ignores_fp8_bookkeeping():
    net = torch.nn.Linear(4, 2)
    state = {"net.weight": torch.ones(2, 4), "net.bias": torch.ones(2), "net.fp8._extra_state": torch.zeros(5)}
    audit = load_dit_weights(net, state)
    assert not audit["missing_parameters"] and audit["ignored_fp8_extra_states"] == 1
    assert torch.equal(net.weight, state["net.weight"])
    with pytest.raises(ValueError, match="missing=.*bias"):
        load_dit_weights(net, {"net.weight": state["net.weight"]})
    with pytest.raises(ValueError, match="unexpected=.*other"):
        load_dit_weights(net, {**state, "net.other": torch.ones(1)})
