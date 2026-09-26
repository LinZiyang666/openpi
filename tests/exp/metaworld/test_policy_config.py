"""``pi05_metaworld`` inference config and ``metaworld_policy`` transforms.

Pins the checkpoint-matching model shape (action horizon 5, no discrete state,
max_token_len 200), the asset id the server resolves norm stats from, quantile
normalization, the single-camera image layout and the 4-dim action slice, and
runs the full policy input chain on synthetic quantile stats.
"""

from __future__ import annotations

import sys

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import metaworld_policy
from openpi.shared import normalize as _normalize
from openpi.training.config import get_config


def test_config_model_shape_without_simulator_import():
    cfg = get_config("pi05_metaworld")
    assert not [
        m for m in sys.modules if m == "metaworld" or m.startswith("metaworld.")
    ]
    assert cfg.model.model_type == _model.ModelType.PI05
    assert (cfg.model.action_horizon, cfg.model.action_dim) == (5, 32)
    assert cfg.model.discrete_state_input is False
    assert cfg.model.max_token_len == 200


def test_data_config_asset_id_quantile_and_chain(tmp_path):
    cfg = get_config("pi05_metaworld")
    dc = cfg.data.create(tmp_path, cfg.model)
    assert dc.asset_id == "metaworld_mt50"
    assert dc.use_quantile_norm is True
    assert [type(t) for t in dc.data_transforms.inputs] == [
        metaworld_policy.MetaworldInputs
    ]
    assert [type(t) for t in dc.data_transforms.outputs] == [
        metaworld_policy.MetaworldOutputs
    ]


def test_inputs_single_camera_layout():
    image = np.random.randint(256, size=(480, 480, 3), dtype=np.uint8)
    out = metaworld_policy.MetaworldInputs()(
        {
            "observation/image": image,
            "observation/state": np.arange(4, dtype=np.float64),
            "prompt": "Press a button",
        }
    )
    assert set(out) == {"state", "image", "image_mask", "prompt"}
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], image)
    for slot in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert out["image"][slot].shape == image.shape and not out["image"][slot].any()
    assert {k: bool(v) for k, v in out["image_mask"].items()} == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert out["state"].dtype == np.float32 and out["state"].tolist() == [0, 1, 2, 3]
    assert out["prompt"] == "Press a button"


def test_inputs_parse_float_chw_image():
    chw = np.full((3, 8, 8), 0.5, dtype=np.float32)
    out = metaworld_policy.MetaworldInputs()(
        {"observation/image": chw, "observation/state": np.zeros(4)}
    )
    assert out["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert out["image"]["base_0_rgb"].dtype == np.uint8
    assert "prompt" not in out


def test_outputs_slice_four_dims():
    actions = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    out = metaworld_policy.MetaworldOutputs()({"actions": actions})
    np.testing.assert_array_equal(out["actions"], actions[:, :4])


def test_full_input_chain_on_quantile_stats(tmp_path):
    cfg = get_config("pi05_metaworld")
    stats = _normalize.NormStats(
        mean=np.zeros(4), std=np.ones(4), q01=-np.ones(4), q99=np.ones(4)
    )
    _normalize.save(tmp_path / "metaworld_mt50", {"state": stats, "actions": stats})
    dc = cfg.data.create(tmp_path, cfg.model)
    assert dc.norm_stats is not None
    chain = _transforms.compose(
        [
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )
    example = metaworld_policy.make_metaworld_example()
    example["observation/state"] = np.full(4, 0.5, dtype=np.float32)
    out = chain(example)
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert out["tokenized_prompt"].shape == (200,)
    assert out["state"].shape == (32,)
    # quantile map of 0.5 on [-1, 1] -> 0.5, then zero padding
    np.testing.assert_allclose(out["state"][:4], 0.5, atol=1e-5)
    assert not out["state"][4:].any()


def test_example_is_a_valid_request():
    example = metaworld_policy.make_metaworld_example()
    assert example["observation/image"].shape == (480, 480, 3)
    assert example["observation/state"].shape == (4,)
