"""Non-manual tests for the ``pi05_robocasa`` inference config registration (T2).

Pins the three failure modes that would otherwise only surface at server
startup or — worse — silently:

* importing the config registry (or the policy module) must NOT pull the
  ``robocasa`` package into the main venv (it only exists in the py3.12 sim
  island; a top-level import would break every ``get_config`` caller);
* ``asset_id`` must be ``"robocasa"`` — without it ``create_trained_policy``
  raises "Asset id is required to load norm stats" because
  ``scripts/serve_policy.py`` never passes ``norm_stats`` explicitly;
* ``use_quantile_norm`` must be False — the checkpoint's norm_stats carry
  q01/q99 = null and quantile normalization raises inside
  ``transforms.Normalize``.

Also pins the transform-chain shape against the rescued
``robocasa_policy`` module so the registered config builds the same stack as
the archived ``serve_robocasa_pi05_ORIGINAL.py`` (structural half of T2-a; the
numeric half needs the checkpoint and lives in
``test_pi05_stack_parity_manual.py``).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import robocasa_policy
from openpi.training.config import get_config


def test_config_registered_without_robocasa_import():
    cfg = get_config("pi05_robocasa")
    loaded = sorted(m for m in sys.modules if m == "robocasa" or m.startswith("robocasa."))
    assert not loaded, f"registering pi05_robocasa pulled robocasa into the main venv: {loaded}"
    assert cfg.model.pi05 is True
    assert cfg.model.max_token_len == 200
    # Checkpoint-matching defaults (pi05_pretrain_human300 training shape).
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.model.discrete_state_input is True


def test_data_config_asset_id_and_quantile_norm():
    cfg = get_config("pi05_robocasa")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    # Load-bearing: create_policy() does not pass norm_stats, so resolution
    # goes through <ckpt>/assets/<asset_id>; None would raise at startup.
    assert dc.asset_id == "robocasa"
    # Load-bearing: the checkpoint's norm_stats have q01/q99 = null.
    assert dc.use_quantile_norm is False


def test_data_transform_chain_shape():
    cfg = get_config("pi05_robocasa")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert [type(t) for t in dc.data_transforms.inputs] == [robocasa_policy.RobocasaInputs]
    assert [type(t) for t in dc.data_transforms.outputs] == [robocasa_policy.RobocasaOutputs]
    inp = dc.data_transforms.inputs[0]
    assert inp.action_dim == 32
    assert inp.model_type == _model.ModelType.PI05


def test_outputs_slice_to_12_dims():
    out = robocasa_policy.RobocasaOutputs()({"actions": np.zeros((50, 32))})
    assert out["actions"].shape == (50, 12)


def test_pi05_inputs_require_right_image():
    inputs = robocasa_policy.RobocasaInputs(action_dim=32, model_type=_model.ModelType.PI05)
    data = {
        "observation/state": np.zeros(16),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "x",
    }
    with pytest.raises(ValueError, match="right_image"):
        inputs(data)


def test_pi05_inputs_full_contract():
    inputs = robocasa_policy.RobocasaInputs(action_dim=32, model_type=_model.ModelType.PI05)
    data = {
        "observation/state": np.arange(16, dtype=np.float64),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "turn on the sink faucet",
    }
    out = inputs(data)
    assert out["state"].shape == (32,)  # padded 16 -> action_dim
    assert np.array_equal(out["state"][:16], np.arange(16))
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is True  # pi05: real camera, not padding
    assert out["prompt"] == "turn on the sink faucet"
