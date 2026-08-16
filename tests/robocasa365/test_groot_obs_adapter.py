"""Non-manual tests for the RoboCasa365 x GR00T N1.5 observation adapter.

These run in the main test environment: nothing here imports ``gr00t``, ``torch``
or the simulator, and the policy is injected as a stub.  The GR00T-dependent
checks live in ``test_groot_data_config_manual.py``.

The expected key lists are hard-coded rather than read back from the module under
test, so a silent reordering fails loudly instead of agreeing with itself.  They
are transcribed from the official RoboCasa365 ``new_embodiment`` configuration
(robocasa @ be22d659, docs/datasets/using_datasets.md:167-210).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from exp.robocasa365 import groot_keys
from exp.robocasa365.groot_policy_adapter import (
    GrootPolicyAdapter,
    build_groot_observation,
    iter_step_actions,
    take_action_chunk,
    validate_action_chunk,
)

OFFICIAL_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]
OFFICIAL_STATE_KEYS = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
]
OFFICIAL_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]
# Transcription of the keys robocasa.utils.env_utils.convert_action produces.
# NOTE: comparing our constants against this is self-consistent by construction --
# it pins our own list but cannot detect robocasa changing env.step's contract.
# That binding lives in test_env_action_contract_manual.py, which calls the real
# convert_action inside the simulation island.
ENV_ACTION_KEYS = set(OFFICIAL_ACTION_KEYS)

OFFICIAL_STATE_DIMS = {
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
}
OFFICIAL_ACTION_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
OFFICIAL_QUATERNION_STATE_KEYS = [
    "state.end_effector_rotation_relative",
    "state.base_rotation",
]

N17_LANGUAGE_KEY = "annotation.human.action.task_description"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _StubPolicy:
    """Stand-in for Gr00tPolicy that returns a canned, well-formed action chunk."""

    def __init__(self, bad_key: str | None = None) -> None:
        self.bad_key = bad_key
        self.seen: dict[str, Any] | None = None

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        self.seen = observations
        out: dict[str, Any] = {
            key: np.zeros((groot_keys.ACTION_HORIZON, dim))
            for key, dim in groot_keys.ACTION_DIMS.items()
        }
        if self.bad_key is not None:
            width = groot_keys.ACTION_DIMS[self.bad_key]
            out[self.bad_key] = np.full((groot_keys.ACTION_HORIZON, width), np.nan)
        return out


def _valid_obs() -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for key in OFFICIAL_VIDEO_KEYS:
        obs[key] = np.zeros((256, 256, 3), dtype=np.uint8)
    for key in OFFICIAL_STATE_KEYS:
        obs[key] = np.zeros(groot_keys.STATE_DIMS[key])
    for key in groot_keys.QUATERNION_STATE_KEYS:
        obs[key] = np.asarray(groot_keys.IDENTITY_QUATERNION_WXYZ)
    obs["annotation.human.task_description"] = "open the drawer"
    return obs


# ------------------------------------------------------------------
# Key contract
# ------------------------------------------------------------------


def test_video_key_order_matches_official_config():
    assert groot_keys.VIDEO_KEYS == OFFICIAL_VIDEO_KEYS


def test_state_key_order_matches_official_config():
    assert groot_keys.STATE_KEYS == OFFICIAL_STATE_KEYS


def test_action_key_order_matches_official_config():
    assert groot_keys.ACTION_KEYS == OFFICIAL_ACTION_KEYS


def test_language_key_is_env_native_not_n17_variant():
    # N1.5 `new_embodiment` uses the env-native key. The parent DataConfig
    # declares the N1.7 ROBOCASA_PANDA variant, which is the easy mistake here.
    assert groot_keys.LANGUAGE_KEYS == ["annotation.human.task_description"]
    assert N17_LANGUAGE_KEY not in groot_keys.LANGUAGE_KEYS


def test_action_keys_match_transcribed_env_contract():
    # Self-consistent check; the binding to the real convert_action is manual.
    assert set(groot_keys.ACTION_KEYS) == ENV_ACTION_KEYS


def test_wire_keys_are_not_renamed():
    wire = groot_keys.wire_observation_keys()
    assert wire == OFFICIAL_VIDEO_KEYS + OFFICIAL_STATE_KEYS + [
        "annotation.human.task_description"
    ]


def test_action_horizon_is_16():
    assert groot_keys.ACTION_HORIZON == 16


# ------------------------------------------------------------------
# Observation translation
# ------------------------------------------------------------------


def test_build_obs_adds_time_axis():
    out = build_groot_observation(_valid_obs())
    for key in OFFICIAL_VIDEO_KEYS:
        assert out[key].shape == (1, 256, 256, 3)
    for key in OFFICIAL_STATE_KEYS:
        assert out[key].shape == (1, groot_keys.STATE_DIMS[key])
    assert out["annotation.human.task_description"].shape == (1,)


def test_build_obs_rejects_missing_key():
    obs = _valid_obs()
    del obs["state.gripper_qpos"]
    with pytest.raises(ValueError, match="state.gripper_qpos"):
        build_groot_observation(obs)


def test_build_obs_rejects_wrong_dtype():
    obs = _valid_obs()
    obs["video.robot0_eye_in_hand"] = np.zeros((256, 256, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        build_groot_observation(obs)


def test_build_obs_rejects_wrong_shape():
    obs = _valid_obs()
    obs["state.base_position"] = np.zeros(7)
    with pytest.raises(ValueError, match="state.base_position"):
        build_groot_observation(obs)


def test_rejects_non_finite_input():
    obs = _valid_obs()
    obs["state.end_effector_position_relative"] = np.asarray([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        build_groot_observation(obs)


# ------------------------------------------------------------------
# Adapter behaviour
# ------------------------------------------------------------------


def test_infer_nests_actions_under_their_own_key():
    """Actions live under "actions", never at the envelope top level.

    The websocket server attaches its own diagnostic fields (server_timing) to
    the reply; nesting keeps those from ever being mistaken for action data.
    """
    result = GrootPolicyAdapter(_StubPolicy()).infer(_valid_obs())
    assert set(result) == {"actions"}
    assert set(result["actions"]) == ENV_ACTION_KEYS


def test_rejects_non_finite_output():
    adapter = GrootPolicyAdapter(_StubPolicy(bad_key="action.end_effector_position"))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.infer(_valid_obs())


def test_infer_rejects_incomplete_policy_output():
    class _Incomplete:
        def get_action(self, observations):
            return {"action.end_effector_position": np.zeros((16, 3))}

    with pytest.raises(ValueError, match="missing action keys"):
        GrootPolicyAdapter(_Incomplete()).infer(_valid_obs())


def test_adapter_propagates_policy_exception():
    class _Boom:
        def get_action(self, observations):
            raise RuntimeError("cuda oom")

    with pytest.raises(RuntimeError, match="cuda oom"):
        GrootPolicyAdapter(_Boom()).infer(_valid_obs())


# ------------------------------------------------------------------
# Action chunk slicing
# ------------------------------------------------------------------


def test_action_chunk_slicing_preserves_order():
    chunk = {
        "action.base_motion": np.arange(groot_keys.ACTION_HORIZON * 4).reshape(-1, 4)
    }
    taken = take_action_chunk(chunk, 5)
    assert taken["action.base_motion"].shape == (5, 4)
    np.testing.assert_array_equal(
        taken["action.base_motion"], chunk["action.base_motion"][:5]
    )


def test_replan_steps_must_not_exceed_horizon():
    chunk = {"action.base_motion": np.zeros((groot_keys.ACTION_HORIZON, 4))}
    with pytest.raises(ValueError, match="exceeds action horizon"):
        take_action_chunk(chunk, groot_keys.ACTION_HORIZON + 1)


# ------------------------------------------------------------------
# Start-up probe input
# ------------------------------------------------------------------


def test_dummy_obs_quaternions_are_unit_norm_not_zeros(tmp_path):
    """A zero quaternion is not a rotation and would poison rotation_6d.

    Guards the probe input itself: the obvious all-zeros smoke observation makes
    the handshake meaningless.  The probe takes its quaternions from the
    checkpoint's own means and renormalises them, so the values are both
    in-distribution and geometrically valid -- the mean of several quaternions
    is not itself unit-norm.
    """
    from exp.robocasa365.serve_groot_n15 import _dummy_observation

    stats = {
        "new_embodiment": {
            "statistics": {
                "state": {
                    key.removeprefix("state."): {
                        "mean": [0.1] * groot_keys.STATE_DIMS[key]
                    }
                    for key in OFFICIAL_STATE_KEYS
                }
            }
        }
    }
    cfg = tmp_path / "experiment_cfg"
    cfg.mkdir()
    (cfg / "metadata.json").write_text(json.dumps(stats))

    obs = _dummy_observation(tmp_path)
    assert len(groot_keys.QUATERNION_STATE_KEYS) == 2
    for key in groot_keys.QUATERNION_STATE_KEYS:
        assert np.isclose(np.linalg.norm(obs[key]), 1.0), f"{key} is not unit-norm"
        assert not np.allclose(obs[key], 0.0)
    for key in OFFICIAL_STATE_KEYS:
        assert np.isfinite(obs[key]).all()
    # And the probe must survive its own validation.
    build_groot_observation(obs)


def test_dummy_obs_falls_back_to_identity_for_degenerate_quaternion(tmp_path):
    """An all-zero mean cannot be normalised; fall back rather than divide by 0."""
    from exp.robocasa365.serve_groot_n15 import _dummy_observation

    stats = {
        "new_embodiment": {
            "statistics": {
                "state": {
                    key.removeprefix("state."): {
                        "mean": [0.0] * groot_keys.STATE_DIMS[key]
                    }
                    for key in OFFICIAL_STATE_KEYS
                }
            }
        }
    }
    cfg = tmp_path / "experiment_cfg"
    cfg.mkdir()
    (cfg / "metadata.json").write_text(json.dumps(stats))

    obs = _dummy_observation(tmp_path)
    for key in groot_keys.QUATERNION_STATE_KEYS:
        np.testing.assert_array_equal(obs[key], np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_identity_quaternion_is_wxyz_convention():
    # pytorch3d (used by GR00T's RotationTransform) puts the real part first.
    # Note this is identity in GR00T's frame: robosuite emits xyzw, so the same
    # tuple is a 180-degree X rotation in the simulator's own convention. It is
    # only a degenerate-case fallback for the probe, never a semantic identity.
    assert groot_keys.IDENTITY_QUATERNION_WXYZ == (1.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------
# Action chunk contract
# ------------------------------------------------------------------


def _good_chunk() -> dict[str, np.ndarray]:
    return {
        key: np.zeros((groot_keys.ACTION_HORIZON, dim))
        for key, dim in groot_keys.ACTION_DIMS.items()
    }


def test_action_dims_cover_every_action_key():
    assert set(groot_keys.ACTION_DIMS) == set(OFFICIAL_ACTION_KEYS)
    assert sum(groot_keys.ACTION_DIMS.values()) == 12


def test_adapter_rejects_wrong_action_horizon():
    """A 15-step chunk still slices cleanly, so it must be rejected up front."""
    chunk = _good_chunk()
    chunk["action.base_motion"] = np.zeros((groot_keys.ACTION_HORIZON - 1, 4))

    class _Short:
        def get_action(self, observations):
            return chunk

    with pytest.raises(ValueError, match="must have shape"):
        GrootPolicyAdapter(_Short()).infer(_valid_obs())


def test_adapter_rejects_wrong_action_width():
    chunk = _good_chunk()
    chunk["action.end_effector_position"] = np.zeros((groot_keys.ACTION_HORIZON, 7))

    class _Wide:
        def get_action(self, observations):
            return chunk

    with pytest.raises(ValueError, match="action.end_effector_position"):
        GrootPolicyAdapter(_Wide()).infer(_valid_obs())


def test_adapter_rejects_unexpected_action_key():
    chunk = _good_chunk()
    chunk["action.mystery"] = np.zeros((groot_keys.ACTION_HORIZON, 2))

    class _Extra:
        def get_action(self, observations):
            return chunk

    with pytest.raises(ValueError, match="unexpected action keys"):
        GrootPolicyAdapter(_Extra()).infer(_valid_obs())


def test_validate_action_chunk_accepts_contract_shape():
    validated = validate_action_chunk(_good_chunk())
    for key, dim in groot_keys.ACTION_DIMS.items():
        assert validated[key].shape == (groot_keys.ACTION_HORIZON, dim)


def test_replan_steps_must_be_positive():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="must be >= 1"):
            take_action_chunk(_good_chunk(), bad)


def test_iter_step_actions_yields_per_step_dicts():
    steps = iter_step_actions(_good_chunk(), 3)
    assert len(steps) == 3
    for step in steps:
        assert set(step) == set(OFFICIAL_ACTION_KEYS)
        for key, dim in groot_keys.ACTION_DIMS.items():
            assert step[key].shape == (dim,)


def test_iter_step_actions_rejects_bad_replan():
    with pytest.raises(ValueError):
        iter_step_actions(_good_chunk(), groot_keys.ACTION_HORIZON + 1)


# ------------------------------------------------------------------
# Client-side chunk validation (the wire boundary)
# ------------------------------------------------------------------


def test_client_split_rejects_wrong_action_horizon():
    """What msgpack decoded is not necessarily what the server believed it sent.

    A 15-step chunk slices cleanly for replan_steps=5, so the client must
    re-check the contract rather than trust the far side.
    """
    chunk = _good_chunk()
    chunk["action.base_motion"] = np.zeros((groot_keys.ACTION_HORIZON - 1, 4))
    with pytest.raises(ValueError, match="must have shape"):
        iter_step_actions(chunk, 5)


def test_client_split_rejects_wrong_action_width():
    chunk = _good_chunk()
    chunk["action.gripper_close"] = np.zeros((groot_keys.ACTION_HORIZON, 3))
    with pytest.raises(ValueError, match="action.gripper_close"):
        iter_step_actions(chunk, 5)


def test_client_split_rejects_missing_key():
    chunk = _good_chunk()
    del chunk["action.control_mode"]
    with pytest.raises(ValueError, match="missing action keys"):
        iter_step_actions(chunk, 5)


def test_client_split_rejects_unexpected_key():
    chunk = _good_chunk()
    chunk["action.mystery"] = np.zeros((groot_keys.ACTION_HORIZON, 1))
    with pytest.raises(ValueError, match="unexpected action keys"):
        iter_step_actions(chunk, 5)


def test_client_split_rejects_non_finite():
    chunk = _good_chunk()
    chunk["action.end_effector_rotation"] = np.full(
        (groot_keys.ACTION_HORIZON, 3), np.inf
    )
    with pytest.raises(ValueError, match="non-finite"):
        iter_step_actions(chunk, 5)


def test_server_and_client_share_one_contract():
    """Both ends must reject the same malformed chunk."""
    bad = _good_chunk()
    bad["action.end_effector_position"] = np.zeros((groot_keys.ACTION_HORIZON - 1, 3))

    class _Short:
        def get_action(self, observations):
            return bad

    with pytest.raises(ValueError, match="must have shape"):
        GrootPolicyAdapter(_Short()).infer(_valid_obs())
    with pytest.raises(ValueError, match="must have shape"):
        iter_step_actions(bad, 5)


# ------------------------------------------------------------------
# Constants anchored to literals
# ------------------------------------------------------------------


def test_state_dims_match_official_config():
    """Anchored to literals: every fixture builds its state *from* this table,
    so without an explicit check a wrong width would certify itself."""
    assert groot_keys.STATE_DIMS == OFFICIAL_STATE_DIMS


def test_action_dims_match_official_config():
    assert groot_keys.ACTION_DIMS == OFFICIAL_ACTION_DIMS


def test_quaternion_state_keys_are_the_two_rotation_fields():
    """Pin the list itself.

    Tests that merely loop over QUATERNION_STATE_KEYS pass vacuously when it is
    empty -- the guard would be switched off by the very value it guards.
    """
    assert groot_keys.QUATERNION_STATE_KEYS == OFFICIAL_QUATERNION_STATE_KEYS


def test_image_resolutions_match_official_pipeline():
    """512 render -> 256 model input, per the official RoboCasa365 wrapper."""
    assert groot_keys.RENDER_RESOLUTION == 512
    assert groot_keys.MODEL_IMAGE_RESOLUTION == 256


# ------------------------------------------------------------------
# Temporal ordering of executed actions
# ------------------------------------------------------------------


def test_iter_step_actions_preserves_temporal_order():
    """Step k of the returned list must be row k of the chunk.

    Reversing the loop leaves shapes, keys and counts untouched, so nothing else
    in the suite notices -- yet the robot would execute the chunk backwards.
    """
    chunk = {
        key: np.arange(groot_keys.ACTION_HORIZON * dim, dtype=float).reshape(-1, dim)
        for key, dim in groot_keys.ACTION_DIMS.items()
    }
    steps = iter_step_actions(chunk, 4)
    for offset, step in enumerate(steps):
        for key in groot_keys.ACTION_KEYS:
            np.testing.assert_array_equal(step[key], chunk[key][offset])


def test_take_action_chunk_keeps_leading_rows():
    chunk = {
        key: np.arange(groot_keys.ACTION_HORIZON * dim, dtype=float).reshape(-1, dim)
        for key, dim in groot_keys.ACTION_DIMS.items()
    }
    taken = take_action_chunk(chunk, 3)
    for key in groot_keys.ACTION_KEYS:
        np.testing.assert_array_equal(taken[key], chunk[key][:3])


def test_build_obs_rejects_wrong_video_shape():
    for bad in ((256, 256), (256, 256, 4)):
        obs = _valid_obs()
        obs["video.robot0_agentview_left"] = np.zeros(bad, dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            build_groot_observation(obs)
