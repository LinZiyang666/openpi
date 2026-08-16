"""Ordered modality-key contract for RoboCasa365 x GR00T N1.5.

This module holds *only* constants and pure functions, and deliberately imports
nothing beyond the standard library.  That is what allows the adapter tests to
run in the main test environment (where ``gr00t`` is absent) instead of being
confined to the GR00T island -- see ``tests/robocasa365/test_groot_obs_adapter.py``.

Why the ordering matters
------------------------
GR00T's ``ConcatTransform`` splices the per-key tensors together following the
declared list order.  A wrong order still yields tensors of the correct shape
and value range, so the failure is silent: the model simply reads one field as
another.  The lists below are therefore transcribed verbatim from the official
RoboCasa365 configuration rather than inferred.

Source of truth (all four lists and the action horizon)
-------------------------------------------------------
``robocasa`` @ pin ``be22d659b02db8f6d7f3a3c3edc742934fdcbaae``,
``docs/datasets/using_datasets.md`` lines 167-210, which spells out the
``modality_configs`` for ``EmbodimentTag("new_embodiment")`` -- the very tag
carried by the checkpoint this experiment serves.  The annotation key is
corroborated by ``robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json``.

Do NOT derive the order from the checkpoint's ``experiment_cfg/metadata.json``:
its ``modalities`` section is alphabetically sorted and carries no ordering
information.
"""

# ------------------------------------------------------------------
# Ordered modality keys (verbatim from the official new_embodiment config)
# ------------------------------------------------------------------

VIDEO_KEYS: list[str] = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]

STATE_KEYS: list[str] = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
]

ACTION_KEYS: list[str] = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]

# N1.5 `new_embodiment` uses the env-native annotation key.  The parent
# SinglePandaGripperDataConfig declares "annotation.human.action.task_description",
# which belongs to N1.7's ROBOCASA_PANDA embodiment and must NOT be used here.
LANGUAGE_KEYS: list[str] = [
    "annotation.human.task_description",
]

# `delta_indices=list(range(16))` in the official action ModalityConfig.
ACTION_HORIZON: int = 16

# ------------------------------------------------------------------
# State field geometry
# ------------------------------------------------------------------

# Per-key state dimensions, matching the checkpoint metadata.
STATE_DIMS: dict[str, int] = {
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
}

# Per-key action widths.  The action head emits ACTION_HORIZON timesteps for
# each key, so a well-formed chunk is exactly (ACTION_HORIZON, ACTION_DIMS[key]).
# A short or narrow chunk is a silent hazard: slicing it still yields plausible
# per-step dicts, and env.step would happily consume the wrong numbers.
ACTION_DIMS: dict[str, int] = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}

# The two quaternion-valued state fields.  GR00T converts these to rotation_6d
# via pytorch3d, whose convention is wxyz (real part first): a zero-filled
# quaternion is not a rotation and propagates non-finite values downstream.
QUATERNION_STATE_KEYS: list[str] = [
    "state.end_effector_rotation_relative",
    "state.base_rotation",
]

# Identity rotation *as GR00T reads it*.  Verified empirically inside the GR00T
# island: pytorch3d.transforms.quaternion_to_matrix([1,0,0,0]) is the identity
# matrix, whereas [0,0,0,1] is not.
# NOTE: robosuite emits its quaternions xyzw, so GR00T is reading them wxyz --
# consistent between training and evaluation, hence harmless, but it means this
# value is the identity in GR00T's frame, not in the simulator's.  It is used
# only to keep the start-up probe finite and unit-norm.
IDENTITY_QUATERNION_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Native render size used by the official wrapper before it downsamples to the
# resolution the model was trained on.
RENDER_RESOLUTION: int = 512
MODEL_IMAGE_RESOLUTION: int = 256


def wire_observation_keys() -> list[str]:
    """Every key the rollout client must put on the wire, in contract order.

    The wire format deliberately reuses GR00T's own key names, so no renaming
    happens anywhere in the pipeline.
    """
    return [*VIDEO_KEYS, *STATE_KEYS, *LANGUAGE_KEYS]
