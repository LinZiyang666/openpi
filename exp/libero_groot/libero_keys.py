"""The LIBERO<->GR00T N1.5 contract, in one place.

Two vocabularies meet here and they do *not* agree on names, unlike the
RoboCasa365 line where both ends already spoke GR00T's dialect:

  * the wire, spoken by ``examples/libero/main.py`` -- ``observation/image``,
    ``observation/wrist_image``, ``observation/state`` (8-D: 3 eef pos + 3
    axis-angle + 2 gripper qpos) and ``prompt``;
  * GR00T, spoken by ``examples/Libero/custom_data_config.py:LiberoDataConfig``
    -- ``video.image`` / ``video.wrist_image``, seven scalar ``state.*`` keys,
    and ``annotation.human.action.task_description``.

The split of the 8-D wire state into GR00T's seven keys is transcribed from
the official evaluator (``examples/Libero/eval/run_libero_eval.py``
``_process_observation``): x/y/z/roll/pitch/yaw are one element each and
``state.gripper`` keeps both qpos entries. Getting that split wrong is the
failure this module exists to prevent: every wrong variant still produces
finite numbers of a plausible shape.

⚠ The language key is the N1.5 LIBERO one (``annotation.human.action.
task_description``), NOT the RoboCasa365 one (``annotation.human.
task_description``). The two differ by one path segment and swapping them
yields a silently unconditioned policy.
"""

from __future__ import annotations

# --- wire side (LIBERO client) --------------------------------------------
WIRE_IMAGE = "observation/image"
WIRE_WRIST = "observation/wrist_image"
WIRE_STATE = "observation/state"
WIRE_PROMPT = "prompt"

WIRE_STATE_DIM = 8  # 3 eef pos + 3 axis-angle + 2 gripper qpos

# --- GR00T side (LiberoDataConfig) ----------------------------------------
VIDEO_KEYS = ("video.image", "video.wrist_image")
SCALAR_STATE_KEYS = (
    "state.x",
    "state.y",
    "state.z",
    "state.roll",
    "state.pitch",
    "state.yaw",
)
GRIPPER_STATE_KEY = "state.gripper"
STATE_KEYS = (*SCALAR_STATE_KEYS, GRIPPER_STATE_KEY)
LANGUAGE_KEY = "annotation.human.action.task_description"

ACTION_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)
ACTION_HORIZON = 16

# Wire image resolution. The official GR00T evaluator feeds the raw 256x256
# render and lets the transform chain crop/resize to 224 itself, so the client
# must be run with ``--resize-size 256`` to reproduce that pre-processing.
# Passing the client's 224 default would apply resize_with_pad *and* the
# chain's crop, i.e. a different field of view than the checkpoint trained on.
WIRE_IMAGE_RESOLUTION = 256


def wire_observation_keys() -> tuple[str, ...]:
    return (WIRE_IMAGE, WIRE_WRIST, WIRE_STATE, WIRE_PROMPT)
