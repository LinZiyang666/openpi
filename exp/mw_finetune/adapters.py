"""CPU-testable wire conventions. No scaling, clipping, or gripper inversion."""
from __future__ import annotations

import hashlib
import numpy as np

CHANNELS = ("x", "y", "z", "gripper")


def validate_observation(obs):
    image = np.asarray(obs["observation/image"])
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    if image.shape != (480, 480, 3) or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HWC (480,480,3)")
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must be finite [x,y,z,gripper], shape (4,)")
    prompt = obs["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    return image, state, prompt


def groot_observation(obs):
    image, state, prompt = validate_observation(obs)
    result = {"video.image": image[None],
              "annotation.human.action.task_description": np.asarray([prompt])}
    result.update({f"state.{key}": state[i].reshape(1, 1) for i, key in enumerate(CHANNELS)})
    return result


def canonical_observation(obs, *, task_name, seed, decision_index):
    image, state, prompt = validate_observation(obs)
    return {"image": image, "state": state, "prompt": prompt, "task_name": task_name,
            "seed": seed, "decision_index": decision_index}


def canonical_to_wire(obs):
    wire = {"observation/image": obs["image"], "observation/state": obs["state"], "prompt": obs["prompt"]}
    validate_observation(wire)
    return wire


def validate_actions(actions, minimum=5):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4 or actions.shape[0] < minimum:
        raise ValueError(f"expected (>={minimum},4) action chunk, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN/Inf")
    return np.ascontiguousarray(actions)


def groot_actions(result):
    parts = []
    for key in CHANNELS:
        a = np.asarray(result[f"action.{key}"])
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2 or a.shape[1] != 1:
            raise ValueError(f"action.{key} must be (H,1) or (1,H,1), got {a.shape}")
        parts.append(a)
    return validate_actions(np.concatenate(parts, axis=1))


def episode_seed(task, idx, seed=7, decision=0):
    text = f"mt50:{seed}:{task}:{idx}:{decision}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")
