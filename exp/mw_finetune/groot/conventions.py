"""Raw MetaWorld conventions, independent of GR00T and CUDA."""
from __future__ import annotations

import numpy as np

CHANNELS = ("x", "y", "z", "gripper")
EMBODIMENT = "metaworld_sawyer"
EMBODIMENT_ID = 31
HORIZON = 16


def modality_metadata():
    def channels(original_key, absolute):
        return {name: {"start": i, "end": i + 1, "original_key": original_key,
                       "absolute": absolute, "dtype": "float32"}
                for i, name in enumerate(CHANNELS)}
    return {"state": channels("observation.state", True),
            "action": channels("action", False),
            "video": {"image": {"original_key": "observation.image"}},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}}}


def to_groot_observation(obs):
    image = np.asarray(obs["image"])
    state = np.asarray(obs["state"], dtype=np.float32)
    if image.dtype != np.uint8 or image.shape != (480, 480, 3):
        raise ValueError("image must be RGB uint8 (480,480,3)")
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must be finite (4,) in x,y,z,gripper order")
    if not isinstance(obs["prompt"], str) or not obs["prompt"].strip():
        raise ValueError("prompt must be nonempty text")
    result = {"video.image": np.ascontiguousarray(image)[None],
              "annotation.human.action.task_description": np.asarray([obs["prompt"]])}
    result.update({f"state.{name}": state[i:i + 1].reshape(1, 1)
                   for i, name in enumerate(CHANNELS)})
    return result


def from_groot_actions(result):
    parts = []
    for name in CHANNELS:
        value = result[f"action.{name}"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (HORIZON, 1):
            raise ValueError(f"action.{name} must have shape (16,1) or (1,16,1), got {value.shape}")
        parts.append(value)
    actions = np.ascontiguousarray(np.concatenate(parts, axis=1))
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN/Inf")
    return actions


def action_chunk(actions, start, horizon=HORIZON):
    """Match GR00T delta-action zero padding at the end of an episode."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4 or not 0 <= start < len(actions):
        raise ValueError("expected (N,4) actions and an in-episode start")
    result = np.zeros((horizon, 4), dtype=np.float32)
    available = min(horizon, len(actions) - start)
    result[:available] = actions[start:start + available]
    return result


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("statistics require finite (N,4) values")
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, .01, axis=0).tolist(),
            "q99": np.quantile(values, .99, axis=0).tolist()}
