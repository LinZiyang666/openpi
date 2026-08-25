"""Bridge the LIBERO wire protocol to a GR00T N1.5 policy.

``WebsocketPolicyServer`` calls ``policy.infer(obs)``; ``Gr00tPolicy`` exposes
``get_action(observations)``. This module is the shim, and it also owns the
rename between the two vocabularies described in ``libero_keys``.

Like its RoboCasa365 sibling it imports neither ``gr00t`` nor ``torch``: the
policy is injected, so the whole translation path is testable in the main venv
with a stub while production wiring lives in ``serve_groot_libero.py``.

The validation is deliberately unforgiving -- no padding, no clipping, no
zero-filling. A malformed observation that gets quietly repaired produces an
action that is structurally perfect and semantically wrong, which is the one
failure mode nothing downstream can catch.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol

import numpy as np

from exp.libero_groot import libero_keys as K


class _ActionPolicy(Protocol):
    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]: ...


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN/Inf); refusing to run inference"
        )


def build_groot_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Translate one wire observation into GR00T's LIBERO input dict.

    Every modality gains a leading time axis (T=1, matching the config's
    ``observation_indices=[0]``). The 8-D wire state is split exactly the way
    the official evaluator splits it: six scalars then the 2-D gripper qpos.
    """
    missing = [k for k in K.wire_observation_keys() if k not in obs]
    if missing:
        raise ValueError(f"observation is missing required keys: {missing}")

    out: dict[str, Any] = {}

    for wire_key, groot_key in ((K.WIRE_IMAGE, "video.image"), (K.WIRE_WRIST, "video.wrist_image")):
        image = np.asarray(obs[wire_key])
        if image.dtype != np.uint8:
            raise ValueError(f"{wire_key} must be uint8, got {image.dtype}")
        expected = (K.WIRE_IMAGE_RESOLUTION, K.WIRE_IMAGE_RESOLUTION, 3)
        if image.shape != expected:
            raise ValueError(
                f"{wire_key} must have shape {expected}, got {image.shape}. "
                "Run the LIBERO client with --resize-size 256: the official "
                "GR00T evaluator feeds the raw render and lets the transform "
                "chain crop to 224, so a pre-resized 224 frame would go through "
                "that crop twice and change the field of view."
            )
        out[groot_key] = image[np.newaxis, ...]

    state = np.asarray(obs[K.WIRE_STATE], dtype=np.float64)
    if state.shape != (K.WIRE_STATE_DIM,):
        raise ValueError(
            f"{K.WIRE_STATE} must have shape ({K.WIRE_STATE_DIM},), got {state.shape}"
        )
    _require_finite(K.WIRE_STATE, state)
    for i, key in enumerate(K.SCALAR_STATE_KEYS):
        out[key] = state[i].reshape(1, 1)
    out[K.GRIPPER_STATE_KEY] = state[6:8].reshape(1, 2)

    out[K.LANGUAGE_KEY] = np.asarray([str(obs[K.WIRE_PROMPT])])
    return out


def validate_action_chunk(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Check the policy output against the seven-key LIBERO action contract."""
    missing = [k for k in K.ACTION_KEYS if k not in raw]
    if missing:
        raise ValueError(f"policy output is missing action keys: {missing}")
    unexpected = [k for k in raw if k not in K.ACTION_KEYS]
    if unexpected:
        raise ValueError(f"policy output has unexpected action keys: {unexpected}")

    actions: dict[str, np.ndarray] = {}
    for key in K.ACTION_KEYS:
        value = np.asarray(raw[key], dtype=np.float64)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != (K.ACTION_HORIZON,):
            raise ValueError(
                f"{key} must have shape ({K.ACTION_HORIZON},), got {value.shape}"
            )
        _require_finite(key, value)
        actions[key] = value
    return actions


def normalize_gripper_action(chunk: np.ndarray) -> np.ndarray:
    """Map the gripper column from GR00T's [0, 1] to LIBERO's [+1, -1], in place.

    Transcribed from the official evaluator's ``normalize_gripper_action``
    (``examples/Libero/eval/utils.py``), which ``_convert_to_libero_action``
    applies with ``binarize=True`` to every action before ``env.step``:
    ``y = 1 - 2 * (x - 0) / (1 - 0)`` then ``sign``.

    The two conventions are *inverted*, not merely scaled: GR00T's action head
    emits gripper openness (1 = open) while robosuite reads +1 as "close" (its
    no-op action is ``[0]*6 + [-1]``). Passing the raw value through therefore
    commands the opposite of the intent at an un-saturated magnitude, and the
    arm reaches for objects it never grips -- every episode runs out its step
    budget with a clean-looking trajectory and no error anywhere.
    """
    chunk[..., -1] = 1.0 - 2.0 * chunk[..., -1]
    chunk[..., -1] = np.sign(chunk[..., -1])
    return chunk


def chunk_to_libero_actions(actions: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the seven action keys into LIBERO's 7-D action array per step.

    LIBERO's ``env.step`` takes ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``;
    the key order in ``ACTION_KEYS`` is that order, so this is a stack rather
    than a lookup table -- and the contract check above guarantees the width.
    The gripper column is converted here, the single place the 7-D array is
    assembled, so both ``infer`` and ``iter_step_actions`` inherit it.
    """
    chunk = np.stack([actions[k] for k in K.ACTION_KEYS], axis=-1)  # [T, 7]
    return normalize_gripper_action(chunk)


def iter_step_actions(raw: dict[str, Any], replan_steps: int) -> Iterator[np.ndarray]:
    """Yield up to ``replan_steps`` single-step 7-D actions from one chunk."""
    chunk = chunk_to_libero_actions(validate_action_chunk(raw))
    for row in chunk[:replan_steps]:
        yield row


class GrootLiberoPolicyAdapter:
    """``infer(obs) -> {"actions": [T, 7]}`` over an injected GR00T policy.

    Episode lifecycle calls are forwarded to the wrapped policy when it
    implements them. This matters for collection: the server dispatches the
    client's ``episode_start``/``episode_end`` control frames only when the
    served object exposes the hooks, so an adapter that swallowed them would
    leave the collector recording every step into an episode it never opened —
    silently producing no HDF5 at all, with the rollout looking perfectly fine.
    """

    def __init__(self, policy: _ActionPolicy) -> None:
        self._policy = policy

    def infer(self, obs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run one inference and return ``{"actions": [T, 7], **side_channel}``.

        A policy may also return dunder-prefixed side-channel fields -- the
        cache interceptor attaches ``__hit_meta__`` that way, and the client
        reads it back off the infer result. They are lifted out *before*
        validation and placed beside ``actions`` rather than inside it:
        ``validate_action_chunk`` rejects unknown keys on purpose, and that
        check has to keep rejecting a genuinely unexpected *action* key.
        """
        del kwargs  # the LIBERO client never passes noise
        raw = self._policy.get_action(build_groot_observation(obs))
        side_channel = {k: v for k, v in raw.items() if k.startswith("__")}
        actions = {k: v for k, v in raw.items() if not k.startswith("__")}
        return {
            "actions": chunk_to_libero_actions(validate_action_chunk(actions)),
            **side_channel,
        }

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._forward(
            "on_episode_start",
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
            extra_metadata=extra_metadata,
        )

    def on_episode_end(self, success: bool) -> None:
        self._forward("on_episode_end", success=success)

    def _forward(self, name: str, **kwargs: Any) -> None:
        hook = getattr(self._policy, name, None)
        if hook is not None:
            hook(**kwargs)
