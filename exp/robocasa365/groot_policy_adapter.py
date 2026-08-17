"""Adapter bridging ``WebsocketPolicyServer`` to a GR00T N1.5 policy.

Why this file exists
--------------------
``openpi.serving.websocket_policy_server.WebsocketPolicyServer`` drives whatever
object it is given through ``policy.infer(obs)``, while ``gr00t``'s
``Gr00tPolicy`` only exposes ``get_action(observations)``.  This module supplies
the missing shim and, along the way, the input/output validation that keeps a
malformed observation from silently becoming a wrong action.

Deliberately dependency-light
-----------------------------
Nothing here imports ``gr00t`` or ``torch``.  The policy is injected, so the
whole translation path is exercisable with a stub in the main test environment
(``tests/robocasa365/test_groot_obs_adapter.py``); only the production wiring in
``serve_groot_n15.py`` needs the GR00T island.

Runtime placement
-----------------
Shared by all three environments, which is exactly why it stays pure-numpy:
the server imports it in the GR00T island
(``/home/weiland/gr00t_n15_venv/.venv``, py3.11 / numpy 1.26.4), the rollout
client imports ``iter_step_actions`` from it in the simulation island
(``.../robocasa365_uv/.venv``, py3.12 / numpy 2.2.5), and the test suite imports
it in the main venv.  Both ends of the wire therefore enforce one contract.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from exp.robocasa365 import groot_keys

# ------------------------------------------------------------------
# Injected policy protocol
# ------------------------------------------------------------------


class _ActionPolicy(Protocol):
    """Minimal surface the adapter needs from a GR00T policy."""

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]: ...


# ------------------------------------------------------------------
# Pure observation translation
# ------------------------------------------------------------------


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN/Inf); refusing to run inference"
        )


def build_groot_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Validate a wire observation and reshape it into GR00T's expected form.

    GR00T consumes every modality with a leading time axis (``T=1`` here, since
    the official config uses ``observation_indices=[0]``).  Key names are passed
    through untouched: the wire format is GR00T's own vocabulary, so there is no
    renaming step anywhere in the pipeline.

    Raises ``ValueError`` -- naming the offending keys -- on a missing key, a
    wrong dtype/shape, or a non-finite value.  It never pads, zero-fills, or
    clips its way around bad input, because such a silent repair is exactly how
    a semantically wrong action reaches the robot looking perfectly healthy.
    """
    missing = [k for k in groot_keys.wire_observation_keys() if k not in obs]
    if missing:
        raise ValueError(f"observation is missing required keys: {missing}")

    out: dict[str, Any] = {}

    for key in groot_keys.VIDEO_KEYS:
        image = np.asarray(obs[key])
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8, got {image.dtype}")
        expected_hw = (
            groot_keys.MODEL_IMAGE_RESOLUTION,
            groot_keys.MODEL_IMAGE_RESOLUTION,
        )
        if image.shape != (*expected_hw, 3):
            # Resolution is part of the contract too: a client that skipped the
            # downsample would send 512x512 frames that are structurally valid
            # but present the model a different field of view than it trained on.
            raise ValueError(
                f"{key} must have shape {(*expected_hw, 3)}, got {image.shape}"
            )
        out[key] = image[np.newaxis, ...]  # [1, H, W, 3]

    for key in groot_keys.STATE_KEYS:
        vector = np.asarray(obs[key], dtype=np.float64)
        expected = groot_keys.STATE_DIMS[key]
        if vector.shape != (expected,):
            raise ValueError(f"{key} must have shape ({expected},), got {vector.shape}")
        _require_finite(key, vector)
        out[key] = vector[np.newaxis, ...]  # [1, D]

    for key in groot_keys.LANGUAGE_KEYS:
        out[key] = np.asarray([str(obs[key])])  # [1]

    return out


def validate_action_chunk(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Check a policy's raw output against the full action contract.

    Every one of the five keys must be exactly ``(ACTION_HORIZON, width)`` and
    finite.  Shape is enforced rather than merely reported because a short or
    narrow chunk fails silently downstream: slicing it still produces per-step
    dicts that ``env.step`` will accept, so the robot is driven by numbers that
    look structurally fine and mean something else.

    Unknown extra keys are rejected too -- their presence means the policy is
    not the one this contract was written against.
    """
    missing = [key for key in groot_keys.ACTION_KEYS if key not in raw]
    if missing:
        raise ValueError(f"policy output is missing action keys: {missing}")
    unexpected = [key for key in raw if key not in groot_keys.ACTION_DIMS]
    if unexpected:
        raise ValueError(f"policy output has unexpected action keys: {unexpected}")

    actions: dict[str, np.ndarray] = {}
    for key in groot_keys.ACTION_KEYS:
        value = np.asarray(raw[key])
        expected = (groot_keys.ACTION_HORIZON, groot_keys.ACTION_DIMS[key])
        if value.shape != expected:
            raise ValueError(f"{key} must have shape {expected}, got {value.shape}")
        _require_finite(key, value)
        actions[key] = value
    return actions


def take_action_chunk(
    actions: dict[str, np.ndarray], replan_steps: int
) -> dict[str, np.ndarray]:
    """Slice the leading ``replan_steps`` timesteps out of an action chunk.

    ``replan_steps`` may not exceed the model's action horizon; asking for more
    steps than were predicted would otherwise silently short-read the chunk.
    """
    if replan_steps < 1:
        raise ValueError(f"replan_steps must be >= 1, got {replan_steps}")
    if replan_steps > groot_keys.ACTION_HORIZON:
        raise ValueError(
            f"replan_steps={replan_steps} exceeds action horizon {groot_keys.ACTION_HORIZON}"
        )
    return {key: value[:replan_steps] for key, value in actions.items()}


def iter_step_actions(
    actions: dict[str, np.ndarray], replan_steps: int
) -> list[dict[str, np.ndarray]]:
    """Validate a chunk, then split it into the per-step dicts ``env.step`` wants.

    The full contract is re-checked here even though the server already applied
    it: this is the client side of a network boundary, and what arrives is
    whatever msgpack decoded, not whatever the server believed it sent.  A chunk
    whose leading dimension is 15 slices perfectly well for ``replan_steps=5``,
    so without this check a short chunk would drive the robot silently.

    Routing the rollout through here also keeps the executed path identical to
    the one the tests cover.
    """
    validated = validate_action_chunk(actions)
    chunk = take_action_chunk(validated, replan_steps)
    return [
        {key: value[offset] for key, value in chunk.items()}
        for offset in range(replan_steps)
    ]


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


class GrootPolicyAdapter:
    """Expose a GR00T policy through the ``infer()`` call the server expects."""

    def __init__(self, policy: _ActionPolicy) -> None:
        self._policy = policy

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run one inference and return ``{"actions": {...}}``.

        Action arrays are returned under a nested ``actions`` mapping so that
        diagnostic fields (timings and the like) can travel alongside them
        without a caller mistaking the envelope for the action data itself.

        A policy may also return dunder-prefixed side-channel fields -- the
        cache interceptor attaches ``__hit_meta__`` this way.  Those are lifted
        out before validation and placed beside ``actions`` rather than inside
        it: ``validate_action_chunk`` rejects unknown keys on purpose, and that
        check has to keep rejecting a genuinely unexpected *action* key.

        Exceptions raised by the underlying policy propagate unchanged -- a
        failed inference must not be papered over into a plausible-looking
        action.
        """
        groot_obs = build_groot_observation(obs)
        raw = self._policy.get_action(groot_obs)
        side_channel = {k: v for k, v in raw.items() if k.startswith("__")}
        actions = {k: v for k, v in raw.items() if not k.startswith("__")}
        return {"actions": validate_action_chunk(actions), **side_channel}

    # ------------------------------------------------------------------
    # Task / episode lifecycle
    # ------------------------------------------------------------------
    #
    # The websocket server probes for these with ``hasattr`` and calls them by
    # keyword.  They are forwarded rather than handled here because the
    # adapter is stateless; only a cache-aware policy has episode state to
    # reset.  A plain policy has no such methods and the forwarding degrades
    # to a no-op, so the teacher-only path is unchanged.

    def on_task_begin(self) -> None:
        self._forward("on_task_begin")

    def on_task_end(self) -> None:
        self._forward("on_task_end")

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
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
