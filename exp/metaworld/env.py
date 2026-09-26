"""MetaWorld MT50 simulator conventions and the closed-loop episode kernel.

Every convention comes from RLinf's standalone MetaWorld evaluation (validated
against its published numbers, ``logs/step_diag_metaworld_selfstart_plan.log.md``
§1-§2): per episode ``MT1(task, seed=BENCH_SEED)`` + ``set_task(train_tasks[idx])``;
camera ``corner2`` (asserted to be camera id 2) moved to ``CAMERA_POS``; a
480x480 render rotated 180 degrees (``[::-1, ::-1]``) and made contiguous;
state ``obs[:4]``; ``SETTLE_STEPS`` zero-action steps after ``reset()``; each
inference's first ``replan_steps`` actions executed open-loop; at most
``MAX_POLICY_STEPS`` policy steps; ``info["success"]`` at any step is a success
and ends the episode. Actions are neither clipped nor scaled here (the
environment clips to [-1, 1]).

``metaworld`` / ``mujoco`` are imported lazily inside ``make_env`` so the
kernel is testable with fake environments in the main venv.
Public interface: ``make_env``, ``render_image``, ``observation``,
``reset_and_settle``, ``run_episode``, ``EpisodeOutcome``.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from exp.metaworld import tasks as T

# decision_callback(step_idx, infer_result) -> None, once per inference call.
DecisionCallback = Callable[[int, dict], None]


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    """Result of one closed-loop episode."""

    success: bool
    n_steps: int  # policy steps executed (settle steps excluded)
    n_decisions: int  # inference calls
    end_reason: str  # success | step_cap | terminated | truncated
    infer_s: float  # wall time blocked on inference


@functools.lru_cache(maxsize=64)
def _benchmark(task: str, seed: int) -> Any:
    # MT1 construction (~1.2 s) is pure data: the seeded task samples. Caching it
    # per (task, seed) keeps every episode's environment fresh and identical.
    import metaworld

    return metaworld.MT1(task, seed=seed)


def make_env(task: str, idx: int, seed: int = T.BENCH_SEED) -> Any:
    """Build a fresh RLinf-convention environment for one ``(task, idx, seed)`` identity."""
    if not 0 <= idx < T.N_TRAIN_TASKS:
        raise IndexError(f"MetaWorld idx {idx} outside 0..{T.N_TRAIN_TASKS - 1}")
    mt1 = _benchmark(task, seed)
    env = mt1.train_classes[task](
        render_mode="rgb_array",
        camera_name=T.CAMERA_NAME,
        width=T.RENDER_SIZE,
        height=T.RENDER_SIZE,
    )
    env.set_task(mt1.train_tasks[idx])
    place_camera(env)
    return env


def place_camera(env: Any) -> None:
    """Move camera ``corner2`` to RLinf's viewpoint; RLinf indexes it by id 2, so assert that id."""
    name = env.model.camera(T.CAMERA_ID).name
    if name != T.CAMERA_NAME:
        raise AssertionError(
            f"camera id {T.CAMERA_ID} is {name!r}, expected {T.CAMERA_NAME!r}"
        )
    env.model.cam_pos[T.CAMERA_ID] = T.CAMERA_POS


def render_image(env: Any) -> np.ndarray:
    """The policy image: the render rotated 180 degrees, as a contiguous uint8 array."""
    return np.ascontiguousarray(np.asarray(env.render())[::-1, ::-1])


def observation(env: Any, obs: np.ndarray, prompt: str) -> dict:
    """One policy request in the ``MetaworldInputs`` wire format."""
    return {
        "observation/image": render_image(env),
        "observation/state": np.asarray(obs[:4], dtype=np.float32),
        "prompt": prompt,
    }


def reset_and_settle(env: Any) -> np.ndarray:
    """``reset()`` then ``SETTLE_STEPS`` zero-action steps; returns the settled observation."""
    obs, _ = env.reset()
    zero = np.zeros(4, dtype=np.float32)
    for _ in range(T.SETTLE_STEPS):
        obs, *_ = env.step(zero)
    return obs


def run_episode(
    env: Any,
    infer: Callable[[dict], dict],
    prompt: str,
    *,
    replan_steps: int = T.REPLAN_STEPS,
    max_steps: int = T.MAX_POLICY_STEPS,
    on_decision: DecisionCallback | None = None,
    on_step: Callable[[int], None] | None = None,
) -> EpisodeOutcome:
    """Run one episode from reset; ``infer`` returns the policy result dict.

    ``on_decision(step_idx, result)`` fires after every inference with the policy
    step index it was issued at (0, replan_steps, 2*replan_steps, ...);
    ``on_step(step_idx)`` fires before every executed policy step.
    """
    if not 0 < replan_steps <= T.REPLAN_STEPS:
        raise ValueError(f"replan_steps must lie in 1..{T.REPLAN_STEPS}")
    obs = reset_and_settle(env)
    plan: list[np.ndarray] = []
    steps = decisions = 0
    infer_s = 0.0
    reason = "step_cap"
    while steps < max_steps:
        if not plan:
            request = observation(env, obs, prompt)
            started = time.perf_counter()
            result = infer(request)
            infer_s += time.perf_counter() - started
            actions = np.asarray(result["actions"])
            if (
                actions.ndim != 2
                or actions.shape[0] < replan_steps
                or actions.shape[1] != 4
            ):
                raise ValueError(
                    f"policy returned actions of shape {actions.shape}; need (>={replan_steps}, 4)"
                )
            decisions += 1
            if on_decision is not None:
                on_decision(steps, result)
            plan = list(actions[:replan_steps])
        if on_step is not None:
            on_step(steps)
        obs, _, terminated, truncated, info = env.step(
            np.asarray(plan.pop(0), dtype=np.float32)
        )
        steps += 1
        if info.get("success", 0):
            reason = "success"
            break
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            break
    return EpisodeOutcome(
        success=reason == "success",
        n_steps=steps,
        n_decisions=decisions,
        end_reason=reason,
        infer_s=infer_s,
    )
