"""RoboCasa365 conductor ``EpisodeRunner``: teacher adapters + liveness watchdog (T3).

Worker-side execution leaf. One runner serves both teachers through a
``TeacherAdapter`` that owns the env<->teacher observation/action contract; the
contracts are FROZEN against the baselines that produced the admission-gate
numbers and must not drift:

* pi0.5  — ``exp/robocasa365/baselines/pi05_step0b_client_ORIGINAL.py:46-66``
  (five-field state concat, resize_with_pad 224, env-default render size,
  actions through ``robocasa.utils.env_utils.convert_action``);
* GR00T  — ``exp/robocasa365/groot_rollout_client.py`` (512-render ->
  256 INTER_AREA, env-native language key, action dicts).

The two adapters deliberately differ in render resolution: the pi0.5 baseline
never passed camera sizes to ``gym.make`` while the GR00T one renders at 512.
Unifying either way would change that teacher's input distribution relative to
its admission gate.

``episode_start.task`` is always the canonical env name
(``task.extra["task_name"]``) for BOTH teachers: it becomes the HDF5 ``task``
attr, which the offline builder copies verbatim into every cache entry's
``task_key`` and the online interceptor later filters by. The natural-language
prompt goes only into the inference observation.

Seeding: ``env.reset(seed=base_seed + orig_init_state_idx)`` — the episode's
initial state is a pure function of the seed, but the rollout itself is
stochastic (flow-matching noise is drawn fresh per inference), so a retried
``task_uid`` yields the same initial state and a fresh rollout. Artifact
selection stays deterministic through the attempt-suffixed episode_name plus
the journal-based manifest rule (see ``verify_collection_artifacts.py``).

``WatchdogRunner`` bounds the two places the stock client stack can block
forever (``WebsocketClientPolicy.__init__``'s unbounded ``_wait_for_server``
loop and a mid-episode ``recv``): a per-invocation, generation-scoped watchdog
closes the current client at ``episode_deadline_s`` and hard-exits the worker
(``os._exit(3)``, letting ``WorkerAgent`` respawn it) after
``terminate_grace_s`` more. The race window between the bounded handshake
probe and the real client construction is NOT eliminated (that would require
changing ``openpi_client``); it is bounded by the watchdog.

Coupling map:
  DEPENDS ON:  openpi.conductor (task/worker), openpi_client (lazy),
               exp.robocasa365.groot_rollout_client / groot_policy_adapter (lazy),
               robocasa + gymnasium (lazy; island A only)
  CONSUMED BY: exp.robocasa365.worker_entry, tests/robocasa365
"""

from __future__ import annotations

import collections
import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np

from openpi.conductor import task as _task
from openpi.conductor.worker import EpisodeRunner, ProgressCallback

# ------------------------------------------------------------------
# Frozen contract constants
# ------------------------------------------------------------------

# Every dispatched EpisodeTask MUST carry these keys in ``extra``; the runner
# fails fast instead of silently falling back to a default (the LIBERO runner's
# ``num_trials_per_task`` precedent).
REQUIRED_EXTRA_KEYS = ("task_name", "layout", "style", "teacher", "base_seed", "replan_steps")

# The natural-language instruction source. Feeds the model observation only —
# never ``episode_start.task``.
PROMPT_SOURCE_KEY = "annotation.human.task_description"

# pi0.5 state concatenation — order transcribed from
# baselines/pi05_step0b_client_ORIGINAL.py:53-59 and pinned by a test that
# re-parses that archive, so this tuple cannot drift without failing loudly.
PI05_STATE_CONCAT_ORDER = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
PI05_RESIZE = 224
PI05_IMAGE_KEYS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}


# ------------------------------------------------------------------
# Teacher adapters
# ------------------------------------------------------------------


class TeacherAdapter(Protocol):
    """Owns the env<->teacher contract for one teacher."""

    def env_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for ``gym.make`` (render resolution differs per teacher)."""

    def build_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        """One env observation -> the server's inference payload."""

    def iter_actions(self, response: dict[str, Any], replan_steps: int) -> Iterator[Any]:
        """One server response -> up to ``replan_steps`` env-ready actions."""


class Pi05TeacherAdapter:
    """pi0.5 contract, byte-faithful to the step0b baseline client.

    ``convert_action`` is injectable so the adapter is unit-testable outside
    island A; the default lazily imports the real
    ``robocasa.utils.env_utils.convert_action`` (pinned by the island-A manual
    binding test so a stale copy cannot certify itself).
    """

    def __init__(self, convert_action: Callable[[Any], Any] | None = None) -> None:
        self._convert_action = convert_action

    def env_kwargs(self) -> dict[str, Any]:
        # Frozen: the baseline's gym.make passed NO camera sizes (env default
        # render). Adding them here would change the teacher's input
        # distribution relative to the admission gate.
        return {}

    def build_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        from openpi_client import image_tools

        payload: dict[str, Any] = {}
        for out_key, env_key in PI05_IMAGE_KEYS.items():
            payload[out_key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(np.ascontiguousarray(obs[env_key]), PI05_RESIZE, PI05_RESIZE)
            )
        payload["observation/state"] = np.concatenate(
            [np.asarray(obs[key]) for key in PI05_STATE_CONCAT_ORDER], axis=0
        )
        payload["prompt"] = prompt
        return payload

    def iter_actions(self, response: dict[str, Any], replan_steps: int) -> Iterator[Any]:
        conv = self._convert_action
        if conv is None:
            from robocasa.utils.env_utils import convert_action as conv  # island A only

            self._convert_action = conv
        chunk = response["actions"]
        for vec in list(chunk)[:replan_steps]:
            yield conv(vec)


class GrootTeacherAdapter:
    """GR00T N1.5 contract, delegating to the already-shipped gate client code."""

    def env_kwargs(self) -> dict[str, Any]:
        from exp.robocasa365 import groot_keys

        return {
            "camera_heights": groot_keys.RENDER_RESOLUTION,
            "camera_widths": groot_keys.RENDER_RESOLUTION,
        }

    def build_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        del prompt  # N1.5 wants the env-native language key, preserved by the selection
        from exp.robocasa365.groot_rollout_client import _select_and_downsample

        return _select_and_downsample(obs)

    def iter_actions(self, response: dict[str, Any], replan_steps: int) -> Iterator[Any]:
        from exp.robocasa365.groot_policy_adapter import iter_step_actions

        return iter_step_actions(response["actions"], replan_steps)


ADAPTERS: dict[str, Callable[[], TeacherAdapter]] = {
    "pi05": Pi05TeacherAdapter,
    "groot_tp": GrootTeacherAdapter,
}


# ------------------------------------------------------------------
# Default (real) collaborators — injectable for tests
# ------------------------------------------------------------------


def default_handshake_probe(server: _task.ServerEndpoint, timeout_s: float) -> None:
    """L1 bounded liveness check: FULL handshake including the first frame.

    The server's first action after the WS handshake is sending its metadata
    frame; a probe that tolerated a missing first frame would classify
    "handshake completes but the server never speaks" as healthy, which is
    exactly the hang L1 exists to fast-fail on. Any failure — TCP refusal, WS
    handshake timeout, or a metadata timeout — propagates to the caller (after
    closing the connection) and becomes one bounded retry.

    Fast-fail only, NOT a guarantee — the server can vanish between this probe
    and the real (unbounded-constructor) client; that window is bounded by the
    WatchdogRunner, not eliminated.
    """
    import websockets.sync.client

    conn = websockets.sync.client.connect(
        f"ws://{server.host}:{server.port}", open_timeout=timeout_s, close_timeout=timeout_s
    )
    try:
        conn.recv(timeout=timeout_s)  # server sends metadata as its first frame
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def default_client_factory(server: _task.ServerEndpoint) -> Any:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=server.host, port=server.port)


def default_gym_make(task_name: str, layout: int, style: int, **kwargs: Any) -> Any:
    import robocasa  # noqa: F401 - registers the robocasa/ namespace with gymnasium
    import gymnasium as gym

    return gym.make(
        f"robocasa/{task_name}",
        # split=None bypasses the branch that would also swap the object pool;
        # only the kitchen may vary between build and eval scenes.
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[(layout, style)],
        **kwargs,
    )


def default_horizon_fn(task_name: str) -> int:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    return get_task_horizon(task_name)


# ------------------------------------------------------------------
# RobocasaEpisodeRunner
# ------------------------------------------------------------------


class RobocasaEpisodeRunner(EpisodeRunner):
    """Runs one RoboCasa365 episode against an assigned collection server."""

    def __init__(
        self,
        adapter: TeacherAdapter,
        *,
        client_factory: Callable[[_task.ServerEndpoint], Any] = default_client_factory,
        gym_make: Callable[..., Any] = default_gym_make,
        horizon_fn: Callable[[str], int] = default_horizon_fn,
        handshake_probe: Callable[[_task.ServerEndpoint, float], None] = default_handshake_probe,
        connect_deadline_s: float = 60.0,
        connect_retries: int = 3,
        max_cached_envs: int | None = None,
    ) -> None:
        self._adapter = adapter
        self._client_factory = client_factory
        self._gym_make = gym_make
        self._horizon_fn = horizon_fn
        self._handshake_probe = handshake_probe
        self._connect_deadline_s = connect_deadline_s
        self._connect_retries = max(1, int(connect_retries))
        # None = legacy unbounded cache (collection ran 1 worker on a 48G GPU
        # where 13 cached kitchens fit). Eval fleets on 8G cards MUST bound it:
        # each cached kitchen holds ~1-1.5G VRAM and several GB host RAM, so an
        # unbounded cache walks a worker straight into OOM as the scheduler
        # rotates it across tasks (found live, ws_search 2026-08-21).
        self._max_cached_envs = None if max_cached_envs is None else max(1, int(max_cached_envs))
        self._client: Any | None = None
        self._client_server: str | None = None
        self._bundle: str | None = None
        self._client_lock = threading.Lock()
        # (task_name, layout, style) -> live env. The conductor queue interleaves
        # episodes of different tasks, so without this cache every episode would
        # pay a full gym.make (kitchen build) again.
        self._envs: dict[tuple[str, int, int], Any] = {}

    # -- client lifecycle ------------------------------------------------

    def close_current_client(self) -> None:
        """Best-effort close so a blocked ``recv`` raises (watchdog L2 hook)."""
        with self._client_lock:
            client, self._client, self._client_server, self._bundle = self._client, None, None, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

    def _ensure_client(self, task: _task.EpisodeTask) -> Any:
        with self._client_lock:
            client = self._client
            same_server = self._client_server == task.server.key
        if client is None or not same_server:
            self.close_current_client()
            # L1: bounded probe with bounded retries. A failure here raises and
            # becomes a failed EpisodeResult in WorkerLoop._run_one — the worker
            # stays alive and pulls the next task.
            last_exc: Exception | None = None
            for _ in range(self._connect_retries):
                try:
                    self._handshake_probe(task.server, self._connect_deadline_s)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001 - classified by the driver, not here
                    last_exc = exc
                    time.sleep(min(2.0, self._connect_deadline_s / 10.0))
            if last_exc is not None:
                raise ConnectionError(
                    f"server {task.server.key} failed {self._connect_retries} bounded "
                    f"handshake probes ({self._connect_deadline_s}s each): {last_exc!r}"
                ) from last_exc
            new_client = self._client_factory(task.server)
            with self._client_lock:
                self._client = new_client
                self._client_server = task.server.key
                self._bundle = None
            client = new_client
        if self._bundle != task.bundle_id:
            # Collection servers only ack the "default" slot; the strategy pins
            # bundle_id to it, so this stays a no-op server-side.
            client.select_bundle(task.bundle_id)
            self._bundle = task.bundle_id
        return client

    # -- env lifecycle ---------------------------------------------------

    def _ensure_env(self, task_name: str, layout: int, style: int) -> Any:
        key = (task_name, layout, style)
        env = self._envs.get(key)
        if env is None:
            if self._max_cached_envs is not None:
                while len(self._envs) >= self._max_cached_envs:
                    evict_key, evict_env = next(iter(self._envs.items()))
                    del self._envs[evict_key]
                    with contextlib.suppress(Exception):
                        evict_env.close()
            env = self._gym_make(task_name, layout, style, **self._adapter.env_kwargs())
            self._envs[key] = env
        return env

    # -- evidence hook ---------------------------------------------------

    def _episode_header_rows(
        self, task: _task.EpisodeTask, *, prompt: str, seed: int
    ) -> list[dict[str, Any]]:
        """Rows prepended to ``per_step`` after the episode's single reset.

        Called exactly once per episode, right after ``env.reset`` produced
        ``prompt``/``seed``. The base runner contributes nothing — collection
        and plain eval keep their per-step row set byte-identical. Evidence
        runners (ws2) override this to emit one header row; they must not copy
        the rollout loop or reset a second time to reach these values.
        """
        del task, prompt, seed
        return []

    # -- one episode -----------------------------------------------------

    def run(self, task: _task.EpisodeTask, report: ProgressCallback) -> _task.EpisodeResult:
        extra = task.extra or {}
        missing = [key for key in REQUIRED_EXTRA_KEYS if key not in extra]
        if missing:
            raise ValueError(
                f"EpisodeTask {task.task_uid!r} is missing extra keys {missing}; the "
                "strategy must stamp the full frozen contract (no worker-side defaults)."
            )
        if "/" in task.experiment:
            # ``experiment`` becomes a directory level in the collector without
            # the traversal guard episode_name gets; reject early.
            raise ValueError(f"experiment {task.experiment!r} must not contain '/'")
        task_name = str(extra["task_name"])
        replan_steps = int(extra["replan_steps"])
        layout, style = int(extra["layout"]), int(extra["style"])

        client = self._ensure_client(task)
        env = self._ensure_env(task_name, layout, style)
        horizon = int(self._horizon_fn(task_name))
        seed = int(extra["base_seed"]) + task.orig_init_state_idx

        obs, _ = env.reset(seed=seed)
        prompt = str(obs[PROMPT_SOURCE_KEY])
        # Canonical task name (NOT the language prompt): becomes the HDF5
        # ``task`` attr and, downstream, every cache entry's ``task_key``.
        client.episode_start(
            experiment=task.experiment,
            task=task_name,
            episode_id=task.episode_idx,
            episode_name=f"{task_name}/episode_{task.episode_idx:04d}_a{task.attempt:02d}",
            extra_metadata={
                "task_uid": task.task_uid,
                "attempt": task.attempt,
                "task_id": task.task_id,
                "orig_init_state_idx": task.orig_init_state_idx,
                "seed": seed,
            },
        )
        per_step: list[dict[str, Any]] = []
        success = False
        step = 0
        started = time.monotonic()
        try:
            # Single-reset capture seam: the true prompt/seed exist only here as
            # locals, so subclasses that need them as evidence rows get exactly
            # one hook call instead of copying the rollout or resetting twice.
            # Inside the try so a raising override still runs episode_end --
            # episode_start has already been sent, and leaving it unbalanced
            # would strand the collector's open episode.
            per_step.extend(self._episode_header_rows(task, prompt=prompt, seed=seed))
            plan: collections.deque = collections.deque()
            while step < horizon:
                if not plan:
                    response = client.infer(self._adapter.build_observation(obs, prompt))
                    plan.extend(self._adapter.iter_actions(response, replan_steps))
                    meta = response.get("__hit_meta__")
                    if meta is not None:
                        # Small summaries only: EpisodeResult frames are capped at
                        # 64 MiB, so no tensors ever ride back on per_step_rows.
                        per_step.append(
                            {
                                "task_uid": task.task_uid,
                                "yaml_id": task.yaml_id,
                                "step_idx": step,
                                "hit_type": meta.get("hit_type"),
                                "winner_id": meta.get("winner_id"),
                                "cp1_score": meta.get("cp1_score"),
                                "searched": meta.get("searched"),
                            }
                        )
                obs, _, _, _, info = env.step(plan.popleft())
                if info.get("success"):
                    success = True
                    break
                step += 1
                report(step, step / max(time.monotonic() - started, 1e-6), None)
        finally:
            # Always sent, including on an exception mid-episode: the collector
            # only flushes the HDF5 on episode_end.
            with contextlib.suppress(Exception):
                client.episode_end(success=success)
        return _task.EpisodeResult(task.task_uid, success=success, n_steps=step, per_step_rows=per_step)

    def close(self) -> None:
        self.close_current_client()
        envs, self._envs = list(self._envs.values()), {}
        for env in envs:
            # Each close attempt is isolated: one failing env must not leak the
            # MuJoCo/EGL contexts of the rest.
            with contextlib.suppress(Exception):
                env.close()


# ------------------------------------------------------------------
# WatchdogRunner — process-level liveness bound (L2/L3)
# ------------------------------------------------------------------


class WatchdogRunner(EpisodeRunner):
    """Bounds every ``run()`` call — client construction included.

    Cancellation lifecycle (frozen): each invocation owns one watchdog thread,
    tagged with a monotonically increasing generation. ``run()``'s ``finally``
    disarms (``event.set()``) and ``join()``s it, so no timer thread survives
    into the next episode; a stale generation can therefore never close the
    next episode's client or exit the process. ``os._exit(3)`` fires only when
    generation matches AND the completion event is unset AND the grace period
    has elapsed.
    """

    def __init__(
        self,
        inner: RobocasaEpisodeRunner,
        *,
        episode_deadline_s: float,
        terminate_grace_s: float,
        exit_fn: Callable[[int], None] = os._exit,
    ) -> None:
        self._inner = inner
        self._deadline_s = float(episode_deadline_s)
        self._grace_s = float(terminate_grace_s)
        self._exit_fn = exit_fn
        self._gen = 0
        self._gen_lock = threading.Lock()

    def _is_current(self, gen: int) -> bool:
        with self._gen_lock:
            return gen == self._gen

    def _watch(self, gen: int, done: threading.Event) -> None:
        if done.wait(self._deadline_s):
            return
        if not self._is_current(gen) or done.is_set():
            return
        # L2: close the live socket so a blocked recv raises. If the episode is
        # still stuck inside the client CONSTRUCTOR there is no object to close
        # — fall straight through to the grace timer.
        with contextlib.suppress(Exception):
            self._inner.close_current_client()
        if done.wait(self._grace_s):
            return
        if not self._is_current(gen) or done.is_set():
            return
        # L3: hard exit; WorkerAgent.supervise_once respawns the worker and the
        # driver's timeout requeue re-dispatches the episode.
        self._exit_fn(3)

    def run(self, task: _task.EpisodeTask, report: ProgressCallback) -> _task.EpisodeResult:
        with self._gen_lock:
            self._gen += 1
            gen = self._gen
        done = threading.Event()
        watchdog = threading.Thread(
            target=self._watch, args=(gen, done), name=f"episode-watchdog-{gen}", daemon=True
        )
        watchdog.start()
        try:
            return self._inner.run(task, report)
        finally:
            done.set()
            watchdog.join()

    def close(self) -> None:
        self._inner.close()
