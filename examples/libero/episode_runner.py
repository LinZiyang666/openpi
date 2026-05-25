"""LIBERO episode execution kernel as an conductor ``EpisodeRunner``.

Worker-side execution leaf (plan §4.2). Rather than re-implement the (verified)
LIBERO inference loop, this adapter *reuses* ``examples.libero.main._run_episode``
and only adapts it to the conductor contract: connection reuse + bundle
binding, episode_start/end lifecycle, progress reporting, and per-step
``__hit_meta__`` collection (plan §11.4). The server connection is reused across
episodes of the same (server, bundle); re-connect/re-select only on change
(plan §6.4).

The LIBERO-specific bits (``run_episode_fn``, ``episode_setup``, client) are
injected so the adapter logic is unit-testable without LIBERO/CUDA; the real
defaults are wired by ``worker_entry``.

Coupling map:
  DEPENDS ON:  openpi.conductor (task/worker), examples.libero.main (lazy)
  CONSUMED BY: examples.libero.worker_entry
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import time
from typing import Any

from openpi.conductor import task as _task
from openpi.conductor.worker import EpisodeRunner
from openpi.conductor.worker import ProgressCallback

# episode_setup(task) -> (env, initial_state, task_description, max_steps)
EpisodeSetup = Callable[[_task.EpisodeTask], tuple]


def _hit_row(task: _task.EpisodeTask, step: int, hit: dict) -> dict:
    return {
        "yaml_id": task.yaml_id,
        "task_id": task.task_id,
        "episode_idx": task.episode_idx,
        "orig_init_state_idx": task.orig_init_state_idx,
        "phase": task.phase,
        "step": step,
        "hit_type": hit.get("hit_type"),
        "start_t": hit.get("start_t"),
        "winner_id": hit.get("winner_id"),
        "cp1_score": hit.get("cp1_score"),
    }


def default_client_factory(server: _task.ServerEndpoint):
    """Connect a real WebSocket client to the assigned server."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=server.host, port=server.port)


def _default_run_episode(*args, **kwargs):
    from examples.libero import main as _m

    return _m._run_episode(*args, **kwargs)  # noqa: SLF001 - intentional reuse of main internals


class LiberoEpisodeRunner(EpisodeRunner):
    """Adapts ``main._run_episode`` to the conductor EpisodeRunner contract."""

    def __init__(
        self,
        args: Any,
        episode_setup: EpisodeSetup,
        *,
        client_factory: Callable[[_task.ServerEndpoint], Any] = default_client_factory,
        run_episode_fn: Callable[..., tuple] = _default_run_episode,
    ) -> None:
        self._args = args
        self._episode_setup = episode_setup
        self._client_factory = client_factory
        self._run_episode_fn = run_episode_fn
        self._client: Any | None = None
        self._client_server: str | None = None
        self._bundle: str | None = None

    def _ensure_client(self, task: _task.EpisodeTask) -> Any:
        if self._client is None or self._client_server != task.server.key:
            self.close()
            self._client = self._client_factory(task.server)
            self._client_server = task.server.key
            self._bundle = None
        if self._bundle != task.bundle_id:
            self._client.select_bundle(task.bundle_id)
            self._bundle = task.bundle_id
        return self._client

    def run(self, task: _task.EpisodeTask, report: ProgressCallback) -> _task.EpisodeResult:
        client = self._ensure_client(task)
        env, initial_state, task_description, max_steps = self._episode_setup(task)
        client.episode_start(
            experiment=task.experiment,
            task=task_description,
            episode_id=task.episode_idx,
            extra_metadata={"task_id": task.task_id, "orig_init_state_idx": task.orig_init_state_idx},
        )
        per_step: list[dict] = []
        t0 = time.monotonic()

        def infer_recorder(step_idx: int, hit_meta: dict) -> None:
            per_step.append(_hit_row(task, step_idx, hit_meta or {}))

        def step_callback(step: int) -> None:
            rate = (step + 1) / max(time.monotonic() - t0, 1e-6)
            report(step, rate, None)

        success = False
        n_steps = 0
        try:
            result = self._run_episode_fn(
                env,
                client,
                initial_state,
                task_description,
                self._args,
                max_steps,
                infer_recorder=infer_recorder,
                step_callback=step_callback,
            )
            # Positional dependency on main._run_episode's 5-tuple return
            # (success, images, timestamps, traj, final_env_timestep); if that
            # contract changes order, n_steps below would silently misread.
            success = bool(result[0])
            n_steps = int(result[4]) if len(result) > 4 and result[4] is not None else len(per_step)
        finally:
            with contextlib.suppress(Exception):
                client.episode_end(success=success)
        return _task.EpisodeResult(task.task_uid, success=success, n_steps=n_steps, per_step_rows=per_step)

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None
        self._client_server = None
        self._bundle = None
