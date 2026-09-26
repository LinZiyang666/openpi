"""MetaWorld MT50 conductor ``EpisodeRunner``.

Adapts ``exp.metaworld.env.run_episode`` to the conductor contract the way
``examples.libero.episode_runner.LiberoEpisodeRunner`` does for LIBERO: one
WebSocket connection reused across episodes of the same (server, bundle),
``episode_start`` / ``episode_end`` around every episode, progress reports, and
one per-step row per inference call.

Identity: the dispatched ``EpisodeTask.task_id`` names the MT50 environment
(``tasks.TASK_NAMES[task_id]``; a strategy that also stamps
``extra["task_name"]`` must agree) and ``orig_init_state_idx`` is the
``MT1.train_tasks`` index. ``episode_start`` carries
``{experiment, task=<env name>, episode_id=episode_idx}`` plus
``task_id / orig_init_state_idx / task_uid / attempt / seed`` metadata, the
identity the warm reset evidence and self-start seeds are keyed on.

Per-step rows: ``step_idx`` is the policy step the inference was issued at
(0, 5, 10, ... for ``replan_steps=5``), ``hit_type`` / ``start_t`` and a few
scalar ``__hit_meta__`` fields are copied (``None`` when the server attached
none; ``miss_nfe`` is a plain / full arm's executed MISS steps), so the driver-stamped rows expose decision count, replan spacing and
outcome. One ``_kind: episode_summary`` row per episode (no ``hit_type``)
records steps, decisions, end reason and inference time.

Public interface: ``MetaworldEpisodeRunner``, ``default_client_factory``.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from typing import Any

from exp.metaworld import env as _env
from exp.metaworld import tasks as T
from openpi.conductor import task as _task
from openpi.conductor.worker import EpisodeRunner, ProgressCallback

# Scalar ``__hit_meta__`` fields copied onto every per-step row.
_HIT_FIELDS = (
    "hit_type",
    "start_t",
    "winner_id",
    "cp1_score",
    "checkpoint",
    "score",
    "searched",
    "miss_nfe",
)
# Scalar ``__hit_meta__["warm_reset"]`` fields copied with a ``warm_reset_`` prefix.
_WARM_RESET_FIELDS = (
    "n_steps",
    "decision_nfe",
    "continuation_nfe",
    "self_direct_nfe",
    "self_seed",
)


def _scalar(value: Any) -> Any:
    """JSON-safe scalar: numpy scalars unwrapped, non-finite floats and containers dropped to None."""
    if hasattr(value, "item") and callable(value.item):
        with contextlib.suppress(TypeError, ValueError):
            value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def decision_row(
    task: _task.EpisodeTask, task_name: str, step_idx: int, result: dict
) -> dict:
    """The per-step row of one inference call."""
    hit = result.get("__hit_meta__")
    hit = hit if isinstance(hit, dict) else {}
    row = {
        "yaml_id": task.yaml_id,
        "task_uid": task.task_uid,
        "phase": task.phase,
        "suite": task.experiment,
        "task_id": task.task_id,
        "task": task_name,
        "subset_init_state_idx": task.episode_idx,
        "orig_init_state_idx": task.orig_init_state_idx,
        "step_idx": int(step_idx),
    }
    for key in _HIT_FIELDS:
        row[key] = _scalar(hit.get(key))
    warm = hit.get("warm_reset")
    if isinstance(warm, dict):
        for key in _WARM_RESET_FIELDS:
            row[f"warm_reset_{key}"] = _scalar(warm.get(key))
    return row


def default_client_factory(server: _task.ServerEndpoint) -> Any:
    """Connect a real WebSocket client to the assigned server."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=server.host, port=server.port)


class MetaworldEpisodeRunner(EpisodeRunner):
    """Runs MetaWorld MT50 episodes for the conductor worker loop."""

    def __init__(
        self,
        *,
        seed: int = T.BENCH_SEED,
        replan_steps: int = T.REPLAN_STEPS,
        env_factory: Callable[[str, int, int], Any] = _env.make_env,
        client_factory: Callable[[_task.ServerEndpoint], Any] = default_client_factory,
        run_episode_fn: Callable[..., _env.EpisodeOutcome] = _env.run_episode,
    ) -> None:
        self._seed = seed
        self._replan_steps = replan_steps
        self._env_factory = env_factory
        self._client_factory = client_factory
        self._run_episode_fn = run_episode_fn
        self._client: Any | None = None
        self._client_server: str | None = None
        self._bundle: str | None = None

    @staticmethod
    def task_name(task: _task.EpisodeTask) -> str:
        """MT50 environment name of a dispatched task (cross-checked against ``extra``)."""
        if not 0 <= task.task_id < len(T.TASK_NAMES):
            raise ValueError(
                f"task {task.task_uid!r}: task_id {task.task_id} is not an MT50 task"
            )
        name = T.TASK_NAMES[task.task_id]
        stamped = (task.extra or {}).get("task_name")
        if stamped is not None and stamped != name:
            raise ValueError(
                f"task {task.task_uid!r}: task_name {stamped!r} disagrees with task_id -> {name!r}"
            )
        return name

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

    def run(
        self, task: _task.EpisodeTask, report: ProgressCallback
    ) -> _task.EpisodeResult:
        """Run one episode; a failure drops the (possibly dead) connection before re-raising."""
        try:
            return self._run(task, report)
        except Exception:
            self.close()
            raise

    def _run(
        self, task: _task.EpisodeTask, report: ProgressCallback
    ) -> _task.EpisodeResult:
        name = self.task_name(task)
        client = self._ensure_client(task)
        env = self._env_factory(name, task.orig_init_state_idx, self._seed)
        per_step: list[dict] = []
        started = time.monotonic()
        outcome = None
        try:
            client.episode_start(
                experiment=task.experiment,
                task=name,
                episode_id=task.episode_idx,
                extra_metadata={
                    "task_id": task.task_id,
                    "orig_init_state_idx": task.orig_init_state_idx,
                    "task_uid": task.task_uid,
                    "attempt": task.attempt,
                    "seed": self._seed,
                },
            )
            last_hit: dict = {"hit_type": None}

            def on_decision(step_idx: int, result: dict) -> None:
                row = decision_row(task, name, step_idx, result)
                last_hit["hit_type"] = row["hit_type"]
                per_step.append(row)

            def on_step(step_idx: int) -> None:
                rate = (step_idx + 1) / max(time.monotonic() - started, 1e-6)
                report(step_idx, rate, last_hit["hit_type"])

            try:
                outcome = self._run_episode_fn(
                    env,
                    client.infer,
                    T.PROMPTS[name],
                    replan_steps=self._replan_steps,
                    max_steps=T.MAX_POLICY_STEPS,
                    on_decision=on_decision,
                    on_step=on_step,
                )
            finally:
                with contextlib.suppress(Exception):
                    client.episode_end(success=bool(outcome and outcome.success))
        finally:
            with contextlib.suppress(Exception):
                env.close()
        per_step.append(
            {
                "_kind": "episode_summary",
                "task_uid": task.task_uid,
                "yaml_id": task.yaml_id,
                "task_id": task.task_id,
                "task": name,
                "subset_init_state_idx": task.episode_idx,
                "orig_init_state_idx": task.orig_init_state_idx,
                "seed": self._seed,
                "n_steps": outcome.n_steps,
                "n_decisions": outcome.n_decisions,
                "end_reason": outcome.end_reason,
                "infer_s": round(outcome.infer_s, 4),
                "episode_s": round(time.monotonic() - started, 4),
            }
        )
        return _task.EpisodeResult(
            task.task_uid,
            success=outcome.success,
            n_steps=outcome.n_steps,
            per_step_rows=per_step,
        )

    def close(self) -> None:
        """Close the persistent server connection, if any."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None
        self._client_server = None
        self._bundle = None
