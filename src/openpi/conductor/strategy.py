"""Programmable experiment-strategy interface and the inter-stage blackboard.

This is the *policy* seam of the conductor framework: the driver core
(mechanism) drives a ``TaskGraph`` produced by an ``ExperimentStrategy`` and
calls its stage-lifecycle hooks; all experiment semantics (which yaml is warmup
vs eval, when to fetch_dump / preload, etc.) live in concrete strategies under
``exp/``. The core itself contains no experiment knowledge.

Coupling map:
  DEPENDS ON:  task.py, openpi_client (ctl, type-only), standard library
  CONSUMED BY: driver.py (calls the hooks), concrete strategies under exp/
  IF CHANGED:  every concrete ExperimentStrategy
  NOTE:        MUST NOT import exp.* or LIBERO — the decoupling boundary.
"""

from __future__ import annotations

import abc
import threading
from typing import TYPE_CHECKING, Any

from openpi.conductor import task as _task

if TYPE_CHECKING:
    # Type-only; the runtime contract is duck-typed so fakes work in tests.
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
else:  # pragma: no cover - alias for runtime annotations
    WebsocketClientPolicy = Any


# ----------------------------------------------------------------------
# StageContext — thread-safe inter-stage blackboard (plan §5.2)
# ----------------------------------------------------------------------


class StageContext:
    """Carries the calibration buffer handoff from a warmup stage to its eval
    consumers. Thread-safe because ``on_stage_complete`` runs on a separate
    thread from the scheduler loop (plan §6.4).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, dict[str, list[float]]] = {}

    def publish(self, calib_id: str, buffer: dict[str, list[float]]) -> None:
        """Store a calibration buffer (warmup on_stage_complete → eval consumers)."""
        with self._lock:
            self._buffers[calib_id] = buffer

    def get(self, calib_id: str) -> dict[str, list[float]] | None:
        """Return the buffer for ``calib_id`` (None if not yet published)."""
        with self._lock:
            return self._buffers.get(calib_id)

    def has(self, calib_id: str) -> bool:
        """Whether a buffer for ``calib_id`` has been published."""
        with self._lock:
            return calib_id in self._buffers

    def drop(self, calib_id: str) -> None:
        """Discard a buffer once its last consuming eval stage no longer needs it."""
        with self._lock:
            self._buffers.pop(calib_id, None)


# ----------------------------------------------------------------------
# ExperimentStrategy ABC (plan §6.1)
# ----------------------------------------------------------------------


class ExperimentStrategy(abc.ABC):
    """A programmable experiment "script".

    The core guarantees the per-stage ordering
    ``upstream.on_stage_complete -> downstream.on_stage_begin -> downstream
    episodes ready -> downstream done -> downstream.on_stage_complete``.
    """

    @abc.abstractmethod
    def plan(
        self,
        yamls: list[str],
        server_assignment: dict[str, _task.ServerEndpoint],
    ) -> _task.TaskGraph:
        """Produce the task graph: per-yaml stage sequence + dependencies.

        ``server_assignment`` maps ``yaml_id -> ServerEndpoint`` (the core's
        yaml->server placement); each stage carries the setup info it needs.
        """

    def on_stage_begin(  # noqa: B027 - optional lifecycle hook, default no-op
        self,
        stage: _task.Stage,
        ctl: WebsocketClientPolicy,
        ctx: StageContext,
    ) -> None:
        """Ordered setup control-ops *before* this stage's episodes go ready.

        Typical (plan §6.1):
          - warmup stage: ctl.load_cache_config(warmup_yaml, yaml_id=...)
          - eval stage:   ctl.preload_normalizer_buffer(eval_id, ctx.get(calib_id))
                          THEN ctl.load_cache_config(eval_yaml, yaml_id=eval_id)
                          (order matters: config load fail-fasts if pool absent)
        Default no-op for stages with no setup.
        """

    def on_stage_complete(  # noqa: B027 - optional lifecycle hook, default no-op
        self,
        stage: _task.Stage,
        ctl: WebsocketClientPolicy,
        ctx: StageContext,
    ) -> None:
        """Handoff / teardown *after* all of this stage's episodes are done.

        Typical:
          - warmup stage: buf = aggregate(ctl.fetch_dump(f"{cleanup_id}__warmup"))
                          ctx.publish(calib_id, buf)
          - eval stage:   ctl.unload_warmup_buffer(eval_id)
        Default no-op.
        """

    def on_resume(  # noqa: B027 - optional lifecycle hook, default no-op
        self,
        stage: _task.Stage,
        ctl: WebsocketClientPolicy,
        ctx: StageContext,
    ) -> None:
        """Triggered server-side self-heal (plan §8.3): the core calls this when
        a stage's server-side precondition (WarmupPool) may be invalid, so the
        strategy can rebuild it (rerun warmup + preload). Default no-op.
        """
