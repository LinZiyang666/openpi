"""Worker execution interface and the pull/run/report worker loop.

The worker is the "dumb" leaf of the conductor framework (plan §2): it
pulls an ``EpisodeTask``, runs it through a pluggable ``EpisodeRunner``, and
reports back. It holds no scheduling policy and does not interpret ``phase`` —
it only forwards the tag. The concrete runner (LIBERO) lives in
``examples/libero/episode_runner.py`` and is injected here.

Coupling map:
  DEPENDS ON:  task.py, protocol.py, standard library
  CONSUMED BY: agent.py (forks worker processes), the standalone libero entry
  IF CHANGED:  agent fork contract, EpisodeRunner implementations
  NOTE:        MUST NOT import exp.* — LIBERO/verdict semantics arrive via the
               injected EpisodeRunner only.
"""

from __future__ import annotations

import abc
# NOTE: ProgressCallback below is a module-level alias evaluated at import time.
# The LIBERO worker env is python 3.8 where collections.abc.Callable is not
# subscriptable; typing.Callable is. Use typing here so worker import succeeds.
from typing import Callable
import contextlib
import dataclasses
import logging
import socket
import time

from openpi.conductor import protocol as _proto
from openpi.conductor import task as _task

logger = logging.getLogger(__name__)

# report(step, actions_per_s, hit_type) — invoked periodically during a run.
ProgressCallback = Callable[[int, float, "str | None"], None]


# ----------------------------------------------------------------------
# EpisodeRunner ABC (plan §6.2)
# ----------------------------------------------------------------------


class EpisodeRunner(abc.ABC):
    """Runs one episode against a server and returns its result.

    Implementations connect to ``task.server``, ``select_bundle(task.bundle_id)``,
    run the episode (env steps + infer), periodically call ``report`` with
    progress, and return an :class:`~openpi.conductor.task.EpisodeResult`.
    A runner may keep a connection open and reuse it across episodes of the same
    ``(server, bundle)`` (plan §6.4); only re-connect when those change.
    """

    @abc.abstractmethod
    def run(self, task: _task.EpisodeTask, report: ProgressCallback) -> _task.EpisodeResult: ...

    def close(self) -> None:  # noqa: B027 - optional hook, default no-op by design
        """Release any persistent connection / resources. Default no-op."""


# ----------------------------------------------------------------------
# WorkerLoop — pull / run / report against the driver (via agent)
# ----------------------------------------------------------------------


class WorkerLoop:
    """Drives one worker process: repeatedly pull a task, run it, report result.

    Transport is a blocking TCP socket to the driver (the agent forwards). On a
    ``none`` assignment the loop backs off; on a dropped connection it retries
    with exponential backoff (plan §5.3). The loop exits on a ``shutdown``
    message or when ``stop`` is set.
    """

    def __init__(
        self,
        worker_id: str,
        server_key: str,
        runner: EpisodeRunner,
        *,
        connect: Callable[[], socket.socket],
        max_backoff_s: float = 30.0,
    ) -> None:
        # ``server_key`` is the bound ServerEndpoint key ("host:port"); the
        # driver schedules per this key (yaml->server affinity, plan §7).
        self._worker_id = worker_id
        self._server_key = server_key
        self._runner = runner
        self._connect = connect
        self._max_backoff_s = max_backoff_s

    # -- one request/response round against the driver --

    def _pull(self, sock: socket.socket) -> _task.EpisodeTask | dict | None:
        """Pull one assignment. Three-state return: an ``EpisodeTask`` (work to
        run), a ``dict`` carrying ``backoff_ms`` (no task ready — idle backoff),
        or ``None`` (shutdown)."""
        _proto.send_message(
            sock,
            _proto.MSG_PULL,
            {"worker_id": self._worker_id, "server_host": self._server_key},
        )
        msg_type, payload = _proto.recv_message(sock)
        if msg_type == _proto.MSG_SHUTDOWN:
            return None
        if msg_type != _proto.MSG_ASSIGN:
            raise _proto.ProtocolError(f"expected assign/shutdown, got {msg_type!r}")
        if payload.get("none"):
            return {"backoff_ms": payload.get("backoff_ms", 1000)}
        return _proto.task_from_wire(payload["task"])

    def _report_result(self, sock: socket.socket, result: _task.EpisodeResult) -> None:
        _proto.send_message(sock, _proto.MSG_REPORT_RESULT, _proto.result_to_wire(result))

    def _run_one(self, sock: socket.socket, task: _task.EpisodeTask) -> None:
        def report(step: int, actions_per_s: float, hit_type: str | None) -> None:
            # Progress is best-effort; a dropped progress frame must not kill the
            # run. The result report (and its retry) is the source of truth.
            with contextlib.suppress(OSError):
                _proto.send_message(
                    sock,
                    _proto.MSG_REPORT_PROGRESS,
                    {
                        "worker_id": self._worker_id,
                        "task_uid": task.task_uid,
                        "step": step,
                        "actions_per_s": actions_per_s,
                        "hit_type": hit_type,
                    },
                )

        # Timed around the runner, including the failure path: an episode that
        # crashed still consumed the worker for however long it ran, and a
        # utilisation figure that silently drops those intervals reads high for
        # exactly the run that went worst.
        started = time.monotonic()
        try:
            result = self._runner.run(task, report)
        except Exception as exc:
            logger.exception("episode %s raised", task.task_uid)
            result = _task.EpisodeResult(task_uid=task.task_uid, success=False, n_steps=0, error=repr(exc))
        # Echo the dispatch attempt so the driver can fence a stale result from a
        # superseded (timed-out + requeued) dispatch of the same task_uid (G2R3).
        # ``duration_s`` is set here rather than in the runner so every runner
        # gets it without knowing about it; a runner that measured it itself is
        # left alone.
        elapsed = time.monotonic() - started
        result = dataclasses.replace(
            result,
            attempt=task.attempt,
            duration_s=result.duration_s if result.duration_s is not None else elapsed,
        )
        self._report_result(sock, result)

    def run_forever(self, stop: Callable[[], bool] | None = None) -> None:
        """Main loop. ``stop`` is an optional predicate to break the loop."""
        backoff = 1.0
        should_stop = stop or (lambda: False)
        try:
            while not should_stop():
                try:
                    sock = self._connect()
                    backoff = 1.0  # reset on a healthy connection
                    idle = 0
                    try:
                        while not should_stop():
                            assignment = self._pull(sock)
                            if assignment is None:
                                return  # shutdown
                            if isinstance(assignment, dict):
                                # No ready task (e.g. a barrier wait): exponential
                                # idle backoff (capped) so many idle workers do not
                                # hammer the driver's pull server (plan §5.3 / R8).
                                base = assignment["backoff_ms"] / 1000.0
                                time.sleep(min(base * (2**idle), self._max_backoff_s))
                                idle = min(idle + 1, 10)
                                continue
                            idle = 0
                            self._run_one(sock, assignment)
                    finally:
                        sock.close()
                except (OSError, _proto.ProtocolError) as exc:
                    logger.warning("worker %s transport error: %s; retry in %.1fs", self._worker_id, exc, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, self._max_backoff_s)
        finally:
            # Runner persists across reconnects; close once when the loop exits.
            self._runner.close()
