"""Conductor driver: the central engine (plan §3, §7, §8).

Wires together the scheduler (what to run), the strategy (how to set up each
stage), the journal (crash-safe resume), and the worker pull transport. The
driver owns:
  - yaml->server placement (``assign_servers``) honouring calib co-location;
  - a ctl connection pool (one control WebSocket per server);
  - the stage lifecycle loop (on_stage_begin / on_stage_complete) run on a
    *separate* thread so a slow barrier (fetch_dump + preload) never blocks the
    worker pull path (plan §6.4);
  - the pull/report handlers (idempotent on task_uid), retry classification, and
    per-step row collection (plan §11.4).

Coupling map:
  DEPENDS ON:  task.py, scheduler.py, journal.py, strategy.py, protocol.py,
               openpi_client (ctl, via injected factory), standard library
  CONSUMED BY: conductor entry scripts / thin exp wrappers
  IF CHANGED:  agent.py forwarding, strategy hook contract
  NOTE:        MUST NOT import exp.* or LIBERO.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
import socket
import threading
import time
from typing import Any

from openpi.conductor import protocol as _proto
from openpi.conductor import scheduler as _sched
from openpi.conductor import strategy as _strat
from openpi.conductor import task as _task
from openpi.conductor.journal import Journal

logger = logging.getLogger(__name__)

# Error substrings that mark a *fatal* (non-retriable) failure: a misconfigured
# config surfaces as ConfigValidationError and must not be retried (plan §9.1).
_FATAL_ERROR_MARKERS = ("ConfigValidationError", "ConfigValidation", "FatalEpisodeError")


# ----------------------------------------------------------------------
# yaml -> server placement (plan §7)
# ----------------------------------------------------------------------


def assign_servers(
    yaml_weights: dict[str, int],
    servers: list[_task.ServerEndpoint],
    colocation: dict[str, str] | None = None,
) -> dict[str, _task.ServerEndpoint]:
    """Greedily place yamls onto servers, balancing by episode weight while
    keeping co-located yamls (those sharing a calibration source) together.

    ``colocation`` maps ``yaml_id -> group_key``; yamls in the same group are
    assigned to the same server (so a shared WarmupPool stays on one process).
    Balances by total group weight onto the least-loaded server (fewest yamls
    naturally follows from one group landing on one server).
    """
    if not servers:
        raise ValueError("no servers to assign to")
    colocation = colocation or {}
    # Build groups: group_key -> [yaml_ids]; ungrouped yamls are singletons.
    groups: dict[str, list[str]] = {}
    for yaml_id in yaml_weights:
        gkey = colocation.get(yaml_id, yaml_id)
        groups.setdefault(gkey, []).append(yaml_id)

    load = {s.key: 0 for s in servers}
    by_key = {s.key: s for s in servers}
    assignment: dict[str, _task.ServerEndpoint] = {}
    # Heaviest groups first for a better balance.
    for gkey in sorted(groups, key=lambda g: -sum(yaml_weights[y] for y in groups[g])):
        target_key = min(load, key=lambda k: load[k])
        for yaml_id in groups[gkey]:
            assignment[yaml_id] = by_key[target_key]
            load[target_key] += yaml_weights[yaml_id]
    return assignment


def is_retriable_error(error: str | None) -> bool:
    """Classify a worker-reported error as retriable (plan §9.1).

    Called only on the failure path (``error`` is non-None). A fatal config
    error (ConfigValidationError, ...) is not retried; everything else
    (network / timeout / crash) is. ``None`` returns False defensively.
    """
    if error is None:
        return False
    return not any(marker in error for marker in _FATAL_ERROR_MARKERS)


# ----------------------------------------------------------------------
# ConductorDriver
# ----------------------------------------------------------------------


class ConductorDriver:
    """Central engine. Construct with a strategy + the yamls/servers, then run."""

    def __init__(
        self,
        strategy: _strat.ExperimentStrategy,
        *,
        yaml_weights: dict[str, int],
        servers: list[_task.ServerEndpoint],
        journal_path: str,
        ctl_factory: Callable[[_task.ServerEndpoint], Any],
        colocation: dict[str, str] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        monitor: Any | None = None,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        episode_timeout_s: float = 1800.0,
    ) -> None:
        self._strategy = strategy
        self._assignment = assign_servers(yaml_weights, servers, colocation)
        self._graph = strategy.plan(sorted(yaml_weights), self._assignment)
        self._scheduler = _sched.EpisodeScheduler(self._graph, **(scheduler_kwargs or {}))
        self._journal = Journal(journal_path)
        _done_uids = self._journal.replay_done_uids()
        self._scheduler.mark_preexisting_done(_done_uids)
        # Resume mode (plan §8.3 B): a non-empty journal means we are resuming;
        # the server-side WarmupPool may be stale/lost, so eval stages self-heal
        # their calibration at setup. (Warmup stages always rerun whole on resume
        # since they are stage-atomic and not episode-journaled.)
        self._resuming = bool(_done_uids)
        self._ctx = _strat.StageContext()
        self._ctl_factory = ctl_factory
        self._monitor = monitor
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._episode_timeout_s = episode_timeout_s

        self._ctls: dict[str, Any] = {}
        self._ctl_lock = threading.Lock()
        self._per_step_rows: list[dict[str, Any]] = []
        self._rows_lock = threading.Lock()
        self._actual_port: int | None = None

    # -- ctl connection pool --

    def _ctl(self, server: _task.ServerEndpoint) -> Any:
        with self._ctl_lock:
            ctl = self._ctls.get(server.key)
            if ctl is None:
                ctl = self._ctl_factory(server)
                self._ctls[server.key] = ctl
            return ctl

    # -- stage lifecycle (runs on its own thread; never blocks pull) --

    def _setup_stage(self, stage: _task.Stage) -> None:
        self._scheduler.mark_setup_running(stage.stage_id)
        ctl = self._ctl(stage.server)
        # warmup atomicity (plan §8.2 / G1R4): clear the dump before any
        # (re)start so a rerun never appends onto a stale file.
        if stage.phase == "warmup" and stage.produces_calib_id:
            calib = self._graph.calibrations.get(stage.produces_calib_id)
            if calib is not None and calib.cleanup_id is not None:
                with contextlib.suppress(Exception):
                    ctl.unload_warmup_buffer(calib.cleanup_id)
        elif self._resuming and stage.phase == "eval":
            # plan §8.3 (B): on resume the server's WarmupPool may be stale/lost;
            # let the strategy drop it so on_stage_begin re-preloads cleanly.
            self._strategy.on_resume(stage, ctl, self._ctx)
        self._strategy.on_stage_begin(stage, ctl, self._ctx)
        self._scheduler.mark_setup_done(stage.stage_id)

    def _complete_stage(self, stage: _task.Stage) -> None:
        self._scheduler.mark_complete_running(stage.stage_id)
        ctl = self._ctl(stage.server)
        self._strategy.on_stage_complete(stage, ctl, self._ctx)
        self._scheduler.mark_complete_done(stage.stage_id)

    def drive_stages_once(self) -> None:
        """Run one pass of stage setup/complete (also a test/driver entry point)."""
        for stage in self._scheduler.pending_setups():
            try:
                self._setup_stage(stage)
            except Exception as exc:
                # Classify (plan §9.1): a fatal config error (ConfigValidationError,
                # ...) must NOT retry forever → FAILED + cascade; a transient ctl
                # error rolls back to SETUP_PENDING and self-recovers (bounded by
                # max_setup_retries).
                logger.exception("stage setup failed: %s", stage.stage_id)
                self._scheduler.mark_setup_failed(stage.stage_id, fatal=not is_retriable_error(repr(exc)))
        for stage in self._scheduler.pending_completes():
            try:
                self._complete_stage(stage)
            except Exception as exc:
                logger.exception("stage complete failed: %s", stage.stage_id)
                self._scheduler.mark_complete_failed(stage.stage_id, fatal=not is_retriable_error(repr(exc)))

    def _stage_loop(self, stop: Callable[[], bool]) -> None:
        while not stop() and not self._scheduler.all_done():
            self.drive_stages_once()
            # Wall-clock episode timeout: requeue tasks stuck on a hung worker
            # (plan §9.2 — worker alive but blocked in infer, never reports).
            self._scheduler.requeue_timed_out(timeout_s=self._episode_timeout_s)
            time.sleep(0.02)
        self.drive_stages_once()  # final flush

    # -- pull / report handlers (transport-agnostic; idempotent) --

    def handle_pull(self, server_key: str) -> tuple[str, dict[str, Any]]:
        """Return an (msg_type, payload) for a worker pull on the given server.

        ``server_key`` is the worker's bound ServerEndpoint key ("host:port").
        """
        task = self._scheduler.next_task(server_key)
        if task is None:
            return _proto.MSG_ASSIGN, {"none": True, "backoff_ms": 200}
        return _proto.MSG_ASSIGN, {"task": _proto.task_to_wire(task)}

    def handle_result(self, payload: dict[str, Any]) -> None:
        result = _proto.result_from_wire(payload)
        retriable = is_retriable_error(result.error) if not result.success else False
        self._scheduler.mark_result(
            result.task_uid, success=result.success, retriable=retriable, attempt=result.attempt
        )
        # Journal terminal records only (a requeued episode is not terminal).
        if result.success or not retriable:
            yaml_id, phase = self._uid_meta(result.task_uid)
            self._journal.record(
                task_uid=result.task_uid,
                yaml_id=yaml_id,
                phase=phase,
                status="done" if result.success else "failed",
                success=result.success,
            )
        if result.per_step_rows:
            with self._rows_lock:
                self._per_step_rows.extend(result.per_step_rows)
        if self._monitor is not None:
            self._monitor.on_result(result)

    def handle_progress(self, payload: dict[str, Any]) -> None:
        if self._monitor is not None:
            self._monitor.on_progress(payload)

    @staticmethod
    def _uid_meta(task_uid: str) -> tuple[str, str]:
        # task_uid == f"{yaml_id}:{phase}:{task_id}:{episode_idx}". rsplit from the
        # right so a yaml_id that itself contains ':' is not truncated.
        parts = task_uid.rsplit(":", 3)
        return (parts[0], parts[1]) if len(parts) == 4 else (task_uid, "eval")

    # -- TCP pull server --

    def _handle_conn(self, conn: socket.socket, stop: Callable[[], bool]) -> None:
        # Track episodes dispatched on this connection but not yet reported, so a
        # worker crash (connection drop) requeues them (plan §9.2).
        inflight: set[str] = set()
        try:
            while not stop():
                try:
                    msg_type, payload = _proto.recv_message(conn)
                except (ConnectionError, _proto.ProtocolError):
                    break
                if msg_type == _proto.MSG_PULL:
                    key = payload.get("server_host", "")
                    # Reuse handle_pull so the assign/none/backoff logic lives in one place.
                    _, assign_payload = self.handle_pull(key)
                    if "task" in assign_payload:
                        inflight.add(assign_payload["task"]["task_uid"])
                    _proto.send_message(conn, _proto.MSG_ASSIGN, assign_payload)
                elif msg_type == _proto.MSG_REPORT_RESULT:
                    # Record first, then drop from inflight: if handle_result
                    # raises (malformed payload), the uid stays inflight and is
                    # requeued by the finally block rather than silently lost.
                    self.handle_result(payload)
                    inflight.discard(payload.get("task_uid", ""))
                elif msg_type == _proto.MSG_REPORT_PROGRESS:
                    self.handle_progress(payload)
        finally:
            for uid in inflight:
                # Crashed mid-episode: requeue (eval) / invalidate warmup stage.
                self._scheduler.mark_result(uid, success=False, retriable=True)
            with contextlib.suppress(OSError):
                conn.close()

    @property
    def scheduler(self) -> _sched.EpisodeScheduler:
        return self._scheduler

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def port(self) -> int | None:
        return self._actual_port

    @property
    def per_step_rows(self) -> list[dict[str, Any]]:
        with self._rows_lock:
            return list(self._per_step_rows)

    def run(self, stop: Callable[[], bool] | None = None, *, poll_s: float = 0.05) -> None:
        """Start the pull server + stage loop; return when all stages are done."""
        should_stop = stop or (lambda: False)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._bind_host, self._bind_port))
        srv.listen(128)
        srv.settimeout(0.2)
        self._actual_port = srv.getsockname()[1]

        done_flag = threading.Event()

        def loop_stop() -> bool:
            return should_stop() or done_flag.is_set()

        stage_thread = threading.Thread(target=self._stage_loop, args=(loop_stop,), daemon=True)
        stage_thread.start()
        conn_threads: list[threading.Thread] = []
        try:
            while not should_stop() and not self._scheduler.all_done():
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                t = threading.Thread(target=self._handle_conn, args=(conn, loop_stop), daemon=True)
                t.start()
                conn_threads.append(t)
        finally:
            done_flag.set()
            srv.close()
            stage_thread.join(timeout=2.0)
