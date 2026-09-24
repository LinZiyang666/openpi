"""Trace-mode server lifecycle shared by every serving entry point (plan §7.4-5).

``serve_with_trace_shutdown`` runs a ``WebsocketPolicyServer`` under the
trace shutdown protocol: SIGTERM / SIGINT and a build-mode writer failure
request a stop instead of killing the process; after the listener closes,
every trace writer is drained within one shared budget and the exit status
reports the outcome (0 only when no episode is unfinished, no write error was
recorded and the drain did not time out; otherwise 3). A watchdog
``os._exit(3)`` bounds a cleanup that hangs in CUDA teardown or a stuck
producer. ``atexit`` stays a best-effort backstop and never sets the status.

Jax-free: imported by ``scripts/serve_policy.py`` and by the GR00T island's
servers alike. Uses the trace writers and model-independent batching core.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

logger = logging.getLogger(__name__)

TRACE_SHUTDOWN_BUDGET_S = 30.0
TRACE_WATCHDOG_S = 60.0
TRACE_EXIT_CODE = 3


def serve_with_trace_shutdown(server, *, build_mode: bool) -> None:
    """Serve ``server`` until a stop is requested, then drain every trace writer.

    CLI ``build_mode`` and effective YAML build writers both enable failure
    monitoring: the first recorded error stops the intake of new episodes
    process-wide, not only on the affected connection (plan §7.4-4).
    """
    from openpi.cache.trace.h5_sink import TraceWriter

    stop_watch = threading.Event()
    shutdown_lock = threading.RLock()
    deadline = None
    watchdog = None
    abort_timer = None

    def _abort_writers() -> None:
        for writer in TraceWriter.all_instances():
            writer.abort("server shutdown deadline exceeded")

    def _arm_shutdown() -> None:
        nonlocal deadline, watchdog, abort_timer
        with shutdown_lock:
            if deadline is not None:
                return
            deadline = time.monotonic() + TRACE_SHUTDOWN_BUDGET_S
            # Start before requesting server shutdown: wait_closed() and the
            # asyncio executor join can both be blocked by an in-flight infer.
            abort_timer = threading.Timer(TRACE_SHUTDOWN_BUDGET_S, _abort_writers)
            abort_timer.daemon = True
            abort_timer.start()
            watchdog = threading.Timer(
                TRACE_WATCHDOG_S,
                lambda: (logger.error("trace shutdown watchdog fired"), os._exit(TRACE_EXIT_CODE)),
            )
            watchdog.daemon = True
            watchdog.start()

    def _request_stop() -> None:
        _arm_shutdown()
        server.request_stop()

    def _on_signal(signum, _frame):
        logger.warning("trace serving: signal %s received; requesting stop", signum)
        _request_stop()

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    def _failure_watch() -> None:
        while not stop_watch.wait(1.0):
            from openpi.serving.batching_core import get_active_coordinator

            coordinator = get_active_coordinator()
            if coordinator is not None and getattr(coordinator, "fatal_error", None) is not None:
                logger.error("trace serving: fatal coordinator failure; requesting stop")
                _request_stop()
                return
            for writer in TraceWriter.all_instances():
                if (build_mode or writer.build_mode) and (writer.errors() or not writer.healthy):
                    logger.error("trace build-cache: writer failure detected; requesting stop")
                    _request_stop()
                    return

    threading.Thread(target=_failure_watch, name="trace-failure-watch", daemon=True).start()

    exit_code = 0
    try:
        server.serve_forever(stop_on_request=True)
    finally:
        _arm_shutdown()
        stop_watch.set()
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        from openpi.serving.batching_core import get_active_coordinator

        coordinator = get_active_coordinator()
        if coordinator is not None and getattr(coordinator, "fatal_error", None) is not None:
            exit_code = TRACE_EXIT_CODE
            _abort_writers()
        threads_alive = False
        for writer in TraceWriter.all_instances():
            remaining = max(0.0, deadline - time.monotonic())
            report = writer.stop(timeout=remaining)
            threads_alive = threads_alive or report.thread_alive
            if not report.ok:
                exit_code = TRACE_EXIT_CODE
                logger.error(
                    "trace writer %s: unfinished=%s errors=%d timed_out=%s fatal=%s",
                    writer.out_dir, list(report.unfinished), len(report.errors), report.timed_out,
                    report.fatal_error,
                )
            else:
                logger.info("trace writer %s: drained cleanly", writer.out_dir)
        abort_timer.cancel()
        if not threads_alive:
            watchdog.cancel()
    if exit_code:
        sys.exit(exit_code)


__all__ = ["TRACE_EXIT_CODE", "TRACE_SHUTDOWN_BUDGET_S", "TRACE_WATCHDOG_S", "serve_with_trace_shutdown"]
