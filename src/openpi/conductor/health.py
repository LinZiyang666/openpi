"""Per-worker health aggregation (plan §9.2, §10).

Tracks each worker's last-seen timestamp, state, current task, and a sliding-
window throughput, from the progress/result/heartbeat stream the driver feeds.
Surfaces stale (silent) workers so the supervisor can act. This is observation
only — requeue/restart decisions live in the driver/agent.

Coupling map:
  DEPENDS ON:  standard library (threading, time, collections)
  CONSUMED BY: monitor.py, driver.py (optional)
"""

from __future__ import annotations

import collections
import threading
import time


class _WorkerHealth:
    def __init__(self) -> None:
        self.state = "idle"
        self.last_ts = time.monotonic()
        self.current_task: str | None = None
        self.last_step = 0
        self.actions_per_s = 0.0
        self.completed = 0


class HealthAggregator:
    """Thread-safe per-worker health view."""

    def __init__(self, *, stale_after_s: float = 120.0) -> None:
        self._stale_after = stale_after_s
        self._lock = threading.Lock()
        self._workers: dict[str, _WorkerHealth] = collections.defaultdict(_WorkerHealth)

    def on_progress(self, payload: dict) -> None:
        wid = payload.get("worker_id", "")
        with self._lock:
            w = self._workers[wid]
            w.state = "running"
            w.last_ts = time.monotonic()
            w.current_task = payload.get("task_uid")
            w.last_step = int(payload.get("step", 0))
            w.actions_per_s = float(payload.get("actions_per_s", 0.0))

    def on_result(self, task_uid: str, *, success: bool) -> None:
        # Result frames carry no worker_id; attribute to the worker whose
        # current_task matches (set on the last progress frame). If the episode
        # ran without ever emitting progress, current_task stays None and that
        # worker's completed count is not incremented (best-effort observability).
        with self._lock:
            for w in self._workers.values():
                if w.current_task == task_uid:
                    w.state = "idle"
                    w.last_ts = time.monotonic()
                    w.current_task = None
                    if success:
                        w.completed += 1
                    return

    def stale_workers(self, *, now: float | None = None) -> list[str]:
        """Workers silent for longer than ``stale_after_s`` (likely dead/hung).

        ``last_ts`` is refreshed by progress/result only (this build has no
        separate heartbeat wire — a worker process crash is caught by the
        driver's connection-drop requeue, plan §9.2; episode wall-clock timeout
        is a follow-up). A worker mid-infer that emits no progress can therefore
        read as stale; treat this as a best-effort observability signal.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            return [wid for wid, w in self._workers.items() if now - w.last_ts > self._stale_after]

    def snapshot(self) -> dict[str, dict]:
        """Per-worker view: {worker_id: {state, current_task, step, actions_per_s, completed}}."""
        with self._lock:
            return {
                wid: {
                    "state": w.state,
                    "current_task": w.current_task,
                    "step": w.last_step,
                    "actions_per_s": round(w.actions_per_s, 2),
                    "completed": w.completed,
                }
                for wid, w in self._workers.items()
            }
