"""Health aggregation + monitor rendering tests (M5)."""

from __future__ import annotations

import time

from openpi.conductor.health import HealthAggregator
from openpi.conductor.monitor import Monitor


def test_health_progress_then_result():
    h = HealthAggregator()
    h.on_progress({"worker_id": "w0", "task_uid": "u", "step": 3, "actions_per_s": 5.0})
    snap = h.snapshot()
    assert snap["w0"]["state"] == "running"
    assert snap["w0"]["step"] == 3
    h.on_result("u", success=True)
    snap = h.snapshot()
    assert snap["w0"]["state"] == "idle"
    assert snap["w0"]["current_task"] is None
    assert snap["w0"]["completed"] == 1


def test_health_stale_detection():
    h = HealthAggregator(stale_after_s=10.0)
    h.on_progress({"worker_id": "w0", "task_uid": "u", "step": 0, "actions_per_s": 1.0})
    now = time.monotonic()
    assert h.stale_workers(now=now + 100.0) == ["w0"]
    assert h.stale_workers(now=now) == []


def test_health_on_result_unknown_task_is_noop():
    h = HealthAggregator()
    h.on_progress({"worker_id": "w0", "task_uid": "u", "step": 1, "actions_per_s": 1.0})
    h.on_result("other", success=True)  # no worker has current_task=="other"
    assert h.snapshot()["w0"]["completed"] == 0


class _FakeSched:
    def progress(self):
        return {"episodes_done": 2, "episodes_failed": 1, "episodes_total": 4}


class _Result:
    def __init__(self, uid, ok):
        self.task_uid = uid
        self.success = ok


def test_monitor_render_aggregate():
    m = Monitor(scheduler=_FakeSched())
    m.on_progress({"worker_id": "w0", "task_uid": "u", "step": 2, "actions_per_s": 4.0})
    m.on_result(_Result("u", ok=True))
    out = m.render()
    assert "2/4" in out
    assert "workers:" in out


def test_monitor_no_scheduler_still_renders():
    m = Monitor()
    m.on_progress({"worker_id": "w0", "task_uid": "u", "step": 0, "actions_per_s": 0.0})
    assert "workers:" in m.render()
