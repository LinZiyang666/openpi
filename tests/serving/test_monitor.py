"""Unit tests for the serving-side monitor master switch + recorder.

Coverage:
* ``MonitorLevel.from_str`` parses names + numeric forms + falls back.
* ``OPENPI_MONITOR_LEVEL`` env var is read once via ``get_monitor_level``.
* ``MetricsRecorder`` ring buffers FIFO-evict + ``dropped_totals`` updates.
* ``MetricsRecorder.dump_all`` snapshots without clearing.
* ``SystemTimer`` is forced no-op when level == OFF, regardless of
  constructor ``enabled=True``.
* ``SystemTimer.on_task_end`` pushes records into the recorder.
* ``take_memory_snapshot`` is rejected below SNAPSHOT level.
* ``enable_memory_history`` is rejected below HISTORY level.
"""

from __future__ import annotations

import pytest

from openpi.serving import monitor as _monitor


@pytest.fixture(autouse=True)
def _reset_recorder():
    """Clear the module-level recorder/level cache before each test."""
    _monitor.reset_recorder_for_tests()
    _monitor._LEVEL = None  # type: ignore[attr-defined]
    yield
    _monitor.reset_recorder_for_tests()
    _monitor._LEVEL = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MonitorLevel parsing
# ---------------------------------------------------------------------------


def test_level_from_str_names():
    assert _monitor.MonitorLevel.from_str("off") == _monitor.MonitorLevel.OFF
    assert _monitor.MonitorLevel.from_str("BASIC") == _monitor.MonitorLevel.BASIC
    assert _monitor.MonitorLevel.from_str("snapshot") == _monitor.MonitorLevel.SNAPSHOT
    assert _monitor.MonitorLevel.from_str("History") == _monitor.MonitorLevel.HISTORY


def test_level_from_str_numeric():
    assert _monitor.MonitorLevel.from_str("0") == _monitor.MonitorLevel.OFF
    assert _monitor.MonitorLevel.from_str("3") == _monitor.MonitorLevel.HISTORY


def test_level_from_str_unknown_falls_back_to_off():
    # Default is OFF: empty/unset/unrecognised must record nothing, so the
    # production serving path stays at zero instrumentation overhead unless
    # OPENPI_MONITOR_LEVEL is explicitly set.
    assert _monitor.MonitorLevel.from_str("garbage") == _monitor.MonitorLevel.OFF
    assert _monitor.MonitorLevel.from_str(None) == _monitor.MonitorLevel.OFF
    assert _monitor.MonitorLevel.from_str("") == _monitor.MonitorLevel.OFF


def test_get_monitor_level_reads_env(monkeypatch):
    monkeypatch.setenv("OPENPI_MONITOR_LEVEL", "snapshot")
    assert _monitor.get_monitor_level() == _monitor.MonitorLevel.SNAPSHOT


def test_get_monitor_level_default_when_unset(monkeypatch):
    monkeypatch.delenv("OPENPI_MONITOR_LEVEL", raising=False)
    assert _monitor.get_monitor_level() == _monitor.MonitorLevel.OFF


def test_set_monitor_level_overrides():
    _monitor.set_monitor_level(_monitor.MonitorLevel.OFF)
    assert _monitor.get_monitor_level() == _monitor.MonitorLevel.OFF


# ---------------------------------------------------------------------------
# MetricsRecorder ring buffers
# ---------------------------------------------------------------------------


def test_recorder_round_trips_event_kinds():
    rec = _monitor.MetricsRecorder()
    rec.record_util({"cpu_proc_pct": 12.5})
    rec.record_batch({"stage": 1, "size": 4})
    rec.record_timing_bulk([{"name": "stage1_vision", "elapsed_ms": 5.0}])
    out = rec.dump_all()
    assert len(out["util"]) == 1 and out["util"][0]["cpu_proc_pct"] == 12.5
    assert len(out["batch"]) == 1 and out["batch"][0]["stage"] == 1
    assert len(out["timing"]) == 1 and out["timing"][0]["elapsed_ms"] == 5.0


def test_recorder_ring_buffer_drops_oldest():
    rec = _monitor.MetricsRecorder(util_capacity=3)
    for i in range(10):
        rec.record_util({"i": i})
    out = rec.dump_all()
    assert [e["i"] for e in out["util"]] == [7, 8, 9]
    assert out["dropped_totals"]["util"] == 7


def test_recorder_dump_does_not_clear():
    rec = _monitor.MetricsRecorder()
    rec.record_util({"x": 1})
    rec.dump_all()
    rec.dump_all()
    out = rec.dump_all()
    assert len(out["util"]) == 1  # still there


def test_recorder_clear_drops_selected():
    rec = _monitor.MetricsRecorder()
    rec.record_util({"x": 1})
    rec.record_batch({"y": 2})
    cleared = rec.clear(util=True)
    assert cleared == {"util": 1}
    out = rec.dump_all()
    assert out["util"] == [] and len(out["batch"]) == 1


def test_recorder_stats_shape():
    rec = _monitor.MetricsRecorder(util_capacity=5)
    rec.record_util({"x": 1})
    stats = rec.stats()
    assert stats["lengths"]["util"] == 1
    assert stats["capacities"]["util"] == 5
    assert stats["appended_totals"]["util"] == 1


def test_get_recorder_is_singleton():
    a = _monitor.get_recorder()
    b = _monitor.get_recorder()
    assert a is b


# ---------------------------------------------------------------------------
# Memory snapshot / history gating
# ---------------------------------------------------------------------------


def test_take_memory_snapshot_refused_below_snapshot_level():
    _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)
    result = _monitor.take_memory_snapshot("attempt")
    assert result["ok"] is False
    assert "SNAPSHOT" in result["reason"]


def test_enable_memory_history_refused_below_history_level():
    _monitor.set_monitor_level(_monitor.MonitorLevel.SNAPSHOT)
    result = _monitor.enable_memory_history()
    assert result["ok"] is False
    assert "HISTORY" in result["reason"]


def test_disable_memory_history_idempotent():
    # Calling stop when already off is safe.
    result = _monitor.disable_memory_history()
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# SystemTimer master-switch behaviour
# ---------------------------------------------------------------------------


def test_system_timer_forced_off_when_level_off():
    _monitor.set_monitor_level(_monitor.MonitorLevel.OFF)
    from openpi.cache.timing import SystemTimer

    t = SystemTimer(enabled=True)  # explicit True must NOT override OFF level
    assert t._enabled is False  # noqa: SLF001
    t.register_probe("p", backend="cpu")
    with t.measure("p"):
        pass
    # No record appended because measure() is a no-op when disabled.
    assert len(list(t._records)) == 0  # noqa: SLF001


def test_system_timer_pushes_to_recorder_on_task_end():
    _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)
    from openpi.cache.timing import SystemTimer

    t = SystemTimer(enabled=True, quiet=True)
    t._resource_monitors = []  # noqa: SLF001 — isolate from real CPU/GPU monitors
    t.register_probe("p", backend="cpu")
    t.on_task_begin()
    with t.measure("p"):
        pass
    t.on_task_end()
    rec = _monitor.get_recorder()
    out = rec.dump_all()
    rows = [r for r in out["timing"] if r["name"] == "p"]
    assert len(rows) == 1
    assert rows[0]["elapsed_ms"] >= 0.0


# --- G2 R1 fix: throughput_summary must not fabricate a rate from 1 sample ---


def test_throughput_summary_insufficient_window_single_sample():
    # One util sample + one batch at the same instant: the elapsed span is ~0.
    # Must report insufficient_window, NOT n_inf / 1e-6 (~10,000,000 inf/s).
    rec = _monitor.MetricsRecorder()
    rec.record_util({"gpu_util_pct": 50, "ts_epoch": 1000.0})
    rec.record_batch({"stage": 1, "size": 10, "ts_epoch": 1000.0})
    s = rec.throughput_summary()
    assert s.get("insufficient_window") is True
    assert "stage1_throughput_inf_per_s" not in s


def test_throughput_summary_two_samples_gives_finite_rate():
    rec = _monitor.MetricsRecorder()
    rec.record_util({"gpu_util_pct": 50, "ts_epoch": 1000.0})
    rec.record_util({"gpu_util_pct": 60, "ts_epoch": 1001.0})
    rec.record_batch({"stage": 1, "size": 10, "ts_epoch": 1000.5})
    s = rec.throughput_summary()
    assert s["elapsed_s"] == 1.0
    assert s["stage1_throughput_inf_per_s"] == 10.0
