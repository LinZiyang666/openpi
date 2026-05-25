"""Unit tests for ResourceMonitor implementation in SystemTimer.

Coverage:
* ``CpuMonitor.sample()`` returns expected keys when psutil present, ``{}`` otherwise.
* ``GpuMonitor.sample()`` returns expected keys when pynvml + GPU present, ``{}`` otherwise.
* ``SystemTimer(enabled=True)`` auto-registers Cpu + Gpu monitors at SNAPSHOT+.
* At BASIC the timer records timing rows but does NOT auto-register per-probe
  resource monitors (no psutil/NVML sampling on the hot path).
* ``SystemTimer(enabled=False)`` registers nothing (zero-overhead contract).
* ``record_resource_snapshot`` merges monitor samples into the latest matching record.
* ``measure()`` finally-block auto-triggers ``record_resource_snapshot`` when monitors exist.
* Monitor ``sample()`` raising an exception does not crash the timer (caught + skipped).
* Multiple monitors namespaced by ``.name`` prefix so keys do not collide.
"""

from __future__ import annotations

from openpi.cache.timing import CpuMonitor
from openpi.cache.timing import GpuMonitor
from openpi.cache.timing import ResourceMonitor
from openpi.cache.timing import SystemTimer
from openpi.serving import monitor as _monitor


def test_cpu_monitor_returns_keys_or_empty():
    m = CpuMonitor()
    snap = m.sample()
    # Either psutil is installed (uv venv ships it) so we get both keys,
    # or it isn't and we get an empty dict.
    if snap:
        assert set(snap.keys()) == {"proc_pct", "rss_mb"}
        assert snap["rss_mb"] >= 0
    else:
        assert snap == {}


def test_gpu_monitor_returns_keys_or_empty():
    m = GpuMonitor()
    snap = m.sample()
    if snap:
        assert set(snap.keys()) == {"util_pct", "mem_used_mb"}
        assert 0 <= snap["util_pct"] <= 100
        assert snap["mem_used_mb"] >= 0
    else:
        assert snap == {}


def test_enabled_timer_auto_registers_monitors():
    # Auto-registration is a SNAPSHOT+ feature (per-probe psutil/NVML sampling).
    _monitor.set_monitor_level(_monitor.MonitorLevel.SNAPSHOT)
    try:
        t = SystemTimer(enabled=True)
        names = {getattr(m, "name", "") for m in t._resource_monitors}
        assert "cpu" in names
        assert "gpu" in names
    finally:
        _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)


def test_basic_level_skips_resource_monitors():
    # At BASIC the timer still records timing rows, but must NOT auto-register
    # per-probe resource monitors (keeps the hot path free of psutil/NVML).
    _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)
    t = SystemTimer(enabled=True)
    assert t._resource_monitors == []


def test_disabled_timer_skips_monitor_registration():
    t = SystemTimer(enabled=False)
    assert t._resource_monitors == []


def test_record_resource_snapshot_attaches_to_latest_matching():
    class FakeMonitor:
        name = "fake"

        def sample(self):
            return {"k1": 1.5, "k2": 7.0}

    t = SystemTimer(enabled=True)
    t._resource_monitors = [FakeMonitor()]  # replace auto-registered ones for isolation
    t.register_probe("probe_a", backend="cpu")
    with t.measure("probe_a"):
        pass

    rec = list(t._records)[-1]
    assert rec.name == "probe_a"
    assert rec.resources is not None
    assert rec.resources["fake.k1"] == 1.5
    assert rec.resources["fake.k2"] == 7.0


def test_measure_auto_triggers_snapshot_only_when_monitors_present():
    t = SystemTimer(enabled=True)
    t._resource_monitors = []  # explicitly no monitors
    t.register_probe("probe_b", backend="cpu")
    with t.measure("probe_b"):
        pass
    rec = list(t._records)[-1]
    # No monitors -> no resources attached.
    assert rec.resources is None


def test_failing_monitor_does_not_crash_timer():
    class BrokenMonitor:
        name = "broken"

        def sample(self):
            raise RuntimeError("oops")

    t = SystemTimer(enabled=True)
    t._resource_monitors = [BrokenMonitor()]
    t.register_probe("probe_c", backend="cpu")
    with t.measure("probe_c"):
        pass
    rec = list(t._records)[-1]
    # Broken monitor produces no entries but does not crash measure().
    assert rec.resources is None or "broken" not in str(rec.resources)


def test_multiple_monitors_namespaced():
    class M1:
        name = "alpha"

        def sample(self):
            return {"x": 1.0}

    class M2:
        name = "beta"

        def sample(self):
            return {"x": 2.0}

    t = SystemTimer(enabled=True)
    t._resource_monitors = [M1(), M2()]
    t.register_probe("probe_d", backend="cpu")
    with t.measure("probe_d"):
        pass
    rec = list(t._records)[-1]
    assert rec.resources is not None
    assert rec.resources["alpha.x"] == 1.0
    assert rec.resources["beta.x"] == 2.0


def test_cpu_monitor_protocol_compliance():
    m = CpuMonitor()
    assert isinstance(m, ResourceMonitor)


def test_gpu_monitor_protocol_compliance():
    m = GpuMonitor()
    assert isinstance(m, ResourceMonitor)
