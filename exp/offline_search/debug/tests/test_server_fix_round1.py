"""Review-1 regressions: durability, identity, startup and targeted taps."""
import copy
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug import schema
from exp.offline_search.debug.server import ServerObserver
from exp.offline_search.debug.server.diagnostics import metric_tap
from exp.offline_search.debug.tests.test_server_observer import setup
from exp.offline_search.debug.tests.test_server_support import no_git, server_json, server_rows


@pytest.mark.parametrize("extras, expected", [
    ({"os_sf_unanimous": 1, "os_sf_extension": 0}, "cache_tail"),
    ({"os_sf_valve": .2}, "cache_tail"),
    ({"os_sf_extension": 2}, "cache_tail"),
    ({"os_sf_extension": 1}, "follow"),
    ({"os_sf_source": 1}, "follow"),
])
@pytest.mark.parametrize("source", ["cache", "cache_blind"])
def test_sf_tail_is_follow_only_for_an_actual_extension(tmp_path, extras, expected, source):
    rt, session, obs = setup(tmp_path)
    session._dec.update(vision=False, hit=True, source=source, extras=extras)
    capture = rt.debug.begin(obs, obs["__debug__"])
    rt.debug.finish(capture, session, {"actions": np.zeros((6, 2), np.float64)}, 0.)
    rt.debug.close()
    record, = server_rows(tmp_path)
    assert record["src"] == expected and record["status"] == "available"


def test_subset_init_and_original_pool_index_are_distinct(tmp_path):
    rt, session, obs = setup(tmp_path)
    session.ep_meta["init"] = 77
    capture = rt.debug.begin(obs, obs["__debug__"])
    rt.debug.finish(capture, session, {"actions": np.zeros((6, 2), np.float64)}, 0.)
    rt.debug.close()
    record, = server_rows(tmp_path)
    assert record["init"] == 20 and record["orig_init_state_idx"] == 77


def test_served_wire_keeps_float64_bits(tmp_path):
    rt, session, obs = setup(tmp_path)
    wire = np.full((6, 2), 1. + 2. ** -35, np.float64)
    assert wire.astype(np.float32).astype(np.float64).tobytes() != wire.tobytes()
    capture = rt.debug.begin(obs, obs["__debug__"])
    rt.debug.finish(capture, session, {"actions": wire}, 0.)
    rt.debug.close()
    record, = server_rows(tmp_path)
    data = schema.read_npz_block(tmp_path / "blocks" / record["blk"])
    assert data["served_wire"].dtype == np.dtype("<f8")
    assert data["served_wire"][0].tobytes() == wire.tobytes()


def test_required_debug_startup_is_fatal_and_off_never_constructs_observer(tmp_path, monkeypatch):
    from exp.offline_search.harness import dims, store
    import exp.offline_search.debug.server as server
    lib = types.SimpleNamespace(L=1, ids=["fixture"],
        action=np.zeros((1, dims.HORIZON["pi05"], dims.ACT_FULL_DIMS), np.float32),
        meta=dict(task_map={"fixture task": 0}), dir=tmp_path / "library/current")
    monkeypatch.setattr(store, "LibraryView", lambda *args: lib)
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError("cannot create capture directory")
    monkeypatch.setattr(server, "ServerObserver", fail)
    argv = ["--os-method", "native", "--os-cell", "pi05_spatial_cache", "--os-root", str(tmp_path),
            "--os-log-dir", str(tmp_path / "legacy")]
    opts, _ = plugin.parse_cli(argv)
    assert plugin.PluginRuntime(opts, "pi05").debug is None and not calls
    opts, _ = plugin.parse_cli(argv + ["--os-debug-dir", str(tmp_path / "debug")])
    with pytest.raises(RuntimeError, match="required debug observer startup failed") as failure:
        plugin.PluginRuntime(opts, "pi05")
    assert isinstance(failure.value.__cause__, OSError) and len(calls) == 1


def test_metric_wrappers_preserve_facade_and_existing_override_without_profiler(monkeypatch):
    class Metric:
        def __init__(self):
            self.factor = 1
            self.B0T = np.zeros((2, 3), np.float32)
            self.calls = 0
        def _dist(self, query):
            self.calls += 1
            xv = np.arange(4, dtype=np.float32) * self.factor
            return (None,) * 5 + (xv, np.array([self.factor], np.float32)) + (None,) * 4
        def _mix(self, rows, w):
            return w / w.sum(), rows, object()
    base, other = Metric(), Metric()
    original_override = types.MethodType(Metric._mix, base)
    base._mix = original_override
    session = types.SimpleNamespace(method=types.SimpleNamespace(base=base))
    target, profile = {}, sys.getprofile()
    monkeypatch.setattr(sys, "setprofile", lambda *args: pytest.fail("no profiler may be installed"))
    with metric_tap(session, target):
        assert sys.getprofile() is profile
        assert "_dist" not in vars(other) and "_mix" not in vars(other)
        original = base._dist
        facade = copy.copy(base)
        facade.factor = 7
        result = original.__func__(facade, object())  # CU/CT's actual saved-bound-method contract
        mixed = base._mix(rows=np.array([2, 8]), w=np.array([1., 3.]))
        assert facade.calls == 1 and base.calls == 0
        np.testing.assert_array_equal(result[5], np.array([0, 7, 14, 21], np.float32))
        assert sys.getprofile() is profile
    assert "_dist" not in vars(base) and base._mix is original_override
    np.testing.assert_array_equal(target["keys_pca_third"], [0, 7])
    np.testing.assert_array_equal(target["keys_pca_wrist"], [14, 21])
    np.testing.assert_array_equal(target["metric_state"], [7])
    np.testing.assert_array_equal(target["weights"], mixed[0])
    np.testing.assert_array_equal(target["rows"], [2, 8])
    with pytest.raises(RuntimeError, match="intentional"):
        with metric_tap(session, {}):
            raise RuntimeError("intentional")
    assert "_dist" not in vars(base) and base._mix is original_override


def test_metric_copy_error_never_changes_result_or_adds_calls(monkeypatch):
    import exp.offline_search.debug.server.diagnostics as diagnostics
    class Metric:
        def _mix(self, rows, w):
            self.calls += 1
            return self.result
    base = Metric()
    base.calls, base.result = 0, (np.ones(1), None, None)
    def fail(*args):
        raise OSError("copy unavailable")
    monkeypatch.setattr(diagnostics, "detached", fail)
    target = {}
    with metric_tap(types.SimpleNamespace(method=base), target):
        assert base._mix(np.array([0]), np.array([1])) is base.result
    assert base.calls == 1 and "copy unavailable" in target["tap_error"]


@pytest.mark.parametrize("tail", [b'{"partial":', b'{"complete":true}', b'{"utf8":"\xe2\x82'])
def test_reader_reports_final_unterminated_line_and_reads_legacy_and_pid_files(tmp_path, tail, caplog):
    (tmp_path / "decisions.jsonl").write_bytes(b'{"legacy":1}\n' + tail)
    (tmp_path / "decisions_123.jsonl").write_bytes(b'{"new":2}\n')
    issues = []
    assert server_rows(tmp_path, issues) == [dict(legacy=1), dict(new=2)]
    assert len(issues) == 1 and issues[0]["line"] == 2 and issues[0]["bytes"] == len(tail)
    assert "unterminated final JSONL line skipped" in caplog.text
    (tmp_path / "decisions_123.jsonl").write_bytes(b'{"bad":}\n')
    with pytest.raises(ValueError):
        server_rows(tmp_path)


def test_episode_ack_survives_hard_kill_and_restart_keeps_process_artifacts(tmp_path):
    code = '''
import signal, sys, types
import numpy as np
from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug.server import ServerObserver
import exp.offline_search.debug.server.manifest as manifest
manifest.git_provenance = lambda: {}
r = types.SimpleNamespace(model="generic", suite="fake", H=2, method=None, tables={}, r4=False, gpu=None,
    opts=types.SimpleNamespace(os_method="fake", kwargs={}, os_fit_artifact=""))
r.debug = ServerObserver(r, sys.argv[1], {"campaign":"crash"})
r.debug.writer.flush_seconds = 60
s = types.SimpleNamespace(rt=r, ep=None, method=None)
s.client_episode_end = lambda success: plugin.PluginSession.finish_episode(s, "episode_end", success)
inner = types.SimpleNamespace(on_episode_end=lambda **kwargs: None)
c = plugin._ConnPolicy(inner, [s], "default")
for i in range(13):
    r.debug.writer.put({"decision_id":"accepted:"+str(i)}, {"state_wire":np.array([i], np.float64)})
c.on_episode_end(success=True)
print("episode_ack", flush=True)
for i in range(2):
    r.debug.writer.put({"decision_id":"inflight:"+str(i)}, {"state_wire":np.array([i], np.float64)})
print("inflight", flush=True)
signal.pause()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "episode_ack"
        assert child.stdout.readline().strip() == "inflight"
        child.send_signal(signal.SIGKILL)  # Only this test's owned child; no serving process or socket exists.
        _, error = child.communicate(timeout=10)
        assert child.returncode == -signal.SIGKILL, error
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    assert [r["decision_id"] for r in server_rows(tmp_path)] == ["accepted:" + str(i) for i in range(13)]
    old_meta = tmp_path / "meta_{}.json".format(child.pid)
    old_stats = tmp_path / "writer_stats_{}.json".format(child.pid)
    old_log = tmp_path / "decisions_{}.jsonl".format(child.pid)
    old_log.write_bytes(old_log.read_bytes() + b'{"killed":')
    before = {p: p.read_bytes() for p in (old_meta, old_stats, old_log)}
    rt = types.SimpleNamespace(model="generic", suite="fake", H=2, method=None, tables={},
                              opts=types.SimpleNamespace(os_method="fake", kwargs={}, os_fit_artifact=""))
    observer = ServerObserver(rt, tmp_path, {"campaign":"restart"})
    observer.writer.put({"decision_id":"restarted:0"}, {"state_wire":np.array([0.], np.float64)})
    observer.close()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (tmp_path / "meta_{}.json".format(os.getpid())).exists()
    assert len(list(tmp_path.glob("meta*.json"))) == len(list(tmp_path.glob("writer_stats*.json"))) == 2
    issues = []
    records = server_rows(tmp_path, issues)
    assert len(records) == 14 and len(issues) == 1
    assert {r["decision_id"] for r in records} == {"accepted:" + str(i) for i in range(13)} | {"restarted:0"}
    assert all((tmp_path / "blocks" / r["blk"]).exists() for r in records)
