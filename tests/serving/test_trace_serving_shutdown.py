"""Trace shutdown protocol of the serving entry points (plan §7.4-5, §12-G).

``serve_with_trace_shutdown``: a build-mode writer failure requests the stop,
SIGTERM requests the stop, the writers are drained and the exit status is 3
when any episode stayed unfinished or failed (0 otherwise); the watchdog
bounds a hanging drain with ``os._exit(3)``. The subprocess cases run a real
``WebsocketPolicyServer`` so the ``stop_on_request`` loop and the signal
path are the production ones; the default ``serve_forever()`` path stays
untouched (``request_stop`` is a no-op there).
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.serving import trace_serving
from tests.cache.trace.conftest import make_identity, make_plan

REPO = pathlib.Path(__file__).resolve().parents[2]


class _FakeServer:
    def __init__(self):
        self.stopped = threading.Event()
        self.stop_requests = 0

    def serve_forever(self, *, stop_on_request=False):
        assert stop_on_request
        self.stopped.wait(timeout=10)

    def request_stop(self):
        self.stop_requests += 1
        self.stopped.set()


def test_build_mode_writer_failure_requests_stop_and_exits_3(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_serving, "TRACE_SHUTDOWN_BUDGET_S", 5.0)
    plan = make_plan(record_noise_actions=True, save_timesteps=(0.9,), fail_loud=True)
    writer = TraceWriter.get(str(tmp_path), queue_steps=4)
    sink = H5TraceSink(tmp_path, plan=plan, writer=writer)
    sink.on_episode_start(make_identity(1, "e1"))
    # An open episode is left behind on purpose: the watch must fire on the
    # recorded error, and the drain must then report the episode unfinished.
    writer.drain(timeout=5)  # the open job is registered on the writer thread
    (job,) = list(writer._jobs.values())  # noqa: SLF001 - inject a failure on it
    writer._record_error(job, "step", "boom")  # noqa: SLF001
    server = _FakeServer()
    with pytest.raises(SystemExit) as exc:
        trace_serving.serve_with_trace_shutdown(server, build_mode=True)
    assert exc.value.code == 3
    assert server.stop_requests >= 1


def test_clean_drain_exits_0(tmp_path, monkeypatch):
    plan = make_plan()
    writer = TraceWriter.get(str(tmp_path), queue_steps=4)
    sink = H5TraceSink(tmp_path, plan=plan, writer=writer)
    sink.on_episode_start(make_identity(1, "e1"))
    sink.on_episode_end(True)
    server = _FakeServer()
    threading.Timer(0.2, server.request_stop).start()
    trace_serving.serve_with_trace_shutdown(server, build_mode=False)  # returns, no SystemExit
    assert writer.drain(timeout=5).ok


_SUBPROCESS = textwrap.dedent(
    """
    import os, sys, signal, threading, time
    sys.path.insert(0, {repo!r})
    from openpi.serving import trace_serving
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
    from tests.cache.trace.conftest import make_identity, make_plan

    class _Policy:
        def infer(self, obs):
            return {{"actions": [[0.0]]}}
        @property
        def metadata(self):
            return {{}}

    plan = make_plan()
    writer = TraceWriter.get({out!r}, queue_steps=4)
    sink = H5TraceSink({out!r}, plan=plan, writer=writer)
    sink.on_episode_start(make_identity(1, "e1"))
    if {finish!r}:
        sink.on_episode_end(True)
    if {hang!r}:
        trace_serving.TRACE_WATCHDOG_S = 1.0
        trace_serving.TRACE_SHUTDOWN_BUDGET_S = 0.5
        real_stop = writer.stop
        def hanging_stop(*a, **k):
            time.sleep(30)
            return real_stop(*a, **k)
        writer.stop = hanging_stop
    server = WebsocketPolicyServer(_Policy(), host="127.0.0.1", port=0)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    trace_serving.serve_with_trace_shutdown(server, build_mode=False)
    print("EXIT-CLEAN", flush=True)
    """
)


def _run(tmp_path, *, finish: bool, hang: bool = False):
    script = _SUBPROCESS.format(repo=str(REPO), out=str(tmp_path / "out"), finish=finish, hang=hang)
    env = dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env)


def test_sigterm_with_unfinished_episode_exits_3(tmp_path):
    proc = _run(tmp_path, finish=False)
    assert proc.returncode == 3, proc.stderr[-2000:]
    assert "EXIT-CLEAN" not in proc.stdout


def test_sigterm_after_clean_close_exits_0(tmp_path):
    proc = _run(tmp_path, finish=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "EXIT-CLEAN" in proc.stdout


def test_watchdog_bounds_a_hanging_drain(tmp_path):
    t0 = time.monotonic()
    proc = _run(tmp_path, finish=True, hang=True)
    assert proc.returncode == 3, proc.stderr[-2000:]
    assert time.monotonic() - t0 < 60
    assert "watchdog" in proc.stderr
