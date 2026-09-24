"""Writer failure protocol (plan §7.4, §12-G).

Every failure point of an episode -- last Step, terminal attrs, fsync,
sidecar publish, H5 rename, failure log unwritable, reservation conflict,
pre-existing final file -- must leave no committed ``.h5``, record the error,
make the build-mode sink sticky, and keep the writer alive for the other
connections. ``drain`` / ``stop`` report unfinished episodes and errors.
"""

from __future__ import annotations

import os

import h5py
import pytest

from openpi.cache.trace import h5_sink as sink_mod
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TraceWriteFailed
from tests.cache.trace.conftest import make_identity, make_plan, make_step


def _writer(out_dir, queue_steps=4) -> TraceWriter:
    return TraceWriter(str(out_dir), queue_steps=queue_steps)


def _run_episode(sink, n_steps: int, name: str, episode_id: int = 1, success: bool = True):
    """Drive one episode; a sticky build-mode failure mid-episode is reported
    back as the return value instead of aborting the test."""
    sink.on_episode_start(make_identity(episode_id, name))
    raised = None
    for i in range(n_steps):
        sink.begin_step()
        try:
            sink.record_step(make_step(i))
        except TraceWriteFailed as exc:
            raised = exc
            sink.finish_step()
            break
        sink.finish_step()
    sink.on_episode_end(success)
    return raised


def _assert_failed(out_dir, name: str, report, stage: str):
    assert not (out_dir / "exp" / f"{name}.h5").exists()
    assert not (out_dir / "exp" / f"{name}.h5.reserved").exists()
    assert any(e.stage == stage for e in report.errors), report.errors
    failures = out_dir / "_trace_failures.jsonl"
    assert failures.exists() and name in failures.read_text()


def test_last_step_write_failure(out_dir, monkeypatch):
    w = _writer(out_dir)
    calls = {"n": 0}
    original = sink_mod.write_step_group

    def flaky(grp, embs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        original(grp, embs)

    monkeypatch.setattr(sink_mod, "write_step_group", flaky)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 3, "s1")
    report = w.drain(timeout=10)
    _assert_failed(out_dir, "s1", report, "step")
    assert (out_dir / "exp" / "s1.h5.failed").exists()
    assert w.healthy
    # Sticky: the next episode of this connection is refused in build mode.
    with pytest.raises(TraceWriteFailed):
        sink.on_episode_start(make_identity(2, "s2"))
    assert sink.sticky_error() is not None


def test_close_rename_failure(out_dir, monkeypatch):
    w = _writer(out_dir)
    real_replace = os.replace

    def bad_replace(src, dst):
        if str(dst).endswith("c1.h5"):
            raise OSError("rename refused")
        return real_replace(src, dst)

    monkeypatch.setattr(sink_mod.os, "replace", bad_replace)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 2, "c1")
    report = w.drain(timeout=10)
    _assert_failed(out_dir, "c1", report, "close")
    assert w.healthy
    with pytest.raises(TraceWriteFailed):
        sink.begin_step()
        sink.record_step(make_step(9))


def test_fsync_failure_is_a_close_failure(out_dir, monkeypatch):
    w = _writer(out_dir)

    def bad_fsync(path):
        raise OSError("fsync failed")

    monkeypatch.setattr(sink_mod, "_fsync_path", bad_fsync)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 1, "f1")
    report = w.drain(timeout=10)
    _assert_failed(out_dir, "f1", report, "close")


def test_sidecar_publish_failure_blocks_h5_commit(out_dir, monkeypatch):
    w = _writer(out_dir)
    real_replace = os.replace

    def bad_replace(src, dst):
        if str(dst).endswith(".trace.jsonl"):
            raise OSError("sidecar rename refused")
        return real_replace(src, dst)

    monkeypatch.setattr(sink_mod.os, "replace", bad_replace)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 1, "sc1")
    report = w.drain(timeout=10)
    _assert_failed(out_dir, "sc1", report, "close")
    assert not (out_dir / "exp" / "sc1.trace.jsonl").exists()


def test_failure_log_unwritable_still_records_error(out_dir, monkeypatch):
    w = _writer(out_dir)
    real_open = open

    def bad_open(path, *args, **kwargs):
        if str(path).endswith("_trace_failures.jsonl"):
            raise OSError("log unwritable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", bad_open)
    original = sink_mod.write_step_group

    def boom(grp, embs):
        raise OSError("boom")

    monkeypatch.setattr(sink_mod, "write_step_group", boom)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 1, "l1")
    report = w.drain(timeout=10)
    assert not (out_dir / "exp" / "l1.h5").exists()
    assert any(e.stage == "step" for e in report.errors)
    assert sink.sticky_error() is not None
    monkeypatch.setattr(sink_mod, "write_step_group", original)


def test_diagnostic_mode_records_but_does_not_raise(out_dir, monkeypatch):
    w = _writer(out_dir)

    def boom(grp, embs):
        raise OSError("boom")

    monkeypatch.setattr(sink_mod, "write_step_group", boom)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=False), writer=w)
    _run_episode(sink, 1, "d1")
    w.drain(timeout=10)
    # Diagnostic: the next episode proceeds; the failed one is still unpublished.
    sink.on_episode_start(make_identity(2, "d2"))
    sink.record_step(make_step(0))
    sink.on_episode_end(True)
    w.drain(timeout=10)
    assert not (out_dir / "exp" / "d1.h5").exists()
    assert sink.sticky_error() is not None


def test_reservation_conflict_refuses_second_writer(out_dir):
    w = _writer(out_dir)
    (out_dir / "exp").mkdir(parents=True)
    reserved = out_dir / "exp" / "r1.h5.reserved"
    reserved.write_text('{"token": "other", "pid": 1}')
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    raised = _run_episode(sink, 1, "r1")
    report = w.drain(timeout=10)
    # Build mode: the open failure surfaces on the first record_step already.
    assert raised is not None or sink.sticky_error() is not None
    assert not (out_dir / "exp" / "r1.h5").exists()
    assert any(e.stage == "open" and "reserved" in e.message for e in report.errors)
    # The foreign reservation is not ours to remove.
    assert reserved.exists()


def test_existing_final_file_is_never_overwritten(out_dir):
    w = _writer(out_dir)
    sink = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink, 1, "x1")
    assert w.drain(timeout=10).ok
    before = (out_dir / "exp" / "x1.h5").read_bytes()
    sink2 = H5TraceSink(out_dir, plan=make_plan(fail_loud=True), writer=w)
    _run_episode(sink2, 2, "x1", episode_id=9)
    report = w.drain(timeout=10)
    assert sink2.sticky_error() is not None
    assert (out_dir / "exp" / "x1.h5").read_bytes() == before
    assert any(e.stage == "open" and "already exists" in e.message for e in report.errors)


def test_drain_and_stop_report_unfinished_episode(out_dir):
    w = _writer(out_dir)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(1, "u1"))
    sink.record_step(make_step(0))
    report = w.drain(timeout=5)
    assert not report.timed_out
    assert len(report.unfinished) == 1
    assert not report.ok
    stopped = w.stop(timeout=5)
    assert not stopped.thread_alive
    assert not (out_dir / "exp" / "u1.h5").exists()
    assert any(e.stage == "stop" for e in stopped.errors)
    with pytest.raises(TraceWriteFailed):
        sink.on_episode_start(make_identity(2, "u2"))


def test_drain_timeout_is_reported(out_dir, monkeypatch):
    w = _writer(out_dir, queue_steps=4)
    import threading

    gate = threading.Event()
    original = w._handle_step

    def slow(*args, **kwargs):
        gate.wait(5.0)
        original(*args, **kwargs)

    monkeypatch.setattr(w, "_handle_step", slow)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(1, "t1"))
    sink.record_step(make_step(0))
    report = w.drain(timeout=0.3)
    assert report.timed_out and not report.ok
    gate.set()
    assert w.drain(timeout=10).unfinished == (sink._token,)
    sink.on_episode_end(True)
    assert w.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "t1.h5", "r") as f:
        assert f.attrs["num_steps"] == 1
