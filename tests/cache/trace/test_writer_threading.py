"""TraceWriter / H5TraceSink threading and ordering contract (plan §7.1, §12-G).

Covers: Step back-pressure blocks only the producer, Open / Close never take a
permit, FIFO order (a Close never overtakes an earlier Step), two connections
interleaving without cross-talk, unknown / out-of-order tokens, and the
in-flight-step assertion on Close.
"""

from __future__ import annotations

import threading
import time

import h5py
import pytest

from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from tests.cache.trace.conftest import make_identity, make_plan, make_step


def _writer(out_dir, queue_steps=2) -> TraceWriter:
    # A fresh writer per test: the ``get`` registry is keyed by directory.
    return TraceWriter(str(out_dir), queue_steps=queue_steps)


def test_open_step_close_produces_committed_file(out_dir):
    w = _writer(out_dir)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(1, "ep_0001"))
    for i in range(3):
        sink.begin_step()
        sink.record_step(make_step(i))
        sink.finish_step()
    sink.on_episode_end(True)
    report = w.drain(timeout=10)
    assert report.ok, report
    final = out_dir / "exp" / "ep_0001.h5"
    assert final.exists()
    assert not (out_dir / "exp" / "ep_0001.h5.tmp").exists()
    assert not (out_dir / "exp" / "ep_0001.h5.reserved").exists()
    with h5py.File(final, "r") as f:
        assert f.attrs["num_steps"] == 3
        assert bool(f.attrs["success"]) is True
        assert bool(f.attrs["trace_closed_ok"]) is True
        assert bool(f.attrs["trace_terminal"]) is True
        assert sorted(k for k in f.keys()) == ["step_0000", "step_0001", "step_0002"]
    sidecar = out_dir / "exp" / "ep_0001.trace.jsonl"
    assert sidecar.exists()
    lines = sidecar.read_text().strip().splitlines()
    assert len(lines) == 4  # header + 3 steps


def test_step_backpressure_blocks_producer_only(out_dir, monkeypatch):
    w = _writer(out_dir, queue_steps=1)
    gate = threading.Event()
    original = w._handle_step

    def slow(*args, **kwargs):
        gate.wait(5.0)
        original(*args, **kwargs)

    monkeypatch.setattr(w, "_handle_step", slow)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(2, "ep_0002"))
    sink.begin_step()
    sink.record_step(make_step(0))  # takes the only permit, writer blocks on gate
    sink.finish_step()
    blocked = threading.Event()
    done = threading.Event()

    def producer():
        sink.begin_step()
        blocked.set()
        try:
            sink.record_step(make_step(1))  # must block until the permit is released
        finally:
            sink.finish_step()
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    blocked.wait(2.0)
    time.sleep(0.3)
    assert not done.is_set(), "second record_step must block while the queue is full"
    # Control messages never take a permit: an in-flight step is an invariant
    # violation the sink reports immediately (it cannot happen on a real
    # connection, whose frames are processed sequentially) instead of letting
    # the Close overtake the blocked Step in the FIFO.
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="in-flight"):
        sink.on_episode_end(True)
    assert time.monotonic() - t0 < 0.2
    gate.set()
    assert done.wait(5.0)
    report = w.drain(timeout=10)
    assert w.healthy
    # A lifecycle invariant violation is Failed, never a published half-file.
    assert not (out_dir / "exp" / "ep_0002.h5").exists()
    assert (out_dir / "exp" / "ep_0002.h5.failed").exists()
    assert any(e.stage == "lifecycle" for e in report.errors)


def test_close_after_last_step_keeps_fifo_order(out_dir, monkeypatch):
    w = _writer(out_dir, queue_steps=1)
    gate = threading.Event()
    original = w._handle_step

    def slow(*args, **kwargs):
        gate.wait(5.0)
        original(*args, **kwargs)

    monkeypatch.setattr(w, "_handle_step", slow)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(3, "ep_0003"))
    done = threading.Event()

    def producer():
        for i in range(3):
            sink.begin_step()
            sink.record_step(make_step(i))
            sink.finish_step()
        sink.on_episode_end(True)  # sequential connection: Close after the last Step
        done.set()

    threading.Thread(target=producer, daemon=True).start()
    time.sleep(0.2)
    assert not done.is_set()
    gate.set()
    assert done.wait(5.0)
    assert w.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "ep_0003.h5", "r") as f:
        assert f.attrs["num_steps"] == 3
        assert bool(f.attrs["trace_terminal"]) is True


def test_two_connections_interleave_without_crosstalk(out_dir):
    w = _writer(out_dir, queue_steps=8)
    a = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    b = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    a.on_episode_start(make_identity(10, "a"))
    b.on_episode_start(make_identity(20, "b"))
    a.record_step(make_step(1))
    b.record_step(make_step(2))
    b.record_step(make_step(3))
    a.record_step(make_step(4))
    b.on_episode_end(False)
    a.on_episode_end(True)
    assert w.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "a.h5", "r") as f:
        assert f.attrs["num_steps"] == 2 and f.attrs["episode_id"] == 10
        assert bool(f.attrs["success"]) is True
    with h5py.File(out_dir / "exp" / "b.h5", "r") as f:
        assert f.attrs["num_steps"] == 2 and f.attrs["episode_id"] == 20
        assert bool(f.attrs["success"]) is False


def test_unknown_token_and_out_of_order_step_are_contained(out_dir):
    w = _writer(out_dir, queue_steps=4)
    w.write_step("no-such-token", 0, make_step(0))  # logged and dropped
    w.close_episode("no-such-token", success=True, terminal=True)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(3, "c"))
    w.write_step(sink._token, 5, make_step(0))  # index 5 is not the next index 0
    sink.on_episode_end(True)
    report = w.drain(timeout=10)
    assert w.healthy
    assert not (out_dir / "exp" / "c.h5").exists()
    assert (out_dir / "exp" / "c.h5.failed").exists()
    assert any(e.stage == "step" for e in report.errors)


def test_close_with_in_flight_step_is_an_assertion(out_dir):
    w = _writer(out_dir, queue_steps=4)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(4, "d"))
    sink.begin_step()
    with pytest.raises(RuntimeError, match="in-flight"):
        sink.on_episode_end(True)
    report = w.drain(timeout=10)
    assert not (out_dir / "exp" / "d.h5").exists()
    assert (out_dir / "exp" / "d.h5.failed").exists()
    assert not report.ok
    assert any(e.stage == "lifecycle" for e in report.errors)


def test_task_end_closes_half_episode(out_dir):
    w = _writer(out_dir, queue_steps=4)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(5, "e"))
    sink.record_step(make_step(0))
    sink.on_task_end()
    assert w.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "e.h5", "r") as f:
        assert bool(f.attrs["trace_terminal"]) is False
        assert f.attrs["num_steps"] == 1


def test_episode_start_twice_closes_previous_as_half_episode(out_dir):
    w = _writer(out_dir, queue_steps=4)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(6, "f1"))
    sink.record_step(make_step(0))
    sink.on_episode_start(make_identity(7, "f2"))
    sink.record_step(make_step(1))
    sink.on_episode_end(True)
    assert w.drain(timeout=10).ok
    with h5py.File(out_dir / "exp" / "f1.h5", "r") as f:
        assert bool(f.attrs["trace_terminal"]) is False
    with h5py.File(out_dir / "exp" / "f2.h5", "r") as f:
        assert bool(f.attrs["trace_terminal"]) is True


def test_fatal_failure_whose_error_record_fails_is_never_a_clean_drain(out_dir, monkeypatch):
    """Plan §7.4: the fatal state itself makes the drain unclean, so a failure
    whose error record could not be written still exits non-zero."""
    w = _writer(out_dir)

    def failing_open(*args, **kwargs):
        raise OSError("injected open failure")

    def failing_record(*args, **kwargs):
        raise RuntimeError("injected error-record failure")

    monkeypatch.setattr(w, "_handle_open", failing_open)
    monkeypatch.setattr(w, "_record_token_error", failing_record)
    sink = H5TraceSink(out_dir, plan=make_plan(), writer=w)
    sink.on_episode_start(make_identity(1, "ep_0001"))
    report = w.drain(timeout=10)
    assert not report.errors and not report.unfinished and not report.timed_out
    assert report.fatal_error is not None and "injected open failure" in report.fatal_error
    assert not report.ok and not w.healthy
    assert not w.stop(timeout=10).ok
