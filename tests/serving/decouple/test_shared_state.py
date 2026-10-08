"""Real frozen method replay across independent CPU processes, no listeners."""
import multiprocessing
import math
from unittest.mock import patch

import pytest
from openpi_client import msgpack_numpy

from exp.decouple.recorded import load
from exp.decouple.replay import frozen_runtime, make_connections, tape
from exp.offline_search.closed_loop import plugin
from openpi.serving.batching_coordinator import BatchingCoordinator
from openpi.serving.split_state import bind_runtime
from tests.serving.decouple.stubs import PiModel


def _runtime(model, path, judge):
    rt, _ = frozen_runtime(model, path)
    rt.judge = plugin.JudgeSpec.parse(judge)
    if judge.startswith("quantile"):
        rt.ctrl = plugin.QuantileController(rt.judge.h, rt.judge.W, rt.judge.tau0)
    return rt


def _worker(pipe, model, root, judge):
    from pathlib import Path
    rt = _runtime(model, Path(root) / f"front_{multiprocessing.current_process().name}", judge)
    bind_runtime(rt, Path(root) / "shared", identity=dict(model=model, judge=judge))
    core = BatchingCoordinator(PiModel(), max_batch_size=1, max_wait_ms=0)
    core.start()
    conn, session, _ = make_connections(rt, load(model, limit=2), core, count=1)[0]
    pipe.send(session.conn)
    try:
        with patch.object(plugin.time, "perf_counter_ns", return_value=1_000_000), \
             patch.object(plugin.time, "perf_counter", return_value=1.), \
             patch.object(plugin.time, "time", return_value=1.):
            while True:
                request = pipe.recv()
                if request is None:
                    break
                start, obs = request
                if session.pending:
                    conn.on_episode_start(**start)
                response = msgpack_numpy.Packer().pack(conn.infer(obs))
                pipe.send((response, rt.decision_count, rt.ctrl.state() if rt.ctrl else None))
    finally:
        core.stop()
        pipe.close()


def test_runtime_identity_ignores_log_destination_and_object_addresses(tmp_path):
    a = _runtime("pi05", tmp_path / "a", "quantile:0.5:10")
    b = _runtime("pi05", tmp_path / "b", "quantile:0.5:10")
    a.opts.judge, b.opts.judge = a.judge, b.judge
    a.opts.os_log_dir, b.opts.os_log_dir = "front0", "front1"
    bind_runtime(a, tmp_path / "shared")
    bind_runtime(b, tmp_path / "shared")
    with a.decision_lock:
        a.decision_count += 1
    assert b.decision_count == 1
    assert next(a._conn_ids) == 0
    assert next(b._conn_ids) == 1
    a.ctrl.push(0.1)
    b.ctrl.push(0.9)
    assert a.ctrl.state() == b.ctrl.state()


def test_shared_quantile_preserves_nonfinite_bisect_order(tmp_path):
    from openpi.serving.split_state import SharedQuantile, SharedState
    direct = plugin.QuantileController(.5, 4)
    shared = SharedState(tmp_path / "window", 4)
    fronts = [SharedQuantile(plugin.QuantileController(.5, 4), shared) for _ in range(2)]
    for i, value in enumerate([.1, float("nan"), .9, float("inf"), .2, -float("inf"), .3]):
        direct.push(value)
        fronts[i % 2].push(value)
        expected, actual = direct.tau(), fronts[(i+1) % 2].tau()
        assert actual == expected or math.isnan(actual) and math.isnan(expected)


@pytest.mark.parametrize("model", ["pi05", "groot"])
@pytest.mark.parametrize("judge", ["periodic:3", "quantile:0.5:10"])
def test_cross_front_decision_clock_quantile_and_synthetic_ids_exact(tmp_path, model, judge):
    rt = _runtime(model, tmp_path / "direct", judge)
    data = load(model, limit=2)
    core = BatchingCoordinator(PiModel(), max_batch_size=1, max_wait_ms=0)
    core.start()
    direct = make_connections(rt, data, core)
    context = multiprocessing.get_context("spawn")
    pipes, children = [], []
    try:
        # Controlled connection and admission order, just as an equivalence
        # replay controls the original concurrent server's scheduling order.
        for i in range(2):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, model, str(tmp_path), judge))
            proc.start()
            child.close()
            pipes.append(parent)
            children.append(proc)
            assert parent.poll(60), "front failed to initialize"
            assert parent.recv() == i
        with patch.object(plugin.time, "perf_counter_ns", return_value=1_000_000), \
             patch.object(plugin.time, "perf_counter", return_value=1.), \
             patch.object(plugin.time, "time", return_value=1.):
            for i, start, obs in tape(data, steps=8):
                conn, session, _ = direct[i]
                if session.pending:
                    conn.on_episode_start(**start)
                expected = msgpack_numpy.Packer().pack(conn.infer(dict(obs)))
                pipes[i].send((start, obs))
                assert pipes[i].poll(30)
                actual, count, ctrl = pipes[i].recv()
                assert actual == expected
                assert count == rt.decision_count
                assert ctrl == (rt.ctrl.state() if rt.ctrl else None)
    finally:
        for pipe in pipes:
            pipe.send(None)
        for proc in children:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            assert proc.exitcode == 0
        for pipe in pipes:
            pipe.close()
        core.stop()
