"""Trace-mode connection lifecycle through a real ``WebsocketPolicyServer`` (plan §6.2, §12-F).

A concurrent server serves ``GrootCacheInterceptor`` stacks in trace build
mode over a process-level ``BatchingCore(GrootStageBatcher)`` whose stage-3
forward is slowed down, so every lifecycle event below lands while a
request's variants are still owned by the worker:

* **disconnect** -- the client closes its socket right after sending an
  infer: the in-flight decision completes on the worker (its inputs read
  back intact), then ``on_task_end`` closes the episode as non-terminal; the
  file is committed with ``trace_terminal=False`` (the audit never admits
  it) and nothing else on the server is disturbed;
* **unload** -- while another connection is in flight, a connection rebinds
  to a second bundle (the server ends its old stack) and the bundle registry
  entry of the in-flight connection is replaced; the in-flight decision is
  served from the stack it was bound to, and each episode ends in the state
  its lifecycle says;
* **drain** -- a stop request arrives while a decision is in flight: the
  listener closes, the handler finishes the decision before ``on_task_end``,
  the writer drains, and no input is released before the worker is done.

Each scenario runs twice: on the CPU stub (host-side ordering, the forward
slowed by a host sleep) and, when CUDA is available, on the CUDA stub with the
worker stream really delayed by ``torch.cuda._sleep`` -- the host returns from
the forward at once while its kernels are still queued, so a reply published
(or an input released) before the stream is clean would be caught: the inputs
are checksummed on the worker stream *after* the forward and read back at the
end, and ``_publish_batch`` records whether the worker stream was idle at the
moment each decision is handed back (a worker without its stream sync fails
all three CUDA scenarios on that record).
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from openpi.cache.groot import batcher as gb
from openpi.cache.groot.batcher import GrootStageBatcher
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TracePlan, TraceRuntime
from openpi.cache.types import VISION_0, VISION_1, VISION_2, groot_n15_schedule
from openpi.serving import batching_core
from openpi.serving import websocket_policy_server as wps
from openpi.serving.batching_core import BatchingCore
from openpi_client import msgpack_numpy
from tests.cache.groot.test_groot_stage3 import _model_with_flow_head
from tests.cache.groot.test_groot_stream_ready_gpu import _cuda_model
from tests.cache.groot.test_groot_stream_ready_gpu import _inputs as _cuda_inputs
from tests.cache.groot.test_trace_groot import _obs

N_STEPS = 4
SCHEDULE = groot_n15_schedule(N_STEPS)
CAMS = (VISION_0, VISION_1, VISION_2)
SLOW_S = 0.8
SLOW_CYCLES = 1_500_000_000  # ~0.8 s of GPU spin on the worker stream


class _Policy:
    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def apply_transforms(self, obs):
        return _cuda_inputs(self.model) if self.device == "cuda" else self.model.build_inputs()

    def unapply_transforms(self, action):
        return {"action.out": action["action"].numpy()}


class _Adapter:
    """Wire adapter: any non-ctrl frame is one decision on the stub observation."""

    def __init__(self, interceptor: GrootCacheInterceptor, bundle_id: str) -> None:
        self._it = interceptor
        self.bundle_id = bundle_id

    def infer(self, obs):
        out = self._it.get_action(_obs())
        return {"actions": out["action.out"], "__hit_meta__": out["__hit_meta__"]}

    def on_task_begin(self):
        self._it.on_task_begin()

    def on_task_end(self):
        self._it.on_task_end()

    def on_episode_start(self, **kw):
        self._it.on_episode_start(**kw)

    def on_episode_end(self, success):
        self._it.on_episode_end(success)


def _plan() -> TracePlan:
    return TracePlan(
        model="groot_n15", schedule=SCHEDULE, checkpoint=None, warm_tiers=(),
        record_noise_actions=True, save_timesteps=SCHEDULE.timesteps, record_prefix_tokens=True,
        record_raw_images=True, record_model_images=False, record_query_keys=True, record_search=True,
        record_tokenized_prompt=True, raw_image_keys=(), rng_isolation="verdict_aware",
        sidecar_jsonl=True, fail_loud=True, concurrent=True,
    )


class _Harness:
    """Model, slowed stage-3 worker, factory, and the in-flight ownership ledger."""

    def __init__(self, out_dir, device: str) -> None:
        self.device = device
        self.model = _cuda_model() if device == "cuda" else _model_with_flow_head(N_STEPS)
        self.policy = _Policy(self.model, device)
        self.lock = threading.Lock()
        self.writer = TraceWriter.get(str(out_dir), queue_steps=16)
        self.out_dir = out_dir
        runner = GrootStagedRunner(self.model, verify_upstream=False)
        # Materialise the lazy head layer before serving: its first forward
        # synchronises the host, which would hide an unsynchronised hand-back.
        with runner.session():
            stage2 = runner.run_stage2_llm(runner.run_stage1(self.policy.apply_transforms(None)))
            z = runner.sample_noise(stage2, generator=torch.Generator(device=device).manual_seed(0))
        gb.run_miss(runner, stage2, z, schedule=SCHEDULE, capture=True)
        if device == "cuda":
            torch.cuda.synchronize()
        batcher = GrootStageBatcher(runner)
        # (checksum at worker entry, checksum computed on the worker stream after the forward)
        self._ledger: list[tuple[float, torch.Tensor]] = []
        # (batch size, worker stream idle) per hand-back: the reply must never
        # leave the worker while its stream still has queued kernels.
        self.published: list[tuple[int, bool]] = []
        self.in_worker = threading.Event()
        real = batcher.run_stage3_miss

        def checksum(payloads) -> torch.Tensor:
            return sum(
                p.noise.double().sum() + p.stage2_out.stage2.backbone_features.double().sum() for p in payloads
            )

        def slow_miss(payloads, **kw):
            before = float(checksum(payloads))  # the worker already waited on the producer events
            self.in_worker.set()
            if device == "cuda":
                torch.cuda._sleep(SLOW_CYCLES)  # noqa: SLF001 - delays the worker stream, not the host
            else:
                time.sleep(SLOW_S)
            out = real(payloads, **kw)
            self._ledger.append((before, checksum(payloads)))  # queued behind the forward, no host sync
            return out

        batcher.run_stage3_miss = slow_miss
        self.core = BatchingCore(batcher, device=device, max_batch_size=8, max_wait_ms=20.0)
        publish = self.core._publish_batch  # noqa: SLF001

        def publish_spy(batch, outputs):
            idle = torch.cuda.current_stream().query() if device == "cuda" else True
            publish(batch, outputs)
            self.published.append((len(batch), idle))

        self.core._publish_batch = publish_spy  # noqa: SLF001
        self.core.start()
        self.bound: list[str] = []

    def ledger(self) -> list[tuple[float, float]]:
        if self.device == "cuda":
            torch.cuda.synchronize()
        return [(b, float(a)) for b, a in self._ledger]

    def factory(self, base_policy, bundle_id: str = "default"):
        runner = GrootStagedRunner(self.model, verify_upstream=False)
        plan = _plan()
        rt = TraceRuntime(plan=plan, sink=H5TraceSink(self.out_dir, plan=plan, writer=self.writer))
        it = GrootCacheInterceptor(
            self.policy, runner, trace=rt, coordinator=self.core, bundle_id=bundle_id,
            model_lock=self.lock, trace_vision_fields=CAMS,
        )
        self.bound.append(bundle_id)
        return _Adapter(it, bundle_id)

    def close(self) -> None:
        self.core.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, harness: _Harness) -> None:
        self.port = _free_port()
        self.server = wps.WebsocketPolicyServer(
            harness.policy, host="127.0.0.1", port=self.port, metadata={"t": 1},
            concurrent=True, connection_policy_factory=harness.factory, allow_dynamic_bundles=True,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"stop_on_request": True},
                                       daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.server.request_stop()
        self.thread.join(15)


class _Client:
    def __init__(self, port: int) -> None:
        from websockets.sync.client import connect

        self.ws = connect(f"ws://127.0.0.1:{port}", max_size=None, compression=None)
        self.packer = msgpack_numpy.Packer()
        msgpack_numpy.unpackb(self.ws.recv())  # metadata

    def call(self, msg: dict) -> dict:
        self.ws.send(self.packer.pack(msg))
        return msgpack_numpy.unpackb(self.ws.recv())

    def send(self, msg: dict) -> None:
        self.ws.send(self.packer.pack(msg))

    def episode_start(self, name: str, uid: str, **extra) -> dict:
        return self.call({"__ctrl__": "episode_start", "__experiment__": "exp", "__task__": "t",
                          "__episode_id__": 1, "__episode_name__": name,
                          "__extra__": {"task_uid": uid, "attempt": 1}, **extra})

    def close(self) -> None:
        self.ws.close()


def _wait_for(pred, timeout=15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def _attrs(path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: f.attrs[k] for k in ("num_steps", "trace_terminal", "trace_closed_ok", "trace_write_errors")}


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    ]
)
def harness(request, tmp_path):
    h = _Harness(tmp_path, request.param)
    saved = dict(wps._bundles)
    yield h
    h.close()
    wps._bundles.clear()
    wps._bundles.update(saved)
    TraceWriter.get(str(tmp_path)).stop(timeout=10)


def test_disconnect_mid_inference_completes_then_closes_non_terminal(harness):
    srv = _Server(harness)
    try:
        a = _Client(srv.port)
        assert a.episode_start("dropped", "u-a")["__ack__"] == "episode_start"
        a.send({"x": np.zeros(1)})
        assert harness.in_worker.wait(10)
        a.close()  # the decision is still on the worker
        path = harness.out_dir / "exp" / "dropped.h5"
        _wait_for(path.exists)
        attrs = _attrs(path)
        assert int(attrs["num_steps"]) == 1  # the in-flight decision was recorded, then the Close
        assert bool(attrs["trace_terminal"]) is False and bool(attrs["trace_closed_ok"]) is True
        # the server is still healthy: a second connection runs a full episode
        b = _Client(srv.port)
        b.episode_start("after", "u-b")
        out = b.call({"x": np.zeros(1)})
        assert out["actions"].shape[-1] > 0
        b.call({"__ctrl__": "episode_end", "__success__": True})
        b.close()
        _wait_for((harness.out_dir / "exp" / "after.h5").exists)
        assert bool(_attrs(harness.out_dir / "exp" / "after.h5")["trace_terminal"]) is True
    finally:
        srv.stop()
    assert harness.core.fatal_error is None
    ledger = harness.ledger()
    assert len(ledger) == 2 and all(b == a for b, a in ledger)  # inputs intact through the forward
    assert harness.published == [(1, True), (1, True)]
    assert not batching_core._FATAL_PAYLOADS
    assert harness.writer.drain(timeout=10).ok


def test_rebind_and_registry_replacement_while_another_connection_is_in_flight(harness):
    for bid in ("b1", "b2"):
        wps._bundles[bid] = SimpleNamespace(cache_config=None, shared_storage=None, yaml_id=bid,
                                            config_path=f"/{bid}.yaml", version=1)
    srv = _Server(harness)
    try:
        b = _Client(srv.port)
        assert b.call({"__ctrl__": "select_bundle", "bundle_id": "b1"})["__ack__"] == "select_bundle"
        b.episode_start("inflight", "u-b")
        result: dict = {}

        def _infer():
            result["out"] = b.call({"x": np.zeros(1)})

        t = threading.Thread(target=_infer)
        t.start()
        assert harness.in_worker.wait(10)
        # unload #1: replace the registry entry the in-flight connection was bound from
        wps._bundles["b1"] = SimpleNamespace(cache_config=None, shared_storage=None, yaml_id="b1",
                                             config_path="/b1-v2.yaml", version=2)
        # unload #2: another connection rebinds (the server ends its old stack)
        c = _Client(srv.port)
        assert c.call({"__ctrl__": "select_bundle", "bundle_id": "b1"})["__ack__"] == "select_bundle"
        c.episode_start("rebound_old", "u-c1")
        assert c.call({"__ctrl__": "select_bundle", "bundle_id": "b2"})["__ack__"] == "select_bundle"
        assert t.is_alive() and not harness.published  # both unloads landed while B was on the worker
        c.episode_start("rebound_new", "u-c2")
        assert c.call({"x": np.zeros(1)})["actions"].shape[-1] > 0
        c.call({"__ctrl__": "episode_end", "__success__": True})
        t.join(20)
        assert "out" in result and result["out"]["__hit_meta__"]["trace"]["executed_arm"] == "full_inference"
        b.call({"__ctrl__": "episode_end", "__success__": True})
        b.close()
        c.close()
        exp = harness.out_dir / "exp"
        _wait_for(lambda: all((exp / f"{n}.h5").exists() for n in ("inflight", "rebound_old", "rebound_new")))
        assert bool(_attrs(exp / "inflight.h5")["trace_terminal"]) is True
        assert int(_attrs(exp / "inflight.h5")["num_steps"]) == 1
        old = _attrs(exp / "rebound_old.h5")
        assert bool(old["trace_terminal"]) is False and int(old["num_steps"]) == 0  # ended by the rebind
        assert bool(_attrs(exp / "rebound_new.h5")["trace_terminal"]) is True
    finally:
        srv.stop()
    assert harness.bound.count("b1") == 2 and harness.bound.count("b2") == 1
    assert harness.core.fatal_error is None
    ledger = harness.ledger()
    assert len(ledger) == 2 and all(b == a for b, a in ledger)
    assert harness.published == [(1, True), (1, True)]
    assert harness.writer.drain(timeout=10).ok


def test_stop_request_while_in_flight_finishes_the_decision_before_release(harness):
    srv = _Server(harness)
    a = _Client(srv.port)
    a.episode_start("drained", "u-d")
    a.send({"x": np.zeros(1)})
    assert harness.in_worker.wait(10)
    srv.server.request_stop()  # listener closes; the handler still owns the decision
    srv.thread.join(20)
    assert not srv.thread.is_alive()
    assert harness.published == [(1, True)]  # the server returned only after the worker handed the decision back
    path = harness.out_dir / "exp" / "drained.h5"
    report = harness.writer.drain(timeout=10)
    assert report.ok
    assert path.exists()
    attrs = _attrs(path)
    assert int(attrs["num_steps"]) == 1 and bool(attrs["trace_terminal"]) is False
    ledger = harness.ledger()
    assert len(ledger) == 1 and all(b == a for b, a in ledger)
    assert harness.core.fatal_error is None
    try:
        a.close()
    except Exception:  # noqa: BLE001 - the server already closed it
        pass
