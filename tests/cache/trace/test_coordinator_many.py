"""Coordinator core under trace load (plan §6.1, §12-A-4, §12-F).

``submit_many_to_stage`` of one decision's variants groups with other
connections' same-key requests; a MISS bucket mixing legacy (``None``) and
trace (full-schedule) requests calls the model once with the union and hands
each request its own subset back; an all-``None`` bucket keeps the legacy
call shape (no ``save_timesteps`` kwarg -- the stub signature is the guard);
a failing bucket fails only its own requests; the Pi0.5 default snapshot set
equals the model signature default.
"""

from __future__ import annotations

import inspect
import threading

import pytest
import torch

from openpi.serving import batching_coordinator as bcmod
from openpi.serving.batching_coordinator import (
    BatchingCoordinator,
    Stage3MissPayload,
    Stage3WarmStartPayload,
    _MODEL_DEFAULT_SAVE_TIMESTEPS,
)
from openpi.cache.types import PI05_V1
from tests.cache.test_serving_optimization import _StubModel

H, D = 50, 32


class _TraceStub(_StubModel):
    """Stub whose ``run_stage3`` accepts ``save_timesteps`` but refuses ``None``."""

    def __init__(self):
        super().__init__()
        self.save_calls: list = []

    def run_stage3(self, stage2, *, noise, num_steps=10, return_intermediates=False, **kw):
        if "save_timesteps" in kw and kw["save_timesteps"] is None:
            raise TypeError("save_timesteps=None is not accepted")
        save = kw.get("save_timesteps", _MODEL_DEFAULT_SAVE_TIMESTEPS)
        self.save_calls.append(("kw" if "save_timesteps" in kw else "default", tuple(save)))
        b = stage2.stage1.state.shape[0]
        self.stage3_batch_sizes.append((b, "miss", None, num_steps))
        action = noise * 2.0
        inter = {t: torch.full((b, H, D), float(t)) for t in save} if return_intermediates else None
        from openpi.models_pytorch.pi0_pytorch import Stage3Output

        return Stage3Output(action_chunk=action, intermediates=inter)


def _stage2(stub, b=1):
    stage1 = stub.run_stage1({"state": torch.zeros(b, 4), "tokens": torch.zeros(b, 2, dtype=torch.int64)})
    return stub.run_stage2(stage1)


def test_default_save_timesteps_matches_model_signature():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    default = inspect.signature(PI0Pytorch.run_stage3).parameters["save_timesteps"].default
    assert tuple(default) == _MODEL_DEFAULT_SAVE_TIMESTEPS


@pytest.mark.parametrize("bucket_first", [False, True])
def test_three_connections_multi_variant_group_by_key(monkeypatch, bucket_first):
    if bucket_first:
        monkeypatch.setenv("OPENPI_STAGE3_BUCKET_FIRST", "1")
    else:
        monkeypatch.delenv("OPENPI_STAGE3_BUCKET_FIRST", raising=False)
    stub = _TraceStub()
    results: dict[int, list] = {}
    errors: dict[int, BaseException] = {}
    barrier = threading.Barrier(3)
    with BatchingCoordinator(stub, device="cpu", max_batch_size=9, max_wait_ms=200.0) as bc:

        def worker(i):
            s2 = _stage2(stub)
            payloads = [
                Stage3MissPayload(s2, torch.full((H, D), float(i)), 10, save_timesteps=PI05_V1.timesteps),
                Stage3WarmStartPayload(s2, torch.zeros(H, D), 0.7, 10),
                Stage3WarmStartPayload(s2, torch.zeros(H, D), 0.5, 10),
            ]
            barrier.wait()
            try:
                results[i] = bc.submit_many_to_stage(3, "default", payloads, request_id=f"c{i}")
            except BaseException as exc:  # noqa: BLE001
                errors[i] = exc

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
    assert not errors, errors
    assert len(results) == 3
    for i, outs in results.items():
        assert len(outs) == 3
        assert torch.equal(outs[0].action_chunk[0], torch.full((H, D), 2.0 * i))
        assert sorted(outs[0].intermediates) == sorted(PI05_V1.timesteps)
        assert outs[1].intermediates is None and outs[2].intermediates is None
    # Each key ran as ONE batch of three across the connections.
    sizes = sorted(stub.stage3_batch_sizes)
    assert sizes == sorted([(3, "miss", None, 10), (3, "warm_start", 0.7, 10), (3, "warm_start", 0.5, 10)])


def test_union_and_per_request_filter_with_legacy_request():
    stub = _TraceStub()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=4, max_wait_ms=200.0) as bc:
        s2 = _stage2(stub)
        legacy = Stage3MissPayload(s2, torch.zeros(H, D), 10)  # save_timesteps=None
        trace = Stage3MissPayload(s2, torch.ones(H, D), 10, save_timesteps=PI05_V1.timesteps)
        outs = {}

        def submit(name, payload):
            outs[name] = bc.submit_to_stage(3, "default", payload, request_id=name, timeout=5.0)

        t1 = threading.Thread(target=submit, args=("legacy", legacy))
        t2 = threading.Thread(target=submit, args=("trace", trace))
        t1.start(); t2.start(); t1.join(10); t2.join(10)
    assert stub.save_calls == [("kw", tuple(sorted(set(PI05_V1.timesteps) | set(_MODEL_DEFAULT_SAVE_TIMESTEPS), reverse=True)))]
    assert sorted(outs["legacy"].intermediates) == sorted(_MODEL_DEFAULT_SAVE_TIMESTEPS)
    assert sorted(outs["trace"].intermediates) == sorted(PI05_V1.timesteps)
    for t, v in outs["legacy"].intermediates.items():
        assert torch.equal(v, torch.full((1, H, D), float(t)))


def test_all_none_bucket_keeps_legacy_call_shape():
    # ``_StubModel.run_stage3`` has no ``save_timesteps`` parameter at all:
    # any kwarg would be a TypeError, so a clean reply proves the legacy shape.
    stub = _StubModel()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=2, max_wait_ms=50.0) as bc:
        s2 = _stage2(stub)
        out = bc.submit_to_stage(3, "default", Stage3MissPayload(s2, torch.zeros(H, D), 10), timeout=5.0)
    assert sorted(out.intermediates) == [0.7]
    assert Stage3MissPayload(s2, torch.zeros(H, D), 10).save_timesteps is None


def test_per_bucket_failure_isolation():
    class _Boom(_TraceStub):
        def run_stage3_from(self, stage2, *, start_x, start_t, num_steps=10):
            if start_t == 0.5:
                raise RuntimeError("bucket boom")
            return super().run_stage3_from(stage2, start_x=start_x, start_t=start_t, num_steps=num_steps)

    stub = _Boom()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=8, max_wait_ms=200.0) as bc:
        s2 = _stage2(stub)
        payloads = [
            Stage3MissPayload(s2, torch.zeros(H, D), 10),
            Stage3WarmStartPayload(s2, torch.zeros(H, D), 0.5, 10),
            Stage3WarmStartPayload(s2, torch.zeros(H, D), 0.7, 10),
        ]
        reqs = [bc.enqueue_to_stage(3, "default", p) for p in payloads]
        got = []
        for r in reqs:
            try:
                got.append(bc.wait_for(r, timeout=5.0))
            except RuntimeError as exc:
                got.append(exc)
        assert isinstance(got[1], RuntimeError) and "bucket boom" in str(got[1])
        assert got[0].action_chunk.shape == (1, H, D) and got[2].action_chunk.shape == (1, H, D)
        stage3_threads = [t for t in bc._stage_threads if "stage3" in t.name]
        assert all(t.is_alive() for t in stage3_threads)
        # submit_many: the failure is raised only after every sibling was harvested.
        with pytest.raises(RuntimeError, match="bucket boom"):
            bc.submit_many_to_stage(3, "default", payloads, timeout=5.0)


def test_submit_many_timeout_uses_shared_deadline(monkeypatch):
    import time

    class _Slow(_TraceStub):
        def run_stage3(self, *a, **k):
            time.sleep(0.5)
            return super().run_stage3(*a, **k)

    stub = _Slow()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=4, max_wait_ms=5.0) as bc:
        s2 = _stage2(stub)
        payloads = [Stage3MissPayload(s2, torch.zeros(H, D), 10) for _ in range(2)]
        with pytest.raises(TimeoutError):
            bc.submit_many_to_stage(3, "default", payloads, timeout=0.1)
        time.sleep(1.2)  # the worker still owns and finishes the requests


def test_record_ready_events_is_empty_on_cpu():
    assert bcmod.record_ready_events("cpu") == ()
    assert bcmod.record_ready_events(None) == ()


def test_interceptor_variants_go_through_submit_many():
    """``_run_trace_variants`` under a coordinator submits all variants at once
    and maps the replies back by name (plan §5 step 10)."""
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.timing import SystemTimer
    from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
    from openpi.cache.trace.types import TraceRuntime
    from tests.cache.trace.conftest import TraceFakePolicy, make_plan

    stub = _TraceStub()
    with BatchingCoordinator(stub, device="cpu", max_batch_size=8, max_wait_ms=50.0) as bc:
        plan = make_plan(save_timesteps=PI05_V1.timesteps, record_noise_actions=True)
        rt = TraceRuntime(plan=plan, sink=H5TraceSink("/tmp/unused-trace", plan=plan,
                                                      writer=TraceWriter("/tmp/unused-trace")), twins=None)
        it = InferenceInterceptor(TraceFakePolicy(), timer=SystemTimer(enabled=False), eager=True,
                                  coordinator=bc, bundle_id="default", trace=rt)
        s2 = _stage2(stub)
        variants = [
            ("full", "miss", {"noise": torch.zeros(1, H, D), "num_steps": 10, "save_timesteps": PI05_V1.timesteps}),
            ("warm_07", "warm", {"start_x": torch.zeros(H, D), "start_t": 0.7, "num_steps": 10}),
        ]
        outs = it._run_trace_variants(s2, variants)
    assert set(outs) == {"full", "warm_07"}
    assert sorted(outs["full"].intermediates) == sorted(PI05_V1.timesteps)
    assert outs["warm_07"].action_chunk.shape == (1, H, D)
    assert stub.stage3_batch_sizes == [(1, "miss", None, 10), (1, "warm_start", 0.7, 10)] or \
        sorted(stub.stage3_batch_sizes) == sorted([(1, "miss", None, 10), (1, "warm_start", 0.7, 10)])
