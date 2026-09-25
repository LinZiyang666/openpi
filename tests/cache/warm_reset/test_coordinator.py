"""Batching coordinator: ``Stage3WarmResetPayload`` buckets, rows, failures and metrics (plan §4.5.2, §9.2, §9.4)."""

from __future__ import annotations

import dataclasses
import threading

import pytest
import torch

from openpi.cache.types import PI05_V1, groot_n15_schedule
from openpi.cache.warm_reset.pi05 import run_pi05_continuation, run_pi05_self_start
from openpi.cache.warm_reset.types import private_noise, resolve_plan, resolve_self_plan
from openpi.serving.batching_coordinator import (
    BatchingCoordinator,
    Pi05StageBatcher,
    Stage3MissPayload,
    Stage3WarmResetPayload,
    Stage3WarmStartPayload,
)
from openpi.serving.batching_core import BatchingCore, StageRequest
from tests.cache.warm_reset._support import D, H, Pi05Model, row_stage2, spec_for

K4_SPEC = spec_for("groot", "warmreset_t0.75")


def _plan(arm: str, t: float):
    return resolve_plan(spec_for("pi05", arm), PI05_V1, t)


def _self_plan(arm: str, t: float):
    return resolve_self_plan(spec_for("pi05", arm), PI05_V1, t)


def _x(seed: int) -> torch.Tensor:
    return torch.randn(H, D, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------
# Bucket keys
# ------------------------------------------------------------------


def test_same_grid_arms_share_a_continuation_bucket():
    key = Pi05StageBatcher.bucket_key
    s2 = row_stage2(0.0)
    arms = ["warmreset_t0.2", "resetfinal_t0.2", "selfwarmreset_t0.2", "selfresetfinal_t0.2"]
    keys = {key(Stage3WarmResetPayload(s2, _x(0), _plan(a, 0.2))) for a in arms}
    assert keys == {("warm_reset", _plan("warmreset_t0.2", 0.2).grid_key())}
    # N, level or the shoot start_t split the bucket
    assert key(Stage3WarmResetPayload(s2, _x(0), _plan("warmreset_t0.3", 0.3))) not in keys
    assert key(Stage3WarmResetPayload(s2, _x(0), _plan("midreset_t0.2", 0.2))) not in keys
    shoot = [key(Stage3WarmResetPayload(s2, _x(0), _plan(f"warmshoot_t{t}", t))) for t in (0.2, 0.3)]
    assert shoot[0] != shoot[1]


def test_self_start_bucket_ignores_t_and_point():
    key = Pi05StageBatcher.bucket_key
    s2 = row_stage2(0.0)
    plans = [_self_plan("selfwarmreset_t0.1", 0.1), _self_plan("selfwarmreset_t0.3", 0.3),
             _self_plan("selfresetfinal_t0.2", 0.2)]
    assert {key(Stage3WarmResetPayload(s2, _x(0), p)) for p in plans} == {("warm_reset_self", ("pi05_v1", 10))}


def test_legacy_bucket_keys_are_unchanged():
    key = Pi05StageBatcher.bucket_key
    s2 = row_stage2(0.0)
    assert key(Stage3MissPayload(s2, _x(0), 10)) == ("miss", None, 10)
    assert key(Stage3WarmStartPayload(s2, _x(0), 0.2, 10)) == ("warm_start", 0.2, 10)
    assert key(object()) == ("unknown", None, None)


def test_new_payload_is_a_sibling_not_a_warm_start_subclass():
    assert not issubclass(Stage3WarmResetPayload, Stage3WarmStartPayload)
    assert [f.name for f in dataclasses.fields(Stage3WarmResetPayload)] == ["stage2_out", "x", "plan", "ready_events"]


# ------------------------------------------------------------------
# run_stage3_warm_reset: row alignment and measured counts (CPU, B = 3)
# ------------------------------------------------------------------


def test_continuation_bucket_rows_equal_single_row_runs():
    arms = [("warmreset_t0.2", 0.2), ("resetfinal_t0.2", 0.2), ("selfwarmreset_t0.2", 0.2)]
    stage2s = [row_stage2(v) for v in (-0.5, 0.1, 0.7)]
    xs = [_x(i) for i in range(3)]
    payloads = [Stage3WarmResetPayload(s, x, _plan(a, t)) for s, x, (a, t) in zip(stage2s, xs, arms)]
    outs = Pi05StageBatcher(Pi05Model(), torch.device("cpu")).run_stage3_warm_reset(payloads)
    assert len(outs) == 3
    for p, out in zip(payloads, outs):
        ref = run_pi05_continuation(Pi05Model(), p.stage2_out, p.x[None], p.plan)
        assert out.action_chunk.shape == (1, H, D)
        assert torch.equal(out.action_chunk, ref.action_chunk)
        assert out.steps_run == ref.steps_run == 2  # per row, not 3 x 2


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_self_bucket_hands_each_row_its_own_capture(order):
    rows = [(_self_plan("selfwarmreset_t0.1", 0.1), 9), (_self_plan("selfwarmreset_t0.3", 0.3), 7),
            (_self_plan("selfresetfinal_t0.2", 0.2), None)]
    stage2s = [row_stage2(v) for v in (-0.4, 0.2, 0.9)]
    noises = [private_noise(s, (H, D)) for s in (11, 12, 13)]
    assert [plan.capture_index for plan, _ in rows] == [9, 7, None]
    payloads = [Stage3WarmResetPayload(stage2s[i], noises[i], rows[i][0]) for i in order]
    outs = Pi05StageBatcher(Pi05Model(), torch.device("cpu")).run_stage3_warm_reset(payloads)
    for p, out in zip(payloads, outs):
        (ref,) = run_pi05_self_start(Pi05Model(), p.stage2_out, p.x[None], [p.plan])
        assert out.action_chunk.shape == (1, H, D) and torch.equal(out.action_chunk, ref.action_chunk)
        if p.plan.capture_index is None:
            assert out.snapshot is None
        else:
            assert out.snapshot.shape == (1, H, D) and torch.equal(out.snapshot, ref.snapshot)
        assert out.steps_run == 10


@pytest.mark.parametrize("bad", ["mixed_types", "mixed_grids", "invalid_row", "wrong_k"])
def test_bucket_refuses_before_the_first_step(bad):
    s2 = row_stage2(0.0)
    good = _plan("warmreset_t0.2", 0.2)
    if bad == "mixed_types":
        plans = [good, _self_plan("selfwarmreset_t0.2", 0.2)]
    elif bad == "mixed_grids":
        plans = [good, _plan("warmreset_t0.3", 0.3)]
    elif bad == "invalid_row":
        plans = [good, dataclasses.replace(good, start_t=0.25)]  # same grid key, unrecoverable t
    else:
        plans = [resolve_plan(K4_SPEC, groot_n15_schedule(4), 0.75)]
    model = Pi05Model()
    with pytest.raises((TypeError, ValueError)):
        Pi05StageBatcher(model, torch.device("cpu")).run_stage3_warm_reset(
            [Stage3WarmResetPayload(s2, _x(0), p) for p in plans]
        )
    assert model.t_log == []


# ------------------------------------------------------------------
# The real coordinator: one batch across connections
# ------------------------------------------------------------------


def test_threads_of_one_grid_run_as_one_batch():
    model = Pi05Model()
    arms = ["warmreset_t0.2", "resetfinal_t0.2", "selfresetfinal_t0.2"]
    results, errors = {}, {}
    barrier = threading.Barrier(3)
    with BatchingCoordinator(model, device="cpu", max_batch_size=3, max_wait_ms=300.0) as bc:

        def worker(i):
            payload = Stage3WarmResetPayload(row_stage2(0.1 * i), _x(i), _plan(arms[i], 0.2))
            barrier.wait()
            try:
                results[i] = bc.submit_to_stage(3, "default", payload, timeout=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors[i] = exc

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
    assert not errors, errors
    assert all(len(bits) == 3 for bits in model.t_log)  # every Euler step ran at B = 3
    assert len(model.t_log) == 2 and all(r.steps_run == 2 for r in results.values())


# ------------------------------------------------------------------
# Adapters without the method fail loudly, bucket-locally
# ------------------------------------------------------------------


class _MissOnlyBatcher:
    """An adapter that predates warm reset (no ``run_stage3_warm_reset``)."""

    @staticmethod
    def bucket_key(payload):
        return ("miss",) if isinstance(payload, Stage3MissPayload) else ("other",)

    def run_stage3_miss(self, payloads, *, num_steps, save_timesteps_per_request):
        return [f"miss-{i}" for i in range(len(payloads))]


def _request(payload, stage_id=3):
    return StageRequest(request_id="r", bundle_id="b", stage_id=stage_id, payload=payload,
                        reply_event=threading.Event(), reply_slot=[None])


def test_adapter_without_the_method_fails_only_its_bucket():
    core = BatchingCore(_MissOnlyBatcher(), device="cpu")
    new = _request(Stage3WarmResetPayload(row_stage2(0.0), _x(0), _plan("warmreset_t0.2", 0.2)))
    miss = _request(Stage3MissPayload(row_stage2(0.0), _x(1), 10))
    core._run_batch(3, [new, miss])
    assert isinstance(new.error, TypeError) and new.reply_event.is_set()
    assert miss.error is None and miss.reply_slot[0] == "miss-0"


def test_groot_stage_batcher_refuses_the_new_payload():
    from openpi.cache.groot.batcher import GrootStageBatcher

    core = BatchingCore(GrootStageBatcher(runner=None), device="cpu")
    req = _request(Stage3WarmResetPayload(row_stage2(0.0), _x(0), _plan("warmreset_t0.2", 0.2)))
    with pytest.raises(TypeError, match="Stage3WarmResetPayload"):
        core._run_stage3_bucket([req])


# ------------------------------------------------------------------
# 3bucket metrics
# ------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.records = []

    def record_batch(self, record):
        self.records.append(record)


class _EchoBatcher(_MissOnlyBatcher):
    def run_stage3_warm(self, payloads, *, start_t, num_steps, capture_first_step):
        return ["warm"] * len(payloads)

    def run_stage3_warm_reset(self, payloads):
        return [type("Out", (), {"steps_run": p.plan.n_steps if hasattr(p.plan, "n_steps") else p.plan.k})()
                for p in payloads]

    @staticmethod
    def bucket_key(payload):
        return type(payload).__name__, getattr(payload, "plan", None)


@pytest.mark.parametrize("bucket_first", [False, True])
def test_metrics_record_plan_values_and_leave_legacy_records_alone(bucket_first):
    core = BatchingCore(_EchoBatcher(), device="cpu")
    core._recorder = _Recorder()
    cont, self_ = _plan("warmreset_t0.2", 0.2), _self_plan("selfresetfinal_t0.2", 0.2)
    buckets = [
        [_request(Stage3MissPayload(row_stage2(0.0), _x(0), 10))],
        [_request(Stage3WarmStartPayload(row_stage2(0.0), _x(0), 0.2, 10))],
        [_request(Stage3WarmResetPayload(row_stage2(0.0), _x(0), cont))],
        [_request(Stage3WarmResetPayload(row_stage2(0.0), _x(0), self_))],
    ]
    for bucket in buckets:
        if bucket_first:
            core._dispatch_stage3_bucket(bucket)
        else:
            core._run_batch(3, bucket)
    records = [r for r in core._recorder.records if r["stage"] == "3bucket"]
    trimmed = [{k: v for k, v in r.items() if k != "forward_ms"} for r in records]
    assert trimmed[0] == {"stage": "3bucket", "mode": "miss", "start_t": -1.0, "num_steps": 10, "size": 1}
    assert trimmed[1] == {"stage": "3bucket", "mode": "warm_start", "start_t": 0.2, "num_steps": 10, "size": 1}
    assert trimmed[2] == {"stage": "3bucket", "mode": "warm_reset", "start_t": 0.2, "num_steps": 2, "size": 1,
                          "grid_key": list(cont.grid_key()), "steps_run": 2}
    assert trimmed[3] == {"stage": "3bucket", "mode": "warm_reset_self", "start_t": -1.0, "num_steps": 10,
                          "size": 1, "grid_key": ["pi05_v1", 10], "steps_run": 10}


# ------------------------------------------------------------------
# Direct entries refuse malformed input before the first denoise_step
# ------------------------------------------------------------------


def _batch_stage2(values):
    from openpi.serving import stage_io

    return stage_io.stack_stage2_output([row_stage2(v) for v in values])


@pytest.mark.parametrize("defect", ["plans_vs_noise", "plans_vs_stage2", "no_plans", "final_bad_t",
                                    "capture_mismatch", "capture_bool"])
def test_self_start_entry_refuses_before_the_first_step(defect):
    plans = [_self_plan("selfwarmreset_t0.1", 0.1), _self_plan("selfresetfinal_t0.2", 0.2)]
    stage2, noise = _batch_stage2([0.0, 0.5]), torch.stack([_x(0), _x(1)])
    if defect == "plans_vs_noise":
        noise = noise[:1]
    elif defect == "plans_vs_stage2":
        stage2 = _batch_stage2([0.0, 0.5, 0.9])
    elif defect == "no_plans":
        plans = []
    elif defect == "final_bad_t":
        plans[1] = dataclasses.replace(plans[1], start_t=0.25)
    elif defect == "capture_mismatch":
        plans[0] = dataclasses.replace(plans[0], capture_index=7)
    else:
        plans[0] = dataclasses.replace(plans[0], capture_index=True)
    model = Pi05Model()
    with pytest.raises(ValueError):
        run_pi05_self_start(model, stage2, noise, plans)
    assert model.t_log == []


def test_continuation_entry_refuses_a_batch_mismatch():
    model = Pi05Model()
    with pytest.raises(ValueError, match="stage-2 batch"):
        run_pi05_continuation(model, _batch_stage2([0.0, 0.5]), _x(0)[None], _plan("warmreset_t0.2", 0.2))
    assert model.t_log == []
