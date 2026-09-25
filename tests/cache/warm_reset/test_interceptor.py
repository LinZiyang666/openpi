"""Pi0.5 ``InferenceInterceptor``: absent block is call-for-call unchanged; the injected executor path (plan §4.5, §9.3, §9.4)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import AlwaysHitJudge, AlwaysWarmStartJudge
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID
from openpi.cache.warm_reset.pi05 import Pi05WarmResetExecutor
from openpi.cache.warm_reset.runtime import WarmResetSession
from openpi.serving.batching_coordinator import (
    Stage3WarmResetPayload,
    Stage3WarmStartPayload,
)
from tests.cache.conftest import insert_entry, make_orchestrator
from tests.cache.test_interceptor import FakePolicy
from tests.cache.warm_reset._support import (
    EPISODE,
    EXTRA,
    D,
    H,
    Pi05Model,
    SyncCoordinator,
    obs_for,
    pi05_payload,
    pi05_stack,
    spec_for,
)

LEGACY_META_KEYS = {"hit_type", "start_t", "winner_id", "cp1_score", "checkpoint", "score", "searched"}


class _RecordingModel(Pi05Model):
    """Records every stage entry with its arguments."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def run_stage1(self, observation):
        self.calls.append(("run_stage1",))
        return super().run_stage1(observation)

    def run_stage2(self, stage1):
        self.calls.append(("run_stage2",))
        return super().run_stage2(stage1)

    def run_stage3(self, stage2, *, noise=None, num_steps=10, return_intermediates=False, **kw):
        self.calls.append(("run_stage3", noise is None, num_steps, return_intermediates, tuple(kw)))
        return Pi05Model.run_stage3(self, stage2, noise=noise, num_steps=num_steps,
                                    return_intermediates=return_intermediates, **kw)

    def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
        self.calls.append(("run_stage3_from", start_x.clone(), start_t, num_steps))
        return Pi05Model.run_stage3_from(self, stage2, start_x, start_t, num_steps=num_steps)


def _orchestrator(model, verdict: str, t: float = 0.2):
    if verdict == "FULL_HIT":
        orch, _, storage = make_orchestrator(vector_dims={"robot_state": D}, judge=AlwaysHitJudge())
    else:
        orch, _, storage = make_orchestrator(vector_dims={"robot_state": D}, judge=AlwaysWarmStartJudge(t))
    entry = None
    if verdict != "MISS":
        entry = insert_entry(storage, CheckpointID.CP1, model.state, pi05_payload(t))
    return orch, entry


def _legacy(model, verdict, *, coordinator=False):
    orch, entry = _orchestrator(model, verdict)
    coord = SyncCoordinator(model) if coordinator else None
    icpt = InferenceInterceptor(FakePolicy(model), timer=SystemTimer(enabled=True), orchestrator=orch,
                                eager=True, coordinator=coord)
    return icpt, entry, coord


# ------------------------------------------------------------------
# Absent block: exactly today's calls, wire and probes
# ------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["FULL_HIT", "WARM_START", "MISS"])
def test_absent_block_direct_calls_and_wire_are_the_legacy_ones(verdict):
    model = _RecordingModel()
    icpt, entry, _ = _legacy(model, verdict)
    keys_before = set(vars(model))
    icpt.on_task_begin()
    out = icpt.infer(obs_for(None))
    names = [c[0] for c in model.calls]
    if verdict == "FULL_HIT":
        assert names == ["run_stage1"]
    elif verdict == "MISS":
        assert names == ["run_stage1", "run_stage2", "run_stage3"]
        assert model.calls[2] == ("run_stage3", True, 10, True, ())
    else:
        assert names == ["run_stage1", "run_stage2", "run_stage3_from"]
        _, start_x, start_t, num_steps = model.calls[2]
        assert start_t == 0.2 and num_steps == 10
        assert torch.equal(start_x, entry.payload.intermediates[0.2][None])
    meta = out["__hit_meta__"]
    assert set(meta) == LEGACY_META_KEYS and "warm_reset" not in meta
    assert meta["hit_type"] == verdict and meta["checkpoint"] == "CP1"
    assert meta["start_t"] == (0.2 if verdict == "WARM_START" else None)
    assert meta["winner_id"] == (None if entry is None else entry.id)
    assert "stage3_self_start" not in icpt._timer._probes
    assert set(vars(model)) == keys_before  # nothing on the model instance was replaced or added


@pytest.mark.parametrize("verdict", ["WARM_START", "MISS"])
def test_absent_block_coordinator_payloads_are_the_legacy_ones(verdict):
    model = _RecordingModel()
    icpt, _, coord = _legacy(model, verdict, coordinator=True)
    icpt.on_task_begin()
    icpt.infer(obs_for(None))
    stage3 = [p for sid, _, p in coord.calls if sid == 3]
    assert len(stage3) == 1
    if verdict == "WARM_START":
        (payload,) = stage3
        assert type(payload) is Stage3WarmStartPayload
        assert (payload.start_t, payload.num_steps, payload.start_x.shape) == (0.2, 10, (H, D))
    else:
        assert type(stage3[0]).__name__ == "Stage3MissPayload"


# ------------------------------------------------------------------
# Construction: refusals, bindings, probes
# ------------------------------------------------------------------


def _executor(arm="warmreset_t0.2"):
    spec = spec_for("pi05", arm)
    return Pi05WarmResetExecutor(spec, WarmResetSession(spec))


@pytest.mark.parametrize("combo", ["trace", "hit_executor", "miss_executor", "shadow_teacher", "cp2"])
def test_construction_refuses_incompatible_components(combo):
    model = Pi05Model()
    orch, _ = _orchestrator(model, "WARM_START")
    kwargs = {"warm_reset": _executor()}
    if combo == "trace":
        kwargs["trace"] = SimpleNamespace(twins=None)
    elif combo == "hit_executor":
        kwargs["hit_executor"] = lambda obs: obs
    elif combo == "miss_executor":
        kwargs["miss_executor"] = lambda obs: obs
    elif combo == "shadow_teacher":
        kwargs["shadow_teacher"] = SimpleNamespace(enabled=True)
    else:
        orch = SimpleNamespace(has_checkpoint=lambda cp: cp == CheckpointID.CP2, artifact_meta={})
    with pytest.raises(ValueError, match="warm_reset"):
        InferenceInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch, **kwargs)


def test_bindings_and_probes_follow_the_injected_executor():
    model = Pi05Model()
    orch, _ = _orchestrator(model, "WARM_START")
    cache = InferenceInterceptor(FakePolicy(model), timer=SystemTimer(), orchestrator=orch, eager=True,
                                 warm_reset=_executor())
    assert "stage3_self_start" not in cache._timer._probes
    orch2, _ = _orchestrator(model, "WARM_START")
    self_ = InferenceInterceptor(FakePolicy(model), timer=SystemTimer(), orchestrator=orch2, eager=True,
                                 coordinator=SyncCoordinator(model), warm_reset=_executor("selfwarmreset_t0.2"))
    assert "stage3_self_start" in self_._timer._probes
    assert self_._stage3_warm_reset_fn.__name__ == "_warm_reset_via_coordinator"
    assert cache._stage3_warm_reset_fn.__name__ == "run"


# ------------------------------------------------------------------
# The executor path
# ------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["warmreset_t0.2", "resetfinal_t0.2", "selfwarmreset_t0.2", "selfmidfinal_t0.2"])
def test_direct_and_coordinator_paths_agree_at_batch_one(tmp_path, arm):
    outs = {}
    for coordinator in (False, True):
        stack = pi05_stack(tmp_path / str(coordinator), arm, coordinator=coordinator)
        stack.served.on_task_begin()
        stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
        outs[coordinator] = stack.served.infer(obs_for(stack))
        if coordinator:
            payloads = [p for sid, _, p in stack.coordinator.calls if sid == 3]
            assert all(type(p) is Stage3WarmResetPayload for p in payloads)
            assert len(payloads) == (2 if arm.startswith("self") else 1)  # self start + continuation
            assert all(p.x.shape == (H, D) and p.x.device == torch.device("cpu") for p in payloads)
    np.testing.assert_array_equal(outs[False]["actions"], outs[True]["actions"])
    # the two stacks differ only in their evidence directory (part of the spec digest)
    metas = [dict(outs[c]["__hit_meta__"], warm_reset=dict(outs[c]["__hit_meta__"]["warm_reset"])) for c in (False, True)]
    for meta in metas:
        meta["warm_reset"].pop("spec_digest")
    assert metas[0] == metas[1]


def test_hit_meta_carries_the_warm_reset_evidence(tmp_path):
    stack = pi05_stack(tmp_path, "selfwarmreset_t0.2")
    stack.served.on_task_begin()
    stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    meta = stack.served.infer(obs_for(stack))["__hit_meta__"]
    assert set(meta) == LEGACY_META_KEYS | {"warm_reset"}
    wr = meta["warm_reset"]
    assert wr["schema"] == "warm_reset_meta_v1" and wr["spec_digest"] == stack.spec.digest()
    assert (wr["kind"], wr["source"], wr["point"], wr["level"]) == ("reset", "self", "snapshot", 1.0)
    assert (wr["start_t"], wr["schedule_id"], wr["k"], wr["n_steps"]) == (0.2, "pi05_v1", 10, 2)
    assert (wr["continuation_nfe"], wr["n_stage3_calls"], wr["self_start_calls"]) == (2, 1, 1)
    assert wr["self_start"] is True and wr["self_direct_nfe"] == 10 and wr["decision_nfe"] == 12
    assert wr["t"] == pytest.approx([1.0, 0.5]) and wr["dt"] == pytest.approx(-0.5)
    assert wr["self_seed"] == stack.parts.session.spec.seed_policy().seed({**EPISODE, **EXTRA}, 0)


def test_payload_step_count_must_match_the_plan(tmp_path):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    executor = stack.parts.executor
    stack.parts.session.begin_episode(**EPISODE, extra_metadata=dict(EXTRA))
    stack.parts.session.begin_decision()
    payload = pi05_payload(0.2)
    payload.denoising_num_steps = 8
    calls = []
    with pytest.raises(ValueError, match="denoising_num_steps"):
        executor.run(stage2=None, cp_result=SimpleNamespace(payload=payload, start_t=0.2),
                     snapshot_x=torch.zeros(1, H, D), run_stage3=lambda *a: calls.append(a),
                     timer=SystemTimer(enabled=False))
    assert calls == []


def test_executor_requires_an_open_decision(tmp_path):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    with pytest.raises(RuntimeError, match="no open decision"):
        stack.parts.executor.run(stage2=None, cp_result=SimpleNamespace(payload=pi05_payload(0.2), start_t=0.2),
                                 snapshot_x=torch.zeros(1, H, D), run_stage3=None, timer=None)


def test_global_rng_is_untouched_by_the_self_start(tmp_path):
    stack = pi05_stack(tmp_path, "selfresetfinal_t0.2")
    stack.served.on_task_begin()
    stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    obs = obs_for(stack)
    state = torch.get_rng_state()
    stack.served.infer(obs)
    assert torch.equal(torch.get_rng_state(), state)
