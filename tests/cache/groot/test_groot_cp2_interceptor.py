"""The CP2-only decision cycle of the GR00T interceptor (ActionCache-style arm).

stage 1 -> stage 2 (LM only) -> encoded key source -> one ``check(CP2)``; then
FULL_HIT replays, WARM_START resumes the transcription at the library snapshot,
MISS runs upstream's ``get_action`` from the same stage-2 output. The stub flow
head counts its prologue encoders and Euler steps, which is the evidence for the
cost formula's ``E`` term: FULL = 1 prologue / 0 steps, WARM@0.875 = 2 / 1,
MISS = 2 / 8. The CP1 cycle is untouched (the legacy tests keep that; here only
the key source's absence is asserted).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootCP2KeySource, GrootStagedRunner, _batch_feature
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID, groot_n15_schedule

from .conftest import ACTION_DIM, ACTION_HORIZON, StubGrootModel
from .test_groot_interceptor import LEGACY_META_KEYS, _StubPolicy
from .test_groot_stage3 import _FlowHeadStub

K = 8
WARM_T = 0.875
LIB_SHA = "b" * 64


class _CountingFlowHead(_FlowHeadStub):
    """Upstream's loop shape with counters on the prologue encoders and the Euler step."""

    def __init__(self, num_steps: int = K) -> None:
        super().__init__(num_steps)
        self.process_calls = 0
        self.state_calls = 0
        self.step_calls = 0
        self.eval()  # nn.Module starts in training mode; the served head is eval

    def process_backbone_output(self, backbone_output):
        self.process_calls += 1
        # BatchFeature-like: subscriptable *and* attribute access, as upstream.
        return _batch_feature({"backbone_features": backbone_output["backbone_features"].float()})

    def state_encoder(self, state, embodiment_id):
        self.state_calls += 1
        return super().state_encoder(state, embodiment_id)

    def action_encoder(self, actions, timesteps, embodiment_id):
        self.step_calls += 1
        return super().action_encoder(actions, timesteps, embodiment_id)


class _CP2Orchestrator:
    def __init__(self, verdict, payload=None, *, start_t=None, artifact_meta=None, serves_cp2=True) -> None:
        self._verdict, self._payload, self._start_t = verdict, payload, start_t
        self._serves_cp2 = serves_cp2
        self.artifact_meta = artifact_meta
        self.calls: list[str] = []
        self.checked: list = []
        self.sources: list = []
        self.broadcast: list[torch.Tensor] = []
        self.buffered: list[tuple] = []
        self.has_checkpoint_calls = 0

    def has_checkpoint(self, checkpoint_id):
        self.has_checkpoint_calls += 1
        return self._serves_cp2 and checkpoint_id is CheckpointID.CP2

    def check(self, checkpoint_id, **stage_outputs):
        self.calls.append("check")
        self.checked.append((checkpoint_id, sorted(stage_outputs)))
        if checkpoint_id is CheckpointID.CP2:
            assert isinstance(stage_outputs["cp2_source"], GrootCP2KeySource)
            assert stage_outputs["stage2"].action_pred is None  # LM-only stage 2
            self.sources.append(stage_outputs["cp2_source"])
        return types.SimpleNamespace(
            hit_type=self._verdict, payload=self._payload, start_t=self._start_t, score=0.5,
            entry_id="traj:7", query_keys={"vlm_out": torch.zeros(3)}, searched=True,
        )

    def broadcast_action(self, chunk):
        self.calls.append("broadcast")
        self.broadcast.append(chunk)

    def buffer_for_write(self, keys, chunk, **kwargs):
        self.calls.append("buffer")
        self.buffered.append((keys, chunk))

    def clear(self):
        self.calls.append("clear")

    def on_task_begin(self):
        pass

    def on_task_end(self):
        pass

    def on_episode_start(self, **kwargs):
        pass

    def on_episode_end(self):
        pass


def _obs():
    return {"state.x": np.zeros((1, 3), dtype=np.float32)}


def _warm_payload(start_t: float = WARM_T) -> CachePayload:
    return CachePayload(
        action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM),
        intermediates={start_t: torch.full((ACTION_HORIZON, ACTION_DIM), 0.25)},
        denoising_num_steps=K, schedule_id=f"groot_n15_k{K}_v1",
    )


def _build(verdict, payload=None, *, start_t=None, timer=None, artifact_meta=None, serves_cp2=True):
    torch.manual_seed(0)
    model = StubGrootModel()
    model.action_head = _CountingFlowHead()
    policy = _StubPolicy(model)
    runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
    meta = {"schedule_id": f"groot_n15_k{K}_v1", "library_sha256": LIB_SHA} if artifact_meta is None else artifact_meta
    orch = _CP2Orchestrator(verdict, payload, start_t=start_t, artifact_meta=meta, serves_cp2=serves_cp2)
    interceptor = GrootCacheInterceptor(policy, runner, orchestrator=orch, timer=timer)
    return model, runner, orch, interceptor


def test_full_hit_encodes_once_and_never_denoises():
    payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
    model, _, orch, interceptor = _build(HitType.FULL_HIT, payload)
    out = interceptor.get_action(_obs())
    head = model.action_head
    assert model.backbone.eagle_model.language_model.calls == 1  # LM once
    assert (head.process_calls, head.state_calls, head.step_calls) == (1, 1, 0)
    assert orch.checked == [(CheckpointID.CP2, ["cp2_source", "stage2"])]
    assert orch.calls == ["check", "broadcast", "buffer", "clear"]
    assert torch.equal(orch.broadcast[0], payload.action_chunk)
    meta = out["__hit_meta__"]
    assert set(meta) == LEGACY_META_KEYS | {"checkpoint", "score", "library_sha256"}
    assert meta["checkpoint"] == "CP2" and meta["hit_type"] == "FULL_HIT" and meta["start_t"] is None
    assert meta["score"] == 0.5 and meta["cp1_score"] is None
    assert meta["library_sha256"] == LIB_SHA


def test_warm_start_at_0875_runs_one_euler_step_from_the_snapshot():
    payload = _warm_payload()
    model, runner, orch, interceptor = _build(HitType.WARM_START, payload, start_t=WARM_T)
    out = interceptor.get_action(_obs())
    head = model.action_head
    assert (head.process_calls, head.state_calls, head.step_calls) == (2, 2, 1)
    assert out["__hit_meta__"]["hit_type"] == "WARM_START" and out["__hit_meta__"]["start_t"] == WARM_T
    # Same numbers as resuming the transcription directly from the same stage-2 output.
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
        ref = runner.run_stage3_from(stage2, payload.intermediates[WARM_T], WARM_T,
                                     schedule=groot_n15_schedule(K)).action_pred
    assert torch.equal(orch.broadcast[0], ref[0].float().cpu())
    assert head.step_calls == 2  # the reference ran exactly one more step


def test_miss_runs_upstreams_full_loop_from_the_same_stage2_output():
    model, runner, orch, interceptor = _build(HitType.MISS)
    torch.manual_seed(1234)
    out = interceptor.get_action(_obs())
    head = model.action_head
    assert (head.process_calls, head.state_calls, head.step_calls) == (2, 2, K)
    assert out["__hit_meta__"]["hit_type"] == "MISS" and out["__hit_meta__"]["checkpoint"] == "CP2"
    # Bit-equal to the plain teacher (run_stage2) under the same seed: the
    # helper consumed no RNG and stage 2 was not altered by the in-place prologue.
    torch.manual_seed(1234)
    with runner.session():
        ref = runner.run_stage2(runner.run_stage1(model.build_inputs())).action_pred
    assert torch.equal(orch.broadcast[0], ref[0].float().cpu())


def test_warm_payload_disagreeing_with_the_library_schedule_is_refused():
    payload = _warm_payload()
    payload.schedule_id = "groot_n15_k4_v1"
    payload.denoising_num_steps = 4
    _, _, orch, interceptor = _build(HitType.WARM_START, payload, start_t=WARM_T)
    with pytest.raises(RuntimeError, match="payload is stamped"):
        interceptor.get_action(_obs())
    assert orch.calls[-1] == "clear"


def test_clear_runs_when_the_cp2_cycle_raises():
    _, _, orch, interceptor = _build(HitType.FULL_HIT, payload=None)
    with pytest.raises(Exception):
        interceptor.get_action(_obs())
    assert orch.calls[-1] == "clear"


def test_persisted_tensors_satisfy_the_storage_contract_on_cp2():
    _, _, orch, interceptor = _build(HitType.MISS)
    interceptor.get_action(_obs())
    tensors = list(orch.broadcast) + [chunk for _, chunk in orch.buffered]
    for tensor in tensors:
        assert tensor.device.type == "cpu" and tensor.dtype is torch.float32
        assert tensor.is_contiguous() and not tensor.is_inference()
    assert orch.broadcast[0].shape == (ACTION_HORIZON, ACTION_DIM)


def test_cp2_probes_replace_cp1_sum_and_time_the_encoder():
    from openpi.serving import monitor

    previous = monitor.get_monitor_level()
    monitor.set_monitor_level(monitor.MonitorLevel.BASIC)
    try:
        timer = SystemTimer(enabled=True)
        payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
        _, _, _, interceptor = _build(HitType.FULL_HIT, payload, timer=timer)
        interceptor.get_action(_obs())
        counts = {name: stats.count for name, stats in timer.summary(task_only=False).items()}
        assert counts.get("cp2_sum") == 1 and counts.get("cp2_encode") == 1
        assert counts.get("stage2_llm") == 1 and counts.get("total_inference") == 1
        assert counts.get("cp1_sum", 0) == 0 and counts.get("stage2_action", 0) == 0
    finally:
        monitor.set_monitor_level(previous)


def test_cp2_only_is_frozen_at_construction():
    _, _, orch, interceptor = _build(HitType.MISS)
    assert orch.has_checkpoint_calls == 1
    interceptor.get_action(_obs())
    interceptor.get_action(_obs())
    assert orch.has_checkpoint_calls == 1  # never re-consulted per decision


def test_cp1_cycle_never_touches_the_key_source():
    """An orchestrator that does not serve CP2 gets the legacy cycle: check(CP1,
    stage1=...) and the head's own prologue only."""
    model, runner, orch, interceptor = _build(HitType.MISS, serves_cp2=False)
    called = []
    original = runner.run_cp2_key_source
    runner.run_cp2_key_source = lambda stage2: called.append(1) or original(stage2)
    interceptor.get_action(_obs())
    head = model.action_head
    assert called == []
    assert orch.checked == [(CheckpointID.CP1, ["stage1"])]
    assert (head.process_calls, head.state_calls, head.step_calls) == (1, 1, K)
    assert orch.calls == ["check", "broadcast", "buffer", "clear"]
