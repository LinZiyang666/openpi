"""Interceptor behaviour: what a hit skips, and what must survive the inference context."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from .conftest import ACTION_DIM, ACTION_HORIZON, StubGrootModel

# The Pi0.5 interceptor's wire schema, which this one must reproduce exactly so
# a single analysis path reads both.
LEGACY_META_KEYS = {"hit_type", "start_t", "winner_id", "cp1_score", "searched"}


class _StubPolicy:
    def __init__(self, model: StubGrootModel) -> None:
        self.model = model
        self._model = model

    def apply_transforms(self, obs):
        return self.model.build_inputs()

    def unapply_transforms(self, action):
        return {"action.out": action["action"].numpy()}


class _StubOrchestrator:
    def __init__(self, verdict: HitType, payload=None) -> None:
        self._verdict = verdict
        self._payload = payload
        self.calls: list[str] = []
        self.broadcast: list[torch.Tensor] = []
        self.buffered: list[tuple] = []

    def check(self, checkpoint_id, **stage_outputs):
        assert checkpoint_id is CheckpointID.CP1
        assert "stage1" in stage_outputs
        self.calls.append("check")
        return types.SimpleNamespace(
            hit_type=self._verdict,
            payload=self._payload,
            score=0.5,
            entry_id="traj:7",
            query_keys={"robot_state": torch.zeros(3)},
            searched=True,
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
        self.calls.append("task_begin")

    def on_task_end(self):
        self.calls.append("task_end")

    def on_episode_start(self, **kwargs):
        self.calls.append(f"episode_start:{kwargs.get('task_key')}")

    def on_episode_end(self):
        self.calls.append("episode_end")


def _obs():
    return {"state.x": np.zeros((1, 3), dtype=np.float32)}


def _build(verdict, payload=None, timer=None):
    model = StubGrootModel()
    policy = _StubPolicy(model)
    runner = GrootStagedRunner(model, timer=timer, verify_upstream=False)
    orch = _StubOrchestrator(verdict, payload)
    interceptor = GrootCacheInterceptor(
        policy, runner, orchestrator=orch, timer=timer
    )
    return model, orch, interceptor


def test_full_hit_never_runs_stage2():
    payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
    model, orch, interceptor = _build(HitType.FULL_HIT, payload)

    interceptor.get_action(_obs())

    assert model.backbone.eagle_model.extract_calls == 1  # stage 1 always runs
    assert model.action_head.calls == 0  # stage 2 skipped entirely
    assert torch.equal(orch.broadcast[0], torch.ones(ACTION_HORIZON, ACTION_DIM))


def test_miss_runs_stage2_exactly_once():
    model, orch, interceptor = _build(HitType.MISS)
    interceptor.get_action(_obs())
    assert model.action_head.calls == 1


def test_bookkeeping_order_matches_pi05():
    payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
    _, orch, interceptor = _build(HitType.FULL_HIT, payload)
    interceptor.get_action(_obs())
    assert orch.calls == ["check", "broadcast", "buffer", "clear"]


def test_clear_runs_even_when_the_cycle_raises():
    _, orch, interceptor = _build(HitType.FULL_HIT, payload=None)  # payload=None -> AttributeError
    with pytest.raises(Exception):
        interceptor.get_action(_obs())
    assert orch.calls[-1] == "clear"


def test_warm_start_is_refused():
    _, _, interceptor = _build(HitType.WARM_START)
    with pytest.raises(RuntimeError, match="WARM_START"):
        interceptor.get_action(_obs())


def test_persisted_tensors_satisfy_the_storage_contract():
    """Including is_inference: a `.cpu()` inside the context does not escape it."""
    model, orch, interceptor = _build(HitType.MISS)
    interceptor.get_action(_obs())

    tensors = list(orch.broadcast) + [chunk for _, chunk in orch.buffered]
    tensors += [t for keys, _ in orch.buffered for t in keys.values()]
    assert tensors
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        assert tensor.dtype is torch.float32
        assert tensor.is_contiguous()
        assert tensor.is_inference() is False
        tensor.add_(0.0)  # would raise on an inference tensor


def test_action_chunk_is_unbatched_for_storage():
    model, orch, interceptor = _build(HitType.MISS)
    interceptor.get_action(_obs())
    assert orch.broadcast[0].shape == (ACTION_HORIZON, ACTION_DIM)


def test_hit_meta_field_set_is_exactly_the_legacy_one():
    payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
    _, _, interceptor = _build(HitType.FULL_HIT, payload)
    out = interceptor.get_action(_obs())
    assert set(out["__hit_meta__"]) == LEGACY_META_KEYS
    assert out["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert out["__hit_meta__"]["start_t"] is None


def test_cache_off_reports_a_miss_placeholder():
    model = StubGrootModel()
    policy = _StubPolicy(model)
    runner = GrootStagedRunner(model, verify_upstream=False)
    interceptor = GrootCacheInterceptor(policy, runner)
    out = interceptor.get_action(_obs())
    assert out["__hit_meta__"]["hit_type"] == "MISS"
    assert model.action_head.calls == 1


def test_lifecycle_is_forwarded_including_the_task_key():
    _, orch, interceptor = _build(HitType.MISS)
    interceptor.on_task_begin()
    interceptor.on_episode_start(task="OpenCabinet", episode_id=3)
    interceptor.on_episode_end(success=True)
    interceptor.on_task_end()
    assert "task_begin" in orch.calls
    assert "episode_start:OpenCabinet" in orch.calls
    assert "episode_end" in orch.calls
    assert "task_end" in orch.calls


def test_probe_counts_are_the_gate_evidence():
    from openpi.serving import monitor

    previous = monitor.get_monitor_level()
    monitor.set_monitor_level(monitor.MonitorLevel.BASIC)
    try:
        payload = CachePayload(action_chunk=torch.ones(ACTION_HORIZON, ACTION_DIM))
        timer = SystemTimer(enabled=True)
        _, _, interceptor = _build(HitType.FULL_HIT, payload, timer=timer)
        interceptor.get_action(_obs())

        counts = {name: stats.count for name, stats in timer.summary(task_only=False).items()}
        # Positive controls first: without them "stage2 recorded nothing" would
        # be indistinguishable from a timer that never recorded anything.
        assert counts.get("stage1_vision") == 1
        assert counts.get("cp1_sum") == 1
        assert counts.get("total_inference") == 1
        assert counts.get("stage2_llm", 0) == 0
        assert counts.get("stage2_action", 0) == 0
    finally:
        monitor.set_monitor_level(previous)
