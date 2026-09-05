"""InferenceInterceptor CP2-only branch: FULL / WARM / MISS, wire fields, capture binding."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.orchestrator import CheckResult
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID


class _Model:
    config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=50, action_dim=32)

    def __init__(self):
        self.stage1_calls = self.stage2_calls = self.capture_calls = self.stage3_calls = 0
        self.stage3_from = []

    def run_stage1(self, observation):
        self.stage1_calls += 1
        return SimpleNamespace(state=torch.randn(1, 32), prefix_embs=torch.randn(1, 10, 2048))

    def run_stage2(self, stage1):
        self.stage2_calls += 1
        return SimpleNamespace(stage1=stage1, past_key_values=None, prefix_out=None)

    def run_stage2_capture(self, stage1):
        self.capture_calls += 1
        return SimpleNamespace(stage1=stage1, past_key_values=None, prefix_out=torch.randn(1, 4, 16))

    def run_stage3(self, stage2, noise=None, num_steps=10, return_intermediates=False, save_timesteps=(0.7, 0.5, 0.3)):
        self.stage3_calls += 1
        inter = {st: torch.randn(1, 50, 32) for st in save_timesteps} if return_intermediates else None
        return SimpleNamespace(action_chunk=torch.randn(1, 50, 32), intermediates=inter)

    def run_stage3_from(self, stage2, start_x, start_t, num_steps=10):
        self.stage3_from.append((tuple(start_x.shape), start_t, num_steps))
        return SimpleNamespace(action_chunk=torch.randn(1, 50, 32), intermediates=None)

    def sample_noise(self, shape, device, generator=None):
        return torch.randn(*shape)


class _Policy:
    def __init__(self, model):
        self._is_pytorch_model = True
        self._model = model
        self._input_transform = lambda x: x
        self._output_transform = lambda x: x
        self._pytorch_device = torch.device("cpu")

    @property
    def metadata(self) -> dict[str, Any]:
        return {}


class _Orch:
    """Orchestrator stand-in: one configured checkpoint, scripted verdict."""

    accepts_client_signal = False
    key_builder = object()

    def __init__(self, cp: CheckpointID, result: CheckResult):
        self._cp = cp
        self._result = result
        self.checks: list[tuple[CheckpointID, dict]] = []
        self.broadcasts = self.buffers = self.clears = 0

    def has_checkpoint(self, cp):
        return cp is self._cp

    def check(self, cp, *, request_context=None, **kw):
        self.checks.append((cp, kw))
        if cp is self._cp:
            return self._result
        return CheckResult(hit_type=HitType.MISS)  # unconfigured probe (CP3 tail)

    def broadcast_action(self, a):
        self.broadcasts += 1

    def buffer_for_write(self, *a, **k):
        self.buffers += 1

    def clear(self):
        self.clears += 1


def _obs():
    return {
        "state": np.random.randn(32).astype(np.float32),
        "image": {"base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.bool_(True)},
    }


def _payload(with_ws=False):
    inter = {0.1: torch.randn(10, 32)} if with_ws else None
    return CachePayload(action_chunk=torch.randn(10, 32), intermediates=inter,
                        denoising_num_steps=10 if with_ws else None)


def _make(cp, result, **kw):
    model = _Model()
    orch = _Orch(cp, result)
    it = InferenceInterceptor(_Policy(model), timer=SystemTimer(enabled=False), orchestrator=orch, **kw)
    return it, model, orch


def test_cp2_full_hit_skips_stage3_and_reports_cp2_wire():
    res = CheckResult(hit_type=HitType.FULL_HIT, payload=_payload(), score=0.97, entry_id="w",
                      query_keys={"vlm_out": torch.zeros(8)})
    it, model, orch = _make(CheckpointID.CP2, res)
    assert it._cp2_only
    out = it.infer(_obs())
    assert out["actions"].shape == (10, 32)
    assert model.capture_calls == 1 and model.stage2_calls == 0 and model.stage3_calls == 0
    assert [c for c, _ in orch.checks] == [CheckpointID.CP2]
    assert orch.checks[0][1]["stage2"].prefix_out is not None
    assert orch.broadcasts == 1 and orch.buffers == 1 and orch.clears == 1
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "FULL_HIT" and meta["checkpoint"] == "CP2"
    assert meta["score"] == 0.97 and meta["cp1_score"] is None and meta["winner_id"] == "w"


def test_cp2_warm_start_runs_partial_stage3_from_cached_x():
    res = CheckResult(hit_type=HitType.WARM_START, payload=_payload(with_ws=True), score=0.93,
                      entry_id="w", start_t=0.1, query_keys={"vlm_out": torch.zeros(8)})
    it, model, orch = _make(CheckpointID.CP2, res)
    out = it.infer(_obs())
    assert model.stage3_from == [((1, 10, 32), 0.1, 10)] and model.stage3_calls == 0
    # Exactly one check per decision: no CP1 before, no CP3 probe after (plan §3.3).
    assert [c for c, _ in orch.checks] == [CheckpointID.CP2]
    assert orch.broadcasts == 1 and orch.clears == 1
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "WARM_START" and meta["start_t"] == 0.1 and meta["checkpoint"] == "CP2"


def test_cp2_miss_runs_full_stage3():
    res = CheckResult(hit_type=HitType.MISS, score=0.5, entry_id="w", query_keys={"vlm_out": torch.zeros(8)})
    it, model, orch = _make(CheckpointID.CP2, res)
    out = it.infer(_obs())
    assert model.capture_calls == 1 and model.stage3_calls == 1
    assert orch.clears == 1 and orch.broadcasts == 1
    assert [c for c, _ in orch.checks] == [CheckpointID.CP2]
    meta = out["__hit_meta__"]
    assert meta["hit_type"] == "MISS" and meta["checkpoint"] == "CP2" and meta["score"] == 0.5
    assert "library_sha256" not in meta  # stub orchestrator exposes no artifact meta


def test_legacy_cp1_wire_and_binding_unchanged():
    res = CheckResult(hit_type=HitType.MISS, score=0.5, entry_id="w", query_keys={"robot_state": torch.zeros(32)})
    it, model, orch = _make(CheckpointID.CP1, res)
    assert not it._cp2_only
    out = it.infer(_obs())
    assert model.stage2_calls == 1 and model.capture_calls == 0
    assert [c for c, _ in orch.checks] == [CheckpointID.CP1, CheckpointID.CP3]
    meta = out["__hit_meta__"]
    assert meta["checkpoint"] == "CP1" and meta["cp1_score"] == meta["score"] == 0.5


def test_cache_off_wire_carries_none_checkpoint():
    it = InferenceInterceptor(_Policy(_Model()), timer=SystemTimer(enabled=False))
    meta = it.infer(_obs())["__hit_meta__"]
    assert meta["checkpoint"] is None and meta["score"] is None and meta["hit_type"] == "MISS"


def test_cp2_rejects_meta_stage_placement():
    res = CheckResult(hit_type=HitType.MISS)
    sc = SimpleNamespace(stage1="cpu", stage2="meta", stage3="cpu", is_legacy_default=False,
                         needs_relocation=False, has_meta_stage=True)
    with pytest.raises(ValueError, match="stage2 and stage3 on real devices"):
        _make(CheckpointID.CP2, res, stage_config=sc)


class _Coord:
    def __init__(self, model):
        self.model = model
        self.stage2_kwargs: list[dict] = []

    def submit_to_stage(self, stage_id, bundle_id, payload, **kw):
        if stage_id == 1:
            return self.model.run_stage1(payload)
        if stage_id == 2:
            self.stage2_kwargs.append(kw)
            fn = self.model.run_stage2_capture if kw.get("requires_stage2_capture") else self.model.run_stage2
            return fn(payload)
        return SimpleNamespace(action_chunk=torch.randn(1, 50, 32), intermediates=None)


def test_coordinator_path_passes_request_capability():
    for cp, expect in ((CheckpointID.CP2, True), (CheckpointID.CP1, False)):
        model = _Model()
        coord = _Coord(model)
        res = CheckResult(hit_type=HitType.MISS, score=0.1, entry_id="w",
                          query_keys={"vlm_out" if expect else "robot_state": torch.zeros(8)})
        orch = _Orch(cp, res)
        it = InferenceInterceptor(_Policy(model), timer=SystemTimer(enabled=False), orchestrator=orch,
                                  coordinator=coord, bundle_id="b")
        it.infer(_obs())
        assert coord.stage2_kwargs == [{"requires_stage2_capture": expect}]
        expect_checks = [CheckpointID.CP2] if expect else [CheckpointID.CP1, CheckpointID.CP3]
        assert [c for c, _ in orch.checks] == expect_checks
