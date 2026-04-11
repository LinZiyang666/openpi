"""Tests for InferenceInterceptor cache integration (T6.1–T6.5).

These tests use FakePolicy/FakeModel to avoid loading real model weights.
All tests run in eager mode (pytorch_compile_mode=None) on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.timing import SystemTimer

from tests.cache.conftest import make_orchestrator


# ---------------------------------------------------------------------------
# Fake Policy / Model
# ---------------------------------------------------------------------------


class FakeModel:
    """Minimal model with staged API. Tracks call counts."""

    config = SimpleNamespace(pytorch_compile_mode=None)

    def __init__(
        self,
        fixed_state: torch.Tensor | None = None,
        fixed_action: torch.Tensor | None = None,
    ) -> None:
        self.stage1_calls = 0
        self.stage2_calls = 0
        self.stage3_calls = 0
        self._fixed_state = fixed_state
        self._fixed_action = fixed_action

    def run_stage1(self, observation):
        self.stage1_calls += 1
        state = (
            self._fixed_state.clone()
            if self._fixed_state is not None
            else torch.randn(1, 32)
        )
        return SimpleNamespace(
            state=state,
            prefix_embs=torch.randn(1, 10, 2048),
        )

    def run_stage2(self, stage1):
        self.stage2_calls += 1
        return SimpleNamespace(kv_cache=None)

    def run_stage3(self, stage2, noise=None, num_steps=10, return_intermediates=False, save_timesteps=(0.7, 0.5, 0.3)):
        self.stage3_calls += 1
        action = (
            self._fixed_action.clone()
            if self._fixed_action is not None
            else torch.randn(1, 50, 32)
        )
        intermediates = None
        if return_intermediates:
            intermediates = {
                st: torch.randn(1, 50, 32)
                for st in save_timesteps
            }
        return SimpleNamespace(action_chunk=action, intermediates=intermediates)


class FakePolicy:
    """Minimal Policy mock for InferenceInterceptor."""

    def __init__(self, model: FakeModel | None = None) -> None:
        self._is_pytorch_model = True
        self._model = model if model is not None else FakeModel()
        self._input_transform = lambda x: x
        self._output_transform = lambda x: x
        self._pytorch_device = torch.device("cpu")

    @property
    def metadata(self) -> dict[str, Any]:
        return {}


def _make_obs() -> dict:
    """Minimal observation dict that passes through identity transforms.

    Observation.from_dict requires 'image', 'image_mask', and 'state'.
    Shape conventions (before batching by interceptor):
      image:      (h, w, c) uint8  -> interceptor adds [None, ...] -> (1, h, w, c)
      image_mask: scalar bool      -> interceptor adds [None, ...] -> (1,)
      state:      (s,) float32     -> interceptor adds [None, ...] -> (1, s)
    """
    return {
        "state": np.random.randn(32).astype(np.float32),
        "image": {
            "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.bool_(True),
        },
    }


# ---------------------------------------------------------------------------
# T6.1: infer without orchestrator returns actions
# ---------------------------------------------------------------------------


def test_infer_without_orchestrator_returns_actions():
    policy = FakePolicy()
    interceptor = InferenceInterceptor(policy, timer=SystemTimer(enabled=False))

    obs = _make_obs()
    result = interceptor.infer(obs)
    assert "actions" in result
    assert "state" in result
    assert policy._model.stage1_calls == 1
    assert policy._model.stage2_calls == 1
    assert policy._model.stage3_calls == 1


# ---------------------------------------------------------------------------
# T6.2: CP1 miss -> full pipeline
# ---------------------------------------------------------------------------


def test_infer_with_orchestrator_cp1_miss_full_pipeline():
    policy = FakePolicy()
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    obs = _make_obs()
    result = interceptor.infer(obs)
    assert "actions" in result
    assert "state" in result
    assert policy._model.stage1_calls == 1
    assert policy._model.stage2_calls == 1
    assert policy._model.stage3_calls == 1


# ---------------------------------------------------------------------------
# T6.3: CP1 hit skips stage2 + stage3
#
# Key: FakeModel must return deterministic state/action BEFORE interceptor
# is constructed, because __init__ caches staged methods into _stage1_fn etc.
# ---------------------------------------------------------------------------


def test_infer_with_orchestrator_cp1_hit_skips_stage2_3():
    from openpi.cache.components.write_policy import AlwaysWritePolicy

    fixed_state = torch.randn(1, 32)
    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=fixed_action)
    policy = FakePolicy(model=model)

    orch, _, _ = make_orchestrator(write_policy=AlwaysWritePolicy())
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    # First call: MISS -> buffers CP1 entry for write.
    obs = _make_obs()
    interceptor.on_task_begin()
    interceptor.infer(obs)
    assert model.stage1_calls == 1
    assert model.stage2_calls == 1
    assert model.stage3_calls == 1

    # Flush buffered data to storage via episode lifecycle.
    interceptor.on_episode_end(success=True)
    interceptor.on_episode_start("test", "test", 1)

    # Reset counters for second call.
    model.stage1_calls = 0
    model.stage2_calls = 0
    model.stage3_calls = 0

    # Second call: same fixed_state -> CP1 hit, skip stage2 + stage3.
    result = interceptor.infer(obs)
    assert "actions" in result
    assert model.stage1_calls == 1   # stage1 still runs (CP1 is after stage1)
    assert model.stage2_calls == 0   # skipped
    assert model.stage3_calls == 0   # skipped


# ---------------------------------------------------------------------------
# T6.4: CP3 consume stub does not skip
# ---------------------------------------------------------------------------


def test_infer_cp3_consume_stub_does_not_skip():
    policy = FakePolicy()
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    obs = _make_obs()
    result = interceptor.infer(obs)
    assert "actions" in result
    assert policy._model.stage1_calls == 1


# ---------------------------------------------------------------------------
# T6.5: WARM_START runs stage2 + run_stage3_from, skips full stage3
# ---------------------------------------------------------------------------


def test_infer_warm_start_calls_stage3_from():
    """WARM_START runs stage2 + run_stage3_from, skips full stage3."""
    import math
    import torch.nn.functional as F
    from openpi.cache.components.judge import ThresholdJudge
    from openpi.cache.storage_types import CachePayload
    from openpi.cache.types import CheckpointID
    from tests.cache.conftest import insert_entry

    # Build two states with known cosine similarity = 0.96
    stored_state = torch.zeros(1, 32)
    stored_state[0, 0] = 1.0

    target_cos = 0.96
    sin_val = math.sqrt(1.0 - target_cos ** 2)
    query_state = torch.zeros(1, 32)
    query_state[0, 0] = target_cos
    query_state[0, 1] = sin_val
    query_state = F.normalize(query_state, dim=1)

    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=query_state, fixed_action=fixed_action)

    stage3_from_calls = [0]

    def fake_run_stage3_from(stage2, start_x, start_t, *, num_steps=10):
        stage3_from_calls[0] += 1
        return SimpleNamespace(action_chunk=fixed_action.clone())

    model.run_stage3_from = fake_run_stage3_from

    policy = FakePolicy(model=model)

    warm_tiers = [{"threshold": 0.95, "start_t": 0.3}]
    judge = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95, warm_tiers=warm_tiers)
    orch, _, storage = make_orchestrator(judge=judge)
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    # Pre-populate storage with entry that has intermediates at start_t=0.3
    payload = CachePayload(
        action_chunk=torch.randn(50, 32),
        intermediates={0.3: torch.randn(50, 32), 0.5: torch.randn(50, 32)},
        denoising_num_steps=10,
    )
    insert_entry(storage, CheckpointID.CP1, stored_state, payload)

    obs = _make_obs()
    interceptor.on_task_begin()
    result = interceptor.infer(obs)

    assert "actions" in result
    assert model.stage1_calls == 1
    assert model.stage2_calls == 1
    assert stage3_from_calls[0] == 1
    assert model.stage3_calls == 0
