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

    def run_stage3(self, stage2, noise=None):
        self.stage3_calls += 1
        action = (
            self._fixed_action.clone()
            if self._fixed_action is not None
            else torch.randn(1, 50, 32)
        )
        return SimpleNamespace(action_chunk=action)


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
    """Minimal observation dict that passes through identity transforms."""
    return {
        "state": np.random.randn(32).astype(np.float32),
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
    fixed_state = torch.randn(1, 32)
    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=fixed_action)
    policy = FakePolicy(model=model)

    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    # First call: MISS -> writes CP1 entry to cache.
    obs = _make_obs()
    interceptor.infer(obs)
    assert model.stage1_calls == 1
    assert model.stage2_calls == 1
    assert model.stage3_calls == 1

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
# T6.5: CP3 consume with scheduled action skips all stages
# ---------------------------------------------------------------------------


def test_infer_cp3_consume_with_scheduled_action_skips_all_stages():
    policy = FakePolicy()
    orch, _, _ = make_orchestrator()

    # Monkey-patch orchestrator to return a scheduled action.
    scheduled_action = torch.randn(50, 32)
    orch.should_skip_inference = lambda: scheduled_action

    clear_called = [False]
    original_clear = orch.clear

    def tracking_clear():
        clear_called[0] = True
        original_clear()

    orch.clear = tracking_clear

    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )

    obs = _make_obs()
    result = interceptor.infer(obs)

    # No stages should have been called.
    assert policy._model.stage1_calls == 0
    assert policy._model.stage2_calls == 0
    assert policy._model.stage3_calls == 0

    # Output shape checks.
    assert "actions" in result
    assert "state" in result
    actions = result["actions"]
    assert actions.shape == (50, 32)

    # clear() must have been called.
    assert clear_called[0]
