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


# ---------------------------------------------------------------------------
# prefill_trajectory: Step 3 trajectory-deviation spawn runner
# ---------------------------------------------------------------------------


def _prefill_interceptor() -> tuple[InferenceInterceptor, FakeModel]:
    """Build an orchestrator-wired interceptor usable for prefill tests."""
    fixed_state = torch.randn(1, 32)
    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=fixed_action)
    policy = FakePolicy(model=model)
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )
    interceptor.on_task_begin()
    interceptor.on_episode_start("exp", "task", 0)
    return interceptor, model


def test_prefill_trajectory_runs_stage1_once_per_step():
    interceptor, model = _prefill_interceptor()

    obs_seq = [_make_obs() for _ in range(3)]
    act_seq = [np.random.randn(50, 32).astype(np.float32) for _ in range(3)]

    interceptor.prefill_trajectory(obs_seq, actions=act_seq)

    # Each prefill step goes through the full ``infer`` pipeline: stage1
    # always runs (CP1 check happens after stage1), but the synthetic CP1
    # FULL_HIT bypasses stage2 + stage3. So stage1 counts equal the number
    # of prefill steps, stage2/3 counts stay at 0.
    assert model.stage1_calls == 3
    assert model.stage2_calls == 0
    assert model.stage3_calls == 0


def test_prefill_trajectory_exits_prefill_mode_after_each_step():
    interceptor, _ = _prefill_interceptor()
    storage = interceptor._orchestrator._storage

    obs_seq = [_make_obs() for _ in range(2)]
    act_seq = [np.random.randn(50, 32).astype(np.float32) for _ in range(2)]

    interceptor.prefill_trajectory(obs_seq, actions=act_seq)

    # After prefill completes the facade must be back to normal mode so a
    # subsequent real ``infer`` hits the backend instead of returning a
    # synthetic hit.
    assert storage._prefill_mode is False
    assert storage._prefill_payload is None


def test_prefill_trajectory_rejects_mismatched_lengths():
    interceptor, _ = _prefill_interceptor()
    obs_seq = [_make_obs() for _ in range(3)]
    act_seq = [np.random.randn(50, 32).astype(np.float32) for _ in range(2)]

    import pytest

    with pytest.raises(ValueError, match="equal length"):
        interceptor.prefill_trajectory(obs_seq, actions=act_seq)


def test_prefill_trajectory_rejects_deferred_modes():
    interceptor, _ = _prefill_interceptor()
    obs_seq = [_make_obs()]
    act_seq = [np.random.randn(50, 32).astype(np.float32)]

    import pytest

    with pytest.raises(NotImplementedError, match="actions=None"):
        interceptor.prefill_trajectory(obs_seq, actions=None)

    with pytest.raises(NotImplementedError, match="record=True"):
        interceptor.prefill_trajectory(obs_seq, actions=act_seq, record=True)

    with pytest.raises(NotImplementedError, match="on_miss"):
        interceptor.prefill_trajectory(obs_seq, actions=act_seq, on_miss="warn")


def test_prefill_trajectory_requires_orchestrator():
    """Without an orchestrator there is no per-connection storage facade."""
    policy = FakePolicy()
    interceptor = InferenceInterceptor(policy, timer=SystemTimer(enabled=False))

    import pytest

    with pytest.raises(RuntimeError, match="requires a cache orchestrator"):
        interceptor.prefill_trajectory(
            [_make_obs()], actions=[np.zeros((50, 32), dtype=np.float32)]
        )


def test_build_prefill_payload_wraps_numpy_action():
    from openpi.cache.storage_types import CachePayload

    action = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    payload = InferenceInterceptor._build_prefill_payload(action)

    assert isinstance(payload, CachePayload)
    assert torch.equal(payload.action_chunk, torch.from_numpy(action))
    assert payload.intermediates is None
    assert payload.denoising_num_steps is None


def test_build_prefill_payload_accepts_tensor_action():
    from openpi.cache.storage_types import CachePayload

    action = torch.arange(50 * 32, dtype=torch.float32).reshape(50, 32)
    payload = InferenceInterceptor._build_prefill_payload(action)

    assert isinstance(payload, CachePayload)
    # Tensor input is passed through without copy.
    assert payload.action_chunk is action


def test_prefill_trajectory_records_query_keys_into_strategy():
    """Trajectory side effect: each prefill step must push the CP1 query_keys
    into the strategy's trajectory history, not just skip stage2/stage3.

    Without this, Step-3 spawn would emit a gap-free action chunk sequence
    but the first post-prefill real ``infer`` would see an empty trajectory
    and lose KNN context. This test guards the invariant at the strategy
    level (where trajectory memory actually lives).
    """
    from openpi.cache.orchestrator import CheckpointID
    from openpi.cache.storage_types import QuerySpec, SearchResultLite

    class _RecordingStrategy:
        """Strategy spy: captures both ``search`` and ``record_query_keys``
        invocations so the test can tell side-effect pathways apart.
        """

        def __init__(self, storage):
            self._storage = storage
            self.query_history: list[dict[str, torch.Tensor]] = []
            self.search_calls = 0

        def search(self, ctx) -> list[SearchResultLite]:
            self.search_calls += 1
            # Production strategies (WeightedRRFKNNStrategy) record inside
            # ``search`` before dispatching to storage — mirror that contract.
            self.record_query_keys(ctx.query_keys)
            return self._storage.search(
                QuerySpec(
                    query_keys=ctx.query_keys,
                    top_k=1,
                    checkpoint_id=ctx.checkpoint_id,
                )
            )

        def record_query_keys(self, query_keys):
            # Clone tensors so the recorded snapshot is not aliased to a
            # later mutated buffer; tests then inspect the snapshot shapes.
            self.query_history.append(
                {k: v.detach().clone() for k, v in query_keys.items()}
            )

        def on_episode_start(self): ...
        def on_episode_end(self): ...

    # Build interceptor with the recording strategy swapped into CP1 slot.
    fixed_state = torch.randn(1, 32)
    fixed_action = torch.randn(1, 50, 32)
    model = FakeModel(fixed_state=fixed_state, fixed_action=fixed_action)
    policy = FakePolicy(model=model)
    orch, _, storage = make_orchestrator()
    recorder = _RecordingStrategy(storage)
    orch._search_strategies[CheckpointID.CP1] = recorder

    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )
    interceptor.on_task_begin()
    interceptor.on_episode_start("exp", "task", 0)

    obs_seq = [_make_obs() for _ in range(4)]
    act_seq = [np.random.randn(50, 32).astype(np.float32) for _ in range(4)]
    interceptor.prefill_trajectory(obs_seq, actions=act_seq)

    # Search dispatched once per prefill step (gate=always_search by default).
    assert recorder.search_calls == 4
    # Trajectory history advanced by exactly one entry per prefill step.
    assert len(recorder.query_history) == 4
    # Each recorded entry must include ``robot_state`` (the configured key field).
    assert all("robot_state" in entry for entry in recorder.query_history)
