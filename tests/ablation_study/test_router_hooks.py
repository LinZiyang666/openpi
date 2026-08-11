"""InferenceInterceptor hit_executor / miss_executor hook tests (plan §9).

Uses the FakePolicy/FakeModel fixtures from tests.cache.test_interceptor and
the conftest orchestrator factory; no model weights, CPU only.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.timing import SystemTimer

from tests.cache.conftest import make_orchestrator
from tests.cache.test_interceptor import FakeModel
from tests.cache.test_interceptor import FakePolicy
from tests.cache.test_interceptor import _make_obs


def _routed_outputs() -> dict:
    return {"actions": np.zeros((10, 7), dtype=np.float32), "state": np.zeros(8)}


class RecordingExecutor:
    """Callable executor stub that records calls and supports close()."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = 0

    def __call__(self, obs: dict) -> dict:
        self.calls.append(obs)
        return _routed_outputs()

    def close(self) -> None:
        self.closed += 1


def _populated(judge=None, **hooks):
    """Build (hooked interceptor, model, obs) whose storage already holds an
    entry matching obs — arranged by a hook-free writer interceptor so the
    write path is identical to the legacy MISS bookkeeping."""
    from openpi.cache.components.write_policy import AlwaysWritePolicy

    model = FakeModel(fixed_state=torch.randn(1, 32), fixed_action=torch.randn(1, 50, 32))
    policy = FakePolicy(model=model)
    orch, _, _ = make_orchestrator(judge=judge, write_policy=AlwaysWritePolicy())
    writer = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch
    )
    obs = _make_obs()
    writer.on_task_begin()
    writer.infer(obs)  # MISS -> buffer entry
    writer.on_episode_end(success=True)
    writer.on_episode_start("test", "test", 1)
    routed = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch, **hooks
    )
    model.stage1_calls = model.stage2_calls = model.stage3_calls = 0
    return routed, model, obs


def test_executors_require_orchestrator():
    with pytest.raises(ValueError, match="orchestrator"):
        InferenceInterceptor(
            FakePolicy(), timer=SystemTimer(enabled=False), hit_executor=lambda o: o
        )


def test_no_hooks_output_unchanged():
    # Regression lock: hooks=None output carries no executor field and the
    # legacy keys are intact.
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch
    )
    result = interceptor.infer(_make_obs())
    assert set(result) >= {"actions", "state", "__hit_meta__"}
    assert "executor" not in result["__hit_meta__"]


def test_hit_executor_routes_full_hit():
    executor = RecordingExecutor()
    interceptor, model, obs = _populated(hit_executor=executor)
    obs_snapshot = copy.deepcopy(obs)

    result = interceptor.infer(obs)
    assert len(executor.calls) == 1
    assert result["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert result["__hit_meta__"]["executor"] == "override"
    np.testing.assert_array_equal(result["actions"], _routed_outputs()["actions"])
    assert model.stage2_calls == 0 and model.stage3_calls == 0
    # obs passed to the executor is byte-identical to the client obs
    # (input transforms must not mutate in place — plan §6.1 precondition).
    passed = executor.calls[0]
    np.testing.assert_array_equal(passed["state"], obs_snapshot["state"])
    for k in obs_snapshot["image"]:
        np.testing.assert_array_equal(passed["image"][k], obs_snapshot["image"][k])


def test_miss_executor_routes_miss_and_skips_stages():
    executor = RecordingExecutor()
    model = FakeModel(fixed_state=torch.randn(1, 32))
    policy = FakePolicy(model=model)
    orch, _, _ = make_orchestrator()  # empty storage -> MISS
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch,
        miss_executor=executor,
    )
    result = interceptor.infer(_make_obs())
    assert len(executor.calls) == 1
    assert result["__hit_meta__"]["hit_type"] == "MISS"
    assert result["__hit_meta__"]["executor"] == "override"
    assert model.stage1_calls == 1  # stage1 always runs (CP1 key source)
    assert model.stage2_calls == 0 and model.stage3_calls == 0


def test_miss_executor_not_called_on_hit():
    executor = RecordingExecutor()
    interceptor, model, obs = _populated(miss_executor=executor)
    result = interceptor.infer(obs)
    assert len(executor.calls) == 0  # FULL_HIT replays the cache, not the sidecar
    assert result["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert "executor" not in result["__hit_meta__"]
    assert model.stage2_calls == 0 and model.stage3_calls == 0


def test_warm_start_with_executor_raises():
    from openpi.cache.components.judge import AlwaysWarmStartJudge

    interceptor, _, obs = _populated(
        judge=AlwaysWarmStartJudge(start_t=0.5), miss_executor=RecordingExecutor()
    )
    with pytest.raises(RuntimeError, match="WARM_START"):
        interceptor.infer(obs)


def test_prefill_with_executor_raises():
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch,
        miss_executor=RecordingExecutor(),
    )
    with pytest.raises(RuntimeError, match="prefill"):
        interceptor.prefill_trajectory([_make_obs()], [np.zeros((10, 7), np.float32)])


def test_on_task_end_closes_executors():
    executor = RecordingExecutor()
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch,
        hit_executor=executor,
    )
    interceptor.on_task_end()
    assert executor.closed == 1


class ExplodingExecutor:
    def __init__(self) -> None:
        self.closed = 0

    def __call__(self, obs: dict) -> dict:
        raise RuntimeError("sidecar down")

    def close(self) -> None:
        self.closed += 1


def test_executor_exception_propagates_and_state_recovers():
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch,
        miss_executor=ExplodingExecutor(),
    )
    with pytest.raises(RuntimeError, match="sidecar down"):
        interceptor.infer(_make_obs())
    # Fail-closed must not wedge the interceptor: swap in a healthy executor
    # and the next inference on the same instance succeeds.
    interceptor._miss_executor = RecordingExecutor()  # noqa: SLF001
    result = interceptor.infer(_make_obs())
    assert result["__hit_meta__"]["executor"] == "override"


def test_collect_meta_preserved_on_routed_miss():
    executor = RecordingExecutor()
    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        FakePolicy(), timer=SystemTimer(enabled=False), orchestrator=orch,
        miss_executor=executor,
        export_collect_meta=True, collect_fields=("robot_state",), collect_kb_id="ph",
    )
    result = interceptor.infer(_make_obs())
    assert result["__hit_meta__"]["executor"] == "override"
    assert "__collect_meta__" in result
    assert result["__collect_meta__"]["kb_id"] == "ph"


def test_miss_routing_under_fake_coordinator():

    model = FakeModel(fixed_state=torch.randn(1, 32))
    policy = FakePolicy(model=model)
    submitted = []

    class FakeCoordinator:
        def submit_to_stage(self, stage, bundle_id, payload):
            submitted.append(stage)
            if stage == 1:
                return model.run_stage1(payload)
            raise AssertionError(f"stage {stage} must not be submitted on a routed MISS")

    orch, _, _ = make_orchestrator()
    interceptor = InferenceInterceptor(
        policy, timer=SystemTimer(enabled=False), orchestrator=orch,
        coordinator=FakeCoordinator(), bundle_id="b",
        miss_executor=RecordingExecutor(),
    )
    result = interceptor.infer(_make_obs())
    assert result["__hit_meta__"]["executor"] == "override"
    assert submitted == [1]  # stage1 only; stages 2/3 never reached the coordinator


def test_client_timing_row_persisted_by_runner():
    # LiberoEpisodeRunner appends one `_kind: client_timing` row per episode
    # when the injected run_episode_fn supports the keyword; doubles without
    # the keyword keep the legacy contract (signature-filtered).
    from openpi.conductor import task as _task
    from examples.libero.episode_runner import LiberoEpisodeRunner

    class _Client:
        def select_bundle(self, b): pass
        def episode_start(self, **kw): pass
        def episode_end(self, success): pass

    def _setup(task):
        return object(), object(), "prompt", 10

    def _fn_with_ct(env, client, state, desc, args, max_steps,
                    infer_recorder=None, step_callback=None, client_timing=None):
        client_timing["infer_ms"] = 12.5
        return True, None, None, None, 3

    runner = LiberoEpisodeRunner(
        type("A", (), {"seed": 0})(), _setup,
        client_factory=lambda s: _Client(), run_episode_fn=_fn_with_ct,
    )
    task = _task.EpisodeTask(
        task_uid=_task.make_task_uid("arm", "eval", 0, 0), yaml_id="arm",
        phase="eval", experiment="suite", task_id=0, episode_idx=0,
        orig_init_state_idx=0, server_host="h", server_port=1, bundle_id="arm",
        extra={"num_trials_per_task": 1},
    )
    result = runner.run(task, lambda *a, **k: None)
    ct_rows = [r for r in result.per_step_rows if r.get("_kind") == "client_timing"]
    assert len(ct_rows) == 1 and ct_rows[0]["infer_ms"] == 12.5

    def _fn_legacy(env, client, state, desc, args, max_steps,
                   infer_recorder=None, step_callback=None):
        return True, None, None, None, 3

    runner2 = LiberoEpisodeRunner(
        type("A", (), {"seed": 0})(), _setup,
        client_factory=lambda s: _Client(), run_episode_fn=_fn_legacy,
    )
    result2 = runner2.run(task, lambda *a, **k: None)
    assert not [r for r in result2.per_step_rows if r.get("_kind") == "client_timing"]
