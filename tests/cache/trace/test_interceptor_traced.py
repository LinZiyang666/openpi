"""``InferenceInterceptor`` under the trace runtime (plan §5, §12-E).

Build-cache form first (no orchestrator): the full inference is the executed
action, ``noise_action_0..9`` are the loop's own inputs, prefix tokens are the
CP1 slice of ``prefix_embs``, the raw observation is captured before the
transform chain, and ``save_timesteps=None`` is never passed to the model.
Also: ``trace=None`` never reaches ``_infer_traced``; construction refuses the
incompatible wrappers; ``verdict_aware`` keeps a MISS step on the global RNG
(bit-identical noise to the legacy path under the same seed).
"""

from __future__ import annotations

import inspect

import h5py
import numpy as np
import pytest
import torch

from openpi.cache import interceptor as interceptor_mod
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TraceRuntime
from openpi.cache.timing import SystemTimer
from openpi.cache.types import PI05_V1
from tests.cache.trace.conftest import (
    TraceFakeModel,
    TraceFakePolicy,
    make_obs,
    make_plan,
)


def _runtime(out_dir, **plan_overrides) -> tuple[TraceRuntime, TraceWriter]:
    plan = make_plan(**plan_overrides)
    writer = TraceWriter(str(out_dir), queue_steps=8)
    sink = H5TraceSink(out_dir, plan=plan, writer=writer)
    return TraceRuntime(plan=plan, sink=sink, twins=None), writer


def _interceptor(policy, runtime):
    return InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True, trace=runtime)


def test_legacy_default_save_timesteps_matches_model_signature():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    default = inspect.signature(PI0Pytorch.run_stage3).parameters["save_timesteps"].default
    assert tuple(default) == interceptor_mod._LEGACY_SAVE_TIMESTEPS


def test_trace_none_never_calls_infer_traced(monkeypatch):
    policy = TraceFakePolicy()
    it = InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True)
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not be called")

    monkeypatch.setattr(it, "_infer_traced", boom)
    out = it.infer(make_obs())
    assert called["n"] == 0
    assert out["actions"].shape == (50, 32)


def test_construction_mutex():
    policy = TraceFakePolicy()
    rt, _ = _runtime("/tmp/never-used")
    with pytest.raises(ValueError, match="hit_executor"):
        InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True,
                             trace=rt, hit_executor=lambda o: {}, orchestrator=object())
    with pytest.raises(ValueError, match="shadow teacher"):
        InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True,
                             trace=rt, shadow_teacher=object())
    with pytest.raises(ValueError, match="export_collect_meta"):
        InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True,
                             trace=rt, export_collect_meta=True)


@pytest.mark.parametrize("build", [False, True])
def test_no_library_trace_records_full_inference(out_dir, build):
    model = TraceFakeModel(seed=1)
    policy = TraceFakePolicy(model)
    rt, writer = _runtime(
        out_dir,
        record_noise_actions=build,
        save_timesteps=PI05_V1.timesteps if build else None,
        fail_loud=build,
    )
    it = _interceptor(policy, rt)
    it.on_task_begin()
    it.on_episode_start("exp", "task", 7, "ep7", {"task_uid": "u7", "attempt": 2})
    torch.manual_seed(123)
    out = it.infer(make_obs(1))
    torch.manual_seed(123)
    expected_noise = model.sample_noise((1, 50, 32), "cpu")
    expected = model.run_stage3(policy._model.run_stage2(None), noise=expected_noise, num_steps=10)
    np.testing.assert_allclose(out["actions"], expected.action_chunk[0].numpy())
    assert out["__hit_meta__"]["hit_type"] == "MISS"
    assert out["__hit_meta__"]["trace"]["executed_arm"] == "full_inference"
    # save_timesteps: build passes the whole schedule, diagnostic passes nothing.
    stage3_calls = [kw for name, kw in model.calls if name == "stage3"]
    assert len(stage3_calls) == 2  # ours + the expected replay above
    assert stage3_calls[0]["return_intermediates"] is True
    if build:
        assert tuple(stage3_calls[0]["save_timesteps"]) == tuple(PI05_V1.timesteps)
    else:
        assert stage3_calls[0]["save_timesteps"] == (0.7, 0.5, 0.3)
    it.on_episode_end(True)
    it.on_task_end()
    assert writer.drain(timeout=10).ok

    with h5py.File(out_dir / "exp" / "ep7.h5", "r") as f:
        assert f.attrs["num_steps"] == 1
        assert f.attrs["prompt"] == "pick up the bowl"
        assert f.attrs["trace_task_uid"] == "u7" and f.attrs["trace_attempt"] == 2
        g = f["step_0000"]
        # Prefix tokens are the CP1 slice of prefix_embs, fp16.
        np.testing.assert_array_equal(
            g["vision_0"][...], model.prefix[0, 0:256].to(torch.float16).numpy()
        )
        np.testing.assert_array_equal(
            g["prompt_emb"][...], model.prefix[0, 768:].to(torch.float16).numpy()
        )
        np.testing.assert_array_equal(g["robot_state"][...], model.state[0].numpy())
        np.testing.assert_allclose(g["clean_action"][...], out["actions"])
        if build:
            np.testing.assert_array_equal(g["noise_action_0"][...], expected_noise[0].numpy())
            for i in range(1, 10):
                assert f"noise_action_{i}" in g
            np.testing.assert_array_equal(
                g["noise_action_1"][...], expected.intermediates[0.9][0].numpy()
                if expected.intermediates else g["noise_action_1"][...],
            )
        else:
            assert not any(k.startswith("noise_action_") for k in g.keys())
        tg = g["trace"]
        assert tg.attrs["executed_arm"] == "full_inference"
        raw = tg["raw_images"]
        assert list(raw.keys()) == ["observation%2Fimage"]
        assert raw["observation%2Fimage"].shape == (224, 224, 3)
        assert tg["raw_state"].shape == (32,)
        np.testing.assert_allclose(tg["actions"]["executed"][...], out["actions"])
        assert "search" not in tg  # no orchestrator: nothing to search
        assert "query_keys" not in tg


def test_verdict_aware_miss_noise_equals_legacy_path(out_dir):
    """Non-concurrent, no library: the trace full inference draws the same
    global-RNG noise the legacy infer would have drawn under the same seed."""
    model_a, model_b = TraceFakeModel(seed=3), TraceFakeModel(seed=3)
    legacy = InferenceInterceptor(TraceFakePolicy(model_a), timer=SystemTimer(enabled=False), eager=True)
    rt, writer = _runtime(out_dir)
    traced = _interceptor(TraceFakePolicy(model_b), rt)
    traced.on_task_begin()
    traced.on_episode_start("exp", "task", 1, "e1")
    torch.manual_seed(77)
    a = legacy.infer(make_obs(2))["actions"]
    torch.manual_seed(77)
    b = traced.infer(make_obs(2))["actions"]
    np.testing.assert_array_equal(a, b)
    traced.on_episode_end(True)
    assert writer.drain(timeout=10).ok


def test_diagnostic_mode_writer_error_is_reported_not_raised(out_dir, monkeypatch):
    rt, writer = _runtime(out_dir, fail_loud=False)
    it = _interceptor(TraceFakePolicy(), rt)
    it.on_task_begin()
    it.on_episode_start("exp", "task", 1, "e1")

    def boom(step):
        raise RuntimeError("queue dead")

    monkeypatch.setattr(rt.sink, "record_step", boom)
    out = it.infer(make_obs())
    assert "writer_error" in out["__hit_meta__"]["trace"]
    rt.plan = make_plan(fail_loud=True)
    with pytest.raises(RuntimeError, match="queue dead"):
        it.infer(make_obs())


def test_same_device_treats_indexless_cuda_as_current():
    """Regression (GPU parity gate): ``"cuda"`` vs a tensor on ``cuda:0`` is the same device."""
    from openpi.cache.interceptor import _same_device

    assert _same_device(torch.device("cpu"), "cpu")
    assert not _same_device(torch.device("cpu"), "cuda")
    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        assert _same_device(torch.device("cuda", cur), "cuda")
        assert _same_device(torch.device("cuda", cur), f"cuda:{cur}")
        assert not _same_device(torch.device("cuda", cur), f"cuda:{cur + 1}")


def test_interceptor_attaches_runtime_twins_to_the_orchestrator():
    """The Pi0.5 interceptor hands ``TraceRuntime.twins`` to the orchestrator."""
    from openpi.cache import interceptor as mod

    calls = []

    class _Orch:
        def attach_trace_twins(self, twins):
            calls.append(twins)

    marker = object()
    mod._attach_trace_twins(_Orch(), types_ns(twins=marker))
    mod._attach_trace_twins(_Orch(), types_ns(twins=None))
    mod._attach_trace_twins(None, types_ns(twins=marker))
    assert calls == [marker]
    with pytest.raises(TypeError, match="cannot take trace twins"):
        mod._attach_trace_twins(object(), types_ns(twins=marker))


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)
