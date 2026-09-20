"""CPU tests for exp/step_diag/pi05.py with the interceptor test fakes: the executed action and the
global RNG are untouched by the shadow channel, plain / full arms pin the executed step count on
the interceptor's own stage-3 binding and count real ``denoise_step`` calls, the shadow sample
matrix is reproducible from the recorded noise ids, and every MISS decision lands as a row."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.step_diag import pi05 as P
from exp.step_diag import recorder as R
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.timing import SystemTimer
from tests.cache.conftest import make_orchestrator
from tests.cache.test_interceptor import FakePolicy, _make_obs

H, D = 50, 32


class _StagedFakeModel:
    """Deterministic staged model: ``denoise_step`` is the only place the loop advances."""

    config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)

    def __init__(self) -> None:
        self.stage1_calls = self.stage2_calls = self.stage3_calls = self.step_calls = 0

    def run_stage1(self, observation):
        self.stage1_calls += 1
        return SimpleNamespace(state=torch.zeros(1, D), prefix_embs=torch.zeros(1, 4, 8))

    def run_stage2(self, stage1):
        self.stage2_calls += 1
        return SimpleNamespace(kv_cache=None)

    def sample_noise(self, shape, device, generator=None):
        return torch.randn(*shape, generator=generator)

    def denoise_step(self, x_t, k):
        self.step_calls += 1
        return x_t * 0.5 + 1.0 / k

    def run_stage3(self, stage2, noise=None, num_steps=10, return_intermediates=False, save_timesteps=(0.7, 0.5, 0.3)):
        self.stage3_calls += 1
        x = noise if noise is not None else torch.randn(1, H, D)
        for _ in range(int(num_steps)):
            x = self.denoise_step(x, num_steps)
        inter = {st: x.clone() for st in save_timesteps} if return_intermediates else None
        return SimpleNamespace(action_chunk=x, intermediates=inter)

    def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
        x = start_x
        for _ in range(int(round(float(start_t) * num_steps))):
            x = self.denoise_step(x, num_steps)
        return SimpleNamespace(action_chunk=x, intermediates=None)


def _spec(mode, **over):
    base = dict(experiment_id="e", env_id="pi05_rc", arm_id=mode, mode=mode, k_full=10, k_set=(1, 3), warm_ts=(0.3,),
                n_primary=4, n_dense_extra=2, action_shape=(H, D), config_sha="c")
    base.update(over)
    return R.DiagSpec(**base)


def _episode_start(icpt, uid="u1"):
    icpt.on_episode_start(experiment="robocasa365", task="CloseFridge", episode_id=0,
                          extra_metadata={"task_uid": uid, "attempt": 1, "task_id": 1, "orig_init_state_idx": 0, "seed": 2_000_000})


def test_plain_arm_pins_steps_on_the_executed_path_and_counts_denoise_steps(tmp_path):
    model = _StagedFakeModel()
    rec = R.DiagRecorder(_spec("plain", k_set=(), warm_ts=(), exec_steps=2), tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), diag=rec, mode="plain", exec_steps=2)
    _episode_start(icpt)
    z = torch.randn(1, H, D)
    out = icpt.infer(_make_obs(), noise=z.numpy())
    # executed action = the model's own 2-step loop from the same noise (no cache verdict involved)
    ref = _StagedFakeModel().run_stage3(None, noise=z, num_steps=2).action_chunk[0]
    assert torch.allclose(torch.as_tensor(np.asarray(out["actions"])), ref)
    assert model.step_calls == 2
    icpt.on_episode_end(True)
    import json
    rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 1 and ok[0]["executed_steps"] == 2 and ok[0]["n_stage3_calls"] == 1 and ok[0]["hit_type"] == "MISS"
    assert rows[-1]["status"] == "finalize" and rows[-1]["terminal"] and rows[-1]["outcome"] is True
    # a second interceptor without the pin (a full arm at K) runs the model default
    model2 = _StagedFakeModel()
    rec2 = R.DiagRecorder(_spec("full", k_set=(), warm_ts=(), exec_steps=10), tmp_path / "full")
    icpt2 = P.Pi05DiagInterceptor(FakePolicy(model2), timer=SystemTimer(enabled=False), diag=rec2, mode="full", exec_steps=10)
    _episode_start(icpt2)
    icpt2.infer(_make_obs(), noise=z.numpy())
    assert model2.step_calls == 10


def _shadow_pair(tmp_path, seed=7):
    """(baseline interceptor output, diag interceptor output, model, recorder) on the same noise / orchestrator."""
    torch.manual_seed(seed)
    z = torch.randn(1, H, D)
    obs = _make_obs()
    # baseline: production interceptor + orchestrator, explicit noise
    torch.manual_seed(seed)
    base_model = _StagedFakeModel()
    orch, _, _ = make_orchestrator()
    base = InferenceInterceptor(FakePolicy(base_model), timer=SystemTimer(enabled=False), orchestrator=orch)
    base.on_episode_start(experiment="robocasa365", task="CloseFridge", episode_id=0, extra_metadata={"task_uid": "u", "attempt": 1})
    base_out = base.infer(obs, noise=z.numpy().copy())
    rng_base = torch.random.get_rng_state().clone()
    # diag: shadow interceptor on the same noise
    torch.manual_seed(seed)
    model = _StagedFakeModel()
    orch2, _, _ = make_orchestrator()
    rec = R.DiagRecorder(_spec("shadow"), tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch2, diag=rec, mode="shadow")
    _episode_start(icpt)
    out = icpt.infer(obs, noise=z.numpy().copy())
    rng_diag = torch.random.get_rng_state().clone()
    return base_out, out, model, rec, icpt, rng_base, rng_diag


def test_shadow_keeps_executed_action_and_global_rng(tmp_path):
    base_out, out, model, rec, icpt, rng_base, rng_diag = _shadow_pair(tmp_path)
    assert np.array_equal(np.asarray(base_out["actions"]), np.asarray(out["actions"]))
    assert torch.equal(rng_base, rng_diag)  # the 4x(10+1+3) shadow forwards drew from private generators only
    # executed loop 10 + samples: 4 full x 10 + 4 x (1 + 3) = 56 extra denoise steps
    assert model.step_calls == 10 + 4 * 10 + 4 * (1 + 3)
    icpt.on_episode_end(False)
    import json
    rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
    ok = rows[0]
    assert ok["status"] == "ok" and ok["executed_steps"] == 10 and ok["n_full"] == 4 and len(ok["noise_ids"]) == 4
    assert ok["n_stage3_calls"] == 1  # the shadow's 16 extra run_stage3 calls are not executed-path entries
    assert ok["warm_status"] == {"0.3000": "no_candidate"}  # empty library: no top-1
    z = np.load(tmp_path / rows[-1]["arrays"])
    # same-noise pairing is reproducible from the recorded ids: a_k1[n] == loop(z_n, 1), a_full[n] == loop(z_n, 10)
    for n, seed in enumerate(ok["noise_ids"]):
        zn = R.make_noise(seed, (H, D))[None]
        ref1 = _StagedFakeModel().run_stage3(None, noise=zn, num_steps=1).action_chunk[0].numpy()
        ref10 = _StagedFakeModel().run_stage3(None, noise=zn, num_steps=10).action_chunk[0].numpy()
        assert np.allclose(z["a_k1_0000"][n], ref1) and np.allclose(z["a_full_0000"][n], ref10)
    assert np.array_equal(z["a_exec_0000"], np.asarray(out["actions"]))


def test_shadow_failure_is_an_error_row_not_an_exception(tmp_path):
    model = _StagedFakeModel()
    orch, _, _ = make_orchestrator()
    rec = R.DiagRecorder(_spec("shadow"), tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch, diag=rec, mode="shadow")
    _episode_start(icpt)
    calls = {"n": 0}
    real = model.run_stage3

    def flaky(stage2, noise=None, num_steps=10, **kw):
        calls["n"] += 1
        if calls["n"] > 1:  # the executed call succeeds, the first shadow sample raises
            raise RuntimeError("cuda oom (simulated)")
        return real(stage2, noise=noise, num_steps=num_steps, **kw)

    model.run_stage3 = flaky
    z = torch.randn(1, H, D)
    out = icpt.infer(_make_obs(), noise=z.numpy())
    assert np.asarray(out["actions"]).shape == (H, D)
    icpt.on_episode_end(True)
    import json
    rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
    assert rows[0]["status"] == "error" and "cuda oom" in rows[0]["error_reason"] and rows[-1]["status"] == "finalize"


@pytest.mark.parametrize("mode", ["plain", "shadow"])
def test_every_decision_gets_a_row_with_its_index(tmp_path, mode):
    model = _StagedFakeModel()
    kw = {"orchestrator": make_orchestrator()[0]} if mode == "shadow" else {}
    rec = R.DiagRecorder(_spec(mode, k_set=(1,) if mode == "shadow" else (), warm_ts=()), tmp_path)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), diag=rec, mode=mode,
                                 exec_steps=3 if mode == "plain" else None, **kw)
    _episode_start(icpt)
    for _ in range(3):
        icpt.infer(_make_obs(), noise=torch.randn(1, H, D).numpy())
    icpt.on_episode_end(False)
    import json
    rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
    assert [r["decision_idx"] for r in rows if r["status"] == "ok"] == [0, 1, 2] and rows[-1]["n_decisions"] == 3
