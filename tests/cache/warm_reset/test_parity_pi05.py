"""Pi0.5 K=10 bit-for-bit parity with the step_diag reference on CPU stubs (plan §9.1).

The stub model runs ``PI0Pytorch``'s own stage-3 bodies over a nonlinear
``denoise_step`` that records the float32 bit pattern of every timestep, so
"equal" below means the same t sequence, the same inputs and the same output
tensor, bit for bit (B = 1). Real-model parity is the manual GPU test's claim.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exp.step_diag import envs as E
from exp.step_diag import pi05 as P
from exp.step_diag import recorder as R
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.timing import SystemTimer
from openpi.cache.types import PI05_V1
from openpi.cache.warm_reset.pi05 import (
    build_pi05_warm_reset,
    run_pi05_continuation,
    run_pi05_self_start,
)
from openpi.cache.warm_reset.types import private_noise, resolve_plan, resolve_self_plan
from tests.cache.test_interceptor import FakePolicy, _make_obs
from tests.cache.warm_reset._support import (
    D,
    H,
    Pi05Model,
    block_of,
    pi05_config,
    pi05_payload,
    row_stage2,
    spec_for,
    warm_orchestrator,
)

PI05_MODES = {**E.WARM_VARIANT_MODES, **E.SELF_VARIANT_MODES}
TS = (0.1, 0.2, 0.3)
CASES = [(mode, t) for mode in PI05_MODES for t in TS]


def _start(seed: int = 7) -> torch.Tensor:
    return torch.randn(1, H, D, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------
# (a) the continuation loop
# ------------------------------------------------------------------


@pytest.mark.parametrize("mode,t", CASES)
def test_continuation_equals_warm_variant_stage3(mode, t):
    ref_model, new_model = Pi05Model(), Pi05Model()
    stage2 = row_stage2(0.3)
    start = _start()
    ref = P.warm_variant_stage3(ref_model, stage2, start, t, num_steps=10, variant=PI05_MODES[mode])
    plan = resolve_plan(spec_for("pi05", f"{mode}_t{t}"), PI05_V1, t)
    new = run_pi05_continuation(new_model, stage2, start, plan)
    assert new_model.t_log == ref_model.t_log
    assert all(torch.equal(a, b) for a, b in zip(new_model.x_log, ref_model.x_log, strict=True))
    assert torch.equal(new.action_chunk, ref.action_chunk)
    assert new.steps_run == len(ref_model.t_log) == plan.n_steps
    # the host replay of the plan is the t the model actually received
    got = torch.tensor([bits[0] for bits in new_model.t_log], dtype=torch.int32).view(torch.float32).tolist()
    assert got == list(plan.flow_times())


# ------------------------------------------------------------------
# (c) the self-start direct inference
# ------------------------------------------------------------------


@pytest.mark.parametrize("t", PI05_V1.timesteps)
@pytest.mark.parametrize("point", ["snapshot", "final"])
def test_self_start_equals_run_stage3_with_intermediates(t, point):
    mode = "selfwarmreset" if point == "snapshot" else "selfresetfinal"
    ref_model, new_model = Pi05Model(), Pi05Model()
    stage2 = row_stage2(-0.2)
    noise = private_noise(99, (H, D))[None, ...]
    ref = ref_model.run_stage3(stage2, noise=noise, num_steps=10, return_intermediates=True, save_timesteps=(t,))
    plan = resolve_self_plan(spec_for("pi05", f"{mode}_t{t}"), PI05_V1, t)
    (new,) = run_pi05_self_start(new_model, stage2, noise, [plan])
    assert new_model.t_log == ref_model.t_log and new.steps_run == 10
    assert torch.equal(new.action_chunk, ref.action_chunk)
    if point == "snapshot":
        assert torch.equal(new.snapshot, ref.intermediates[t])
    else:
        assert new.snapshot is None


# ------------------------------------------------------------------
# (b) + (d) the whole served decision vs Pi05DiagInterceptor
# ------------------------------------------------------------------

_META = {"task_uid": "arm:eval:4:3", "attempt": 1, "task_id": 4, "orig_init_state_idx": 3}


def _obs():
    torch.manual_seed(0)
    np.random.seed(0)
    return _make_obs()


def _step_diag_decision(tmp_path, mode: str, t: float):
    model = Pi05Model()
    orch, _ = warm_orchestrator(model, t, pi05_payload(t))
    spec = R.DiagSpec(experiment_id="parity", env_id="pi05_libero_10", arm_id=f"{mode}_t{t}", mode=mode,
                      k_full=10, action_shape=(H, D))
    icpt = P.Pi05DiagInterceptor(
        FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch,
        diag=R.DiagRecorder(spec, tmp_path / "diag"), mode=mode, warm_variant=PI05_MODES[mode],
        self_start=mode in E.SELF_VARIANT_MODES,
    )
    icpt.on_task_begin()
    icpt.on_episode_start(experiment="libero_10", task="put both pots", episode_id=3, extra_metadata=dict(_META))
    seed = icpt._diag.self_start_seed()
    return icpt.infer(_obs()), seed, model


def _production_decision(tmp_path, mode: str, t: float, seed: int):
    model = Pi05Model()
    orch, _ = warm_orchestrator(model, t, pi05_payload(t))
    spec = spec_for("pi05", f"{mode}_t{t}", evidence_dir=str(tmp_path / "evidence"))
    parts = build_pi05_warm_reset(pi05_config(block_of(spec), start_t=t), bundle_id="b", yaml_id="y", yaml_path=None)
    # Same seed integer as the reference (the production recipe differs by design, plan Q2).
    parts.session.self_seed = lambda: seed
    served = parts.wrap(
        InferenceInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch,
                             warm_reset=parts.executor)
    )
    served.on_task_begin()
    served.on_episode_start(experiment="libero_10", task="put both pots", episode_id=3, extra_metadata=dict(_META))
    return served.infer(_obs()), model


@pytest.mark.parametrize("mode,t", CASES)
def test_served_decision_equals_the_step_diag_interceptor(tmp_path, mode, t):
    ref, seed, ref_model = _step_diag_decision(tmp_path, mode, t)
    new, new_model = _production_decision(tmp_path, mode, t, seed)
    assert ref["__hit_meta__"]["hit_type"] == new["__hit_meta__"]["hit_type"] == "WARM_START"
    np.testing.assert_array_equal(new["actions"], ref["actions"])
    assert new_model.t_log == ref_model.t_log
    meta = new["__hit_meta__"]["warm_reset"]
    n = PI05_V1.remaining_steps(t)
    assert meta["continuation_nfe"] == n and meta["n_stage3_calls"] == 1
    if mode in E.SELF_VARIANT_MODES:
        assert meta["self_seed"] == seed and meta["self_direct_nfe"] == 10 and meta["decision_nfe"] == 10 + n
    else:
        assert "self_seed" not in meta and meta["decision_nfe"] == n
