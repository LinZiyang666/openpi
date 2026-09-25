"""Real-checkpoint Pi0.5 warm reset parity (plan §9.1 / §9.2 manual; the only real-model bit claim).

SD_ENV_ID=<pi05_rc | pi05_libero_spatial | pi05_libero_10> SD_CKPT=<checkpoint directory>
SD_OBSERVATION=<npz with one production wire observation, allow_pickle=False>
python -m pytest tests/cache/warm_reset/test_parity_manual.py --run-manual -q

B = 1: every step_diag Pi0.5 arm (cache and self) through the new entries is
bit-equal to the step_diag functions, and the coordinator adapter with one
request per bucket is bit-equal to the direct call. B = 4: routing, per-row
captures and measured counts are asserted; the numerical deviation from the
per-row B = 1 runs is only reported. Missing assets or GPU FAIL once
--run-manual is requested (a skip is not evidence).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual

TS = (0.1, 0.2, 0.3)


def _model_and_stage2():
    import jax

    from exp.step_diag import envs as E
    from openpi.models import model as _model
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training import config
    from scripts.serve_policy import _disable_compile_for_serving

    assert torch.cuda.is_available(), "run on the assigned serving GPU"
    env = E.resolve_env(os.environ["SD_ENV_ID"])
    assert env.policy == "pi05"
    checkpoint = pathlib.Path(os.environ["SD_CKPT"])
    assert checkpoint.is_dir()
    name = "pi05_robocasa" if env.benchmark == "robocasa365" else "pi05_libero"
    policy = create_trained_policy(_disable_compile_for_serving(config.get_config(name)), checkpoint,
                                   pytorch_device="cuda")
    with np.load(os.environ["SD_OBSERVATION"], allow_pickle=False) as data:
        wire = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
    model = policy._model  # noqa: SLF001
    inputs = policy._input_transform(jax.tree.map(lambda x: x, wire))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda")[None, ...], inputs)
    with torch.no_grad():
        stage1 = model.run_stage1(_model.Observation.from_dict(inputs))
        stage2 = model.run_stage2(stage1)
    return model, stage2, (model.config.action_horizon, model.config.action_dim)


@torch.no_grad()
def test_real_pi05_parity_and_batching():
    from exp.step_diag import envs as E
    from exp.step_diag import pi05 as P
    from openpi.cache.types import PI05_V1
    from openpi.cache.warm_reset.pi05 import run_pi05_continuation, run_pi05_self_start
    from openpi.cache.warm_reset.types import (
        private_noise,
        resolve_plan,
        resolve_self_plan,
    )
    from openpi.serving.batching_coordinator import (
        Pi05StageBatcher,
        Stage3WarmResetPayload,
    )
    from tests.cache.warm_reset._support import spec_for

    model, stage2, shape = _model_and_stage2()
    device = stage2.stage1.state.device
    noise = private_noise(20260925, shape)[None, ...].to(device)
    modes = {**E.WARM_VARIANT_MODES, **E.SELF_VARIANT_MODES}
    batcher = Pi05StageBatcher(model, device)
    for t in TS:
        full = model.run_stage3(stage2, noise=noise, num_steps=10, return_intermediates=True, save_timesteps=(t,))
        for mode, variant in modes.items():
            arm = f"{mode}_t{t}"
            plan = resolve_plan(spec_for("pi05", arm), PI05_V1, t)
            start = full.action_chunk if plan.point == "final" else full.intermediates[t]
            ref = P.warm_variant_stage3(model, stage2, start, t, num_steps=10, variant=variant)
            new = run_pi05_continuation(model, stage2, start, plan)
            assert torch.equal(new.action_chunk, ref.action_chunk), arm
            one = batcher.run_stage3_warm_reset([Stage3WarmResetPayload(stage2, start[0], plan)])[0]
            assert torch.equal(one.action_chunk, ref.action_chunk) and one.steps_run == plan.n_steps, arm
            if mode in E.SELF_VARIANT_MODES:
                self_plan = resolve_self_plan(spec_for("pi05", arm), PI05_V1, t)
                (direct,) = run_pi05_self_start(model, stage2, noise, [self_plan])
                want = full.action_chunk if self_plan.capture_index is None else full.intermediates[t]
                got = direct.action_chunk if self_plan.capture_index is None else direct.snapshot
                assert torch.equal(got, want) and direct.steps_run == 10, arm
                (bucket_one,) = batcher.run_stage3_warm_reset([Stage3WarmResetPayload(stage2, noise[0], self_plan)])
                assert torch.equal(bucket_one.action_chunk, direct.action_chunk), arm

    # B = 4, one continuation bucket and one self bucket: routing / counts asserted, deviation reported.
    stage2s = [stage2] * 4
    xs = [private_noise(100 + i, shape).to(device) for i in range(4)]
    cont = resolve_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.2)
    outs = batcher.run_stage3_warm_reset([Stage3WarmResetPayload(s, x, cont) for s, x in zip(stage2s, xs)])
    self_plans = [resolve_self_plan(spec_for("pi05", "selfwarmreset_t0.2"), PI05_V1, t) for t in TS]
    self_plans.append(resolve_self_plan(spec_for("pi05", "selfresetfinal_t0.2"), PI05_V1, 0.2))
    selfs = batcher.run_stage3_warm_reset([Stage3WarmResetPayload(s, x, p) for s, x, p in zip(stage2s, xs, self_plans)])
    deltas = []
    for x, out in zip(xs, outs):
        ref = run_pi05_continuation(model, stage2, x[None], cont)
        assert out.steps_run == 2
        deltas.append(float((out.action_chunk - ref.action_chunk).abs().max()))
    for x, plan, out in zip(xs, self_plans, selfs):
        (ref,) = run_pi05_self_start(model, stage2, x[None], [plan])
        assert out.steps_run == 10 and (out.snapshot is None) == (plan.capture_index is None)
        if out.snapshot is not None:
            deltas.append(float((out.snapshot - ref.snapshot).abs().max()))
    print(f"B=4 max|delta| vs per-row B=1: {max(deltas):.3e}")
