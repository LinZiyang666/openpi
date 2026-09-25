"""Real-checkpoint GR00T warm reset parity (plan §9.1 manual; the only real-model bit claim).

Run on island B with the gr00t worktree on PYTHONPATH (check the skip count:
nothing here may be skipped once --run-manual is given)::

    SD_ENV_ID=<groot_rc | groot_libero_spatial | groot_libero_10> SD_CKPT=<checkpoint directory>
    SD_OBSERVATION=<npz with one production wire observation, allow_pickle=False>
    python -m pytest tests/cache/groot/test_warm_reset_groot_manual.py --run-manual -q

B = 1, inside the runner session: every step_diag GR00T arm (RoboCasa K = 4 or
LIBERO K = 8, cache and self starts) through ``run_groot_continuation`` /
``groot_self_start`` is bit-equal to ``groot_warm_variant_stage3`` /
``exp.step_diag.groot.groot_self_start`` on the same stage-2 handle.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual


def _runner_and_stage2(env):
    from gr00t.model.policy import Gr00tPolicy

    from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values
    from openpi.cache.groot.staged import GrootStagedRunner

    if env.benchmark == "robocasa365":
        from exp.robocasa365.groot_data_config import (
            RoboCasa365DataConfig as DataConfig,
        )
        from exp.robocasa365.groot_policy_adapter import build_groot_observation
    else:
        from custom_data_config import LiberoDataConfig as DataConfig

        from exp.libero_groot.policy_adapter import build_groot_observation

    checkpoint = pathlib.Path(os.environ["SD_CKPT"])
    assert checkpoint.is_dir()
    dc = DataConfig()
    policy = Gr00tPolicy(model_path=str(checkpoint), embodiment_tag="new_embodiment",
                         modality_config=dc.modality_config(), modality_transform=dc.transform(),
                         denoising_steps=env.k_full, device="cuda:0")
    with np.load(os.environ["SD_OBSERVATION"], allow_pickle=False) as data:
        wire = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
    obs = build_groot_observation(wire)
    if not _is_batched(obs):
        obs = _unsqueeze_values(obs)
    obs = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in obs.items()}
    runner = GrootStagedRunner(policy.model)
    normalized = policy.apply_transforms(obs)
    with runner.session():
        stage1 = runner.run_stage1(normalized)
        stage2 = runner.run_stage2_llm(stage1)
    return runner, stage2


def test_real_groot_parity():
    from exp.step_diag import envs as E
    from exp.step_diag import groot as G
    from exp.step_diag.serve_diag_groot import GROOT_VARIANT_MODES
    from openpi.cache.types import groot_n15_schedule
    from openpi.cache.warm_reset.groot import groot_self_start, run_groot_continuation
    from openpi.cache.warm_reset.types import (
        private_noise,
        resolve_plan,
        resolve_self_plan,
    )
    from tests.cache.warm_reset._arms import spec_for

    assert torch.cuda.is_available(), "run on the assigned serving GPU"
    env = E.resolve_env(os.environ["SD_ENV_ID"])
    assert env.policy == "groot"
    runner, stage2 = _runner_and_stage2(env)
    schedule = groot_n15_schedule(env.k_full)
    assert runner.live_schedule() == schedule
    if env.k_full == 4:
        arms = [f"{mode}_t{s}" for mode in GROOT_VARIANT_MODES for s in (0.75, 0.5)]
    else:
        arms = [a for a in E.LIBERO_SELF_ARMS_BY_POLICY["groot"] if E.warm_steps_of(a) is not None]
    head = runner._model.action_head  # noqa: SLF001
    shape = (int(head.config.action_horizon), int(head.config.action_dim))
    with runner.session():
        for arm in arms:
            mode, start_t = E.warm_mode_of(arm), E.warm_t_of(arm)
            variant = GROOT_VARIANT_MODES[mode]
            like = private_noise(7, shape)  # stands in for a host [H, D] library snapshot
            ref = G.groot_warm_variant_stage3(runner, stage2, like, start_t, schedule=schedule, variant=variant,
                                              num_steps=E.warm_steps_of(arm))
            plan = resolve_plan(spec_for("groot", arm), schedule, start_t)
            new = run_groot_continuation(runner, stage2, like, plan, schedule=schedule)
            assert torch.equal(new.action_pred, ref.action_pred) and new.steps_run == plan.n_steps, arm
            if E.is_self_mode(mode):
                ref_start = G.groot_self_start(runner, stage2, like, start_t, schedule=schedule, variant=variant,
                                               seed=4242)
                noise = private_noise(4242, shape)[None, ...].to(device=like.device)
                x, steps = groot_self_start(runner, stage2, noise,
                                            resolve_self_plan(spec_for("groot", arm), schedule, start_t),
                                            schedule=schedule)
                new_start = x.to(device=like.device, dtype=like.dtype).reshape(like.shape)
                assert torch.equal(new_start, ref_start) and steps == env.k_full, arm
