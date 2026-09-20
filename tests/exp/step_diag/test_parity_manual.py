"""Real-checkpoint G2/Verify parity gate, one environment per serving Python.

SD_ENV_ID=<one of envs.ENVS> SD_CKPT=<checkpoint directory>
SD_OBSERVATION=<npz containing one production wire observation, allow_pickle=False>
python -m pytest tests/exp/step_diag/test_parity_manual.py --run-manual -q

Run for all six environments on the assigned serving hosts. Missing assets/GPU
FAIL once --run-manual is requested. Inputs are real wire observations captured
before intervention; string prompts are scalar unicode arrays in the NPZ.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib

import numpy as np
import pytest
import torch

from exp.step_diag import envs as E
from exp.step_diag import recorder as R

pytestmark = pytest.mark.manual


def _state():
    return [
        torch.random.get_rng_state().clone(),
        *[s.clone() for s in torch.cuda.get_rng_state_all()],
    ]


def _equal(a, b):
    assert set(a) == set(b)
    for key in a:
        if key in ("policy_timing", "__hit_meta__"):
            continue
        np.testing.assert_array_equal(
            np.asarray(a[key]), np.asarray(b[key]), err_msg=key
        )


@contextlib.contextmanager
def _restore_methods(model):
    originals = {
        key: getattr(model, key)
        for key in (
            "run_stage1",
            "run_stage2",
            "run_stage3",
            "run_stage3_from",
            "denoise_step",
        )
        if hasattr(model, key)
    }
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(model, key, value)


def test_real_plain_and_shadow_parity(tmp_path):
    from exp.step_diag.groot import GrootDiagPolicy
    from openpi.cache.timing import SystemTimer

    assert torch.cuda.is_available(), "run on the assigned serving GPU"
    env = E.resolve_env(os.environ["SD_ENV_ID"])
    checkpoint = pathlib.Path(os.environ["SD_CKPT"])
    assert checkpoint.is_dir()
    with np.load(os.environ["SD_OBSERVATION"], allow_pickle=False) as data:
        wire = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
    if env.policy == "pi05":
        from exp.step_diag.pi05 import Pi05DiagInterceptor
        from openpi.training import config
        from openpi.policies.policy_config import create_trained_policy
        from scripts.serve_policy import _disable_compile_for_serving

        policy = create_trained_policy(
            _disable_compile_for_serving(
                config.get_config(
                    "pi05_robocasa" if env.benchmark == "robocasa365" else "pi05_libero"
                )
            ),
            checkpoint,
            pytorch_device="cuda",
        )
        model, obs = policy._model, wire
    else:
        from gr00t.model.policy import Gr00tPolicy
        from openpi.cache.groot.staged import GrootStagedRunner

        if env.benchmark == "robocasa365":
            from exp.robocasa365.groot_data_config import RoboCasa365DataConfig
            from exp.robocasa365.groot_policy_adapter import build_groot_observation

            dc = RoboCasa365DataConfig()
        else:
            from custom_data_config import LiberoDataConfig
            from exp.libero_groot.policy_adapter import build_groot_observation

            dc = LiberoDataConfig()
        policy = Gr00tPolicy(
            model_path=str(checkpoint),
            embodiment_tag="new_embodiment",
            modality_config=dc.modality_config(),
            modality_transform=dc.transform(),
            denoising_steps=env.k_full,
            device="cuda:0",
        )
        model, obs = policy.model, build_groot_observation(wire)
    results = []
    captured_stage2 = {}
    for mode, k in [("plain", k) for k in (*env.k_set, env.k_full)] + [
        ("shadow", env.k_full)
    ]:
        spec = R.DiagSpec(
            experiment_id="manual_parity",
            env_id=env.env_id,
            arm_id=f"{mode}_{k}",
            mode=mode,
            k_full=env.k_full,
            k_set=env.k_set if mode == "shadow" else (),
            warm_ts=env.warm_ts if mode == "shadow" else (),
            action_shape=(env.action_horizon, env.action_dim),
            exec_steps=k,
        )
        rec = R.DiagRecorder(spec, tmp_path / f"{mode}_{k}")
        calls = []
        if env.policy == "pi05":
            policy._sample_kwargs["num_steps"] = k

            def reference():
                return policy.infer(dict(obs))
        else:
            model.action_head.num_inference_timesteps = k

            def reference():
                return policy.get_action(dict(obs))
        torch.manual_seed(314159)
        baseline = reference()
        baseline_rng = _state()
        if mode == "plain" and k == env.k_full:
            torch.manual_seed(271828)
            other = reference()
            keys = set(baseline) - {"policy_timing", "__hit_meta__"}
            assert any(
                not np.array_equal(np.asarray(baseline[key]), np.asarray(other[key]))
                for key in keys
            ), "different full-loop noise produced identical actions"
        with _restore_methods(model):
            if env.policy == "pi05":
                run2 = model.run_stage2

                def capture_stage2(*args, **kwargs):
                    result = run2(*args, **kwargs)
                    captured_stage2["value"] = result
                    return result

                model.run_stage2 = capture_stage2
                served = Pi05DiagInterceptor(
                    policy,
                    timer=SystemTimer(enabled=False),
                    diag=rec,
                    mode=mode,
                    exec_steps=k if mode == "plain" else None,
                )
            else:
                runner = GrootStagedRunner(model)
                served = GrootDiagPolicy(
                    policy,
                    runner,
                    orchestrator=None,
                    diag=rec,
                    schedule=env.schedule if mode == "shadow" else None,
                    shadow=mode == "shadow",
                )
                handle = model.action_head.action_encoder.register_forward_pre_hook(
                    lambda *args: calls.append(1)
                )
            served.on_episode_start(
                experiment=env.benchmark,
                task="parity",
                episode_id=0,
                extra_metadata=dict(
                    task_uid="parity", attempt=1, orig_init_state_idx=0, seed=2_000_000
                ),
            )
            torch.manual_seed(314159)
            actual = (
                served.infer(dict(obs))
                if env.policy == "pi05"
                else served.get_action(dict(obs))
            )
            actual_rng = _state()
            served.on_episode_end(True)
            if env.policy == "groot":
                handle.remove()
        # Timing/hit metadata is additive; compare every action field.
        action_keys = set(baseline) - {"policy_timing", "__hit_meta__"}
        _equal(
            {k: baseline[k] for k in action_keys}, {k: actual[k] for k in action_keys}
        )
        assert all(torch.equal(a, b) for a, b in zip(baseline_rng, actual_rng))
        rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
        row = rows[0]
        assert (
            row["status"] == "ok"
            and row["executed_steps"] == k
            and row["n_stage3_calls"] == 1
        )
        if env.policy == "groot":
            assert len(calls) == k + row["shadow_nfe"]
        results.append(
            {
                "mode": mode,
                "k": k,
                "actual_calls": len(calls) if calls else row["executed_steps"],
            }
        )
    if env.policy == "pi05":
        stage2 = captured_stage2["value"]
        with _restore_methods(model), torch.no_grad():
            calls = []
            step = model.denoise_step

            def counted(*args, **kwargs):
                calls.append(1)
                return step(*args, **kwargs)

            model.denoise_step = counted
            full = model.run_stage3(
                stage2,
                num_steps=env.k_full,
                return_intermediates=True,
                save_timesteps=env.warm_ts,
            )
            assert len(calls) == env.k_full
            for t in env.warm_ts:
                calls.clear()
                resumed = model.run_stage3_from(
                    stage2, full.intermediates[t], t, num_steps=env.k_full
                )
                assert len(calls) == env.remaining_steps(t)
                assert torch.equal(resumed.action_chunk, full.action_chunk)
                results.append({"warm_t": t, "actual_calls": len(calls)})
    else:
        from openpi.cache.groot.interceptor import _unsqueeze_values

        with runner.session():
            normalized = policy.apply_transforms(_unsqueeze_values(dict(obs)))
            stage2 = runner.run_stage2_llm(runner.run_stage1(normalized))
            snapshots = []
            hook = model.action_head.action_encoder.register_forward_pre_hook(
                lambda module, args: snapshots.append(args[0].detach().clone())
            )
            full = runner.run_stage3(stage2)
            hook.remove()
            assert len(snapshots) == env.k_full
            for t in env.warm_ts:
                calls = []
                hook = model.action_head.action_encoder.register_forward_pre_hook(
                    lambda *args: calls.append(1)
                )
                index = int(round(t * env.k_full))
                resumed = runner.run_stage3_from(
                    stage2, snapshots[index][0].float().cpu(), t, schedule=env.schedule
                )
                hook.remove()
                assert len(calls) == env.remaining_steps(
                    t
                ) and resumed.steps_run == len(calls)
                assert torch.equal(resumed.action_pred, full.action_pred)
                results.append({"warm_t": t, "actual_calls": len(calls)})
    evidence = pathlib.Path("exp/step_diag/analysis") / f"parity_{env.env_id}.json"
    evidence.write_text(
        json.dumps(
            {
                "env_id": env.env_id,
                "checkpoint_sha256": E.sha256_tree(checkpoint),
                "results": results,
                "torch": torch.__version__,
            },
            indent=1,
        )
    )
