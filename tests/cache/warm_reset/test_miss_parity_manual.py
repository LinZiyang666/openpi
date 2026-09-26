"""Real-checkpoint parity: per-bundle ``plain_k`` / ``full`` vs the process-level step_diag arm (B = 1).

SD_ENV_ID=<pi05_rc | pi05_libero_spatial | pi05_libero_10> SD_CKPT=<checkpoint directory>
SD_OBSERVATION=<npz with one production wire observation, allow_pickle=False>
python -m pytest tests/cache/warm_reset/test_miss_parity_manual.py --run-manual -q -s

For every k in (1, 2, 3, 10) and one fixed noise: the production stack
``_wrap_policy`` builds for a library-free ``miss`` bundle -- direct and through
the Pi0.5 coordinator adapter with one request per bucket -- returns the same
action bits as the step_diag plain / full arm (``Pi05DiagInterceptor``, no
cache, ``exec_steps`` pinned for the process). The production stacks run first:
the step_diag interceptor wraps the model instance's stage-3 entries (counting
wrappers that forward unchanged). Missing assets or GPU FAIL once --run-manual
is requested (a skip is not evidence).
"""

from __future__ import annotations

import os
import pathlib
import types

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.manual

KS = (1, 2, 3, 10)
EPISODE = {"experiment": "parity", "task": "parity task", "episode_id": 0}
EXTRA = {"task_uid": "y:eval:0:0", "attempt": 1, "task_id": 0, "orig_init_state_idx": 0}


def _policy():
    from exp.step_diag import envs as E
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
    return env, policy, wire


class _OneRequestCoordinator:
    """The coordinator's Pi0.5 adapter with exactly one request per bucket (B = 1)."""

    def __init__(self, model, device) -> None:
        from openpi.serving.batching_coordinator import Pi05StageBatcher

        self._batcher = Pi05StageBatcher(model, device)
        self.miss_steps: list = []

    def submit_to_stage(self, stage_id, bundle_id, payload, **kwargs):
        from openpi.serving.batching_coordinator import Stage3MissPayload

        if stage_id == 1:
            return self._batcher.run_stage1_batch([payload])[0]
        if stage_id == 2:
            return self._batcher.run_stage2_batch([payload], capture=False)[0]
        assert isinstance(payload, Stage3MissPayload)
        self.miss_steps.append(payload.num_steps)
        return self._batcher.run_stage3_miss([payload], num_steps=payload.num_steps,
                                             save_timesteps_per_request=[payload.save_timesteps])[0]


def _production(tmp_path, monkeypatch, env, policy, k: int, coordinator):
    from exp.warm_reset.envs import get_env
    from exp.warm_reset.plan import library_free_base
    from openpi.cache.config import build_shared_storage, load_cache_config
    from openpi.serving import websocket_policy_server as wps
    from scripts import serve_policy

    raw = library_free_base(get_env(env.env_id))
    raw["miss"] = {"num_steps": k, "evidence_dir": str(tmp_path / f"ev_{k}")}
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"miss_k{k}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_cache_config(path)
    bundle = types.SimpleNamespace(cache_config=cfg, shared_storage=build_shared_storage(cfg), yaml_id=f"k{k}",
                                   config_path=str(path))
    monkeypatch.setattr(wps, "get_current_cache_bundle", lambda bundle_id=None: bundle)
    served = serve_policy._wrap_policy(policy, serve_policy.Args(), quiet=True, eager=True,
                                       shared_cache={"coordinator": coordinator} if coordinator else None,
                                       bundle_id=f"k{k}")
    served.on_task_begin()
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    return served


@torch.no_grad()
def test_real_per_bundle_miss_equals_the_process_level_plain_arm(tmp_path, monkeypatch):
    from exp.step_diag import pi05 as P
    from exp.step_diag import recorder as R
    from openpi.cache.timing import SystemTimer
    from openpi.cache.warm_reset.types import private_noise

    env, policy, wire = _policy()
    model = policy._model  # noqa: SLF001
    shape = (model.config.action_horizon, model.config.action_dim)
    noises = dict.fromkeys(KS, private_noise(20260925, shape).numpy())  # one noise: only k differs
    produced = {}
    for k in KS:
        direct = _production(tmp_path / "direct", monkeypatch, env, policy, k, None)
        coord = _OneRequestCoordinator(model, torch.device("cuda"))
        batched = _production(tmp_path / "coord", monkeypatch, env, policy, k, coord)
        produced[k] = (direct.infer(dict(wire), noise=noises[k]), batched.infer(dict(wire), noise=noises[k]))
        assert coord.miss_steps == [k]
    for k in KS:
        mode = "full" if k == env.k_full else "plain"
        spec = R.DiagSpec(experiment_id="parity", env_id=env.env_id, arm_id="full" if mode == "full" else f"plain_k{k}",
                          mode=mode, k_full=env.k_full, action_shape=shape, exec_steps=k)
        ref_icpt = P.Pi05DiagInterceptor(policy, timer=SystemTimer(enabled=False), eager=True,
                                         diag=R.DiagRecorder(spec, tmp_path / f"diag_{k}"), mode=mode, exec_steps=k)
        ref_icpt.on_task_begin()
        ref_icpt.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
        ref = ref_icpt.infer(dict(wire), noise=noises[k])
        direct, batched = produced[k]
        assert np.array_equal(direct["actions"], ref["actions"]), k
        assert np.array_equal(batched["actions"], ref["actions"]), k
        assert direct["__hit_meta__"]["miss_nfe"] == batched["__hit_meta__"]["miss_nfe"] == k
        print(f"k={k}: per-bundle direct / coordinator == process-level step_diag (bit-equal)")
    distinct = {k: produced[k][0]["actions"].tobytes() for k in KS}
    assert len(set(distinct.values())) == len(KS)  # same noise: every k really ran a different loop
