"""G0-C against real RoboCasa-k4 and LIBERO-k8 checkpoints on island B.

Run with the GR00T venv and ``--run-manual -s``; PYTHONPATH must contain this
checkout's root/src, /home/weiland/gr00t_n15 and its examples/Libero directory.
An explicit manual run fails on missing dependencies/assets, never skips.
The upstream reference runs the entire model and supplies every snapshot via
an action-encoder hook. Resume inputs round-trip through the HDF5 reader used
by the artifact builders, including its float32 storage conversion.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import pathlib

import h5py
import numpy as np
import pytest
import torch

from exp.robocasa365.bench_groot_stages import (
    assert_host,
    assert_idle_gpu,
    checkpoint_identity,
    gpu_provenance,
    sha256_file,
)
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.storage_types import CachePayload
from openpi.collect.h5_intermediates import episode_schedule, read_step_intermediates

pytestmark = pytest.mark.manual


@pytest.fixture(scope="module", params=["robocasa_k4", "libero_k8"])
def real_stack(request):
    """Load each production checkpoint with its serving schedule and source pins."""
    from gr00t.model.policy import Gr00tPolicy

    assert_host()
    assert_idle_gpu(0, os.getpid())
    torch.cuda.set_device(0)
    if request.param == "robocasa_k4":
        from exp.robocasa365.groot_data_config import RoboCasa365DataConfig
        from tests.robocasa365.test_groot_cache_manual import CHECKPOINT

        checkpoint = CHECKPOINT
        data_config = RoboCasa365DataConfig()
        steps = 4
    else:
        from custom_data_config import LiberoDataConfig

        checkpoint = pathlib.Path("/home/weiland/ckpt_n15_libero_spatial")
        data_config = LiberoDataConfig()
        steps = 8
    assert checkpoint.is_dir(), checkpoint
    identity = checkpoint_identity(checkpoint)
    policy = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag="new_embodiment",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=steps,
        device="cuda:0",
    )
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    source = pathlib.Path(inspect.getsourcefile(type(policy.model.action_head)))
    evidence = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": identity,
        "schedule_id": runner.live_schedule().schedule_id,
        "action_head_source": str(source),
        "action_head_sha256": sha256_file(source),
        "torch": torch.__version__,
        **gpu_provenance(0),
    }
    try:
        yield request.param, policy, runner, evidence
    finally:
        del runner, policy
        gc.collect()
        torch.cuda.empty_cache()
        assert_idle_gpu(0, os.getpid())


def _inputs(kind, policy, step):
    if kind == "robocasa_k4":
        from tests.robocasa365.test_groot_cache_manual import _normalized, _observation

        return _normalized(policy, _observation(step))

    from exp.libero_groot import libero_keys as keys
    from exp.libero_groot.policy_adapter import build_groot_observation
    from openpi.cache.groot.interceptor import _unsqueeze_values

    rng = np.random.default_rng(1000 + step)
    wire = {
        keys.WIRE_IMAGE: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        keys.WIRE_WRIST: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        keys.WIRE_STATE: np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.01, -0.01]) + 0.001 * step,
        keys.WIRE_PROMPT: "pick up the black bowl"
        if step == 0
        else "put the bowl on the plate",
    }
    shaped = _unsqueeze_values(build_groot_observation(wire))
    return policy.apply_transforms(
        {key: np.asarray(value) for key, value in shaped.items()}
    )


@pytest.mark.parametrize("observation_step", [0, 1])
def test_full_and_h5_resumed_loops_match_real_upstream(
    real_stack, observation_step, monkeypatch, tmp_path
):
    """Require bit-exact full/resumed actions and effective noise/time controls."""
    kind, policy, runner, evidence = real_stack
    assert_idle_gpu(0, os.getpid())
    inputs = _inputs(kind, policy, observation_step)
    head = policy.model.action_head
    schedule = runner.live_schedule()
    with runner.session():
        # Prime kernel choices before comparing separate full-model calls.
        policy.model.get_action(inputs)
        stage2 = runner.run_stage2_llm(runner.run_stage1(inputs))
    noise = torch.randn(
        1,
        head.config.action_horizon,
        head.config.action_dim,
        device="cuda:0",
        generator=torch.Generator(device="cuda:0").manual_seed(
            12345 + observation_step
        ),
    )
    captures = []
    draws = []

    def capture(module, args):
        del module
        captures.append(args[0].detach().clone())

    def fixed_randn(*args, **kwargs):
        shape = kwargs.get("size", args[0] if args else None)
        assert tuple(shape) == tuple(noise.shape)
        draws.append(shape)
        assert len(draws) == 1, "upstream RNG contract changed"
        return noise.to(device=kwargs["device"], dtype=kwargs["dtype"]).clone()

    handle = head.action_encoder.register_forward_pre_hook(capture)
    try:
        with monkeypatch.context() as patch, runner.session():
            patch.setattr(torch, "randn", fixed_randn)
            reference = policy.model.get_action(inputs)["action_pred"].clone()
    finally:
        handle.remove()
    assert len(draws) == 1
    assert len(captures) == schedule.num_steps

    with runner.session():
        full = runner.run_stage3(stage2, noise=noise).action_pred
        repeated = runner.run_stage3(stage2, noise=noise).action_pred
        changed = runner.run_stage3(stage2, noise=-noise).action_pred
    full_delta = float((full - reference).abs().max())
    assert torch.equal(full, reference), f"full vs upstream max_abs={full_delta}"
    assert torch.equal(full, repeated)
    assert not torch.equal(full, changed), "different noise must change the action"

    h5_path = tmp_path / "snapshots.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["denoise_schedule_id"] = schedule.schedule_id
        h5.attrs["denoising_num_steps"] = schedule.num_steps
        group = h5.create_group("step_0000")
        group["clean_action"] = reference[0].float().cpu().numpy()
        for index, snapshot in enumerate(captures):
            group[f"noise_action_{index}"] = snapshot[0].float().cpu().numpy()
    with h5py.File(h5_path, "r") as h5:
        recovered_schedule = episode_schedule(h5)
        snapshots, steps = read_step_intermediates(h5["step_0000"], recovered_schedule)
        payload = CachePayload(
            action_chunk=torch.from_numpy(h5["step_0000/clean_action"][:]),
            intermediates=snapshots,
            denoising_num_steps=steps,
            schedule_id=recovered_schedule.schedule_id,
        )

    resume_deltas = {}
    controls = {}
    for index in range(1, schedule.num_steps):
        start_t = schedule.snapshot_t(index)
        payload.validate_for_warm_start(schedule, start_t)
        with runner.session():
            resumed = runner.run_stage3_from(
                stage2, payload.intermediates[start_t], start_t, schedule=schedule
            )
            wrong = runner.run_stage3_from(
                stage2, captures[index - 1], start_t, schedule=schedule
            ).action_pred
        resume_deltas[str(start_t)] = float(
            (resumed.action_pred - reference).abs().max()
        )
        controls[str(start_t)] = float((wrong - reference).abs().max())
        assert resumed.steps_run == schedule.num_steps - index
        assert torch.equal(resumed.action_pred, reference), resume_deltas
        assert not torch.equal(wrong, reference), (
            f"wrong snapshot at t={start_t} was inert"
        )
    assert_idle_gpu(0, os.getpid())
    print(
        "G0_C_EVIDENCE="
        + json.dumps(
            {
                **evidence,
                "observation_step": observation_step,
                "full_max_abs": full_delta,
                "resume_max_abs": resume_deltas,
                "wrong_snapshot_max_abs": controls,
                "different_noise_max_abs": float((changed - full).abs().max()),
                "fixed_noise_replay_equal": True,
            },
            sort_keys=True,
        )
    )
