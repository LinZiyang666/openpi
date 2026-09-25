"""Shared CPU fixtures of the warm reset tests (plan warm_continuation_first_class §9).

* ``Pi05Model`` runs ``PI0Pytorch``'s own stage-3 bodies (``run_stage3`` with
  intermediates, ``run_stage3_from``, ``_stage3_with_intermediates``) over a
  nonlinear ``denoise_step`` that records the float32 bit pattern of every
  timestep row and every input, so parity tests compare the production loops.
* the plan §4.3.4 arm table (``spec_for`` ...) lives in the jax-free
  ``_arms`` module and is re-exported here.
* ``pi05_config`` / ``warm_orchestrator`` build a validated config with a
  ``warm_reset`` block and a WARM_START orchestrator over one stored entry.
* ``pi05_stack`` / ``run_episode`` / ``expected_for`` drive the whole served
  Pi0.5 stack (interceptor + executor + evidence wrapper) through an episode
  and build its trusted admission expectation.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Optional

import torch
from transformers import DynamicCache

from openpi.cache.components.judge import AlwaysWarmStartJudge
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    GateConfig,
    JudgeConfig,
    KeyFieldConfig,
    KeysConfig,
    SearchStrategyConfig,
    WarmResetConfig,
    _dict_to_dataclass,
)
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import PI05_V1, CheckpointID
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Stage1Output, Stage2Output
from tests.cache.conftest import insert_entry, make_orchestrator
from tests.cache.warm_reset._arms import (  # noqa: F401 - re-exported for the tests
    BASE_MODES,
    DEFAULT_SEED_KEYS,
    block_of,
    spec_for,
    split_mode,
    warm_arms,
)

H, D = 10, 8


# ------------------------------------------------------------------
# Pi0.5 model stand-in
# ------------------------------------------------------------------


def t_bits(timestep: torch.Tensor) -> list[int]:
    """float32 bit patterns of a timestep vector (exact comparison of t)."""
    return timestep.detach().to(torch.float32).contiguous().view(torch.int32).tolist()


class Pi05Model:
    """Production stage-3 bodies over a recording, row-wise nonlinear ``denoise_step``."""

    run_stage3 = PI0Pytorch.run_stage3
    run_stage3_from = PI0Pytorch.run_stage3_from
    _stage3_with_intermediates = PI0Pytorch._stage3_with_intermediates

    def __init__(self, state: Optional[torch.Tensor] = None) -> None:
        self.config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)
        self.state = torch.linspace(-1.0, 1.0, D)[None] if state is None else state
        self.t_log: list[list[int]] = []
        self.x_log: list[torch.Tensor] = []
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.t_log.append(t_bits(timestep))
        self.x_log.append(x_t.detach().clone())
        t = timestep.to(torch.float32)[:, None, None]
        s = state.to(torch.float32).mean(dim=-1)[:, None, None]
        return torch.sin(3.0 * x_t + 2.0 * t) * (1.0 + t * t) + 0.25 * s

    def _stage3_action_expert(self, state, prefix_pad_masks, past_key_values, noise, num_steps):
        out, _ = self._stage3_with_intermediates(state, prefix_pad_masks, past_key_values, noise, num_steps, ())
        return out

    def run_stage1(self, observation):
        state = observation["state"] if isinstance(observation, dict) else observation.state
        b = state.shape[0] if state.ndim >= 2 else 1
        return stage1_of(self.state.expand(b, -1).clone())

    def run_stage2(self, stage1):
        return stage2_of(stage1)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)


def stage1_of(state: torch.Tensor) -> Stage1Output:
    b = state.shape[0]
    return Stage1Output(
        state=state,
        prefix_embs=torch.zeros(b, 2, 4),
        prefix_pad_masks=torch.ones(b, 2, dtype=torch.bool),
        prefix_att_2d_masks_4d=torch.zeros(b, 1, 2, 2),
        prefix_position_ids=torch.zeros(b, 2, dtype=torch.int64),
    )


def stage2_of(stage1: Stage1Output) -> Stage2Output:
    b = stage1.state.shape[0]
    cache = DynamicCache()
    cache.update(torch.zeros(b, 1, 2, 4), torch.zeros(b, 1, 2, 4), layer_idx=0)
    return Stage2Output(stage1=stage1, past_key_values=cache)


def row_stage2(value: float) -> Stage2Output:
    """A B=1 stage-2 handle whose state (and so the stub's velocity) is row-specific."""
    return stage2_of(stage1_of(torch.full((1, D), float(value))))


# ------------------------------------------------------------------
# Config / orchestrator
# ------------------------------------------------------------------


def pi05_config(block: Optional[dict], *, start_t: float = 0.2, **extra) -> CacheConfig:
    """A Pi0.5 ``always_warm_start`` config (legacy schedule reading) with an optional block."""
    cfg = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": D}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(type="always_warm_start", start_t=start_t),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
        warm_reset=None if block is None else _dict_to_dataclass(WarmResetConfig, block),
    )
    for key, value in extra.items():
        setattr(cfg, key, value)
    return cfg


def pi05_payload(start_t: float, *, seed: int = 0) -> CachePayload:
    """A stored Pi0.5 entry with a snapshot at ``start_t`` and a distinct final action."""
    gen = torch.Generator().manual_seed(seed)
    return CachePayload(
        action_chunk=torch.randn(H, D, generator=gen),
        intermediates={start_t: torch.randn(H, D, generator=gen)},
        denoising_num_steps=PI05_V1.num_steps,
        schedule_id=PI05_V1.schedule_id,
    )


def warm_orchestrator(model: Pi05Model, start_t: float, payload: CachePayload):
    """``(orchestrator, storage)`` returning WARM_START(start_t) on the stored ``payload``."""
    orch, _, storage = make_orchestrator(vector_dims={"robot_state": D}, judge=AlwaysWarmStartJudge(start_t))
    insert_entry(storage, CheckpointID.CP1, model.state, payload)
    return orch, storage


# ------------------------------------------------------------------
# A whole served Pi0.5 stack: interceptor + executor + evidence wrapper
# ------------------------------------------------------------------

EPISODE = {"experiment": "libero_10", "task": "put both pots on the stove", "episode_id": 3}
EXTRA = {"task_uid": "y:eval:4:3", "attempt": 1, "task_id": 4, "orig_init_state_idx": 3}


class SyncCoordinator:
    """Synchronous stand-in for ``BatchingCoordinator``: B = 1 through the real Pi0.5 adapter."""

    def __init__(self, model: Pi05Model) -> None:
        from openpi.serving.batching_coordinator import Pi05StageBatcher

        self._model = model
        self._batcher = Pi05StageBatcher(model, torch.device("cpu"))
        self.calls: list = []

    def submit_to_stage(self, stage_id, bundle_id, payload, **kwargs):
        from openpi.serving.batching_coordinator import (
            Stage3MissPayload,
            Stage3WarmResetPayload,
            Stage3WarmStartPayload,
        )

        self.calls.append((stage_id, bundle_id, payload))
        if stage_id == 1:
            return self._model.run_stage1(payload)
        if stage_id == 2:
            return self._model.run_stage2(payload)
        if isinstance(payload, Stage3MissPayload):
            return self._batcher.run_stage3_miss(
                [payload], num_steps=payload.num_steps, save_timesteps_per_request=[payload.save_timesteps]
            )[0]
        if isinstance(payload, Stage3WarmStartPayload):
            return self._batcher.run_stage3_warm(
                [payload], start_t=payload.start_t, num_steps=payload.num_steps, capture_first_step=False
            )[0]
        if isinstance(payload, Stage3WarmResetPayload):
            return self._batcher.run_stage3_warm_reset([payload])[0]
        raise TypeError(type(payload).__name__)


def pi05_stack(tmp_path, arm: str, t: float = 0.2, *, coordinator: bool = False,
               yaml_id: str = "y", bundle_id: str = "b"):
    """Build the served stack ``_wrap_policy`` builds for a warm reset yaml (eager, CPU)."""
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.timing import SystemTimer
    from openpi.cache.warm_reset.pi05 import build_pi05_warm_reset
    from tests.cache.test_interceptor import FakePolicy

    model = Pi05Model()
    payload = pi05_payload(t)
    orch, _ = warm_orchestrator(model, t, payload)
    spec = spec_for("pi05", arm, evidence_dir=str(tmp_path / "evidence"))
    yaml_path = pathlib.Path(tmp_path) / f"{arm}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"# {arm}\n")
    parts = build_pi05_warm_reset(
        pi05_config(block_of(spec), start_t=t), bundle_id=bundle_id, yaml_id=yaml_id, yaml_path=str(yaml_path)
    )
    coord = SyncCoordinator(model) if coordinator else None
    interceptor = InferenceInterceptor(
        FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch, eager=True,
        coordinator=coord, bundle_id=bundle_id, warm_reset=parts.executor,
    )
    return SimpleNamespace(model=model, payload=payload, parts=parts, spec=spec, interceptor=interceptor,
                           served=parts.wrap(interceptor), coordinator=coord, yaml_path=yaml_path, t=t,
                           yaml_id=yaml_id, bundle_id=bundle_id)


def obs_for(stack) -> dict:
    """One wire-shaped observation (the interceptor adapts it to either path)."""
    import numpy as np

    from tests.cache.test_interceptor import _make_obs

    del stack
    np.random.seed(0)
    return _make_obs()


def run_episode(stack, n: int, *, success: bool = True, extra: Optional[dict] = None) -> list:
    """Drive one episode of ``n`` decisions through the served stack; returns the responses."""
    served = stack.served
    served.on_task_begin()
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA if extra is None else extra))
    outs = [served.infer(obs_for(stack)) for _ in range(n)]
    served.on_episode_end(success)
    return outs


def read_rows(path) -> list[dict]:
    import json

    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def expected_for(stack, *, n: int, outcome: bool = True, **overrides):
    """The trusted expectation of one ``run_episode`` (caller-side sources only)."""
    import hashlib

    from openpi.cache.warm_reset.evidence import ExpectedEpisode

    identity = {**EPISODE, **EXTRA}
    fields = dict(
        task_uid=EXTRA["task_uid"], attempt=EXTRA["attempt"], outcome=outcome, n_decisions=n,
        yaml_id=stack.yaml_id, bundle_id=stack.bundle_id, spec=stack.spec, spec_digest=stack.spec.digest(),
        yaml_sha256=hashlib.sha256(stack.yaml_path.read_text().encode("utf-8")).hexdigest(),
        schedule_id=PI05_V1.schedule_id, k=PI05_V1.num_steps, start_t=stack.t, identity=identity,
    )
    fields.update(overrides)
    return ExpectedEpisode(**fields)
