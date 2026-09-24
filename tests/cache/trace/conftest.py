"""Shared fakes for the trace serving mode tests.

``TraceFakeModel`` is a CPU stand-in for ``PI0Pytorch`` with the staged API
the interceptor calls (``run_stage1/2/3``, ``run_stage3_from``,
``sample_noise``) and a prefix long enough for the CP1 token slice. Its
stage 3 is a deterministic function of the noise so tests can check that the
executed action really is the variant the verdict selected and that the
recorded ``noise_action_*`` are the loop's own inputs. ``make_step`` builds a
minimal ``StepTrace`` for writer tests without a model at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from openpi.cache.trace.types import EpisodeIdentity, StepTrace, TracePlan
from openpi.cache.types import PI05_V1
from openpi.collect.data_collector import InferenceEmbeddings

PREFIX_TOKENS = 768 + 20
EMB = 2048
H, D = 50, 32


class TraceFakeModel:
    """Deterministic staged-API fake: action = f(noise, start point)."""

    config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=H, action_dim=D)

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.prefix = torch.randn(1, PREFIX_TOKENS, EMB, generator=g)
        self.state = torch.randn(1, D, generator=g)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_stage1(self, observation):
        self.calls.append(("stage1", {}))
        return SimpleNamespace(state=self.state.clone(), prefix_embs=self.prefix.clone())

    def run_stage2(self, stage1):
        self.calls.append(("stage2", {}))
        return SimpleNamespace(stage1=stage1, past_key_values=None)

    def sample_noise(self, shape, device, generator=None):
        if generator is None:
            return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device, generator=generator)

    def run_stage3(self, stage2, *, noise=None, num_steps=10, return_intermediates=False, **kwargs):
        if "save_timesteps" in kwargs and kwargs["save_timesteps"] is None:
            raise TypeError("save_timesteps=None is not accepted by the real model")
        save_timesteps = kwargs.get("save_timesteps", (0.7, 0.5, 0.3))
        self.calls.append(("stage3", {"num_steps": num_steps, "save_timesteps": save_timesteps,
                                       "return_intermediates": return_intermediates}))
        if noise is None:
            noise = self.sample_noise((1, H, D), "cpu")
        x = noise.clone()
        dt = -1.0 / num_steps
        save_at = {round((1.0 - st) * num_steps): st for st in save_timesteps}
        inter: dict[float, torch.Tensor] = {}
        for step in range(num_steps):
            if step in save_at:
                inter[save_at[step]] = x.clone()
            x = x + dt * (x * 0.1 + 1.0)
        return SimpleNamespace(action_chunk=x, intermediates=inter if return_intermediates else None)

    def run_stage3_from(self, stage2, start_x, start_t, *, num_steps=10):
        self.calls.append(("stage3_from", {"start_t": start_t}))
        x = start_x.clone()
        dt = -1.0 / num_steps
        steps = round(start_t * num_steps)
        for _ in range(steps):
            x = x + dt * (x * 0.1 + 1.0)
        return SimpleNamespace(action_chunk=x)


class TraceFakePolicy:
    def __init__(self, model: TraceFakeModel | None = None) -> None:
        self._is_pytorch_model = True
        self._model = model if model is not None else TraceFakeModel()
        self._input_transform = _fake_input_transform
        self._output_transform = lambda x: x
        self._pytorch_device = torch.device("cpu")

    @property
    def metadata(self) -> dict[str, Any]:
        return {}


def _fake_input_transform(obs: dict) -> dict:
    # Mirrors the real chain's observable effects: ``prompt`` is popped, the
    # wire images become model slots + masks, state is kept.
    out = {
        "state": obs["observation/state"],
        "image": {"base_0_rgb": obs["observation/image"]},
        "image_mask": {"base_0_rgb": np.bool_(True)},
    }
    return out


def make_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.standard_normal(D).astype(np.float32),
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the bowl",
    }


def make_plan(**overrides) -> TracePlan:
    base = dict(
        model="pi05",
        schedule=PI05_V1,
        checkpoint=None,
        warm_tiers=(),
        record_noise_actions=False,
        save_timesteps=None,
        record_prefix_tokens=True,
        record_raw_images=True,
        record_model_images=False,
        record_query_keys=True,
        record_search=True,
        record_tokenized_prompt=True,
        raw_image_keys=("observation/image",),
        rng_isolation="verdict_aware",
        sidecar_jsonl=True,
        fail_loud=False,
    )
    base.update(overrides)
    return TracePlan(**base)


def make_identity(episode_id: int = 1, name: str = "", **extra) -> EpisodeIdentity:
    return EpisodeIdentity(
        experiment="exp", task="task", episode_id=episode_id, episode_name=name,
        extra_metadata=dict(extra),
    )


def make_step(seed: int = 0, *, noise: bool = False) -> StepTrace:
    rng = np.random.default_rng(seed)
    legacy = InferenceEmbeddings(
        vision_embs=[rng.standard_normal((256, 8)).astype(np.float16) for _ in range(3)],
        prompt_emb=rng.standard_normal((20, 8)).astype(np.float16),
        robot_state=rng.standard_normal(D).astype(np.float32),
        noise_action_steps=[rng.standard_normal((H, D)).astype(np.float32) for _ in range(9)] if noise else [],
        clean_action=rng.standard_normal((H, D)).astype(np.float32),
        input_images={"base_0_rgb": rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)},
        init_noise=rng.standard_normal((H, D)).astype(np.float32) if noise else None,
    )
    return StepTrace(
        legacy=legacy,
        raw_images={"observation/image": rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)},
        prompt="p",
        raw_state=rng.standard_normal(8).astype(np.float32),
        model_images=None,
        image_mask=None,
        tokenized_prompt=np.arange(5, dtype=np.int64),
        query_keys={"robot_state": rng.standard_normal(D).astype(np.float32)},
        search=None,
        cp3_twin_json=None,
        action_full_hit=None,
        action_warm={},
        action_warm_exec=None,
        action_executed=legacy.clean_action.copy(),
        executed_arm="full_inference",
        verdict={"hit_type": "MISS", "searched": True},
        tier_status={},
        error_proxies={},
        timing_ms={"wall_ms": 1.0},
    )


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "trace_out"
