"""A LIBERO-shaped stub GR00T model for the CP2 island tooling tests.

Two image runs of 256 tokens (image + wrist_image) and a prompt whose length
follows the instruction string, on top of ``tests/cache/groot``'s stub. The
policy's ``apply_transforms`` is the seam the island scripts go through
(``cp2_reconstruct.normalized_input_for``), and ``load_groot_libero_policy`` is
monkeypatched to hand it out where a script would load the real checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exp.libero_groot import libero_keys as K
from openpi.cache.groot.staged import GrootStagedRunner
from tests.cache.groot.conftest import (
    IMAGE_TOKEN_ID,
    STATE_VALID,
    STATE_WIDTH,
    TOKENS_PER_IMAGE,
    StubGrootModel,
)

N_CAMERAS = 2  # LIBERO: image + wrist_image


class LiberoStubModel(StubGrootModel):
    """Two image runs, and a prompt length that follows the instruction string."""

    def build_inputs(self, prompt_tokens=None):
        n_prompt = self.prompt_tokens if prompt_tokens is None else prompt_tokens
        ids = [1] * n_prompt
        for _ in range(N_CAMERAS):
            ids.extend([IMAGE_TOKEN_ID] * TOKENS_PER_IMAGE)
            ids.extend([2] * n_prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        state = torch.zeros(1, 1, STATE_WIDTH)
        state_mask = torch.zeros(1, 1, STATE_WIDTH, dtype=torch.bool)
        state_mask[0, 0, :STATE_VALID] = True
        return {
            "eagle_input_ids": input_ids,
            "eagle_attention_mask": torch.ones_like(input_ids),
            "eagle_pixel_values": torch.zeros(N_CAMERAS, 3, 4, 4),
            "eagle_image_sizes": torch.zeros(N_CAMERAS, 2, dtype=torch.long),
            "state": state,
            "state_mask": state_mask,
            "embodiment_id": torch.tensor([0]),
        }


class StubLiberoPolicy:
    """``apply_transforms`` keyed on the instruction: a different task string gives a
    different prompt length, as the real tokenizer would."""

    def __init__(self, model: LiberoStubModel) -> None:
        self.model = model
        self.denoising_steps = model.action_head.num_inference_timesteps
        self.calls: list[dict] = []

    def apply_transforms(self, obs):
        self.calls.append(obs)
        task = str(np.asarray(obs[K.LANGUAGE_KEY]).reshape(-1)[0])
        return self.model.build_inputs(prompt_tokens=3 + len(task.split()))


@pytest.fixture
def groot_harness():
    model = LiberoStubModel()
    runner = GrootStagedRunner(model, verify_upstream=False)
    return model, StubLiberoPolicy(model), runner


@pytest.fixture
def stub_island_policy(monkeypatch):
    """Route every island script's ``load_groot_libero_policy`` to a k=8 stub policy
    and switch the runner's upstream drift guards off for the stub."""
    model = LiberoStubModel()
    model.action_head.num_inference_timesteps = 8
    policy = StubLiberoPolicy(model)
    for guard in ("_verify_upstream_forward", "_verify_upstream_action_head"):
        monkeypatch.setattr(f"openpi.cache.groot.staged.GrootStagedRunner.{guard}", lambda self: None)
    for module in ("exp.libero_groot.build_cp2_artifact_groot", "exp.libero_groot.build_shadow_table_groot",
                   "exp.libero_groot.bench_cp2_overhead_groot", "exp.libero_groot.groot_cp2_parity"):
        monkeypatch.setattr(f"{module}.load_groot_libero_policy",
                            lambda checkpoint, denoising_steps=8, device="cpu": policy)
    return policy
