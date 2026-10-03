"""Deterministic CPU GR00T-shaped fake for offline batch parity/profiling.

Runs the production runner's stage-1 and stage-2 implementations. Toy vision,
language and action operations give dispatch evidence, not GPU speed estimates.
"""
from types import SimpleNamespace

import numpy as np
import torch

from openpi.cache.groot.staged import GrootStagedRunner
from .models import GrootModel, measure


class _Language:
    model = SimpleNamespace(layers=[None])

    def __init__(self):
        self.batches = []

    def get_input_embeddings(self):
        return lambda ids: ids[:, :, None].float().expand(-1, -1, 32).clone()

    def __call__(self, inputs_embeds, **kwargs):
        self.batches.append(len(inputs_embeds))
        return SimpleNamespace(hidden_states=[inputs_embeds, inputs_embeds * 2.])


class _Eagle:
    image_token_index = 99

    def __init__(self):
        self.language_model = _Language()
        self.batches = []

    def extract_feature(self, pixels):
        self.batches.append(len(pixels) // 2)
        return pixels.reshape(len(pixels), -1).mean(dim=1)[:, None, None].expand(-1, 256, 32).clone()


class _Raw:
    training = False
    device = "cpu"

    def __init__(self):
        self.backbone = SimpleNamespace(eagle_model=_Eagle(), select_layer=1, eagle_linear=lambda x: x)
        self.action_head = SimpleNamespace(num_inference_timesteps=8, training=False,
                                          process_backbone_output=lambda x: x)

    def prepare_input(self, obs):
        n = 530 + len(obs.get("prompt", ""))
        ids = torch.arange(n)[None].remainder(50)
        ids[:, :256] = 99
        ids[:, 270:526] = 99
        images = np.stack([obs["observation/image"], obs["observation/wrist_image"]])
        state = torch.tensor(np.pad(obs["observation/state"], (0, 24)), dtype=torch.float32)[None, None]
        eagle = dict(eagle_input_ids=ids, eagle_attention_mask=torch.ones((1, n), dtype=torch.bool),
                     eagle_pixel_values=torch.tensor(images, dtype=torch.float32))
        action = dict(state=state, state_mask=(torch.arange(32)[None, None] < 8),
                      embodiment_id=torch.tensor([obs.get("embodiment_id", 1)]))
        return eagle, action


class _Runner(GrootStagedRunner):
    def __init__(self, raw):
        super().__init__(raw, verify_upstream=False)
        self.action_batches = []

    def run_stage3(self, stage2, noise):
        self.action_batches.append(len(noise))
        # Each row is independent, including reduction order across batches.
        mean = stage2.backbone_features.mean(dim=(1, 2))
        return SimpleNamespace(action_pred=noise + mean[:, None, None] / 1000.)


class CpuGrootModel(GrootModel):
    fake = True
    def __init__(self, stage_mode="batched"):
        raw = _Raw()
        self.policy = SimpleNamespace(model=raw)
        self.runner = _Runner(raw)
        self.device, self.H, self.action_dim, self.stage_mode = "cpu", 16, 32, stage_mode
        self.prepared_rows = 0

    def prepare(self, observations):
        with measure(self, "transform"):
            self.prepared_rows += len(observations)
            return observations
