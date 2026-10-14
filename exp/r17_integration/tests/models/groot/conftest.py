"""Tiny one-camera GR00T-shaped vendor stub; true source guards remain exercised."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.models.groot.manifest import CHANNELS, GrootManifest, PROTOCOL, digest


class Language:
    def __init__(self):
        self.model = NS(layers=[0, 1])
        self.embed = torch.nn.Embedding(100, 8)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(800).reshape(100, 8) / 800)
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def __call__(self, *, inputs_embeds, **kwargs):
        self.calls += 1
        return NS(hidden_states=(inputs_embeds, inputs_embeds + 1, inputs_embeds * 2))


class Eagle:
    image_token_index = 99

    def __init__(self):
        self.language_model = Language()
        self.extract_calls = 0

    def extract_feature(self, pixels):
        self.extract_calls += 1
        value = pixels.flatten(1).mean(1)[:, None, None]
        return torch.arange(256 * 8).reshape(1, 256, 8).float() / 2048 + value

    def forward(self, inputs):
        ids = inputs["eagle_input_ids"]
        embeds = self.language_model.get_input_embeddings()(ids)
        embeds[ids == self.image_token_index] = self.extract_feature(inputs["eagle_pixel_values"]).reshape(-1, 8)
        return self.language_model(inputs_embeds=embeds).hidden_states[2]


class Head(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.num_inference_timesteps = k
        self.num_timestep_buckets = 1000
        self.action_horizon = 16
        self.config = NS(action_horizon=16, action_dim=32, add_pos_embed=False)
        self.future_tokens = torch.nn.Embedding(1, 32)
        torch.nn.init.zeros_(self.future_tokens.weight)
        self.nfe = 0
        self.eval()

    def process_backbone_output(self, output):
        return BatchFeature({"backbone_features": output["backbone_features"].float()})

    def state_encoder(self, state, embodiment):
        return state[..., :32].float()

    def action_encoder(self, x, t, embodiment):
        return x + t[:, None, None].float() / 10000

    def model(self, *, hidden_states, encoder_hidden_states, timestep):
        self.nfe += 1
        return hidden_states + encoder_hidden_states.float().mean((1, 2))[:, None, None]

    def action_decoder(self, x, embodiment):
        return x * .05

    def get_action(self, output, action):
        # Independent unsplit loop, not a call to the staged transcription.
        vl = self.process_backbone_output(output).backbone_features
        emb = action["embodiment_id"]
        state = self.state_encoder(action["state"], emb)
        x = torch.randn((vl.shape[0], 16, 32), dtype=vl.dtype, device=vl.device)
        for index in range(self.num_inference_timesteps):
            t = torch.full((vl.shape[0],), int(index / self.num_inference_timesteps * 1000), device=vl.device)
            encoded = self.action_encoder(x, t, emb)
            future = self.future_tokens.weight[None].expand(vl.shape[0], -1, -1)
            predicted = self.action_decoder(self.model(hidden_states=torch.cat((state, future, encoded), 1),
                                                      encoder_hidden_states=vl, timestep=t), emb)
            x = x + (1.0 / self.num_inference_timesteps) * predicted[:, -16:]
        return {"action_pred": x}


class Model:
    training = False
    device = "cpu"

    def __init__(self, k=4):
        self.backbone = NS(eagle_model=Eagle(), select_layer=2, eagle_linear=torch.nn.Identity())
        self.action_head = Head(k)
        self.validated = 0

    def prepare_input(self, data):
        return BatchFeature({k: v for k, v in data.items() if k.startswith("eagle_")}), BatchFeature({k: v for k, v in data.items() if not k.startswith("eagle_")})

    def validate_data(self, *args, **kwargs):
        self.validated += 1


class Backend:
    def __init__(self, k=4):
        self.model = Model(k)
        self.apply_count = self.inverse_count = 0

    def normalize_state_only(self, state):
        return np.asarray(state, np.float32) * 2 - 1

    def apply_transforms(self, obs):
        self.apply_count += 1
        assert obs["video.image"].shape == (1, 1, 480, 480, 3)
        # Explicit nonidentity native minmax normalization, with no GR LIBERO sign inversion.
        state = torch.zeros(1, 1, 64)
        for i, name in enumerate(CHANNELS):
            state[..., i] = float(obs[f"state.{name}"].reshape(-1)[0]) * 2 - 1
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[..., :4] = True
        prompt = str(obs["annotation.human.action.task_description"].reshape(-1)[0])
        ids = torch.tensor([[1] * (len(prompt) % 5 + 1) + [99] * 256 + [2]])
        return {"eagle_input_ids": ids, "eagle_attention_mask": torch.ones_like(ids),
                "eagle_pixel_values": torch.tensor(np.array(obs["video.image"], dtype=np.float32)).reshape(1, -1).mean(1).reshape(1, 1, 1, 1),
                "state": state, "state_mask": mask, "embodiment_id": torch.tensor([31])}

    def unapply_transforms(self, data):
        self.inverse_count += 1
        x = data["action"].detach().cpu().numpy()
        assert x.shape == (1, 16, 32)
        return {f"action.{name}": x[..., i:i+1] * 3 + .25 for i, name in enumerate(CHANNELS)}

    def get_action(self, obs):
        native = {k: v[None] for k, v in obs.items()}
        normalized = self.apply_transforms(native)
        from openpi.cache.groot.staged import GrootStagedRunner
        # Upstream-shaped unsplit backbone→head, both under upstream autocast context.
        with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16):
            features = self.model.backbone.eagle_model.forward(normalized)
            action = self.model.action_head.get_action(BatchFeature({"backbone_features": features}), normalized)
        return self.unapply_transforms({"action": action["action_pred"].detach().cpu().float().clone()})


@pytest.fixture
def guarded_stub(monkeypatch):
    import openpi.cache.groot.staged as staged
    # Pin test vendor source instead of turning verification off. Separate tests exercise original pins.
    monkeypatch.setattr(staged, "UPSTREAM_FORWARD_SHA256", hashlib.sha256(inspect.getsource(Eagle.forward).encode()).hexdigest())
    monkeypatch.setattr(staged, "UPSTREAM_ACTION_HEAD_SHA256", hashlib.sha256(Path(inspect.getsourcefile(Head)).read_bytes()).hexdigest())
    return Backend


@pytest.fixture
def manifest():
    return GrootManifest(*(digest(s) for s in ("weights", "norm", "code", "transforms", "tokenizer")), hidden_width=8)


@pytest.fixture
def obs():
    from exp.r17_integration.benchmarks.metaworld import decision_seed
    from exp.metaworld.tasks import task_id_of
    return {"image": np.full((480, 480, 3), 2, np.uint8), "state": np.array([.1, .2, .3, .4], np.float32), "prompt": "reach",
            "__request__": {"campaign": "cpu-P5", "task_name": "reach-v3", "task_id": task_id_of("reach-v3"),
                            "original_init_idx": 0, "seed": 7, "explicit_seed": decision_seed("reach-v3", 0, 0)}}


def begin_episode(policy, **kwargs):
    """Explicit fake-client declaration, identical to the real P2 ack contract."""
    from exp.metaworld.tasks import task_id_of
    extra = {"campaign": "cpu-P5", "task_name": "reach-v3", "task_id": task_id_of("reach-v3"),
             "original_init_idx": 0, "seed": 7, "ack_protocol": "r17.execution.v1"}
    extra.update(kwargs.pop("extra_metadata", {}))
    policy.on_episode_start(extra_metadata=extra, **kwargs)


@pytest.fixture
def context(manifest):
    return RequestContext("cpu-P5", manifest.benchmark_protocol, "reach-v3", 0, 0)


def training_metadata():
    modalities = {group: {name: {"shape": [1], "absolute": group == "state", "continuous": True}
                          for name in CHANNELS} for group in ("state", "action")}
    modalities["video"] = {"image": {"resolution": [480, 480], "channels": 3, "fps": 20}}
    return {"metaworld_sawyer": {"embodiment_tag": "metaworld_sawyer", "modalities": modalities,
             "statistics": {group: {name: {s: [v] for s, v in (("min", -1.), ("max", 1.), ("mean", 0.), ("std", .5), ("q01", -.9), ("q99", .9))}
                                for name in CHANNELS} for group in ("state", "action")}}}
