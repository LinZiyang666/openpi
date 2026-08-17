"""Stub GR00T model shaped like the real one, so the split is testable off-island.

The runner is duck-typed on purpose: `gr00t` lives in its own virtualenv on one
machine, and a test that could only run there would in practice never run. The
stub reproduces the attribute paths the split actually touches — the eagle
model's embedding table, feature extractor and truncated language model, the
backbone's `select_layer` / `eagle_linear`, and the action head — with tiny
tensors.
"""

from __future__ import annotations

import types

import pytest
import torch

EMB_DIM = 8
IMAGE_TOKEN_ID = 99
TOKENS_PER_IMAGE = 256
N_CAMERAS = 3
ACTION_HORIZON = 4
ACTION_DIM = 6
STATE_WIDTH = 10
STATE_VALID = 5


class _StubLanguageModel:
    def __init__(self, n_layers: int) -> None:
        self.model = types.SimpleNamespace(layers=list(range(n_layers)))
        self._embed = torch.nn.Embedding(200, EMB_DIM)
        torch.nn.init.normal_(self._embed.weight, std=0.5)
        self.calls = 0

    def get_input_embeddings(self):
        return self._embed

    def __call__(self, **kwargs):
        self.calls += 1
        embeds = kwargs["inputs_embeds"]
        # hidden_states[i] is the input to layer i; the final entry is the
        # normed output, which is what select_layer == len(layers) selects.
        n = len(self.model.layers)
        states = tuple(embeds + i for i in range(n)) + (embeds * 2.0,)
        return types.SimpleNamespace(hidden_states=states)


class _StubEagle:
    def __init__(self, n_layers: int) -> None:
        self.language_model = _StubLanguageModel(n_layers)
        self.image_token_index = IMAGE_TOKEN_ID
        self.extract_calls = 0

    def extract_feature(self, pixel_values):
        self.extract_calls += 1
        n_images = pixel_values.shape[0]
        base = torch.arange(n_images, dtype=torch.float32).view(n_images, 1, 1)
        return base + torch.ones(n_images, TOKENS_PER_IMAGE, EMB_DIM)

    def forward(self):  # only present so the hash guard has something to read
        return None


class _StubActionHead:
    def __init__(self) -> None:
        self.calls = 0

    def get_action(self, backbone_outputs, action_inputs):
        self.calls += 1
        features = backbone_outputs["backbone_features"]
        value = features.float().mean()
        pred = torch.full((1, ACTION_HORIZON, ACTION_DIM), value)
        return {"action_pred": pred}


class StubGrootModel:
    """Enough of GR00T_N1_5 for the two-stage split to run end to end."""

    def __init__(self, n_layers: int = 3, prompt_tokens: int = 5) -> None:
        self.backbone = types.SimpleNamespace(
            eagle_model=_StubEagle(n_layers),
            select_layer=n_layers,
            eagle_linear=torch.nn.Identity(),
        )
        self.action_head = _StubActionHead()
        self.training = False
        self.device = "cpu"
        self.prompt_tokens = prompt_tokens
        self.validate_calls = 0

    # -- inputs ---------------------------------------------------------

    def build_inputs(self, prompt_tokens: int | None = None) -> dict:
        """A normalised input whose image runs sit after a variable-length prompt."""
        n_prompt = self.prompt_tokens if prompt_tokens is None else prompt_tokens
        ids = []
        for _ in range(N_CAMERAS):
            ids.extend([1] * n_prompt)
            ids.extend([IMAGE_TOKEN_ID] * TOKENS_PER_IMAGE)
        ids.extend([2] * n_prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        state = torch.zeros(1, 1, STATE_WIDTH)
        state[0, 0, :STATE_VALID] = torch.linspace(0.1, 0.5, STATE_VALID)
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

    def prepare_input(self, inputs):
        backbone = {k: v for k, v in inputs.items() if k.startswith("eagle_")}
        action = {k: v for k, v in inputs.items() if not k.startswith("eagle_")}
        return backbone, action

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        self.validate_calls += 1


@pytest.fixture
def stub_model():
    return StubGrootModel()


@pytest.fixture
def runner(stub_model):
    from openpi.cache.groot.staged import GrootStagedRunner

    return GrootStagedRunner(stub_model, verify_upstream=False)
