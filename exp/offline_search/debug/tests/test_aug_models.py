import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from exp.offline_search.debug.aug.models import Pi05Model, GrootModel


def test_pi05_adapter_actual_stage_io_and_private_noise():
    from openpi.models_pytorch.pi0_pytorch import Stage1Output
    torch.set_num_threads(1)
    class PolicyModel:
        def __init__(self):
            self.backbone_calls = 0
        def sample_noise(self, shape, device, generator):
            return torch.randn(shape, device=device, generator=generator)
        def run_stage2(self, stage1):
            self.backbone_calls += 1
            return stage1
        def run_stage3(self, stage2, noise, num_steps):
            assert num_steps == 10
            return SimpleNamespace(action_chunk=noise + stage2.state.mean(dim=1)[:, None, None])
    class Batcher:
        def run_stage1_batch(self, inputs):
            return [Stage1Output(state=torch.tensor(p["state"])[None], prefix_embs=torch.ones((1, 768, 32)),
                                 prefix_pad_masks=torch.ones((1, 768), dtype=torch.bool),
                                 prefix_att_2d_masks_4d=torch.zeros((1, 1, 768, 768)),
                                 prefix_position_ids=torch.arange(768)[None]) for p in inputs]
    adapter = Pi05Model.__new__(Pi05Model)
    adapter.policy = SimpleNamespace(_input_transform=lambda obs: dict(state=np.pad(obs["observation/state"], (0, 24)).astype(np.float32)))
    adapter._model = PolicyModel()
    adapter.batcher = Batcher()
    adapter.device, adapter.steps, adapter.H, adapter.action_dim = torch.device("cpu"), 10, 10, 32
    observations = [{"observation/state": np.arange(8, dtype=np.float64)}, {"observation/state": np.ones(8)}]
    before = torch.get_rng_state().clone()
    encoded = adapter.encode(observations)
    assert encoded["v0"].shape == (2, 512)
    output = adapter.draw(encoded, [1, 2])
    adapter.draw(encoded, [3, 4])
    assert adapter._model.backbone_calls == 1
    reverse = adapter.draw(adapter.encode(observations[::-1]), [2, 1])
    assert np.array_equal(output, reverse[::-1])
    assert torch.equal(before, torch.get_rng_state())


def test_groot_adapter_uses_matching_shapes_and_private_noise():
    from openpi.cache.groot.staged import GrootStage1Output, GrootStage2Output
    from openpi.cache.types import groot_n15_schedule
    class Runner:
        _device_type = "cpu"
        def __init__(self):
            self.batch_shapes = []
            self.backbone_calls = 0
        def session(self):
            return contextlib.nullcontext()
        def live_schedule(self):
            return groot_n15_schedule(8)
        def run_stage1(self, norm):
            n = 530 + len(str(norm["annotation.human.action.task_description"][0, 0]))
            mask = torch.zeros((1, n), dtype=torch.bool)
            mask[:, :256] = True
            mask[:, 270:526] = True
            state = torch.ones((1, 1, 32))
            state_mask = torch.arange(32)[None, None] < 8
            inputs = dict(state=state, state_mask=state_mask, embodiment_id=torch.tensor([1]))
            return GrootStage1Output(input_embeds=torch.ones((1, n, 32)), attention_mask=torch.ones((1, n), dtype=torch.bool),
                                     image_token_mask=mask, action_inputs=inputs)
        def run_stage2_llm(self, stage1):
            self.backbone_calls += 1
            return GrootStage2Output(backbone_features=stage1.input_embeds, attention_mask=stage1.attention_mask,
                                     action_inputs=stage1.action_inputs)
        def _head_inputs(self, stage2):
            return dict(backbone_features=stage2.backbone_features)
        def run_stage3(self, stage2, noise):
            self.batch_shapes.append(tuple(stage2.backbone_features.shape))
            return SimpleNamespace(action_pred=noise + stage2.backbone_features.shape[1] / 1000.)
    adapter = GrootModel.__new__(GrootModel)
    adapter.policy = SimpleNamespace(apply_transforms=lambda obs: obs,
                                    model=SimpleNamespace(action_head=SimpleNamespace(process_backbone_output=lambda value: value)))
    adapter.runner = Runner()
    adapter.stage_mode = "serial"
    adapter.device, adapter.H, adapter.action_dim = "cpu", 16, 32
    base = {"observation/state": np.ones(8), "observation/image": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.ones((256, 256, 3), np.uint8)}
    observations = [dict(base, prompt="one"), dict(base, prompt="longer"), dict(base, prompt="one")]
    before = torch.get_rng_state().clone()
    encoded = adapter.encode(observations)
    assert encoded["v0"].shape == (3, 512)
    output = adapter.draw(encoded, [1, 2, 3])
    assert sorted(shape[0] for shape in adapter.runner.batch_shapes) == [1, 2]
    adapter.draw(encoded, [4, 5, 6])
    assert adapter.runner.backbone_calls == 3
    reverse = adapter.draw(adapter.encode(observations[::-1]), [3, 2, 1])
    assert np.array_equal(output, reverse[::-1])
    assert torch.equal(before, torch.get_rng_state())


def test_pi05_subset_reuses_actual_stage2_and_independent_kv_shards():
    from openpi.models_pytorch.pi0_pytorch import Stage1Output, Stage2Output
    from openpi.serving import stage_io
    class Model:
        calls = 0
        def run_stage2(self, stage1):
            self.calls += 1
            b = len(stage1.state)
            cache = ((torch.arange(b * 12).reshape(b, 1, 3, 4).float(), torch.ones((b, 1, 3, 4))),)
            return Stage2Output(stage1=stage1, past_key_values=cache, prefix_out=stage1.prefix_embs)
        def sample_noise(self, shape, device, generator):
            return torch.randn(shape, device=device, generator=generator)
        def run_stage3(self, stage2, noise, num_steps):
            return SimpleNamespace(action_chunk=noise + stage2.stage1.state.mean(dim=1)[:, None, None])
    adapter = Pi05Model.__new__(Pi05Model)
    adapter.device, adapter.steps, adapter.H, adapter.action_dim = torch.device("cpu"), 10, 10, 32
    adapter._model = Model()
    stage1 = Stage1Output(state=torch.arange(96).reshape(3, 32).float(), prefix_embs=torch.ones((3, 768, 32)),
        prefix_pad_masks=torch.ones((3, 768), dtype=torch.bool), prefix_att_2d_masks_4d=torch.zeros((3, 1, 768, 768)),
        prefix_position_ids=torch.arange(768)[None].expand(3, -1))
    encoded = dict(stage1=stage_io.split_stage1_output(stage1, 3), stage2=None)
    before = torch.get_rng_state().clone()
    full = adapter.draw(encoded, [21, 22, 23])
    selected = adapter.select(encoded, [2, 0])
    assert np.array_equal(adapter.draw(selected, [23, 21]), full[[2, 0]])
    assert adapter._model.calls == 1
    original = encoded["stage2"].past_key_values[0][0].clone()
    encoded["stage2_shards"][0].past_key_values[0][0].fill_(0)
    assert torch.equal(original, encoded["stage2"].past_key_values[0][0])
    assert torch.equal(before, torch.get_rng_state())
