"""Prove the frozen shortcut preserves action predictions and proprio gradients."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from exp.mw_finetune.fastwam.lora import LoRALinear


def test_lora_starts_identical_and_updates_low_rank_branch():
    base = nn.Linear(5, 7)
    layer = LoRALinear(base, rank=2, alpha=2)
    assert layer.weight is base.weight and layer.bias is base.bias
    x = torch.randn(3, 5)
    torch.testing.assert_close(layer(x), base(x))
    layer(x).square().sum().backward()
    assert layer.lora_B.grad.norm() > 0
    assert base.weight.grad is None and base.bias.grad is None


def test_current_frame_loss_matches_full_causal_action_loss_and_gradient():
    pytest.importorskip("lerobot")
    from lerobot.policies.fastwam.wan import ActionDiT, FastWAM, MoT, WanVideoDiT
    from exp.mw_finetune.fastwam.loss import frozen_video_loss

    torch.set_num_threads(2)
    torch.manual_seed(1)
    video = WanVideoDiT(hidden_dim=32, in_dim=4, ffn_dim=64, out_dim=4,
        text_dim=16, freq_dim=8, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        attn_head_dim=16, num_layers=2, seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal")
    action = ActionDiT(hidden_dim=16, action_dim=4, ffn_dim=32, text_dim=16,
        freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=16, num_layers=2)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)

    class VAE(nn.Module):
        temporal_downsample_factor = 4

        def encode(self, pixels, **kw):
            encoded = F.avg_pool3d(pixels[:, :, ::4], kernel_size=(1, 4, 4))
            return torch.cat([encoded, encoded[:, :1]], dim=1)

    core = FastWAM(video, action, mot, VAE(), text_dim=16, proprio_dim=4,
        device="cpu", torch_dtype=torch.float32, loss_lambda_video=0)
    for name, parameter in core.named_parameters():
        if "video" in name:
            parameter.requires_grad_(False)
    sample = {"video": torch.rand(2, 3, 9, 16, 16), "action": torch.rand(2, 32, 4),
        "proprio": torch.rand(2, 1, 4), "context": torch.rand(2, 5, 16),
        "context_mask": torch.ones(2, 5, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 32, dtype=torch.bool),
        "image_is_pad": torch.zeros(2, 9, dtype=torch.bool)}
    core.train_action_scheduler.sample_training_t = lambda batch_size, device, dtype: torch.full((batch_size,), 0.4, device=device, dtype=dtype)
    torch.manual_seed(17)
    noise = torch.randn_like(sample["action"])
    t = torch.full((2,), 0.4)
    inputs = core.build_inputs(sample)
    latent = inputs["input_latents"] + torch.randn_like(inputs["input_latents"])
    latent[:, :, :1] = inputs["first_frame_latents"]
    targets = {"latents": latent, "timestep_video": torch.full((2,), 0.7),
        "noisy_action": core.train_action_scheduler.add_noise(sample["action"], noise, t), "timestep_action": t}
    _, prediction = core._run_training_mot(inputs, targets)
    target = core.train_action_scheduler.training_target(sample["action"], noise, t)
    reference = (F.mse_loss(prediction, target, reduction="none").mean((1, 2)) * core.train_action_scheduler.training_weight(t)).mean()
    reference.backward()
    reference_grad = core.proprio_encoder.weight.grad.clone()
    core.zero_grad(set_to_none=True)
    torch.manual_seed(17)
    shortcut, _ = frozen_video_loss(SimpleNamespace(model=core), sample)
    shortcut.backward()
    torch.testing.assert_close(shortcut, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(core.proprio_encoder.weight.grad, reference_grad, rtol=2e-4, atol=2e-6)
    assert reference_grad.norm() > 0
