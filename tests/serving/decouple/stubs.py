"""Deterministic stage stubs with the production inter-stage types."""
from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_pytorch import Stage1Output, Stage2Output, Stage3Output


class PiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
        self.config = SimpleNamespace(action_horizon=10, action_dim=32, pytorch_compile_mode=None)

    def run_stage1(self, obs):
        state = obs.state if hasattr(obs, "state") else obs["state"]
        # A deterministic stage stub for arbitrary recorded state/images.
        n = state.shape[0]
        if isinstance(obs, dict) and "tokens" in obs:
            prefix = obs["tokens"]
        else:
            prefix = state[:, None, :].expand(n, 800, 32).contiguous()
        length = prefix.shape[1]
        return Stage1Output(state, prefix, torch.ones(n, length, dtype=torch.bool),
                            torch.zeros(n, 1, length, length), torch.arange(length).expand(n, length))

    def run_stage2(self, stage1):
        n = stage1.state.shape[0]
        kv = stage1.state.reshape(n, 1, 1, -1).clone()
        return Stage2Output(stage1, ((kv, kv.clone()),))

    run_stage2_capture = run_stage2

    def run_stage3(self, stage2, *, noise, num_steps=10, return_intermediates=False,
                   save_timesteps=(0.7, 0.5, 0.3)):
        action = noise.clone()
        for _ in range(num_steps):
            action = action + stage2.stage1.state[:, None, :] * 0.001
        inter = {t: action * t for t in save_timesteps} if return_intermediates else None
        return Stage3Output(action, inter)

    def run_stage3_from(self, stage2, *, start_x, start_t, num_steps):
        return self.run_stage3(stage2, noise=start_x, num_steps=num_steps)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)
