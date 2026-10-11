"""In-place video LoRA preserving the released FastWAM action/projector layout.

Targets/rank/alpha follow EasyWAM configs/model/{backbone/wan22,lora/video_dit}.yaml.
"""
import math

import torch
from torch import nn

TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")


class LoRALinear(nn.Module):
    def __init__(self, base, rank=128, alpha=128):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        # Wan's projection helper inspects Linear.weight.dtype before calling us.
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B) * self.scale


def inject_video_lora(policy, rank=128, alpha=128):
    targets = [(name, module) for name, module in policy.named_modules()
        if isinstance(module, nn.Linear) and ".blocks.video." in name and any(name.endswith(t) for t in TARGETS)]
    if len(targets) != 30 * len(TARGETS):
        raise ValueError(f"Expected exactly 300 video linear targets, found {len(targets)}")
    for name, module in targets:
        parent_name, child = name.rsplit(".", 1)
        parent = policy.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(module, rank, alpha))
    return {"rank": rank, "alpha": alpha, "dropout": 0, "bias": "none", "targets": list(TARGETS), "modules": len(targets)}
