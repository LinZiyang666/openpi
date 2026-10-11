"""A's stateless serving contract. No queue, cross-call cache, or gripper toggle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import denormalize, normalize, resize_rgb, validate_observation


class Policy:
    def __init__(self, policy, metadata, denoising_steps=10):
        self.policy = policy
        self.metadata = metadata
        self.denoising_steps = int(denoising_steps)
        if self.denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")

    def reset(self):
        self.policy.reset()

    def get_metadata(self):
        return {**self.metadata, "model": "fastwam", "action_horizon": 32,
            "denoising_steps": self.denoising_steps, "stateless": True,
            "rng": "obs.seed; CPU torch.Generator", "image_size": [224, 224],
            "action_channels": ["x", "y", "z", "gripper"], "gripper_transform": "none"}

    def infer(self, obs):
        import torch
        image, state, prompt = validate_observation(obs)
        pixels = torch.from_numpy(resize_rgb(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        state = torch.from_numpy(normalize(state, self.metadata["stats"]["observation.state"])).unsqueeze(0)
        model = self.policy.model
        prompt = self.policy.config.prompt_template.format(task=prompt)
        # All KV caching occurs only inside this infer_action call.
        with torch.inference_mode():
            result = model.infer_action(prompt=prompt, input_image=pixels,
                proprio=state, action_horizon=32, num_inference_steps=self.denoising_steps,
                text_cfg_scale=1.0, seed=int(obs.get("seed", 42)), rand_device="cpu",
                compile_action_infer=False)
        actions = result["action"].detach().float().cpu().numpy().reshape(32, 4)
        actions = denormalize(actions, self.metadata["stats"]["action"])
        if not np.isfinite(actions).all():
            raise ValueError("Fast-WAM emitted nonfinite actions")
        return {"actions": np.ascontiguousarray(actions, dtype=np.float32)}


def load_policy(checkpoint, device="cuda:0", **kw):
    from .model import build_model, load_delta
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "mt50.json").read_text())
    if metadata.get("format") != "fastwam-mt50-delta-v1":
        raise ValueError("Expected an MT50 delta checkpoint with its training normalization")
    base = kw.pop("base", metadata["base"])
    wan = kw.pop("wan_root", metadata["wan_root"])
    steps = kw.pop("denoising_steps", 10)
    if kw:
        raise TypeError(f"Unknown policy options: {sorted(kw)}")
    policy = build_model(base=base, device=device, wan_root=wan, checkpointing=False, mode=metadata["mode"])
    load_delta(policy, checkpoint)
    policy.eval()
    metadata = {**metadata, "checkpoint": str(checkpoint.resolve())}
    return Policy(policy, metadata, steps)
