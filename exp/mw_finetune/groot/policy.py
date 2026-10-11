"""A's pure-inference serving contract; GR00T owns normalization and resizing."""
from __future__ import annotations

import json
from pathlib import Path

from .conventions import EMBODIMENT, EMBODIMENT_ID, HORIZON, from_groot_actions, to_groot_observation


class MetaWorldPolicy:
    def __init__(self, backend, checkpoint, denoising_steps):
        self.backend = backend
        self.checkpoint = str(checkpoint)
        self.denoising_steps = denoising_steps

    def reset(self):
        # No action queue, model history, retrieval, or cross-request caches.
        pass

    def infer(self, obs):
        return {"actions": from_groot_actions(self.backend.get_action(to_groot_observation(obs)))}

    def get_metadata(self):
        return {"model": "GR00T-N1.5-MT50", "checkpoint": self.checkpoint,
                "embodiment": EMBODIMENT, "embodiment_id": EMBODIMENT_ID,
                "action_horizon": HORIZON, "action_dim": 4, "state_dim": 4,
                "denoising_steps": self.denoising_steps, "pure_inference": True,
                "action_channels": ["x", "y", "z", "gripper"],
                "policy_rng": "caller-seeded global Torch RNG"}


def load_policy(checkpoint, device="cuda:0", **kw):
    from .embodiment import register_embodiment
    register_embodiment()
    from .data_config import MetaWorldDataConfig
    from gr00t.model.policy import Gr00tPolicy

    steps = kw.pop("denoising_steps", None)
    steps = 4 if steps is None else steps
    if kw:
        raise TypeError(f"unknown policy options: {sorted(kw)}")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("denoising_steps must be positive")
    metadata_path = Path(checkpoint) / "experiment_cfg" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing fine-tuned MetaWorld metadata: {metadata_path}")
    if EMBODIMENT not in json.loads(metadata_path.read_text()):
        raise ValueError(f"checkpoint has no {EMBODIMENT} training normalization")
    cfg = MetaWorldDataConfig()
    backend = Gr00tPolicy(model_path=str(checkpoint), embodiment_tag=EMBODIMENT,
                          modality_config=cfg.modality_config(), modality_transform=cfg.transform(),
                          denoising_steps=steps, device=device)
    return MetaWorldPolicy(backend, checkpoint, steps)
