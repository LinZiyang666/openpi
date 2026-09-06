"""Stage-1 reconstruction paths for the offline CP2 tooling (plan §3.8).

Two ways to get a ``Stage1Output`` for one stored H5 step:

- ``offline`` — ``exp/common/build_in_memory_cache_artifact._build_fake_stage1_with_masks``:
  the stored ``vision_*`` / ``prompt_emb`` tensors, no SigLIP forward. It marks
  every stored camera slot valid, including the zero-frame right-wrist slot that
  the serving path masks out (the collector captures its SigLIP embedding
  anyway), so the backbone sees 784 valid prefix tokens instead of 528 and its
  output — the CP2 key source — diverges from serving (parity 2026-09-04,
  libero_spatial: cosine 0.70). Kept only so ``parity_check`` can keep
  measuring the gap.
- ``online`` — the serving path itself: ``input_images`` + ``task`` attr ->
  ``Observation.from_dict`` -> ``model.run_stage1``. Exact by construction;
  the libraries, shadow tables and overhead runs use this (§3.8 fallback).

The artifact records ``stage1_path``; shadow / overhead runs must use the
same path as the library they score against.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

STAGE1_PATHS = ("offline", "online")
DEFAULT_STAGE1_PATH = "online"


def observation_from_h5(group: h5py.Group, task: str, tokenizer, model, device: torch.device):
    """Rebuild the model ``Observation`` of one step from the raw H5 fields
    (``input_images`` are the post-transform uint8 frames the collector saw)."""
    images = {}
    masks = {}
    stored = group["input_images"]
    for key in IMAGE_KEYS:
        if key in stored:
            img = torch.from_numpy(np.array(stored[key]))
            images[key] = img[None, ...].to(device)
            masks[key] = torch.tensor([True], device=device)
        else:
            # Absent camera: zero frame + mask False, exactly what the LIBERO
            # input transform sends for the unused wrist slot.
            ref = next(iter(images.values())) if images else torch.zeros(1, 224, 224, 3, dtype=torch.uint8, device=device)
            images[key] = torch.zeros_like(ref)
            masks[key] = torch.tensor([False], device=device)
    state_np = np.array(group["robot_state"], dtype=np.float32)
    state = torch.from_numpy(state_np)[None, ...].to(device)
    tok_state = state_np if model.config.discrete_state_input else None
    tokens_np, mask_np = tokenizer.tokenize(task, state=tok_state)
    return _model.Observation.from_dict({
        "image": images, "image_mask": masks, "state": state,
        "tokenized_prompt": torch.from_numpy(tokens_np).long()[None, ...].to(device),
        "tokenized_prompt_mask": torch.from_numpy(mask_np).bool()[None, ...].to(device),
    })


def rebuild_stage1(group: h5py.Group, task: str, tokenizer, model, device: torch.device, path: str):
    """``Stage1Output`` (or the offline look-alike) of one H5 step via ``path``."""
    if path == "online":
        return model.run_stage1(observation_from_h5(group, task, tokenizer, model, device))
    if path == "offline":
        from exp.common.build_in_memory_cache_artifact import _build_fake_stage1_with_masks

        return _build_fake_stage1_with_masks(group, task_str=task, tokenizer=tokenizer, model=model, device=device)
    raise ValueError(f"unknown stage1 path {path!r}; expected one of {STAGE1_PATHS}")


__all__ = ["DEFAULT_STAGE1_PATH", "STAGE1_PATHS", "observation_from_h5", "rebuild_stage1"]
