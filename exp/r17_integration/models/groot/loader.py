"""Load B's MW policy only on an explicit call; never touch training checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import checkpoint_manifest, validate_training_metadata


def load_backend(checkpoint, *, device="cuda:0", denoising_steps=4, policy_loader=None):
    validate_training_metadata(json.loads((Path(checkpoint) / "experiment_cfg/metadata.json").read_text()))
    if type(denoising_steps) is not int or denoising_steps not in (1, 2, 4):
        raise ValueError("endpoint K must be 4, 2, or 1")
    if policy_loader is None:
        from exp.mw_finetune.groot.policy import load_policy
        policy_loader = load_policy
    policy = policy_loader(checkpoint, device=device, denoising_steps=denoising_steps)
    meta = policy.get_metadata()
    expected = {"embodiment": "metaworld_sawyer", "embodiment_id": 31, "action_horizon": 16,
                "action_dim": 4, "state_dim": 4, "denoising_steps": denoising_steps}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("loaded policy metadata differs from MetaWorld embodiment")
    if policy.backend.model.training:
        raise ValueError("loaded GR00T must be in eval mode")
    return policy.backend


def load_adapter(spec, *, checkpoint=None, device="cuda:0", backend=None, manifest=None,
                 tokenizer_files=(), code_files=(), **kwargs):
    """A supplied backend is useful for CPU conformance; production uses B's loader."""
    from .provider import GrootMetaWorldAdapter
    if manifest is None:
        if hasattr(spec, "model_id"):
            if checkpoint is None:
                raise ValueError("ModelSpec loading requires an immutable manifest or checkpoint content inputs")
            manifest = checkpoint_manifest(checkpoint, denoising_steps=spec.default_sampler.num_steps,
                                           tokenizer_files=tokenizer_files, code_files=code_files)
        else:
            manifest = spec
    if backend is None:
        if checkpoint is None:
            raise ValueError("explicit immutable checkpoint required")
        backend = load_backend(checkpoint, device=device, denoising_steps=manifest.denoising_steps)
    adapter = GrootMetaWorldAdapter(backend, manifest, **kwargs)
    if hasattr(spec, "model_id") and adapter.metadata() != spec:
        raise ValueError("loaded adapter spec differs from requested ModelSpec")
    return adapter


def load_checkpoint(checkpoint, *, device="cuda:0", denoising_steps=4, tokenizer_files=(), code_files=()):
    manifest = checkpoint_manifest(checkpoint, denoising_steps=denoising_steps,
                                   tokenizer_files=tokenizer_files, code_files=code_files)
    backend = load_backend(checkpoint, device=device, denoising_steps=denoising_steps)
    return backend, manifest
