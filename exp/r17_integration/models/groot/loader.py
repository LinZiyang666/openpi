"""Load native benchmark policies only on an explicit call."""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import checkpoint_manifest, validate_training_metadata


def checkpoint_paths(checkpoint, *, suite=None):
    """Historical LIBERO checkpoints are separate siblings, one per suite."""
    if isinstance(checkpoint, dict):
        return checkpoint
    path = Path(checkpoint)
    if (path / "config.json").is_file():
        return str(path)
    suites = (suite,) if suite else ("libero_spatial", "libero_10")
    return {s:str(path / ("n15_" + s)) for s in suites}


def load_backend(checkpoint, *, device="cuda:0", denoising_steps=4, policy_loader=None, benchmark="metaworld"):
    if isinstance(checkpoint, dict):
        return {suite:load_backend(path, device=device, denoising_steps=denoising_steps,
            policy_loader=policy_loader, benchmark=benchmark) for suite,path in checkpoint.items()}
    if benchmark == "metaworld":
        validate_training_metadata(json.loads((Path(checkpoint) / "experiment_cfg/metadata.json").read_text()))
    if type(denoising_steps) is not int or denoising_steps not in (1, 2, 3, 4):
        raise ValueError("endpoint K must be 1-4")
    if policy_loader is None:
        if benchmark == "libero":
            from custom_data_config import LiberoDataConfig
            from gr00t.model.policy import Gr00tPolicy
            config = LiberoDataConfig()
            return Gr00tPolicy(model_path=str(checkpoint), embodiment_tag="new_embodiment",
                modality_config=config.modality_config(), modality_transform=config.transform(),
                denoising_steps=denoising_steps, device=device)
        else:
            from exp.mw_finetune.groot.policy import load_policy
            policy_loader = load_policy
    policy = policy_loader(checkpoint, device=device, denoising_steps=denoising_steps)
    if benchmark == "libero":
        return policy
    meta = policy.get_metadata()
    expected = {"embodiment": "metaworld_sawyer", "embodiment_id": 31, "action_horizon": 16,
                "action_dim": 4, "state_dim": 4, "denoising_steps": denoising_steps}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("loaded policy metadata differs from MetaWorld embodiment")
    if policy.backend.model.training:
        raise ValueError("loaded GR00T must be in eval mode")
    return policy.backend


def load_adapter(spec, *, checkpoint=None, device="cuda:0", backend=None, manifest=None,
                 tokenizer_files=(), code_files=(), benchmark="metaworld", benchmark_protocol=None, **kwargs):
    """A supplied backend is useful for CPU conformance; production uses B's loader."""
    from .provider import GrootMetaWorldAdapter
    if manifest is None:
        if hasattr(spec, "model_id"):
            if checkpoint is None:
                raise ValueError("ModelSpec loading requires an immutable manifest or checkpoint content inputs")
            manifest = checkpoint_manifest(checkpoint, denoising_steps=spec.default_sampler.num_steps,
                                           tokenizer_files=tokenizer_files, code_files=code_files,
                                           benchmark=benchmark, benchmark_protocol=benchmark_protocol)
        else:
            manifest = spec
    if backend is None:
        if checkpoint is None:
            raise ValueError("explicit immutable checkpoint required")
        backend = load_backend(checkpoint, device=device, denoising_steps=manifest.denoising_steps, benchmark=manifest.benchmark)
    adapter = GrootMetaWorldAdapter(backend, manifest, **kwargs)
    if hasattr(spec, "model_id") and adapter.metadata() != spec:
        raise ValueError("loaded adapter spec differs from requested ModelSpec")
    return adapter


def load_checkpoint(checkpoint, *, device="cuda:0", denoising_steps=4, tokenizer_files=(), code_files=(),
                    benchmark="metaworld", benchmark_protocol=None):
    manifest = checkpoint_manifest(checkpoint, denoising_steps=denoising_steps,
                                   tokenizer_files=tokenizer_files, code_files=code_files,
                                   benchmark=benchmark, benchmark_protocol=benchmark_protocol)
    backend = load_backend(checkpoint, device=device, denoising_steps=denoising_steps, benchmark=benchmark)
    return backend, manifest
