"""Initialize from released LIBERO weights; save small, strict MT50 deltas."""
from __future__ import annotations

import json
from pathlib import Path
import re
from contextlib import contextmanager

BASE = "/data/mw_finetune/models/fastwam_libero"
WAN = "/data/xwam/wan22_5b"


@contextmanager
def skip_overwritten_initialization():
    """Avoid randomly filling ~6B weights that immediately load from LIBERO.

    All tensors are strictly audited below, and all new embodiment layers are
    explicitly reset after construction. Restore Torch initializers on errors.
    """
    import torch
    names = ("kaiming_uniform_", "xavier_uniform_", "normal_", "uniform_")
    saved = {name: getattr(torch.nn.init, name) for name in names}
    try:
        for name in names:
            setattr(torch.nn.init, name, lambda tensor, *args, **kw: tensor)
        yield
    finally:
        for name, function in saved.items():
            setattr(torch.nn.init, name, function)


def canonical_key(key):
    # safetensors stores one of several aliased module paths. Normalize all.
    if key.startswith("model.dit."):
        key = "model.mot." + key[len("model.dit."):]
    for name in ("video", "action"):
        if key.startswith(f"model.{name}_expert."):
            key = f"model.mot.mixtures.{name}." + key[len(f"model.{name}_expert."):]
    return re.sub(r"model\.mot\.mixtures\.([^.]+)\.blocks\.(\d+)\.", r"model.mot.layers.\2.blocks.\1.", key)


def make_config(base=BASE, device="cuda", wan_root=WAN, checkpointing=True):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    # Use checkpoint config rather than LeRobot defaults (action shift is 1).
    cfg = FastWAMConfig.from_pretrained(base)
    cfg.device = device
    cfg.dtype = torch.bfloat16
    cfg.action_dim = cfg.proprio_dim = 4
    cfg.action_horizon = cfg.n_action_steps = 32
    cfg.image_size = (224, 224)
    cfg.num_video_frames = 33
    cfg.action_video_freq_ratio = 4
    cfg.toggle_action_dimensions = []
    cfg.freeze_video_expert = True
    cfg.loss = {"lambda_action": 1.0, "lambda_video": 0.0}
    cfg.tokenizer_model_id = str(Path(wan_root) / "google/umt5-xxl")
    cfg.model_id = wan_root
    cfg.load_text_encoder = True
    cfg.use_gradient_checkpointing = checkpointing
    cfg.mot_checkpoint_mixed_attn = checkpointing
    for expert in (cfg.video_dit_config, cfg.action_dit_config):
        expert["action_dim"] = 4
        expert["use_gradient_checkpointing"] = checkpointing
    cfg.input_features = {
        "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (4,))}
    cfg.validate_features()
    return cfg


def build_model(base=BASE, device="cuda", wan_root=WAN, checkpointing=True, fp32_trainable=False, mode="frozen_video"):
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from .raw_wan import install_raw_loaders

    install_raw_loaders(wan_root)
    cfg = make_config(base, device, wan_root, checkpointing)
    torch.manual_seed(42)
    with skip_overwritten_initialization():
        policy = FastWAMPolicy(cfg)
    torch.manual_seed(42)
    policy.model.action_expert.action_encoder.reset_parameters()
    policy.model.action_expert.head.reset_parameters()
    policy.model.proprio_encoder.reset_parameters()
    state = {canonical_key(k): v for k, v in load_file(str(Path(base) / "model.safetensors"), device="cpu").items()}
    # Do not use permissive upstream loading: audit every unique parameter.
    params = {canonical_key(k): v for k, v in policy.named_parameters()}
    mismatched = {k for k, p in params.items() if k in state and tuple(state[k].shape) != tuple(p.shape)}
    allowed = ("action_encoder.", "mixtures.action.head.", "proprio_encoder.")
    if any(not any(part in k for part in allowed) for k in mismatched):
        raise RuntimeError(f"Unexpected LIBERO shape changes: {sorted(mismatched)}")
    reset = {k for k in params if any(part in k for part in allowed)}
    missing = set(params) - set(state)
    if missing:
        raise RuntimeError(f"LIBERO is missing model parameters: {sorted(missing)[:20]}")
    with torch.no_grad():
        for key, param in params.items():
            if key not in reset:
                param.copy_(state[key].to(device=param.device, dtype=param.dtype))
    del state
    # Frozen blocks were reparented onto MoT layers. Verify both paths.
    for name, p in policy.named_parameters():
        video = ".blocks.video." in name or "video_expert." in name or ".mixtures.video." in name
        if video:
            p.requires_grad_(False)
    if mode == "video_lora":
        from .lora import inject_video_lora
        lora = inject_video_lora(policy)
        policy.config.loss["lambda_video"] = policy.model.loss_lambda_video = 1.0
    elif mode == "frozen_video":
        lora = None
    else:
        raise ValueError(f"Unknown training mode {mode}")
    if fp32_trainable:
        for p in policy.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    policy._mt50_load_audit = {"reset": sorted(reset), "loaded_unique_tensors": len(params) - len(reset),
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in policy.parameters() if not p.requires_grad), "lora": lora, "mode": mode}
    return policy


def delta_state(policy):
    return {canonical_key(k): p.detach().cpu().contiguous() for k, p in policy.named_parameters() if p.requires_grad}


def load_delta(policy, checkpoint):
    from safetensors.torch import load_file
    import torch
    state = load_file(str(Path(checkpoint) / "action.safetensors"))
    params = {canonical_key(k): p for k, p in policy.named_parameters() if p.requires_grad}
    if set(state) != set(params):
        raise ValueError("MT50 delta parameter keys differ from the exact expected trainable set")
    with torch.no_grad():
        for k, p in params.items():
            if p.shape != state[k].shape:
                raise ValueError(f"MT50 delta shape mismatch: {k}")
            p.copy_(state[k].to(device=p.device, dtype=p.dtype))


def save_delta(policy, destination, metadata):
    from safetensors.torch import save_file
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    save_file(delta_state(policy), str(dest / "action.safetensors.tmp"))
    (dest / "action.safetensors.tmp").replace(dest / "action.safetensors")
    (dest / "mt50.json.tmp").write_text(json.dumps(metadata, indent=2))
    (dest / "mt50.json.tmp").replace(dest / "mt50.json")
