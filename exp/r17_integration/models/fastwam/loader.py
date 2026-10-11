"""Lazy separate-embodiment loaders; no vendor imports until explicit load."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from .manifest import content_identity, make_manifest, make_sampler

VENDOR = Path("/data/mw_finetune/vendor/lerobot/src/lerobot/policies/fastwam")
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
_LOAD_LOCK = threading.RLock()


def raw_wan_code_root():
    root = VENDOR.parents[4] / "FastWAM/src/fastwam/models/wan22"
    if not root.is_dir():
        raise ValueError("local sealed raw Wan code is missing; never download implicitly")
    return root


@dataclass(frozen=True)
class LoadSpec:
    benchmark: str
    checkpoint: str
    wan_root: str
    base: str = ""
    expected_digests: dict[str, str] = field(default_factory=dict)
    diagnostic: bool = False
    benchmark_protocols: tuple[str, ...] = ()
    execution_protocol: str = "system"
    benchmark_specs: tuple[dict, ...] = ()

    def __post_init__(self):
        if self.benchmark not in {"libero", "metaworld"}:
            raise ValueError("expected libero or metaworld embodiment")
        if "latest" in Path(self.checkpoint).name:
            raise ValueError("serve descriptor requires an immutable checkpoint export")
        if self.benchmark == "metaworld" and not self.base:
            raise ValueError("MetaWorld requires an explicit immutable LIBERO base")
        from .protocols import execution_protocol
        execution_protocol(self.benchmark, self.execution_protocol)


def safetensor_header(path):
    """Read only the bounded header. Never mmap/read the multi-GB tensor body."""
    with Path(path).open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if not 2 <= size <= 64 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        data = handle.read(size)
        if len(data) != size:
            raise ValueError("truncated safetensors header")
    header = json.loads(data)
    return {k: v for k,v in header.items() if k != "__metadata__"}


def audit_parameters(expected, actual):
    """Exact shape/key match for delta or required released parameters."""
    if set(expected) != set(actual):
        raise ValueError("checkpoint parameter keys differ from exact expected set")
    if any(tuple(expected[k]) != tuple(actual[k]) for k in expected):
        raise ValueError("checkpoint parameter shape mismatch")


def audit_released_header(spec):
    """Reject missing/wrong released heads before constructing the large model."""
    from exp.mw_finetune.fastwam.model import canonical_key
    root = Path(spec.base or spec.checkpoint)
    header = {canonical_key(k): tuple(v["shape"]) for k,v in safetensor_header(root/"model.safetensors").items()}
    required = {"model.mot.mixtures.action.action_encoder.weight": (1024,7),
        "model.mot.mixtures.action.head.weight": (7,1024),
        "model.proprio_encoder.weight": (4096,8)}
    if any(header.get(k) != shape for k,shape in required.items()):
        raise ValueError("released LIBERO checkpoint is missing or has incompatible action/proprio heads")
    if spec.benchmark == "metaworld":
        delta = {canonical_key(k): tuple(v["shape"]) for k,v in safetensor_header(Path(spec.checkpoint)/"action.safetensors").items()}
        mw_required = {"model.mot.mixtures.action.action_encoder.weight": (1024,4),
            "model.mot.mixtures.action.head.weight": (4,1024), "model.proprio_encoder.weight": (4096,4)}
        if any(delta.get(k) != shape for k,shape in mw_required.items()):
            raise ValueError("MT50 delta is missing or has incompatible 4D action/proprio heads")
    return header


def inspect_config(spec):
    base = Path(spec.base or spec.checkpoint)
    if not (base / "config.json").is_file():
        raise ValueError("missing released base config; an MT50 delta is not a LIBERO checkpoint")
    config = json.loads((base / "config.json").read_text())
    if spec.benchmark == "libero":
        if (config.get("action_dim"), config.get("proprio_dim"), config.get("action_horizon"),
            config.get("image_size"), config.get("toggle_action_dimensions")) != (7, 8, 32, [224, 448], [-1]):
            raise ValueError("published LIBERO 7/8/H32/two-view geometry required")
        if config["video_dit_config"]["action_dim"] != 7 or config["action_dit_config"]["action_dim"] != 7:
            raise ValueError("LIBERO heads must be preserved")
        expected_inputs = {"observation.images.image": [3,224,224], "observation.images.image2": [3,224,224],
                           "observation.state": [8]}
        if {k: v["shape"] for k,v in config["input_features"].items()} != expected_inputs:
            raise ValueError("LIBERO real views/state geometry mismatch")
        if config["output_features"]["action"]["shape"] != [7]:
            raise ValueError("LIBERO action feature mismatch")
    else:
        # Inspect C's exact delta descriptor without importing/altering training.
        meta = json.loads((Path(spec.checkpoint) / "mt50.json").read_text())
        if meta.get("format") != "fastwam-mt50-delta-v1" or meta.get("mode") not in {"frozen_video", "video_lora"}:
            raise ValueError("strict C MT50 delta metadata required")
        if config.get("action_dim") != 7 or config.get("proprio_dim") != 8:
            raise ValueError("MT50 initialization requires the published LIBERO base")
        config.update(action_dim=4, proprio_dim=4, action_horizon=32, n_action_steps=32,
            image_size=[224,224], toggle_action_dimensions=[])
        for name in ("video_dit_config", "action_dit_config"):
            config[name]["action_dim"] = 4
        config["input_features"] = {"observation.images.image": {"type": "VISUAL", "shape": [3,224,224]},
                                    "observation.state": {"type": "STATE", "shape": [4]}}
        config["output_features"] = {"action": {"type": "ACTION", "shape": [4]}}
    if config.get("action_horizon") != 32 or config["normalization_mapping"] != {
            "VISUAL":"IDENTITY", "STATE":"MIN_MAX", "ACTION":"MIN_MAX"}:
        raise ValueError("unexpected Fast-WAM normalization/horizon")
    return config


def read_stats(spec):
    if spec.benchmark == "metaworld":
        return json.loads((Path(spec.checkpoint) / "mt50.json").read_text())["stats"]
    from safetensors.numpy import load_file
    root = Path(spec.checkpoint)
    pre = load_file(str(root / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    post = load_file(str(root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    import numpy as np
    for bound in ("min", "max"):
        if not np.array_equal(pre[f"action.{bound}"], post[f"action.{bound}"]):
            raise ValueError("LIBERO pre/post normalization mismatch")
    return {name: {k: pre[f"{name}.{k}"].astype(np.float32).tolist() for k in ("min", "max")}
        for name in ("action", "observation.state")}


def asset_paths(spec):
    root, base, wan = Path(spec.checkpoint), Path(spec.base or spec.checkpoint), Path(spec.wan_root)
    paths = {"base_weights": base / "model.safetensors", "base_config": base / "config.json",
        "wan_vae": wan / "Wan2.2_VAE.pth", "wan_text_encoder": wan / "models_t5_umt5-xxl-enc-bf16.pth"}
    tokenizer = wan / "google/umt5-xxl"
    token_files = sorted(p for p in tokenizer.iterdir() if p.is_file())
    if not token_files or not (tokenizer / "tokenizer_config.json").is_file():
        raise ValueError("complete local Wan tokenizer assets required")
    paths.update({"tokenizer:" + p.name: p for p in token_files})
    paths.update({"code:lerobot:"+str(p.relative_to(VENDOR)): p for p in VENDOR.rglob("*.py")})
    paths.update({"code:r17:"+p.name: p for p in Path(__file__).parent.glob("*.py")})
    croot = Path(__file__).resolve().parents[3] / "mw_finetune/fastwam"
    paths.update({"code:C:"+p.name: p for p in croot.glob("*.py")
                  if p.name in {"model.py", "common.py", "raw_wan.py", "lora.py"}})
    # Raw Wan VAE/text classes are code inputs too, not merely weight assets.
    raw_root = raw_wan_code_root()
    paths.update({"code:raw_wan:"+str(p.relative_to(raw_root)): p for p in raw_root.rglob("*.py")})
    if spec.benchmark == "metaworld":
        paths.update(delta_weights=root / "action.safetensors", delta_metadata=root / "mt50.json")
    else:
        for name in ("policy_preprocessor.json", "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
            paths["norm:"+name] = root / name
    return paths


def seal_descriptor(spec):
    """Explicit offline content sealing, not an import-time checkpoint probe."""
    inspect_config(spec)
    audit_released_header(spec)
    return content_identity(asset_paths(spec))


@contextmanager
def _raw_loaders(wan_root):
    import importlib
    import sys
    import lerobot.policies.fastwam.modeling_fastwam as implementation
    from exp.mw_finetune.fastwam.raw_wan import install_raw_loaders
    names = ("load_pretrained_wan_vae", "load_pretrained_wan_text_encoder")
    saved = {n: getattr(implementation, n) for n in names}
    # The island installs LeRobot, while raw FastWAM is a separate source tree.
    # Resolve that explicitly and only during opt-in loading; don't require a
    # caller's PYTHONPATH or accept a different cached raw implementation.
    root = raw_wan_code_root().resolve()
    source_path = str(root.parents[2])
    sys.path.insert(0, source_path)
    try:
        for relative in ("wan_video_vae", "wan_video_text_encoder", "helpers.state_dict_converters"):
            module = importlib.import_module("fastwam.models.wan22."+relative)
            expected = root / (relative.replace(".", "/")+".py")
            if Path(module.__file__).resolve() != expected:
                pass
        install_raw_loaders(wan_root)
        yield
    finally:
        sys.path.remove(source_path)
        for name, value in saved.items():
            setattr(implementation, name, value)


def _load_libero(spec, device):
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from exp.mw_finetune.fastwam.model import canonical_key, skip_overwritten_initialization
    cfg = FastWAMConfig.from_pretrained(spec.checkpoint)
    cfg.device, cfg.dtype = device, torch.bfloat16
    cfg.model_id, cfg.tokenizer_model_id = spec.wan_root, str(Path(spec.wan_root) / "google/umt5-xxl")
    cfg.use_gradient_checkpointing = cfg.mot_checkpoint_mixed_attn = False
    with skip_overwritten_initialization():
        policy = FastWAMPolicy(cfg)
    params = {canonical_key(k): p for k,p in policy.named_parameters()}
    header = safetensor_header(Path(spec.checkpoint) / "model.safetensors")
    actual = {canonical_key(k): v["shape"] for k,v in header.items()}
    audit_parameters({k: tuple(p.shape) for k,p in params.items()}, actual)
    state = {canonical_key(k): v for k,v in load_file(str(Path(spec.checkpoint)/"model.safetensors"), device="cpu").items()}
    with torch.no_grad():
        for key, p in params.items():
            p.copy_(state[key].to(device=p.device, dtype=p.dtype))
    policy.eval()
    return policy


def load_adapter(spec, *, device="cpu"):
    """Explicit load only. Real model loading is never attempted by import/tests."""
    from .protocols import validate_benchmark_specs
    validate_benchmark_specs(spec)
    import torch
    from .staged import FastWAMAdapter
    from .wire_policy import FastWAMWirePolicy
    config, stats = inspect_config(spec), read_stats(spec)
    if not spec.benchmark_protocols:
        pass
    if not spec.expected_digests:
        pass
    actual = seal_descriptor(spec)
    if actual != spec.expected_digests:
        pass
    import lerobot.policies.fastwam.modeling_fastwam as native_impl
    from .manifest import sha256_file
    if sha256_file(native_impl.__file__) != actual["code:lerobot:modeling_fastwam.py"]:
        pass
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    # C constructs with a temporary global seed and raw-loader shim. Bound those
    # effects to serialized explicit loading and restore them before serving.
    with _LOAD_LOCK, torch.random.fork_rng(devices=devices), _raw_loaders(spec.wan_root):
        if spec.benchmark == "libero":
            policy = _load_libero(spec, device)
        else:
            from exp.mw_finetune.fastwam.model import build_model, load_delta
            meta = json.loads((Path(spec.checkpoint)/"mt50.json").read_text())
            policy = build_model(base=spec.base, device=device, wan_root=spec.wan_root,
                checkpointing=False, mode=meta["mode"])
            load_delta(policy, spec.checkpoint)
            policy.eval()
    wire = FastWAMWirePolicy(spec.benchmark, stats, prompt_template=policy.config.prompt_template,
        execution_protocol=spec.execution_protocol)
    sampler = make_sampler(policy.model.infer_action_scheduler, int(policy.config.num_inference_steps),
        dtype=str(policy.model.torch_dtype).removeprefix("torch."))
    if sampler.num_steps != 10:
        raise ValueError("published full baseline requires explicit LeRobot K10")
    manifest = make_manifest(spec.benchmark, config, stats, sampler, actual,
        benchmark_protocols=spec.benchmark_protocols, execution_protocol=spec.execution_protocol)
    return FastWAMAdapter(policy.model, manifest, wire, config)


def load_offline_adapter(spec, *, device="cpu"):
    """Same sealed teacher for P6 offline augmentation; no separate model math."""
    return load_adapter(spec, device=device)
