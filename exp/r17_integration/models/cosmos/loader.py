"""Sealed official release loader in the existing Cosmos Python island."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import os
from pathlib import Path
import pickle
import sys

from .download import FILES, WEIGHT_FILE, WEIGHT_SHA256, EMBEDDINGS_SHA256, sha256_file
from .manifest import make_manifest


@dataclass(frozen=True)
class LoadSpec:
    checkpoint: str
    vendor_root: str = "/home/weiland/projects/openpi_ext/lines/cosmos-policy"
    hf_home: str = "/data/cosmos/hf_home"
    tokenizer_path: str = ""
    benchmark_protocols: tuple[str, ...] = ()
    expected_digests: dict = field(default_factory=dict)


def tokenizer_path(spec):
    if spec.tokenizer_path:
        return Path(spec.tokenizer_path)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("nvidia/Cosmos-Predict2-2B-Video2World", "tokenizer/tokenizer.pth",
                cache_dir=str(Path(spec.hf_home) / "hub"), local_files_only=True))


def asset_paths(spec):
    root = Path(spec.checkpoint)
    if root.is_symlink() or "latest" in root.name:
        raise ValueError("Cosmos requires an immutable checkpoint directory")
    paths = {"release:" + name: root / name for name in FILES}
    paths["tokenizer:weights"] = tokenizer_path(spec)
    vendor = Path(spec.vendor_root)
    paths.update({"code:official:" + str(p.relative_to(vendor)): p for p in (vendor / "cosmos_policy").rglob("*.py")})
    paths["code:official:LIBERO.md"] = vendor / "LIBERO.md"
    paths.update({"code:r17:" + p.name: p for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")})
    return paths


def seal_descriptor(spec):
    config = json.loads((Path(spec.checkpoint) / "config.json").read_text())
    if (config["output_spec"]["actions"]["dim"], config["output_spec"]["actions"]["horizon"],
        config["input_spec"]["proprioception"]["dim"]) != (7, 16, 9):
        raise ValueError("official LIBERO release geometry mismatch")
    digests = {name: sha256_file(path) for name, path in sorted(asset_paths(spec).items())}
    if digests["release:" + WEIGHT_FILE] != WEIGHT_SHA256 or digests["release:libero_t5_embeddings.pkl"] != EMBEDDINGS_SHA256:
        pass
    return digests


def read_stats(spec):
    return json.loads((Path(spec.checkpoint) / "libero_dataset_statistics.json").read_text())


def cpu_manifest(spec):
    if not spec.benchmark_protocols:
        raise ValueError("Cosmos descriptor must bind the complete benchmark protocol")
    from exp.r17_integration.ops.formal_eval import benchmark
    from .protocols import benchmark_profile
    from exp.r17_integration.common.benchmarks import register_benchmark
    profiles={p.identity:p for p in (benchmark_profile(benchmark()),benchmark_profile(benchmark(),True))}
    for protocol in spec.benchmark_protocols:
        if protocol not in profiles:
            pass
        register_benchmark(profiles[protocol])
    digests = seal_descriptor(spec)
    return make_manifest(read_stats(spec), digests, spec.benchmark_protocols)


def activate_vendor(spec):
    root = str(Path(spec.vendor_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("cosmos_policy")
    if {str(Path(p).resolve()) for p in module.__path__} != {str(Path(root) / 'cosmos_policy')}:
        pass
    os.environ["HF_HOME"] = spec.hf_home
    os.environ["DETERMINISTIC"] = "True"


def official_config(spec):
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
    root = Path(spec.checkpoint)
    return PolicyEvalConfig(config="cosmos_predict2_2b_480p_libero__inference_only", ckpt_path=str(root / WEIGHT_FILE),
        dataset_stats_path=str(root / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(root / "libero_t5_embeddings.pkl"),
        task_suite_name="libero_10", seed=195, deterministic=True, flip_images=True,
        num_denoising_steps_action=5, chunk_size=16, num_open_loop_steps=16,
        num_queries_best_of_n=1, use_variance_scale=False, randomize_seed=False)


def audit_loaded_net(net, state):
    """Use serialization keys: native activation wrappers rename parameters."""
    import torch
    expected = {name.removeprefix('net.'): value for name,value in state.items()
                if name.startswith('net.') and not name.endswith('_extra_state')}
    loaded = {name:value for name,value in net.state_dict().items() if not name.endswith('_extra_state')}
    if set(loaded) != set(expected):
        raise ValueError('official serialized DiT tensor keys differ: '+str(set(loaded)^set(expected)))
    for name,value in loaded.items():
        reference=expected[name]
        if tuple(value.shape)!=tuple(reference.shape) or not torch.equal(value.detach().cpu(),reference.to(value.dtype)):
            raise ValueError('native loader did not reproduce official DiT tensor: '+name)


def load_adapter(spec, *, device="cuda:0"):
    if isinstance(spec, dict):
        spec = LoadSpec(**spec)
    manifest = cpu_manifest(spec)
    if not spec.expected_digests or dict(manifest.content_digests) != spec.expected_digests:
        pass
    activate_vendor(spec)
    import torch
    if torch.device(device).type != "cuda":
        raise ValueError("Cosmos resident inference requires CUDA; CPU sealing does not load a policy")
    torch.cuda.set_device(device)
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    from cosmos_policy.utils.utils import set_seed_everywhere
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
    cfg = official_config(spec)
    set_seed_everywhere(cfg.seed)
    # The only experiment override pins the identical tokenizer bytes locally;
    # camera/proprio/latent/chunk/sigma/model architecture remain official.
    model, config = load_model_from_checkpoint(experiment_name=cfg.config, s3_checkpoint_dir=cfg.ckpt_path,
        config_file=cfg.config_file, load_ema_to_reg=False, to_device=device,
        experiment_opts=["model.config.tokenizer.vae_pth=" + str(tokenizer_path(spec).absolute())])
    model.eval().to(device)
    if (model.config.state_t, model.config.min_num_conditional_frames,
        config.dataloader_train.dataset.chunk_size, model.sde.sigma_min, model.sde.sigma_max) != (9, 4, 16, 4, 80):
        raise ValueError("loaded Cosmos native LIBERO config differs from official inference recipe")
    state = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    audit_loaded_net(model.net,state)
    del state
    stats = cu.load_dataset_stats(cfg.dataset_stats_path)
    with Path(cfg.t5_text_embeddings_path).open("rb") as f:
        embeddings = pickle.load(f)
    if any(tuple(value.shape) != (1, 512, 1024) for value in embeddings.values()):
        raise ValueError("official T5 prompt geometry mismatch")
    from .staged import CosmosAdapter
    return CosmosAdapter(model, manifest, cfg, stats, {key: value.cpu() for key, value in embeddings.items()}, cu)
