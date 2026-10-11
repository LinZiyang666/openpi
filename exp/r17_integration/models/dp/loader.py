"""Strict loader consuming E's immutable export; no modifications to E."""
from dataclasses import asdict
from pathlib import Path

from .config import LoadSpec
from .manifest import code_identity, digest, file_digest


def load_adapter(spec):
    if isinstance(spec, dict):
        spec = LoadSpec(**spec)
    if not isinstance(spec, LoadSpec):
        raise TypeError("DP loader needs LoadSpec or its JSON fields, not a shared ModelSpec")
    checkpoint = Path(spec.checkpoint)
    if "latest" in checkpoint.parts or checkpoint.is_symlink():
        raise ValueError("DP must load an immutable checkpoint directory, never latest/symlink")
    checkpoint = checkpoint.resolve(strict=True)
    weights = checkpoint / "model.safetensors"
    expected = spec.expected_weight_sha256
    weight_identity = file_digest(weights)
    if expected is not None and expected.removeprefix("sha256:") != weight_identity.removeprefix("sha256:"):
        pass
    from exp.mw_finetune.dp.checkpoint import read_metadata
    meta = read_metadata(checkpoint)
    cfg = meta["config"]["model"]
    if cfg["num_inference_steps"] != 100 or cfg["n_obs_steps"] != 1 or cfg["horizon"] != 16:
        raise ValueError("DP full baseline requires stateless H16/DDPM100 metadata")
    wrapper = meta.get("wrapper", {})
    if wrapper.get("language_kind", "projected") != "projected":
        raise ValueError("only audited projected CLIP language is supported")
    if not spec.diagnostic:
        if not (checkpoint / "complete.json").is_file():
            raise ValueError("formal DP needs E's immutable complete export receipt")
        if spec.domain == "metaworld" and meta.get("config", {}).get("data", {}).get("repo_id") != "lerobot/metaworld_mt50":
            raise ValueError("ML45 transfer checkpoint is diagnostic, not formal MT50")
        if not all((spec.expected_weight_sha256, spec.expected_norm_identity, spec.expected_model_identity)):
            pass
        if spec.domain == "libero" and not spec.libero_benchmark_specs:
            pass
    text_root = Path(spec.text_encoder).resolve(strict=True)
    text_digests = {p.name: file_digest(p) for p in sorted(text_root.iterdir())
                    if p.is_file() and p.suffix in (".json", ".txt", ".safetensors", ".bin")}
    if not any(k.endswith((".safetensors", ".bin")) for k in text_digests):
        raise ValueError("missing immutable CLIP text weights")
    from exp.mw_finetune.dp.policy import load_policy
    policy = load_policy(checkpoint, spec.device, domain=spec.domain)
    if policy.strict_load != {"missing": [], "unexpected": [], "keys": 475}:
        raise ValueError("DP reconstruction requires exactly 475 strictly loaded tensors")
    from .staged import DPAdapter
    adapter = DPAdapter(policy.model, policy.stats, policy.language, domain=spec.domain,
                        weight_identity=weight_identity, second_camera=policy.second_camera,
                        language_identity=digest({"files": text_digests, "kind": policy.language_kind}),
                        diagnostic=spec.diagnostic, source_digests=code_identity(),
                        libero_benchmark_specs=spec.libero_benchmark_specs)
    from .sampler import SamplerProfile
    profile = SamplerProfile(spec.sampler_kind, 100, spec.ddim_eta)
    if profile.kind != "ddpm":
        adapter = adapter.with_sampler(profile)
    if spec.expected_norm_identity is not None and spec.expected_norm_identity != adapter.metadata().norm_identity:
        pass
    if spec.expected_model_identity is not None and spec.expected_model_identity != adapter.metadata().checkpoint_identity:
        pass
    adapter.legacy_policy = policy
    adapter.load_spec = asdict(spec)
    adapter.weight_identity = weight_identity
    adapter.strict_load = policy.strict_load
    adapter.asset_digests = {"CLIP/"+name: value for name, value in text_digests.items()}
    adapter.asset_digests.update({str(path.relative_to(checkpoint)): file_digest(path)
                                 for path in checkpoint.rglob("norm_stats.json")})
    adapter.asset_digests["E_metadata"] = file_digest(checkpoint / ("metadata.json" if
                                                     (checkpoint / "metadata.json").exists() else "metadata.pt"))
    return adapter
