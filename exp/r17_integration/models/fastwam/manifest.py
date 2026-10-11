"""Content identities and CPU-safe Fast-WAM model descriptors (R17 v1)."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, FeatureSpec, ModelManifest,
    ModelSpec, SamplerSpec, digest,
)

SAMPLER_ID = "fastwam_shifted_flow_v1"
ISLAND = "/data/mw_finetune/envs/fastwam/bin/python"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_identity(paths):
    """Explicit, streaming hash; never substitutes mtime/size for content."""
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def make_sampler(scheduler, k=10, *, dtype="bfloat16", sigma_shift=None):
    import torch
    if type(k) is not int or k < 1:
        raise ValueError("Fast-WAM K must be a positive integer")
    torch_dtype = getattr(torch, dtype)
    times, deltas = scheduler.build_inference_schedule(
        num_inference_steps=k, device=torch.device("cpu"), dtype=torch_dtype,
        shift_override=sigma_shift,
    )
    return SamplerSpec(SAMPLER_ID, "shifted_flow", k,
        tuple(times.float().tolist()), tuple(deltas.float().tolist()),
        {"shift": float(scheduler.shift if sigma_shift is None else sigma_shift),
         "num_train_timesteps": int(scheduler.num_train_timesteps),
         "dtype": dtype, "cfg": 1.0, "update": "native_scheduler.step",
         "snapshot_semantics": "before_update"}, "index_encoding")


def make_manifest(benchmark, config, stats, sampler, content_digests, *, hidden_width=None,
                  benchmark_protocols=None, execution_protocol="system"):
    if benchmark not in {"libero", "metaworld"}:
        raise ValueError("Fast-WAM benchmark must be libero or metaworld")
    libero = benchmark == "libero"
    from .protocols import execution_protocol as get_protocol
    protocol = get_protocol(benchmark, execution_protocol)
    a, s = (7, 8) if libero else (4, 4)
    norm_identity = digest({"stats": stats, "epsilon": 1e-8 if libero else 1e-6,
        "denominator": "zero_only" if libero else "clamp_min", "toggle": [-1] if libero else []})
    layout = ActionStateLayout(32, protocol.execution_steps, a, tuple(range(s)), tuple(range(a-1)), a-1,
        "published toggle sign(-(2*x-1)); physical closed polarity unqualified" if libero else "native_MW_command; physical polarity unqualified",
        ("state",), "LIBERO_OSC_POSE" if libero else "MetaWorld_xyz_gripper",
        "native controller; physical scale unqualified", "native benchmark frame",
        "fastwam.libero.minmax.toggle.v1" if libero else "fastwam.mw.minmax.v1")
    checkpoint_identity = digest(content_digests)
    # Wan VAE downsamples spatially by 16, then native patch embedding by 2.
    # Actual runtime meta is checked before emitting a retrieval key.
    h, w = config["image_size"]
    grid = (1, h // 32, w // 32)
    width = hidden_width or config["video_dit_config"]["hidden_dim"]
    features = FeatureSpec("video_expert.pre_dit.tokens",
        ("image", "image2") if libero else ("image",),
        {"vision_0": width * 16, "robot_state": s}, grid, width,
        "adaptive_avg_4x4.fused_real_views.v1", "reject", False, "float32",
        tuple(range(s)), checkpoint_identity, norm_identity,
        digest({"image_size": config["image_size"], "benchmark": benchmark, "execution_protocol": asdict(protocol),
                "camera_order": "sorted observation.images.*", "config": config}))
    supported = {name: Capability("supported", "CPU stub/native-math certificate; real GPU qualification pending")
        for name in ("full", "k_step", "cp1_retrieval", "concurrent", "bundle", "episode", "debug_capture")}
    for name in ("warmup_library", "recipe", "offline_replay",
                 "conductor", "worker", "analysis", "trace_twins", "native_warm_resume", "native_warm_reset", "cp2"):
        supported[name] = Capability("implemented_unverified", "shared P1/P2/P6 integration or real GPU certificate pending")
    for name in ("batch_stage1", "batch_stage2", "batch_stage3"):
        supported[name] = Capability("unsupported", "production B>1 disabled until GPU batch-composition parity certified; CPU vector hooks only")
    for name in ("split", "shadow_routing", "online_rit", "flow_shoot"):
        supported[name] = Capability("unsupported", "no end-to-end certificate for this combination")
    spec = ModelSpec("fastwam_libero" if libero else "fastwam_mt50", checkpoint_identity,
        norm_identity, layout, features, sampler, Capabilities(supported),
        tuple(benchmark_protocols or (("libero_r17_v1",) if libero else ("metaworld_mt50_r17_v1",))),
        digest({k: v for k, v in content_digests.items() if k.startswith("code:")}), ISLAND)
    return ModelManifest(spec, dict(content_digests))


def manifest_json(manifest):
    return asdict(manifest)
