"""CPU-sealed official assets, native EDM schedule and explicit capability limits."""
from __future__ import annotations

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, FeatureSpec, ModelManifest, ModelSpec, SamplerSpec, digest,
)
from .download import sha256_file


def make_sampler(k=5):
    # K2 counts the clean step in the requested budget; the upstream one-step
    # branch nevertheless performs just one sigma_max evaluation (NFE1).
    if type(k) is not int or k not in (1, 2, 3, 5):
        raise ValueError("Cosmos supports native K1/K2/K3/K5")
    import torch
    n = k - 1 if k > 1 else 1
    sigmas = (80 ** (1 / 7) + torch.arange(n + 1, dtype=torch.float64) / n * (4 ** (1 / 7) - 80 ** (1 / 7))) ** 7
    grid = tuple(sigmas.tolist()) if k > 1 else (float(sigmas[0]),)
    config = {"sigma_min": 4.0, "sigma_max": 80.0, "rho": 7, "solver": "2ab",
                "sample_clean": True, "churn": 0.0, "cfg": False, "variance_scale": False,
                "net_dtype": "bfloat16", "solver_dtype": "float64", "initial_noise_dtype": "float32",
                "initial_noise": "native arch_invariant_rand; current request seed", "num_samples": 1}
    if k == 2:
        config.update(expected_nfe=1, arm_label="K2（含清理步）", evaluated_sigma_indices=[0],
            upstream_branch="num_steps=2 minus clean budget -> one-step sigma_max; no sigma_min evaluation")
    return SamplerSpec("cosmos_edm_2ab_k2_incl_clean" if k == 2 else "cosmos_policy_edm_2ab_v1",
        "cosmos_edm", k, grid, config=config, point_kind="sigma")


def make_manifest(stats, digests, protocols):
    norm = digest({"stats": stats, "proprio": "minmax; no clipping; g0,g1,xyz,quat_xyzw",
                   "actions": "native minmax inverse; no clipping/sign/scaling"})
    checkpoint = digest(digests)
    layout = ActionStateLayout(16, 16, 7, tuple(range(9)), tuple(range(6)), 6,
        "native LIBERO gripper command; unchanged from official runner",
        ("gripper_qpos[2]", "eef_pos[3]", "eef_quat_xyzw[4]"), "LIBERO_OSC_POSE",
        "official native controller", "LIBERO base frame", "cosmos.libero.official_minmax.v1")
    feature = FeatureSpec("Cosmos tokenizer.encode current camera latent frames",
        ("agentview", "robot0_eye_in_hand"), {"vision_0": 256, "vision_1": 256, "robot_state": 9},
        (9, 28, 28), 16, "per_camera_latent_channels.adaptive_avg_4x4.v1", "reject",
        False, "float32", tuple(range(9)), checkpoint, norm,
        digest({"jpeg": 95, "image_aug": "official center crop 90% area", "resolution": 224,
                "wire_resolution": 256, "wire_rot180_to_native_flipud": "undo horizontal flip", "latent_order": "wrist,agentview",
                "state_t": 9, "conditioning_frames": 4}))
    supported = {n: Capability("supported", "native implementation; receipts in HANDBACK_COSMOS.md") for n in
        ("full", "k_step", "concurrent", "bundle", "episode", "conductor", "worker", "analysis")}
    supported['cp1_retrieval'] = Capability('implemented_unverified',
        'native VAE keys implemented; Cosmos CP1 HIT/recipe quality has no GPU certificate')
    supported['recipe'] = Capability('implemented_unverified',
        'sealed B500 cache-study K5/C16 admission and key-only HIT CPU verified; GPU hot-switch smoke pending')
    supported['debug_capture'] = Capability('supported',
        'minimal P6 teacher allowlist only; native K5/seed195; snapshots=False; no model trace products')
    for n in ("warmup_library", "offline_replay", "native_warm_resume",
              "native_warm_reset", "cp2", "trace_twins", "batch_stage1", "batch_stage2", "batch_stage3",
              "split", "shadow_routing", "online_rit", "flow_shoot"):
        supported[n] = Capability("unsupported", "no Cosmos-specific end-to-end certificate; rejected explicitly")
    model = ModelSpec("cosmos_libero", checkpoint, norm, layout, feature, make_sampler(), Capabilities(supported),
        tuple(protocols), digest({k: v for k, v in digests.items() if k.startswith("code:")}),
        "cosmos-policy/python3.10/torch2.7-cu128")
    manifest = ModelManifest(model, dict(digests))
    from .library_admission import remember_manifest
    remember_manifest(manifest)
    return manifest
