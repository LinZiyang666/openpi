"""Content identities and strict MW library geometry (no checkpoint tensors loaded)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np

PROVIDER = "r17_groot_mw_v1"
BUILDER = "r17_groot_mw_cp1_spatial16_v1"
CHANNELS = ("x", "y", "z", "gripper")
EMBODIMENT = "metaworld_sawyer"
PROTOCOL = "mt50_seed7_corner2_r17_v1"


def default_protocol():
    # P2 owns the complete benchmark protocol, including task order/camera/splits.
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    return FORMAL_SPEC.identity


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return "sha256:" + result.hexdigest()


def validate_training_metadata(document):
    if EMBODIMENT not in document:
        raise ValueError("checkpoint has no metaworld_sawyer normalization")
    metadata = document[EMBODIMENT]
    if metadata.get("embodiment_tag") != EMBODIMENT:
        raise ValueError("training metadata embodiment_tag differs from metaworld_sawyer")
    modalities = metadata["modalities"]
    for group in ("state", "action"):
        if set(modalities[group]) != set(CHANNELS):
            raise ValueError(f"{group} must have exactly x,y,z,gripper")
        for name in CHANNELS:
            channel = modalities[group][name]
            # Saved DatasetMetadata uses per-key shape, not conversion modality.json start/end.
            if channel.get("shape") != [1] or channel.get("absolute") is not (group == "state") or channel.get("continuous") is not True:
                raise ValueError(f"wrong {group}.{name} saved channel geometry/convention")
            stats = metadata["statistics"][group][name]
            for statistic in ("min", "max", "mean", "std", "q01", "q99"):
                values = np.asarray(stats[statistic])
                if values.shape != (1,) or not np.isfinite(values).all():
                    raise ValueError(f"{group}.{name}/{statistic} must be one finite scalar")
            if stats["min"][0] > stats["max"][0] or stats["q01"][0] > stats["q99"][0] or stats["std"][0] < 0:
                raise ValueError(f"invalid normalization range for {group}.{name}")
        if set(metadata["statistics"][group]) != set(CHANNELS):
            raise ValueError(f"wrong {group} statistics geometry")
    if set(modalities["video"]) != {"image"}:
        raise ValueError("MetaWorld requires exactly video.image; no wrist")
    video = modalities["video"]["image"]
    if video.get("channels") != 3 or video.get("resolution") != [480, 480]:
        raise ValueError("training image must be native RGB480")
    return metadata


@dataclass(frozen=True)
class GrootManifest:
    checkpoint_identity: str
    norm_identity: str
    code_identity: str
    transform_identity: str
    tokenizer_identity: str
    denoising_steps: int = 4
    hidden_width: int = 2048
    benchmark_protocol: str = field(default_factory=default_protocol)
    interface_version: str = "r17.contracts.v1"

    def __post_init__(self):
        if type(self.denoising_steps) is not int or self.denoising_steps not in (1, 2, 3, 4):
            raise ValueError("GR00T-MW endpoint K must be 1-4")
        if type(self.hidden_width) is not int or self.hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        for name in ("checkpoint_identity", "norm_identity", "code_identity", "transform_identity", "tokenizer_identity"):
            value = getattr(self, name)
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{name} must be a content SHA256 identity")

    @property
    def layout(self):
        return {"horizon": 16, "replan_steps": 5, "model_action_width": 32,
                "valid_action_dim": 4, "valid_action_mask": [i < 4 for i in range(32)],
                "valid_state_indices": [0, 1, 2, 3], "motion_channels": [0, 1, 2],
                "gripper_channel": 3, "gripper_closed_convention": "native_metaworld",
                "raw_state_fields": list(CHANNELS), "action_space_name": "metaworld_sawyer_native4",
                "units": "native controller; physical scale unverified", "frame": "native_metaworld",
                "output_transform_id": self.transform_identity}

    @property
    def feature_spec(self):
        return {"cut": "eagle_scatter_before_llm", "real_cameras": ["video.image"],
                "vision_fields": ["vision_0"], "token_grid": [16, 16], "hidden_width": self.hidden_width,
                "pooling": "spatial4x4_float32_v1", "prompt_dependent": False,
                "state_indices": [0, 1, 2, 3], "dtype": "float32", "builder": BUILDER,
                "preprocess_identity": self.transform_identity, "norm_identity": self.norm_identity}

    @property
    def sampler_spec(self):
        k = self.denoising_steps
        return {"schedule_id": f"groot_n15_k{k}_v1", "num_steps": k, "family": "flow_euler",
                "native_points": [i / k for i in range(k)], "deltas": [1 / k] * k,
                "direction": "noise_to_clean_asc", "endpoint_fixed_k": True}

    @property
    def capabilities(self):
        core = ("full", "k_step", "cp1", "concurrent", "bundle", "episode", "debug_capture", "library", "native_resume", "trace", "batch_stage3", "cp2_key_capture", "warm_reset_hook")
        result = {name: {"status": "implemented_unverified", "reason": "CPU conformance; GPU qualification pending"} for name in core}
        for name in ("batch_stage1", "batch_stage2", "split", "cp2", "online_rit"):
            result[name] = {"status": "unsupported", "reason": "no MW qualification for this hook"}
        result["dual_camera_ablation"] = {"status": "not_applicable", "reason": "one real camera"}
        return result

    @property
    def identities(self):
        spec = self.model_spec()
        return {"model_id": spec.model_id, "checkpoint_identity": self.checkpoint_identity,
                "norm_identity": self.norm_identity, "layout_identity": spec.layout.identity,
                "feature_identity": spec.feature_spec.identity, "sampler_identity": spec.default_sampler.identity,
                "benchmark_protocol": self.benchmark_protocol}

    def model_spec(self):
        from exp.r17_integration.common.contracts import ActionStateLayout, Capabilities, Capability, FeatureSpec, ModelSpec, SamplerSpec
        layout = ActionStateLayout(16, 5, 4, (0, 1, 2, 3), (0, 1, 2), 3,
                                  gripper_closed_convention="native_metaworld", raw_state_fields=CHANNELS,
                                  action_space_name="metaworld_sawyer_native4", units="native controller; physical scale unverified",
                                  frame="native_metaworld", output_transform_id=self.transform_identity, model_action_width=32)
        feature = FeatureSpec("eagle_scatter_before_llm", ("video.image",),
                              {"vision_0": 16 * self.hidden_width, "robot_state": 4},
                              token_grid=(16, 16), hidden_width=self.hidden_width, pooling="spatial4x4_float32_v1",
                              active_state_indices=(0, 1, 2, 3), policy_digest=self.checkpoint_identity,
                              norm_digest=self.norm_identity, preprocess_digest=self.transform_identity)
        k = self.denoising_steps
        sampler = SamplerSpec(f"groot_n15_k{k}_v1", "flow_euler", k, tuple(i / k for i in range(k)),
                              (1 / k,) * k, {"direction": "noise_to_clean_asc", "endpoint_fixed_k": True}, "flow_time")
        caps = {name: Capability("supported", "Generic serving path available; CPU evidence only, GPU qualification pending")
                for name in ("full", "k_step", "cp1_retrieval", "concurrent", "bundle", "episode", "debug_capture")}
        hook_limits = {
            "library": "Full-action CP1 subset only; complete library fit/replay/augmentation qualification pending",
            "native_resume": "Direct adapter API only; generic warm-tier dispatcher rejects configuration",
            "trace": "Direct stock trace hook only; generic trace executor rejects configuration",
            "batch_stage3": "Direct stock S3 hook only; generic server does not submit through BatchingCore",
            "cp2_key_capture": "Direct key hook only; generic CP2 execution rejects configuration",
            "warm_reset_hook": "Direct stock hook only; generic warm-reset execution rejects configuration",
        }
        caps.update({name: Capability("implemented_unverified", reason) for name, reason in hook_limits.items()})
        caps.update({name: Capability("unsupported", item["reason"]) for name, item in self.capabilities.items()
                     if item["status"] == "unsupported"})
        caps["dual_camera_ablation"] = Capability("not_applicable", "one real camera")
        return ModelSpec("groot_n15_metaworld_sawyer", self.checkpoint_identity, self.norm_identity, layout, feature,
                         sampler, Capabilities(caps), (self.benchmark_protocol,), self.code_identity, "GR00T vendor island")

    def to_dict(self):
        return {**asdict(self), "provider": PROVIDER, "embodiment": EMBODIMENT, "embodiment_id": 31,
                "layout": self.layout, "feature_spec": self.feature_spec, "sampler_spec": self.sampler_spec,
                "capabilities": self.capabilities, **self.identities,
                "r17_model_spec": asdict(self.model_spec()),
                "r17_samplers": {str(self.denoising_steps): asdict(self.model_spec().default_sampler)}}

    def endpoint_spec(self, host, port, *, capacity=1):
        from exp.r17_integration.conductor.manifest import EndpointSpec
        from openpi.conductor.task import ServerEndpoint
        return EndpointSpec(ServerEndpoint(host, port), self.checkpoint_identity, self.norm_identity,
                            self.model_spec().layout.identity, self.benchmark_protocol,
                            allowed_k=(self.denoising_steps,), capacity=capacity)


def validate_library_model(model, manifest):
    from exp.r17_integration.common.recipe import retrieval_model_identity
    if retrieval_model_identity(model) != retrieval_model_identity(manifest.model_spec()):
        raise ValueError("library immutable model/norm/layout/feature/code differs from GR00T-MW")


def validate_teacher_sampler(meta, manifest):
    """Validate teacher provenance independently; native resume remains grid-exact."""
    from exp.r17_integration.common.contracts import digest as contract_digest
    schedule = meta.get("denoise_schedule")
    schedules = {f"groot_n15_k{k}_v1": k for k in (4, 2, 1)}
    if schedule not in schedules:
        raise ValueError("library denoise_schedule is not a declared GR00T teacher schedule")
    teacher = replace(manifest, denoising_steps=schedules[schedule]).model_spec().default_sampler
    if meta.get("sampler_identity") != teacher.identity:
        raise ValueError("library sampler_identity differs from declared teacher sampler")
    if meta.get("library_sampler_identity", teacher.identity) != teacher.identity:
        raise ValueError("library_sampler_identity differs from teacher provenance")
    if "library_sampler" in meta and contract_digest(meta["library_sampler"]) != teacher.identity:
        raise ValueError("library sampler native grid/config differs from declared teacher")
    return teacher


def validate_library(meta, manifest, *, vector_dims=None):
    for key, expected in manifest.identities.items():
        if key == "sampler_identity":
            continue  # teacher sampler is provenance; MISS sampler belongs to the endpoint
        if meta.get(key) != expected:
            raise ValueError(f"library {key} differs from served GR00T-MW endpoint")
    if meta.get("key_builder_type") != BUILDER:
        raise ValueError("library key builder differs")
    expected = {"vision_0": 16 * manifest.hidden_width, "robot_state": 4}
    if (vector_dims if vector_dims is not None else meta.get("vector_dims")) != expected:
        raise ValueError("library must contain one real vision field and state4")
    if tuple(meta.get("action_shape", ())) != (16, 32) or meta.get("active_action_indices") != [0, 1, 2, 3]:
        raise ValueError("library must preserve H16×32 storage with first4 active")
    validate_teacher_sampler(meta, manifest)


def library_metadata(manifest):
    from exp.r17_integration.common.recipe import retrieval_model_identity
    return {**manifest.identities, "interface_version": "r17.contracts.v1", "key_builder_type": BUILDER,
            "r17_binding": {**manifest.identities, "retrieval_model_identity": retrieval_model_identity(manifest.model_spec())},
            "library_sampler_identity": manifest.model_spec().default_sampler.identity,
            "library_sampler": asdict(manifest.model_spec().default_sampler),
            "vector_dims": {"vision_0": 16 * manifest.hidden_width, "robot_state": 4},
            "action_shape": [16, 32], "active_action_indices": [0, 1, 2, 3],
            "real_cameras": ["video.image"], "denoise_schedule": manifest.sampler_spec["schedule_id"]}


def checkpoint_manifest(checkpoint, *, denoising_steps=4, tokenizer_files=(), code_files=()):
    """Hash immutable inference assets. Caller must supply tokenizer/vendored source inputs."""
    checkpoint = Path(checkpoint)
    metadata_path = checkpoint / "experiment_cfg/metadata.json"
    metadata = validate_training_metadata(json.loads(metadata_path.read_text()))
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    head = config.get("action_head_cfg", config.get("action_head_config", {}))
    if head.get("action_horizon") != 16 or head.get("action_dim") != 32:
        raise ValueError("checkpoint head must preserve H16/action32")
    if not tokenizer_files or not code_files:
        raise ValueError("tokenizer_files and code_files content inputs are required")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        weights = ["model.safetensors"]
    if not weights:
        raise ValueError("checkpoint weight index is empty")
    for name in weights:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard must be inside checkpoint")
    weight_hashes = {name: file_digest(checkpoint / name) for name in weights}
    code_hashes = sorted((Path(p).name, file_digest(p)) for p in code_files)
    tokenizer_hashes = sorted((Path(p).name, file_digest(p)) for p in tokenizer_files)
    code_id = digest(code_hashes)
    tokenizer_id = digest(tokenizer_hashes)
    norm_id = digest(metadata)
    transform_id = digest({"code": code_id, "norm": norm_id, "tokenizer": tokenizer_id, "embodiment": EMBODIMENT})
    identity = digest({"weights": weight_hashes, "config": file_digest(config_path), "metadata": file_digest(metadata_path),
                       "code": code_id, "tokenizer": tokenizer_id, "transform": transform_id})
    return GrootManifest(identity, norm_id, code_id, transform_id, tokenizer_id, denoising_steps)
