"""Thin π0.5 adapter: benchmark-owned transforms and original native stages.

Imports are model-island-free. Explicit construction/prepare loads torch; only
load_adapter() loads weights.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import numpy as np

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, CaptureSpec, FeatureSpec, ModelSpec,
    NativeSnapshot, PreparedObs, SampleResult, SamplerSpec, Stage1, Stage2, StateView, WireChunk, digest,
)
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC

KEY_BUILDER_ID = "pi05_metaworld_cp1_spatial16_v1"


def sampler_for_steps(k):
    if type(k) is not int or k not in (1, 2, 3, 4, 5, 10):
        raise ValueError("π-MW grids are K1-K5/K10")
    dt, t = np.float32(-1/k), np.float32(1)
    times = []
    for _ in range(k):
        times.append(float(t))
        t = np.float32(t + dt)
    return SamplerSpec("pi05_v1_k" + str(k), "flow", k, tuple(times), (float(dt),)*k,
                       {"update": "native PI0Pytorch float32 accumulated Euler", "direction": "descending"}, "native_time")


def model_spec(checkpoint_identity, norm_identity, *, replan_steps=5, code_digest="",
               benchmark="metaworld", benchmark_spec=None):
    layout = ActionStateLayout(5, replan_steps, 4, (0, 1, 2, 3), (0, 1, 2), 3,
                               gripper_closed_convention="positive native Sawyer close command",
                               raw_state_fields=("hand_x", "hand_y", "hand_z", "gripper_aperture"),
                               action_space_name="MetaWorld native Sawyer xyz delta + gripper",
                               units="native command; controller scale not calibrated here", frame="world",
                               output_transform_id="pi05_metaworld.quantile_inverse.v1", model_action_width=32)
    features = FeatureSpec("PI0Pytorch.run_stage1.prefix_embs", ("corner2",),
                           {"vision_0": 32768, "robot_state": 4}, (16, 16), 2048,
                           "spatial16.adaptive_avg_pool2d.v1", "two masked zero wrists retain native compute", False,
                           active_state_indices=(0, 1, 2, 3), policy_digest=checkpoint_identity,
                           norm_digest=norm_identity, preprocess_digest=code_digest)
    supported = ("full", "k_step", "cp1_retrieval", "concurrent", "bundle", "episode", "debug_capture")
    caps = {k: Capability("supported", "CPU hook conformance; real checkpoint GPU qualification pending") for k in supported}
    for k in ("warmup", "library", "recipe", "offline", "replay", "conductor", "worker", "analysis"):
        caps[k] = Capability("implemented_unverified", "framework/benchmark tools require cross-package qualification")
    for k in ("batch_stage1", "batch_stage2", "batch_stage3", "split", "cp2", "warm_reset", "trace_twins", "routing"):
        caps[k] = Capability("unsupported", "generic R17 executor not yet wired/qualified for this feature")
    caps["warm_resume"] = Capability("implemented_unverified", "native suffix hook; GPU parity pending")
    domain = benchmark
    if domain == "libero":
        if benchmark_spec is None:
            from exp.r17_integration.ops.formal_eval import benchmark as libero_spec
            benchmark_spec = libero_spec()
        layout = ActionStateLayout(10, replan_steps, 7, tuple(range(8)), tuple(range(6)), 6,
            gripper_closed_convention="positive LIBERO close command",
            raw_state_fields=("eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper_qpos_0", "gripper_qpos_1"),
            action_space_name="LIBERO xyz/axis-angle delta + gripper", units="native controller", frame="world",
            output_transform_id="pi05_libero.quantile_inverse.v1", model_action_width=32)
        features = replace(features, real_cameras=("agentview", "robot0_eye_in_hand"),
            fields={"vision_0": 32768, "vision_1": 32768, "robot_state": 8},
            active_state_indices=tuple(range(8)), missing_branch_policy="one masked zero wrist retains native compute")
        protocol = benchmark_spec
    else:
        protocol = FORMAL_SPEC if replan_steps == 5 else replace(FORMAL_SPEC, replan_steps=replan_steps,
                           protocol_id=f"mt50_seed7_corner2_r17_replan{replan_steps}_v1")
    from exp.r17_integration.common.benchmarks import register_benchmark
    register_benchmark(protocol)
    return ModelSpec("pi05_libero" if domain == "libero" else "pi05_rlinf_mt50", checkpoint_identity, norm_identity, layout, features,
                     sampler_for_steps(10), Capabilities(caps), (protocol.identity,), code_digest, "openpi PyTorch")


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.array(value, copy=True)


def _tensor_tree(tree, device):
    import torch
    if isinstance(tree, dict):
        return {k: _tensor_tree(v, device) for k, v in tree.items()}
    return torch.from_numpy(np.array(tree)).to(device)[None, ...]


class Pi05Adapter:
    def __init__(self, policy, spec, *, observation_factory=None):
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("π-MW staged adapter requires the native PyTorch policy")
        self.policy, self.model, self.spec = policy, policy._model, spec
        if (self.model.config.action_horizon, self.model.config.action_dim) != (spec.layout.horizon, 32):
            raise ValueError("π checkpoint geometry differs from benchmark layout")
        self.observation_factory = observation_factory
        self.stage_executor = None

    def metadata(self):
        return self.spec

    def drain_queue_wait_ms(self):
        """This connection-local adapter has no shared model-lock acquisition.

        Zero attests only adapter lock wait. CUDA scheduling/contention remains
        part of host service wall time, not a measured GPU execution time.
        """
        return 0.0

    def prepare(self, wire_obs, request):
        clean = {k: v for k, v in wire_obs.items() if k not in {"__debug__", "__request__", "__execution__"}}
        state = np.asarray(clean["observation/state"])
        state_dim = len(self.spec.layout.valid_state_indices)
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"expected finite state{state_dim}")
        image_size = 256 if state_dim == 8 else 480
        image_keys = ("observation/image", "observation/wrist_image") if state_dim == 8 else ("observation/image",)
        for name in image_keys:
            image = np.asarray(clean[name])
            if image.shape != (image_size, image_size, 3) or image.dtype != np.uint8:
                raise ValueError(f"expected uint8 RGB{image_size}; model transform performs resize")
        if not isinstance(clean.get("prompt"), str) or not clean["prompt"]:
            raise ValueError("canonical prompt required")
        # Copy all arrays; stock transforms may modify containers.
        clean = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in clean.items()}
        inputs = self.policy._input_transform(clean)
        batched = _tensor_tree(inputs, self.policy._pytorch_device)
        factory = self.observation_factory
        if factory is None:
            from openpi.models.model import Observation
            factory = Observation.from_dict
        return PreparedObs(factory(batched), request, inputs["state"])

    def run_stage1(self, prepared):
        import torch
        with torch.no_grad():
            output = self.model.run_stage1(prepared.data)
        return self.finish_stage1(output, prepared)

    def finish_stage1(self, output, prepared):
        return Stage1(output, fields={"model_input_base_rgb": prepared.data.images["base_0_rgb"]}, state=output.state)

    def bind_stage_executor(self, executor):
        self.stage_executor = executor

    def batching_group(self):
        return id(self.model)

    def stage1_batch_input(self, prepared):
        import jax
        # Pi05StageBatcher stacks unbatched observations, whereas prepare is B1.
        return jax.tree.map(lambda x: x[0], prepared.data)

    def stage3_batch_input(self, s2):
        return s2.data

    def retrieval_fields(self, s1):
        import torch
        import torch.nn.functional as F
        prefix = s1.data.prefix_embs
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[1] < 768 or prefix.shape[2] != 2048:
            raise ValueError("native π prefix must retain all three slots and language tokens")
        fields = {}
        for i in range(len(self.spec.feature_spec.real_cameras)):
            # Match cp1_spatial_pool_16: pool in the native prefix dtype,
            # then transfer as float32. GR00T has its own pre-pool cast.
            tokens = prefix[0, i*256:(i+1)*256]
            pool = F.adaptive_avg_pool2d(tokens.reshape(16, 16, 2048).permute(2, 0, 1)[None], (4, 4))
            fields[f"vision_{i}"] = pool[0].permute(1, 2, 0).reshape(-1).detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        state = s1.data.state[0, list(self.spec.layout.valid_state_indices)].detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if any(not torch.isfinite(key).all() for key in fields.values()) or not torch.isfinite(state).all():
            raise ValueError("nonfinite retrieval fields")
        return {**fields, "robot_state": state}

    def run_stage2(self, s1, *, capture=False):
        import torch
        with torch.no_grad():
            output = self.model.run_stage2_capture(s1.data) if capture else self.model.run_stage2(s1.data)
        return Stage2(output, s1)

    def sample(self, s2, sampler, rng, capture=CaptureSpec()):
        import torch
        if sampler != sampler_for_steps(sampler.num_steps):
            raise ValueError("foreign π sampler grid/config")
        noise = torch.from_numpy(rng.normal((self.spec.layout.horizon, 32))).to(s2.data.stage1.state.device)[None]
        points = sampler.native_timesteps[1:] if capture.snapshots else ()
        with torch.no_grad():
            if self.stage_executor is not None:
                output = self.stage_executor.miss(s2, noise[0], sampler, points)
            else:
                output = self.model.run_stage3(s2.data, noise=noise, num_steps=sampler.num_steps,
                                              return_intermediates=capture.snapshots, save_timesteps=points)
        normalized = _numpy(output.action_chunk[0])
        self._validate_chunk(normalized)
        snapshots = []
        for point, x in (getattr(output, "intermediates", None) or {}).items():
            index = min(range(sampler.num_steps), key=lambda i: abs(sampler.native_timesteps[i] - point))
            snapshots.append(NativeSnapshot(self.spec.checkpoint_identity, self.spec.norm_identity,
                                            self.spec.layout.identity, sampler.identity, sampler.schedule_digest,
                                            index, sampler.native_timesteps[index], _numpy(x[0])))
        return SampleResult(normalized, sampler.num_steps, sampler, tuple(snapshots),
                            {"normalized_chunk": normalized.copy(), "initial_noise": _numpy(noise[0])} if capture.enabled else {}, rng.receipt())

    def capture_fields(self, s1, s2, result):
        fields = {"prefix_embeddings": _numpy(s1.data.prefix_embs[0]),
                  "prefix_mask": _numpy(s1.data.prefix_pad_masks[0]),
                  "normalized_state": _numpy(s1.data.state[0]),
                  "model_input_base_rgb": _numpy(s1.fields["model_input_base_rgb"][0])}
        if getattr(s2.data, "prefix_out", None) is not None:
            fields["prefix_out"] = _numpy(s2.data.prefix_out[0])
        if result.capture:
            fields.update({k: _numpy(v) for k, v in result.capture.items()})
        return fields

    def resume(self, s2, snapshot, sampler, rng, capture=CaptureSpec()):
        import torch
        if sampler != sampler_for_steps(sampler.num_steps):
            raise ValueError("foreign π sampler")
        snapshot.validate(self.spec, sampler)
        x = np.asarray(snapshot.x_t)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("snapshot requires finite float32 normalized x_t")
        with torch.no_grad():
            start_x = torch.from_numpy(x.copy()).to(s2.data.stage1.state.device)
            if self.stage_executor is not None:
                out = self.stage_executor.warm(s2, start_x, snapshot, sampler)
            else:
                out = self.model.run_stage3_from(s2.data, start_x[None], snapshot.native_timestep,
                                               num_steps=sampler.num_steps)
        normalized = _numpy(out.action_chunk[0])
        self._validate_chunk(normalized)
        return SampleResult(normalized, sampler.num_steps - snapshot.loop_index, sampler, rng_receipt=rng.receipt())

    def _validate_chunk(self, x):
        if x.shape != (self.spec.layout.horizon, 32) or not np.isfinite(x).all():
            raise ValueError("expected finite normalized native horizon/action32")

    def to_wire(self, normalized_chunk, prepared):
        x = _numpy(normalized_chunk)
        self._validate_chunk(x)
        output = self.policy._output_transform({"state": np.array(prepared.state, copy=True), "actions": x})
        actions = np.array(output["actions"], copy=True)
        if actions.shape != (self.spec.layout.horizon, self.spec.layout.valid_action_dim) or not np.isfinite(actions).all():
            raise ValueError("native inverse returned invalid wire actions")
        return WireChunk(actions)

    def extract_state_only(self, wire_obs):
        state = np.asarray(wire_obs["observation/state"], dtype=np.float32)
        if state.shape != (len(self.spec.layout.valid_state_indices),) or not np.isfinite(state).all():
            raise ValueError("expected finite native state")
        # Invoke only the original Normalize transform, on state alone. Vision
        # resize, tokenizer and model-input padding never run on a free tail.
        transforms = getattr(self.policy._input_transform, "transforms", ())
        normalizers = [t for t in transforms if type(t).__name__ == "Normalize"]
        if len(normalizers) != 1:
            raise ValueError("native policy requires one explicit state normalizer")
        normalizer = normalizers[0]
        stats = (normalizer.norm_stats or {}).get("state")
        if stats is not None:
            fn = normalizer._normalize_quantile if normalizer.use_quantiles else normalizer._normalize
            state = fn(state, stats)
        return StateView(np.array(state, dtype=np.float32, copy=True))

    def make_stage_batcher(self):
        from openpi.serving.batching_coordinator import Pi05StageBatcher
        return Pi05StageBatcher(self.model, self.policy._pytorch_device)

    def make_trace_adapter(self):
        return Pi05TraceAdapter()

    def cpu_blueprint(self):
        raise NotImplementedError("split qualification pending")


# Historical import retained for MT50 callers; both benchmarks use one adapter.
Pi05MetaWorldAdapter = Pi05Adapter


class Pi05TraceAdapter:
    def capture(self, decision):
        # Consume executed results only; no model/RNG invocation.
        return {"schema": decision.schema, "source": decision.source, "cursor": decision.cursor,
                "executed_count": decision.executed_count, "actual_nfe": decision.actual_nfe,
                "stage_calls": decision.stage_calls, "normalized_chunk": _numpy(decision.normalized_chunk),
                "wire_actions": _numpy(decision.wire_actions),
                "fields": {k: _numpy(v) for k, v in decision.fields.items()},
                "sampler_receipt": dict(decision.sampler_receipt)}


def _tokenizer_content():
    # Mirror the original downloader's local cache path without importing it,
    # creating directories, changing permissions, or fetching anything.
    cache = Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    path = cache / "big_vision/paligemma_tokenizer.model"
    return {"source": "gs://big_vision/paligemma_tokenizer.model", "cache_path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "status": "sealed" if path.is_file() else "missing_local_content"}


def _verify_loaded_tokenizer(policy, expected_digest):
    tokenizers = [getattr(t, "tokenizer", None) for t in policy._input_transform.transforms]
    processors = [t._tokenizer for t in tokenizers if t is not None and hasattr(t, "_tokenizer")]
    if len(processors) != 1:
        raise ValueError("expected exactly one native PaliGemma tokenizer")
    actual = hashlib.sha256(processors[0].serialized_model_proto()).hexdigest()
    if actual != expected_digest:
        raise ValueError("loaded tokenizer content differs from sealed manifest")


def norm_path(checkpoint, benchmark="metaworld"):
    if benchmark == "libero":
        # The historical converted checkpoint contains weights/config only.
        # Stock policy_config resolves this config-owned LIBERO asset.
        return Path(__file__).resolve().parents[4] / "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
    return Path(checkpoint) / "assets/metaworld_mt50/norm_stats.json"


def content_manifest(checkpoint, *, benchmark="metaworld", benchmark_spec=None):
    root = Path(checkpoint).resolve()
    weights = root / "model.safetensors"
    norm = norm_path(root, benchmark)
    if not weights.is_file() or not norm.is_file():
        raise ValueError("local PyTorch weights and benchmark norm stats required")
    contents = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() and not p.resolve().is_relative_to(root):
            raise ValueError("checkpoint symlink escapes immutable root")
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                while block := f.read(4*1024*1024):
                    h.update(block)
            contents[p.relative_to(root).as_posix()] = h.hexdigest()
    # Bind code used by both the adapter and stock loader/transforms/stages.
    repo = Path(__file__).resolve().parents[4]
    code = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), repo / f"src/openpi/policies/{'libero' if benchmark == 'libero' else 'metaworld'}_policy.py",
             repo / "src/openpi/models_pytorch/pi0_pytorch.py", repo / "src/openpi/policies/policy.py",
             repo / "src/openpi/policies/policy_config.py", repo / "src/openpi/training/config.py",
             repo / "src/openpi/transforms.py", repo / "src/openpi/models/tokenizer.py")}
    tokenizer = _tokenizer_content()
    norm_key = "assets/pi05_libero/physical-intelligence/libero/norm_stats.json" if benchmark == "libero" else "assets/metaworld_mt50/norm_stats.json"
    contents[norm_key] = hashlib.sha256(norm.read_bytes()).hexdigest()
    checkpoint_identity = digest({"config": "pi05_libero" if benchmark == "libero" else "pi05_metaworld", "files": contents, "tokenizer": tokenizer["sha256"]})
    code_identity = digest(code)
    return {"schema": "r17.model_manifest.v1", "checkpoint_identity": checkpoint_identity,
            "norm_identity": contents[norm_key], "content_digests": contents, "norm_path": str(norm),
            "tokenizer": tokenizer, "code_digest": code_identity, "source_digests": code, "checkpoint_root": str(root),
            "spec": asdict(model_spec(checkpoint_identity, contents[norm_key],
                                      code_digest=code_identity, benchmark=benchmark, benchmark_spec=benchmark_spec)),
            "available_samplers": [asdict(sampler_for_steps(k)) for k in (10, 2, 1)],
            "components": {"weights": "model.safetensors", "base": "embedded in complete PyTorch checkpoint",
                           "delta": None, "norm": str(norm) if benchmark == "libero" else norm_key},
            "qualification": {"model_loaded": False, "gpu": "unexecuted"}}


def load_adapter(spec):
    """Explicit load only. No norm fallback, no checkpoint URI side effects."""
    checkpoint = spec["checkpoint"]
    benchmark = spec.get("benchmark", "metaworld")
    benchmark_spec = spec.get("benchmark_spec")
    if benchmark == "libero" and isinstance(benchmark_spec, dict):
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
        from dataclasses import fields
        names = {f.name for f in fields(LiberoBenchmarkSpec)}
        benchmark_spec = LiberoBenchmarkSpec(**{k:v for k,v in benchmark_spec.items() if k in names})
    manifest = content_manifest(checkpoint, benchmark=benchmark, benchmark_spec=benchmark_spec)
    if manifest["tokenizer"]["sha256"] is None:
        raise ValueError("local PaliGemma tokenizer content must be sealed before model load; set OPENPI_DATA_HOME")
    expected = spec.get("checkpoint_identity")
    if expected is not None and expected != manifest["checkpoint_identity"]:
        raise ValueError("immutable checkpoint identity mismatch")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    options = {}
    if benchmark == "libero":
        from openpi.shared.normalize import load
        options["norm_stats"] = load(norm_path(checkpoint, benchmark).parent)
    policy = create_trained_policy(get_config("pi05_libero" if benchmark == "libero" else "pi05_metaworld"), checkpoint,
                                   sample_kwargs={"num_steps": 10}, pytorch_device=spec.get("device", "cpu"), **options)
    _verify_loaded_tokenizer(policy, manifest["tokenizer"]["sha256"])
    return Pi05MetaWorldAdapter(policy, model_spec(manifest["checkpoint_identity"], manifest["norm_identity"],
                                                 code_digest=manifest["code_digest"], benchmark=benchmark, benchmark_spec=benchmark_spec))
