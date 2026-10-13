"""Thin π0.5-MT50 adapter: original transforms and original native stages.

Imports are model-island-free. Explicit construction/prepare loads torch; only
load_adapter() loads weights. No LIBERO transform or surrogate camera key.
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


def model_spec(checkpoint_identity, norm_identity, *, replan_steps=5, code_digest=""):
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
    benchmark = FORMAL_SPEC if replan_steps == 5 else replace(FORMAL_SPEC, replan_steps=replan_steps,
                       protocol_id=f"mt50_seed7_corner2_r17_replan{replan_steps}_v1")
    from exp.r17_integration.common.benchmarks import register_benchmark
    register_benchmark(benchmark)
    return ModelSpec("pi05_rlinf_mt50", checkpoint_identity, norm_identity, layout, features,
                     sampler_for_steps(10), Capabilities(caps), (benchmark.identity,), code_digest, "openpi PyTorch")


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


class Pi05MetaWorldAdapter:
    def __init__(self, policy, spec, *, observation_factory=None):
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("π-MW staged adapter requires the native PyTorch policy")
        self.policy, self.model, self.spec = policy, policy._model, spec
        if (self.model.config.action_horizon, self.model.config.action_dim) != (5, 32):
            raise ValueError("π-MW checkpoint geometry must be H5/action32")
        self.observation_factory = observation_factory

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
        image = np.asarray(clean["observation/image"])
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("expected finite MetaWorld state4")
        if image.shape != (480, 480, 3) or image.dtype != np.uint8:
            raise ValueError("expected uint8 corner2 RGB480; model transform performs resize")
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
        return Stage1(output, fields={"model_input_base_rgb": prepared.data.images["base_0_rgb"]}, state=output.state)

    def retrieval_fields(self, s1):
        import torch
        import torch.nn.functional as F
        prefix = s1.data.prefix_embs
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[1] < 768 or prefix.shape[2] != 2048:
            raise ValueError("native π prefix must retain all three slots and language tokens")
        tokens = prefix[0, :256]
        pool = F.adaptive_avg_pool2d(tokens.reshape(16, 16, 2048).permute(2, 0, 1)[None], (4, 4))
        key = pool[0].permute(1, 2, 0).reshape(-1).detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        state = s1.data.state[0, :4].detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if not torch.isfinite(key).all() or not torch.isfinite(state).all():
            raise ValueError("nonfinite retrieval fields")
        return {"vision_0": key, "robot_state": state}

    def run_stage2(self, s1, *, capture=False):
        import torch
        with torch.no_grad():
            output = self.model.run_stage2_capture(s1.data) if capture else self.model.run_stage2(s1.data)
        return Stage2(output, s1)

    def sample(self, s2, sampler, rng, capture=CaptureSpec()):
        import torch
        if sampler != sampler_for_steps(sampler.num_steps):
            raise ValueError("foreign π sampler grid/config")
        noise = torch.from_numpy(rng.normal((5, 32))).to(s2.data.stage1.state.device)[None]
        points = sampler.native_timesteps[1:] if capture.snapshots else ()
        with torch.no_grad():
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
            out = self.model.run_stage3_from(s2.data, torch.from_numpy(x.copy()).to(s2.data.stage1.state.device)[None],
                                             snapshot.native_timestep, num_steps=sampler.num_steps)
        normalized = _numpy(out.action_chunk[0])
        self._validate_chunk(normalized)
        return SampleResult(normalized, sampler.num_steps - snapshot.loop_index, sampler, rng_receipt=rng.receipt())

    def _validate_chunk(self, x):
        if x.shape != (5, 32) or not np.isfinite(x).all():
            raise ValueError("expected finite normalized H5/action32")

    def to_wire(self, normalized_chunk, prepared):
        x = _numpy(normalized_chunk)
        self._validate_chunk(x)
        output = self.policy._output_transform({"state": np.array(prepared.state, copy=True), "actions": x})
        actions = np.array(output["actions"], copy=True)
        if actions.shape != (5, 4) or not np.isfinite(actions).all():
            raise ValueError("native MetaWorld inverse must return finite H5/action4")
        return WireChunk(actions)

    def extract_state_only(self, wire_obs):
        state = np.asarray(wire_obs["observation/state"], dtype=np.float32)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("expected finite state4")
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
        return StateView(np.array(state, copy=True))

    def make_stage_batcher(self):
        from openpi.serving.batching_coordinator import Pi05StageBatcher
        return Pi05StageBatcher(self.model, self.policy._pytorch_device)

    def make_trace_adapter(self):
        return Pi05TraceAdapter()

    def cpu_blueprint(self):
        raise NotImplementedError("split qualification pending")


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


def content_manifest(checkpoint):
    root = Path(checkpoint).resolve()
    weights = root / "model.safetensors"
    norm = root / "assets/metaworld_mt50/norm_stats.json"
    if not weights.is_file() or not norm.is_file():
        raise ValueError("immutable local PyTorch weights and checkpoint-local MW norm stats required")
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
            (Path(__file__), repo / "src/openpi/policies/metaworld_policy.py",
             repo / "src/openpi/models_pytorch/pi0_pytorch.py", repo / "src/openpi/policies/policy.py",
             repo / "src/openpi/policies/policy_config.py", repo / "src/openpi/training/config.py",
             repo / "src/openpi/transforms.py", repo / "src/openpi/models/tokenizer.py")}
    tokenizer = _tokenizer_content()
    checkpoint_identity = digest({"config": "pi05_metaworld", "files": contents, "tokenizer": tokenizer["sha256"]})
    code_identity = digest(code)
    return {"schema": "r17.model_manifest.v1", "checkpoint_identity": checkpoint_identity,
            "norm_identity": contents["assets/metaworld_mt50/norm_stats.json"], "content_digests": contents,
            "tokenizer": tokenizer, "code_digest": code_identity, "source_digests": code, "checkpoint_root": str(root),
            "spec": asdict(model_spec(checkpoint_identity, contents["assets/metaworld_mt50/norm_stats.json"],
                                      code_digest=code_identity)),
            "available_samplers": [asdict(sampler_for_steps(k)) for k in (10, 2, 1)],
            "components": {"weights": "model.safetensors", "base": "embedded in complete PyTorch checkpoint",
                           "delta": None, "norm": "assets/metaworld_mt50/norm_stats.json"},
            "qualification": {"model_loaded": False, "gpu": "unexecuted"}}


def load_adapter(spec):
    """Explicit load only. No norm fallback, no checkpoint URI side effects."""
    checkpoint = spec["checkpoint"]
    manifest = content_manifest(checkpoint)
    if manifest["tokenizer"]["sha256"] is None:
        raise ValueError("local PaliGemma tokenizer content must be sealed before model load; set OPENPI_DATA_HOME")
    expected = spec.get("checkpoint_identity")
    if expected is not None and expected != manifest["checkpoint_identity"]:
        raise ValueError("immutable checkpoint identity mismatch")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    policy = create_trained_policy(get_config("pi05_metaworld"), checkpoint,
                                   sample_kwargs={"num_steps": 10}, pytorch_device=spec.get("device", "cpu"))
    _verify_loaded_tokenizer(policy, manifest["tokenizer"]["sha256"])
    return Pi05MetaWorldAdapter(policy, model_spec(manifest["checkpoint_identity"], manifest["norm_identity"],
                                                 code_digest=manifest["code_digest"]))
