"""Staged execution of E's reconstruction, without editing or patching it."""
from dataclasses import asdict, replace

import numpy as np
import torch

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, CPUModelBlueprint, FeatureSpec,
    ModelSpec, NativeSnapshot, PreparedObs, SampleResult, Stage1, Stage2, StateView, WireChunk,
)

from .manifest import code_identity, digest, feature_spec, framework_identity, layout, tensor_digest
from .normalization import Normalization
from .request_noise import RequestNoise
from .sampler import SamplerProfile, checked_profile


class DPAdapter:
    def __init__(self, model, stats, language, *, domain, weight_identity,
                 second_camera="zero", language_identity="diagnostic", diagnostic=True,
                 source_digests=None, libero_benchmark_specs=None):
        if domain not in ("libero", "metaworld") or second_camera not in ("zero", "duplicate"):
            raise ValueError("invalid DP domain/second-camera policy")
        d, s = (7, 8) if domain == "libero" else (4, 4)
        if (model.cfg["action_dim"], model.cfg["lowdims"][0]["dim"], model.cfg["horizon"]) != (d, s, 16):
            raise ValueError("DP checkpoint action/state/horizon disagree with domain")
        if model.training:
            raise ValueError("DP staged inference requires eval mode")
        order = tuple(model._policy.obs_encoder.feature_order)
        if len(order) != 4 or set(order) != {"base_0_rgb", "left_wrist_0_rgb", "state", "lang_emb"}:
            raise ValueError("DP feature order must contain all four native features exactly once")
        self.model, self.language, self.domain = model, language, domain
        self.weight_identity, self.language_identity = weight_identity, language_identity
        self.device = next(model.parameters()).device
        self.second_camera, self.diagnostic = second_camera, diagnostic
        self.batch_diagnostic_label = None
        self.batch_config = None
        self.batch_ledger = None
        self.libero_wire_protocols = {}
        if libero_benchmark_specs:
            if domain != "libero":
                raise ValueError("LIBERO benchmark specs cannot bind MetaWorld")
            from exp.r17_integration.common.benchmarks import register_benchmark
            from .protocols import libero_spec_from_document
            for document in libero_benchmark_specs:
                benchmark = libero_spec_from_document(document, replan_steps=layout(domain)["replan_steps"])
                register_benchmark(benchmark)
                self.libero_wire_protocols[benchmark.identity] = benchmark.wire_size
        self.normalization = Normalization(model, stats)
        self.source_digests = source_digests or code_identity()
        from importlib.metadata import version
        self.dependency_versions = {name: version(name) for name in
                                    ("torch", "torchvision", "diffusers", "transformers", "numpy", "safetensors")}
        layout_data = layout(domain)
        layout_data.pop("valid_action_mask")
        layout_data["raw_state_fields"] = (layout_data["raw_state_fields"],)
        for name in ("valid_state_indices", "motion_channels"):
            layout_data[name] = tuple(layout_data[name])
        native_layout = ActionStateLayout(**layout_data)
        info = feature_spec(domain, second_camera)
        fields = {"vision_0": 64, "robot_state": s}
        if domain == "libero":
            fields["vision_1"] = 64
        code_digest = digest({"sources": self.source_digests, "dependencies": self.dependency_versions})
        # Model identity also binds language weights, reconstructed conventions,
        # feature order, absent branch and adapter source, not only DP tensors.
        identity = digest({"weights": weight_identity, "language": language_identity,
                           "cfg": model.cfg, "feature_order": order, "domain": domain,
                           "second_camera": second_camera, "code": code_digest,
                           "norm": self.normalization.identity})
        features = FeatureSpec(source=info["cutpoint"], real_cameras=tuple(info["real_cameras"]),
                               fields=fields, hidden_width=64, pooling=info["pooling"],
                               missing_branch_policy="wire" if domain == "libero" else second_camera,
                               prompt_dependent=True, active_state_indices=tuple(range(s)),
                               policy_digest=identity, norm_digest=self.normalization.identity,
                               preprocess_digest=digest(info))
        capabilities = {name: Capability("supported", "CPU conformance; GPU unqualified") for name in
                        ("full", "k_step", "cp1_retrieval",
                         "debug_capture")}
        capabilities.update({name: Capability("unsupported", "GPU batch composition changes outputs; production B1 only; labelled offline batch diagnostics available")
                             for name in ("batch_stage1", "batch_stage2", "batch_stage3")})
        capabilities.update({name: Capability("implemented_unverified", "shared runtime/GPU qualification pending")
                             for name in ("native_warm_resume", "native_warm_reset", "library", "recipe", "warmup",
                                          "offline", "replay", "conductor", "worker", "analysis", "trace_twins")})
        capabilities["k_step"] = Capability("supported", "Trailing DDIM math; reduced quality unqualified; K1 lossy, binding restricted to labelled plain step-reduction study")
        capabilities.update({name: Capability("supported", "P1 framework path; CPU connection conformance")
                             for name in ("concurrent", "bundle", "episode")})
        capabilities.update({name: Capability("unsupported", "native/shared conformance certificate pending")
                             for name in ("cp2", "online_rit", "shadow", "routing", "split")})
        capabilities["flow_shoot"] = Capability("not_applicable", "DDPM/DDIM are discrete diffusion, not flow ODE")
        if domain == "metaworld":
            capabilities["two_camera_ablation"] = Capability("not_applicable", "one real MW camera")
            from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
            protocols = (FORMAL_SPEC.identity,)
        else:
            protocols = tuple(self.libero_wire_protocols) or ("libero_r17_v1",)
        self._spec = ModelSpec(model_id=f"dp_t_{domain}" + ("_diagnostic" if diagnostic else ""),
                               checkpoint_identity=identity, norm_identity=self.normalization.identity,
                               layout=native_layout, feature_spec=features,
                               default_sampler=SamplerProfile().to_spec(model.cfg),
                               capabilities=Capabilities(capabilities),
                               allowed_benchmark_protocols=protocols, code_digest=code_digest,
                               python_island="/data/mw_finetune/dp/env/bin/python")

    def metadata(self):
        return self._spec

    def for_batch_diagnostic(self, label):
        """Explicit offline clone; cannot register or serve as a provider."""
        import copy
        if not isinstance(label, str) or not label.strip():
            raise ValueError("DP batch diagnostic requires a nonempty label")
        own = copy.copy(self)
        own.batch_diagnostic_label = label
        return own

    def _check_batch(self, count):
        if self.batch_config is not None:
            if self.domain not in ("libero", "metaworld") or not 1 <= count <= self.batch_config.max_batch:
                raise ValueError("DP batch exceeds the explicit execution configuration")
            return
        if count > 1 and not self.batch_diagnostic_label:
            raise ValueError("DP production B>1 disabled: GPU batch composition changes outputs")

    def with_sampler(self, profile):
        """An explicit DDIM research baseline gets its own ModelSpec/endpoint."""
        import copy
        _, sampler = checked_profile(profile, self.model.cfg)
        own = copy.copy(self)
        suffix = f"_{sampler.family}_baseline" if sampler.family != "ddpm" else ""
        own._spec = replace(self._spec, model_id=self._spec.model_id+suffix, default_sampler=sampler)
        return own

    def _state(self, wire_obs):
        key = "state" if self.domain == "metaworld" and "state" in wire_obs else "observation/state"
        value = np.asarray(wire_obs[key], np.float32)
        if value.shape != (len(self._spec.layout.valid_state_indices),) or not np.isfinite(value).all():
            raise ValueError("invalid DP wire state")
        return value.copy()

    def extract_state_only(self, wire_obs):
        return StateView(self._state(wire_obs))

    @torch.inference_mode()
    def prepare(self, wire_obs, request):
        from exp.mw_finetune.dp.preprocess import image_tensor
        image_key = "image" if self.domain == "metaworld" and "image" in wire_obs else "observation/image"
        if self.domain == "libero":
            sizes = []
            for key in (image_key, "observation/wrist_image"):
                value = np.asarray(wire_obs[key])
                if value.dtype != np.uint8 or value.shape not in ((96, 96, 3), (256, 256, 3)):
                    raise ValueError("DP LIBERO wire requires paired96 or raw256 uint8 RGB; resampled224 is rejected")
                sizes.append(value.shape[:2])
            if sizes[0] != sizes[1]:
                raise ValueError("DP LIBERO paired views must have the same wire resolution")
            expected = self.libero_wire_protocols.get(request.benchmark_protocol)
            if expected is not None and sizes[0] != (expected, expected):
                raise ValueError("DP LIBERO image resolution differs from the bound benchmark protocol")
        state = self._state(wire_obs)
        prompt = wire_obs["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("DP prompt must be a nonempty string")
        image = image_tensor(wire_obs[image_key])
        second = image_tensor(wire_obs["observation/wrist_image"]) if self.domain == "libero" else (
            image.clone() if self.second_camera == "duplicate" else torch.zeros(3, 96, 96))
        language = self.language([prompt])
        if tuple(language.shape) != (1, self.model.cfg["lang_emb_dim"]) or not torch.isfinite(language).all():
            raise ValueError("invalid projected DP language embedding")
        data = {"inputs": {"base_0_rgb": image.unsqueeze(0), "left_wrist_0_rgb": second.unsqueeze(0),
                           "state": torch.from_numpy(self.normalization.state(state)).unsqueeze(0),
                           "lang_emb": language}, "prompt": prompt}
        data["inputs"] = {k: v.to(self.device) for k, v in data["inputs"].items()}
        return PreparedObs(data=data, request=request, state=state)

    @torch.inference_mode()
    def run_stage1(self, prepared):
        return self.run_stage1_batch([prepared])[0]

    @torch.inference_mode()
    def run_stage1_batch(self, prepareds):
        self._check_batch(len(prepareds))
        from exp.mw_finetune.dp.model import crop
        core = self.model._policy
        obs = {k: torch.cat([p.data["inputs"][k] for p in prepareds], 0)
               for k in prepareds[0].data["inputs"]}
        normalized = {k: core.normalizer.normalize(k, v) for k, v in obs.items()}
        # These are exactly E's ObservationEncoder branches after its ReLU.
        features = {k: torch.relu(net(crop(normalized[k], False), normalized["lang_emb"]))
                    for k, net in core.obs_encoder.obs_nets.items()}
        features.update(state=normalized["state"], lang_emb=normalized["lang_emb"])
        outputs = []
        for i, p in enumerate(prepareds):
            own_features = {k: v[i:i+1].clone() for k, v in features.items()}
            fields = {"vision_0": own_features["base_0_rgb"][0].float().cpu().numpy().copy(),
                      "robot_state": np.asarray(p.state, np.float32).copy()}
            if self.domain == "libero":
                fields["vision_1"] = own_features["left_wrist_0_rgb"][0].float().cpu().numpy().copy()
            outputs.append(Stage1(data={"prepared": p, "features": own_features}, fields=fields, state=p.state))
        return outputs

    def retrieval_fields(self, s1):
        return {k: np.ascontiguousarray(v, dtype=np.float32).copy() for k, v in s1.fields.items()}

    @torch.inference_mode()
    def run_stage2(self, s1, *, capture=False):
        features = s1.data["features"]
        cond = torch.cat([features[k] for k in self.model._policy.obs_encoder.feature_order], -1).unsqueeze(1)
        return Stage2(data={"conditioning": cond}, stage1=s1)

    def _generator(self, rng):
        if isinstance(rng, RequestNoise):
            return rng.generator
        # P1 RequestRNG gives a semantic initial seed. All posterior draws
        # continue this same generator, matching E's sequential sampler.
        return rng.torch_generator(device=self.device)

    def _snapshot(self, s2, sampler, index, x, generator, *, conditioning_scope=None):
        spec = self._spec
        return NativeSnapshot(model_identity=spec.checkpoint_identity, norm_identity=spec.norm_identity,
                              layout_identity=spec.layout.identity, sampler_identity=sampler.identity,
                              schedule_digest=sampler.schedule_digest, loop_index=index,
                              native_timestep=sampler.native_timesteps[index], x_t=x[0].clone(),
                              noise_state=generator.get_state().clone(),
                              conditioning_scope=conditioning_scope or tensor_digest(s2.data["conditioning"]))

    @torch.inference_mode()
    def _loop(self, s2, sampler, rng, capture, *, snapshot=None, initial_noise=None):
        return self.sample_batch([s2], sampler, [rng], [capture], snapshots=[snapshot],
                                 initial_noises=[initial_noise])[0]

    @torch.inference_mode()
    def sample_batch(self, stages, sampler, rngs, captures, *, snapshots=None, initial_noises=None):
        self._check_batch(len(stages))
        profile, sampler = checked_profile(sampler, self.model.cfg)
        vector = self.batch_config is not None
        if vector and (sampler.num_steps not in (100, 10, 5) or profile.eta != 0):
            raise ValueError("DP production batching supports DDPM100 or trailing eta0 DDIM10/5")
        # Scheduler.step has no per-row mutable state for these audited
        # profiles. A batch-local scheduler applies elementwise coefficients;
        # diffusers' generator-list helper still draws each (1,H,D) noise from
        # its own stream, preserving B1 posterior draws exactly.
        schedulers = [profile.scheduler(self.model.cfg)] * len(stages) if vector else [profile.scheduler(self.model.cfg) for _ in stages]
        snapshots = snapshots or [None] * len(stages)
        initial_noises = initial_noises or [None] * len(stages)
        if not (len(stages) == len(rngs) == len(captures) == len(snapshots) == len(initial_noises)):
            raise ValueError("DP batch envelopes must have identical lengths")
        generators = [self._generator(rng) for rng in rngs]
        starts = []
        trajectories = []
        scopes = [tensor_digest(s.data["conditioning"]) if snap is not None or (c.enabled and c.snapshots)
                  else None for s, snap, c in zip(stages, snapshots, captures, strict=True)]
        for s2, snap, z, g, scope in zip(stages, snapshots, initial_noises, generators, scopes, strict=True):
            cond = s2.data["conditioning"]
            shape = (1, 16, self.model.cfg["action_dim"])
            if snap is not None:
                snap.validate(self._spec, sampler, conditioning_scope=scope)
                if snap.x_t.dtype != cond.dtype or snap.x_t.device != cond.device or not torch.isfinite(snap.x_t).all():
                    raise ValueError("snapshot dtype/device/nonfinite mismatch")
                if snap.noise_state is None:
                    raise ValueError("DP suffix requires private posterior RNG state")
                g.set_state(snap.noise_state.cpu())
                trajectories.append(snap.x_t.unsqueeze(0).clone())
                starts.append(snap.loop_index)
            else:
                x = torch.randn(shape, dtype=cond.dtype, device=cond.device, generator=g) if z is None else z
                if tuple(x.shape) == shape[1:]:
                    x = x.unsqueeze(0)
                if tuple(x.shape) != shape or x.dtype != cond.dtype or x.device != cond.device or not torch.isfinite(x).all():
                    raise ValueError("initial noise shape/dtype/device mismatch")
                trajectories.append(x.clone())
                starts.append(0)
        if len(set(starts)) != 1:
            raise ValueError("native resume points must be bucketed separately")
        start = starts[0]
        saved = [[] for _ in stages]
        first_inputs = [None] * len(stages)
        first_xs = [None] * len(stages)
        cond = torch.cat([s.data["conditioning"] for s in stages], 0)
        x = torch.cat(trajectories, 0)
        for index in range(start, sampler.num_steps):
            t = schedulers[0].timesteps[index]
            for i, (s, c, g) in enumerate(zip(stages, captures, generators, strict=True)):
                if c.enabled and c.snapshots and index > 0:
                    saved[i].append(self._snapshot(s, sampler, index, x[i:i+1], g, conditioning_scope=scopes[i]))
                if c.enabled and index == start:
                    first_inputs[i] = x[i].clone()
            prediction = self.model._policy.model(x, t, cond)
            # Keep B1 scheduler arithmetic and B1 noise shapes per request.
            # A shared generator or batched draw changes E's sequential stream.
            if vector:
                x = profile.step(schedulers[0], prediction, t, x, generators)
            else:
                x = torch.cat([profile.step(sch, prediction[i:i+1], t, x[i:i+1], g)
                               for i, (sch, g) in enumerate(zip(schedulers, generators, strict=True))], 0)
            if not vector and not torch.isfinite(x).all():
                raise ValueError("nonfinite DP native update")
            if index == start:
                for i, c in enumerate(captures):
                    if c.enabled:
                        first_xs[i] = x[i].clone()
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite DP native update")
        result = []
        for i, g in enumerate(generators):
            receipt = {"stream": "sequential_initial_then_posterior_v1",
                       "seed": g.initial_seed(), "state_digest": tensor_digest(g.get_state()),
                       "posterior_draws": sum(1 for t in sampler.native_timesteps[start:] if t > 0)
                       if profile.kind == "ddpm" else (sampler.num_steps-start if profile.eta else 0)}
            result.append(SampleResult(normalized_chunk=x[i].clone(), actual_nfe=sampler.num_steps-start,
                                       sampler=sampler, snapshots=tuple(saved[i]),
                                       capture={"first_step_input": first_inputs[i], "first_step_x": first_xs[i],
                                                **({"batch_diagnostic_label": self.batch_diagnostic_label}
                                                   if self.batch_diagnostic_label else {}),
                                                "conditioning": stages[i].data["conditioning"].clone()
                                                if captures[i].enabled else None},
                                       rng_receipt=receipt))
        return result

    def sample(self, s2, sampler, rng, capture):
        return self._loop(s2, sampler, rng, capture)

    def resume(self, s2, snapshot, sampler, rng, capture):
        return self._loop(s2, sampler, rng, capture, snapshot=snapshot)

    @torch.inference_mode()
    def warm_reset(self, s2, normalized_clean, *, loop_index, sampler, rng, capture):
        profile, sampler = checked_profile(sampler, self.model.cfg)
        if type(loop_index) is not int or not 0 < loop_index < sampler.num_steps:
            raise ValueError("native reset requires an interior loop index")
        clean = normalized_clean.unsqueeze(0)
        cond = s2.data["conditioning"]
        if clean.shape != (1, 16, self.model.cfg["action_dim"]) or clean.dtype != cond.dtype or clean.device != cond.device:
            raise ValueError("native reset clean chunk shape/dtype/device mismatch")
        g = self._generator(rng)
        z = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=g)
        timestep = torch.tensor([sampler.native_timesteps[loop_index]], device=clean.device)
        x = profile.scheduler(self.model.cfg).add_noise(clean, z, timestep)
        snapshot = self._snapshot(s2, sampler, loop_index, x, g)
        return self._loop(s2, sampler, rng, capture, snapshot=snapshot)

    @torch.inference_mode()
    def to_wire(self, normalized_chunk, prepared):
        expected = (16, self.model.cfg["action_dim"])
        if tuple(normalized_chunk.shape) != expected:
            raise ValueError("DP action chunk shape mismatch")
        normalized_chunk = torch.as_tensor(normalized_chunk, device=self.device,
                                            dtype=self.model._policy.normalizer.params_dict["action"]["scale"].dtype)
        return WireChunk(self.normalization.to_wire(normalized_chunk))

    def make_stage_batcher(self):
        from .batcher import DPStageBatcher
        return DPStageBatcher(self)

    def make_trace_adapter(self):
        from .trace import DPTraceAdapter
        return DPTraceAdapter(self)

    def cpu_blueprint(self):
        return CPUModelBlueprint(self._spec, {"domain": self.domain, "config": dict(self.model.cfg),
                                             "second_camera": self.second_camera,
                                             "diagnostic": self.diagnostic, "split_qualified": False})

    def manifest(self):
        return {"schema": "r17.model_manifest.v1", "spec": asdict(self._spec),
                "content_digests": {**self.source_digests, "DP_weights": self.weight_identity,
                                    "language_tokenizer": self.language_identity,
                                    "normalization": self._spec.norm_identity, **getattr(self, "asset_digests", {})},
                "dependency_versions": self.dependency_versions,
                "framework_digests": framework_identity(),
                "strict_load": getattr(self, "strict_load", {"status": "diagnostic_toy_no_weight_load"}),
                "diagnostic": self.diagnostic, "reconstruction": "E checkpoint-compatible reconstructed wrapper"}
