"""GR00T's native stage hooks, consuming P1's r17.contracts.v1 unchanged."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
import numpy as np
import torch

from exp.r17_integration.common.contracts import (
    CaptureSpec, NativeSnapshot, PreparedObs, SampleResult, Stage1, Stage2, StateView, WireChunk,
)
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import CheckpointID
from .key_builder import GrootMetaWorldCP1SpatialPool16KeyBuilder
from .manifest import PROVIDER, validate_library, validate_library_model
from .wire_policy import from_groot_actions, observation_field, to_groot_observation


class _TimedLockView:
    """Connection-local acquisition timer over the process-shared native RLock.

    Exclusion is exactly the shared lock's; only this connection's time spent
    waiting to acquire it is accumulated, so P1 can separate queueing from
    host service time (``drain_queue_wait_ms``). No model/RNG state is touched.
    """
    def __init__(self, lock):
        self._lock = lock
        self._wait_ms = 0.0

    def __enter__(self):
        begin = time.perf_counter()
        self._lock.acquire()
        self._wait_ms += (time.perf_counter() - begin) * 1000.0
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False

    def drain(self):
        wait, self._wait_ms = self._wait_ms, 0.0
        return wait


class GrootMetaWorldAdapter:
    def __init__(self, backend, manifest, *, model_lock=None, runner_factory=GrootStagedRunner):
        self.backend = backend
        self.manifest = manifest
        self.spec = manifest.model_spec()
        self.model_lock = model_lock if model_lock is not None else threading.RLock()
        # Shared exclusion, connection-local queue-wait ledger (N5).
        self._timed_lock = _TimedLockView(self.model_lock)
        self.runner = runner_factory(backend.model)  # source drift guards stay enabled
        self.key_builder = GrootMetaWorldCP1SpatialPool16KeyBuilder(hidden_width=manifest.hidden_width)
        self._state_transforms = []
        if not hasattr(backend, "normalize_state_only"):
            transforms = getattr(getattr(backend, "_modality_transform", None), "transforms", ())
            state_keys = {"state." + name for name in ("x", "y", "z", "gripper")}
            self._state_transforms = [t for t in transforms if set(getattr(t, "apply_to", ())) == state_keys
                                      and type(t).__name__ in ("StateActionToTensor", "StateActionTransform")]
            if [type(t).__name__ for t in self._state_transforms] != ["StateActionToTensor", "StateActionTransform"]:
                raise ValueError("backend must expose the exact four-channel state normalization transforms")
        self._require_live()
        head = backend.model.action_head
        if head.config.action_horizon != 16 or head.config.action_dim != 32:
            raise ValueError("GR00T-MW model must have H16/action32 storage")

    def _require_live(self, sampler=None):
        live = self.runner.live_schedule()
        if live.num_steps != self.manifest.denoising_steps:
            raise RuntimeError("shared GR00T head K changed after endpoint creation")
        if sampler is not None and sampler != self.spec.default_sampler:
            raise ValueError("sampler differs from endpoint; dispatch to a compatible K4/2/1 process")
        return live

    def metadata(self):
        return self.spec

    def drain_queue_wait_ms(self):
        """Return and reset this connection's shared model-lock acquisition wait."""
        return self._timed_lock.drain()

    def on_task_begin(self):
        self.key_builder.clear()
        self.key_builder.on_episode_start()

    def on_episode_start(self, **kwargs):
        self.on_task_begin()

    def on_episode_end(self, **kwargs):
        self.key_builder.clear()

    def on_task_end(self):
        self.key_builder.clear()

    def prepare(self, wire_obs, request):
        if request.benchmark_protocol != self.manifest.benchmark_protocol:
            raise ValueError("request benchmark protocol differs from model binding")
        # Match stock interceptor's unbatched→B1 transform contract, once.
        native = to_groot_observation(wire_obs)
        batched = {name: np.expand_dims(value, 0) for name, value in native.items()}
        with self._timed_lock:
            normalized = self.backend.apply_transforms(batched)
        return PreparedObs(normalized, request, np.asarray(observation_field(wire_obs, "state"), dtype=np.float32).copy())

    def run_stage1(self, prepared):
        self._require_live()
        with self._timed_lock, self.runner.session():
            native = self.runner.run_stage1(prepared.data)
        self.key_builder.collect(CheckpointID.CP1, stage1=native)
        try:
            fields = self.key_builder.build(CheckpointID.CP1)
        finally:
            self.key_builder.clear()
        return Stage1(native, fields, native.state[0, -1, :4].detach().cpu().float().clone())

    def retrieval_fields(self, s1):
        return {name: value.clone() for name, value in s1.fields.items()}

    def run_stage2(self, s1, *, capture=False):
        del capture
        self._require_live()
        with self._timed_lock, self.runner.session():
            native = self.runner.run_stage2_llm(s1.data)
        return Stage2(native, s1)

    def sample(self, s2, sampler, rng, capture=CaptureSpec()):
        self._require_live(sampler)
        snapshots = []

        def observe(index, x_in, x_out):
            del x_out
            if index:
                snapshots.append(NativeSnapshot(self.spec.checkpoint_identity, self.spec.norm_identity,
                    self.spec.layout.identity, sampler.identity, sampler.schedule_digest,
                    index, sampler.native_timesteps[index], x_in[0].detach().cpu().float().clone(),
                    noise_state={"native_dtype": str(x_in.dtype), "storage_dtype": "float32"}))

        with self._timed_lock, self.runner.session():
            generator = rng.torch_generator(device=str(s2.data.backbone_features.device))
            noise = self.runner.sample_noise(s2.data, generator=generator)
            out = self.runner.run_stage3(s2.data, noise=noise,
                                        on_step=observe if capture.enabled and capture.snapshots else None)
        normalized = out.action_pred[0].detach().cpu().float().clone()
        if normalized.shape != (16, 32) or not torch.isfinite(normalized).all():
            raise ValueError("GR00T returned invalid normalized action chunk")
        captured = {"initial_noise": noise[0].detach().cpu().clone()} if capture.enabled else {}
        snapshots = tuple(replace(snap, x_t=snap.x_t.clone()) for snap in snapshots)
        return SampleResult(normalized, out.steps_run, sampler, snapshots, captured, rng.receipt())

    def resume(self, s2, snapshot, sampler, rng, capture=CaptureSpec()):
        schedule = self._require_live(sampler)
        snapshot.validate(self.spec, sampler)
        x_t = torch.as_tensor(snapshot.x_t)
        if not torch.isfinite(x_t).all():
            raise ValueError("native snapshot contains non-finite values")
        with self._timed_lock, self.runner.session():
            expected_dtype = self.runner._noise_dtype(s2.data)
            dtype_receipt = snapshot.noise_state or {}
            if x_t.dtype != expected_dtype:
                # BF16 snapshots are exactly representable in portable float32 NPZ/H5.
                # Permit that declared lossless transport, never an implicit cast.
                if (x_t.dtype != torch.float32 or dtype_receipt.get("native_dtype") != str(expected_dtype)
                        or dtype_receipt.get("storage_dtype") != "float32"
                        or not torch.equal(x_t.to(expected_dtype).float(), x_t)):
                    raise ValueError("native snapshot dtype differs from live action loop")
                x_t = x_t.to(expected_dtype)
            out = self.runner.run_stage3_from(s2.data, x_t, float(snapshot.native_timestep),
                                              schedule=schedule, capture_first_step=capture.enabled)
        normalized = out.action_pred[0].detach().cpu().float().clone()
        if not torch.isfinite(normalized).all():
            raise ValueError("resumed action is not finite")
        captured = {}
        if capture.enabled:
            captured = {"first_step_input": out.first_step_input.detach().cpu().clone(),
                        "first_step_output": out.first_step_x.detach().cpu().clone()}
        return SampleResult(normalized, out.steps_run, sampler, capture=captured, rng_receipt=rng.receipt())

    def to_wire(self, normalized_chunk, prepared):
        del prepared
        chunk = torch.as_tensor(normalized_chunk).detach().cpu().float().clone()
        if chunk.shape != (16, 32) or not torch.isfinite(chunk).all():
            raise ValueError("normalized chunk must be finite H16×32")
        with self._timed_lock:
            result = self.backend.unapply_transforms({"action": chunk[None]})
        return WireChunk(from_groot_actions(result))

    def extract_state_only(self, wire_obs):
        state = np.asarray(observation_field(wire_obs, "state"), dtype=np.float32)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("state-only tail requires finite state4")
        with self._timed_lock:
            if hasattr(self.backend, "normalize_state_only"):
                normalized = torch.as_tensor(self.backend.normalize_state_only(state.copy()))
            else:
                data = {"state." + name: state[i:i+1].reshape(1, 1, 1).copy()
                        for i, name in enumerate(("x", "y", "z", "gripper"))}
                for transform in self._state_transforms:
                    data = transform(data)
                normalized = torch.cat([data["state." + name].reshape(-1) for name in ("x", "y", "z", "gripper")])
            # prepare_input casts state to the head's dtype before stage1; match that quantization.
            normalized = normalized.to(getattr(self.backend.model.action_head, "dtype", torch.float32)).float()
        if normalized.shape != (4,) or not torch.isfinite(normalized).all():
            raise ValueError("state normalization returned invalid active4 values")
        return StateView(normalized.detach().cpu().numpy().copy())

    def make_stage_batcher(self):
        from .batcher import GrootMetaWorldStageBatcher
        return GrootMetaWorldStageBatcher(self.runner, self.manifest)

    def make_trace_adapter(self):
        from .trace import make_trace_adapter
        return make_trace_adapter(self.manifest)

    def capture_fields(self, s1, s2, result):
        """Observe actual stages only. Called by P1 after a MISS with a capture sink."""
        native = s1.data
        tokens = native.input_embeds[0, native.image_token_mask[0]].detach().cpu().float().clone()
        return {"vision_0_tokens": tokens, "normalized_state": native.state[0, -1, :4].detach().cpu().float().clone(),
                "state_mask": native.state_mask[0, -1].detach().cpu().clone(),
                "image_token_mask": native.image_token_mask[0].detach().cpu().clone(),
                "backbone_features": s2.data.backbone_features[0].detach().cpu().float().clone(),
                **dict(result.capture)}

    def make_cp2_key_builder(self, *, token_len, state_feat_dim, seed, d=500, p=.01):
        from .key_builder import GrootMetaWorldCP2KeyBuilder
        return GrootMetaWorldCP2KeyBuilder(token_len=token_len, feature_dim=self.manifest.hidden_width,
                                           state_feat_dim=state_feat_dim, seed=seed, d=d, p=p)

    def cp2_fields(self, s2, builder):
        self._require_live()
        with self._timed_lock, self.runner.session():
            source = self.runner.run_cp2_key_source(s2.data)
        builder.collect(CheckpointID.CP2, cp2_source=source)
        try:
            return builder.build(CheckpointID.CP2)
        finally:
            builder.clear()

    def make_warm_reset_parts(self, config, *, bundle_id, yaml_id=None, yaml_path=None):
        from openpi.cache.warm_reset.groot import build_groot_warm_reset
        self._require_live()
        if config.denoise_schedule != self.runner.live_schedule().schedule_id:
            raise ValueError("warm reset must use the immutable endpoint schedule")
        return build_groot_warm_reset(config, bundle_id=bundle_id, yaml_id=yaml_id, yaml_path=yaml_path)

    def cpu_blueprint(self):
        raise NotImplementedError("GR00T-MW split/CPU blueprint is not qualified")


def make_provider(backend, manifest, *, model_lock=None, runner_factory=GrootStagedRunner):
    from exp.r17_integration.common.provider import AdapterProvider
    lock = model_lock if model_lock is not None else threading.RLock()

    def adapter():
        return GrootMetaWorldAdapter(backend, manifest, model_lock=lock, runner_factory=runner_factory)

    def sampler(k):
        if type(k) is not int or k != manifest.denoising_steps:
            raise ValueError("requested K differs from immutable GR00T endpoint K")
        return manifest.model_spec().default_sampler

    def artifact(config, storage):
        del config
        if storage._backend.count():
            meta = storage.artifact_meta or {}
            sealed = meta.get("r17_manifest")
            if sealed:
                binding = sealed.get("binding", {})
                validate_library_model(binding.get("model", {}), manifest)
                if (binding.get("benchmark_protocol") != manifest.benchmark_protocol
                        or storage._backend.vector_dims != dict(manifest.model_spec().feature_spec.fields)):
                    raise ValueError("P6 library binding/geometry differs from GR00T-MW")
            else:
                model = meta.get("model")
                if isinstance(model, dict) and model.get("r17_binding"):
                    model = model.get("model_spec", {name: value for name, value in model.items() if name != "r17_binding"})
                    validate_library_model(model, manifest)
                    if storage._backend.vector_dims != dict(manifest.model_spec().feature_spec.fields):
                        raise ValueError("stock library model/geometry differs from GR00T-MW")
                else:
                    validate_library(meta, manifest, vector_dims=storage._backend.vector_dims)

    # Generic connection policy passes adapter's already-computed named fields.
    # Stock interceptor uses the specialized raw-stage key builder separately.
    return AdapterProvider(PROVIDER, manifest.model_spec(), adapter, sampler,
                           artifact_validator=artifact, trace_factory=lambda adapter: adapter.make_trace_adapter())


def register_provider(provider=None, *, backend=None, manifest=None, **kwargs):
    from exp.r17_integration.common.provider import register_provider as register, register_trace_provider
    if provider is None:
        if backend is None or manifest is None:
            raise ValueError("explicit backend and immutable manifest are required")
        provider = make_provider(backend, manifest, **kwargs)
    register(provider)
    register_trace_provider(provider.model_spec.model_id, provider.trace_factory)
    return provider
