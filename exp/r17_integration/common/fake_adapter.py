"""Deterministic CPU staged adapter for framework conformance, never quality."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .contracts import (ActionStateLayout, Capabilities, Capability, CaptureSpec,
                        CPUModelBlueprint, FeatureSpec, ModelSpec, NativeSnapshot,
                        PreparedObs, SampleResult, SamplerSpec, Stage1, Stage2, StateView, WireChunk)
from .provider import AdapterProvider, register_provider


def fake_sampler(k=4):
    return SamplerSpec(f"fake_flow_k{k}.v1", "diagnostic_flow", k,
                       tuple(1 - i / k for i in range(k)), tuple(-1 / k for _ in range(k)))


def fake_spec(*, horizon=16, action_dim=4, views=1, task_count=50, model_id="fake_mw", width=None):
    layout = ActionStateLayout(horizon, min(5, horizon), action_dim, tuple(range(action_dim)),
                               tuple(range(action_dim - 1)), action_dim - 1,
                               output_transform_id="fake_scale2.v1", model_action_width=width)
    features = {f"vision_{i}": 4 for i in range(views)} | {"robot_state": action_dim}
    caps = Capabilities({k: Capability("supported", evidence=("tests/common",)) for k in
                         ("full", "k_step", "concurrent", "bundle", "episode", "cp1_retrieval", "debug_capture", "native_resume")})
    return ModelSpec(model_id, "fake.weights.sha256", "fake.norm.sha256", layout,
                     FeatureSpec("fake.policy.stage1", tuple(f"camera{i}" for i in range(views)), features),
                     fake_sampler(), caps, ("fake_mw.v1", "fake_libero.v1"), code_digest="diagnostic")


class FakeAdapter:
    def drain_queue_wait_ms(self):
        # This CPU diagnostic adapter acquires no model lock.
        return 0.0
    def __init__(self, spec=None):
        self.spec = spec or fake_spec()
        self.counts = {"prepare": 0, "stage1": 0, "stage2": 0, "nfe": 0, "inverse": 0}
        self.lifecycle = []

    def metadata(self):
        return self.spec

    def prepare(self, wire_obs, request):
        assert not any(k.startswith("__") for k in wire_obs), "envelope entered model"
        self.counts["prepare"] += 1
        return PreparedObs(dict(wire_obs), request, self.extract_state_only(wire_obs).state)

    def extract_state_only(self, wire_obs):
        state = np.asarray(wire_obs["state"], np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("fake state must be finite vector")
        return StateView(state)

    def run_stage1(self, prepared):
        self.counts["stage1"] += 1
        named = {}
        for name, dim in self.spec.feature_spec.fields.items():
            named[name] = (prepared.state[:dim].copy() if name == "robot_state" else
                           np.asarray(prepared.data.get(name, np.arange(dim)), np.float32).copy())
        return Stage1(prepared, named, prepared.state[None])

    def retrieval_fields(self, s1):
        return {k: v.copy() for k, v in s1.fields.items()}

    def run_stage2(self, s1, *, capture=False):
        self.counts["stage2"] += 1
        target = np.resize(s1.data.state, (self.spec.layout.horizon, self.spec.layout.action_width)).astype(np.float32)
        return Stage2(target, s1)

    def _loop(self, s2, sampler, rng, capture, x, start):
        snapshots = []
        for i in range(start, sampler.num_steps):
            if capture.snapshots and i > 0:
                snapshots.append(NativeSnapshot(self.spec.checkpoint_identity, self.spec.norm_identity,
                    self.spec.layout.identity, sampler.identity, sampler.schedule_digest, i,
                    sampler.native_timesteps[i], x.copy()))
            self.counts["nfe"] += 1
            # Deliberately stable closed-form diagnostic, all K share target.
            x = x + (s2.data - x) / np.float32(sampler.num_steps - i)
        return SampleResult(x, sampler.num_steps - start, sampler, tuple(snapshots), rng_receipt=rng.receipt())

    def sample(self, s2, sampler, rng, capture):
        x = rng.normal(s2.data.shape)
        return self._loop(s2, sampler, rng, capture, x, 0)

    def resume(self, s2, snapshot, sampler, rng, capture):
        snapshot.validate(self.spec, sampler)
        return self._loop(s2, sampler, rng, capture, np.array(snapshot.x_t, copy=True), snapshot.loop_index)

    def to_wire(self, normalized_chunk, prepared):
        self.counts["inverse"] += 1
        return WireChunk(np.asarray(normalized_chunk, np.float32)[:, self.spec.layout.active_actions] * np.float32(2))

    def make_trace_adapter(self):
        return FakeTraceAdapter()

    def make_stage_batcher(self):
        return FakeStageBatcher(self)

    def cpu_blueprint(self):
        return CPUModelBlueprint(self.spec, {"diagnostic": True})

    def on_task_begin(self):
        self.lifecycle.append("begin")

    def on_task_end(self):
        self.lifecycle.append("end")


class FakeTraceAdapter:
    def capture(self, decision):
        return {"fields": {k: np.array(v, copy=True) for k, v in decision.fields.items()},
                "normalized": np.array(decision.normalized_chunk, copy=True),
                "wire": np.array(decision.wire_actions, copy=True)}


def fake_provider(provider_id="fake.v1", spec=None):
    spec = spec or fake_spec()
    from .benchmarks import register_benchmark
    from .contracts import BenchmarkSpec
    for protocol in spec.allowed_benchmark_protocols:
        if protocol.startswith("fake_"):
            register_benchmark(BenchmarkSpec("fake", "r17.fixture.v1", protocol,("reach",),("reach",),5,0,160),
                               protocol_identity=protocol)
    p = AdapterProvider(provider_id, spec, lambda: FakeAdapter(spec), fake_sampler)
    register_provider(p)
    return p


class FakeStageBatcher:
    """Vector CPU S3; stock BatchingCore scheduling/fault tests use this."""
    def __init__(self, adapter):
        self.adapter = adapter

    def bucket_key(self, payload):
        if hasattr(payload, "noise"):
            return (self.adapter.spec.identity, tuple(payload.noise.shape), str(payload.noise.dtype),
                    payload.num_steps, payload.save_timesteps)
        return (self.adapter.spec.identity, type(payload).__name__)

    def run_stage1_batch(self, payloads):
        return [self.adapter.run_stage1(p) for p in payloads]

    def run_stage2_batch(self, payloads, *, capture=False):
        return [self.adapter.run_stage2(p, capture=capture) for p in payloads]

    def run_stage3_miss(self, payloads, *, num_steps, save_timesteps_per_request):
        import torch
        from openpi.serving.batching_core import Stage3VariantOutput
        if any(not torch.isfinite(p.noise).all() for p in payloads):
            raise ValueError("fake batch nonfinite noise")
        x = torch.stack([p.noise for p in payloads])
        target = torch.stack([torch.as_tensor(p.stage2_out.data) for p in payloads])
        snaps = {}
        for i in range(num_steps):
            t = round(1 - i / num_steps, 4)
            snaps[t] = x.clone()
            x = x + (target - x) / (num_steps - i)
        return [Stage3VariantOutput(x[j:j+1].clone(),
                    {t: snaps[t][j:j+1].clone() for t in (requested or ())})
                for j, requested in enumerate(save_timesteps_per_request)]

    def run_stage3_warm(self, *args, **kwargs):
        raise ValueError("use native FakeAdapter.resume for diagnostic snapshots")
