"""Production B1 admission and labelled diagnostics after Q's GPU failure."""
from types import SimpleNamespace
import sys

import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.models.dp import probe, serve
from exp.r17_integration.models.dp.provider import CoordinatedAdapter, make_connection_factory, register_provider
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.models.dp.wire_policy import WirePolicy
from exp.r17_integration.tests.models.dp.conftest import observation


@pytest.mark.parametrize("domain", ["libero", "metaworld"])
@pytest.mark.parametrize("entry", ["stage1", "stage2", "sample", "miss", "warm"])
def test_production_rejects_B2_before_forward_or_rng_mutation(make_adapter, req_context, monkeypatch, domain, entry):
    adapter = make_adapter(domain)
    batcher = adapter.make_stage_batcher()
    prepared = adapter.prepare(observation(domain), req_context)
    s1 = adapter.run_stage1(prepared)
    s2 = adapter.run_stage2(s1)
    profile = SamplerProfile(num_steps=2)
    noises = [RequestNoise(seed) for seed in (18, 37)]
    captures = [CaptureSpec()] * 2
    snapshots = adapter.sample(s2, profile, RequestNoise(18), CaptureSpec(True, True)).snapshots
    payloads = [batcher.miss_payload(s2, profile, n, CaptureSpec()) for n in noises]
    warm = [batcher.warm_payload(s2, snapshots[0], profile, n, CaptureSpec()) for n in noises]
    states = [n.generator.get_state().clone() for n in noises]
    def forbidden(*args, **kwargs):
        pytest.fail("B2 production gate ran a forward")
    monkeypatch.setattr(adapter.model._policy.model, "forward", forbidden)
    for net in adapter.model._policy.obs_encoder.obs_nets.values():
        monkeypatch.setattr(net, "forward", forbidden)
    calls = {"stage1": lambda: adapter.run_stage1_batch([prepared] * 2),
             "stage2": lambda: batcher.run_stage2_batch([s1] * 2, capture=False),
             "sample": lambda: adapter.sample_batch([s2] * 2, profile, noises, captures),
             "miss": lambda: batcher.run_stage3_miss(payloads, num_steps=2, save_timesteps_per_request=[None] * 2),
             "warm": lambda: batcher.run_stage3_warm(warm, start_t=.5, num_steps=1, capture_first_step=False)}
    with pytest.raises(ValueError, match="production B>1 disabled"):
        calls[entry]()
    assert all(torch.equal(n.generator.get_state(), state) for n, state in zip(noises, states))


def test_batch_diagnostic_is_labelled_but_cannot_register_or_serve(make_adapter, req_context, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.config import build_shared_storage
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    config = serve.plain_config(provider)
    diagnostic = adapter.for_batch_diagnostic("CPU_explicit_batch_study")
    assert adapter.batch_diagnostic_label is None
    for name in ("batch_stage1", "batch_stage2", "batch_stage3"):
        assert diagnostic.metadata().capabilities.features[name].status == "unsupported"
    metadata = WirePolicy(adapter).get_metadata()
    assert metadata["r17_batch_policy"]["production_max_batch_size"] == 1
    assert "GPU batch composition changes outputs" in metadata["r17_batch_policy"]["reason"]
    with pytest.raises(ValueError, match="cannot be registered"):
        register_provider(diagnostic)
    with pytest.raises(ValueError, match="cannot serve"):
        make_connection_factory(diagnostic, provider=provider, default_config=config,
                                default_storage=build_shared_storage(config))
    ps = [diagnostic.prepare(observation(prompt=p), req_context) for p in ("reach", "drawer")]
    stages = [diagnostic.run_stage2(s) for s in diagnostic.run_stage1_batch(ps)]
    results = diagnostic.sample_batch(stages, SamplerProfile(num_steps=2),
                                     [RequestNoise(18), RequestNoise(37)], [CaptureSpec()] * 2)
    assert all(r.actual_nfe == 2 and r.capture["batch_diagnostic_label"] == "CPU_explicit_batch_study" for r in results)


@pytest.mark.parametrize("label", ["", " ", None])
def test_batch_diagnostic_requires_explicit_label(make_adapter, label):
    with pytest.raises(ValueError, match="nonempty label"):
        make_adapter().for_batch_diagnostic(label)


def test_production_factory_rejects_B2_core_and_rechecks_hot_updates(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.common.config import build_shared_storage
    monkeypatch.setattr(registry, "_providers", {})
    adapter = make_adapter()
    provider = register_provider(adapter)
    config = serve.plain_config(provider)
    core = SimpleNamespace(_max_batch_size=2)
    with pytest.raises(ValueError, match="max_batch_size=1"):
        make_connection_factory(adapter, provider=provider, default_config=config,
                                default_storage=build_shared_storage(config), stage_core=core)
    with pytest.raises(ValueError, match="max_batch_size=1"):
        CoordinatedAdapter(adapter, core)
    core._max_batch_size = 1
    coordinated = CoordinatedAdapter(adapter, core)
    core._max_batch_size = 4
    with pytest.raises(ValueError, match="max_batch_size=1"):
        coordinated.run_stage1(None)  # must reject before calling the core


@pytest.mark.parametrize("size", [0, 2, 4])
def test_serve_rejects_non_B1_before_checkpoint_load(monkeypatch, size):
    monkeypatch.setattr(sys, "argv", ["dp.serve", "--load-spec", "/must/not/load.json",
                                     "--port", "8765", "--batch-size", str(size)])
    with pytest.raises(SystemExit) as exc:
        serve.main()
    assert exc.value.code == 2


def test_GPU_drift_repro_is_diagnostic_not_a_production_tolerance():
    one = torch.full((16, 7), .123, dtype=torch.float32)
    two = one.clone()
    two.view(-1)[:5] += 6.565451622009277e-5
    # The previous probe failed for this measured magnitude at small outputs.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(one, two, rtol=1e-4, atol=1e-5)
    result = SimpleNamespace(normalized_chunk=one, rng_receipt={"state": "same"})
    paired = SimpleNamespace(result=SimpleNamespace(normalized_chunk=two, rng_receipt=result.rng_receipt))
    report = probe.batch_comparison([result], [paired])
    assert report["status"] == "diagnostic_only" and report["production_batching"] == "unsupported"
    assert report["old_tolerance"]["mismatches"] == 5 and report["max_abs"] > 6.5e-5
    assert report["rng_states_exact"] is True
