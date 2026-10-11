from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from openpi.serving.batching_core import BatchingCore
from exp.r17_integration.tests.models.dp.conftest import observation


@pytest.mark.parametrize("kind,eta", [("auto", 0), ("ddim", .5)])
def test_B2_noise_order_capture_and_no_alias(make_adapter, req_context, kind, eta, record_property):
    adapter = make_adapter().for_batch_diagnostic("CPU_B2_math_rng_alias_study")
    b = adapter.make_stage_batcher()
    ps = [adapter.prepare(observation(prompt=p), req_context) for p in ("reach", "drawer")]
    singles = [adapter.run_stage1(p) for p in ps]
    batched = b.run_stage1_batch(ps)
    for one, two in zip(singles, batched, strict=True):
        for name in one.fields:
            np.testing.assert_allclose(one.fields[name], two.fields[name], rtol=1e-5, atol=2e-6)
    s2s = b.run_stage2_batch(batched, capture=True)
    profile = SamplerProfile(kind, 4, eta)
    seed = [18, 37]
    single = [adapter.sample(s2, profile, RequestNoise(k), CaptureSpec(True, True)) for s2, k in zip(s2s, seed)]
    payloads = [b.miss_payload(s2, profile, RequestNoise(k), CaptureSpec(True, True)) for s2, k in zip(s2s, seed)]
    paired = b.run_stage3_miss(payloads, num_steps=4, save_timesteps_per_request=[None, None])
    record_property("B1_B2_max_abs", max(float((one.normalized_chunk-two.result.normalized_chunk).abs().max())
                                       for one, two in zip(single, paired)))
    for one, two in zip(single, paired):
        torch.testing.assert_close(one.normalized_chunk, two.result.normalized_chunk, rtol=2e-5, atol=3e-6)
        assert one.rng_receipt == two.result.rng_receipt and two.actual_nfe == 4
    reversed_payloads = [b.miss_payload(s2s[i], profile, RequestNoise(seed[i]), CaptureSpec(True, True)) for i in (1, 0)]
    reverse = b.run_stage3_miss(reversed_payloads, num_steps=4, save_timesteps_per_request=[None, None])
    record_property("reversed_B2_max_abs", max(float((paired[i].action-other.action).abs().max())
                                             for i, other in enumerate(reversed(reverse))))
    for i, other in enumerate(reversed(reverse)):
        torch.testing.assert_close(paired[i].action, other.action, rtol=2e-5, atol=3e-6)
        assert paired[i].result.rng_receipt == other.result.rng_receipt
    assert paired[0].action.data_ptr() != paired[1].action.data_ptr()
    assert paired[0].result.snapshots[0].x_t.data_ptr() != paired[1].result.snapshots[0].x_t.data_ptr()
    before = paired[1].action.clone()
    with torch.inference_mode():
        paired[0].action.zero_()
    assert torch.equal(paired[1].action, before)


def test_stock_BatchingCore_async_resume_and_fault_buckets(make_adapter, req_context):
    adapter = make_adapter()
    b = adapter.make_stage_batcher()
    with BatchingCore(b, device="cpu", max_batch_size=1, max_wait_ms=8) as core:
        prepared = [adapter.prepare(observation(prompt=p), req_context) for p in ("reach", "drawer")]
        s1 = core.submit_many_to_stage(1, "p4", prepared, timeout=20)
        s2 = core.submit_many_to_stage(2, "p4", s1, timeout=20)
        profile = SamplerProfile(num_steps=4)
        payloads = [b.miss_payload(s, profile, RequestNoise(i), CaptureSpec(True, True)) for i, s in enumerate(s2)]
        results = core.submit_many_to_stage(3, "p4", payloads, timeout=20)
        warm = [b.warm_payload(s, r.result.snapshots[1], profile, RequestNoise(i), CaptureSpec())
                for i, (s, r) in enumerate(zip(s2, results))]
        suffix = core.submit_many_to_stage(3, "p4", warm, timeout=20)
        for r, resumed in zip(results, suffix):
            assert torch.equal(r.action, resumed.action) and resumed.actual_nfe == 2
        bad = b.miss_payload(s2[0], profile, RequestNoise(4), CaptureSpec())
        bad.noise = torch.zeros(2, 2)  # separate invalid-shape bucket
        valid = b.miss_payload(s2[1], profile, RequestNoise(3), CaptureSpec())
        reqs = [core.enqueue_to_stage(3, "p4", p) for p in (bad, valid)]
        with pytest.raises(ValueError, match="noise"):
            core.wait_for(reqs[0], timeout=20)
        assert core.wait_for(reqs[1], timeout=20).actual_nfe == 4
        assert reqs[0].reply_event.is_set() and reqs[1].reply_event.is_set()


def test_buckets_separate_K_sampler_shape_capture_native_point(make_adapter, req_context):
    adapter = make_adapter()
    b = adapter.make_stage_batcher()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    variants = [(SamplerProfile(num_steps=2), CaptureSpec()), (SamplerProfile(num_steps=4), CaptureSpec()),
                (SamplerProfile("ddim", 2, .5), CaptureSpec()), (SamplerProfile(num_steps=2), CaptureSpec(True))]
    keys = [b.bucket_key(b.miss_payload(s2, profile, RequestNoise(1), capture)) for profile, capture in variants]
    assert len(set(keys)) == 4
    full = adapter.sample(s2, SamplerProfile(num_steps=4), RequestNoise(1), CaptureSpec(True, True))
    warm = [b.bucket_key(b.warm_payload(s2, snap, full.sampler, RequestNoise(1), CaptureSpec())) for snap in full.snapshots]
    assert len(set(warm)) == 3
