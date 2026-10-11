"""Review B round-3 regressions for Fast-WAM (reviewer-owned fixes)."""
import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.fastwam import staged
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation, request


def _counting_digest(monkeypatch):
    calls = []
    original = staged._scope_digest
    def counted(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(staged, "_scope_digest", counted)
    return calls


@pytest.mark.parametrize("benchmark", ["libero", "metaworld"])
def test_b3_key_and_capture_off_miss_never_hash_conditioning_inputs(benchmark, monkeypatch):
    """The N4 scope hash (ms per request) must not be paid on HIT/key or capture-off MISS."""
    a = make_adapter(benchmark, allow_unqualified_batch=False)
    calls = _counting_digest(monkeypatch)
    req = request(explicit_seed=42)
    prepared = a.prepare(observation(benchmark), req)
    s1 = a.run_stage1(prepared)
    a.retrieval_fields(s1)                                    # HIT/key path ends here
    assert calls == []
    s2 = a.run_stage2(s1)
    result = a.sample(s2, a.sampler_for_steps(10), RequestRNG(req), CaptureSpec())
    assert calls == [] and result.snapshots == ()


@pytest.mark.parametrize("benchmark", ["libero", "metaworld"])
def test_b3_lazy_scope_is_computed_once_and_equals_eager_scope(benchmark, monkeypatch):
    a = make_adapter(benchmark, allow_unqualified_batch=False)
    req = request(explicit_seed=42)
    prepared = a.prepare(observation(benchmark), req)
    eager = staged.conditioning_scope(prepared)
    calls = _counting_digest(monkeypatch)
    s2 = a.run_stage2(a.run_stage1(prepared))
    sampler, rng = a.sampler_for_steps(10), RequestRNG(req)
    full = a.sample(s2, sampler, rng, CaptureSpec(True, True))
    assert len(calls) == 1 and len(full.snapshots) == 9
    assert {s.conditioning_scope for s in full.snapshots} == {eager}
    # Exact suffix resume on an independently prepared identical observation.
    same = a.run_stage2(a.run_stage1(a.prepare(observation(benchmark), req)))
    resumed = a.resume(same, full.snapshots[4], sampler, rng)
    torch.testing.assert_close(resumed.normalized_chunk, full.normalized_chunk, rtol=0, atol=0)
    assert len(calls) == 2
    # Changed state with the same prompt still fails closed before head calls.
    other_obs = observation(benchmark)
    other_obs["state"] = other_obs["state"] + np.float32(0.25)
    other = a.run_stage2(a.run_stage1(a.prepare(other_obs, req)))
    heads = len(a.model.calls["head"])
    with pytest.raises(ValueError, match="conditioning"):
        a.resume(other, full.snapshots[4], sampler, rng)
    assert len(a.model.calls["head"]) == heads


def test_b3_lazy_scope_rejects_gpu_prepared_inputs_at_stage1():
    a = make_adapter("metaworld", allow_unqualified_batch=False)
    prepared = a.prepare(observation("metaworld"), request(explicit_seed=42))
    prepared.data["input_image"] = prepared.data["input_image"].to("meta")
    with pytest.raises(ValueError, match="CPU-prepared"):
        a.run_stage1(prepared)
