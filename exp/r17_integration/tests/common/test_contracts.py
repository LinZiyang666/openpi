from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_sampler, fake_spec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call


def test_contract_import_has_no_model_side_effects():
    code = "import sys; import exp.r17_integration.common.contracts; assert not any(x in sys.modules for x in ['torch','jax','mujoco','transformers']); from exp.r17_integration.common.provider import _providers; assert not _providers"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_request_noise_threads_order_restart_and_paired_arms():
    r = RequestContext("campaign", "mt50.v1", "push-back-v3", 8, 12)
    records = [replace(r, arm=str(i), attempt=i, bundle_generation=f"v{i}") for i in range(8)]
    def draw(context):
        return RequestRNG(context).normal((16, 4))
    expected = draw(r)
    with ThreadPoolExecutor(4) as pool:
        for value in pool.map(draw, reversed(records)):
            np.testing.assert_array_equal(value, expected)
    assert not np.array_equal(draw(replace(r, decision_seq=13)), expected)
    assert not np.array_equal(draw(replace(r, task_name="push-v3")), expected)
    assert not np.array_equal(RequestRNG(r).posterior_noise((16, 4), 2), RequestRNG(r).posterior_noise((16, 4), 3))


def test_explicit_private_torch_stream_matches_original_draw_order():
    import torch
    r = RequestRNG(RequestContext("c", "b", "task", 1, 2, explicit_seed=123))
    got = r.torch_generator()
    original = torch.Generator().manual_seed(123)
    for _ in range(5):
        assert torch.equal(torch.randn((3, 4), generator=got), torch.randn((3, 4), generator=original))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_native_k_actual_calls_and_snapshot_suffix(k):
    a = FakeAdapter()
    r = RequestContext("c", "fake_mw.v1", "task", 0, 0)
    obs = {"state": np.arange(4, dtype=np.float32)}
    result, wire = full_call(a, obs, r, sampler=fake_sampler(k), capture=CaptureSpec(True, True))
    assert result.actual_nfe == k and a.counts["nfe"] == k
    if k > 1:
        p = a.prepare(obs, r)
        s2 = a.run_stage2(a.run_stage1(p))
        resumed = a.resume(s2, result.snapshots[0], result.sampler, RequestRNG(r), CaptureSpec())
        np.testing.assert_array_equal(resumed.normalized_chunk, result.normalized_chunk)
        assert resumed.actual_nfe == k - result.snapshots[0].loop_index
        with pytest.raises(ValueError, match="identity"):
            replace(result.snapshots[0], norm_identity="foreign").validate(a.spec, result.sampler)
        with pytest.raises(ValueError, match="timestep"):
            replace(result.snapshots[0], native_timestep=-9).validate(a.spec, result.sampler)
        with pytest.raises(ValueError, match="dtype"):
            replace(result.snapshots[0], x_t=result.snapshots[0].x_t.astype(np.float64)).validate(a.spec, result.sampler)


def test_native_schedule_registry_does_not_alias_legacy_flow():
    import uuid
    from exp.r17_integration.common.native import register_native_schedule_provider, native_schedule_provider
    from openpi.cache.types import schedule_from_id
    class Native:
        def sampler_for_steps(self,k):return fake_sampler(k)
        def validate_snapshot(self,snapshot,sampler):snapshot.validate(fake_spec(),sampler)
    name="native."+uuid.uuid4().hex
    p=Native();register_native_schedule_provider(name,p)
    assert native_schedule_provider(name) is p
    with pytest.raises(ValueError):schedule_from_id(name)
    with pytest.raises(ValueError):register_native_schedule_provider(name,Native())


def test_provider_metadata_mutation_fails_closed(provider):
    from exp.r17_integration.common.provider import get_provider
    provider.model_spec.feature_spec.fields["vision_0"]+=1
    with pytest.raises(ValueError,match="mutated"):get_provider(provider.provider_id)


def test_binding_constructor_rejects_explicit_zero_bool_and_invalid_horizon(provider):
    from exp.r17_integration.common.provider import binding_for
    for value in (0, False, -1):
        with pytest.raises(ValueError,match="positive integer"):
            binding_for(provider,"fake_mw.v1",num_steps=value)
    for value in (0, False, -1, provider.model_spec.layout.horizon+1):
        with pytest.raises(ValueError,match="commit"):
            binding_for(provider,"fake_mw.v1",commit_horizon=value)


@pytest.mark.parametrize("h,d,v", [(5,4,1), (16,4,1), (32,7,2)])
def test_layout_no_padding_and_commit_bound(h,d,v):
    spec = fake_spec(horizon=h, action_dim=d, views=v)
    spec.layout.validate_commit(h - h % spec.layout.replan_steps)
    with pytest.raises(ValueError):
        spec.layout.validate_commit(h+1)
    result, wire = full_call(FakeAdapter(spec), {"state": np.arange(d, dtype=np.float32)},
                             RequestContext("c", "b", "t", 0, 0))
    assert result.normalized_chunk.shape == wire.actions.shape == (h,d)


def test_legacy_positional_payload_and_missing_new_attributes():
    import torch
    from exp.r17_integration.common.native import CachePayload
    from openpi.cache.types import CheckpointID, PI05_V1
    payload = CachePayload(torch.ones(5,4), None, None, "task", None, None)
    del payload.sampler_state
    del payload.native_schedule_meta
    payload.validate_for_checkpoint(CheckpointID.CP1)
    assert getattr(payload, "native_schedule_meta", None) is None
    assert PI05_V1.remaining_steps(.7) == 7
