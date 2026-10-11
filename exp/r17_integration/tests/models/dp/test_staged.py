from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from exp.mw_finetune.dp.preprocess import image_tensor, quantile_unnormalize
from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture, NativeSnapshot
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.dp.manifest import code_identity
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.models.dp.wire_policy import WirePolicy
from exp.r17_integration.tests.models.dp.conftest import observation


@pytest.mark.parametrize("domain", ["libero", "metaworld"])
@pytest.mark.parametrize("steps", [1, 2, 100])
def test_staged_vs_reference_native_loop_and_exact_two_inverse(make_adapter, req_context, domain, steps):
    adapter = make_adapter(domain)
    p = adapter.prepare(observation(domain), req_context)
    with torch.inference_mode():
        if steps == 100:
            native = adapter.model.sample(p.data["inputs"], generator=torch.Generator().manual_seed(18), steps=steps)[0]
        else:
            from diffusers import DDIMScheduler
            original = adapter.model.scheduler()
            scheduler = DDIMScheduler.from_config(original.config, timestep_spacing="trailing")
            scheduler.set_timesteps(steps)
            g = torch.Generator().manual_seed(18)
            x = torch.randn((1,16,adapter.model.cfg["action_dim"]), generator=g)
            condition = adapter.model.condition(p.data["inputs"])
            for t in scheduler.timesteps:
                x = scheduler.step(adapter.model._policy.model(x,t,condition), t, x, generator=g).prev_sample
            native = adapter.model._policy.normalizer.unnormalize("action", x)[0]
    expected = quantile_unnormalize(native.numpy(), adapter.normalization.stats["actions"]).astype(np.float32)
    s1 = adapter.run_stage1(p)
    s2 = adapter.run_stage2(s1)
    result = adapter.sample(s2, SamplerProfile(num_steps=steps), RequestNoise(18), CaptureSpec())
    got = adapter.to_wire(result.normalized_chunk, p).actions
    assert np.array_equal(got, expected)
    assert result.actual_nfe == steps
    assert result.normalized_chunk.shape == (16, 7 if domain == "libero" else 4)


@pytest.mark.parametrize("domain", ["libero", "metaworld"])
def test_visual_key_exact_post_relu_real_views_crop_and_condition_order(make_adapter, req_context, domain):
    adapter = make_adapter(domain)
    p = adapter.prepare(observation(domain), req_context)
    assert torch.equal(p.data["inputs"]["base_0_rgb"][0], image_tensor(observation(domain)[
        "observation/image" if domain == "libero" else "image"]))
    s1 = adapter.run_stage1(p)
    s2 = adapter.run_stage2(s1)
    with torch.inference_mode():
        assert torch.equal(s2.data["conditioning"], adapter.model.condition(p.data["inputs"]))
    fields = adapter.retrieval_fields(s1)
    assert set(fields) == ({"vision_0", "vision_1", "robot_state"} if domain == "libero" else {"vision_0", "robot_state"})
    for name in fields:
        assert fields[name].dtype == np.float32 and fields[name].flags.c_contiguous
    assert fields["vision_0"].shape == (64,) and np.all(fields["vision_0"] >= 0)
    assert np.array_equal(fields["vision_0"], s1.data["features"]["base_0_rgb"][0].numpy())
    if domain == "metaworld":
        assert not torch.count_nonzero(p.data["inputs"]["left_wrist_0_rgb"])
        changed = adapter.run_stage1(adapter.prepare(observation(domain, "close the drawer"), req_context))
        assert not torch.equal(s1.data["features"]["left_wrist_0_rgb"], changed.data["features"]["left_wrist_0_rgb"])
        assert "vision_1" not in fields


@pytest.mark.parametrize("kind,eta", [("auto", 0.0), ("ddim", 0.5)])
def test_suffix_resume_exact_and_capture_does_not_change_stream(make_adapter, req_context, kind, eta):
    adapter = make_adapter()
    p = adapter.prepare(observation(), req_context)
    s1 = adapter.run_stage1(p)
    s2 = adapter.run_stage2(s1)
    sampler = SamplerProfile(kind, 8, eta).to_spec(adapter.model.cfg)
    counts = {"vision": 0, "head": 0}
    handles = [net.register_forward_hook(lambda *args: counts.__setitem__("vision", counts["vision"]+1))
               for net in adapter.model._policy.obs_encoder.obs_nets.values()]
    handles.append(adapter.model._policy.model.register_forward_hook(lambda *args: counts.__setitem__("head", counts["head"]+1)))
    full = adapter.sample(s2, sampler, RequestNoise(72), CaptureSpec(True, True))
    assert counts == {"vision": 0, "head": 8}
    plain = adapter.sample(s2, sampler, RequestNoise(72), CaptureSpec())
    assert torch.equal(full.normalized_chunk, plain.normalized_chunk)
    assert full.rng_receipt == plain.rng_receipt
    assert len(full.snapshots) == 7
    for snapshot in (full.snapshots[0], full.snapshots[3], full.snapshots[-1]):
        resumed = adapter.resume(s2, snapshot, sampler, RequestNoise(72), CaptureSpec())
        assert torch.equal(resumed.normalized_chunk, full.normalized_chunk)
        assert resumed.actual_nfe == 8-snapshot.loop_index
        assert resumed.rng_receipt["state_digest"] == full.rng_receipt["state_digest"]
    before = dict(counts)
    trace = adapter.make_trace_adapter().from_stages(p, s1, full, adapter.to_wire(full.normalized_chunk, p))
    assert counts == before and trace["arrays"]["native_grid"].dtype == np.int64
    for h in handles:
        h.remove()


@pytest.mark.parametrize("field,value", [("native_timestep", 500), ("sampler_identity", "wrong"),
                                        ("norm_identity", "wrong"), ("model_identity", "wrong"),
                                        ("layout_identity", "wrong"), ("schedule_digest", "wrong"),
                                        ("conditioning_scope", "wrong"), ("loop_index", 0),
                                        ("noise_state", None)])
def test_resume_rejects_wrong_native_snapshot(make_adapter, req_context, field, value):
    adapter = make_adapter()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    sampler = SamplerProfile(num_steps=2).to_spec(adapter.model.cfg)
    result = adapter.sample(s2, sampler, RequestNoise(1), CaptureSpec(True, True))
    with pytest.raises(ValueError):
        adapter.resume(s2, replace(result.snapshots[0], **{field: value}), sampler, RequestNoise(1), CaptureSpec())


def test_snapshot_numeric_roundtrip_can_resume_without_pickle(make_adapter, req_context, tmp_path):
    adapter = make_adapter()
    p = adapter.prepare(observation(), req_context)
    s1 = adapter.run_stage1(p)
    s2 = adapter.run_stage2(s1)
    result = adapter.sample(s2, SamplerProfile(num_steps=2), RequestNoise(13), CaptureSpec(True, True))
    trace = adapter.make_trace_adapter()
    captured = trace.from_stages(p, s1, result, adapter.to_wire(result.normalized_chunk, p))
    trace.save(captured, tmp_path)
    meta = json.loads((tmp_path / "capture.json").read_text())
    with np.load(tmp_path / "capture.npz", allow_pickle=False) as saved:
        snap = NativeSnapshot(**meta["snapshots"][0], x_t=torch.from_numpy(saved["snapshot_x"][0]),
                              noise_state=torch.from_numpy(saved["snapshot_rng_state"][0]))
        for key, arr in captured["arrays"].items():
            assert np.array_equal(arr, saved[key])
    resumed = adapter.resume(s2, snap, result.sampler, RequestNoise(13), CaptureSpec())
    assert torch.equal(resumed.normalized_chunk, result.normalized_chunk)


def test_warm_reset_uses_native_add_noise_and_full_suffix(make_adapter, req_context):
    adapter = make_adapter()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    sampler = SamplerProfile(num_steps=4)
    clean = torch.linspace(-0.8, 0.9, 64).reshape(16, 4)
    g = torch.Generator().manual_seed(15)
    z = torch.randn((1, 16, 4), generator=g)
    sch = sampler.scheduler(adapter.model.cfg)
    expected = sch.add_noise(clean[None], z, sch.timesteps[2:3])
    for t in sch.timesteps[2:]:
        with torch.inference_mode():
            expected = sch.step(adapter.model._policy.model(expected, t, s2.data["conditioning"]), t,
                                expected, generator=g).prev_sample
    got = adapter.warm_reset(s2, clean, loop_index=2, sampler=sampler, rng=RequestNoise(15), capture=CaptureSpec(True))
    assert torch.equal(got.capture["first_step_input"], sch.add_noise(clean[None], z, sch.timesteps[2:3])[0])
    assert torch.equal(got.normalized_chunk, expected[0]) and got.actual_nfe == 2


def test_request_rng_pairing_and_order_are_independent(make_adapter, req_context):
    adapter = make_adapter()
    variants = [req_context, replace(req_context, arm="other", attempt=99, bundle_generation="other", task_id=42)]
    for r in variants:
        assert RequestRNG(r).seed_for() == RequestRNG(req_context).seed_for()
    assert RequestRNG(replace(req_context, decision_seq=1)).seed_for() != RequestRNG(req_context).seed_for()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    a = adapter.sample(s2, SamplerProfile(num_steps=2), RequestRNG(req_context), CaptureSpec())
    torch.manual_seed(999)
    torch.randn(64)
    b = adapter.sample(s2, SamplerProfile(num_steps=2), RequestRNG(variants[-1]), CaptureSpec())
    assert torch.equal(a.normalized_chunk, b.normalized_chunk)


def test_validation_norm_identity_and_wire_seedless_default(make_adapter, req_context):
    adapter = make_adapter()
    stats = adapter.normalization.stats
    from exp.r17_integration.models.dp.normalization import Normalization
    changed = json.loads(json.dumps(stats))
    changed["actions"]["q99"][0] += 0.1
    assert Normalization(adapter.model, changed).identity != adapter.metadata().norm_identity
    changed["state"]["q01"] = [0]
    with pytest.raises(ValueError, match="external"):
        Normalization(adapter.model, changed)
    bad = observation()
    bad["state"][0] = np.nan
    with pytest.raises(ValueError, match="state"):
        adapter.prepare(bad, req_context)
    policy = WirePolicy(adapter)
    first = policy.infer(observation())["actions"]
    torch.manual_seed(789)
    policy.reset()
    assert np.array_equal(first, policy.infer(observation())["actions"])


def test_default_off_import_no_torch_jax_registry_or_E_edits():
    before = code_identity()
    code = "import sys; import exp.r17_integration.models.dp; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules; assert 'exp.r17_integration.common.provider' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
    after = code_identity()
    assert before == after


def test_cpu_blueprint_has_no_tensor_vendor_model(make_adapter):
    from dataclasses import asdict
    blueprint = make_adapter().cpu_blueprint()
    json.dumps(asdict(blueprint), allow_nan=False)
    assert blueprint.configuration["split_qualified"] is False


def test_explicit_trace_variants_match_initial_noise_and_account_extra_NFE(make_adapter, req_context):
    adapter = make_adapter()
    s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(), req_context)))
    calls = []
    hook = adapter.model._policy.model.register_forward_hook(lambda *args: calls.append(1))
    rng = RequestNoise(65)
    before = rng.receipt()
    profiles = [SamplerProfile(num_steps=2), SamplerProfile(num_steps=4), SamplerProfile("ddim", 2, .5)]
    variants = adapter.make_trace_adapter().sample_variants(s2, profiles, rng, CaptureSpec(True, True))
    assert len(calls) == sum(result.actual_nfe for result in variants.values()) == 8
    assert rng.receipt() == before
    inputs = [r.capture["first_step_input"] for r in variants.values()]
    assert all(torch.equal(inputs[0], x) for x in inputs[1:])
    for profile, result in zip(profiles, variants.values()):
        single = adapter.sample(s2, profile, RequestNoise(65), CaptureSpec())
        assert torch.equal(result.normalized_chunk, single.normalized_chunk)
    hook.remove()
