"""Regressions for Opus review B4/B5/B6/B7/B9/B10 and the shared seams."""
from dataclasses import asdict, replace
import json
import subprocess
import sys
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.fastwam.measure import analytical_split, measure_adapter
from exp.r17_integration.models.fastwam.protocols import execution_protocol
from exp.r17_integration.models.fastwam.provider import register_provider, make_connection_factory
from exp.r17_integration.models.fastwam.serve import serve_metadata
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, native_defs, observation, request
from exp.r17_integration.tests.models.fastwam.test_fastwam_serving import config_for


@pytest.mark.parametrize("benchmark", ["libero", "metaworld"])
def test_b4_key_does_not_encode_text_or_pre_dit_and_miss_reuses_patch(benchmark, monkeypatch):
    a = make_adapter(benchmark)
    obs = observation(benchmark)
    original = a.model._prepare_infer_context
    def forbidden(*args, **kwargs):
        pytest.fail("key/HIT path must not invoke UMT5")
    monkeypatch.setattr(a.model, "_prepare_infer_context", forbidden)
    p = a.prepare(obs, request())
    s1 = a.run_stage1(p)
    s_other = a.run_stage1(a.prepare({**obs, "prompt": "different prompt", "state": obs["state"]+.1}, request(1)))
    assert s1.fields["vision_0"].tobytes() == s_other.fields["vision_0"].tobytes()
    assert a.model.calls["vae"] == [1,1] and a.model.calls["patch"] == [1,1]
    assert not a.model.calls["pre"] and not a.model.calls["context"]
    assert "context" not in s1.data and "attention_mask" not in s1.data
    monkeypatch.setattr(a.model, "_prepare_infer_context", original)
    s2 = a.run_stage2(s1)
    assert a.model.calls["context"] == [1] and a.model.calls["pre"] == [1]
    assert a.model.calls["patch"] == [1,1]  # native pre_dit reused stage1 convolution
    assert a.model.calls["prefill"] == [1]
    expected_context, _ = original([p.data["prompt"]], None, None, p.data["proprio"])
    torch.testing.assert_close(s2.data["context"], expected_context, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("benchmark", ["libero","metaworld"])
def test_b4_unchanged_native_pre_dit_reuses_identical_conv_tokens(benchmark,dtype,monkeypatch):
    from einops import rearrange
    from exp.r17_integration.common.serving import full_call
    from exp.r17_integration.tests.models.fastwam.conftest import ToyModel
    native = native_defs("pre_dit_native.py.txt", {"rearrange":rearrange})
    # Run unchanged vendor pre_dit against tiny real torch modules. Keep CUDA
    # autocast disabled explicitly on this CPU-only fixture (native fp32 time).
    monkeypatch.setattr(torch.amp,"autocast",lambda *args,**kwargs:nullcontext())
    cls = type("TinyNativeVideo",(),{k:native[k] for k in ("patchify","_validate_forward_inputs","pre_dit")})
    video = cls()
    video.patch_embedding = torch.nn.Conv3d(3,8,(1,2,2),stride=(1,2,2)).to(dtype)
    video.time_embedding = torch.nn.Linear(8,8)
    video.time_projection = torch.nn.Linear(8,48)
    video.text_embedding = torch.nn.Linear(8,8).to(dtype)
    video.patch_size = (1,2,2)
    video.hidden_dim = video.freq_dim = 8
    video.seperated_timestep = video.fuse_vae_embedding_in_latents = video.action_conditioned = True
    video.video_attention_mask_mode = "first_frame_causal"
    video.training = False
    video.freqs = torch.arange(5)
    a = make_adapter(benchmark)
    a.model = ToyModel(a.metadata().layout.action_width,len(a.metadata().layout.valid_state_indices),dtype=dtype)
    a.model.video_expert = video
    p = a.prepare(observation(benchmark),request())
    calls=[]
    hook=video.patch_embedding.register_forward_hook(lambda *args:calls.append(1))
    expected = a.model.infer_action(prompt=p.data["prompt"],input_image=p.data["input_image"],
        proprio=p.data["proprio"],action_horizon=32,num_inference_steps=10,seed=RequestRNG(request()).seed_for(),
        rand_device="cpu",compile_action_infer=False)["action"]
    actual, _ = full_call(a,observation(benchmark),request(),sampler=a.sampler_for_steps(10))
    torch.testing.assert_close(actual.normalized_chunk,expected,rtol=0,atol=0)
    assert calls == [1,1]  # one native full convolution, one staged convolution
    hook.remove()


@pytest.mark.parametrize("camera", ["image", "image2"])
@pytest.mark.parametrize("size", [224,128,480])
def test_b5_libero_rejects_client_resized_or_wrong_camera_resolution(camera, size):
    a = make_adapter("libero")
    obs = observation("libero")
    obs[camera] = np.zeros((size,size,3),np.uint8)
    with pytest.raises(ValueError,match="original 256x256"):
        a.prepare(obs,request())


def test_b6_released_protocol_is_explicit_and_changes_layout_seed_and_binding(tmp_path):
    a = make_adapter("libero", protocol="released_libero")
    system = make_adapter("libero")
    p = execution_protocol("libero", "released_libero")
    assert p.label == "fastwam.libero.released_protocol.v1"
    assert p.execution_steps == 32 and p.fixed_seed == 42
    assert dict(p.suite_control_caps) == {"libero_spatial":280,"libero_object":280,"libero_goal":300,"libero_10":520}
    assert p.wire_resolution == (256,256) and p.identity != system.wire.protocol.identity
    assert a.metadata().identity != system.metadata().identity
    assert a.metadata().feature_spec.identity != system.metadata().feature_spec.identity
    assert a.metadata().layout.replan_steps == 32
    a.metadata().layout.validate_commit(32)
    with pytest.raises(ValueError):
        a.metadata().layout.validate_commit(5)
    with pytest.raises(ValueError):
        system.metadata().layout.validate_commit(32)
    with pytest.raises(ValueError,match="explicit_seed=42"):
        a.prepare(observation("libero"),request(explicit_seed=43))
    with pytest.raises(ValueError,match="K10"):
        a.sampler_for_steps(2)
    provider = register_provider(a,provider_id="p3_released_"+tmp_path.name)
    from exp.r17_integration.common.provider import binding_for
    binding = binding_for(provider,a.metadata().allowed_benchmark_protocols[0])
    assert binding["commit_horizon"] == 32 and binding["num_steps"] == 10
    from exp.r17_integration.common.serving import full_call
    result, wire = full_call(a,observation("libero"),request(explicit_seed=42))
    assert wire.actions.shape == (32,7) and result.actual_nfe == 10
    assert a.cpu_blueprint().configuration["execution_protocol"] == asdict(p)


@pytest.mark.parametrize("profile", ["system","released_libero"])
def test_b6_p2_original_libero_kernel_uses_profile_execution_and_seeds(profile,tmp_path):
    # P2's fixture compiles the unchanged original LIBERO loop; its env is CPU
    # only. Drive the actual P1 policy and P3 model hooks, not a controller oracle.
    from exp.r17_integration.tests.benchmarks.test_libero import spec,args,task,Env,native_kernel
    from exp.r17_integration.benchmarks.libero_bridge import R17LiberoEpisodeRunner,decision_seed
    from exp.r17_integration.common.config import build_shared_storage
    protocol = spec()
    if profile == "released_libero":
        protocol = replace(protocol,seed=42,seed_mode="fixed",replan_steps=32,
            max_steps_by_suite=(280,280,300,520),protocol_id="libero_fastwam_released_reference_v1")
    from exp.r17_integration.models.fastwam.loader import LoadSpec
    from exp.r17_integration.models.fastwam.protocols import validate_benchmark_specs
    validate_benchmark_specs(LoadSpec("libero","unused_immutable","unused",execution_protocol=profile,
        benchmark_protocols=(protocol.identity,),benchmark_specs=(protocol.manifest(),)))
    a = make_adapter("libero",protocol=profile,allow_unqualified_batch=False)
    a.manifest = replace(a.manifest,spec=replace(a.metadata(),allowed_benchmark_protocols=(protocol.identity,)))
    p = register_provider(a,provider_id="p3_profile_kernel_"+profile+tmp_path.name)
    cfg = config_for(p,k=10,commit=32 if profile=="released_libero" else 10)
    cfg.diagnostic_context = False
    records=[]
    policy = make_connection_factory(p,cfg,build_shared_storage(cfg),capture_sink=records.append)()
    class Client:
        def select_bundle(self,bundle):
            pass
        def episode_start(self,**kwargs):
            policy.on_episode_start(**kwargs)
        def infer(self,obs):
            return policy.infer(obs)
        def episode_end(self,**kwargs):
            policy.on_episode_end(**kwargs)
        def close(self):
            pass
    t = task(protocol)
    env = Env(done_at=34)
    runner = R17LiberoEpisodeRunner(args(protocol),lambda _: (env,np.array([7.,8.]),protocol.tasks[t.task_id][1],280),
        spec=protocol,client_factory=lambda _:Client(),run_episode_fn=native_kernel())
    result = runner.run(t,lambda *args:None)
    assert result.success and result.n_steps == 34
    acks = [row for row in result.per_step_rows if row.get("_kind")=="execution_ack"]
    if profile == "released_libero":
        assert [row["executed_count"] for row in acks] == [32,2]
        assert [r.request.explicit_seed for r in records] == [42,42]
        assert [r.stage_calls for r in records] == [(1,1,10)]*2
    else:
        assert [row["executed_count"] for row in acks] == [5]*6+[4]
        assert any(r.stage_calls==(0,0,0) for r in records)
        assert [r.request.explicit_seed for r in records] == [decision_seed(protocol.tasks[t.task_id][0],7,i) for i in range(7)]
    assert all(r.request.original_init_idx==7 and r.request.task_id==13 for r in records)
    assert sum(row["executed_count"] for row in acks) == 34
    assert policy._awaiting is None and policy._chunk is None
    runner.close()


def test_b7_advanced_and_batch_capabilities_fail_closed():
    caps = make_adapter().metadata().capabilities
    for name in ("cp2","native_warm_resume","native_warm_reset"):
        assert caps.features[name].status == "implemented_unverified"
        with pytest.raises(ValueError):
            caps.require(name)
    for name in ("batch_stage1","batch_stage2","batch_stage3"):
        assert caps.features[name].status == "unsupported"


def test_b9_ordinary_capture_skips_cp2_and_marks_trace_unpriceable():
    a = make_adapter()
    s1 = a.run_stage1(a.prepare(observation(),request()))
    ordinary = a.run_stage2(s1,capture=True)
    assert "cp2_fields" not in ordinary.data
    explicit = a.run_stage2(s1,capture_cp2=True)
    assert explicit.data["cp2_fields"]["vlm_out"].size == 32
    torch.testing.assert_close(ordinary.data["video_kv_cache"][0]["k"],explicit.data["video_kv_cache"][0]["k"],rtol=0,atol=0)
    off = a.sample(ordinary,a.sampler_for_steps(2),RequestRNG(request()))
    on = a.sample(ordinary,a.sampler_for_steps(2),RequestRNG(request()),CaptureSpec(True,True))
    torch.testing.assert_close(off.normalized_chunk,on.normalized_chunk,rtol=0,atol=0)
    assert off.capture["pricing_eligible"] and not on.capture["pricing_eligible"]
    cp2 = a.sample(explicit,a.sampler_for_steps(2),RequestRNG(request()))
    assert not cp2.capture["pricing_eligible"]
    assert not a.make_trace_adapter().sample_fields(s1,ordinary,on)["pricing_eligible"]


def test_b10_production_adapter_and_coordinator_reject_batch2(tmp_path):
    from openpi.serving.batching_core import Stage3MissPayload
    a = make_adapter(allow_unqualified_batch=False)
    ps = [a.prepare(observation(variant=i),request(i)) for i in range(2)]
    with pytest.raises(ValueError,match="B>1 disabled"):
        a.run_stage1_batch(ps)
    batcher = a.make_stage_batcher()
    with pytest.raises(ValueError,match="B>1 disabled"):
        batcher.run_stage1_batch(ps)
    s1s = [a.run_stage1(p) for p in ps]
    with pytest.raises(ValueError,match="B>1 disabled"):
        a.run_stage2_batch(s1s)
    with pytest.raises(ValueError,match="B>1 disabled"):
        batcher.run_stage2_batch(s1s)
    s2s = [a.run_stage2(s) for s in s1s]
    payloads = [Stage3MissPayload(s,torch.zeros(32,4),2) for s in s2s]
    with pytest.raises(ValueError,match="B>1 disabled"):
        batcher.run_stage3_miss(payloads,num_steps=2,save_timesteps_per_request=[None,None])
    with pytest.raises(ValueError,match="coordinator disabled"):
        register_provider(a,provider_id="p3_unsafe_"+tmp_path.name,coordinator=SimpleNamespace())


def test_b10_serve_rejects_batch2_before_loading_weights(tmp_path):
    desc = tmp_path/"descriptor.json"
    desc.write_text(json.dumps({"benchmark":"libero","checkpoint":"unused_immutable","wan_root":"unused"}))
    proc = subprocess.run([sys.executable,"-m","exp.r17_integration.models.fastwam.serve",
        "--descriptor",str(desc),"--cache-config","unused","--port","9009","--max-batch-size","2"],
        capture_output=True,text=True,timeout=15)
    assert proc.returncode == 2 and "requires --max-batch-size=1" in proc.stderr
    assert "content-sealed" not in proc.stderr


@pytest.mark.parametrize("benchmark", ["libero","metaworld"])
def test_b4_measurement_exposes_key_full_cost_split_without_capture(benchmark):
    a = make_adapter(benchmark,allow_unqualified_batch=False)
    receipt = measure_adapter(a,[observation(benchmark)],repeats=2)
    assert receipt["status"] == "passed" and len(receipt["rows"]) == 2
    assert a.phase_observer is None
    for row in receipt["rows"]:
        assert set(row["key_phases_ms"]) == {"vae","patchify","key_pool"}
        assert "text_context" in row["full_phases_ms"]
        assert row["full_phases_ms"]["text_context"] > 0
        assert row["actual_nfe"] == 10 and 0 < row["key_over_full"] < 1
        assert row["pricing_eligible"] and not row["capture_enabled"]
    assert a.model.calls["context"] == [1]*3 and a.model.calls["patch"] == [1]*3
    estimate = analytical_split(benchmark)
    assert estimate["umt5_tflop_estimate"] == pytest.approx(1.1776)
    assert estimate["vae_patchify_tflop"] is None and estimate["measured_key_full_ratio"] is None


def test_a5_serve_metadata_matches_worker_modelspec_contract():
    a = make_adapter()
    metadata = serve_metadata(a)
    assert metadata["r17_model_spec"] == asdict(a.metadata())
    assert metadata["r17_execution_protocol"]["wire_resolution"] == (480,480)
    assert "unverified" in metadata["loader_parity"]


def test_b1_b2_reviewer_libero_metadata_repro_fails_closed(tmp_path):
    from exp.r17_integration.common.config import build_shared_storage
    a = make_adapter("libero")
    provider = register_provider(a,provider_id="p3_repro_"+tmp_path.name)
    cfg = config_for(provider)
    cfg.diagnostic_context = False
    c = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    with pytest.raises(ValueError,match="ack_protocol"):
        c.on_episode_start(experiment="libero_spatial",task="pick up the black bowl",episode_id=7,
            extra_metadata={"task_id":3,"orig_init_state_idx":7})
    cfg5 = config_for(provider,commit=5)
    cfg5.diagnostic_context = False
    c5 = make_connection_factory(provider,cfg5,build_shared_storage(cfg5))()
    c5.on_episode_start(experiment="libero_spatial",task="pick up the black bowl",episode_id=7,
        extra_metadata={"task_id":3,"orig_init_state_idx":7})
    assert c5._episode["original_init_idx"] == 7
    with pytest.raises(ValueError,match="missing R17 request context"):
        c5.infer(observation("libero"))
