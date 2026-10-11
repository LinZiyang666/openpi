from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.fastwam.manifest import make_sampler
from exp.r17_integration.tests.models.fastwam.conftest import FIXTURES, Scheduler, make_adapter, native_defs, observation, request


@pytest.mark.parametrize("benchmark", ["libero", "metaworld"])
@pytest.mark.parametrize("k", [1,2,10])
def test_native_unsplit_staged_parity(benchmark,k):
    adapter = make_adapter(benchmark)
    obs, req = observation(benchmark), request()
    prep = adapter.prepare(obs,req)
    rng = RequestRNG(req)
    before = torch.random.get_rng_state().clone()
    # Actual vendor infer_action reference source, explicit LeRobot K (raw default20).
    original = adapter.model.infer_action(prompt=prep.data["prompt"], input_image=prep.data["input_image"],
        action_horizon=32, proprio=prep.data["proprio"], num_inference_steps=k,
        seed=rng.seed_for(), rand_device="cpu", compile_action_infer=False)["action"]
    result,wire = full_call(adapter,obs,req,sampler=adapter.sampler_for_steps(k))
    torch.testing.assert_close(result.normalized_chunk,original,rtol=0,atol=0)
    assert wire.actions.tobytes() == adapter.to_wire(original,prep).actions.tobytes()
    assert result.actual_nfe == k
    assert torch.equal(before,torch.random.get_rng_state())
    assert adapter.model.calls["vae"] == [1,1]
    assert adapter.model.calls["prefill"] == [1,1]
    assert adapter.metadata().default_sampler.num_steps == 10


@pytest.mark.parametrize("k", [1,2,10])
@pytest.mark.parametrize("dtype", ["float32","bfloat16"])
def test_actual_native_grid(k,dtype):
    scheduler = Scheduler(1000,5)
    spec = make_sampler(scheduler,k,dtype=dtype)
    t,d = scheduler.build_inference_schedule(k,torch.device("cpu"),getattr(torch,dtype))
    assert spec.native_timesteps == tuple(t.float().tolist())
    assert spec.native_deltas == tuple(d.float().tolist())
    assert spec.native_timesteps[0] == 1000
    if k == 2:
        assert spec.native_timesteps[1] > 800  # native shifted grid, never uniform500
    x = torch.arange(8,dtype=getattr(torch,dtype)).reshape(2,4)
    z = -x
    torch.testing.assert_close(scheduler.add_noise(x,z,t[-1]),
        (1-t[-1]/1000)*x+(t[-1]/1000)*z,rtol=0,atol=0)
    assert spec.config["cfg"] == 1


@pytest.mark.parametrize("benchmark", ["libero","metaworld"])
def test_snapshot_suffix_exact_and_identity_rejection(benchmark):
    a = make_adapter(benchmark)
    p=a.prepare(observation(benchmark),request())
    s1=a.run_stage1(p); s2=a.run_stage2(s1,capture=True)
    rng=RequestRNG(request()); sampler=a.sampler_for_steps(10)
    result=a.sample(s2,sampler,rng,CaptureSpec(True,True))
    assert len(result.snapshots)==9
    for snapshot in result.snapshots:
        resumed=a.resume(s2,snapshot,sampler,rng)
        torch.testing.assert_close(resumed.normalized_chunk,result.normalized_chunk,rtol=0,atol=0)
        assert resumed.actual_nfe == 10-snapshot.loop_index
    snap=result.snapshots[4]
    for change in ({"norm_identity":"wrong"},{"sampler_identity":"wrong"},
        {"schedule_digest":"wrong"},{"native_timestep":123},{"loop_index":0},
        {"conditioning_scope":"wrong"},{"x_t":snap.x_t.double()},
        {"x_t":torch.full_like(snap.x_t,float("nan"))}):
        with pytest.raises(ValueError):
            a.resume(s2,replace(snap,**change),sampler,rng)
    with pytest.raises(ValueError):
        a.resume(s2,snap,a.sampler_for_steps(2),rng)
    warm=a.reset_snapshot(s2,result.normalized_chunk,torch.ones_like(result.normalized_chunk),sampler,5)
    expected=a.model.infer_action_scheduler.add_noise(result.normalized_chunk,
        torch.ones_like(result.normalized_chunk),torch.tensor(sampler.native_timesteps[5]))
    torch.testing.assert_close(warm.x_t,expected,rtol=0,atol=0)
    assert a.resume(s2,warm,sampler,rng).actual_nfe==5
    with pytest.raises(ValueError):
        a.reset_snapshot(s2,result.normalized_chunk,torch.ones_like(result.normalized_chunk),a.sampler_for_steps(1),0)


def test_capture_has_no_forward_or_rng_and_native_fixture(adapter,tmp_path):
    obs,req=observation(),request()
    before=torch.random.get_rng_state().clone()
    p=adapter.prepare(obs,req);s1=adapter.run_stage1(p);s2=adapter.run_stage2(s1,capture=True,capture_cp2=True)
    rng=RequestRNG(req)
    off=adapter.sample(s2,adapter.sampler_for_steps(10),rng,CaptureSpec())
    on=adapter.sample(s2,adapter.sampler_for_steps(10),rng,CaptureSpec(True,True))
    assert off.normalized_chunk.numpy().tobytes()==on.normalized_chunk.numpy().tobytes()
    assert adapter.model.calls["head"]==[1]*20
    assert torch.equal(before,torch.random.get_rng_state())
    trace=adapter.make_trace_adapter()
    wire=adapter.to_wire(on.normalized_chunk,p).actions
    decision=DecisionCapture(req,"policy",0,0,(1,1,10),10,on.normalized_chunk,wire,
        s1.fields,{"identity":on.sampler.identity},{"rng_receipt":on.rng_receipt})
    counts={k:list(v) for k,v in adapter.model.calls.items()}
    captured=trace.capture(decision);fields=trace.sample_fields(s1,s2,on)
    assert adapter.model.calls==counts
    assert captured["real_cameras"]==["image"]
    assert fields["actual_nfe"]==10
    assert len(fields["native_steps"])==10
    assert len(fields["native_snapshots"])==9
    assert fields["cp2_fields"]["vlm_out"].size==32
    assert fields["native_grid"]["native_timesteps"]==on.sampler.native_timesteps
    # Native data roundtrips through numeric NPZ without pickle.
    path=tmp_path/"capture.npz"
    np.savez(path,vision_0=fields["retrieval_fields"]["vision_0"],
             actions=fields["normalized_actions"],x_t=fields["native_snapshots"][0]["x_t"])
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f["actions"],on.normalized_chunk.numpy())
    captured["decision"]["fields"]["vision_0"][:]=0
    assert np.any(s1.fields["vision_0"]!=0)


def test_key_pool_metadata_and_no_fake_view(adapter):
    s1=adapter.run_stage1(adapter.prepare(observation(),request()))
    tokens=s1.data["video_pre"]["tokens"]
    # Independent explicit adaptive-bin reference checks pooling order/boundaries.
    ref=[]
    grid=tokens.numpy().reshape(7,7,8)
    for i in range(4):
        for j in range(4):
            ref.extend(grid[(i*7)//4:((i+1)*7+3)//4,(j*7)//4:((j+1)*7+3)//4].mean((0,1)))
    np.testing.assert_allclose(s1.fields["vision_0"],ref,rtol=1e-6)
    assert set(s1.fields)=={"vision_0","robot_state"}
    builder=adapter._key_builder
    builder.collect("cp1",stage1=s1)
    keys=builder.build("cp1"); builder.clear()
    assert keys["vision_0"].shape==(128,)
    assert not builder.cached_data
    with pytest.raises(ValueError):
        builder.build("cp1")
    bad=dict(s1.data["video_pre"])
    bad["meta"]={**bad["meta"],"grid_sizes":torch.tensor([[1,8,8]])}
    with pytest.raises(ValueError):
        builder.fields_from_tokens(bad,s1.state)


def test_native_variants_matched_seed_cost_and_blueprint(adapter):
    p=adapter.prepare(observation(),request());s2=adapter.run_stage2(adapter.run_stage1(p))
    trace=adapter.make_trace_adapter()
    variants=trace.variants(s2,RequestRNG(request()))
    assert sum(v["extra_nfe"] for v in variants.values())==13
    for k in (10,2,1):
        assert variants[f"native_k{k}"]["extra_stage_calls"]==(0,0,1)
    assert adapter.model.calls["vae"]==[1]
    assert adapter.model.calls["prefill"]==[1]
    blueprint=adapter.cpu_blueprint()
    from dataclasses import asdict
    json.dumps(asdict(blueprint),allow_nan=False)
    adapter.metadata().layout.validate_commit(10)
    adapter.metadata().layout.validate_commit(30)
    with pytest.raises(ValueError):
        adapter.metadata().layout.validate_commit(32)
    with pytest.raises(ValueError):
        adapter.metadata().layout.validate_commit(33)
    with pytest.raises(ValueError):
        adapter.metadata().capabilities.require("split")


def test_true_vendor_sources_are_unchanged():
    import hashlib
    vendor=__import__("exp.r17_integration.models.fastwam.loader",fromlist=["VENDOR"]).VENDOR
    receipt=json.loads((FIXTURES/"vendor_sources.json").read_text())
    if not vendor.is_dir():
        pytest.skip("local pinned vendor not installed; checked-in native definitions still exercised")
    for name,entry in receipt.items():
        assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==entry["sha256"]


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_native_capture_consumed_by_p6_closed_npz_h5_bridge(benchmark,tmp_path):
    from dataclasses import asdict
    from exp.r17_integration.debug.capture_bridge import CaptureBridge,capture_native_product,export_h5
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.validate import model_binding
    import h5py
    a=make_adapter(benchmark);p=a.prepare(observation(benchmark),request())
    s1=a.run_stage1(p);s2=a.run_stage2(s1,capture=True)
    result=a.sample(s2,a.sampler_for_steps(2),RequestRNG(request()),CaptureSpec(True,True))
    product=a.make_trace_adapter().sample_fields(s1,s2,result)
    protocol=a.metadata().allowed_benchmark_protocols[0]
    binding=model_binding(asdict(a.metadata()),protocol,sampler=asdict(result.sampler))
    directory=tmp_path/"capture"
    bridge=CaptureBridge(directory,enabled=True,binding=binding)
    episode={"campaign":"p3_cpu_fixture","arm":"native_k2","bundle_generation":"k2_v1",
        "benchmark_protocol":protocol,"task_name":"fixture","orig_init_idx":0,
        "task_data_hash":"synthetic_fixture","task_uid":"p3:test:0:0","attempt":1,"decision_seq":0}
    counts={k:list(v) for k,v in a.model.calls.items()}
    capture_native_product(bridge,product,episode)
    stats=bridge.close()
    assert stats["drained"] and not stats["errors"]
    assert a.model.calls==counts
    rows=list(CaptureReader(directory,expected_binding=binding).decisions())
    assert len(rows)==1
    record,arrays,keys=rows[0]
    assert record["actual_nfe"]==2
    np.testing.assert_array_equal(arrays["normalized_actions"],result.normalized_chunk.numpy())
    np.testing.assert_array_equal(arrays["native_grid"],result.sampler.native_timesteps)
    assert set(keys)=={"vision_0","robot_state"}
    destination=export_h5(directory,tmp_path/"capture.h5")
    with h5py.File(destination,"r") as h5:
        assert h5.attrs["r17_closed_ok"]
        np.testing.assert_array_equal(h5["step_000000/keys/vision_0"][:],s1.fields["vision_0"])
