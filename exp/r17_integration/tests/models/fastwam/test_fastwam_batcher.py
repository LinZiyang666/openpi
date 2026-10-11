from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch

from openpi.serving.batching_core import BatchingCore, Stage3MissPayload, Stage3WarmStartPayload
from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation, request


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
@pytest.mark.parametrize("k",[1,2,10])
def test_b2_vector_native_matches_single_no_alias(benchmark,k):
    a=make_adapter(benchmark);batcher=a.make_stage_batcher()
    ps=[a.prepare(observation(benchmark,i),request(i)) for i in range(2)]
    singles=[]
    for p in ps:
        s2=a.run_stage2(a.run_stage1(p))
        singles.append(a.sample(s2,a.sampler_for_steps(k),RequestRNG(p.request),CaptureSpec()))
    a.model.calls={name:[] for name in a.model.calls}
    s1s=batcher.run_stage1_batch(ps)
    s2s=batcher.run_stage2_batch(s1s,capture=True)
    noises=[torch.randn((32,a.metadata().layout.action_width),generator=RequestRNG(p.request).torch_generator()) for p in ps]
    payloads=[Stage3MissPayload(s2,n,num_steps=k) for s2,n in zip(s2s,noises)]
    requested=[None,(0.,)] if k==1 else [(.5,) if k==2 else (.2,.7),None]
    outs=batcher.run_stage3_miss(payloads,num_steps=k,save_timesteps_per_request=requested)
    assert a.model.calls["vae"]==[2]
    assert a.model.calls["pre"]==[2]
    assert a.model.calls["prefill"]==[2]
    assert a.model.calls["head"]==[2]*k
    for i,out in enumerate(outs):
        torch.testing.assert_close(out.action[0],singles[i].normalized_chunk,rtol=0,atol=0)
        assert out.actual_nfe==k
        assert set(out.intermediates or {})==set(requested[i] or ())
    assert not torch.equal(outs[0].action,outs[1].action)
    first_before=s1s[1].data["video_pre"]["tokens"].clone()
    with torch.inference_mode():
        s1s[0].data["video_pre"]["tokens"].zero_()
    assert torch.equal(first_before,s1s[1].data["video_pre"]["tokens"])
    kv_before=s2s[1].data["video_kv_cache"][0]["k"].clone()
    with torch.inference_mode():
        s2s[0].data["video_kv_cache"][0]["k"].zero_()
    assert torch.equal(kv_before,s2s[1].data["video_kv_cache"][0]["k"])
    second=outs[1].action.clone()
    with torch.inference_mode():
        outs[0].action.zero_()
    assert torch.equal(second,outs[1].action)
    assert torch.equal(noises[0],payloads[0].noise)


def test_real_batching_core_stage_threads_and_fault_bucket(adapter):
    batcher=adapter.make_stage_batcher()
    ps=[adapter.prepare(observation(variant=i),request(i)) for i in range(2)]
    with BatchingCore(batcher,device="cpu",max_batch_size=8,max_wait_ms=5) as core:
        s1=core.submit_many_to_stage(1,"cpu",ps,timeout=3)
        s2=core.submit_many_to_stage(2,"cpu",s1,requires_stage2_capture=True,timeout=3)
        noises=[torch.randn((32,4),generator=RequestRNG(p.request).torch_generator()) for p in ps]
        payloads=[Stage3MissPayload(s,n,num_steps=2,save_timesteps=(.5,)) for s,n in zip(s2,noises)]
        outs=core.submit_many_to_stage(3,"cpu",payloads,timeout=3)
        warms=[Stage3WarmStartPayload(s,out.intermediates[.5][0],.5,2,True) for s,out in zip(s2,outs)]
        resumed=core.submit_many_to_stage(3,"cpu",warms,timeout=3)
        for full,warm in zip(outs,resumed):
            torch.testing.assert_close(full.action,warm.action,rtol=0,atol=0)
            assert warm.actual_nfe==1
            assert warm.first_step_input is not None
        bad=Stage3MissPayload(s2[0],torch.full((32,4),float("nan")),num_steps=1)
        good=Stage3MissPayload(s2[1],noises[1],num_steps=2)
        reqs=[core.enqueue_to_stage(3,"cpu",p) for p in (bad,good)]
        with pytest.raises(ValueError):
            core.wait_for(reqs[0],timeout=3)
        assert core.wait_for(reqs[1],timeout=3).actual_nfe==2
        assert all(r.reply_event.is_set() for r in reqs)
    assert adapter.model.calls["vae"]==[2]
    assert adapter.model.calls["prefill"]==[2]
    assert 2 in adapter.model.calls["head"]


def test_bucket_identity_schedule_capture_and_wrong_points(adapter):
    s2=adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation(),request())))
    batcher=adapter.make_stage_batcher()
    p=Stage3MissPayload(s2,torch.ones(32,4),num_steps=2)
    assert batcher.bucket_key(p)!=batcher.bucket_key(replace(p,num_steps=10))
    assert batcher.bucket_key(p)!=batcher.bucket_key(replace(p,noise=p.noise.double()))
    # Capture sets do not fragment otherwise compatible forwards.
    assert batcher.bucket_key(p)==batcher.bucket_key(replace(p,save_timesteps=(.5,)))
    other=make_adapter("libero")
    s2_other=other.run_stage2(other.run_stage1(other.prepare(observation("libero"),request())))
    assert batcher.bucket_key(p)!=other.make_stage_batcher().bucket_key(Stage3MissPayload(s2_other,torch.ones(32,7),2))
    for point in (.25,1.,500.,float("nan")):
        with pytest.raises((ValueError,OverflowError)):
            batcher.run_stage3_warm([Stage3WarmStartPayload(s2,torch.ones(32,4),point,2)],
                start_t=point,num_steps=2,capture_first_step=False)
    from exp.r17_integration.common.contracts import Stage2
    foreign=Stage2({**s2.data,"model_identity":"wrong"},s2.stage1)
    with pytest.raises(ValueError):
        batcher.bucket_key(replace(p,stage2_out=foreign))
    with pytest.raises(ValueError):
        adapter.sample(foreign,adapter.sampler_for_steps(2),RequestRNG(request()))


def test_request_order_and_threads_preserve_noise(adapter):
    def run(i):
        p=adapter.prepare(observation(variant=i),request(i))
        return adapter.sample(adapter.run_stage2(adapter.run_stage1(p)),adapter.sampler_for_steps(2),RequestRNG(p.request))
    reference={i:run(i).normalized_chunk.numpy().tobytes() for i in range(2)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={i:pool.submit(run,i) for i in (1,0)}
        assert {i:f.result().normalized_chunk.numpy().tobytes() for i,f in futures.items()}==reference


def test_shared_core_adapter_routes_from_connection_hooks(adapter,tmp_path):
    from exp.r17_integration.models.fastwam.provider import register_provider
    from exp.r17_integration.common.serving import full_call
    batcher=adapter.make_stage_batcher()
    with BatchingCore(batcher,device="cpu",max_batch_size=2,max_wait_ms=20) as core:
        provider=register_provider(adapter,provider_id="p3_batch_"+tmp_path.name,coordinator=core)
        clients=[provider.load_adapter() for _ in range(2)]
        assert clients[0].adapter.stage1_lock is clients[1].adapter.stage1_lock
        def run(i):
            return full_call(clients[i],observation(variant=i),request(i),sampler=adapter.sampler_for_steps(2),capture=CaptureSpec(True,True))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,i) for i in range(2)]
            outputs=[f.result() for f in futures]
        assert adapter.model.calls["vae"]==[2]
        assert adapter.model.calls["prefill"]==[2]
        assert adapter.model.calls["head"]==[2,2]
        assert all(result.actual_nfe==2 and len(result.snapshots)==1 for result,_ in outputs)


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_qualification_command_core_is_cpu_testable(benchmark):
    from exp.r17_integration.models.fastwam.qualify import qualify_adapter
    a=make_adapter(benchmark)
    result=qualify_adapter(a,[observation(benchmark,i%2) for i in range(3)],batch2=True,rtol=0,atol=0)
    assert result["status"]=="passed",result
    assert len(result["rows"])==11
    assert result["device"]=="cpu"
    partial=qualify_adapter(a,[observation(benchmark)],deadline_seconds=0)
    assert partial["status"]=="partial"
