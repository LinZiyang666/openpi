"""Real native S3 B2 certificate; S1/S2 batching remains unsupported."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from .assets import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    job,run=json.loads(Path(a.job).read_text()),Path(a.run)
    report=dict(status='PARTIAL',scope='native S3 B2 same geometry only',checks=[],quality_claim=False,
        limits=['S1/S2 batching unsupported','throughput/queue pricing unqualified'])
    write_json(run/'native.json',report);begin=time.monotonic()
    try:
        import torch
        from .model_runtime import load,wire_observations
        from exp.r17_integration.common.contracts import RequestContext,CaptureSpec
        from exp.r17_integration.common.request_rng import RequestRNG
        from openpi.serving.batching_core import Stage3MissPayload
        from openpi.cache.groot.batcher import stage3_input
        adapter,provider=load(job)
        spec=adapter.metadata();sampler=spec.default_sampler;k=sampler.num_steps
        obs=wire_observations(job)[0]
        req=RequestContext('Q-GR-B2',spec.allowed_benchmark_protocols[0],'reach-v3',0,0,explicit_seed=18)
        stages=[];singles=[];payloads=[]
        for i in range(2):
            context=replace(req,original_init_idx=i,explicit_seed=18+i)
            stage=adapter.run_stage2(adapter.run_stage1(adapter.prepare(obs,context)))
            result=adapter.sample(stage,sampler,RequestRNG(context),CaptureSpec(True,True))
            stages.append(stage);singles.append(result)
            payloads.append(Stage3MissPayload(stage3_input(adapter.runner,stage.data),result.capture['initial_noise'],k))
        batcher=adapter.make_stage_batcher()
        replies=batcher.run_stage3_miss(payloads,num_steps=k,save_timesteps_per_request=[None,None])
        for single,reply in zip(singles,replies):
            actual=reply.action[0].detach().cpu().float()
            torch.testing.assert_close(actual,single.normalized_chunk,rtol=1e-3,atol=1e-3)
            report['checks'].append(dict(status='PASS',rtol=1e-3,atol=1e-3,max_abs=float((actual-single.normalized_chunk).abs().max()),actual_nfe=k))
        before=replies[1].action.clone();replies[0].action[0,0,0]=999
        assert torch.equal(before,replies[1].action)
        report.update(status='PASS',request_buffers_independent=True,exact_noise_per_request=True)
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_seconds']=time.monotonic()-begin;write_json(run/'native.json',report)


if __name__=='__main__': main()
