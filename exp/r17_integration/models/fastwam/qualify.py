"""Bounded future native qualification; import and CPU fixtures use no GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def qualify_adapter(adapter, observations, *, deadline_seconds=780, batch2=False, rtol=1e-3, atol=1e-3):
    import torch
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.common.serving import full_call
    start=time.monotonic()
    rows=[]
    receipt={"schema":"r17.fastwam.qualification.v1","status":"partial", "model":asdict(adapter.manifest),
        "device":str(adapter.model.device),"rtol":rtol,"atol":atol,"rows":rows,
        "limits":["native model mechanics only; no physical episode/SR, WS/replica GPU or throughput certificate"]}
    def remaining():
        return time.monotonic()-start < deadline_seconds
    try:
        if batch2 and not adapter.allow_unqualified_batch:
            raise ValueError("production B>1 disabled; no GPU batch-composition parity certificate")
        for i,obs in enumerate(observations[:3]):
            if not remaining():
                return receipt
            req=RequestContext("p3_qualification",adapter.metadata().allowed_benchmark_protocols[0],
                "fixed_observation",i,0,explicit_seed=adapter.wire.protocol.fixed_seed or 42+i)
            prepared=adapter.prepare(obs,req)
            for k in ((10,) if adapter.wire.protocol.fixed_seed is not None else (10,2,1)):
                if not remaining():
                    return receipt
                t=time.monotonic()
                sampler=adapter.sampler_for_steps(k)
                native=adapter.model.infer_action(prompt=prepared.data["prompt"],input_image=prepared.data["input_image"],
                    proprio=prepared.data["proprio"],action_horizon=32,num_inference_steps=k,
                    seed=RequestRNG(req).seed_for(),rand_device="cpu",text_cfg_scale=1.0,compile_action_infer=False)["action"]
                result,wire=full_call(adapter,obs,req,sampler=sampler,capture=CaptureSpec(True,True))
                torch.testing.assert_close(native,result.normalized_chunk,rtol=rtol,atol=atol)
                expected=adapter.to_wire(native,prepared).actions
                import numpy as np
                np.testing.assert_allclose(wire.actions,expected,rtol=rtol,atol=atol)
                rows.append({"obs_index":i,"k":k,"actual_nfe":result.actual_nfe,
                    "capture_enabled": True, "pricing_eligible": False,
                    "native_grid":asdict(sampler),"max_abs_error":float((native-result.normalized_chunk).abs().max()),
                    "elapsed_wall_seconds":time.monotonic()-t, "status":"passed"})
            if i==0 and remaining():
                s1=adapter.run_stage1(prepared);s2=adapter.run_stage2(s1,capture=True)
                rng=RequestRNG(req);sampler=adapter.sampler_for_steps(10)
                full=adapter.sample(s2,sampler,rng,CaptureSpec(True,True))
                off=adapter.sample(s2,sampler,rng,CaptureSpec())
                torch.testing.assert_close(off.normalized_chunk,full.normalized_chunk,rtol=0,atol=0)
                snap=full.snapshots[4]
                suffix=adapter.resume(s2,snap,sampler,rng,CaptureSpec())
                torch.testing.assert_close(full.normalized_chunk,suffix.normalized_chunk,rtol=0,atol=0)
                rows.append({"check":"debug_bytes_and_native_resume","remaining_nfe":suffix.actual_nfe,"status":"passed"})
        if batch2 and len(observations)>=2 and remaining():
            from openpi.serving.batching_core import Stage3MissPayload
            ps=[adapter.prepare(obs,RequestContext("p3_qualification",adapter.metadata().allowed_benchmark_protocols[0],
                "fixed_observation",i,0,explicit_seed=42+i)) for i,obs in enumerate(observations[:2])]
            batcher=adapter.make_stage_batcher()
            s1s=batcher.run_stage1_batch(ps);s2s=batcher.run_stage2_batch(s1s,capture=True)
            payloads=[Stage3MissPayload(s2,torch.randn((32,adapter.metadata().layout.action_width),
                generator=RequestRNG(p.request).torch_generator()),2) for p,s2 in zip(ps,s2s)]
            outs=batcher.run_stage3_miss(payloads,num_steps=2,save_timesteps_per_request=[None,None])
            for p,out in zip(ps,outs):
                single=adapter.sample(adapter.run_stage2(adapter.run_stage1(p)),adapter.sampler_for_steps(2),RequestRNG(p.request))
                torch.testing.assert_close(single.normalized_chunk,out.action[0],rtol=rtol,atol=atol)
            rows.append({"check":"native_batch2","status":"passed"})
        receipt["status"]="passed" if len(observations)>=3 and remaining() else "partial"
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    receipt["elapsed_seconds"]=time.monotonic()-start
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--descriptor",required=True)
    parser.add_argument("--observations",required=True,help="NPZ image/state/prompt, image2 for LIBERO; allow_pickle=False")
    parser.add_argument("--receipt",required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--batch2",action="store_true")
    parser.add_argument("--deadline-seconds",type=float,default=780)
    args=parser.parse_args()
    if args.batch2:
        parser.error("production B>1 disabled until GPU batch-composition parity is certified")
    import numpy as np
    from .loader import LoadSpec,load_adapter
    started=time.monotonic()
    output=Path(args.receipt)
    spec=LoadSpec(**json.loads(Path(args.descriptor).read_text()))
    try:
        adapter=load_adapter(spec,device=args.device)
        with np.load(args.observations,allow_pickle=False) as data:
            observations=[{k:str(data[k][i]) if k=="prompt" else data[k][i].copy()
                for k in data.files} for i in range(min(3,len(data["state"])))]
        remaining=max(0,args.deadline_seconds-(time.monotonic()-started))
        receipt=qualify_adapter(adapter,observations,deadline_seconds=remaining,batch2=args.batch2)
        receipt["elapsed_with_load_seconds"]=time.monotonic()-started
    except Exception as exc:
        receipt={"schema":"r17.fastwam.qualification.v1","status":"failed",
            "error":f"{type(exc).__name__}: {exc}","elapsed_with_load_seconds":time.monotonic()-started}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    if receipt["status"]!="passed":
        raise SystemExit(1)


if __name__=="__main__":
    main()
