"""CPU preload then <=4.8s GPU window for the last K5/r2/B32 configuration."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from exp.r17_integration.models.dp.samples import dynamic_batch_measure as base

OUT = Path("exp/r17_integration/models/dp/samples/dynamic_batch_gpu_cap32_final_20261010").resolve()


def worker(index):
    from exp.r17_integration.models.dp.loader import load_adapter
    from exp.r17_integration.models.dp.provider import register_provider
    from exp.r17_integration.models.dp.batch_runtime import FrozenLanguageMemo
    from exp.r17_integration.qualification import model_runtime
    import torch
    directory = OUT / f"replica{index}"
    desc = json.loads((base.ASSETS / "load.json").read_text())
    desc["device"] = "cpu"
    adapter = load_adapter(desc)
    provider = register_provider(adapter)
    adapter.language = FrozenLanguageMemo(adapter.language)
    for obs in base.observations():
        adapter.language([obs["prompt"]])
    assert all(p.device.type == "cpu" for p in adapter.model.parameters())
    assert not torch.cuda.is_initialized()
    base.write(directory / "CPU_READY.json", dict(pid=os.getpid(), device="cpu", cuda_initialized=False, strict_load=adapter.strict_load))
    if sys.stdin.readline().strip() != "activate":
        return
    adapter.model.to("cuda:0")
    adapter.device = torch.device("cuda:0")
    from exp.r17_integration.common.contracts import RequestContext, CaptureSpec
    from exp.r17_integration.models.dp.sampler import SamplerProfile
    from exp.r17_integration.models.dp.request_noise import RequestNoise
    refs = []
    with torch.inference_mode():
        for i, obs in enumerate(base.observations()):
            context = RequestContext("dp_cap32",adapter.metadata().allowed_benchmark_protocols[0],"fixed_real_frame",i,0,explicit_seed=501+i)
            s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(obs,context)))
            result = adapter.sample(s2,SamplerProfile(num_steps=5),RequestNoise(501+i,adapter.device),CaptureSpec())
            refs.append(adapter.to_wire(result.normalized_chunk,None).actions)
    np.savez(directory / "reference.npz", actions=np.stack(refs))
    model_runtime.load = lambda *args,**kwargs: (adapter,provider)
    from exp.r17_integration.models.dp.formal_batch_server import serve
    campaign = json.loads((OUT / "campaign.json").read_text())
    serve(campaign,campaign["ports"][index],directory,cuda_graph=True)


def main():
    if len(sys.argv)>1:
        return worker(int(sys.argv[1]))
    OUT.mkdir(parents=True,exist_ok=False)
    from exp.r17_integration.models.dp.batch_runtime import BatchConfig
    import shutil
    assets = OUT / "assets"
    assets.mkdir()
    for filename in ("load.json","model_spec.json","benchmark.json","wire_observations.npz"):
        shutil.copyfile(base.ASSETS / filename, assets / filename)
    shutil.copyfile(base.ASSETS / "k5.yaml",assets / "full.yaml")
    ports = [base.free_port(),base.free_port()]
    base.write(OUT / "campaign.json",dict(model="dp",domain="libero",assets=str(assets),device="cuda:0",memory_budget_gib=40,
        replicas=2,dp_batching=BatchConfig(32,5).metadata(),denoising_steps=5,commit_horizon=5,library=None,ports=ports))
    processes, logs, started = [], [], None
    try:
        for i in range(2):
            child = OUT / f"replica{i}"
            child.mkdir()
            log = (child / "native.log").open("w")
            logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false")
            processes.append(subprocess.Popen([base.PYTHON,"-u",str(Path(__file__).resolve()),str(i)],env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True))
        base.write(OUT / "owned_pids.json",[p.pid for p in processes])
        until=time.monotonic()+90
        while not all((OUT / f"replica{i}/CPU_READY.json").is_file() for i in range(2)):
            if any(p.poll() is not None for p in processes) or time.monotonic()>until:
                raise RuntimeError("CPU preload failed")
            time.sleep(.05)
        # CPU preparation above did not initialize CUDA. Acquire the same GPU lock
        # before sending activation; a hard timer signals only our own PIDs.
        with open("/tmp/r17_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            started=time.monotonic()
            def expired():
                for p in processes:
                    if p.poll() is None:
                        p.kill()
            timer=threading.Timer(4.8,expired)
            timer.start()
            try:
                for p in processes:
                    p.stdin.write("activate\n")
                    p.stdin.flush()
                while not all((OUT / f"replica{i}/ready.json").is_file() for i in range(2)):
                    if any(p.poll() is not None for p in processes):
                        raise TimeoutError("4.8s GPU cutoff reached before readiness")
                    time.sleep(.01)
                with np.load(OUT / "replica0/reference.npz",allow_pickle=False) as data:
                    reference=data["actions"].copy().view(base.References)
                reference.K=5
                rows,board,elapsed=asyncio.run(base.workload(ports,reference,12,1,50))
                start=rows[0]["measurement_start"]
                timely=[r for r in rows if start<=r["completed_monotonic"]<=start+1]
                power=[r for r in board if start<=r["time"]<=start+1]
                result=dict(K=5,replicas=2,max_batch=32,batch_wait_ms=5,clients=12,calls=len(timely),latency_samples=len(rows),measurement_seconds=1,drain_seconds=elapsed-1,
                    calls_per_second=len(timely),actions_per_second=5*len(timely),p50_ms=float(np.percentile([r["latency_ms"] for r in rows],50)),
                    p90_ms=float(np.percentile([r["latency_ms"] for r in rows],90)),watts=float(np.mean([r["watts"] for r in power])),max_diff=max(r["max_diff"] for r in rows),
                    mean_stage3_batch=float(np.mean([r["batch_size"] for r in rows])),max_stage3_batch=max(r["batch_size"] for r in rows),quality_sr_claim=False)
                base.write(OUT / "result.json",result)
                base.write(OUT / "calls.json",rows)
                base.write(OUT / "power.json",board)
                print(json.dumps(result),flush=True)
            finally:
                expired()
                for p in processes:
                    p.wait(timeout=2)
                timer.cancel()
                base.write(OUT / "GPU_WINDOW.json",dict(GPU_seconds=time.monotonic()-started,hard_cutoff_seconds=4.8,lock="/tmp/r17_gpu.lock",own_children_closed=True))
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=2)
        base.write(OUT / "cleanup.json",[dict(pid=p.pid,exit_code=p.returncode,owned=True) for p in processes])
        for log in logs:
            log.close()


if __name__=="__main__":
    main()
