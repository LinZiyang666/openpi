"""Bounded shared A/B probes, never attaching to or signalling live campaigns.

DP uses P4's unchanged real formal-resident load generator. Fast-WAM uses
unchanged native hooks and S07 with 32 independent closed-loop clients. Private
MPS is enabled only in this process tree; probe children are explicitly owned.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .mps import Session, client_env, command, gpu_snapshot, inspect
from exp.r17_integration.ops.formal_process import write_json

GPU = "GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9"
ROOT = Path(__file__).resolve().parents[5]


def contexts():
    text = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    rows = []
    for line in text.splitlines():
        pid, uuid, memory = [x.strip() for x in line.split(",")]
        stat = Path("/proc") / pid / "stat"
        rows.append(dict(pid=int(pid), gpu=uuid, memory_mib=float(memory),
                         start_ticks=stat.read_text().rsplit(") ",1)[1].split()[19]))
    return rows


def background(directories):
    counts = {}
    for directory in directories:
        calls, decisions = 0, 0
        for path in Path(directory).glob("servers/replica_*/serving.jsonl"):
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        decisions += 1
                        calls += row.get("source") == "policy"
                    except ValueError:
                        pass
        counts[Path(directory).name] = dict(policy_calls=calls, decisions=decisions)
    return dict(time=time.time(), counts=counts, contexts=contexts())


def rates(before, after):
    seconds = after["time"] - before["time"]
    return {name: (row["policy_calls"]-before["counts"][name]["policy_calls"])/seconds
            for name,row in after["counts"].items()}


def fast_worker(directory, clients, seconds, observations, descriptor, *, gpu_uuid=GPU, start_file=None, memory_gib=None):
    import numpy as np
    import torch
    from ..loader import load_adapter, LoadSpec
    from ..batch_runtime import BatchConfig, BatchLedger
    from ..batcher import FastWAMBatchedAdapter
    from ..batch_profile import BoardSampler
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    from openpi.serving.batching_core import BatchingCore
    directory = Path(directory)
    if memory_gib is not None:
        torch.cuda.set_per_process_memory_fraction(memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    adapter = load_adapter(LoadSpec(**json.loads(Path(descriptor).read_text())), device="cuda:0")
    adapter.batch_config = BatchConfig(8,5.)
    adapter.batch_ledger = BatchLedger(directory / "batching.jsonl", adapter.batch_config)
    with np.load(observations, allow_pickle=False) as data:
        obs = [{k: str(data[k][i]) if k == "prompt" else data[k][i].copy() for k in data.files} for i in range(3)]
    requests = [RequestContext("mps_probe", adapter.metadata().allowed_benchmark_protocols[0],
        "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        1272, i, explicit_seed=42+i) for i in range(3)]
    sampler = adapter.sampler_for_steps(10)
    references = []
    with torch.inference_mode():
        for i in range(3):
            result,_ = full_call(adapter,obs[i],requests[i],sampler=sampler)
            references.append(result.normalized_chunk.float().cpu().numpy())
    np.savez(directory/"reference.npz", actions=np.stack(references))
    ready, go = threading.Barrier(clients+1), threading.Event()
    rows, outputs, timing = [], [], {}
    lock = threading.Lock()
    with BatchingCore(adapter.make_stage_batcher(),device=adapter.model.device,max_batch_size=8,max_wait_ms=5.) as core:
        def one(index):
            staged = FastWAMBatchedAdapter(adapter,core)
            row = index % 3
            with torch.inference_mode():
                full_call(staged,obs[row],requests[row],sampler=sampler)
                ready.wait(timeout=120)
                go.wait(timeout=120)
                while time.monotonic() < timing["deadline"]:
                    started = time.monotonic()
                    result,_ = full_call(staged,obs[row],requests[row],sampler=sampler)
                    actions = result.normalized_chunk.float().cpu().numpy()
                    capture = result.capture.get("batching",{})
                    record = dict(client=index, frame=row, latency_ms=(time.monotonic()-started)*1000,
                        max_diff=float(np.max(np.abs(actions-references[row]))),
                        batch_size=capture["stage3"]["size"])
                    with lock:
                        rows.append(record)
                        outputs.append(actions)
                    time.sleep(.05)
        with ThreadPoolExecutor(clients) as pool:
            tasks = [pool.submit(one,i) for i in range(clients)]
            ready.wait(timeout=120)
            torch.cuda.reset_peak_memory_stats()
            write_json(directory/"ready.json",dict(pid=os.getpid(),clients=clients,stage_core="S07"))
            if start_file is not None:
                bound=time.monotonic()+120
                while not Path(start_file).exists():
                    if time.monotonic()>bound:
                        raise TimeoutError('bounded cross-replica start barrier')
                    time.sleep(.02)
            with BoardSampler(gpu_uuid) as board:
                timing["start"] = time.monotonic()
                timing["deadline"] = timing["start"]+seconds
                timing["epoch_start"] = time.time()
                go.set()
                for future in tasks:
                    future.result(timeout=seconds+120)
                end = time.monotonic()
                epoch_end = time.time()
            result = dict(model="fastwam",replicas=1,max_batch=8,clients=clients,calls=len(rows),
                elapsed_seconds=end-timing["start"],calls_per_second=len(rows)/(end-timing["start"]),
                p50_ms=float(np.median([r["latency_ms"] for r in rows])),
                max_diff=max(r["max_diff"] for r in rows),
                mean_stage3_batch=float(np.mean([r["batch_size"] for r in rows])),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                started_epoch=timing["epoch_start"],ended_epoch=epoch_end,
                **board.summary(timing["start"],end))
    np.savez(directory/"actions.npz",actions=np.stack(outputs),frame=np.array([r["frame"] for r in rows]))
    write_json(directory/"calls.json",rows)
    write_json(directory/"result.json",result)


def run(out, directories, budget, seconds, models=('dp','fastwam'), rounds=2, settings=None):
    from exp.r17_integration.models.dp.samples import dynamic_batch_measure as dp
    out.mkdir(exist_ok=False)
    began, results = time.monotonic(), []
    deadline = began+budget
    initial = background(directories)
    write_json(out/"receipt.json",dict(started=time.time(),GPU=GPU,budget_seconds=budget,
        before=initial,probe_mode="shared; no exclusive lock; own new clients only"))
    # A/B/A/B per topology, then repeat Fast-WAM. Loading is counted in budget.
    modes=tuple(x=='on' for x in settings) if settings is not None else ((False,True)*rounds)
    cases = [("dp",r,b,m) for r,b in ((1,16),(2,8)) for m in modes] if 'dp' in models else []
    if 'fastwam' in models:
        cases += [("fastwam",1,8,m) for m in modes]
    try:
        for index,(model,replicas,batch,mps) in enumerate(cases):
            if time.monotonic()+100+seconds > deadline-20:
                raise TimeoutError("bounded total GPU allowance: reserve cleanup time")
            free = gpu_snapshot()[0]["free_mib"]
            needed = 34000 if model=="fastwam" else 8192
            if free < needed+4096:
                results.append(dict(model=model,replicas=replicas,max_batch=batch,mps=mps,status="SKIP_MEMORY",free_mib=free))
                continue
            case = out/f"{index:02d}_{model}_{'on' if mps else 'off'}"
            case.mkdir()
            session, previous = None, dict(os.environ)
            try:
                if mps:
                    session = Session(case)
                    os.environ.update(client_env(session.start()))
                else:
                    # Explicit bypass, only for this probe tree, even if an
                    # unrelated default MPS daemon happens to exist.
                    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(case/"nonexistent")
                before = background(directories)
                if model=="dp":
                    power_class = dp.BoardPower
                    measurement = {}
                    class Power(power_class):
                        def __enter__(self):
                            measurement["before"] = background(directories)
                            if session:
                                own = json.loads((case/f"k100_r{replicas}_b{batch}_c32"/"owned_pids.json").read_text())
                                from .mps import account
                                write_json(case/"attachment.json",account(session.state,own))
                            return super().__enter__()
                        def __exit__(self,*args):
                            super().__exit__(*args)
                            measurement["after"] = background(directories)
                    dp.BoardPower = Power
                    try:
                        result = dp.measure(case,replicas,batch,100,32,seconds,50,deadline)
                    finally:
                        dp.BoardPower = power_class
                    result["background_calls_per_second"] = rates(measurement["before"],measurement["after"])
                else:
                    log = (case/"native.log").open("w")
                    env = dict(os.environ,PYTHONPATH=str(ROOT)+":"+str(ROOT/"src")+":"+str(ROOT/"packages/openpi-client/src")+
                        ":/data/mw_finetune/vendor/lerobot/src:/data/mw_finetune/vendor/FastWAM/src",
                        OPENPI_DISABLE_STAGE_STREAMS="0",BATCHING_STAGE1_WORKERS="1",BATCHING_STAGE2_WORKERS="1",BATCHING_STAGE3_WORKERS="1")
                    process = subprocess.Popen(["/data/mw_finetune/envs/fastwam/bin/python","-u","-m","exp.r17_integration.models.fastwam.ops.mps_probe",
                        "--fast-worker",str(case),"--seconds",str(seconds)],env=env,stdout=log,stderr=subprocess.STDOUT)
                    write_json(case/"owned_pid.json",dict(pid=process.pid))
                    try:
                        while process.poll() is None and not (case/"ready.json").exists():
                            if time.monotonic()>min(deadline-15,began+budget):
                                raise TimeoutError("Fast-WAM bounded startup")
                            time.sleep(.2)
                        if process.poll() is not None:
                            raise RuntimeError("owned Fast-WAM failed: "+str(case/"native.log"))
                        at_start = background(directories)
                        if session:
                            from .mps import account
                            write_json(case/"attachment.json",account(session.state,[process.pid]))
                        process.wait(timeout=min(seconds+60,max(1,deadline-time.monotonic()-15)))
                        if process.returncode:
                            raise RuntimeError("owned Fast-WAM failed: "+str(case/"native.log"))
                        at_end = background(directories)
                        result = json.loads((case/"result.json").read_text())
                        result["background_calls_per_second"] = rates(at_start,at_end)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=15)
                        log.close()
                        write_json(case/"cleanup.json",dict(pid=process.pid,exit_code=process.returncode,closed=True))
                result.update(model=model,mps=mps,status="PASS",case=str(case),before=before,after=background(directories))
                results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('before','after')}),flush=True)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                if session:
                    write_json(case/"mps_closed.json",session.close())
            write_json(out/"results.json",results)
    finally:
        write_json(out/"results.json",results)
        final = background(directories)
        old = {(r["pid"],r["start_ticks"],r["gpu"]) for r in initial["contexts"]}
        current = {(r["pid"],r["start_ticks"],r["gpu"]) for r in final["contexts"]}
        write_json(out/"closed.json",dict(elapsed_seconds=time.monotonic()-began,results=len(results),
            background_contexts_preserved=old <= current,after=final,budget_seconds=budget))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    p.add_argument("--background",type=Path,nargs="*",default=[])
    p.add_argument("--seconds",type=float,default=16)
    p.add_argument("--budget-seconds",type=int,default=1100)
    p.add_argument("--models",choices=('dp','fastwam'),nargs='+',default=['dp','fastwam'])
    p.add_argument("--rounds",type=int,choices=(1,2),default=2)
    p.add_argument("--settings",choices=('off','on'),nargs='+')
    p.add_argument("--fast-worker",type=Path)
    p.add_argument("--clients",type=int,default=32)
    p.add_argument("--gpu-uuid",default=GPU)
    p.add_argument("--start-file",type=Path)
    p.add_argument("--memory-gib",type=float)
    p.add_argument("--descriptor",type=Path,default=ROOT/"exp/r17_integration/models/fastwam/measurements/mps_descriptor.json")
    args=p.parse_args()
    if args.fast_worker:
        return fast_worker(args.fast_worker,args.clients,args.seconds,
            "/data/mw_finetune/qualification/observations/libero_original256_fix2_v1/fastwam_observations.npz",
            str(args.descriptor),gpu_uuid=args.gpu_uuid,start_file=args.start_file,memory_gib=args.memory_gib)
    if args.out is None or not 60<=args.budget_seconds<=1150:
        p.error("--out and GPU budget60..1150 required")
    return run(args.out,args.background,args.budget_seconds,args.seconds,args.models,args.rounds,args.settings)


if __name__=="__main__":
    main()
