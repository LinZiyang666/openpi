"""Same actual formal entry as dynamic_batch_measure; reuse own loaded residents.

Every row starts/stops its actual server/core. Native model/graph workspaces and
frozen CPU text memo remain resident. Only one or two owned GPU processes live.
The ledger/ready/closed receipts remain distinct per configuration.
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from exp.r17_integration.models.dp.samples import dynamic_batch_measure as base


def worker():
    from exp.r17_integration.qualification import model_runtime
    from exp.r17_integration.models.dp import batch_graph
    from exp.r17_integration.models.dp.formal_batch_server import serve
    original_load, original_check = model_runtime.load, batch_graph.check_graph
    arms = (int(os.environ["R17_DP_FAST_ARM"]),) if "R17_DP_FAST_ARM" in os.environ else (100,10,5)
    if any(k not in (100,10,5) for k in arms):
        raise ValueError("unknown bounded qualification arm")
    cached, commands, exiting = {}, queue.Queue(), threading.Event()
    def reader():
        for line in sys.stdin:
            doc = json.loads(line)
            if doc["type"] in ("stop", "exit"):
                if doc["type"] == "exit":
                    exiting.set()
                os.kill(os.getpid(), signal.SIGTERM)  # this process's own formal request_stop handler
            else:
                commands.put(doc)
        exiting.set()
    threading.Thread(target=reader, daemon=True).start()
    def load(job, device):
        if not cached:
            adapter, provider = original_load(job, device)
            from exp.r17_integration.models.dp.batch_runtime import FrozenLanguageMemo
            from exp.r17_integration.common.contracts import RequestContext, CaptureSpec
            from exp.r17_integration.models.dp.request_noise import RequestNoise
            from exp.r17_integration.models.dp.sampler import SamplerProfile
            import torch
            # Exact B1 CPU text function; retain its immutable values between rows.
            adapter.language = FrozenLanguageMemo(adapter.language)
            references, receipts = {}, {}
            with torch.inference_mode():
                stages = []
                for i, obs in enumerate(base.observations()):
                    context = RequestContext("dp_batch_measure", adapter.metadata().allowed_benchmark_protocols[0],
                        "fixed_real_frame", i, 0, explicit_seed=501+i)
                    stages.append(adapter.run_stage2(adapter.run_stage1(adapter.prepare(obs, context))))
                for k in arms:
                    rows, rng = [], []
                    for i, stage in enumerate(stages):
                        result = adapter.sample(stage, SamplerProfile(num_steps=k), RequestNoise(501+i, adapter.device), CaptureSpec())
                        rows.append(adapter.to_wire(result.normalized_chunk, None).actions)
                        rng.append(result.rng_receipt)
                    references[k], receipts[k] = np.stack(rows), rng
            cached.update(adapter=adapter, provider=provider, references=references, receipts=receipts, stages=stages)
        # Revalidate the exact benchmark, even when the same native resident is reused.
        from exp.r17_integration.qualification.benchmark_binding import benchmark_spec
        benchmark_spec(job)
        return cached["adapter"], cached["provider"]
    def check(adapter, observations, *, steps):
        if "graph" not in cached:
            cached["graph"] = original_check(adapter, base.observations(), steps=arms)
            # Native full-B32 qualification: three real paired frames and
            # private request seeds repeated across rows. No serving threads yet.
            from exp.r17_integration.models.dp.batch_runtime import BatchConfig
            from exp.r17_integration.models.dp.sampler import SamplerProfile
            from exp.r17_integration.models.dp.request_noise import RequestNoise
            from exp.r17_integration.common.contracts import CaptureSpec
            import torch
            batch_graph.warm_batch_signatures(adapter, cached["graph"]["graph"], cached["stages"][0], 32,
                SamplerProfile(num_steps=100).scheduler(adapter.model.cfg).timesteps[0])
            prior, audit = adapter.batch_config, []
            adapter.batch_config = BatchConfig(32, 5)
            try:
                with torch.inference_mode():
                    for k in arms:
                        results = adapter.sample_batch([cached["stages"][i%3] for i in range(32)], SamplerProfile(num_steps=k),
                            [RequestNoise(501+i%3, adapter.device) for i in range(32)], [CaptureSpec()]*32)
                        for i, result in enumerate(results):
                            assert result.rng_receipt == cached["receipts"][k][i%3], "B32 private RNG diverged from B1"
                        drift = max(float(np.max(np.abs(adapter.to_wire(result.normalized_chunk, None).actions-cached["references"][k][i%3]))) for i,result in enumerate(results))
                        audit.append(dict(K=k, batch=32, full_horizon=16, checked_requests=32, private_rng_exact=True, max_wire_diff=drift))
            finally:
                adapter.batch_config = prior
            cached["graph"]["batch32_noise_audit"] = audit
        return dict(cached["graph"], checks=[r for r in cached["graph"]["checks"] if r["K"] in steps])
    model_runtime.load, batch_graph.check_graph = load, check
    while not exiting.is_set():
        try:
            command = commands.get(timeout=.5)
        except queue.Empty:
            continue
        campaign = json.loads(Path(command["campaign"]).read_text())
        directory = Path(command["directory"])
        # serve calls load before readiness; instrument it to persist references.
        def row_load(job, device):
            adapter, provider = load(job, device)
            k = campaign["denoising_steps"]
            np.savez(directory / "reference.npz", actions=cached["references"][k])
            base.write(directory / "reference.json", dict(scope="default B1 eager wire; exact CPU text memo", seeds=[501,502,503], rng=cached["receipts"][k]))
            return adapter, provider
        model_runtime.load = row_load
        serve(campaign, command["port"], directory, cuda_graph=True)
        # After the server/core are closed, SIGTERM should also release this
        # owned idle resident, rather than re-stop an already closed loop.
        signal.signal(signal.SIGTERM, lambda *unused: exiting.set())


class ResidentGroup:
    def __init__(self, out, replicas):
        self.processes, self.logs, self.ports, self.children = [], [], [], []
        for i in range(replicas):
            log = (out / f"resident_r{replicas}_{i}.log").open("w")
            self.logs.append(log)
            self.ports.append(base.free_port())
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=base.GPU, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
            self.processes.append(subprocess.Popen([base.PYTHON,"-u",str(Path(__file__).resolve()),"--worker"],
                env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True))
        base.write(out / f"owned_r{replicas}.json", [p.pid for p in self.processes])
    def send(self, index, value):
        self.processes[index].stdin.write(json.dumps(value)+"\n")
        self.processes[index].stdin.flush()
    def start(self, campaign, directory, deadline):
        self.children = []
        for i in range(len(self.processes)):
            child = directory / f"replica{i}"
            child.mkdir()
            self.children.append(child)
            self.send(i, dict(type="next", campaign=str(campaign.resolve()), directory=str(child.resolve()), port=self.ports[i]))
        while not all((child / "ready.json").is_file() for child in self.children):
            if any(p.poll() is not None for p in self.processes) or time.monotonic() > deadline:
                raise RuntimeError(f"resident startup failed/deadline: {directory}")
            time.sleep(.1)
    def stop(self):
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                self.send(i, dict(type="stop"))
        until = time.monotonic()+20
        while not all((child / "closed.json").is_file() for child in self.children):
            if any(p.poll() is not None for p in self.processes) or time.monotonic() > until:
                raise RuntimeError("owned formal server did not close")
            time.sleep(.05)
    def close(self, out):
        rows = []
        for process in self.processes:
            # Children are idle after stop(); terminating them releases their GPU contexts.
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=10)
            rows.append(dict(pid=process.pid, exit_code=code, owned=True))
        base.write(out / f"cleanup_r{len(self.processes)}.json", rows)
        for log in self.logs:
            log.close()


def measure(out, group, maximum, k, clients, seconds, deadline):
    from exp.r17_integration.models.dp.batch_runtime import BatchConfig
    replicas = len(group.processes)
    directory = out / f"k{k}_r{replicas}_b{maximum}_c{clients}"
    directory.mkdir()
    assets = directory / "assets"
    assets.mkdir()
    for filename in ("load.json","model_spec.json","benchmark.json","wire_observations.npz"):
        shutil.copyfile(base.ASSETS / filename, assets / filename)
    shutil.copyfile(base.ASSETS / f"k{k}.yaml", assets / "full.yaml")
    campaign = dict(model="dp",domain="libero",assets=str(assets.resolve()),device="cuda:0",memory_budget_gib=40,
        replicas=replicas,dp_batching=BatchConfig(maximum,5).metadata(),denoising_steps=k,commit_horizon=5,library=None,
        graph_probe_observations=str(base.ASSETS / "wire_observations.npz"))
    base.write(directory / "campaign.json", campaign)
    group.start(directory / "campaign.json", directory, min(deadline,time.monotonic()+90))
    try:
        with np.load(group.children[0] / "reference.npz", allow_pickle=False) as data:
            reference = data["actions"].copy().view(base.References)
        reference.K = k
        for child in group.children[1:]:
            with np.load(child / "reference.npz", allow_pickle=False) as data:
                assert np.array_equal(reference, data["actions"])
        rows, board, elapsed = asyncio.run(base.workload(group.ports, reference, clients, seconds, 50))
        start = rows[0]["measurement_start"]
        timely = [r for r in rows if start <= r["completed_monotonic"] <= start+seconds]
        power = [r for r in board if start <= r["time"] <= start+seconds]
        result = dict(K=k,replicas=replicas,max_batch=maximum,batch_wait_ms=5,clients=clients,calls=len(timely),
            latency_samples=len(rows),measurement_seconds=seconds,drain_seconds=elapsed-seconds,
            calls_per_second=len(timely)/seconds,actions_per_second=5*len(timely)/seconds,
            p50_ms=float(np.percentile([r["latency_ms"] for r in rows],50)),p90_ms=float(np.percentile([r["latency_ms"] for r in rows],90)),
            watts=float(np.mean([r["watts"] for r in power])),max_diff=max(r["max_diff"] for r in rows),
            mean_stage3_batch=float(np.mean([r["batch_size"] for r in rows])),max_stage3_batch=max(r["batch_size"] for r in rows),
            mean_util=float(np.mean([r["util"] for r in power])),peak_board_memory_mib=max(r["memory_mib"] for r in power),
            scope="actual formal WebSocket + S07; completions/board power in fixed window; latency of all window-started calls including drain",
            numerics_reference="three real paired frames/seeds501,502,503/default B1 eager physical actions",quality_sr_claim=False)
        base.write(directory / "calls.json",rows)
        base.write(directory / "power.json",board)
        base.write(directory / "result.json",result)
        print(json.dumps(result),flush=True)
        return result
    finally:
        group.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--replicas",type=int,nargs="+",default=[1,2])
    parser.add_argument("--batches",type=int,nargs="+",default=[1,4,8,16,24,32])
    parser.add_argument("--steps",type=int,nargs="+",default=[100,10,5])
    parser.add_argument("--clients",type=int,default=64)
    parser.add_argument("--seconds",type=float,default=6)
    parser.add_argument("--gpu-budget-seconds",type=int,default=600)
    args=parser.parse_args()
    if args.worker:
        return worker()
    if os.environ.get("R17_DP_FLOCK_OWNED")!="1" or not args.out:
        parser.error("flock ownership and --out required")
    def interrupted(*unused):
        raise TimeoutError("closing only owned residents")
    signal.signal(signal.SIGTERM,interrupted)
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    deadline=started+args.gpu_budget_seconds
    base.write(args.out/"receipt.json",dict(status="RUNNING",pid=os.getpid(),argv=sys.argv,GPU=base.GPU,lock="/tmp/r17_gpu.lock",
        seal=json.loads((base.ASSETS/"receipt.json").read_text()),workspaces_retained_between_rows=True))
    results=[]
    try:
        for replicas in args.replicas:
            group=ResidentGroup(args.out,replicas)
            try:
                for k in args.steps:
                    for batch in args.batches:
                        if time.monotonic()+args.seconds+(3 if "R17_DP_FAST_ARM" in os.environ else 10)>deadline:
                            raise TimeoutError("GPU budget reached")
                        results.append(measure(args.out,group,batch,k,args.clients,args.seconds,deadline))
                        base.write(args.out/"results.json",results)
            finally:
                group.close(args.out)
        status="PASS"
    except BaseException as exc:
        base.write(args.out/"completed.json",dict(status="FAIL",error=repr(exc),rows=len(results),GPU_seconds=time.monotonic()-started,own_children_closed=True))
        raise
    base.write(args.out/"completed.json",dict(status=status,rows=len(results),GPU_seconds=time.monotonic()-started,own_children_closed=True,quality_sr_claim=False))


if __name__=="__main__":
    main()
