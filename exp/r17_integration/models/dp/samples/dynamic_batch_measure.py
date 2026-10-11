"""Bounded, flock-owned GPU sweep of actual DP formal WebSocket residents.

No simulator fleets are launched. The CPU clients replay three sealed real
paired frames, with independent fixed request seeds and actual execution acks.
Only child PIDs started here are signalled. Run under flock /tmp/r17_gpu.lock.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np

GPU = "GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9"
ASSETS = Path("exp/r17_integration/models/dp/samples/dynamic_batch_assets").resolve()
PYTHON = "/data/mw_finetune/dp/env/bin/python"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def observations():
    with np.load(ASSETS / "wire_observations.npz", allow_pickle=False) as data:
        return [{key: str(data[key][i]) if key == "prompt" else data[key][i].copy()
                 for key in data.files} for i in range(3)]


def worker(campaign_path, port, directory):
    # Instrument only this owned measurement child, before the exact formal
    # entry runs; its load/factory/queue/server implementations stay unchanged.
    from exp.r17_integration.qualification import model_runtime
    original = model_runtime.load
    campaign = json.loads(Path(campaign_path).read_text())
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    def load(job, device):
        adapter, provider = original(job, device)
        from exp.r17_integration.common.contracts import RequestContext, CaptureSpec
        from exp.r17_integration.models.dp.request_noise import RequestNoise
        from exp.r17_integration.models.dp.sampler import SamplerProfile
        import torch
        refs, receipts = [], []
        with torch.inference_mode():
            for i, obs in enumerate(observations()):
                context = RequestContext("dp_batch_measure", adapter.metadata().allowed_benchmark_protocols[0],
                    "fixed_real_frame", i, 0, explicit_seed=501+i)
                s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(obs, context)))
                result = adapter.sample(s2, SamplerProfile(num_steps=campaign["denoising_steps"]),
                    RequestNoise(501+i, adapter.device), CaptureSpec())
                refs.append(adapter.to_wire(result.normalized_chunk, None).actions)
                receipts.append(result.rng_receipt)
        np.savez(out / "reference.npz", actions=np.stack(refs))
        write(out / "reference.json", dict(scope="default B1 eager wire actions", seeds=[501,502,503], rng=receipts))
        return adapter, provider
    model_runtime.load = load
    from exp.r17_integration.models.dp.formal_batch_server import serve
    serve(campaign, port, out, cuda_graph=True)


class BoardPower:
    def __init__(self):
        self.rows, self.stop = [], threading.Event()
    def __enter__(self):
        def run():
            while not self.stop.is_set():
                begin = time.monotonic()
                text = subprocess.check_output(["nvidia-smi", "-i", GPU,
                    "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
                power, util, memory = map(float, text.strip().split(","))
                self.rows.append(dict(time=(begin+time.monotonic())/2, watts=power, util=util, memory_mib=memory))
                self.stop.wait(.25)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(5)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def workload(ports, references, count, seconds, think_ms):
    from websockets.asyncio.client import connect
    from openpi_client import msgpack_numpy
    frames = observations()
    benchmark = json.loads((ASSETS / "benchmark.json").read_text())
    packer = msgpack_numpy.Packer()
    ready, begin = asyncio.Event(), {}
    entered, rows, errors = [], [], []
    async def client(index):
        row = index % 3
        ws = await connect(f"ws://127.0.0.1:{ports[index%len(ports)]}", compression=None, max_size=None, ping_timeout=None)
        async def exchange(message):
            await ws.send(packer.pack(message))
            value = await ws.recv()
            if isinstance(value, str):
                raise RuntimeError(value)
            value = msgpack_numpy.unpackb(value)
            if value.get("__ack__") == "error":
                raise RuntimeError(value)
            return value
        try:
            metadata = msgpack_numpy.unpackb(await ws.recv())
            assert metadata["cuda_graph_self_check"]["passed"]
            extra = dict(campaign="dp_batch_measure", task_name=benchmark["tasks"][0][0], task_id=0,
                original_init_idx=index%50, seed=7, explicit_seed=501+row, ack_protocol="r17.execution.v1",
                task_table_identity=benchmark["task_table_identity"], arm="matched_batch_load")
            await exchange(dict(__ctrl__="episode_start", __extra__=extra, __episode_id__=index))
            sequence = None
            async def infer():
                nonlocal sequence
                obs = dict(frames[row], __request__={"explicit_seed":501+row})
                if sequence is not None:
                    obs["__execution__"] = dict(decision_seq=sequence, executed_count=5)
                start = time.monotonic()
                value = await exchange(obs)
                elapsed = time.monotonic()-start
                hit = value["__hit_meta__"]
                sequence = hit["decision_seq"]
                assert value["actions"].shape == (5, 7)
                assert hit["actual_nfe"] == begin["K"]
                stage3 = hit["batching"]["stages"]["stage3"]
                return dict(latency_ms=elapsed*1000, started_monotonic=start, completed_monotonic=time.monotonic(),
                    max_diff=float(np.max(np.abs(value["actions"]-references[row,:5]))),
                    batch_size=stage3["size"], batch_id=stage3["batch_id"], client=index)
            # Concurrent warmup captures encountered signatures; excluded.
            await infer()
            entered.append(index)
            if len(entered) == count:
                ready.set()
            await ready.wait()
            while "deadline" not in begin:
                await asyncio.sleep(.005)
            while time.monotonic() < begin["deadline"]:
                rows.append(await infer())
                await asyncio.sleep(think_ms/1000)
            await exchange(dict(__ctrl__="episode_end", __execution__=dict(decision_seq=sequence, executed_count=5)))
        except BaseException as exc:
            errors.append(repr(exc))
            ready.set()
            raise
        finally:
            await ws.close()
    tasks = [asyncio.create_task(client(i)) for i in range(count)]
    begin["K"] = references.K
    await asyncio.wait_for(ready.wait(), timeout=90)
    if errors:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise RuntimeError(errors)
    start = time.monotonic()
    with BoardPower() as board:
        begin["deadline"] = start+seconds
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=seconds+90)
        end = time.monotonic()
    for row in rows:
        row.update(measurement_start=start, measurement_deadline=begin["deadline"])
    return rows, board.rows, end-start


class References(np.ndarray):
    """Carries the expected measured arm to the client without globals."""
    pass


def measure(out, replicas, maximum, k, clients, seconds, think_ms, deadline):
    from exp.r17_integration.models.dp.batch_runtime import BatchConfig
    name = f"k{k}_r{replicas}_b{maximum}_c{clients}"
    directory = out / name
    directory.mkdir(parents=True, exist_ok=False)
    assets = directory / "assets"
    assets.mkdir()
    for filename in ("load.json", "model_spec.json", "benchmark.json", "wire_observations.npz"):
        shutil.copyfile(ASSETS / filename, assets / filename)
    shutil.copyfile(ASSETS / f"k{k}.yaml", assets / "full.yaml")
    campaign = dict(model="dp", domain="libero", assets=str(assets.resolve()), device="cuda:0",
        memory_budget_gib=40, replicas=replicas, dp_batching=BatchConfig(maximum, 5).metadata(),
        denoising_steps=k, commit_horizon=5, library=None, graph_probe_observations=str(ASSETS / "wire_observations.npz"))
    write(directory / "campaign.json", campaign)
    processes, logs, ports = [], [], []
    try:
        for replica in range(replicas):
            child = directory / f"replica{replica}"
            child.mkdir()
            port = free_port()
            ports.append(port)
            log = (child / "native.log").open("w")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
            process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker",
                str((directory / "campaign.json").resolve()), "--port", str(port), "--directory", str(child.resolve())],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
        write(directory / "owned_pids.json", [p.pid for p in processes])
        wait_deadline = min(time.monotonic()+90, deadline)
        while not all((directory / f"replica{i}" / "ready.json").is_file() for i in range(replicas)):
            if any(p.poll() is not None for p in processes) or time.monotonic() > wait_deadline:
                raise RuntimeError(f"resident did not become ready: {directory}")
            time.sleep(.1)
        with np.load(directory / "replica0/reference.npz", allow_pickle=False) as data:
            reference = data["actions"].copy().view(References)
        reference.K = k
        for i in range(1, replicas):
            with np.load(directory / f"replica{i}/reference.npz", allow_pickle=False) as data:
                assert np.array_equal(reference, data["actions"]), "replica B1 references differ"
        rows, power, elapsed = asyncio.run(workload(ports, reference, clients, seconds, think_ms))
        latencies = [r["latency_ms"] for r in rows]
        result = dict(K=k, replicas=replicas, max_batch=maximum, batch_wait_ms=5, clients=clients,
            calls=len(rows), elapsed_seconds=elapsed, calls_per_second=len(rows)/elapsed,
            actions_per_second=5*len(rows)/elapsed, p50_ms=float(np.percentile(latencies, 50)),
            p90_ms=float(np.percentile(latencies, 90)), watts=float(np.mean([r["watts"] for r in power])),
            max_diff=max(r["max_diff"] for r in rows), mean_stage3_batch=float(np.mean([r["batch_size"] for r in rows])),
            max_stage3_batch=max(r["batch_size"] for r in rows), mean_util=float(np.mean([r["util"] for r in power])),
            peak_board_memory_mib=max(r["memory_mib"] for r in power), think_ms=think_ms,
            scope="actual formal resident / shared stock websocket + S07 / closed loop / local transport / real paired frames",
            quality_sr_claim=False)
        write(directory / "calls.json", rows)
        write(directory / "power.json", power)
        write(directory / "result.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        # Every handle here was created above; no discovery/signalling of others.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        closed = []
        for process in processes:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=10)
            closed.append(dict(pid=process.pid, exit_code=code, owned=True))
        write(directory / "cleanup.json", closed)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--port", type=int)
    parser.add_argument("--directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[100,10,5])
    parser.add_argument("--replicas", type=int, nargs="+", default=[1,2])
    parser.add_argument("--batches", type=int, nargs="+", default=[1,4,8,16,24,32])
    parser.add_argument("--clients", type=int, default=64)
    parser.add_argument("--think-ms", type=float, default=50)
    parser.add_argument("--gpu-budget-seconds", type=int, default=1100)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.port, args.directory)
    if not args.out or os.environ.get("R17_DP_FLOCK_OWNED") != "1":
        parser.error("--out and R17_DP_FLOCK_OWNED=1 inside flock are required")
    def interrupted(*unused):
        raise TimeoutError("owned measurement interrupted; closing only owned children")
    signal.signal(signal.SIGTERM, interrupted)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+args.gpu_budget_seconds
    info = subprocess.check_output(["nvidia-smi", "-i", GPU,
        "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    write(args.out / "receipt.json", dict(status="RUNNING", GPU=info, started=time.time(), argv=sys.argv,
        seal=json.loads((ASSETS / "receipt.json").read_text()), lock="/tmp/r17_gpu.lock", budget_seconds=args.gpu_budget_seconds))
    results = []
    try:
        for k in args.steps:
            for replicas in args.replicas:
                for batch in args.batches:
                    if time.monotonic()+args.seconds+100 > deadline:
                        raise TimeoutError("bounded GPU measurement budget reached")
                    results.append(measure(args.out, replicas, batch, k, args.clients, args.seconds, args.think_ms, deadline))
                    write(args.out / "results.json", results)
        write(args.out / "completed.json", dict(status="PASS", rows=len(results), quality_sr_claim=False,
            GPU_seconds=time.monotonic()-started, own_children_closed=True))
    except BaseException as exc:
        write(args.out / "completed.json", dict(status="FAIL", error=repr(exc), rows=len(results),
            GPU_seconds=time.monotonic()-started, own_children_closed=True))
        raise


if __name__ == "__main__":
    main()
