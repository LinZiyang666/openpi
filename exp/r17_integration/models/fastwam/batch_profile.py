"""Bounded local-GPU batch sweep; explicit load, capture off, private RNGs.

Run with flock /tmp/r17_gpu.lock. --seal-only performs only CPU asset reads.
Power/utilization are NVML board measurements, not SM occupancy measurements.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import threading
import time
import uuid


class BoardSampler:
    def __init__(self, expected_uuid):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByUUID(expected_uuid.encode(), ctypes.byref(self.handle)))
        self.samples = []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML status {code}")

    def run(self):
        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        while not self.done.is_set():
            power, util = ctypes.c_uint(), Util()
            a = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
            b = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
            if not a and not b:
                self.samples.append((time.monotonic(), power.value / 1000, util.gpu))
            self.done.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.done.set()
        self.thread.join(2)
        self.lib.nvmlShutdown()

    def summary(self, begin, end):
        rows = [r for r in self.samples if begin <= r[0] <= end]
        return dict(power_samples=len(rows), mean_power_w=sum(r[1] for r in rows)/len(rows) if rows else None,
                    max_power_w=max((r[1] for r in rows), default=None),
                    mean_gpu_util_percent=sum(r[2] for r in rows)/len(rows) if rows else None)


def numerical_deviation(outputs, references):
    import numpy as np
    deltas = np.concatenate([np.abs(o.normalized_chunk.numpy()-r.normalized_chunk.numpy()).ravel()
                             for o, r in zip(outputs, references, strict=True)])
    return {"max_abs": float(deltas.max()), "median_abs": float(np.median(deltas)),
            "p95_abs": float(np.percentile(deltas, 95)), "mean_abs": float(deltas.mean())}


def sweep(adapter, observations, *, deadline, board, repeats=3, batch_sizes=(1,2,4,8,16), receipt_path=None):
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.common.serving import full_call
    from openpi.serving.batching_core import BatchingCore, Stage3MissPayload
    from .batcher import FastWAMBatchedAdapter
    from .batch_runtime import BatchConfig,BatchLedger
    adapter.allow_unqualified_batch = True  # Only this explicit measurement probe.
    ledger=BatchLedger(Path(receipt_path).with_suffix('.batches.jsonl'),BatchConfig(16,5)) if receipt_path else None
    adapter.batch_ledger=ledger
    sampler = adapter.sampler_for_steps(10)
    requests = [RequestContext("fastwam_batch_measurement", adapter.metadata().allowed_benchmark_protocols[0],
        "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        1272, i, explicit_seed=42+i) for i in range(len(observations))]
    report = dict(schema="r17.fastwam.batch_measurement.v1", status="partial", model=asdict(adapter.manifest),
                  capture_enabled=False, sampler=asdict(sampler), profile={}, direct=[], concurrent=[],
                  sm_occupancy=None, sm_occupancy_reason="not collected; NVML utilization is not occupancy")

    def save():
        if receipt_path:
            Path(receipt_path).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def check():
        if time.monotonic() >= deadline - 15:
            raise TimeoutError("bounded GPU slot ending; reserve teardown time")

    def direct(indices):
        ps = [adapter.prepare(observations[i], requests[i]) for i in indices]
        if ledger is not None:
            for j,p in enumerate(ps):
                p.data['batch_member']={'connection_id':f'direct_row_{j}','request_id':uuid.uuid4().hex,
                                        'request':asdict(p.request)}
        batcher = adapter.make_stage_batcher()
        s1 = batcher.run_stage1_batch(ps)
        s2 = batcher.run_stage2_batch(s1)
        noise = [torch.randn((32,7), generator=RequestRNG(p.request).torch_generator()) for p in ps]
        outs = batcher.run_stage3_miss([Stage3MissPayload(s, n, 10) for s,n in zip(s2,noise)],
                                     num_steps=10, save_timesteps_per_request=[None]*len(ps))
        for p, o in zip(ps, outs):
            adapter.to_wire(o.native_result.normalized_chunk, p)
        return [o.native_result for o in outs]

    try:
        with torch.inference_mode():
            check()
            print("warming B1", flush=True)
            direct([0])
            torch.cuda.synchronize()
            references = []
            for i in range(len(observations)):
                check()
                references.append(direct([i])[0])
            phases = {}
            def observer(name, operation):
                torch.cuda.synchronize()
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t = time.monotonic()
                start_event.record()
                result = operation()
                end_event.record()
                end_event.synchronize()
                rows = phases.setdefault(name, [])
                rows.append(dict(wall_ms=(time.monotonic()-t)*1000, device_ms=start_event.elapsed_time(end_event)))
                return result
            adapter.phase_observer = observer
            begin = time.monotonic()
            direct([0])
            torch.cuda.synchronize()
            end = time.monotonic()
            adapter.phase_observer = None
            report['profile'] = dict(total_wall_ms=(end-begin)*1000,
                phases={name:dict(wall_ms=sum(r['wall_ms'] for r in rows),
                                 device_ms=sum(r['device_ms'] for r in rows), calls=len(rows)) for name,rows in phases.items()},
                other_wall_ms=(end-begin)*1000-sum(r['wall_ms'] for rows in phases.values() for r in rows),
                **board.summary(begin,end))
            save()
            print("PROFILE "+json.dumps(report['profile']), flush=True)
            for b in batch_sizes:
                check()
                row = dict(batch=b, status="partial", repeats=repeats, latencies_seconds=[], batch_members=[])
                report['direct'].append(row)
                try:
                    indices=[i%len(observations) for i in range(b)]
                    direct(indices)  # One B-specific warmup, excluded.
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    outputs=[]
                    begin=time.monotonic()
                    for j in range(repeats):
                        check()
                        indices=[(i+j)%len(observations) for i in range(b)]
                        start=time.monotonic()
                        current=direct(indices)
                        torch.cuda.synchronize()
                        row['latencies_seconds'].append(time.monotonic()-start)
                        row['batch_members'].append(indices)
                        outputs.extend(zip(current,[references[i] for i in indices]))
                    end=time.monotonic()
                    row.update(status="passed", decisions_per_second=b*repeats/(end-begin),
                        mean_request_latency_seconds=float(np.mean(row['latencies_seconds'])),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        numerics=numerical_deviation([p[0] for p in outputs],[p[1] for p in outputs]),
                        **board.summary(begin,end))
                except torch.OutOfMemoryError as exc:
                    row.update(status="oom", error=str(exc))
                    gc.collect()
                    torch.cuda.empty_cache()
                save()
                print("DIRECT "+json.dumps(row), flush=True)

            # Real S07 queues: distinct connection adapters, simultaneous calls,
            # producer events, one worker per stage, actual native batches.
            for b in [r['batch'] for r in report['direct'] if r['status']=='passed']:
                check()
                if ledger is not None:
                    ledger.config=BatchConfig(b,5)
                adapter.batch_config=BatchConfig(b,5)
                row=dict(max_batch=b, batch_wait_ms=5, status="partial", repeats=repeats)
                report['concurrent'].append(row)
                with BatchingCore(adapter.make_stage_batcher(), device=adapter.model.device,
                                  max_batch_size=b,max_wait_ms=5) as core:
                    clients=[FastWAMBatchedAdapter(adapter,core) for _ in range(b)]
                    latencies=[]
                    outputs=[]
                    barrier=threading.Barrier(b)
                    def one(i,j):
                        index=(i+j)%len(observations)
                        barrier.wait(timeout=10)
                        t=time.monotonic()
                        result,_=full_call(clients[i], observations[index], requests[index])
                        return result, references[index], time.monotonic()-t
                    with ThreadPoolExecutor(max_workers=b) as pool:
                        for f in [pool.submit(one,i,0) for i in range(b)]:
                            f.result(timeout=180)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        begin=time.monotonic()
                        for j in range(repeats):
                            check()
                            for f in [pool.submit(one,i,j) for i in range(b)]:
                                a, ref, latency=f.result(timeout=180)
                                outputs.append((a,ref));latencies.append(latency)
                        end=time.monotonic()
                    row.update(status="passed",decisions_per_second=b*repeats/(end-begin),
                        mean_request_latency_seconds=float(np.mean(latencies)),
                        p95_request_latency_seconds=float(np.percentile(latencies,95)),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        numerics=numerical_deviation([p[0] for p in outputs],[p[1] for p in outputs]),
                        decision_batches=[{stage:dict(batch_id=receipt['batch_id'],size=receipt['size'],row=receipt['row'],
                                            members=receipt['members']) for stage,receipt in p[0].capture.get('batching',{}).items()}
                                          for p in outputs],
                        mean_actual_batch_size={stage:float(np.mean([p[0].capture['batching'][stage]['size'] for p in outputs]))
                                                for stage in ('stage1','stage2','stage3')} if ledger else {},
                        **board.summary(begin,end))
                save()
                print("CONCURRENT "+json.dumps(row),flush=True)
            report['status']='passed'
    except Exception as exc:
        adapter.phase_observer=None
        report.update(status='partial',error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    save()
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--descriptor',required=True)
    p.add_argument('--benchmark-spec')
    p.add_argument('--seal-only',action='store_true')
    p.add_argument('--observations')
    p.add_argument('--receipt')
    p.add_argument('--expected-gpu-uuid')
    p.add_argument('--deadline-seconds',type=float,default=1400)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--refresh-code-seal',action='store_true',
                   help='rehash current source files; all weight/assets are still verified by load_adapter')
    a=p.parse_args(argv)
    from .loader import LoadSpec,load_adapter,seal_descriptor
    if a.seal_only:
        from exp.r17_integration.conductor.manifest import load_benchmark_spec
        benchmark=load_benchmark_spec(a.benchmark_spec)
        spec=LoadSpec('libero','/data/mw_finetune/models/fastwam_libero','/data/xwam/wan22_5b',
                     benchmark_protocols=(benchmark.identity,),benchmark_specs=(benchmark.manifest(),))
        spec=replace(spec,expected_digests=seal_descriptor(spec))
        Path(a.descriptor).write_text(json.dumps(asdict(spec),indent=2)+'\n')
        print('CPU sealed descriptor',flush=True)
        return
    if not a.expected_gpu_uuid or not a.observations or not a.receipt or not 0<a.deadline_seconds<=1450:
        p.error('bounded explicit local GPU UUID, observations and receipt required')
    import torch
    import numpy as np
    begin=time.monotonic()
    spec=LoadSpec(**json.loads(Path(a.descriptor).read_text()))
    if a.refresh_code_seal:
        from .loader import asset_paths
        from .manifest import sha256_file
        paths=asset_paths(spec)
        assets={key:value for key,value in spec.expected_digests.items() if not key.startswith('code:')}
        if set(assets)!={key for key in paths if not key.startswith('code:')}:
            pass
        assets.update({key:sha256_file(path) for key,path in paths.items() if key.startswith('code:')})
        spec=replace(spec,expected_digests=assets)
        Path(a.descriptor).write_text(json.dumps(asdict(spec),indent=2)+'\n')
    if spec.benchmark!='libero':
        p.error('this measurement is LIBERO only')
    with BoardSampler(a.expected_gpu_uuid) as board:
        print('loading sealed Fast-WAM',flush=True)
        adapter=load_adapter(spec,device='cuda:0')
        load_seconds=time.monotonic()-begin
        print(f'loaded in {time.monotonic()-begin:.1f}s; allocated {torch.cuda.memory_allocated()/2**30:.2f}GiB',flush=True)
        with np.load(a.observations,allow_pickle=False) as data:
            observations=[{k:str(data[k][i]) if k=='prompt' else data[k][i].copy() for k in data.files}
                          for i in range(len(data['state']))]
        report=sweep(adapter,observations,deadline=begin+a.deadline_seconds,board=board,
                     repeats=a.repeats,receipt_path=a.receipt)
        report.update(load_seconds=load_seconds,
                      elapsed_with_load_seconds=time.monotonic()-begin,gpu_uuid=a.expected_gpu_uuid,
                      gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__)
        Path(a.receipt).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        del adapter
        gc.collect();torch.cuda.empty_cache()
    print('GPU measurement closed normally',flush=True)


if __name__=='__main__':
    main()
