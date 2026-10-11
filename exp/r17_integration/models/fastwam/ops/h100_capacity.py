"""Bounded exclusive H100 native/S07 capacity probe. Own children only.

48 closed-loop clients, recorded real LIBERO frames, K10/C5, 50 ms think.
Reuse each resident for its batch/window sweep; recreate S07 between windows.
No simulator workers, formal campaign, or external process is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from . import mps, mps_probe
from exp.r17_integration.ops.formal_process import write_json

GPU = 'GPU-7ea0f760-d3aa-7023-c19b-95f49123311b'
OBS = '/data/mw_finetune/qualification/observations/libero_original256_fix2_v1/fastwam_observations.npz'


def summarize(rows, start, end):
    """Completion rate excludes priming and final drain, both still saved."""
    import numpy as np
    measured = [r for r in rows if start <= r['finished_epoch'] < end]
    if not measured or end <= start:
        raise ValueError('empty measurement window')
    return dict(calls=len(measured), calls_per_second=len(measured)/(end-start),
                decisions_per_second=len(measured)/(end-start),
                p50_ms=float(np.percentile([r['latency_ms'] for r in measured],50)),
                p90_ms=float(np.percentile([r['latency_ms'] for r in measured],90)),
                max_diff=max(r['max_diff'] for r in rows),
                measured_start_epoch=start, measured_end_epoch=end)


def required_memory(replicas, b16_reserved_mib):
    # Larger single-resident batches have different workspaces; they must not
    # prevent admission of a smaller B16 topology that actually fits.
    return replicas*(max(b16_reserved_mib,27000)+1024)+4096


def refresh(descriptor):
    """CPU-only code reseal; non-code digests retained and verified on load."""
    from ..loader import LoadSpec, asset_paths
    from ..manifest import sha256_file
    spec = LoadSpec(**json.loads(Path(descriptor).read_text()))
    paths = asset_paths(spec)
    assets = {k:v for k,v in spec.expected_digests.items() if not k.startswith('code:')}
    if set(assets) != {k for k in paths if not k.startswith('code:')}:
        raise ValueError('non-code asset set changed')
    assets.update({k:sha256_file(p) for k,p in paths.items() if k.startswith('code:')})
    write_json(descriptor, asdict(replace(spec,expected_digests=assets)))


def worker(args):
    import numpy as np
    import torch
    from ..loader import load_adapter, LoadSpec
    from ..batch_runtime import BatchConfig, BatchLedger
    from ..batcher import FastWAMBatchedAdapter
    from ..batch_profile import BoardSampler
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    from openpi.serving.batching_core import BatchingCore
    directory = args.worker
    torch.set_num_threads(1)
    if args.memory_gib:
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    began = time.monotonic()
    adapter = load_adapter(LoadSpec(**json.loads(args.descriptor.read_text())),device='cuda:0')
    print('loaded',time.monotonic()-began,flush=True)
    with np.load(OBS,allow_pickle=False) as data:
        observations = [{k:str(data[k][i]) if k=='prompt' else data[k][i].copy() for k in data.files} for i in range(3)]
    requests = [RequestContext('h100_capacity',adapter.metadata().allowed_benchmark_protocols[0],
        'libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
        1272,i,explicit_seed=42+i) for i in range(3)]
    sampler = adapter.sampler_for_steps(10)
    with torch.inference_mode():
        references = [full_call(adapter,o,r,sampler=sampler)[0].normalized_chunk.float().cpu().numpy()
                      for o,r in zip(observations,requests)]
    np.savez(directory/'reference.npz',actions=np.stack(references))
    for index,(batch,wait,clients) in enumerate(json.loads(args.configs)):
        case = directory/f'case{index}'
        case.mkdir()
        torch.cuda.empty_cache()
        adapter.batch_config = BatchConfig(batch,wait)
        adapter.batch_ledger = BatchLedger(case/'batching.jsonl',adapter.batch_config)
        rows, outputs = [], []
        lock = threading.Lock()
        ready,go = threading.Barrier(clients+1),threading.Event()
        timing = {}
        with BatchingCore(adapter.make_stage_batcher(),device=adapter.model.device,
                         max_batch_size=batch,max_wait_ms=wait) as core:
            def one(client):
                staged = FastWAMBatchedAdapter(adapter,core)
                frame = client%3
                with torch.inference_mode():
                    full_call(staged,observations[frame],requests[frame],sampler=sampler)
                    ready.wait(timeout=90)
                    go.wait(timeout=120)
                    while time.time() < timing['end']:
                        start = time.monotonic()
                        result,_ = full_call(staged,observations[frame],requests[frame],sampler=sampler)
                        action = result.normalized_chunk.float().cpu().numpy()
                        row = dict(client=client,frame=frame,finished_epoch=time.time(),
                            latency_ms=(time.monotonic()-start)*1000,
                            max_diff=float(np.max(np.abs(action-references[frame]))),
                            committed_max_diff=float(np.max(np.abs(action[:5]-references[frame][:5]))),
                            batch_size=result.capture['batching']['stage3']['size'])
                        with lock:
                            rows.append(row);outputs.append(action)
                        time.sleep(.05)
            with ThreadPoolExecutor(clients) as pool:
                tasks = [pool.submit(one,i) for i in range(clients)]
                ready.wait(timeout=120)
                torch.cuda.reset_peak_memory_stats()
                write_json(case/'ready.json',dict(pid=os.getpid(),max_batch=batch,wait_ms=wait,clients=clients))
                while not (case/'GO.json').exists():
                    if time.monotonic()>args.deadline:
                        raise TimeoutError('probe parent deadline')
                    time.sleep(.02)
                timing.update(json.loads((case/'GO.json').read_text()))
                with BoardSampler(GPU) as board:
                    go.set()
                    for future in tasks:
                        future.result(timeout=args.seconds+50)
                result = summarize(rows,timing['start']+3,timing['end'])
                result.update(max_batch=batch,batch_wait_ms=wait,clients=clients,K=10,C=5,
                    think_ms=50,observations_sha256='403187908f0bbbbbc6196b2e7f6d1224acc382ce99dee34d4199dba33cadb3d2',
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    max_committed_diff=max(r['committed_max_diff'] for r in rows),
                    **board.summary(time.monotonic()-(time.time()-timing['start']-3),
                                    time.monotonic()-(time.time()-timing['end'])))
                fills=[]
                for line in (case/'batching.jsonl').read_text().splitlines():
                    r=json.loads(line)
                    if r.get('stage')=='stage3' and timing['start']+3<=r['time']<timing['end']:
                        fills.append(r['size'])
                result['stage3_fills']=fills
                np.savez(case/'actions.npz',actions=np.stack(outputs),frame=np.array([r['frame'] for r in rows]))
                write_json(case/'calls.json',rows)
                write_json(case/'result.json',result)
                print(json.dumps({k:v for k,v in result.items() if k!='stage3_fills'}),flush=True)
    write_json(directory/'closed.json',dict(pid=os.getpid(),status='CLOSED'))


def parent(args):
    import numpy as np
    out=args.out
    out.mkdir(exist_ok=False)
    initial=mps_probe.contexts()
    if initial:
        raise RuntimeError('exclusive H100 probe refused: existing compute contexts')
    started=time.monotonic(); deadline=started+args.budget_seconds
    results=[]
    write_json(out/'receipt.json',dict(initial=initial,budget_seconds=args.budget_seconds,
        started_epoch=time.time(),GPU=GPU,mode='native full MISS through stock S07; local closed-loop clients'))
    peak_single=args.prior_b16_reserved_mib
    groups=[(1,False,[(16,5,48),(24,5,48),(32,5,48),(16,10,48),(24,10,48),(32,10,48)]),
            (2,False,[(16,5,24),(16,10,24)]),(2,True,[(16,5,24),(16,10,24)]),
            (3,True,[(16,5,16)])]
    if args.multi_only:groups=groups[1:]
    try:
        for group,(replicas,enabled,configs) in enumerate(groups):
            root=out/f'g{group}_r{replicas}_{"on" if enabled else "off"}'; root.mkdir()
            free=mps.gpu_snapshot()[0]['free_mib']
            # Admission is based on observed S07 allocator peak, plus context
            # reserve. Do not allocate a third model speculatively into OOM.
            needed=required_memory(replicas,peak_single)
            if free<needed:
                results.append(dict(status='SKIP_MEMORY',replicas=replicas,mps=enabled,
                    max_batch=16,free_mib=free,needed_mib=needed));continue
            if time.monotonic()+100+args.seconds*len(configs)>deadline-25:
                results.append(dict(status='SKIP_BUDGET',replicas=replicas,mps=enabled));continue
            session=None; processes=[]; logs=[]
            try:
                extra={'CUDA_VISIBLE_DEVICES':GPU,'CUDA_MPS_PIPE_DIRECTORY':str(root/'absent')}
                if enabled:
                    session=mps.Session(root)
                    extra=mps.client_env(session.start())
                for replica in range(replicas):
                    child=root/f'replica{replica}';child.mkdir()
                    log=(child/'native.log').open('w');logs.append(log)
                    env={k:v for k,v in os.environ.items() if not k.startswith('CUDA_MPS_')}
                    env.update(extra,OPENPI_DISABLE_STAGE_STREAMS='0',BATCHING_STAGE1_WORKERS='1',
                        BATCHING_STAGE2_WORKERS='1',BATCHING_STAGE3_WORKERS='1')
                    argv=[sys.executable,'-u','-m',__name__ if __name__!='__main__' else
                        'exp.r17_integration.models.fastwam.ops.h100_capacity','--worker',str(child),
                        '--descriptor',str(args.descriptor),'--configs',json.dumps(configs),
                        '--seconds',str(args.seconds),'--deadline',str(deadline),
                        '--memory-gib',str(74/replicas)]
                    process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
                    processes.append(process)
                write_json(root/'owned_pids.json',[dict(pid=p.pid,start_ticks=mps.ticks(p.pid)) for p in processes])
                for index,(batch,wait,clients) in enumerate(configs):
                    cases=[root/f'replica{r}'/f'case{index}' for r in range(replicas)]
                    while not all((c/'ready.json').exists() for c in cases):
                        if any(p.poll() is not None for p in processes):
                            raise RuntimeError('own native child failed; inspect '+str(root))
                        if time.monotonic()>deadline-args.seconds-25:
                            raise TimeoutError('bounded startup deadline')
                        time.sleep(.05)
                    if session and index==0:
                        write_json(root/'attachments.json',mps.account(session.state,[p.pid for p in processes]))
                    when=time.time()+.2
                    for case in cases:
                        write_json(case/'GO.json',dict(start=when,end=when+args.seconds))
                    while not all((c/'result.json').exists() for c in cases):
                        if any(p.poll() is not None and p.returncode for p in processes):
                            raise RuntimeError('own measurement failed; inspect '+str(root))
                        if time.monotonic()>deadline-20:
                            raise TimeoutError('bounded measurement deadline')
                        time.sleep(.1)
                    values=[json.loads((c/'result.json').read_text()) for c in cases]
                    rows=[r for c in cases for r in json.loads((c/'calls.json').read_text())]
                    result=summarize(rows,when+3,when+args.seconds)
                    fills=[f for v in values for f in v['stage3_fills']]
                    result.update(status='PASS',replicas=replicas,mps=enabled,max_batch=batch,
                        batch_wait_ms=wait,clients=clients*replicas,K=10,C=5,case=str(root),
                        fill_p50=float(np.percentile(fills,50)),fill_p90=float(np.percentile(fills,90)),
                        mean_gpu_util=float(np.mean([v['mean_gpu_util_percent'] for v in values])),
                        mean_power_w=float(np.mean([v['mean_power_w'] for v in values])),
                        max_committed_diff=max(v['max_committed_diff'] for v in values),
                        peak_reserved_mib=sum(v['peak_reserved_mib'] for v in values),
                        peak_allocated_mib=sum(v['peak_allocated_mib'] for v in values))
                    if replicas==1 and batch==16: peak_single=max(peak_single,result['peak_reserved_mib'])
                    results.append(result)
                    write_json(out/'results.json',results)
                    print(json.dumps(result),flush=True)
                for process in processes:
                    process.wait(timeout=min(20,max(1,deadline-time.monotonic()-15)))
            finally:
                for process in processes:
                    if process.poll() is None: process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=5)
                for log in logs:log.close()
                write_json(root/'cleanup.json',[dict(pid=p.pid,start_ticks=mps.ticks(p.pid),
                    exit_code=p.returncode,closed=True) for p in processes])
                if session: write_json(root/'mps_closed.json',session.close())
    finally:
        write_json(out/'results.json',results)
        write_json(out/'closed.json',dict(elapsed_seconds=time.monotonic()-started,
            budget_seconds=args.budget_seconds,after=mps_probe.contexts(),status='CLOSED'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--refresh-only',action='store_true')
    p.add_argument('--worker',type=Path)
    p.add_argument('--configs',default='[]')
    p.add_argument('--seconds',type=float,default=23)
    p.add_argument('--budget-seconds',type=float,default=860)
    p.add_argument('--deadline',type=float,default=0)
    p.add_argument('--memory-gib',type=float)
    p.add_argument('--multi-only',action='store_true')
    p.add_argument('--prior-b16-reserved-mib',type=float,default=0)
    args=p.parse_args()
    if args.refresh_only:return refresh(args.descriptor)
    if args.worker:return worker(args)
    if args.out is None or not 60<=args.budget_seconds<=875:
        p.error('new --out and bounded 60..875 second GPU budget required')
    return parent(args)


if __name__=='__main__':main()
