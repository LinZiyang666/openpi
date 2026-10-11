"""Two new Fast-WAM B8 residents vs unchanged live Cosmos contexts, bounded.

Standalone deployed under a new probe directory; existing release is read-only.
Uses P3 native S07 load generation, private MPS, and 16 clients per resident.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from exp.r17_integration.models import fastwam


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',type=Path)
    p.add_argument('--seconds',type=float,default=16)
    p.add_argument('--budget-seconds',type=float,default=300)
    args=p.parse_args()
    OUT=args.out
    fastwam.__path__.insert(0,str(OUT/'code/fastwam'))
    from exp.r17_integration.models.fastwam.ops import mps,mps_probe
    from exp.r17_integration.ops.formal_process import write_json

    GPU='GPU-7ea0f760-d3aa-7023-c19b-95f49123311b'
    DESC='/data/mw_finetune/eval/hs3_fast_b8w5_h100_k10_c5_control_20261010_125303/assets/load.json'
    COSMOS='/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043'

    if args.worker:
        mps_probe.fast_worker(args.worker,16,args.seconds,
            '/data/mw_finetune/qualification/observations/libero_original256_fix2_v1/fastwam_observations.npz',
            DESC,gpu_uuid=GPU,start_file=args.worker.parent/'GO',memory_gib=29)
        raise SystemExit(0)

    started=time.monotonic()
    deadline=started+args.budget_seconds
    results=[]
    initial=mps_probe.background([COSMOS])
    write_json(OUT/'receipt.json',dict(before=initial,started=time.time(),budget_seconds=args.budget_seconds,
        topology='two owned Fast-WAM B8 / 16 clients each + unchanged live Cosmos2 outside private MPS',GPU=GPU))
    try:
        for index,enabled in enumerate((False,True,False,True)):
            if time.monotonic()+args.seconds+100>deadline-20:
                break
            case=OUT/f'{index:02d}_{"on" if enabled else "off"}'
            case.mkdir(exist_ok=False)
            free=mps.gpu_snapshot()[0]['free_mib']
            if free<2*29*1024+4096:
                results.append(dict(status='SKIP_MEMORY',free_mib=free,mps=enabled));break
            session=None
            processes=[]
            logs=[]
            try:
                extra={}
                if enabled:
                    session=mps.Session(case)
                    extra=mps.client_env(session.start())
                else:
                    extra['CUDA_MPS_PIPE_DIRECTORY']=str(case/'nonexistent')
                for replica in range(2):
                    child=case/f'replica{replica}';child.mkdir()
                    log=(child/'native.log').open('w');logs.append(log)
                    env=dict(os.environ,**extra,OPENPI_DISABLE_STAGE_STREAMS='0',BATCHING_STAGE1_WORKERS='1',
                        BATCHING_STAGE2_WORKERS='1',BATCHING_STAGE3_WORKERS='1')
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(OUT),
                        '--worker',str(child),'--seconds',str(args.seconds)],env=env,stdout=log,stderr=subprocess.STDOUT)
                    processes.append(process)
                write_json(case/'owned_pids.json',[dict(pid=p.pid,start_ticks=mps.ticks(p.pid)) for p in processes])
                while not all((case/f'replica{i}'/'ready.json').exists() for i in range(2)):
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError('owned H100 native child failed; inspect '+str(case))
                    if time.monotonic()>deadline-25:
                        raise TimeoutError('bounded H100 startup deadline')
                    time.sleep(.1)
                before=mps_probe.background([COSMOS])
                if session:
                    write_json(case/'attachment.json',mps.account(session.state,[p.pid for p in processes]))
                (case/'GO').touch()
                for process in processes:
                    process.wait(timeout=min(args.seconds+60,max(1,deadline-time.monotonic()-15)))
                    if process.returncode:
                        raise RuntimeError('owned H100 native child failed; inspect '+str(case))
                after=mps_probe.background([COSMOS])
                rows=[json.loads((case/f'replica{i}'/'result.json').read_text()) for i in range(2)]
                elapsed=max(r['ended_epoch'] for r in rows)-min(r['started_epoch'] for r in rows)
                result=dict(mps=enabled,status='PASS',replicas=2,max_batch=8,clients=32,case=str(case),
                    calls=sum(r['calls'] for r in rows),calls_per_second=sum(r['calls'] for r in rows)/elapsed,
                    elapsed_seconds=elapsed,max_diff=max(r['max_diff'] for r in rows),
                    background_calls_per_second=mps_probe.rates(before,after),replica_results=rows)
                results.append(result)
                print(json.dumps(result),flush=True)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=15)
                for log in logs:log.close()
                write_json(case/'cleanup.json',[dict(pid=p.pid,exit_code=p.returncode,closed=True) for p in processes])
                if session:write_json(case/'mps_closed.json',session.close())
            write_json(OUT/'results.json',results)
    finally:
        write_json(OUT/'results.json',results)
        final=mps_probe.background([COSMOS])
        before={(r['pid'],r['start_ticks'],r['gpu']) for r in initial['contexts']}
        after={(r['pid'],r['start_ticks'],r['gpu']) for r in final['contexts']}
        write_json(OUT/'closed.json',dict(elapsed_seconds=time.monotonic()-started,results=len(results),
            budget_seconds=args.budget_seconds,after=final,background_contexts_preserved=before<=after))


if __name__ == "__main__":
    main()
