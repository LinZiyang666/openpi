"""Bounded CPU handler replay, real /data journals and concurrent fake workers.

The old handlers are extracted verbatim from the baseline's sealed runtime,
including the scheduler lock and per-row fsync. No latency is injected.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tarfile
import threading
import time
from types import SimpleNamespace

from openpi.conductor.task import EpisodeTask, ServerEndpoint
from exp.r17_integration.conductor.manifest import EndpointSpec
from exp.r17_integration.conductor.strategy import MetaWorldStrategy
from exp.r17_integration.qualification.driver import ProbeRecordingDriver
from .formal_driver_fast import AsyncCampaignDriver
from .formal_process import write_json


def legacy_handlers(source, directory):
    tree=ast.parse(source)
    body=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run').body
    selected=[node for node in body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and
              node.name in ('append','CampaignDriver','steps')]
    if len(selected)!=3:pass
    namespace=dict(directory=directory,campaign={},ProbeRecordingDriver=ProbeRecordingDriver,json=json,os=os,time=time)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'sealed_baseline_handlers.py','exec'),namespace)
    return namespace['CampaignDriver'],namespace['steps']


def percentile(values,p):
    ordered=sorted(values)
    return round(ordered[min(len(ordered)-1,int((len(ordered)-1)*p))]*1000,4) if ordered else None


def replay(directory,source,mode,*,workers=24,episodes=96,steps=48):
    directory.mkdir(exist_ok=False)
    endpoint=ServerEndpoint('fake',1234)
    tasks=[EpisodeTask(f'yaml:eval:0:{i}','yaml','eval','libero_spatial',i%10,i//10,i,endpoint.host,
        endpoint.port,'formal_full_v1',extra=dict(model_identity='model',norm_identity='norm',layout_identity='layout',
        benchmark_protocol='benchmark',num_steps=100,original_init_idx=i//10)) for i in range(episodes)]
    endpoints=[EndpointSpec(endpoint,'model','norm','layout','benchmark',allowed_k=(100,),capacity=workers)]
    strategy=MetaWorldStrategy(tasks,endpoints=endpoints)
    kwargs=dict(dispatch_path=directory/'dispatch.jsonl',rejection_path=directory/'rejections.json',
        expected_workers=workers,shutdown_seconds=.1,yaml_weights={'yaml':1},servers=[endpoint],
        journal_path=str(directory/'journal.jsonl'),ctl_factory=lambda ep:SimpleNamespace(close=lambda:None),
        scheduler_kwargs=dict(eval_concurrency=workers,max_episode_retries=2))
    if mode=='legacy':
        cls,writer=legacy_handlers(source,directory);kwargs['per_step_writer']=writer
    else:cls=AsyncCampaignDriver;kwargs['evidence_directory']=directory
    driver=cls(strategy,**kwargs);driver.drive_stages_once()
    barrier=threading.Barrier(workers+1);latencies=[];reports=[];cycles=[];errors=[];counts=[]
    metrics_lock=threading.Lock();deadline=time.monotonic()+180
    def worker(index):
        try:
            driver._shutdown_local.worker_id='fake_'+str(index)
            barrier.wait();last_report=None;completed=0
            while time.monotonic()<deadline:
                started=time.monotonic();kind,payload=driver.handle_pull(endpoint.key);end=time.monotonic()
                if 'task' not in payload:
                    if driver.scheduler.all_done():break
                    time.sleep(.002);continue
                with metrics_lock:
                    latencies.append(end-started)
                    if last_report is not None:cycles.append(end-last_report)
                task=payload['task']
                time.sleep(.003) # CPU fake episode work; NOT an injected I/O delay.
                rows=[dict(decision_seq=i,step_idx=i*5,actual_nfe=100,source='policy',stage_calls=[1,1,100],
                    capture_enabled=False,service_ms=400.,queue_wait_ms=10.,requested_k=100,executed_nfe=100,
                    model_identity='model',norm_identity='norm',layout_identity='layout',benchmark_protocol='benchmark')
                    for i in range(steps)]
                last_report=time.monotonic()
                driver.handle_result(dict(task_uid=task['task_uid'],attempt=task['attempt'],success=True,n_steps=steps*5,
                    duration_s=.003,per_step_rows=rows))
                with metrics_lock:reports.append(time.monotonic()-last_report)
                completed+=1
            else:raise TimeoutError('CPU replay deadline')
            counts.append(completed)
        except BaseException as exc:errors.append(repr(exc))
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(workers)]
    for thread in threads:thread.start()
    started=time.monotonic();barrier.wait()
    while any(thread.is_alive() for thread in threads) and time.monotonic()<deadline:
        driver.drive_stages_once();time.sleep(.005)
    for thread in threads:thread.join(1)
    stats=driver.close();elapsed=time.monotonic()-started
    if errors or any(thread.is_alive() for thread in threads):raise RuntimeError(errors or 'CPU worker did not close')
    records=[json.loads(line) for line in (directory/'journal.jsonl').read_text().splitlines()]
    accepted=[r for r in records if r['accepted']]
    if len(accepted)!=episodes or len({r['task_uid'] for r in accepted})!=episodes:
        pass
    per_step=[json.loads(line) for line in (directory/'per_step.jsonl').read_text().splitlines()]
    if len(per_step)!=episodes*steps or any(not r['accepted'] for r in per_step):pass
    report=dict(mode=mode,workers=workers,episodes=episodes,steps_per_episode=steps,seconds=elapsed,
        episodes_per_second=episodes/elapsed,all_cpu_workers_closed=True,workers_with_assignments=sum(n>0 for n in counts),
        fsyncs=episodes*(steps+3) if mode=='legacy' else stats['fsyncs'],group_commit=stats,
        pull_ms={f'p{int(p*100)}':percentile(latencies,p) for p in (.5,.95,.99)},
        reporting_ms={f'p{int(p*100)}':percentile(reports,p) for p in (.5,.95,.99)},
        reporting_to_next_assignment_ms={f'p{int(p*100)}':percentile(cycles,p) for p in (.5,.95,.99)},
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        input_task_sha256=hashlib.sha256(json.dumps([asdict(t) for t in tasks],sort_keys=True).encode()).hexdigest(),
        outcomes_sha256=hashlib.sha256(json.dumps(sorted((r['task_uid'],r['attempt'],r['success']) for r in accepted)).encode()).hexdigest())
    write_json(directory/'REPLAY.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--episodes',type=int,default=96)
    parser.add_argument('--steps',type=int,default=48)
    args=parser.parse_args(argv)
    if args.workers<2 or not args.workers<=args.episodes<=512 or not 1<=args.steps<=128:
        parser.error('bounded replay: 2+ workers, workers<=episodes<=512, 1..128 steps')
    with tarfile.open(args.baseline/'runtime.tar') as archive:
        source=archive.extractfile('exp/r17_integration/ops/formal_driver.py').read().decode()
    args.output.mkdir(exist_ok=False)
    results=[replay(args.output/mode,source,mode,workers=args.workers,episodes=args.episodes,steps=args.steps)
             for mode in ('legacy','group-commit')]
    if len({r['input_task_sha256'] for r in results})!=1 or len({r['outcomes_sha256'] for r in results})!=1:
        pass
    write_json(args.output/'RESULTS.json',dict(status='PASS',cpu_only=True,latency_injection=False,results=results))
    print(json.dumps(results))


if __name__=='__main__':main()
