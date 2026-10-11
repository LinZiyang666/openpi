"""One-GPU formal queue with one campaign of CPU preparation ahead of dispatch.

Jobs are a JSON array of argv arrays (no shell evaluation), each with a unique
--run-name. Use --paired-control @earlier_run_name to reference a queued control.
--prepare-only stages the list without spawning replicas, workers or tunnels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from . import formal_eval as launch
from .formal_process import alive, stop_group, ticks, write_json


def jobs_from_file(path):
    jobs=json.loads(Path(path).read_text())
    if not isinstance(jobs,list) or not jobs:
        raise ValueError('nonempty JSON array of argument arrays required')
    result, names, gpu = [], set(), None
    for job in jobs:
        if not isinstance(job,list) or not all(isinstance(x,str) for x in job):
            raise ValueError('each campaign must be an argv array')
        if any(x.split('=')[0] in ('--run','--prepare-only','--prepared-run','--output','--defer-control-check') for x in job):
            raise ValueError('queue owns run/prepare/output flags')
        job=list(job)
        for index,token in enumerate(job):
            if token.startswith('@'):
                if index==0 or job[index-1]!='--paired-control' or token[1:] not in names:
                    raise ValueError('control reference must name an earlier queued campaign')
                job[index]=str(launch.FORMAL_ROOT/token[1:])
        args=launch.parse_args([*job,'--prepare-only'])
        if args.paired_control and args.runtime_from is None and args.resume_from is None:
            job += ['--runtime-from',str(args.paired_control)]
            args.runtime_from = args.paired_control
        name=args.run_name
        if not name or name in names or len(name)>150 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('each queue campaign needs a unique plain --run-name')
        target=(args.server_node,args.gpu)
        if gpu is not None and target!=gpu:
            pass
        gpu=target
        names.add(name)
        result.append(dict(name=name,argv=job,args=args,run=str(launch.FORMAL_ROOT/name)))
    return result


class Runner:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.lock=threading.Lock()
        self.children=[]
        self.stopped=False
        self.events=[]

    def event(self,**fields):
        with self.lock:
            self.events.append(dict(time=time.time(),**fields))
            write_json(self.directory/'QUEUE_EVENTS.json',self.events)

    def invoke(self,job,phase,argv):
        env=dict(os.environ,PYTHONPATH=launch.pythonpath(str(launch.ROOT)),OMP_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        if phase=='prepare':
            env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
        # GPU execution requires the explicit queue --run, and the launcher's
        # per-resident device selection. The coordinator itself remains CPU.
        log=self.directory/(job['name']+'_'+phase+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen([launch.CONTROL_PYTHON,'-u','-m','exp.r17_integration.ops.formal_eval',*argv],
                cwd=launch.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry=dict(pid=child.pid,pgid=child.pid,start_ticks=ticks(child.pid),phase=phase,campaign=job['name'])
            with self.lock:
                self.children.append((child,entry))
                write_json(self.directory/'QUEUE_OWNED.json',[e for _,e in self.children])
            self.event(event='start',**entry)
            bound=time.monotonic()+(3600 if phase=='prepare' else job['args'].seconds+1200)
            while child.poll() is None:
                if self.stopped or time.monotonic()>=bound:
                    if phase=='prepare':
                        # CPU seal/transfer children share this owned session.
                        # Fence its leader before signalling the recorded pgid.
                        stop_group(entry)
                    raise TimeoutError('queue stopped or owned '+phase+' deadline exceeded')
                time.sleep(.2)
            entry['exit_code']=child.returncode
            self.event(event='exit',**entry)
            if child.returncode:
                raise RuntimeError(job['name']+' '+phase+' failed; see '+str(log))
        return job['run']

    def close(self):
        # Signal only receipt-bound direct children. Formal launchers own and
        # close their own remote fleets/residents; never scan process patterns.
        self.stopped=True
        for child,entry in self.children:
            if child.poll() is None and alive(entry):
                if entry['phase']=='prepare':
                    stop_group(entry)
                else:
                    child.terminate()
        deadline=time.monotonic()+600
        while any(child.poll() is None for child,_ in self.children) and time.monotonic()<deadline:
            time.sleep(.2)
        closed=[]
        for child,entry in self.children:
            entry['exit_code']=child.poll()
            closed.append(dict(entry,closure_confirmed=not alive(entry)))
        write_json(self.directory/'QUEUE_CLEANUP.json',closed)
        return all(r['closure_confirmed'] for r in closed)


def prepare_job(runner,job):
    argv=[*job['argv'],'--prepare-only']
    if job['args'].paired_control:
        argv.append('--defer-control-check')
    result=runner.invoke(job,'prepare',argv)
    launch_directory=Path(result)
    from .formal_prestage import validate_prepared
    validate_prepared(launch_directory)
    return result


def check_seals(results, runner, *, wait=False):
    """Switch after GPU release, stop on late failure, seal queue after all pulls."""
    from .formal_finalize import wait_seal
    for result in results:
        path=Path(result['run'])
        receipt=json.loads((path/'RECEIPT.json').read_text())
        if receipt.get('status')=='PENDING_EVIDENCE' and not wait:
            continue
        seal=wait_seal(path,stopped=lambda:runner.stopped)
        result.update(status='PASS',evidence_seal=seal)


def run_jobs(jobs,runner,prepare_only=False,startup_retries=2):
    results=[]
    if prepare_only:
        for job in jobs:
            results.append(dict(name=job['name'],prepared=prepare_job(runner,job)))
        return results
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='r17_cpu_prepare') as pool:
        try:
            pending=pool.submit(prepare_job,runner,jobs[0])
            for index,job in enumerate(jobs):
                check_seals(results,runner)
                prepared=pending.result()
                check_seals(results,runner)
                pending=pool.submit(prepare_job,runner,jobs[index+1]) if index+1<len(jobs) else None
                from .queue_retry import run_job
                results.append(run_job(dict(job,run=prepared),runner,limit=startup_retries))
            check_seals(results,runner,wait=True)
        except BaseException:
            runner.stopped=True
            if pending is not None:
                pending.cancel()
            raise
    return results


def prepared_jobs(directory):
    """Reuse an all-CPU prepared queue without rebuilding occupied run dirs."""
    from .formal_prestage import validate_prepared
    directory=Path(directory).resolve(strict=True)
    receipt=json.loads((directory/'QUEUE_RECEIPT.json').read_text())
    plan=json.loads((directory/'QUEUE_PLAN.json').read_text())
    if receipt.get('status')!='PASS' or not receipt.get('cpu_only') or not receipt.get('all_owned_closed'):
        pass
    jobs=plan['jobs']
    if len(jobs)!=len(receipt['results']) or not jobs:
        pass
    gpu=None
    for job,result in zip(jobs,receipt['results']):
        if job['name']!=result['name'] or job['run']!=result['prepared']:
            pass
        campaign,_=validate_prepared(job['run'])
        if campaign['run_name']!=job['name'] or Path(job['run']).resolve()!=launch.FORMAL_ROOT/job['name']:
            pass
        target=(campaign['server_node'],campaign['device'])
        if gpu is not None and target!=gpu:
            pass
        gpu=target
        # The preparation seal already binds the complete configuration. Do
        # not re-interpret frozen arms through subsequently edited model CLIs.
        job['args']=argparse.Namespace(seconds=campaign['seconds'])
    return jobs


def run_prepared_jobs(jobs,runner,startup_retries=2):
    results=[]
    for job in jobs:
        check_seals(results,runner)
        from .queue_retry import run_job
        results.append(run_job(job,runner,limit=startup_retries,initial_failure=job.get('initial_failure',False)))
    check_seals(results,runner,wait=True)
    return results


def refresh_prepared_queue(directory,jobs,mode):
    """Refresh only CPU orchestration; keep the original queue plan/receipt."""
    from .formal_prestage import refresh_driver
    directory=Path(directory)
    stamp=str(time.time_ns())
    for name in ('QUEUE_PLAN.json','QUEUE_RECEIPT.json'):
        write_json(directory/(name+'.before_driver_'+stamp),json.loads((directory/name).read_text()))
    results=[]
    # A partial refresh is explicitly FAIL; it cannot become a runnable queue.
    write_json(directory/'QUEUE_RECEIPT.json',dict(status='FAIL',cpu_only=True,all_owned_closed=True,
               error='orchestration refresh not complete'))
    for job in jobs:
        result=refresh_driver(job['run'],mode)
        argv=job['argv']
        if '--driver-mode' in argv:
            argv[argv.index('--driver-mode')+1]=mode
        else:argv.extend(['--driver-mode',mode])
        results.append(dict(name=job['name'],prepared=job['run'],driver_refresh=result))
    write_json(directory/'QUEUE_PLAN.json',dict(jobs=[{k:v for k,v in job.items() if k!='args'} for job in jobs],
               cpu_only=True,prestaging_depth=1,driver_mode=mode))
    write_json(directory/'QUEUE_RECEIPT.json',dict(status='PASS',cpu_only=True,all_owned_closed=True,
               driver_mode=mode,results=results))
    return results


def rebind_prepared_queue(directory, jobs, control):
    from .formal_prestage import rebind_control, validate_prepared
    from .formal_runtime import paired_lineage
    directory = Path(directory)
    # Refuse an unrelated control before changing any preparation.
    for job in jobs:
        campaign,_ = validate_prepared(job['run'])
        candidate = dict(campaign, paired_control=str(control.resolve(strict=True)))
        paired_lineage(candidate, candidate['runtime_from'])
    stamp = str(time.time_ns())
    for name in ('QUEUE_PLAN.json','QUEUE_RECEIPT.json'):
        write_json(directory/(name+'.before_control_'+stamp),json.loads((directory/name).read_text()))
    write_json(directory/'QUEUE_RECEIPT.json',dict(status='FAIL',cpu_only=True,all_owned_closed=True,
               error='control refresh not complete'))
    results = []
    for job in jobs:
        refreshed = rebind_control(job['run'],control)
        job['argv'][job['argv'].index('--paired-control')+1] = str(control.resolve())
        results.append(dict(name=job['name'],prepared=job['run'],control_refresh=refreshed))
    write_json(directory/'QUEUE_PLAN.json',dict(jobs=[{k:v for k,v in job.items() if k!='args'} for job in jobs],
               cpu_only=True,prestaging_depth=1))
    write_json(directory/'QUEUE_RECEIPT.json',dict(status='PASS',cpu_only=True,all_owned_closed=True,results=results))
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--jobs',type=Path)
    source.add_argument('--prepared-queue',type=Path,help='Launch an already prepared CPU-only PASS queue without new preparation')
    source.add_argument('--continue-queue',type=Path,
                        help='Wait for natural queue closure, skip closed PASS jobs, retry only zero-acceptance failures')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--refresh-driver',choices=('group-commit',),
                        help='CPU refresh of a sealed, undispatched --prepared-queue')
    parser.add_argument('--rebind-control',type=Path,
                        help='CPU-only rebind of undispatched arms to a verified resumed control')
    parser.add_argument('--startup-retries',type=int,default=2,
                        help='Fresh sealed copies after closed zero-acceptance failures (0..3; default 2)')
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run',action='store_true')
    action.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if not 0 <= args.startup_retries <= 3:
        parser.error('--startup-retries must be between 0 and 3')
    if args.continue_queue and not args.run:
        parser.error('--continue-queue requires --run')
    if args.refresh_driver and (not args.prepared_queue or not args.prepare_only or args.output):
        parser.error('--refresh-driver requires --prepared-queue --prepare-only, without --output')
    if args.rebind_control and (not args.prepared_queue or not args.prepare_only or args.output or args.refresh_driver):
        parser.error('--rebind-control requires --prepared-queue --prepare-only without --output/--refresh-driver')
    if args.prepared_queue and not args.run and not (args.refresh_driver or args.rebind_control):
        parser.error('--prepared-queue requires --run or --prepare-only --refresh-driver')
    if args.continue_queue:
        from .queue_retry import continued_jobs
        jobs=continued_jobs(args.continue_queue)
    else:
        jobs=prepared_jobs(args.prepared_queue) if args.prepared_queue else jobs_from_file(args.jobs)
    if args.refresh_driver or args.rebind_control:
        try:
            results=(rebind_prepared_queue(args.prepared_queue,jobs,args.rebind_control) if args.rebind_control else
                     refresh_prepared_queue(args.prepared_queue,jobs,args.refresh_driver))
        except BaseException as exc:
            write_json(args.prepared_queue/'QUEUE_RECEIPT.json',dict(status='FAIL',cpu_only=True,
                       all_owned_closed=True,error=repr(exc)))
            print(json.dumps(dict(status='FAIL',error=repr(exc),output=str(args.prepared_queue))))
            return 2
        print(json.dumps(dict(status='PASS',cpu_only=True,results=results,output=str(args.prepared_queue))))
        return 0
    directory=args.output or launch.FORMAL_ROOT/('formal_queue_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    if directory.resolve().parent!=launch.FORMAL_ROOT:
        parser.error('queue output must be a direct child of formal root')
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'QUEUE_PLAN.json',dict(jobs=[{k:v for k,v in j.items() if k!='args'} for j in jobs],
        cpu_only=args.prepare_only,prestaging_depth=1,startup_retries=args.startup_retries,
        continuation_source=str(args.continue_queue) if args.continue_queue else None))
    runner=Runner(directory)
    previous={sig:signal.signal(sig,lambda *unused:setattr(runner,'stopped',True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    report=dict(status='FAIL',cpu_only=args.prepare_only)
    try:
        report.update(results=(run_prepared_jobs(jobs,runner,args.startup_retries) if args.prepared_queue or args.continue_queue else
                               run_jobs(jobs,runner,args.prepare_only,args.startup_retries)),status='PASS')
    except BaseException as exc:
        report['error']=repr(exc)
    finally:
        report['all_owned_closed']=runner.close()
        if not report['all_owned_closed']:
            report['status']='FAIL'
        for sig,handler in previous.items():
            signal.signal(sig,handler)
        write_json(directory/'QUEUE_RECEIPT.json',report)
    print(json.dumps(dict(report,output=str(directory))))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
