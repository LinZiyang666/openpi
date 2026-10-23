"""Stage model-contiguous queues; only explicit coordinator commands control them."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

from .formal_process import alive,write_json
from .formal_prestage import validate_prepared
from .immutable_release import sha


def hold_next(lane,model):
    from .cache_lane import lane_processes
    lane=Path(lane).resolve(strict=True)
    state=json.loads((lane/'LANE_STATE.json').read_text())
    job=state['jobs'][model]
    if job['status']!='PREPARED':pass
    run=Path(job['run']).resolve(strict=True)
    with __import__('exp.r17_integration.ops.formal_prestage',fromlist=['preparation_lock']).preparation_lock(run):
        campaign,_=validate_prepared(run)
        record=dict(schema='r17.dispatch_hold.v1',lane=str(lane),model=model,run=str(run),
            prepared_sha256=sha(run/'PREPARED.json'),active_parent=lane_processes(lane),
            reason='coordinator model-contiguous ordering; no live child is signalled')
        write_json(run/'DISPATCH_HOLD.json',record)
    return record


def cancel_waiter(lane):
    """Coordinator explicitly retires only a pure waiting parent, never a fleet."""
    from .cache_lane import lane_processes
    lane=Path(lane).resolve(strict=True)
    state=json.loads((lane/'LANE_STATE.json').read_text())
    if state['status']!='WAITING_PREDECESSOR':pass
    owned=lane/'QUEUE_OWNED.json'
    if owned.exists() and any(alive(e) for e in json.loads(owned.read_text())):
        raise ValueError('waiter has live owned children; refuse signals')
    entries=lane_processes(lane)
    if len(entries)!=1:pass
    entry=entries[0]
    if Path('/proc/'+str(entry['pid'])).stat().st_uid!=os.getuid() or not alive(entry):
        pass
    os.kill(entry['pid'],signal.SIGTERM)
    record=dict(cpu_only=True,signal='SIGTERM',parent=entry,no_child_signals=True)
    write_json(lane/'COORDINATOR_WAIT_CANCEL.json',record)
    return record


def stage(directory,fast,cosmos500,cosmos_sizes,predecessor):
    """Only write queue configurations; no deployment, launch or arming."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    plans=[]
    for label,runs in [('fast_sizes',[fast]),('cosmos_all',[cosmos500,cosmos_sizes])]:
        jobs=[]
        for run in runs:
            run=Path(run).resolve(strict=True);c,_=validate_prepared(run)
            expected='fastwam' if label=='fast_sizes' else 'cosmos'
            if c['model']!=expected or c.get('execution_worker_nodes')!={'timan107':45,'timan1':40}:
                raise ValueError('wrong model or external85 fleet')
            jobs.append(dict(name=run.name,run=str(run),seconds=c['seconds'],
                prepared_sha256=sha(run/'PREPARED.json'),model=c['model']))
        q=directory/label;q.mkdir()
        plan=dict(schema='r17.cache_model_order.v1',status='STAGED',cpu_only=True,
            jobs=jobs,after_run=str(Path(predecessor).resolve()) if label=='fast_sizes' else None,
            after_queue=str(directory/'fast_sizes') if label=='cosmos_all' else None,
            launch_authorized=False,one_model_only=True)
        write_json(q/'MODEL_QUEUE_PLAN.json',plan);plans.append(plan)
    write_json(directory/'ORDER.json',dict(status='STAGED',cpu_only=True,armed=False,
        order=['Fast-WAM500 continuation','Fast-WAM50..400','Cosmos500','Cosmos50..400'],plans=plans,
        resident_note='Each five-size campaign hot-switches all VIEW/YAMLs on one resident fleet. Existing500 launcher closes its own residents; no unsafe adoption of live residents.'))
    return directory


def wait_source(path,queue=False,interval=15):
    from .formal_finalize import wait_seal,closed_proof
    path=Path(path)
    name='MODEL_QUEUE_RECEIPT.json' if queue else 'RECEIPT.json'
    while True:
        if (path/name).exists():
            receipt=json.loads((path/name).read_text())
            if (not queue and receipt.get('status')=='PENDING_EVIDENCE' and
                    receipt.get('gpu_released') is True and
                    receipt.get('driver_completion',{}).get('verified_completion') is True and
                    (path/'GPU_RELEASED.json').exists()):
                # Ordering only: no pending result becomes a control/resume source.
                closed_proof(receipt)
                return
            if receipt.get('status')=='PASS' and receipt.get('all_owned_closed') is True:
                if not queue:wait_seal(path)
                return
            if receipt.get('status') in ('FAIL','PARTIAL'):
                pass
        time.sleep(interval)


def run(directory):
    import fcntl
    from .formal_queue import Runner
    from .queue_retry import run_job
    from .cache_lane import resume_job
    from .formal_finalize import wait_seal
    directory=Path(directory).resolve(strict=True)
    plan=json.loads((directory/'MODEL_QUEUE_PLAN.json').read_text())
    with (directory/'.model_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .formal_process import ticks
        write_json(directory/'MODEL_QUEUE_OWNER.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),directory=str(directory)))
        if plan.get('after_run'):wait_source(plan['after_run'])
        if plan.get('after_queue'):wait_source(plan['after_queue'],queue=True)
        runner=Runner(directory);results=[];status='FAIL'
        try:
            for job in plan['jobs']:
                c,_=validate_prepared(job['run'])
                held=Path(job['run'])/'DISPATCH_HOLD.json'
                if held.exists():
                    record=json.loads(held.read_text())
                    if (not plan.get('after_queue') or record['prepared_sha256']!=job['prepared_sha256'] or
                            any(alive(e) for e in record['active_parent'])):
                        pass
                    held.rename(held.with_name('DISPATCH_HOLD_RELEASED.json'))
                current=dict(job,args=argparse.Namespace(seconds=job['seconds']))
                def resume(j):return resume_job(dict(j,model=c['model']+'_libero'))
                results.append(run_job(current,runner,limit=2,resume_handler=resume))
            for result in results:wait_seal(result['run'])
            status='PASS'
        finally:
            closed=runner.close()
            write_json(directory/'MODEL_QUEUE_RECEIPT.json',dict(status=status,
                all_owned_closed=closed,results=results,cpu_only=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='mode',required=True)
    hold=s.add_parser('hold-next');hold.add_argument('--lane',type=Path,required=True);hold.add_argument('--model',default='cosmos_libero')
    cancel=s.add_parser('cancel-waiter');cancel.add_argument('--lane',type=Path,required=True)
    go=s.add_parser('run');go.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='hold-next':print(json.dumps(hold_next(a.lane,a.model)))
    elif a.mode=='cancel-waiter':print(json.dumps(cancel_waiter(a.lane)))
    else:run(a.directory)


if __name__=='__main__':main()
