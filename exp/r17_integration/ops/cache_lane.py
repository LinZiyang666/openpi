"""CPU prestage and resume-safe resident cache-study queues, one GPU lane each.

--prepare-only never loads CUDA or dispatches workers. --run is a separate
explicit coordinator action. Each model is one campaign/fleet across both
suites and all physical groups, not one server launch per YAML.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import shlex
import time

from . import cache_study as study
from .formal_process import write_json, alive, ticks
from .immutable_release import sha

ROOT=study.ROOT
RUN_ROOT=Path('/data/mw_finetune/eval')
RUNTIME_SOURCES={'dp_libero':'hs5dp_ctl_142812','fastwam_libero':'hs6fw_ctl_143332',
                 'cosmos_libero':study.CONTROLS['cosmos_libero']}
PROFILES={
    'dp_libero':dict(node='local',workers=32,ram=4096,vram=576,batch=8,budget=40,startup=1,port=23400,driver=23419,mps=True,mode='group-commit'),
    'fastwam_libero':dict(node='h100',workers=48,ram=4096,vram=768,batch=16,budget=70,startup=2,port=23420,driver=23439,mps=True,mode='group-commit'),
    'cosmos_libero':dict(node='h100',workers=8,ram=4096,vram=768,batch=1,budget=20,startup=1,port=23440,driver=23459,mps=False,mode='legacy')}


@contextmanager
def gpu_launch_lock(node, phase, path='/tmp/r17_gpu.lock'):
    """Lock only the local execution lane, until owned residents close/handoff."""
    if node != 'local' or phase != 'run':
        yield
        return
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            pass
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def launch_args(model,manifest,name,*,smoke=False,size_count=1,worker_counts=None):
    p=PROFILES[model]
    argv=['--model',model,'--cache-study',str(Path(manifest).resolve()),'--official-50',
          '--suites',*study.SUITES,'--server-node',p['node'],'--gpu','cuda:0',
          '--replicas','2','--replica-startup-width',str(p['startup']),
          '--max-batch',str(p['batch']),'--batch-wait-ms','5','--memory-budget-gib',str(p['budget']),
          '--driver-mode',p['mode'],'--worker-node','timan107','--extra-workers','0',
          '--workers',str(p['workers']),'--worker-memory-mib',str(p['ram']),
          '--worker-vram-mib',str(p['vram']),'--worker-max-episodes','4','--worker-restart-limit','2',
          '--retries','2','--seconds',str(7200 if smoke else 86400*size_count),
          '--episode-seconds',str(3600 if model=='fastwam_libero' else 1800),
          '--port',str(p['port']),'--driver-port',str(p['driver']),'--run-name',name,
          '--cache-control',str(RUN_ROOT/study.CONTROLS[model]),
          '--cache-control-runtime',str(RUN_ROOT/RUNTIME_SOURCES[model])]
    if p['mps']:argv.append('--mps')
    if model=='fastwam_libero':argv.extend(['--worker-egl-devices','0,1,2,3,4,5,6,7'])
    if smoke:argv.extend(['--smoke','--smoke-episodes','2'])
    if worker_counts:
        from .local_workers import counts
        counts(worker_counts)
        argv.extend(['--worker-counts',worker_counts])
    return argv


def select_horizons(path,output,long_dp,long_fast):
    """Reuse sealed C-specific artifacts; fit only newly selected B-only heads."""
    import yaml
    original=study.validate_manifest(path)
    wanted={'dp_libero':long_dp,'fastwam_libero':long_fast,'cosmos_libero':16}
    parents={}
    for model,c in wanted.items():
        if type(c)!=int or c<(16 if model=='cosmos_libero' else 5):raise ValueError('L must be at least the native cache commitment')
        for row in original['models'][model]:
            if row['library'] not in parents:parents[row['library']]=study.read(Path(row['library'])/'READY.json')
            h=parents[row['library']]['binding']['model']['layout']['horizon']
            if c>h:raise ValueError('L exceeds native chunk for '+model)
    if all(all(r['settings']['commit']==study.group_settings(m,r['group'],wanted[m])['commit'] for r in rows)
           for m,rows in original['models'].items()):return Path(path).resolve()
    output=Path(output);output.mkdir(exist_ok=False)
    result=copy.deepcopy(original);result['artifacts_pending']=[]
    result['horizon_selection'].update(dp=long_dp,fastwam=long_fast,
        status='coordinator_selected_L; provisional_until_horizon_RESULTS_freeze')
    for model,rows in result['models'].items():
        for row in rows:
            c=study.group_settings(model,row['group'],wanted[model])['commit']
            doc=yaml.safe_load((Path(path).parent/row['file']).read_text())
            if c!=row['settings']['commit']:
                row['settings']['commit']=c
                artifact=output/'artifacts'/study.size_label(row)/(model+'_'+row['suite'])/('C'+str(c))
                row.update(fit=str(artifact/'fit'),stall=str(artifact/'stall'),
                           fit_digest='PENDING_C_SPECIFIC_FIT',stall_digest='PENDING_B_Q3_CALIBRATION')
                doc['adapter_binding']['commit_horizon']=c
                doc['recipe'].update(fitted_path=row['fit'],fitted_digest=row['fit_digest'])
                if row['settings']['gated']:
                    doc['recipe'].update(stall_path=row['stall'],stall_digest=row['stall_digest'])
                result['artifacts_pending'].append(model+'/'+row['suite']+'/'+row['group'])
            target=output/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(yaml.safe_dump(doc,sort_keys=False));row['sha256']=sha(target)
    result.pop('digest');result['digest']=study.digest(result)
    write_json(output/'STUDY.json',result)
    # fit_inputs refreshes all config bindings/digests. Unchanged original
    # fitted heads and stall artifacts remain read-only absolute references.
    study.fit_inputs(output/'STUDY.json')
    study.validate_manifest(output/'STUDY.json')
    return output/'STUDY.json'


def jobs(lane,manifest,prefix,*,size_count=1,worker_counts=None):
    models=['dp_libero'] if lane=='4090' else ['fastwam_libero','cosmos_libero']
    return [dict(name=prefix+{'dp_libero':'d','fastwam_libero':'f','cosmos_libero':'c'}[m],model=m,
                 run=str(RUN_ROOT/(prefix+{'dp_libero':'d','fastwam_libero':'f','cosmos_libero':'c'}[m])),
                 argv=launch_args(m,manifest,prefix+{'dp_libero':'d','fastwam_libero':'f','cosmos_libero':'c'}[m],size_count=size_count,worker_counts=worker_counts)) for m in models]


def resume_job(job,worker_counts=None,replicas=None):
    from .cache_resume import prepare
    source=Path(job['run'])
    capacity=(dict(replicas=replicas,local_vram_mib=576,local_server_growth_mib=8192) if replicas is not None else {})
    return prepare(source,job['model'].split('_')[0]+'c'+str(time.time_ns())[-10:],refresh_workers=True,worker_counts=worker_counts,**capacity)


def scoped_overrides(values,models,*,fleet=False):
    """Explicit model:map/path overrides; no ambiguity across unequal fleets."""
    from .local_workers import counts
    result={}
    for value in values:
        model,setting=value.split(':',1)
        if model not in models or model in result:
            raise ValueError('one override per model in the selected lane required')
        result[model]=counts(setting) if fleet else str(Path(setting).resolve(strict=True))
    return result


def reprepare_fleet(source,model,counts,directory,replicas=None):
    """Clone an undispatched campaign; preserve its old preparation read-only."""
    from .queue_retry import capture,clone
    from .formal_prestage import validate_prepared
    validate_prepared(source)
    name=model.split('_')[0]+'p'+str(time.time_ns())[-10:]
    snapshot=Path(directory)/('topology_inputs_'+name)
    capture(source,snapshot)
    return clone(snapshot,name,dict(owner_authorized_redistribution=True,cpu_only=True),
                 refresh_cache_workers=True,worker_counts=counts,
                 **(dict(replicas=replicas,local_vram_mib=576,local_server_growth_mib=8192) if replicas is not None else {}))


def lane_processes(directory):
    """Exact argv/path match, with PID birth ticks; never signal predecessors."""
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if 'exp.r17_integration.ops.cache_lane' not in argv or '--directory' not in argv:continue
            if Path(argv[argv.index('--directory')+1]).resolve()!=Path(directory).resolve():continue
            entry=dict(pid=int(proc.name),start_ticks=ticks(int(proc.name)))
            if alive(entry):result.append(entry)
        except (OSError,ValueError,IndexError):continue
    return result


def predecessor(path,lane,directory):
    path=Path(path).resolve(strict=True)
    if path==directory:raise ValueError('lane cannot wait on itself')
    plan=study.read(path/'LANE_PLAN.json')
    if plan['lane']!=lane:pass
    return dict(directory=str(path),plan_sha256=sha(path/'LANE_PLAN.json'),
                processes=lane_processes(path))


def wait_predecessor(record, *, interval=30, wait_for_pass=False):
    """Wait for process exit, then hold its lane lock through our GPU handoff.

    Failure or unsealed ownership refuses launch. A coordinator may resume the
    predecessor and repeat this command; new PID generations are also watched.
    """
    directory=Path(record['directory'])
    handle=(directory/'.lane.lock').open('r')
    try:
        while True:
            if sha(directory/'LANE_PLAN.json')!=record['plan_sha256']:
                pass
            active=any(alive(e) for e in record['processes']) or bool(lane_processes(directory))
            acquired=False
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
            except BlockingIOError:pass
            if acquired and not active:
                if wait_for_pass:
                    state=study.read(directory/'LANE_STATE.json')
                    receipt=study.read(directory/'LANE_RECEIPT.json')
                    if (state.get('status')!='PASS' or receipt.get('status')!='PASS' or
                            receipt.get('all_owned_closed') is not True):
                        fcntl.flock(handle,fcntl.LOCK_UN)
                        print('Waiting for predecessor '+str(directory)+' to finish closed PASS; current status '+
                              str(state.get('status'))+'; coordinator may resume it',flush=True)
                        time.sleep(interval)
                        continue
                break
            if acquired:fcntl.flock(handle,fcntl.LOCK_UN)
            print('Waiting for predecessor lane '+str(directory)+' to exit and release ownership',flush=True)
            time.sleep(interval)
        state=study.read(directory/'LANE_STATE.json')
        receipt=study.read(directory/'LANE_RECEIPT.json')
        if state.get('status')!='PASS' or receipt.get('status')!='PASS' or receipt.get('all_owned_closed') is not True:
            pass
        from .formal_finalize import wait_seal
        for result in state['results']:
            wait_seal(result['run'])
            proof=study.read(Path(result['run'])/'RECEIPT.json')
            if proof.get('status')!='PASS' or proof.get('all_owned_closed') is not True:
                pass
        return handle
    except BaseException:
        handle.close();raise


def main(argv=None):
    from . import formal_eval as launch
    from .formal_queue import Runner
    from .formal_prestage import validate_prepared
    from .formal_finalize import wait_seal
    from .queue_retry import run_job,zero_acceptance
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=('4090','h100'),required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--prefix',default='cs1010')
    p.add_argument('--after-lane',type=Path,help='wait for this predecessor lane and require its complete closed PASS')
    p.add_argument('--wait-for-pass',action='store_true',help='keep an armed --run waiting while a failed predecessor is repaired')
    p.add_argument('--study',type=Path,default=study.HERE/'yaml/STUDY.json')
    p.add_argument('--long-dp',type=int,choices=(10,15,16),default=10)
    p.add_argument('--long-fast',type=int,choices=(10,16,20,32),default=10)
    p.add_argument('--worker-counts',help='per-node fleet, e.g. timan107=8,local=16; frozen at prepare')
    p.add_argument('--redistribute-workers',action='append',default=[],metavar='MODEL:NODE=N,...',
                   help='--run only: cache continuation execution fleet; canonical controls remain pinned')
    p.add_argument('--dp-replicas',type=int,choices=(3,4),help='owner cache-only DP capacity exception; requires explicit local fleet')
    p.add_argument('--resume-source',action='append',default=[],metavar='MODEL:RUN',
                   help='--run only: explicitly select a closed failed segment instead of an unlaunched preparation')
    phase=p.add_mutually_exclusive_group(required=True)
    phase.add_argument('--print',action='store_true');phase.add_argument('--prepare-only',action='store_true');phase.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    if args.wait_for_pass and not args.run:p.error('--wait-for-pass requires --run')
    if (args.redistribute_workers or args.resume_source) and not args.run:
        p.error('redistribution/source selection requires --run')
    models=['dp_libero'] if args.lane=='4090' else ['fastwam_libero','cosmos_libero']
    fleets=scoped_overrides(args.redistribute_workers,models,fleet=True)
    if args.dp_replicas and (not args.run or args.lane!='4090' or set(fleets.get('dp_libero',{}))!={'local'}):
        p.error('--dp-replicas requires --run --lane4090 and explicit DP local workers')
    sources=scoped_overrides(args.resume_source,models)
    if len(args.prefix)>24 or not args.prefix.isalnum():p.error('plain alphanumeric prefix of at most24 characters required')
    directory=args.directory.resolve()
    if args.print:
        count=len(study.validate_manifest(args.study).get('library_sizes',[500]))
        for job in jobs(args.lane,args.study,args.prefix,size_count=count,worker_counts=args.worker_counts):
            print(shlex.join([launch.CONTROL_PYTHON,'-u','-m','exp.r17_integration.ops.formal_eval',*job['argv'],'--prepare-only']))
        print('L: DP='+str(args.long_dp)+' Fast-WAM='+str(args.long_fast)+' Cosmos=16; --prepare-only freezes the selection; --run reuses it')
        return 0
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lane.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=directory/'LANE_PLAN.json'
        if args.prepare_only and not plan_path.exists():
            manifest=select_horizons(args.study,directory/'yaml',args.long_dp,args.long_fast)
            plan=dict(schema='r17.cache_lane.v1',lane=args.lane,long_dp=args.long_dp,long_fast=args.long_fast,
                      study=str(manifest),study_sha256=sha(manifest),worker_counts=args.worker_counts,jobs=jobs(args.lane,manifest,args.prefix,size_count=len(study.validate_manifest(manifest).get('library_sizes',[500])),worker_counts=args.worker_counts),
                      expected_episodes=(8000 if args.lane=='4090' else 12000)*len(study.validate_manifest(manifest).get('library_sizes',[500])),
                      pending_horizon_freeze=True,one_resident_fleet_per_model=True,cpu_only=True)
            if args.after_lane:plan['after_lane']=predecessor(args.after_lane,args.lane,directory)
            write_json(plan_path,plan)
        else:
            plan=study.read(plan_path)
            if (plan['lane']!=args.lane or plan['long_dp']!=args.long_dp or plan['long_fast']!=args.long_fast or
                    sha(plan['study'])!=plan['study_sha256']):pass
            if args.prepare_only:
                if [j['name'] for j in plan['jobs']]!=[j['name'] for j in jobs(args.lane,args.study,args.prefix)]:
                    pass
            if args.worker_counts and args.worker_counts!=plan.get('worker_counts'):
                pass
        if not args.prepare_only and args.after_lane and str(args.after_lane.resolve())!=plan.get('after_lane',{}).get('directory'):
            pass
        state_path=directory/'LANE_STATE.json'
        state=study.read(state_path) if state_path.exists() else {'jobs':{},'status':'PREPARING'}
        resource_choices=state.setdefault('continuation_resources',{})
        for model in models:
            choice=dict(resource_choices.get(model,{}))
            if model in fleets:choice['worker_counts']=fleets[model]
            if model=='dp_libero' and args.dp_replicas:choice['replicas']=args.dp_replicas
            if choice:resource_choices[model]=choice
            if model not in fleets and choice.get('worker_counts'):fleets[model]=choice['worker_counts']
        class CacheRunner(Runner):
            def invoke(self,job,phase,argv):
                if phase=='run':
                    state.update(status='RUNNING',cpu_only=False)
                    state.pop('error',None)
                    state['jobs'][job['model']]=dict(run=job['run'],status='RUNNING')
                    write_json(state_path,state)
                with gpu_launch_lock(PROFILES[job['model']]['node'],phase):
                    return super().invoke(job,phase,argv)
        runner=CacheRunner(directory);results=[]
        def continue_job(job):
            replica_choice=resource_choices.get(job['model'],{}).get('replicas')
            return (resume_job(job,fleets.get(job['model']),replica_choice) if replica_choice is not None
                    else resume_job(job,fleets.get(job['model'])))
        gate=None
        recovered=[]
        try:
            if args.prepare_only and (directory/'QUEUE_OWNED.json').exists():
                recovered=study.read(directory/'QUEUE_OWNED.json')
                if any(r.get('phase')!='prepare' for r in recovered):
                    raise ValueError('cannot prepare over a lane with recorded GPU dispatch')
                while any(alive(r) for r in recovered):
                    print('Waiting for recorded CPU preparation children to finish',flush=True)
                    time.sleep(30)
            if not args.prepare_only and plan.get('after_lane'):
                state.update(status='WAITING_PREDECESSOR');write_json(state_path,state)
                gate=wait_predecessor(plan['after_lane'],wait_for_pass=args.wait_for_pass)
            for job in plan['jobs']:
                if args.prepare_only:
                    destination=Path(job['run'])
                    if not destination.exists():runner.invoke(job,'prepare',[*job['argv'],'--prepare-only'])
                    elif not (destination/'PREPARED.json').exists():
                        from .cache_cpu_prepare import finish
                        finish(destination,plan['study'])
                    campaign,_=validate_prepared(destination)
                    if campaign.get('cache_study')!=plan['study']:pass
                    state['jobs'][job['model']]=dict(run=str(destination),status='PREPARED')
                    write_json(state_path,state);results.append(dict(model=job['model'],run=str(destination)));continue
                if plan.get('owner_capacity_request') and state['jobs'].get(job['model'],{}).get('status')=='PREPARED':
                    from .cache_capacity import wait_admission
                    job=wait_admission(job,plan,directory)
                    state['jobs'][job['model']]=dict(run=job['run'],status='PREPARED')
                    write_json(state_path,state)
                current=state['jobs'].get(job['model'],{'run':job['run']})['run'];destination=Path(current)
                if job['model'] in sources:
                    if (destination/'LAUNCH.json').exists() and not (destination/'RECEIPT.json').exists():
                        pass
                    destination=Path(sources[job['model']]);current=str(destination)
                    original=study.read(Path(job['run'])/'campaign.json')
                    replacement=study.read(destination/'campaign.json')
                    for key in ('model','domain','checkpoint','workers','cache_study','cache_control','cache_control_runtime',
                                'runtime_identity','fastwam_batch_treatment','dp_batching'):
                        if original.get(key)!=replacement.get(key):
                            pass
                chosen=fleets.get(job['model'])
                replica_choice=resource_choices.get(job['model'],{}).get('replicas')
                capacity=(dict(replicas=replica_choice,local_vram_mib=576,local_server_growth_mib=8192)
                          if replica_choice is not None else {})
                if (destination/'RECEIPT.json').exists():
                    receipt=study.read(destination/'RECEIPT.json')
                    if receipt.get('status')=='PASS':
                        wait_seal(destination)
                        results.append(dict(name=job['name'],run=current,status=receipt['status']));continue
                    if receipt.get('status')=='PENDING_EVIDENCE':
                        from .formal_finalize import closed_proof
                        closed_proof(receipt)
                        if not receipt.get('gpu_released') or not (destination/'GPU_RELEASED.json').exists():
                            pass
                        results.append(dict(name=job['name'],run=current,status=receipt['status']));continue
                    if receipt.get('all_owned_closed') is not True:pass
                    try:zero_acceptance(destination);startup=True
                    except ValueError:
                        current=str(continue_job(dict(job,run=str(destination))));startup=False
                    if startup and chosen:
                        # A closed startup failure can also change only its fleet.
                        from .queue_retry import clone
                        current=str(clone(destination/'.startup_inputs',job['model'].split('_')[0]+'p'+str(time.time_ns())[-10:],
                            dict(owner_authorized_redistribution=True,all_owned_closed=True),refresh_cache_workers=True,worker_counts=chosen,**capacity))
                        startup=False
                else:
                    prepared,_=validate_prepared(destination);startup=False
                    if chosen and (prepared.get('execution_worker_nodes',prepared['worker_nodes'])!=chosen or
                            replica_choice is not None and prepared['replicas']!=replica_choice):
                        current=str(reprepare_fleet(destination,job['model'],chosen,directory,
                            **(dict(replicas=replica_choice) if replica_choice is not None else {})))
                selected=dict(job,name=Path(current).name,run=current,args=argparse.Namespace(seconds=int(job['argv'][job['argv'].index('--seconds')+1])))
                result=run_job(selected,runner,limit=2,initial_failure=startup,resume_handler=continue_job)
                results.append(result)
                state['jobs'][job['model']]=dict(run=result['run'],status=result['status']);write_json(state_path,state)
                # GPU-first handoff permits next model while CPU sealing runs.
                for previous in results:
                    value=study.read(Path(previous['run'])/'RECEIPT.json')
                    if value.get('status') not in ('PASS','PENDING_EVIDENCE'):pass
            if not args.prepare_only:
                for result in results:wait_seal(result['run'])
            state.update(status='PREPARED' if args.prepare_only else 'PASS',cpu_only=bool(args.prepare_only),results=results)
            write_json(state_path,state)
        except BaseException as exc:
            state.update(status='FAIL',error=repr(exc));write_json(state_path,state);raise
        finally:
            closed=runner.close()
            if recovered:
                proof=[dict(r,closure_confirmed=not alive(r)) for r in recovered]
                write_json(directory/'CPU_PREPARE_RECOVERY.json',dict(cpu_only=True,owned=proof,
                    artifacts_verified=state['status']=='PREPARED'))
                closed=closed and all(r['closure_confirmed'] for r in proof)
            write_json(directory/'LANE_RECEIPT.json',dict(status=state['status'],cpu_only=bool(args.prepare_only),all_owned_closed=closed,results=results))
            if gate is not None:gate.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
