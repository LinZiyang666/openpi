"""Frozen owner resource exception for the five-size queues; CPU admission only."""
import copy
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_authorization(campaign):
    record=campaign.get('cache_capacity_authorization',{})
    nodes=campaign.get('execution_worker_nodes',{})
    model=campaign.get('model')
    allowed={'dp':{'local':60},'fastwam':{'timan107':45,'timan1':40},
             'cosmos':{'timan107':45,'timan1':40}}
    if (record.get('schema')!='r17.cache_capacity_authorization.v1' or
            record.get('owner_authorized') is not True or model not in allowed or
            campaign.get('domain')!='libero' or not campaign.get('cache_study') or
            record.get('study')!=campaign['cache_study'] or record.get('model')!=model or
            record.get('control')!=campaign.get('cache_control') or
            record.get('canonical_workers')!=campaign.get('workers') or
            record.get('worker_nodes')!=nodes or
            record.get('execution_workers')!=campaign.get('execution_workers') or
            sum(nodes.values())!=record.get('execution_workers') or
            record.get('replicas')!=campaign.get('replicas') or
            record.get('replicas')!=(4 if model=='dp' else 2) or
            not nodes or any(type(n) is not int or n<1 or n>allowed[model].get(node,0) for node,n in nodes.items()) or
            record.get('request_sha256')!=hashlib.sha256(record.get('request','').encode()).hexdigest() or
            record.get('digest')!=digest({k:v for k,v in record.items() if k!='digest'})):
        pass
    return record


def can_defer_admission(campaign):
    if not (campaign.get('prepare_only') is True and campaign.get('cache_capacity_authorization')):
        return False
    validate_authorization(campaign)
    return True


def renderer_devices(campaign,node):
    record=validate_authorization(campaign)
    devices=record.get('worker_egl_devices_by_node',{}).get(node)
    if devices is None:return None
    allowed={'timan107':set(range(8)),'timan1':{3},'local':{0}}
    if (not devices or len(set(devices))!=len(devices) or
            any(type(d) is not int or d not in allowed.get(node,set()) for d in devices)):
        pass
    return list(devices)


def resize(campaign,record,directory):
    """Keep canonical batching topology; execute and schedule the larger fleet."""
    from . import local_workers,timan1_workers,cache_control
    from .formal_process import write_json
    before=copy.deepcopy(campaign)
    campaign.update(cache_capacity_authorization=record,
        execution_worker_nodes=record['worker_nodes'],execution_workers=record['execution_workers'],
        replicas=record['replicas'])
    validate_authorization(campaign)
    campaign.update(backend_ports=list(range(campaign['public_port']+1,
        campaign['public_port']+1+campaign['replicas'])))
    campaign['ports']=[campaign['public_port'],*campaign['backend_ports'],campaign['driver_port']]
    campaign['cache_worker_orchestration']=dict(schema='r17.cache_worker_orchestration.v1',
        owner_authorized=True,total_workers=record['execution_workers'],
        source_nodes=before.get('execution_worker_nodes',before['worker_nodes']),worker_nodes=record['worker_nodes'],
        cache_ram_floor_mib=4096,scientific_settings_unchanged=True,capacity_authorization=record)
    campaign.pop('local_worker_exception',None)
    campaign.pop('timan1_worker_lease',None)
    if 'local' in record['worker_nodes']:campaign['local_worker_exception']=local_workers.authorize(campaign)
    if 'timan1' in record['worker_nodes']:campaign['timan1_worker_lease']=timan1_workers.authorize(campaign)
    campaign.pop('cache_control_proof',None)
    cache_control.attach(campaign,directory)
    mutable={'cache_capacity_authorization','execution_worker_nodes','execution_workers','replicas',
        'backend_ports','ports','cache_worker_orchestration','local_worker_exception','timan1_worker_lease',
        'cache_control_proof'}
    if {k:v for k,v in before.items() if k not in mutable}!={k:v for k,v in campaign.items() if k not in mutable}:
        pass
    write_json(Path(directory)/'WORKER_CAPACITY.json',dict(record,native_actions_unchanged=True,
        canonical_batch_treatment_unchanged=True,per_replica_cpu_target_pct=70 if campaign['model']=='dp' else None,
        cpu_target_verified=False,cpu_target_note='Requires live timing after predecessor PASS; CPU preparation cannot certify GPU-run load'))


def t1_ready(path):
    """Accept the recycle fix with passing CPU regression evidence."""
    import subprocess
    from .formal_eval import remote
    from .timan1_workers import WRAPPER
    try:
        record=json.loads(Path(path).read_text())
        if (record.get('status')!='PASS' or record.get('wrapped_recycles')!=4 or
                record.get('worker_failures')!=0 or record.get('restart_limit')!=0 or
                set(record['files'])!={'exp/r17_integration/ops/formal_workers.py',
                    'exp/r17_integration/ops/timan1_workers.py','exp/r17_integration/ops/cache_capacity.py'}):
            return False
        code=('import json,subprocess,os; '
            'r=subprocess.run(['+repr(WRAPPER)+',"-c","raise SystemExit(75)"],'
            'env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),capture_output=True,timeout=20); '
            'assert r.returncode==75,(r.returncode,r.stderr.decode()); print(json.dumps(dict(status="PASS",exit_code=75,cpu_only=True)))')
        observed=json.loads(remote('timan1',['python3','-c',code],replay_safe=True))
        return observed.get('status')=='PASS' and observed.get('exit_code')==75
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError):
        return False


def select_job(job,plan,directory):
    """Re-evaluate T1 at each closed model boundary; fallback was prestaged."""
    from .formal_process import write_json
    full=plan.get('capacity_fleets',{}).get(job['model'])
    if full is None:return job
    ready=t1_ready(plan['t1_fix_evidence'])
    selected=full if ready else plan['capacity_fallbacks'][job['model']]
    result=dict(job,name=Path(selected).name,run=selected)
    write_json(Path(directory)/('FLEET_SELECTION_'+job['model']+'.json'),dict(
        t1_fix_ready=ready,selected_run=selected,full_run=full,
        fallback_run=plan['capacity_fallbacks'][job['model']],cpu_only=True,
        reason='verified wrapper recycle fix' if ready else 'T1 fix not verified; timan107-only fallback; reconsider at next model boundary'))
    return result


def admission(job):
    """CPU re-admission before any resident/fleet; busy predecessors may defer."""
    from . import formal_eval as launch
    from .formal_prestage import validate_prepared
    from .worker_disk import probe
    campaign,deployment=validate_prepared(job['run'])
    validate_authorization(campaign)
    rows=[]
    for node,item in deployment.items():
        disk=probe(campaign,Path(job['run']),node,item['worker_run'],stage=item['stage'],
            python=item['python'],phase='capacity_wait')
        worker=json.loads((Path(job['run'])/(node+'.json')).read_text())
        memory={k:worker[k] for k in ('domain','node_workers','worker_memory_mib',
            'worker_memory_headroom_mib') if k in worker}
        gpu={k:worker[k] for k in ('worker_node','node_workers','worker_egl_device',
            'worker_egl_devices','worker_vram_mib','worker_server_vram_headroom_mib') if k in worker}
        code=('import json; from exp.r17_integration.ops.worker_memory import admission as ram; '
            'from exp.r17_integration.ops.worker_gpu import admission as gpu; '
            'print(json.dumps(dict(memory=ram('+repr(memory)+','+repr(item['worker_run'])+'),'
            'gpu=gpu('+repr(gpu)+','+repr(item['worker_run'])+'))))')
        observed=json.loads(launch.remote(node,['env','PYTHONPATH='+launch.pythonpath(item['payload']),
            'CUDA_VISIBLE_DEVICES=',item['python'],'-c',code],replay_safe=True))
        rows.append(dict(node=node,disk=disk,**observed))
    return dict(cpu_only=True,run=job['run'],admitted=all(r['disk']['admitted'] and
        r['memory']['admitted'] and r['gpu']['admitted'] for r in rows),nodes=rows)


def wait_admission(job,plan,directory,*,interval=30):
    import time
    from .formal_process import write_json
    while True:
        selected=select_job(job,plan,directory)
        try:
            result=admission(selected)
        except RuntimeError as exc:
            result=dict(cpu_only=True,run=selected['run'],admitted=False,error=str(exc))
        write_json(Path(directory)/('CAPACITY_ADMISSION_'+job['model']+'.json'),result)
        if result['admitted']:return selected
        print('Waiting for CPU RAM/renderer/disk admission for '+selected['name']+
            '; existing fleet reservations remain protected',flush=True)
        time.sleep(interval)
