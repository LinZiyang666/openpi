"""CPU-only larger size queues, with sealed T1-free fallbacks."""
import argparse
import copy
import hashlib
from pathlib import Path

from . import cache_study as study,cache_lane as lane,cache_capacity as capacity
from .formal_process import write_json
from .immutable_release import sha
from .formal_prestage import validate_prepared
from .queue_retry import capture,clone,validate_inputs

FLEETS={'dp_libero':{'local':60},'fastwam_libero':{'timan107':45,'timan1':40},
        'cosmos_libero':{'timan107':45,'timan1':40}}
DP_RESOURCES=dict(worker_ram_reservation_mib=3584,local_worker_memory_mib=3584,
    local_worker_vram_mib=717,worker_memory_headroom_mib=8192)
DP_RESOURCE_REQUEST=('Coordinator: CPU preparation only; DP local=60, replicas=4; '
    'renderer reservation 0.7 GiB (717 MiB), RAM reservation 3.5 GiB (3584 MiB), '
    'global RAM headroom 8 GiB; account running workers as max(reservation, measured RSS). '
    'No launches, arming, nohup, restart, stop, or campaign process control.')


def finish_copy(snapshot,destination,authorization):
    """Complete our undispatched resource copy, retaining its failed CPU evidence."""
    from . import formal_eval as launch,cache_control,driver_release
    from .formal_prestage import preparation_lock,seal_prepared
    snapshot,destination=Path(snapshot),Path(destination)
    source,_=validate_inputs(snapshot)
    with preparation_lock(destination):
        if any((destination/n).exists() for n in ('LAUNCH.json','RECEIPT.json','OWNED.json','PREPARED.json')):
            pass
        campaign=study.read(destination/'campaign.json')
        if campaign.get('prepare_only') is not True or campaign.get('cache_capacity_authorization')!=authorization:
            pass
        capacity.validate_authorization(campaign);cache_control.check(campaign)
        pins={n:sha(snapshot/n) for n in study.read(snapshot/'PREPARED.json')['files']
            if n.startswith('assets/') or n in ('EPISODE_PLAN.json','RUNTIME_RELEASE.json','FROZEN_RUNTIME.json')}
        pins['runtime.tar']=sha(snapshot/'runtime.tar')
        if any(sha(destination/n)!=pin for n,pin in pins.items()):
            pass
        driver_release.validate(campaign,destination)
        campaign['cache_worker_refresh']=dict(archive=str(snapshot/'workers.tar'),
            archive_sha256=sha(snapshot/'workers.tar'),files={f'exp/r17_integration/ops/{n}.py':sha(launch.ROOT/f'exp/r17_integration/ops/{n}.py')
                for n in ('formal_workers','worker_gpu','worker_disk')})
        write_json(destination/'campaign.json',campaign)
        with launch.retry_scope(destination):deployment=launch.prepare_workers(campaign,destination)
        if any(sha(destination/n)!=pin for n,pin in pins.items()):
            pass
        preflight=study.read(destination/'PREFLIGHT.json')
        preflight.update(worker_nodes=list(deployment),execution_worker_nodes=campaign['execution_worker_nodes'],
            execution_workers=campaign['execution_workers'],replicas=campaign['replicas'],ports=campaign['ports'],
            memory_admissible=all(study.read(destination/(node+'_MEMORY_ADMISSION.json'))['admitted'] for node in deployment),
            admission_deferred_until_predecessor_and_live_gates=True)
        write_json(destination/'PREFLIGHT.json',preflight)
        write_json(destination/'CAPACITY_CPU_RECOVERY.json',dict(status='PASS',cpu_only=True,
            original_failure_preserved=True,source=str(snapshot),native_pins=pins))
        seal_prepared(campaign,destination,deployment);validate_prepared(destination)


def prepare(which,directory,prefix):
    import fcntl
    directory=Path(directory).resolve();directory.mkdir(exist_ok=True)
    with (directory/'.lane.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=lane.RUN_ROOT/('cszedp' if which=='4090' else 'cszeh')
        plan=copy.deepcopy(study.read(source/'LANE_PLAN.json'))
        manifest=study.validate_manifest(plan['study'])
        assert manifest['library_sizes']==[50,100,200,300,400] and not manifest['artifacts_pending']
        request=(study.ROOT/'exp/r17_integration/REQUESTS/COORD_to_P6b.md').read_text().split('## 2026-10-10 21:1x',1)[1]
        if which=='4090':request += '\n'+DP_RESOURCE_REQUEST
        write_json(directory/'CAPACITY_REQUEST.json',dict(request=request,request_sha256=hashlib.sha256(request.encode()).hexdigest(),
            desired_dp_workers=60,prepared_dp_workers=60,live_admission_verified=False,
            dp_resource_reservations=DP_RESOURCES,
            dp_limit='717MiB renderer reservation; existing8192MiB DP growth reserve and10% GPU headroom remain; live admission deferred',
            local_reserved_for_dp=True,owner_authorized=True))
        state=dict(status='PREPARING',cpu_only=True,jobs={});results=[];full={};fallback={}
        write_json(directory/'LANE_STATE.json',state)
        for job in plan['jobs']:
            model=job['model'];original=Path(job['run']);validate_prepared(original)
            snapshot=directory/('inputs_'+model.split('_')[0])
            if not snapshot.exists():capture(original,snapshot)
            campaign,_=validate_inputs(snapshot)
            name=prefix+{'dp_libero':'d','fastwam_libero':'f','cosmos_libero':'c'}[model]
            variants=[(name,FLEETS[model])]
            if which=='h100':variants.append((name+'0',{'timan107':45}))
            for short,nodes in variants:
                record=dict(schema='r17.cache_capacity_authorization.v1',owner_authorized=True,
                    study=plan['study'],model=campaign['model'],control=campaign['cache_control'],
                    canonical_workers=campaign['workers'],execution_workers=sum(nodes.values()),worker_nodes=nodes,
                    replicas=4 if model=='dp_libero' else 2,request=request,
                    request_sha256=hashlib.sha256(request.encode()).hexdigest())
                record['digest']=capacity.digest(record)
                size_auth=dict(schema='r17.cache_size_lane_authorization.v1',owner_authorized=True,
                    study_date='2026-10-10',study=plan['study'],study_sha256=plan['study_sha256'],
                    study_digest=manifest['digest'],model=campaign['model'],workers=campaign['workers'],
                    worker_nodes=nodes,control=campaign['cache_control'],local_server_growth_mib=8192)
                if model=='dp_libero':size_auth['resource_reservations']=dict(DP_RESOURCES)
                print('CPU capacity prepare',short,nodes,'replicas',record['replicas'],flush=True)
                destination=lane.RUN_ROOT/short
                if not destination.exists():
                    destination=clone(snapshot,short,dict(cpu_only=True,owner_requested_capacity=True),
                        refresh_cache_workers=True,worker_counts=nodes,size_lane_authorization=size_auth,
                        capacity_authorization=record)
                elif not (destination/'PREPARED.json').exists():
                    finish_copy(snapshot,destination,record)
                prepared,deployment=validate_prepared(destination)
                capacity.validate_authorization(prepared)
                assert prepared['cache_capacity_authorization']==record
                assert set(deployment)==set(nodes) and sum(d['workers'] for d in deployment.values())==sum(nodes.values())
                assert sha(destination/'EPISODE_PLAN.json')==sha(snapshot/'EPISODE_PLAN.json')
                roster=[]
                for node,count in nodes.items():
                    roster.extend(dict(worker_id=f'{short}_{node}_{i}',node=node,index=i) for i in range(count))
                write_json(destination/'WORKER_ROSTER.json',dict(schema='r17.worker_roster.v1',workers=roster,
                    runtime_node_pid_birth_device_evidence='Each node formal_workers receipt.json records every generation'))
                if short==name:full[model]=str(destination)
                else:fallback[model]=str(destination)
            job.update(name=name,run=full[model],execution_worker_nodes=FLEETS[model])
            job['argv'][job['argv'].index('--run-name')+1]=name
            # Args document the operational schedule; dispatch always uses --prepared-run.
            job['argv'][job['argv'].index('--replicas')+1]='4' if model=='dp_libero' else '2'
            state['jobs'][model]=dict(run=job['run'],status='PREPARED')
            results.append(dict(model=model,run=job['run']));write_json(directory/'LANE_STATE.json',state)
        plan.update(worker_counts=None,source_prepared_lane=str(source),cpu_only=True,
            execution_worker_nodes_by_model={j['model']:FLEETS[j['model']] for j in plan['jobs']},
            owner_capacity_request=str(directory/'CAPACITY_REQUEST.json'),
            after_lane=lane.predecessor(plan['after_lane']['directory'],which,directory))
        if which=='h100':plan.update(capacity_fleets=full,capacity_fallbacks=fallback,
            t1_fix_evidence=str(study.HERE/'sizes/evidence/t1_recycle_fix.json'))
        write_json(directory/'LANE_PLAN.json',plan)
        state.update(status='PREPARED',results=results);write_json(directory/'LANE_STATE.json',state)
        write_json(directory/'LANE_RECEIPT.json',dict(status='PREPARED',cpu_only=True,all_owned_closed=True,results=results))
        print(which+' capacity size lane PREPARED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=('4090','h100'),required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--prefix',default='cszm')
    args=p.parse_args();prepare(args.lane,args.directory,args.prefix)


if __name__=='__main__':main()
