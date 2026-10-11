"""CPU-only split copies of the frozen five-size queues; originals stay sealed."""
import argparse
import copy
from pathlib import Path

from . import cache_study as study,cache_lane as lane
from .formal_process import write_json
from .immutable_release import sha
from .formal_prestage import validate_prepared
from .queue_retry import clone,validate_inputs

FLEETS={'dp_libero':{'timan107':16,'local':16},
        'fastwam_libero':{'timan107':16,'local':16,'timan1':16},
        'cosmos_libero':{'timan1':8}}


def prepare(which,directory,*,prefix='cszs',growth_mib=8192,names=None):
    import fcntl
    directory=Path(directory).resolve(strict=True)
    with (directory/'.lane.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=study.read(directory/'SOURCE_PREPARATION.json')
        original=Path(source['source'])
        if sha(original/'LANE_PLAN.json')!=source['source_lane_plan_sha256']:
            pass
        plan=copy.deepcopy(study.read(original/'LANE_PLAN.json'))
        if plan['lane']!=which or sha(plan['study'])!=plan['study_sha256']:
            pass
        manifest=study.validate_manifest(plan['study'])
        if manifest.get('library_sizes')!=[50,100,200,300,400] or manifest['artifacts_pending']:
            pass
        state=dict(status='PREPARING',cpu_only=True,jobs={})
        write_json(directory/'LANE_STATE.json',state)
        results=[]
        for job in plan['jobs']:
            model=job['model'];snapshot=Path(source['snapshots'][model])
            campaign,_=validate_inputs(snapshot)
            counts=FLEETS[model]
            if sum(counts.values())!=campaign['workers']:
                raise ValueError('split preparation changes model worker count')
            authorization=dict(schema='r17.cache_size_lane_authorization.v1',owner_authorized=True,
                study_date='2026-10-10',study=plan['study'],study_sha256=plan['study_sha256'],
                study_digest=manifest['digest'],model=campaign['model'],workers=campaign['workers'],
                worker_nodes=counts,control=campaign['cache_control'],local_server_growth_mib=growth_mib)
            name=(names or {}).get(model,prefix+{'dp_libero':'d','fastwam_libero':'f','cosmos_libero':'c'}[model])
            destination=lane.RUN_ROOT/name
            if not destination.exists():
                print('CPU split preparation '+model+' '+str(counts),flush=True)
                destination=clone(snapshot,name,dict(cpu_only=True,owner_authorized_redistribution=True),
                    refresh_cache_workers=True,worker_counts=counts,size_lane_authorization=authorization)
            prepared,deployment=validate_prepared(destination)
            if (prepared.get('execution_worker_nodes')!=counts or set(deployment)!=set(counts) or
                    prepared.get('cache_size_lane_authorization')!=authorization or
                    sha(destination/'EPISODE_PLAN.json')!=sha(snapshot/'EPISODE_PLAN.json')):
                pass
            job.update(name=name,run=str(destination),prepared_source=source['snapshots'][model],
                execution_worker_nodes=counts)
            job['argv'][job['argv'].index('--run-name')+1]=name
            state['jobs'][model]=dict(run=str(destination),status='PREPARED')
            results.append(dict(model=model,run=str(destination)))
            write_json(directory/'LANE_STATE.json',state)
        plan.update(worker_counts=None,execution_worker_nodes_by_model={j['model']:FLEETS[j['model']] for j in plan['jobs']},
            local_server_growth_mib=growth_mib,source_prepared_lane=str(original),
            after_lane=lane.predecessor(plan['after_lane']['directory'],which,directory))
        write_json(directory/'LANE_PLAN.json',plan)
        state.update(status='PREPARED',results=results)
        write_json(directory/'LANE_STATE.json',state)
        write_json(directory/'LANE_RECEIPT.json',dict(status='PREPARED',cpu_only=True,
            all_owned_closed=True,results=results))
        print(which+' split size queue PREPARED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',choices=('4090','h100'),required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prefix',default='cszs')
    parser.add_argument('--local-server-growth-mib',type=int,choices=(8192,16384),default=8192)
    parser.add_argument('--job-name',action='append',default=[],metavar='MODEL:NAME')
    args=parser.parse_args()
    names={}
    for item in args.job_name:
        model,name=item.split(':',1)
        if model not in FLEETS or model in names:parser.error('distinct known model names required')
        names[model]=name
    prepare(args.lane,args.directory,prefix=args.prefix,growth_mib=args.local_server_growth_mib,names=names)


if __name__=='__main__':main()
