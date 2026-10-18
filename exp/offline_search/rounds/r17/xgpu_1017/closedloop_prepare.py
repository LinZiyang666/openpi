"""CPU-only exact formal task manifests, one model and two independently analysed arms.

Does not deploy, start workers or invoke formal_eval's legacy transfer helpers.
The resulting manifests need a coordinator-scheduled renderer fleet.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from artifacts import digest, sha, write_json
from concurrent_server import checked_pin
from replay import verify_runtime


def prepare(bundle, output, inits):
    if inits not in (2,10,50):
        raise ValueError('2 init cost pilot, 10 init paired pilot or 50 init campaign required')
    bundle,output=Path(bundle).resolve(),Path(output).resolve()
    pin=checked_pin(bundle)
    source=json.loads((bundle/'source_campaign.json').read_text())
    runtime=Path(source['runtime'])
    verify_runtime(runtime,pin['runtime_identity'])
    sys.path[:0]=[str(runtime),str(runtime/'src'),str(runtime/'packages/openpi-client/src')]
    from exp.r17_integration.ops.formal_driver import plan
    from exp.r17_integration.common.contracts import digest as formal_digest
    output.mkdir(parents=True,exist_ok=False)
    groups=json.loads((bundle/'assets/selected_groups.json').read_text())
    cache_group=groups[0]
    original_seal=json.loads((Path(source['local_assets'])/'CACHE_STUDY.json').read_text())
    reports={}
    for arm in ('pure','cache'):
        root=output/arm;assets=root/'assets';assets.mkdir(parents=True)
        for f in (bundle/'assets').rglob('*'):
            if f.is_file():
                target=assets/f.relative_to(bundle/'assets');target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(f,target)
        c={k:v for k,v in source.items() if not k.startswith(('cache_resume','resume_'))}
        c.update(run_name='xgpu_preregister_v1_'+pin['model']+'_'+arm,
            semantic_campaign='xgpu_preregister_v1_'+pin['model'],
            cache_plan_name='xgpu_preregister_v1_'+pin['model'],local_assets=str(assets),assets=str(assets),
            endpoint_host='127.0.0.1',public_port=29600,endpoint_port=29600,
            endpoint='127.0.0.1:29600',driver_port=29690,smoke=False,library=False,
            cache_study=arm=='cache',init_count=inits,suites=[cache_group['suite']],
            workers=pin['clients'],execution_workers=pin['clients'],
            eval_pool='A',retries=0)
        if source['domain']=='metaworld' and arm=='pure':c['suites']=None
        if arm=='cache':
            group=dict(cache_group,indices=list(range(inits)),num_trials_per_task=50)
            seal=dict(original_seal,groups=[group]);seal.pop('digest',None)
            seal['digest']=formal_digest(seal)
            write_json(assets/'CACHE_STUDY.json',seal)
        write_json(root/'campaign.json',c)
        benchmark,model,tasks,endpoints=plan(c)
        serial=[asdict(t) for t in tasks]
        expected=(50 if source['domain']=='metaworld' else 10)*inits
        if len(serial)!=expected or len({t.task_uid for t in tasks})!=expected:
            raise ValueError('formal planner did not produce the preregistered paired population')
        scientific=dict(model=pin['model'],arm=arm,suite=cache_group['suite'],
            pin=pin['identity'],benchmark_protocol=benchmark.identity,tasks=serial)
        # Host/port values remain a transport binding; scientific task/seed
        # fields are generated once here and must not be regenerated per GPU.
        write_json(root/'PLAN.json',dict(scientific,identity=digest(scientific)))
        reports[arm]=dict(episodes=len(tasks),plan_sha256=sha(root/'PLAN.json'),
            cache_group=cache_group['group'] if arm=='cache' else None)
    report=dict(schema='xgpu.closedloop.preregistration.v1',model=pin['model'],
        margin=.02,confidence=.95,arms=reports,inits_per_task=inits,
        runtime_identity=pin['runtime_identity'],source_campaign_sha256=pin['source_campaign_sha256'],
        run_status='PREPARED_NOT_LAUNCHED',renderer_admission='REQUIRED',
        analysis='per model/suite/arm; exact paired CI, exact McNemar, task-cluster sensitivity',
        transport='no push/pull/expose launcher is called by this preparation')
    write_json(output/'PREREGISTRATION.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--inits',type=int,default=50)
    a=p.parse_args();print(json.dumps(prepare(a.bundle,a.output,a.inits)))
