"""Materialise the twenty fixed-count formal pilot campaigns; CPU and local only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

W = Path(__file__).resolve().parent
REPO = W.parents[4]
sys.path.insert(0,str(REPO))
MODELS = ('pi05','groot','dp','fastwam','cosmos')
BUNDLES = dict(zip(MODELS,('pi05','groot_longseg_recal1018','dp_longseg_recal1018','fastwam_4090_reduced','cosmos')))
NODES = {'pi05':{'timan1':64,'timan107':48}, 'groot':{'timan1':20,'timan107':20},
         'dp':{'timan107':48},'fastwam':{'timan107':12},'cosmos':{'timan107':44}}
PYTHONS = {'4090':dict(zip(MODELS,('/home/weiland/projects/openpi/.venv/bin/python',
    '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python',
    '/home/weiland/projects/openpi/.venv/bin/python','/data/mw_finetune/envs/fastwam/bin/python',
    '/home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python'))),
    'h100':dict(zip(MODELS,('/home/exouser/openpi/.venv/bin/python',
    '/home/exouser/gr00t_n15_venv/.venv/bin/python','/home/exouser/openpi/.venv/bin/python',
    '/data/mw_finetune/envs/fastwam/bin/python','/data/cosmos/cosmos-policy/.venv/bin/python')))}

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def planner(campaign, output):
    c=read(campaign)
    code=('import json;from dataclasses import asdict;from pathlib import Path;'
          'from exp.r17_integration.ops.formal_driver import plan;'
          'c=json.loads(Path('+repr(str(campaign))+').read_text());'
          '_,_,tasks,_=plan(c);print(json.dumps([asdict(t) for t in tasks]))')
    runtime=c.get('driver_runtime',c['runtime'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=':'.join([runtime+'/packages/openpi-client/src',runtime+'/src',runtime]))
    r=subprocess.run([str(REPO/'.venv/bin/python'),'-c',code],env=env,cwd=runtime,
                     capture_output=True,text=True,check=True,timeout=120)
    tasks=json.loads(r.stdout);write(output,tasks);return tasks

def scientific(tasks):
    return [{k:v for k,v in t.items() if k not in ('server_host','server_port')} for t in tasks]

def pilot_metadata(root, model):
    # The native OFF cache planner honours this field; the older canonical
    # planner inferred it from indices. Match its ten-init MT50 plan exactly.
    path=root/'assets/CACHE_STUDY.json'
    if not path.exists():
        shutil.copyfile(W/'closedloop_pilot_1018'/model/'cache/assets/CACHE_STUDY.json',path)
    if model in ('pi05','groot'):
        from exp.r17_integration.common.contracts import digest
        seal=read(path)
        for group in seal['groups']:group['num_trials_per_task']=10
        seal.pop('digest',None);seal['digest']=digest(seal);write(path,seal)

def materialise():
    index=W/'closedloop_launch_1018';index.mkdir(exist_ok=True)
    if (index/'CAMPAIGNS.json').exists():raise ValueError('already materialised; do not overwrite campaigns')
    jobs=[]
    for host in ('4090','h100'):
      for model in MODELS:
       bundle=W/'concurrent/bundles'/BUNDLES[model];pin=read(bundle/'PIN.json')
       for arm in ('pure','cache'):
        number=len(jobs);root=W/('P'+('0123456789ABCDEFGHIJ'[number]))
        draft=dict(host=host,model=model,arm=arm)
        if root.exists():
            if not (root/'XGPU_DRAFT.json').exists() or read(root/'XGPU_DRAFT.json')!=draft:
                raise ValueError('short directory is not this pilot draft')
            if any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json')):
                raise ValueError('cannot regenerate prepared/launched campaign')
        else:
            root.mkdir();write(root/'XGPU_DRAFT.json',draft)
        original=W/'closedloop_pilot_1018'/model/arm
        shutil.copytree(original/'assets',root/'assets',dirs_exist_ok=True)
        pilot_metadata(root,model)
        shutil.copyfile(original/'PLAN.json',root/'PLAN.json')
        c=read(original/'campaign.json');source=Path(pin['source'])
        # Baseline source may precede the longseg overlay; its archives are identical.
        inheritance=Path(c.get('runtime_from') or pin['source'])
        if sha(source/'runtime.tar')!=sha(inheritance/'runtime.tar'):
            raise ValueError('source worker/runtime inheritance changed')
        source=inheritance
        for name in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json'):
            shutil.copyfile(source/name,root/name)
        for key in list(c):
            if key.startswith(('autoscale','resume_','cache_resume')) or key in ('live_fleets','local_worker_exception','timan1_worker_lease'):
                c.pop(key,None)
        name='xp'+('4' if host=='4090' else 'h')+'_'+model+'_'+arm
        remote_root='/home/exouser/xgpu_1017/pilot/'+root.name
        c.update(run_name=name,filter_failed_source=str(source),runtime_from=str(source),
                 local_assets=str(root/'assets'),assets=str(root/'assets') if host=='4090' else remote_root+'/assets',
                 server_run=str(root/'servers') if host=='4090' else remote_root+'/servers',
                 server_node='local' if host=='4090' else 'h100',server_python=PYTHONS[host][model],
                 worker_nodes=copy.deepcopy(NODES[model]),execution_worker_nodes=copy.deepcopy(NODES[model]),
                 workers=sum(NODES[model].values()),execution_workers=sum(NODES[model].values()),
                 worker_egl_devices_by_node={'timan1':[2],'timan107':list(range(8))},
                 worker_egl_devices=list(range(8)),worker_egl_device=0,
                 public_port=23180,driver_port=23196 if host=='4090' else 23197,
                 backend_host='127.0.0.1',seconds=14400,prepare_only=True)
        if model=='groot' and host=='4090':
            c.update(replicas=6,memory_budget_gib=42,vendor='/home/weiland/projects/openpi_ext/third_party/gr00t_n15')
        if model=='cosmos' and host=='4090':c['vendor']='/home/weiland/projects/openpi_ext/lines/cosmos-policy'
        c['backend_ports']=list(range(23181,23181+c['replicas']))
        c['ports']=[23180,*c['backend_ports'],c['driver_port']]
        if c.get('cache_worker_orchestration'):
            c['cache_worker_orchestration'].update(worker_nodes=c['execution_worker_nodes'],total_workers=c['execution_workers'])
        from exp.r17_integration.ops.formal_eval import open_tunnels
        open_tunnels(c,root)
        if host=='h100':c['server_campaign']='/tmp/r17_formal/'+name+'/server/campaign.json'
        # Per-process aliases retain original recipe/YAML bytes and task identities.
        aliases={}
        if host=='h100':
            for path in (bundle/'data.files').read_text().splitlines():
                aliases[path]='/home/exouser/xgpu_1017/rootfs'+path
            if model=='dp':
                graph='/data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz'
                aliases[graph]='/home/exouser/xgpu_1017/rootfs'+graph
        if host=='4090' and model in ('groot','cosmos'):
            from concurrent_server import relocate_paths
            old='/home/exouser/gr00t_n15' if model=='groot' else '/data/cosmos/cosmos-policy'
            write(root/'assets/load.json',relocate_paths(read(root/'assets/load.json'),{old:c['vendor']}))
        c['xgpu_pilot']=dict(schema='xgpu.formal_pilot.v1',host=host,model=model,arm=arm,
            canonical_plan=str(original/'PLAN.json'),canonical_plan_sha256=sha(original/'PLAN.json'),
            serving_pin=pin['identity'],bundle=str(bundle),autoscale=False,
            native_worker_count=sum(NODES[model].values()),fixed_worker_nodes=NODES[model],
            capacity_arm=model=='groot' and host=='4090',read_aliases=aliases,
            numerics='native',trace='native client/ACK, all episodes, no RGB payloads',
            no_model_gpu_renderers=True)
        # seal_preserved hooks/evidence as ordinary prepared assets
        for filename in ('client_trace.py','pilot_trace.py','pilot_readmap.py'):
            shutil.copyfile(W/filename,root/'assets'/filename)
        evidence=Path(pin['source'])/'assets/ffk_evidence.py'
        if evidence.exists():shutil.copyfile(evidence,root/'assets/ffk_evidence.py')
        if host=='h100':
            aliases.update({str(p):c['assets']+'/'+str(p.relative_to(root/'assets'))
                            for p in (root/'assets').rglob('*') if p.is_file()})
        write(root/'campaign.json',c)
        # Bind the actual native OFF orchestration before constructing its
        # runnable task manifest (including the longseg task namespace).
        from pilot_formal import context,check
        from exp.r17_integration.ops import formal_runtime,driver_release
        with context(root):
            formal_runtime.attach(c,source)
            driver_release.prepare(c,root,REPO)
        write(root/'campaign.json',c)
        tasks=planner(root/'campaign.json',root/'PLANNED_TASKS.json')
        assert len(tasks)==500
        write(root/'EPISODE_PLAN.json',dict(model=read(root/'assets/model_spec.json'),tasks=tasks,
            expected_episodes=500,endpoint=c['endpoint'],xgpu_plan_sha256=sha(original/'PLAN.json')))
        check(root)
        # Frozen MPS paths, on the exact local/remote server root.
        from exp.r17_integration.models.fastwam.ops.mps import paths
        mps=paths(str(Path(c['server_run']).parent),c['device'])
        assert len(os.fsencode(mps[1]))<=85
        job=dict(name=name,root=str(root),host=host,model=model,arm=arm,episodes=500,
            replicas=c['replicas'],workers=c['execution_workers'],worker_nodes=NODES[model],
            plan_sha256=sha(root/'PLAN.json'),mps=[str(p) for p in mps])
        jobs.append(job)
        for mode in ('prepare','launch'):
            script=index/(mode+'_'+name+'.sh')
            script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(REPO)+'\n'+
                str(REPO/'.venv/bin/python')+' '+str(W/'pilot_formal.py')+' '+mode+' --run '+str(root)+'\n')
            script.chmod(0o755)
    write(index/'CAMPAIGNS.json',dict(schema='xgpu.formal.pilots.v1',jobs=jobs,
        status='MATERIALISED_CPU_ONLY',autoscale=False,launches=0))
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    print(json.dumps(materialise(),indent=2))
