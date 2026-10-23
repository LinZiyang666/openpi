"""Fresh native cache-study definitions with pinned decision-head clocks.

No deployment occurs here. Existing definitions, calibrated inputs and model
descriptors are read-only. YAML and task namespaces are preserved exactly.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex

import yaml

from .audit import HERE,EVAL
from . import io_budget as budget
from .runtime import installer,source as patch_source
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,sha,REPO

COMMANDS=HERE/'commands'

def commands(name):
    COMMANDS.mkdir(exist_ok=True)
    for verb,module in (('prepare','exp.offline_search.rounds.r17.filter_failed.stage'),
                        ('launch','exp.offline_search.rounds.r17.filter_failed_knob.launch')):
        path=COMMANDS/f'{verb}_{name}.sh'
        # Replace, never rewrite in place: this runs inside the very prepare script
        # being replaced, and bash keeps reading the open file by offset.
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
            shlex.quote(str(REPO/'.venv/bin/python'))+' -m '+module+' --run '+shlex.quote(str(EVAL/name))+'\n')
        tmp.chmod(0o755);tmp.replace(path)

def pin(root,c):
    from exp.offline_search.rounds.r17.filter_failed_knob.build import freeze_bootstrap
    freeze_bootstrap(root,c)
    path=root/'assets/ffk_bootstrap.py';body=path.read_text()
    marker="ffk_runtime=dict(__name__='__main__')"
    assert body.count(marker)==1
    body=body.replace(marker,installer(patch_source())+'\n'+marker,1)
    path.write_text(body);c['ffk_bootstrap_sha256']=sha(path)
    c.setdefault('longseg_1017',{}).update(clock_unit='segment_decision',tails='delivery_only',
        bootstrap_sha256=c['ffk_bootstrap_sha256'],no_refit=True,
        calibration_quantum='Q3 exec_steps per segment head; frozen artifact bytes')
    if c['longseg_1017'].get('recalibration'):
        c['longseg_1017'].update(no_refit=False,retrieval_fit_unchanged=True,corrector_unchanged=True,
            Q3_recalibrated=True,calibration_quantum='C10 decision units exec=commit=1; C5 unchanged')
    if c.get('longseg_1017_control_kind'):
        from .control_evidence import SUFFIX
        path=root/'assets/ffk_evidence.py'
        body=path.read_text()
        marker='    namespace=dict(vars(study),_ffk_empty_uids=empty_uids)'
        assert marker in body
        body=body.replace(marker,'    source=source.replace("cache_enabled=True,debug_capture=False", "cache_enabled=not campaign.get(\'longseg_1017_control_kind\'),debug_capture=False")\n'+marker,1)
        path.write_text(body+SUFFIX)
        c['ffk_evidence_sha256']=sha(path)

def variant(root,c,seal,kind):
    """Use existing fits/levels; only add OFF or a library-independent control."""
    from exp.r17_integration.common.config import config_from_dict,RecipeConfig
    from exp.r17_integration.common.recipe import bundle_identity,recipe_identity
    from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
    candidates={}
    for row in sorted(seal['groups'],key=lambda r:r['library_size']):
        key=row['suite'] if kind in ('long','pure') else (row['suite'],row['library_size'])
        candidates[key]=row
    result=[]
    for old in candidates.values():
        row=copy.deepcopy(old);doc=yaml.safe_load((root/'assets'/row['file']).read_text())
        base=row['group'].split('_')[0]
        if kind=='off':
            row.update(group=base+'_koff',periodic_gap=None)
            doc['recipe']['periodic_gap']=None
        else:
            row.update(group=base+'_pure',periodic_gap=0.,settings=dict(corrector=False,gated=False,
                       commit=10 if kind=='long' else 5,alias=None))
            doc.update(enabled=False,checkpoints={},recipe=asdict(RecipeConfig(corrector=False)))
            doc['adapter_binding']['commit_horizon']=row['settings']['commit']
            c['filter_failed_knob']=dict(method='frozen_fit_candidate_rows_only',refit=False,selection={},cells={},arm='unfiltered')
        row.update(indices=list(range(50)),serve_loeo=False,num_trials_per_task=50)
        row.pop('tasks',None)
        if kind=='off':doc['recipe']['serve_loeo']=False
        register_metadata(Path(c['filter_failed_source']),doc)
        cfg=config_from_dict(doc);identity=bundle_identity(cfg)
        row.update(bundle_identity=identity,recipe_identity=recipe_identity(cfg),
            bundle_id='ls_'+row['suite']+'_'+str(row['library_size'])+'_'+row['group']+'_'+identity[:12],
            file='cache_yamls/ls_'+row['suite']+'_'+str(row['library_size'])+'_'+row['group']+'.yaml')
        path=root/'assets'/row['file'];path.write_text(yaml.safe_dump(doc,sort_keys=False));row['sha256']=sha(path)
        result.append(row)
    seal['groups']=result

def clone(src,name,*,workers=None,groups=None,mode=None,stage=None,kind=None,edit=None):
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan,publish_study
    from exp.offline_search.rounds.r17.filter_failed.resume import topology
    src=Path(src).resolve();root=EVAL/name
    if root.exists():
        if any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json')):raise ValueError('Never change a prepared/launched rerun')
        if (root/'FFK_DEFINITION.json').exists():return read(root/'FFK_DEFINITION.json')
        import time
        real=root.resolve()  # set the real (/nvme) directory aside; a /data symlink is only a pointer
        real.rename(real.with_name(root.name+'.incomplete_'+str(time.time_ns())))
        if root.is_symlink():root.unlink()
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(root)  # real dir on /nvme, /data path a symlink
    original=read(src/'campaign.json');c=copy.deepcopy(original)
    stage_source=src
    while not (stage_source/'WORKER_DEPLOYMENT.json').exists():
        parent=read(stage_source/'campaign.json')
        ancestor=parent.get('filter_failed_source') or parent.get('runtime_from')
        if not ancestor or Path(ancestor)==stage_source:raise ValueError('No sealed worker deployment ancestor: '+str(src))
        stage_source=Path(ancestor)
    for key in list(c):
        if key.startswith(('cache_resume','resume_','cache_worker_refresh')) or key in ('orchestration','runtime_source_pins','server_campaign'):
            c.pop(key,None)
    c.update(run_name=name,cache_plan_name=c.get('cache_plan_name',original['run_name']),
        prepare_only=True,assets=str(root/'assets'),local_assets=str(root/'assets'),
        server_run=str(root/'servers'),runtime_from=str(stage_source),filter_failed_source=str(stage_source),cache_study=str(root/'STUDY.json'))
    # Same source-attested R4 admission presentation as the standard ffk build.
    admission=stage_source/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json'
    c['ffk_dp_capacity_bridge']=(c['model']=='dp' and c['domain']=='libero' and
        c['replicas'] in (3,4) and admission.exists())
    if not c.get('server_python'):
        from exp.offline_search.rounds.r17.filter_failed_knob.common import PYTHON
        c['server_python']=PYTHON[c['server_node'],c['model']]
        if c['model']=='dp' and read(src/'assets/model_manifest.json').get('dependency_versions',{}).get('numpy')=='2.2.6':
            c['server_python']='/data/mw_finetune/dp/env/bin/python'
    budget.tree(src/'assets',root/'assets')
    for relative in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json'):
        budget.copy(src/relative,root/relative)
    seal=read(root/'assets/CACHE_STUDY.json')
    if groups is not None:seal['groups']=copy.deepcopy(groups)
    # Pure-cache and unfiltered variant lanes use the same resident entry with
    # an empty row-selection map. No fitted quantity is computed here.
    c.setdefault('filter_failed_knob',dict(method='frozen_fit_candidate_rows_only',
        refit=False,selection={},cells={},source=str(src),arm='unfiltered'))
    if kind:variant(root,c,seal,kind)
    if kind in ('long','pure'):c['longseg_1017_control_kind']=kind
    if stage=='levels':
        # Deferred units are instantiated from frozen source ladders only
        # after OFF; apply the owner's new head-clock inputs before sealing.
        from .recalibrate import recipe_inputs,pin_plan_evidence
        recipe_inputs(root,c,seal)
    # Caller-supplied rewrite of the group YAMLs/rows and deployment fields (longseg_off_1022).
    if edit:edit(root,c,seal)
    # Preserve YAML bytes and task UIDs. These frozen YAMLs address the source
    # cache_inputs tree; H100 preparation stages that exact immutable tree too.
    roots=set()
    for row in seal['groups']:
        doc=yaml.safe_load((root/'assets'/row['file']).read_text())
        for key in ('store_path','fitted_path','stall_path'):
            value=doc.get('recipe',{}).get(key)
            if value and '/cache_inputs/' in value:roots.add(value.split('/cache_inputs/')[0]+'/cache_inputs')
    c['longseg_1017_input_roots']=sorted(roots)
    publish_study(root,src,c,seal)
    pin(root,c)
    if stage=='levels':pin_plan_evidence(root,c)
    from .ports import route
    route(c,root,workers)
    budget.dump(root/'OFFLINE_READY.json',dict(status='READY_FOR_COORDINATOR_STAGE',no_launch=True))
    budget.dump(root/'campaign.json',c)
    if workers:
        topology(root,','.join(f'{k}={v}' for k,v in workers.items()));c=read(root/'campaign.json')
    benchmark,model,tasks,endpoints=cpu_plan(c)
    original_plan_equal=None
    if groups is None and kind is None:
        old=copy.deepcopy(original);old.pop('cache_resume_selection',None)
        old['local_assets']=str(src/'assets')
        _,_,before,_=cpu_plan(old)
        assert [t.task_uid for t in before]==[t.task_uid for t in tasks]
        assert [t.extra for t in before]==[t.extra for t in tasks]
        original_plan_equal=True
    plan=dict(schema='r17.episode_plan.v1',model=model,tasks=[asdict(t) for t in tasks],
              expected_episodes=len(tasks),endpoint=c.get('endpoint'))
    budget.dump(root/'EPISODE_PLAN.json',plan,compact=True)
    job=dict(name=name,run=str(root),source=str(src),episodes=len(tasks),model=c['model'],
        benchmark='mt50' if c['domain']=='metaworld' else 'libero',mode=mode,stage=stage,
        gpu='H100' if c['server_node']=='h100' else 'RTX 4090',replicas=c['replicas'],
        batch=(c.get('dp_batching') or c.get('fastwam_batching') or {}).get('max_batch',1),
        workers=c.get('execution_worker_nodes',c['worker_nodes']),python=c['server_python'],runtime=c['runtime'],
        unfiltered=not bool(c['filter_failed_knob']['selection']),pipeline='R17',
        source_full_native_plan_equal=original_plan_equal,
        task_uid_sha256=hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest())
    budget.dump(root/'FFK_DEFINITION.json',job);commands(name)
    print(json.dumps(dict(name=name,episodes=len(tasks),gpu=job['gpu'],workers=job['workers'])),flush=True)
    return job

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--worker-counts',default='timan1=40,timan107=40')
    a=p.parse_args();clone(a.source,a.name,workers={k:int(v) for k,v in (s.split('=') for s in a.worker_counts.split(','))})

if __name__=='__main__':main()
