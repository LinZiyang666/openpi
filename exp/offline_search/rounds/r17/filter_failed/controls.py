"""Source-fit unfiltered controls in the filtered arm's chosen environment."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import copy
from pathlib import Path
import shlex
import shutil
import yaml

from .audit_counts import HERE, REPO, read
from .build import register_metadata, write, cell_id


def create(job, cells, baseline):
    from exp.r17_integration.ops import cache_study
    from exp.r17_integration.common.config import config_from_dict
    from exp.r17_integration.common.recipe import fit_identity, bundle_identity, recipe_identity
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.ops.immutable_release import sha
    oldroot, source = Path(job['run']), Path(job['source'])
    name = job['name']+'_u'
    root = oldroot.with_name(name)
    if (root/'LAUNCH.json').exists():raise ValueError('control already launched')
    if (root/'PREPARED.json').exists():
        saved=read(HERE/'CONTROL_JOBS.json')['jobs']
        return copy.deepcopy(next(j for j in saved if j['name']==name))
    root.mkdir(exist_ok=True)
    c = copy.deepcopy(read(oldroot/'campaign.json'))
    c.update(run_name=name, cache_plan_name=name, prepare_only=True, assets=str(root/'assets'),
             local_assets=str(root/'assets'), server_run=str(root/'servers'),
             filter_failed_arm='unfiltered_control', filter_failed_pair=job['name'])
    c['filter_failed'] = dict(method='unfiltered_original_size_N_control', refit=False)
    for key in ('server_campaign','orchestration','driver_runtime','driver_runtime_identity'):
        c.pop(key,None)
    shutil.copytree(oldroot/'assets', root/'assets', dirs_exist_ok=True)
    for filename in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json'):
        shutil.copyfile(oldroot/filename, root/filename)
    study_root = HERE/'studies'/(name+'_unfiltered')
    study_root.mkdir(parents=True,exist_ok=True)
    study = copy.deepcopy(read(job['study']))
    study.update(models={c['model']+'_libero':[]},groups=['c'],library_sizes=[],artifacts_pending=False)
    groups, input_rows, selected_cells = [], [], []
    original_seal = read(source/'assets/CACHE_STUDY.json')
    old_inputs = read(oldroot/'assets/cache_inputs/manifest.json')
    total=0
    for cell in cells:
        records=[r for r in baseline[cell_id(cell)] if r['source']==job['source']]
        if not records:continue
        original = copy.deepcopy(next(g for g in original_seal['groups'] if g['group']=='c'
                                     and g['suite']==cell['suite'] and g['library_size']==cell['size']))
        original.pop('tasks',None);original.pop('indices',None);original.pop('candidate_view',None)
        document=yaml.safe_load((source/'assets'/original['file']).read_text())
        register_metadata(source,document)
        for relative in (original['view_rel'],original['fit_rel']):
            shutil.copytree(source/'assets/cache_inputs'/relative, root/'assets/cache_inputs'/relative, dirs_exist_ok=True)
        document['recipe'].update(store_path=str(root/'assets/cache_inputs'/original['view_rel']),
                                  fitted_path=str(root/'assets/cache_inputs'/original['fit_rel']),
                                  fitted_digest=original['fit_digest'])
        document['recipe']['fit_identity']=fit_identity(config_from_dict(document))
        config=config_from_dict(document)
        identity=bundle_identity(config)
        target=root/'assets/cache_yamls'/f"n{cell['size']}_{cell['suite']}_c.yaml"
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(document,sort_keys=False))
        row=dict(original,file=str(target.relative_to(root/'assets')),sha256=sha(target),
                 fit=str(source/'assets/cache_inputs'/original['fit_rel']),num_trials_per_task=50,
                 bundle_id=f"ffunfiltered_n{cell['size']}_{cache_study.suite_label(cell['suite'])}_{identity[:12]}",
                 bundle_identity=identity,recipe_identity=recipe_identity(config))
        input_rows.append(row)
        by_task=defaultdict(list)
        for record in records:by_task[record['task']].append(record['init'])
        by_indices=defaultdict(list)
        for task,indices in by_task.items():by_indices[tuple(sorted(indices))].append(task)
        for indices,tasks in by_indices.items():groups.append(dict(row,tasks=sorted(tasks),indices=list(indices)))
        total+=len(records);selected_cells.append(cell_id(cell))
        # Reviewable standard study descriptors retain canonical original views.
        study_doc=copy.deepcopy(document)
        study_doc['recipe']['store_path']=str(cache_study.view_path(original))
        study_doc['recipe']['fitted_path']=str(source/'assets/cache_inputs'/original['fit_rel'])
        relative=f"dp_libero/{cell['suite']}/n{cell['size']}_c.yaml"
        path=study_root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(study_doc,sort_keys=False))
        study['models'][c['model']+'_libero'].append(dict(row,file=relative,sha256=sha(path)))
        study['library_sizes'].append(cell['size'])
    study['library_sizes']=sorted(set(study['library_sizes']))
    study['digest']=digest({k:v for k,v in study.items() if k!='digest'})
    write(study_root/'STUDY.json',study);c['cache_study']=str(study_root/'STUDY.json')
    old_inputs.update(groups=input_rows,source_study_digest=study['digest'])
    old_inputs['files']={str(p.relative_to(root/'assets/cache_inputs')):sha(p)
                         for p in (root/'assets/cache_inputs').rglob('*') if p.is_file() and p.name!='manifest.json'}
    old_inputs['digest']=digest({k:v for k,v in old_inputs.items() if k!='digest'})
    write(root/'assets/cache_inputs/manifest.json',old_inputs)
    seal=copy.deepcopy(read(oldroot/'assets/CACHE_STUDY.json'))
    seal.update(groups=groups,inputs_digest=old_inputs['digest'],source_study_digest=study['digest'])
    seal['digest']=digest({k:v for k,v in seal.items() if k!='digest'})
    write(root/'assets/CACHE_STUDY.json',seal);write(root/'campaign.json',c)
    _,model,tasks,_=cache_study.plan(c)
    assert len(tasks)==total
    plan=read(oldroot/'EPISODE_PLAN.json');plan.update(model=model,tasks=[asdict(t) for t in tasks],expected_episodes=total)
    write(root/'EPISODE_PLAN.json',plan)
    write(root/'OFFLINE_READY.json',dict(status='READY_FOR_COORDINATOR_STAGE',cpu_only=True,episodes=total,
                                       unfiltered_control=True,source=job['source'],empty_tasks_not_excluded=True))
    result=dict(job,name=name,run=str(root),episodes=total,study=str(study_root/'STUDY.json'),
                arm='unfiltered_control',paired_filtered=job['name'],replaces_source=job['source'],cells=selected_cells)
    for phase in ('prepare','launch'):
        module='stage' if phase=='prepare' else 'launch'
        command=['env',*(['CUDA_VISIBLE_DEVICES='] if phase=='prepare' else []),
                 'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH=.:src',
                 str(REPO/'.venv/bin/python'),'-m','exp.offline_search.rounds.r17.filter_failed.'+module,'--run',str(root)]
        path=HERE/'commands'/f'{phase}_{name}.sh'
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+shlex.join(command)+'\n')
        result[phase+'_command']='bash '+str(path)
    return result


def main():
    document=read(HERE/'CAMPAIGNS.json')
    checks={r['run']:r for r in read(HERE/'STARTUP_CHECKS.json')['jobs']}
    cells=read(HERE/'CELLS.json')['cells'];baseline=read(HERE/'BASELINE.json')
    jobs=[j for j in document['jobs'] if j.get('arm')!='unfiltered_control']
    controls=[]
    for job in jobs:
        if job['domain']=='r10_libero':continue
        check=checks[job['name']]
        assert check['status']=='PASS'
        if not check['equals_source_actual']:controls.append(create(job,[c for c in cells if c['source']=='R17' and c['model']==job['model']],baseline))
    document['jobs']=jobs+controls
    document['fairness']='same serving environment; rerun unfiltered controls replace historical outcomes where identities differ'
    write(HERE/'CAMPAIGNS.json',document)
    write(HERE/'CONTROL_JOBS.json',dict(jobs=controls,extra_simulations_by_model={m:sum(j['episodes'] for j in controls if j['model']==m)
          for m in ('pi05','groot','dp','fastwam','cosmos')}))
    print('Created',len(controls),'unfiltered controls:',sum(j['episodes'] for j in controls),'simulations')


if __name__=='__main__':main()
