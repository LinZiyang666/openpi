"""Clone a knob lane's definitions and sealed inputs; never stage or launch.

The original fit, view, Q3 and every calibrated gap stay byte-identical. The
resident entry applies filter_failed.row_filter to the verified fit in memory.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import shutil
from collections import defaultdict

import yaml

from .common import HERE, EVAL, REPO, PYTHON, cells, dump, read, sha, source_jobs
from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
from exp.r17_integration.common.contracts import digest


def cpu_plan(campaign):
    """Standard planner, memoizing only hashes of its immutable benchmark."""
    from contextlib import ExitStack
    from unittest.mock import patch
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    from exp.r17_integration.ops import cache_study
    benchmark=load_benchmark_spec(Path(campaign['local_assets'])/'benchmark.json')
    cls=type(benchmark)
    with ExitStack() as stack:
        for name in ('identity','task_table_identity'):
            prop=getattr(cls,name);values={}
            def get(self,prop=prop,values=values):
                if id(self) not in values:values[id(self)]=(self,prop.fget(self))
                return values[id(self)][1]
            stack.enter_context(patch.object(cls,name,property(get)))
        return cache_study.plan(campaign)


def publish_study(root,source,campaign,seal):
    """Keep the ordinary cache-study manifest/launcher format."""
    if (root/'LAUNCH.json').exists():
        raise ValueError('Cannot change a launched definition')
    original=Path(read(source/'campaign.json')['cache_study'])
    if not original.is_absolute():original=REPO/original
    manifest=copy.deepcopy(read(original))
    bundle=campaign['model']+('_mw' if campaign['domain']=='metaworld' else '_libero')
    # A resumed lane can split a config into several task/init cohorts. The
    # study lists each config once; CACHE_STUDY retains all dispatch cohorts.
    configurations={(r['suite'],r.get('library_size',500),r['group']):r for r in seal['groups']}
    manifest.update(models={bundle:[dict(r,file='assets/'+r['file']) for r in configurations.values()]},
                    groups=list(dict.fromkeys(r['group'] for r in seal['groups'])),
                    library_sizes=sorted({r.get('library_size',500) for r in seal['groups']}),
                    eval_pool=seal.get('eval_pool','A'),filter_failed_knob=dict(source=str(source),refit=False))
    manifest['digest']=digest({k:v for k,v in manifest.items() if k!='digest'})
    dump(root/'STUDY.json',manifest)
    inputs=read(root/'assets/cache_inputs/manifest.json')
    inputs['source_study_digest']=manifest['digest']
    inputs['digest']=digest({k:v for k,v in inputs.items() if k!='digest'})
    dump(root/'assets/cache_inputs/manifest.json',inputs)
    seal.update(source_study_digest=manifest['digest'],inputs_digest=inputs['digest'])
    seal['digest']=digest({k:v for k,v in seal.items() if k!='digest'})
    dump(root/'assets/CACHE_STUDY.json',seal)


def freeze_bootstrap(root,campaign):
    if (root/'LAUNCH.json').exists():
        raise ValueError('Cannot change a launched definition')
    from .launch import bootstrap
    path=root/'assets/ffk_bootstrap.py'
    path.write_text(bootstrap())
    campaign['ffk_bootstrap_sha256']=sha(path)
    path=root/'assets/ffk_evidence.py'
    path.write_bytes((HERE/'evidence.py').read_bytes())
    campaign['ffk_evidence_sha256']=sha(path)


def direct_routes(campaign,root):
    """Use the source pipeline's current direct-port allocation, never a broker."""
    from exp.r17_integration.ops.formal_eval import open_tunnels
    if campaign['domain']=='libero':
        port={'dp':23110,'cosmos':23112,'fastwam':23113 if campaign['server_node']=='local' else 23111}[campaign['model']]
        old=campaign['driver_port'];campaign['driver_port']=port
        campaign['ports']=[port if p==old else p for p in campaign.get('ports',[])]
    open_tunnels(campaign,root)


def complete_cache_inputs(root, official, groups, inputs):
    """Fill raised cells from official A, preserving every existing input byte."""
    other=read(official/'manifest.json')
    for name in inputs['files'].keys() & other['files'].keys():
        pin=inputs['files'][name]
        if other['files'][name]!=pin or sha(root/name)!=pin or sha(official/name)!=pin:
            raise ValueError(f'Raise/official-A cache input differs: {name}')
    required={}
    for row in groups:
        if not any((g['suite'],g['library_size'])==(row['suite'],row['library_size']) for g in other['groups']):
            raise ValueError(f'Official A lacks raised cell: {row["suite"]}/{row["library_size"]}')
        for prefix in (row['view_rel'],row['fit_rel'],row['stall_rel']):
            entries={n:p for n,p in other['files'].items() if n.startswith(prefix+'/')}
            if not entries:raise ValueError(f'Official A lacks cache inputs: {prefix}')
            required.update(entries)
        parent=row['suite']+'/parent/READY.json'
        required[parent]=other['files'][parent]
    missing={n:p for n,p in required.items() if n not in inputs['files']}
    # Verify all additions before writing; a sparse or changed A source is not
    # an alternative model/fit release to silently substitute.
    for name,pin in missing.items():
        if sha(official/name)!=pin:raise ValueError(f'Official-A input seal differs: {name}')
    for name in missing:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(official/name,target)
    inputs['files'].update(missing)


def build(source, name, *, table=None, unfiltered=False, off=False, ladder_source=None, records=None, cache_inputs_source=None, test_levels=None, test_cells=None):
    from exp.r17_integration.common.config import config_from_dict
    from exp.r17_integration.common.recipe import bundle_identity, recipe_identity
    from exp.r17_integration.ops import cache_study
    source = Path(source).resolve()
    if not name.startswith(('ffk1015_','fa_')): raise ValueError('New ffk1015_ or formal-A fa_ name required')
    root = EVAL/name
    root.mkdir(exist_ok=False)
    campaign = copy.deepcopy(read(source/'campaign.json'))
    for key in list(campaign):
        if key.startswith(('resume_', 'cache_resume', 'cache_worker_refresh')) or key in (
                'orchestration', 'runtime_source_pins', 'server_campaign'):
            campaign.pop(key, None)
    campaign.update(run_name=name, cache_plan_name=campaign.get('cache_plan_name',campaign['run_name']), prepare_only=True,
                    local_assets=str(root/'assets'), assets=str(root/'assets'), server_run=str(root/'servers'),
                    runtime_from=str(source), filter_failed_source=str(source),
                    cache_study=str(root/'STUDY.json'))
    campaign['server_python'] = PYTHON[campaign['server_node'], campaign['model']]
    # fsreddp2 was sealed with the newer, separate DP environment.
    if read(source/'assets/model_manifest.json').get('dependency_versions',{}).get('numpy') == '2.2.6' and campaign['model']=='dp':
        campaign['server_python'] = '/data/mw_finetune/dp/env/bin/python'
    campaign['ffk_dp_capacity_bridge'] = (campaign['model']=='dp' and campaign['domain']=='libero'
        and campaign['replicas'] in (3,4) and (source/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json').exists())
    direct_routes(campaign,root)
    # Copy, never rewrite source assets. All model descriptors keep exact bytes.
    shutil.copytree(source/'assets', root/'assets', ignore=shutil.ignore_patterns('.eval.lock','TRANSFER_SHA256.json'))
    for filename in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json'):
        shutil.copyfile(source/filename, root/filename)
    seal = read(source/'assets/CACHE_STUDY.json')
    ladder_source = Path(ladder_source or source)
    original_groups = copy.deepcopy(read(ladder_source/'assets/CACHE_STUDY.json')['groups'])
    if table is not None:
        existing={(r['suite'],r['library_size'],r['group'].rsplit('_',1)[-1]) for r in original_groups}
        for row in list(original_groups):
            if row['group'].endswith('_koff') and (row['suite'],row['library_size'],'pure') not in existing:
                pure=copy.deepcopy(row)
                pure.update(_source_file=pure['file'],file=pure['file'].replace('_koff','_pure'),
                            group=pure['group'].replace('_koff','_pure'),periodic_gap=0.)
                original_groups.append(pure)
    selected, selection, summaries = [], {}, {}
    index = cells()
    for original in original_groups:
        row = copy.deepcopy(original)
        if row['group'].endswith('_pure') and table is None:
            # Library-independent pure references always come from the source.
            continue
        suite, size = row['suite'], row['library_size']
        if test_cells is not None and (suite,size) not in test_cells:continue
        cell = index[campaign['model'],suite,size]
        summary_key = f'{suite}/{size}'
        summaries[summary_key] = {k:cell[k] for k in ('collected','kept','failed','empty_tasks','candidate_view','labels')}
        if not unfiltered: selection[row['fit_digest']] = cell['kept_episode_ids']
        level = row['group'].rsplit('_',1)[-1]
        if test_levels is not None and level not in test_levels:continue
        if off and level not in ('koff', 'pure'):
            row['group'] += '_koff'
            level = 'koff'
        if off and level != 'koff': continue
        doc = yaml.safe_load((ladder_source/'assets'/row.pop('_source_file',row['file'])).read_text())
        doc['recipe']['periodic_gap']=row['periodic_gap']
        names = [t for t in doc['recipe']['ordered_tasks'] if suite=='mt50' or t.startswith(suite+'/')]
        if table is not None or test_levels is not None:
            values = table.get({'pi05':'π0.5','groot':'GR00T','dp':'DP','fastwam':'Fast-WAM','cosmos':'Cosmos'}[campaign['model']],table) if table is not None else {}
            chosen = values.get(suite,{}).get(str(size),{})
            row['tasks'] = [t for t in names if chosen.get(t)==level] if table is not None else names
            if not row['tasks']: continue
            # DEV templates become official-A frozen/raise test rows.
            row.update(indices=list(range(50)), serve_loeo=False, num_trials_per_task=50)
            doc['recipe']['serve_loeo'] = False
        # Restore the whole planned cohort even when the source is a resume.
        if table is None and test_levels is None:
            row.pop('tasks',None)
            row['indices'] = list(range(50 if campaign['domain']=='metaworld' or off else 30))
        doc['recipe'].update(store_path=str(root/'assets/cache_inputs'/row['view_rel']),
                             fitted_path=str(root/'assets/cache_inputs'/row['fit_rel']),
                             stall_path=str(root/'assets/cache_inputs'/row['stall_rel']))
        register_metadata(source, doc)
        cfg = config_from_dict(doc)
        identity = bundle_identity(cfg)
        row.update(bundle_identity=identity,recipe_identity=recipe_identity(cfg),
                   bundle_id=f'ffk_n{size}_{cache_study.suite_label(suite)}_{row["group"]}_{identity[:12]}')
        path = root/'assets'/row['file']
        path.write_text(yaml.safe_dump(doc,sort_keys=False))
        row['sha256'] = sha(path)
        if records is None:
            selected.append(row)
        else:
            by_task=defaultdict(list)
            for t in records:
                ex=t['extra']
                if (ex['cache_group'],ex['library_size'])==(row['group'],size) and (suite=='mt50' or t['experiment']==suite):
                    by_task[ex['task_name']].append(t['orig_init_state_idx'])
            buckets=defaultdict(list)
            for task,indices in by_task.items():buckets[tuple(sorted(indices))].append(task)
            for indices,tasks in buckets.items():
                selected.append(dict(row,tasks=sorted(tasks),indices=list(indices),num_trials_per_task=50))
    seal.update(groups=selected)
    if table is not None or test_levels is not None: seal['eval_pool']='A'
    seal['digest'] = digest({k:v for k,v in seal.items() if k!='digest'})
    dump(root/'assets/CACHE_STUDY.json', seal)
    # Input manifest is a byte inventory, not a fitter or calibration step.
    inputs = read(source/'assets/cache_inputs/manifest.json')
    if cache_inputs_source is not None and Path(cache_inputs_source).resolve()!=source:
        complete_cache_inputs(root/'assets/cache_inputs',Path(cache_inputs_source)/'assets/cache_inputs',selected,inputs)
    inputs.update(groups=selected)
    inputs['digest'] = digest({k:v for k,v in inputs.items() if k!='digest'})
    dump(root/'assets/cache_inputs/manifest.json',inputs)
    seal['inputs_digest'] = inputs['digest']
    seal['digest'] = digest({k:v for k,v in seal.items() if k!='digest'})
    dump(root/'assets/CACHE_STUDY.json',seal)
    publish_study(root,source,campaign,seal)
    campaign['filter_failed_knob'] = dict(method='frozen_fit_candidate_rows_only',refit=False,
        selection=selection, cells=summaries, source=str(source), arm='unfiltered' if unfiltered else 'filtered')
    freeze_bootstrap(root,campaign)
    dump(root/'campaign.json',campaign)
    _, model, tasks, _ = cpu_plan(campaign)
    plan = read(source/'EPISODE_PLAN.json')
    plan.update(model=model,tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks),endpoint=campaign['endpoint'])
    dump(root/'EPISODE_PLAN.json',plan)
    dump(root/'OFFLINE_READY.json',dict(status='READY_FOR_COORDINATOR_STAGE',episodes=len(tasks),
                                      native_runtime_unchanged=True,worker_archive_unchanged=True))
    row = dict(name=name,run=str(root),source=str(source),episodes=len(tasks),model=campaign['model'],
               benchmark='mt50' if campaign['domain']=='metaworld' else 'libero',
               gpu='RTX 4090' if campaign['server_node']=='local' else 'H100',
               workers=campaign.get('execution_worker_nodes',campaign['worker_nodes']),replicas=campaign['replicas'],
               python=campaign['server_python'],runtime=campaign['runtime'],unfiltered=unfiltered)
    dump(root/'FFK_DEFINITION.json',row)
    return row


def initial():
    jobs=[]
    for source in source_jobs():
        if source['key']=='pi05_mt50_full' and (HERE/'MT50_PI05_FULL_ORIGINS.json').exists():
            from .origins import audit
            cohorts=None
            for i,group in enumerate(read(HERE/'MT50_PI05_FULL_ORIGINS.json')['groups'],1):
                name=f'ffk1015_pi05_mt50_full_descent_{i:02d}';root=EVAL/name
                if root.exists():job=read(root/'FFK_DEFINITION.json')
                else:
                    cohorts=cohorts or audit(source['source'])
                    job=build(group['source'],name,records=cohorts[group['source']])
                jobs.append(dict(job,mode='full',stage='descent'))
            dump(HERE/'CAMPAIGNS.json',dict(jobs=jobs))
            continue
        name='ffk1015_'+source['key']+'_'+source['stage']
        root=EVAL/name
        job=read(root/'FFK_DEFINITION.json') if root.exists() else build(source['source'],name)
        jobs.append(dict(job,mode=source['mode'],stage=source['stage']))
        dump(HERE/'CAMPAIGNS.json',dict(jobs=jobs))
    from .common import OFF
    for model, source in OFF.items():
        for unfiltered in ([False,True] if model=='cosmos' else [False]):
            name=f'ffk1015_{model}_libero_reduced_off'+('_u' if unfiltered else '')
            root=EVAL/name
            job=read(root/'FFK_DEFINITION.json') if root.exists() else build(EVAL/source,name,unfiltered=unfiltered,off=True)
            jobs.append(dict(job,mode='reduced',stage='off'))
            dump(HERE/'CAMPAIGNS.json',dict(jobs=jobs))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',action='store_true')
    p.add_argument('--source',type=Path);p.add_argument('--name');p.add_argument('--table',type=Path)
    p.add_argument('--ladder-source',type=Path)
    p.add_argument('--unfiltered',action='store_true');p.add_argument('--off',action='store_true')
    args=p.parse_args()
    if args.initial: initial()
    else: print(build(args.source,args.name,table=read(args.table) if args.table else None,unfiltered=args.unfiltered,off=args.off,ladder_source=args.ladder_source))


if __name__=='__main__':main()
