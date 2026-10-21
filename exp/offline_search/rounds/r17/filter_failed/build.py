"""CPU-only frozen-fit Cache ablation preparation. Never contacts any worker."""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import yaml

from .audit_counts import HERE, REPO, read
from . import sources
from .row_filter import load_r10, save_r10, r17_candidates, r10_candidates
from exp.r17_integration.common.contracts import digest
from exp.r17_integration.ops.immutable_release import sha

TAG = 'keep_success_1015_norefit'
_metadata_samplers = {}


def register_metadata(source, document):
    """Register recorded native descriptors for CPU config parsing/planning.

    The actual resident registers its original provider on launch. This CPU
    provider has no model loader and only returns the sealed sampler identity.
    """
    from exp.r17_integration.ops.cache_study import model_spec
    from exp.r17_integration.common.provider import AdapterProvider, get_provider, register_provider
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    register_benchmark(load_benchmark_spec(Path(source)/'assets/benchmark.json'))
    binding=document['adapter_binding'];identity=binding['provider']
    samplers=_metadata_samplers.setdefault(identity,{})
    samplers[binding['num_steps']]=binding['sampler_identity']
    try:get_provider(identity)
    except ValueError:
        spec=model_spec(read(Path(source)/'assets/model_spec.json'))
        register_provider(AdapterProvider(identity,spec,lambda:None,
            lambda k:SimpleNamespace(num_steps=k,identity=samplers[k],family=spec.default_sampler.family)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def cell_id(cell):
    return f"{cell['model']}/{cell['suite']}/{cell['size']}"


def candidate_view(cell):
    selection = Path(cell['selection'])
    kept = set(cell['kept_episode_ids'])
    if cell['source']=='R17':
        original = read(selection)
        parent = Path(cell['parent'])
        destination = selection.parent.with_name(selection.parent.name + '_' + TAG)
        ready = read(parent / 'READY.json')
        indices = np.array([i for i,r in enumerate(ready['rows']) if r['library_episode_id'] in kept], np.int64)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / 'row_indices.npy', indices, allow_pickle=False)
        view = dict(original, rows=len(indices), episode_ids=sorted(kept), size=len(kept))
        view['metadata'] = dict(original['metadata'], collected_size=cell['size'], fit_policy='original size-N fit; no refit',
                                original_view_digest=original['content_digest'], successes_per_task=cell['successes_per_task'])
        view['metadata'].pop('episodes_per_task', None)
        view['row_indices_sha256'] = sha(destination / 'row_indices.npy')
        view['content_digest'] = digest({k:v for k,v in view.items() if k!='content_digest'})
        write(destination / 'VIEW.json', view)
        return str(destination / 'VIEW.json')
    original_rows = np.load(selection.with_suffix('.npy'), allow_pickle=False)
    ep = np.load(Path(cell['parent']) / 'episode.npy', mmap_mode='r')
    indices = original_rows[np.isin(ep[original_rows], list(kept))]
    destination = selection.with_name(selection.stem + '_' + TAG + '.json')
    np.save(destination.with_suffix('.npy'), indices, allow_pickle=False)
    write(destination, dict(parent=cell['parent'], collected_size=cell['size'], kept=cell['kept'],
                            episode_ids=sorted(kept), rows=len(indices), rows_sha256=sha(destination.with_suffix('.npy')),
                            original_selection=str(selection), fit_policy='original R10 A fit; no refit'))
    return str(destination)


def artifact(cell, source):
    from exp.r17_integration.search.fit import load_fitted_recipe, save_fitted_recipe
    if cell['source']=='R10':
        original = Path(cell['run_fit'])
        destination = Path(cell['original_fit']).with_name(original.stem + '_' + TAG + '.pkl')
        if not destination.exists():
            blob = load_r10(original)
            save_r10(r10_candidates(blob, cell['kept_episode_ids']), destination)
        return dict(path=str(destination), original=str(original), original_sha256=sha(original),
                    sha256=sha(destination))
    row = sources.group(source, cell['suite'], cell['size'])
    original = sources.served_fit(source, row)
    ready = read(original / 'READY.json')
    canonical = Path(row['fit'])
    if not canonical.is_absolute():
        canonical = REPO / canonical
    destination = canonical.with_name(canonical.name + '_' + TAG + '_' + ready['digest'][:8])
    if not destination.exists():
        fitted = load_fitted_recipe(original, expected_binding=ready['receipt']['binding'])
        selected = r17_candidates(fitted, cell['kept_episode_ids'])
        new = save_fitted_recipe(selected, ready['receipt'], destination)
        new['candidate_filter'] = dict(original_fit_digest=ready['digest'], original_fit=str(original),
                                       collected_size=cell['size'], kept_episodes=cell['kept_episode_ids'],
                                       candidate_view=cell['candidate_view'], refit=False)
        new['digest'] = digest({k:v for k,v in new.items() if k!='digest'})
        write(destination / 'READY.json', new)
    return dict(path=str(destination), original=str(original), original_digest=ready['digest'],
                original_sha256=sha(original / 'fit.npz'), digest=read(destination / 'READY.json')['digest'],
                original_manifest_sha256=sha(original / 'READY.json'))


def study_for(model, domain, cells):
    from exp.r17_integration.common.recipe import fit_identity
    from exp.r17_integration.common.config import config_from_dict
    bundle = model + ('_mw' if domain=='mt50' else '_libero')
    first = cells[0]
    origin = first['baseline'][0]['source']
    original_study = Path(read(Path(origin) / 'campaign.json')['cache_study'])
    manifest = copy.deepcopy(read(original_study))
    manifest.update(models={bundle:[]}, groups=['c'], library_sizes=sorted({c['size'] for c in cells}),
                    suites=['mt50'] if domain=='mt50' else ['libero_spatial','libero_10'],
                    artifacts_pending=[], ordinary_miss=False, knobs=False, eval_pool='A', inits=50,
                    filter_failed=dict(method='frozen_fit_candidate_rows_only', refit=False))
    root = HERE / 'studies' / bundle
    for cell in cells:
        source = cell['baseline'][0]['source']
        row = copy.deepcopy(sources.group(source, cell['suite'], cell['size']))
        document = yaml.safe_load((Path(source) / 'assets' / row['file']).read_text())
        register_metadata(source,document)
        filtered = cell['artifacts'][source]
        # Descriptor/receipt continue to bind the ORIGINAL full size-N store.
        # Only candidate tables inside the frozen-format serving artifact differ.
        row.update(fit=filtered['path'], fit_digest=filtered['digest'], library_size=cell['size'],
                   candidate_view=cell['candidate_view'], original_fit_digest=filtered['original_digest'],
                   tasks=[t for t,n in cell['successes_per_task'].items() if n],
                   status='ARTIFACTS_READY_MODEL_ADMISSION_PENDING')
        file = f"{bundle}/{cell['suite']}/n{cell['size']}_c.yaml"
        row['file'] = file
        document['recipe'].update(fitted_path=row['fit'], fitted_digest=row['fit_digest'])
        from exp.r17_integration.ops.cache_study import view_path
        document['recipe']['store_path']=str(view_path(row))
        document['recipe']['fit_identity'] = fit_identity(config_from_dict(document))
        target = root / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(document, sort_keys=False))
        row['sha256'] = sha(target)
        manifest['models'][bundle].append(row)
    manifest['digest'] = digest({k:v for k,v in manifest.items() if k!='digest'})
    write(root / 'STUDY.json', manifest)
    return str(root / 'STUDY.json')


def prepare_r17(source, selected, study_path, number):
    """Materialize a local campaign; standard remote CPU staging is coordinator-only."""
    from exp.r17_integration.common.config import config_from_dict
    from exp.r17_integration.common.recipe import fit_identity, bundle_identity, recipe_identity
    from exp.r17_integration.ops import cache_study
    source = Path(source)
    original = read(source / 'campaign.json')
    name = f"ff1015_{original['model']}_{'mt' if original['domain']=='metaworld' else 'li'}_{number:02d}"
    root = sources.EVAL / name
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(root, exist_ok=True)  # real dir on /nvme, /data path a symlink
    campaign = copy.deepcopy(original)
    for key in list(campaign):
        if key.startswith(('resume_', 'cache_resume', 'cache_capacity', 'cache_worker_refresh')) or key in (
            'runtime_source_pins', 'runtime_from', 'resume_from', 'orchestration', 'cache_control_proof',
            'cache_size_lane_authorization', 'cache_plan_name'):
            campaign.pop(key, None)
    campaign.update(run_name=name, cache_plan_name=name, cache_study=study_path, prepare_only=True,
                    local_assets=str(root / 'assets'), assets=str(root / 'assets'),
                    server_run=str(root / 'servers'), runtime_from=str(source),
                    filter_failed_source=str(source), filter_failed=dict(method='frozen_fit_candidate_rows_only',refit=False))
    if (HERE/'SERVER_ENVIRONMENTS.json').exists():
        selected=read(HERE/'SERVER_ENVIRONMENTS.json')['jobs'].get(name)
        if selected:campaign['server_python']=selected['python']
    campaign.pop('server_campaign', None)
    port = 23150 if campaign['server_node']=='local' else 23420
    from exp.r17_integration.ops.formal_eval import h100_host
    host = 'ziyanglin.com' if campaign['server_node']=='local' else h100_host()
    campaign.update(public_port=port,backend_ports=list(range(port+1,port+1+campaign['replicas'])),
                    driver_port=23159,driver_public_port=23159,driver_host='ziyanglin.com',
                    endpoint_host=host,endpoint_port=port,endpoint=f'{host}:{port}')
    campaign['ports']=[port,*campaign['backend_ports'],23159]
    # Preserve actual execution topology, batch settings, renderers and native runtime.
    # Reuse original ports sequentially; no resource is claimed during preparation.
    assets = root / 'assets'
    assets.mkdir(exist_ok=True)
    for p in (source / 'assets').iterdir():
        if p.is_file() and p.name not in ('CACHE_STUDY.json','TRANSFER_SHA256.json','.eval.lock'):
            shutil.copyfile(p, assets / p.name)
    for name_ in ('runtime.tar', 'workers.tar', 'RUNTIME_RELEASE.json'):
        if not (root/name_).exists():
            shutil.copyfile(source/name_, root/name_)
    source_seal = read(source / 'assets/CACHE_STUDY.json')
    groups = []
    inputs = dict(schema='r17.cache_inputs.v1', source_study_digest=read(study_path)['digest'],
                  model=campaign['model']+('_mw' if campaign['domain']=='metaworld' else '_libero'),
                  groups=[], eval_pool='A', prereg_sha256=read(study_path)['prereg_sha256'],
                  rulings_sha256=read(study_path)['rulings_sha256'],
                  horizon_selection=source_seal['horizon_selection'], files={})
    for cell, records in selected:
        row = copy.deepcopy(sources.group(str(source), cell['suite'], cell['size']))
        filtered = cell['artifacts'][str(source)]
        for relative, names in ((row['view_rel'], ('VIEW.json','row_indices.npy')),
                                (row['fit_rel'], ('READY.json','fit.npz'))):
            target = assets / 'cache_inputs' / relative
            target.mkdir(parents=True, exist_ok=True)
            src = source / 'assets/cache_inputs' / relative if relative==row['view_rel'] else Path(filtered['path'])
            for name_ in names:
                shutil.copyfile(src / name_, target / name_)
        # The view references parent metadata at ../.., as in the original seal.
        parent_rel = str(Path(row['view_rel']).parent.parent / 'READY.json')
        target = assets / 'cache_inputs' / parent_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source / 'assets/cache_inputs' / parent_rel, target)
        document = yaml.safe_load((source / 'assets' / row['file']).read_text())
        register_metadata(source,document)
        document['recipe'].update(store_path=str(assets/'cache_inputs'/row['view_rel']),
                                  fitted_path=str(assets/'cache_inputs'/row['fit_rel']), fitted_digest=filtered['digest'])
        document['recipe']['fit_identity'] = fit_identity(config_from_dict(document))
        config = config_from_dict(document)
        assert config.recipe.corrector is False and config.recipe.miss_mode=='never'
        assert not config.recipe.guard and config.recipe.periodic_gap is None
        identity = bundle_identity(config)
        target = assets / 'cache_yamls' / f"n{cell['size']}_{cell['suite']}_c.yaml"
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(document, sort_keys=False))
        row.update(file=str(target.relative_to(assets)),sha256=sha(target),fit=filtered['path'],
                   fit_digest=filtered['digest'],library_size=cell['size'],candidate_view=cell['candidate_view'],
                   num_trials_per_task=50,
                   bundle_id=f"ffcache_n{cell['size']}_{cache_study.suite_label(cell['suite'])}_c_{identity[:12]}",
                   bundle_identity=identity,recipe_identity=recipe_identity(config))
        by_task = defaultdict(list)
        for record in records:
            by_task[record['task']].append(record['init'])
        by_indices = defaultdict(list)
        for task, indices in by_task.items():
            by_indices[tuple(sorted(indices))].append(task)
        for indices, tasks in by_indices.items():
            groups.append(dict(row,tasks=sorted(tasks),indices=list(indices)))
        inputs['groups'].append(row)
    for p in (assets/'cache_inputs').rglob('*'):
        if p.is_file():inputs['files'][str(p.relative_to(assets/'cache_inputs'))]=sha(p)
    inputs['digest'] = digest(inputs)
    write(assets/'cache_inputs/manifest.json',inputs)
    seal = dict(source_seal,groups=groups,inputs_digest=inputs['digest'],source_study_digest=read(study_path)['digest'])
    seal['digest'] = digest({k:v for k,v in seal.items() if k!='digest'})
    write(assets/'CACHE_STUDY.json',seal)
    write(root/'campaign.json',campaign)
    # This standard planner already supports per-row tasks/indices. No serving
    # implementation changes or model/worker archive overlays are required.
    _, model, tasks, _ = cache_study.plan(campaign)
    expected = sum(len(rows) for _,rows in selected)
    assert len(tasks)==expected
    plan = {k:v for k,v in read(source/'EPISODE_PLAN.json').items() if k!='tasks'}
    plan.update(model=model,tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks),endpoint=campaign['endpoint'])
    write(root/'EPISODE_PLAN.json',plan)
    write(root/'OFFLINE_READY.json',dict(status='READY_FOR_COORDINATOR_STAGE',cpu_only=True,
          source=str(source),native_runtime_unchanged=sha(root/'runtime.tar')==sha(source/'runtime.tar'),
          worker_archive_unchanged=sha(root/'workers.tar')==sha(source/'workers.tar'),episodes=len(tasks)))
    return dict(name=name,run=str(root),source=str(source),model=campaign['model'],domain=campaign['domain'],
                gpu='RTX 4090' if campaign['server_node']=='local' else 'H100',replicas=campaign['replicas'],
                batch=max((campaign.get(k) or {}).get('max_batch',1) for k in ('dp_batching','fastwam_batching','stage_batching')),
                workers=campaign.get('execution_worker_nodes',campaign['worker_nodes']),episodes=len(tasks),
                runtime=campaign['runtime'],runtime_identity=campaign['runtime_identity'],study=study_path)


def prepare_r10(model, cells):
    from exp.offline_search.closed_loop.ops.emit_arms import main as emit
    root = sources.LEGACY_RUNS / f'r17_filter_failed_{model}'
    root.mkdir(exist_ok=True)
    original = sources.LEGACY_RUNS / f'r10_size_{model}'
    specs = read(original / 'arms_in.json')
    rows = []
    for cell in cells:
        short = 'spatial' if cell['suite']=='libero_spatial' else 'l10'
        oldname = f"r10_{model}_{short}_{cell['size']}_A"
        row = copy.deepcopy(next(r for r in specs if r['name']==oldname))
        name = f"ff1015_{model}_{short}_n{cell['size']}_c"
        row['name'] = name
        fit = root/'fits'/f'{name}.pkl'
        fit.parent.mkdir(exist_ok=True)
        shutil.copyfile(cell['artifacts'][str(original)]['path'],fit)
        args = row['plugin_args'];i=args.index('--os-fit-artifact')
        args[i+1]=str(fit)
        manifest = root/'manifests'/f'{name}.json'
        chosen = [[t,i] for t in range(10) if str(t) not in set(map(str,cell['empty_tasks'])) for i in range(50)]
        write(manifest,chosen)
        row['manifest']=str(manifest)
        rows.append(row)
        cell['filtered_run']=str(root);cell['filtered_arm']=name
    write(root/'arms_in.json',rows)
    emit(['--run-root',str(root),'--spec',str(root/'arms_in.json')])
    frozen={r['arm']:r for r in read(original/'arms.json')}
    emitted=read(root/'arms.json')
    for row in emitted:
        short=row['suite_short'];size=row['kwargs']['size']
        old=frozen[f'r10_{model}_{short}_{size}_A']
        shutil.copyfile(old['yaml'],row['yaml'])
        row['yaml_sha256']=sha(row['yaml'])
        row['src_yaml']=old['yaml'];row['src_sha256']=row['yaml_sha256']
    write(root/'arms.json',emitted)
    # The original deployment used these topologies (chain.log + owner receipt).
    return dict(name=root.name,run=str(root),model=model,domain='r10_libero',gpu='H100',replicas=4,batch=1,
                workers={'timan107':48} if model=='pi05' else {'timan108':40},
                ports='23240,23241,23242,23243' if model=='pi05' else '23230,23231,23232,23233',
                wps=12 if model=='pi05' else 10,episodes=6000,arms=[r['name'] for r in rows])


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-views',action='store_true',help='Resume definitions after the candidate views were already written')
    args=parser.parse_args()
    assert read(HERE/'selection_proof/PROOF.json')['status']=='PASS'
    cells = read(HERE/'COUNTS.json')['cells']
    baseline_path = HERE/'BASELINE.json'
    baseline_index = read(baseline_path) if baseline_path.exists() else {}
    for cell in cells:
        key = cell_id(cell)
        if key not in baseline_index:
            baseline_index[key] = sources.baseline(cell)
            write(baseline_path,baseline_index)
        cell['baseline']=baseline_index[key]
        print('Prepare candidates '+key,flush=True)
        if args.reuse_views:
            p=Path(cell['selection'])
            cell['candidate_view']=str(p.parent.with_name(p.parent.name+'_'+TAG)/'VIEW.json' if cell['source']=='R17'
                                       else p.with_name(p.stem+'_'+TAG+'.json'))
            assert Path(cell['candidate_view']).exists()
        else:cell['candidate_view']=candidate_view(cell)
        cell['artifacts']={source:artifact(cell,source) for source in sorted({r['source'] for r in cell['baseline']})}
    write(HERE/'CELLS.json',dict(method='frozen_fit_candidate_rows_only',refit=False,
          cells=[{k:v for k,v in c.items() if k!='baseline'} for c in cells]))
    studies = {}
    by_study=defaultdict(list)
    for cell in cells:
        if cell['source']=='R17':by_study[cell['model'],'mt50' if cell['suite']=='mt50' else 'libero'].append(cell)
    for key, selected in by_study.items():studies[key]=study_for(*key,selected)
    by_source=defaultdict(list)
    for cell in cells:
        if cell['source']!='R17':continue
        by_origin=defaultdict(list)
        for r in cell['baseline']:
            if r['task'] not in cell['empty_tasks']:by_origin[r['source']].append(r)
        for source, records in by_origin.items():by_source[source].append((cell,records))
    jobs=[]
    for i,(source,selected) in enumerate(sorted(by_source.items()),1):
        cell=selected[0][0];domain='mt50' if cell['suite']=='mt50' else 'libero'
        print('Prepare campaign '+source,flush=True)
        jobs.append(prepare_r17(source,selected,studies[cell['model'],domain],i))
    for model in ('pi05','groot'):
        jobs.append(prepare_r10(model,[c for c in cells if c['source']=='R10' and c['model']==model]))
    # Baseline outcomes have a separate immutable audit file; do not duplicate
    # 90,000 rows in every cell/campaign definition.
    for cell in cells:cell.pop('baseline')
    write(HERE/'CELLS.json',dict(method='frozen_fit_candidate_rows_only',refit=False,cells=cells))
    write(HERE/'CAMPAIGNS.json',dict(status='READY_FOR_COORDINATOR_STAGE',jobs=jobs))
    from .commands import generate
    generate()
    print(json.dumps(dict(cells=len(cells),campaigns=len(jobs),episodes=sum(j['episodes'] for j in jobs))))


if __name__=='__main__':
    main()
