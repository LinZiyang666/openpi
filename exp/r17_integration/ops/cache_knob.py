"""Build per-task IR knob studies on CPU; print preparation/launch commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import shlex

import yaml

from . import cache_lane, cache_study as study

GAPS = Path('/home/weiland/projects/openpi_ext/scratch/r17_ops/knob/gaps.jsonl')
REDUCED = {
    'dp_libero': (10, 'hs5dp_ddim10s0'),
    'fastwam_libero': (1, 'hs6fw_k1'),
    'cosmos_libero': (2, 'hs2_cosmos_h100_k2incl_c16_t107_20261010_115120'),
    'pi05_libero': (1, 'pl_k1a'),
    'groot_libero': (1, 'gl_k1a'),
}
VLA_PROFILES = {
    'pi05_libero': dict(node='local', replicas=5, startup=3, workers=24, budget=47, short='pl'),
    'groot_libero': dict(node='h100', replicas=2, startup=2, workers=32, budget=48, short='gl'),
}
SOURCES = (study.HERE/'sizes/STUDY.json', study.HERE/'yaml/STUDY.json')
SIZES = (50, 100, 200, 300, 400, 500)
MW_SIZES = (250, 500, 1000, 1500, 2000, 2500)
MW_SOURCES = {m: Path("/data/r17_library/studies_mt50")/s/"STUDY.json"
              for m,s in zip(study.MW_MODELS, ("dp_k10", "fw_k3", "pi_k2", "gr_k2"))}
MW_STEPS = {"dp_mw": (100, 10), "fastwam_mw": (10, 3), "pi05_mw": (10, 2), "groot_mw": (4, 2)}
MW_PROFILES = {
    "dp_mw": dict(node="h100", replicas=4, startup=3, batch=8, budget=24, workers="timan107=30,timan1=20", port=23180, driver=23189),
    "fastwam_mw": dict(node="h100", replicas=3, startup=3, batch=16, budget=76, workers="timan107=30,timan1=20", port=23170, driver=23179),
    "pi05_mw": dict(node="local", replicas=5, startup=3, batch=1, budget=47, workers="timan107=24,timan1=20", port=23160, driver=23169),
    # owner 10-20: 4-5 GR00T replicas are enough (R11 ran the H100 at 5% GPU); never 11.
    "groot_mw": dict(node="h100", replicas=5, startup=4, batch=1, budget=36, workers="timan107=40,timan1=20", port=23160, driver=23169),
}
LEVELS = {'k50':'0.5', 'k40':'0.4', 'k30':'0.3', 'k20':'0.2', 'k10':'0.1', 'koff':None, 'pure':None}


def model_name(value):
    value = value.lower().replace('-', '')
    name = value if value.endswith('_mw') else value.removesuffix('_libero')+'_libero'
    if name not in study.MODELS:
        raise argparse.ArgumentTypeError('model must be a LIBERO or MT50 cache-study selector')
    return name


def mw_sampler(bundle, miss_steps):
    """Native CPU sampler metadata from the supplied MT50 stage profiles."""
    k = MW_STEPS[bundle][miss_steps == 'reduced']
    path = study.ROOT/'exp/r17_integration/profile/results_mt50'/(bundle+'_mt50.json')
    return study.read(path)['cells'][str(k)]['sampler']


def build(output, *, gaps=GAPS, model=None, table=None, miss_steps='full', partial=False, sources=None, sizes=None, levels=None, mode=None):
    """Reuse full-system YAMLs: LIBERO B30, MT50 direct seed-7 A50 descent."""
    sources = [(Path(path), study.validate_manifest(path)) for path in (sources or ([MW_SOURCES[model]] if model in MW_SOURCES else SOURCES))]
    sizes = tuple(sizes or (MW_SIZES if model in study.MW_MODELS else SIZES))
    base = {}
    for path, manifest in sources:
        for bundle, rows in manifest['models'].items():
            for row in rows:
                if row['group'] == study.full_group(bundle):
                    base[bundle, row['suite'], study.library_size(row)] = path.parent, row
    gap_rows = {}
    if Path(gaps).exists():
        for line in Path(gaps).read_text().splitlines():
            if line.strip() and line.strip()!='DONE':
                entry = json.loads(line)
                gap_rows[model_name(entry['model']), entry['suite'], int(entry['size']), entry.get('miss_steps', 'full')] = entry['levels']
    mt50 = model in study.MW_MODELS
    test = mt50 or ((mode == 'test') if mode else table is not None)
    levels_table = study.read(table) if table else None
    if table:
        label = {'pi05_libero':'π0.5', 'groot_libero':'GR00T', 'dp_libero':'DP',
                 'fastwam_libero':'Fast-WAM', 'cosmos_libero':'Cosmos',
                 'dp_mw':'DP', 'fastwam_mw':'Fast-WAM', 'pi05_mw':'π0.5', 'groot_mw':'GR00T'}.get(model)
        if label in levels_table:levels_table = levels_table[label]
    # Existing knob sources cover their own model cohort; new cache-study
    # selectors need their libraries before they can join a knob generation.
    models = (model,) if model else tuple(dict.fromkeys(bundle for bundle,_,_ in base))
    if table and model not in study.MODELS:
        raise ValueError('TEST requires one model')
    output = Path(output).resolve()
    result = copy.deepcopy(sources[0][1])
    result.update(models={}, groups=[], library_sizes=list(sizes), knobs=True, miss_steps=miss_steps,
                  eval_pool='A' if test else 'B', inits=50 if test else 30, artifacts_pending=[])
    documents = []
    missing = []
    for bundle in models:
        sampler = None
        if bundle in study.MW_MODELS:
            sampler = mw_sampler(bundle, miss_steps)
        elif bundle in VLA_PROFILES:
            if bundle == 'pi05_libero':
                from exp.r17_integration.models.pi05.adapter import sampler_for_steps
            else:
                from exp.r17_integration.models.groot.manifest import sampler_for_steps
            # Native schedule, independent of a future GPU receipt.
            sampler = asdict(sampler_for_steps(1))
        elif miss_steps == 'reduced':
            k, run = REDUCED[bundle]
            sampler = study.read(Path('/data/mw_finetune/eval')/run/'assets/execution_sampler.json')
            if sampler['num_steps'] != k:
                raise ValueError(f'{run}: expected reduced K{k}')
        rows = result['models'][bundle] = []
        for size in sizes:
            for suite in study.suites_for(bundle):
                source, original = base[bundle, suite, size]
                document = yaml.safe_load((source/original['file']).read_text())
                names = [name for name in document['recipe']['ordered_tasks'] if suite == 'mt50' or name.startswith(suite+'/')]
                if table:
                    # partial: retest rounds list only the raised tasks; absent tasks/cells are skipped.
                    cell = levels_table.get(suite, {}).get(str(size), {}) if partial else levels_table[suite][str(size)]
                    chosen = {name if suite == 'mt50' or name.startswith(suite+'/') else suite+'/'+name:level for name,level in cell.items()}
                    if partial and not chosen:
                        continue
                    if (not partial and (set(chosen)!=set(names)) or not set(chosen)<=set(names)
                            or any(level not in LEVELS for level in chosen.values())):
                        raise ValueError(f'{suite}/{size}: table must assign a valid level to each of its tasks')
                    selected_levels = [level for level in LEVELS if level in chosen.values()]
                else:
                    selected_levels = list(levels or list(LEVELS)[:-1])
                    if levels is None and sampler is not None and size == max(sizes) and bundle not in study.MW_MODELS:
                        selected_levels.append('pure')
                for level in selected_levels:
                    gap = None
                    if level=='pure':
                        gap = 0.0
                    elif LEVELS[level] is not None:
                        try:
                            gap_key = (bundle, suite, size, miss_steps)
                            entry = gap_rows.get(gap_key) if bundle in study.MW_MODELS else gap_rows.get(gap_key, gap_rows.get((bundle,suite,size,'full')))
                            gap = entry[LEVELS[level]]['gap']
                        except (KeyError, TypeError):
                            missing.append(f'{bundle}/{suite}/{size}/{level}')
                            continue
                        if isinstance(gap,bool) or not isinstance(gap,(int,float)) or not math.isfinite(gap) or gap<0:
                            raise ValueError(f'invalid gap for {bundle}/{suite}/{size}/{level}')
                    row = copy.deepcopy(original)
                    row.update(group=original['group']+'_'+level, library_size=size,
                               periodic_gap=gap, serve_loeo=not test)
                    for key in ('library','fit','stall'):
                        path = Path(row[key])
                        row[key] = str((study.ROOT/path).resolve())
                    row['file'] = bundle+'/'+study.suite_label(suite)+'/n'+str(size)+'_'+row['group']+'.yaml'
                    if table:
                        row['tasks'] = [name for name in names if chosen.get(name)==level]
                    if not test:
                        row['indices'] = list(range(30))
                    doc = copy.deepcopy(document)
                    if sampler is not None:
                        doc['adapter_binding'].update(num_steps=sampler['num_steps'], sampler_identity=study.digest(sampler))
                        row["num_steps"] = sampler["num_steps"]
                    doc['recipe'].update(store_path=str(study.view_path(row)), fitted_path=row['fit'],
                                         stall_path=row['stall'], periodic_gap=gap, serve_loeo=not test)
                    rows.append(row)
                    documents.append((row,doc))
                    if row['group'] not in result['groups']:
                        result['groups'].append(row['group'])
    if missing:
        raise ValueError(f'{gaps}: missing {len(missing)} gaps; wait for the coordinator to finish: '+', '.join(missing))
    output.mkdir(parents=True, exist_ok=False)
    for row, document in documents:
        path = output/row['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        row['sha256'] = study.file_sha(path)
    result.pop('digest', None)
    result['digest'] = study.digest(result)
    study.atomic_json(output/'STUDY.json', result)
    study.validate_manifest(output/'STUDY.json')
    return output/'STUDY.json'


def calibrate(output, *, sources, model=None, miss_steps='full', sizes=None):
    """The existing R17 B-only proxy calibration (vision=0, miss=1)."""
    from types import SimpleNamespace
    from exp.r17_integration.common.ir import calibrate_periodic, library_schedule
    from exp.r17_integration.search.fit import load_fitted_recipe
    from exp.r17_integration.search.teacher_library import RowIndexStore
    rows = {}
    for source in map(Path, sources):
        for bundle, entries in study.validate_manifest(source)['models'].items():
            if model and bundle != model:continue
            for row in entries:
                if row['group'] == study.full_group(bundle) and (not sizes or study.library_size(row) in sizes):
                    rows[bundle,row['suite'],study.library_size(row)] = source.parent,row
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for (bundle,suite,size),(root,row) in sorted(rows.items()):
            doc = yaml.safe_load((root/row['file']).read_text())
            r,binding = doc['recipe'],doc['adapter_binding']
            if bundle in study.MW_MODELS:
                sampler = mw_sampler(bundle, miss_steps)
                binding.update(num_steps=sampler['num_steps'], sampler_identity=study.digest(sampler))
            parent = study.read(Path(row['library'])/'READY.json')
            fit = load_fitted_recipe(r['fitted_path'], expected_binding=parent['binding'])
            view = RowIndexStore(study.view_path(row))
            config = SimpleNamespace(adapter_binding=binding,
                recipe=SimpleNamespace(ordered_tasks=tuple(r['ordered_tasks']), guard=False))
            replay = library_schedule(fit,view,config)
            entry = dict(model=bundle,suite=suite,size=size,commit=fit.commit,miss_steps=miss_steps,
                num_steps=binding['num_steps'],
                calibration_scope='selected B library LOEO; model-call proxy, vision=0/miss=1',levels={})
            for target in (.5,.4,.3,.2,.1):
                solved = calibrate_periodic(target,replay,dict(vision=0,miss=1))
                entry['levels'][str(target)] = dict(gap=solved['setting'], pred=solved['pred_IR_rounded'], status=solved['solve_status'])
            stream.write(json.dumps(entry)+'\n');stream.flush()
    return output


def commands(manifest, model, *, miss_steps='full', mode='dev', run_name=None):
    """Print lane placeholders or the owner's knob topology; never launch."""
    if model in study.MW_MODELS:
        return mw_commands(manifest, model, miss_steps=miss_steps, mode=mode, run_name=run_name)
    if model in VLA_PROFILES:
        p = VLA_PROFILES[model]
        name = run_name or p['short']+('_kd' if mode == 'dev' else '_kt')
        argv = ['--model',model,'--cache-study',str(Path(manifest).resolve()),'--official-50',
            '--suites',*study.SUITES,'--server-node',p['node'],'--gpu','cuda:0',
            '--replicas',str(p['replicas']),'--replica-startup-width',str(p['startup']),
            '--max-batch','1','--mps','--memory-budget-gib',str(p['budget']),
            '--driver-mode','group-commit','--worker-node','timan107','--extra-workers','0',
            '--workers',str(p['workers']),'--worker-memory-mib','6144','--worker-vram-mib','1024',
            '--worker-egl-devices','0,1,2,3,4,5,6,7','--worker-max-episodes','4',
            '--episode-seconds','1800','--seconds','518400','--port','23160','--driver-port','23169','--run-name',name]
        prefix = ['env',"CUDA_VISIBLE_DEVICES=",'JAX_PLATFORMS=cpu',
            'R17_INHERIT_WORKER_DEPLOYMENT=/data/mw_finetune/eval/cosmossizes145_r8c/WORKER_DEPLOYMENT.json']
        if model == 'pi05_libero':prefix = ['taskset','-c','22-43,66-87',*prefix]
        python = str(study.ROOT/'.venv/bin/python')
        entry = [*prefix,python,'-u','-m','exp.r17_integration.ops.formal_eval']
        return (shlex.join([*entry,*argv,'--prepare-only']),
                shlex.join([*entry,'--run','--prepared-run','/data/mw_finetune/eval/'+name]))
    argv = cache_lane.launch_args(model, manifest, '${RUN_NAME}', size_count=len(SIZES))
    if miss_steps == 'reduced':
        # Same topology as scratch/r17_ops/knob/prepare_dev.sh; fresh names
        # and ports leave the full-step campaigns alone. These are only text.
        # Drivers sit in ziyanglin.com's forwarded 23100-23199 block (workers connect directly).
        replicas, workers, fleet, ram, vram, port, driver = {
            'dp_libero': (4, 48, 'local=48', 2048, 717, 23400, 23110),
            'fastwam_libero': (2, 85, 'timan107=45,timan1=40', 2048, 768, 23420, 23111),
            'cosmos_libero': (8, 85, 'timan107=45,timan1=40', 3072, 768, 23440, 23112),
        }[model]
        name = 'knob'+mode+{'dp_libero':'dp', 'fastwam_libero':'fw', 'cosmos_libero':'cos'}[model]+'_reduced'
        values = (('--replicas',str(replicas)), ('--workers',str(workers)),
                  ('--worker-node','local' if model == 'dp_libero' else 'timan107'),
                  ('--worker-memory-mib',str(ram)), ('--worker-vram-mib',str(vram)),
                  ('--memory-budget-gib','76' if model == 'cosmos_libero' else argv[argv.index('--memory-budget-gib')+1]),
                  ('--port',str(port)), ('--driver-port',str(driver)), ('--run-name',name))
        argv += ['--worker-counts',fleet]
        if model == 'cosmos_libero':argv.append('--mps')
    else:
        values = (('--workers','${WORKERS}'), ('--port','${PORT}'), ('--driver-port','${DRIVER_PORT}'))
        argv += ['--worker-counts','${WORKER_COUNTS}']
    for flag, value in values:
        argv[argv.index(flag)+1] = value
    def quote(value):
        return '"'+value+'"' if value.startswith('${') else shlex.quote(value)
    prepare = ' '.join(quote(value) for value in ['bash',str(study.ROOT/'exp/r17_integration/ops/formal_eval.sh'),*argv,'--prepare-only'])
    run = 'bash '+str(study.ROOT/'exp/r17_integration/ops/formal_eval.sh')+' --prepared-run "/data/mw_finetune/eval/${RUN_NAME}" --run'
    if miss_steps == 'reduced':
        prepare = ('R17_INHERIT_WORKER_DEPLOYMENT=/data/mw_finetune/eval/cosmossizes145_r8c/WORKER_DEPLOYMENT.json '+prepare)
        run = 'bash '+str(study.ROOT/'exp/r17_integration/ops/formal_eval.sh')+' --prepared-run /data/mw_finetune/eval/'+name+' --run'
        if model == 'cosmos_libero':
            root = '/data/mw_finetune/eval/'+name
            nvme = '/tmp/r17_driver/'+name+'/driver'
            run = (f'mkdir -p {nvme}\n'
                   f'[ -e {root}/driver ] && [ ! -L {root}/driver ] && mv {root}/driver {root}/driver.prepared_empty\n'
                   f'[ -L {root}/driver ] || ln -s {nvme} {root}/driver\n'+run)
    return prepare, run


def mw_commands(manifest, model, *, miss_steps='full', mode='dev', run_name=None, reference=None):
    """Exact MT50 prepare/run text; shared workers require serialized launches."""
    p = MW_PROFILES[model]
    name = run_name or model.removesuffix('_mw')+'mk_'+mode+'_'+miss_steps
    size_count = len(study.read(manifest).get('library_sizes', MW_SIZES)) if manifest and Path(manifest).exists() else len(MW_SIZES)
    seconds = 172800 if reference else 86400*size_count
    argv = ['--model', model, '--server-node', p['node'], '--gpu', 'cuda:0',
        '--replicas', str(p['replicas']), '--replica-startup-width', str(p['startup']),
        '--max-batch', str(p['batch']), '--batch-wait-ms', '5', '--mps',
        '--memory-budget-gib', str(p['budget']), '--driver-mode', 'group-commit',
        '--worker-node', 'timan107', '--extra-workers', '0', '--worker-counts', p['workers'],
        '--worker-memory-mib', '2048', '--worker-vram-mib', '1024',
        '--worker-egl-devices', '0,1,2,3,4,5,6,7', '--worker-max-episodes', '0',
        '--worker-restart-limit', '2', '--retries', '2', '--seconds', str(seconds),
        '--episode-seconds', '1800', '--port', str(p['port']), '--driver-port', str(p['driver']), '--run-name', name]
    if reference:
        argv += ['--eval-pool', 'A', '--init-count', '50', '--denoising-steps', str(MW_STEPS[model][miss_steps == 'reduced'])]
    else:
        argv += ['--cache-study', str(Path(manifest).resolve())]
    entry = ['env', 'CUDA_VISIBLE_DEVICES=', 'JAX_PLATFORMS=cpu', str(study.ROOT/'.venv/bin/python'),
             '-u', '-m', 'exp.r17_integration.ops.formal_eval']
    if model == 'pi05_mw':
        entry = ['taskset', '-c', '22-43,66-87', *entry]
    return (shlex.join([*entry, *argv, '--prepare-only']),
            shlex.join([*entry, '--run', '--prepared-run', '/data/mw_finetune/eval/'+name]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    calibration = sub.add_parser('calibrate')
    calibration.add_argument('--source', type=Path, action='append', required=True)
    calibration.add_argument('--model', type=model_name)
    calibration.add_argument('--out', type=Path, required=True)
    calibration.add_argument('--miss-steps', choices=('full','reduced'), default='full')
    calibration.add_argument('--sizes', type=int, nargs='+')
    dev = sub.add_parser('dev', help='LIBERO DEV or MT50 direct seed-7 descent batch')
    test = sub.add_parser('test')
    for p in (dev, test):
        p.add_argument('--model', type=model_name, required=p is test)
        p.add_argument('--table', type=Path, help='Per-task levels; omit for a complete MT50 descent batch')
        p.add_argument('--gaps', type=Path, required=p is dev, default=GAPS)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--miss-steps', choices=('full','reduced'), default='full')
        p.add_argument('--partial', action='store_true', help='Only tasks/cells listed in the table')
        p.add_argument('--source', type=Path, action='append')
        p.add_argument('--run-name')
        p.add_argument('--sizes', type=int, nargs='+')
    dev.add_argument('--levels', choices=tuple(LEVELS), nargs='+', help='Optional levels; MT50 defaults to all six seed-7 levels')
    command = sub.add_parser('commands', help='Print commands only; never prepare or launch')
    command.add_argument('--model', type=model_name, required=True)
    command.add_argument('--study', type=Path)
    command.add_argument('--reference', choices=('test',), help='Same-step pure seed-7 baseline; MT50 has no DEV set')
    command.add_argument('--mode', dest='command_mode', choices=('dev','test'), default='dev')
    command.add_argument('--miss-steps', choices=('full','reduced'), default='full')
    command.add_argument('--run-name')
    args = parser.parse_args(argv)
    if args.mode == 'commands':
        if args.reference:
            pair = mw_commands(None, args.model, run_name=args.run_name, reference=args.reference,
                               mode=args.reference, miss_steps=args.miss_steps)
        else:
            pair = commands(args.study, args.model, miss_steps=args.miss_steps, mode=args.command_mode, run_name=args.run_name)
        print('\n'.join(pair)); return 0
    if args.mode == 'calibrate':
        print('GAPS='+str(calibrate(args.out, sources=args.source, model=args.model, miss_steps=args.miss_steps, sizes=args.sizes))); return 0
    manifest = build(args.out, gaps=args.gaps, model=args.model, table=args.table,
                     miss_steps=args.miss_steps, partial=args.partial, sources=args.source,
                     sizes=args.sizes, levels=getattr(args,'levels',None), mode=args.mode)
    print('STUDY='+str(manifest))
    for model, rows in study.read(manifest)['models'].items():
        count = sum(len(r.get('tasks',range(50 if model in study.MW_MODELS else 10)))*len(r.get('indices',range(50))) for r in rows)
        print(f'{model}: {count} episodes; {args.miss_steps} model calls')
        print('\n'.join(commands(manifest, model, miss_steps=args.miss_steps, mode=args.mode, run_name=args.run_name)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
