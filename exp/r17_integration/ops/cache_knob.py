"""Build per-task IR knob studies on CPU; print preparation/launch commands."""
from __future__ import annotations

import argparse
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
}
SOURCES = (study.HERE/'sizes/STUDY.json', study.HERE/'yaml/STUDY.json')
SIZES = (50, 100, 200, 300, 400, 500)
LEVELS = {'k50':'0.5', 'k40':'0.4', 'k30':'0.3', 'k20':'0.2', 'k10':'0.1', 'koff':None, 'pure':None}


def model_name(value):
    name = value.lower().replace('-', '').removesuffix('_libero')+'_libero'
    if name not in study.MODELS:
        raise argparse.ArgumentTypeError('model must be dp, fastwam or cosmos')
    return name


def build(output, *, gaps=GAPS, model=None, table=None, miss_steps='full', partial=False):
    """Copy complete-group YAMLs and artifacts; never fit or launch anything."""
    sources = [(path, study.validate_manifest(path)) for path in SOURCES]
    base = {}
    for path, manifest in sources:
        for bundle, rows in manifest['models'].items():
            for row in rows:
                if row['group'] == ('crg' if bundle=='cosmos_libero' else 'chrg'):
                    base[bundle, row['suite'], study.library_size(row)] = path.parent, row
    gap_rows = {}
    if Path(gaps).exists():
        for line in Path(gaps).read_text().splitlines():
            if line.strip() and line.strip()!='DONE':
                entry = json.loads(line)
                gap_rows[model_name(entry['model']), entry['suite'], int(entry['size'])] = entry['levels']
    test = table is not None
    levels_table = study.read(table) if test else None
    # Existing knob sources cover their own model cohort; new cache-study
    # selectors need their libraries before they can join a knob generation.
    models = (model,) if test else tuple(dict.fromkeys(bundle for bundle,_,_ in base))
    if test and model not in study.MODELS:
        raise ValueError('TEST requires one model')
    output = Path(output).resolve()
    result = copy.deepcopy(sources[0][1])
    result.update(models={}, groups=[], library_sizes=list(SIZES), knobs=True,
                  eval_pool='A' if test else 'B', inits=50 if test else 30, artifacts_pending=[])
    documents = []
    missing = []
    for bundle in models:
        sampler = None
        if miss_steps == 'reduced':
            k, run = REDUCED[bundle]
            sampler = study.read(Path('/data/mw_finetune/eval')/run/'assets/execution_sampler.json')
            if sampler['num_steps'] != k:
                raise ValueError(f'{run}: expected reduced K{k}')
        rows = result['models'][bundle] = []
        for size in SIZES:
            for suite in study.SUITES:
                source, original = base[bundle, suite, size]
                document = yaml.safe_load((source/original['file']).read_text())
                names = [name for name in document['recipe']['ordered_tasks'] if name.startswith(suite+'/')]
                if test:
                    # partial: retest rounds list only the raised tasks; absent tasks/cells are skipped.
                    cell = levels_table.get(suite, {}).get(str(size), {}) if partial else levels_table[suite][str(size)]
                    chosen = {name if name.startswith(suite+'/') else suite+'/'+name:level for name,level in cell.items()}
                    if partial and not chosen:
                        continue
                    if (not partial and (len(chosen)!=10 or set(chosen)!=set(names)) or not set(chosen)<=set(names)
                            or any(level not in LEVELS for level in chosen.values())):
                        raise ValueError(f'{suite}/{size}: table must assign a valid level to each of its ten tasks')
                    levels = [level for level in LEVELS if level in chosen.values()]
                else:
                    levels = list(LEVELS)[:-1]
                    if sampler is not None and size == 500:
                        levels.append('pure')
                for level in levels:
                    gap = None
                    if level=='pure':
                        gap = 0.0
                    elif LEVELS[level] is not None:
                        try:
                            gap = gap_rows[bundle, suite, size][LEVELS[level]]['gap']
                        except KeyError:
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
                    row['file'] = bundle+'/'+('sp' if suite=='libero_spatial' else 'l10')+'/n'+str(size)+'_'+row['group']+'.yaml'
                    if test:
                        row['tasks'] = [name for name in names if chosen.get(name)==level]
                    else:
                        row['indices'] = list(range(30))
                    doc = copy.deepcopy(document)
                    if sampler is not None:
                        doc['adapter_binding'].update(num_steps=sampler['num_steps'], sampler_identity=study.digest(sampler))
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


def commands(manifest, model, *, miss_steps='full', mode='dev'):
    """Print lane placeholders or the owner's knob topology; never launch."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    dev = sub.add_parser('dev')
    dev.add_argument('--gaps', type=Path, required=True)
    dev.add_argument('--out', type=Path, required=True)
    dev.add_argument('--miss-steps', choices=('full','reduced'), default='full')
    test = sub.add_parser('test')
    test.add_argument('--model', type=model_name, required=True)
    test.add_argument('--table', type=Path, required=True)
    test.add_argument('--gaps', type=Path, default=GAPS)
    test.add_argument('--out', type=Path, required=True)
    test.add_argument('--miss-steps', choices=('full','reduced'), default='full')
    test.add_argument('--partial', action='store_true', help='retest: only the tasks listed in the table')
    args = parser.parse_args(argv)
    manifest = build(args.out, gaps=args.gaps, model=getattr(args,'model',None), table=getattr(args,'table',None),
                     miss_steps=args.miss_steps, partial=getattr(args,'partial',False))
    print('STUDY='+str(manifest))
    for model in study.read(manifest)['models']:
        profile = cache_lane.PROFILES[model]
        count = 6000 if args.mode == 'test' else 22200 if args.miss_steps == 'reduced' else 21600
        if args.miss_steps == 'reduced':
            print(f'{model}: {count} episodes; reduced MISS K={REDUCED[model][0]}')
        else:
            print(f'{model}: {count} episodes; server={profile["node"]}; '
                  f'suggested WORKERS={profile["workers"]} PORT={profile["port"]} DRIVER_PORT={profile["driver"]}; '
                  'set RUN_NAME and WORKER_COUNTS (e.g. timan107=N or timan107=N,timan1=M,local=L)')
        print('\n'.join(commands(manifest, model, miss_steps=args.miss_steps, mode=args.mode)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
