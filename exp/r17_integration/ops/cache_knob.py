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
SOURCES = (study.HERE/'sizes/STUDY.json', study.HERE/'yaml/STUDY.json')
SIZES = (50, 100, 200, 300, 400, 500)
LEVELS = {'k50':'0.5', 'k40':'0.4', 'k30':'0.3', 'k20':'0.2', 'k10':'0.1', 'koff':None, 'pure':None}


def model_name(value):
    name = value.lower().replace('-', '').removesuffix('_libero')+'_libero'
    if name not in study.MODELS:
        raise argparse.ArgumentTypeError('model must be dp, fastwam or cosmos')
    return name


def build(output, *, gaps=GAPS, model=None, table=None):
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
    models = (model,) if test else study.MODELS
    if test and model not in study.MODELS:
        raise ValueError('TEST requires one model')
    output = Path(output).resolve()
    result = copy.deepcopy(sources[0][1])
    result.update(models={}, groups=[], library_sizes=list(SIZES), knobs=True,
                  eval_pool='A' if test else 'B', inits=50 if test else 30, artifacts_pending=[])
    documents = []
    missing = []
    for bundle in models:
        rows = result['models'][bundle] = []
        for size in SIZES:
            for suite in study.SUITES:
                source, original = base[bundle, suite, size]
                document = yaml.safe_load((source/original['file']).read_text())
                names = [name for name in document['recipe']['ordered_tasks'] if name.startswith(suite+'/')]
                if test:
                    cell = levels_table[suite][str(size)]
                    chosen = {name if name.startswith(suite+'/') else suite+'/'+name:level for name,level in cell.items()}
                    if len(chosen)!=10 or set(chosen)!=set(names) or any(level not in LEVELS for level in chosen.values()):
                        raise ValueError(f'{suite}/{size}: table must assign a valid level to each of its ten tasks')
                    levels = [level for level in LEVELS if level in chosen.values()]
                else:
                    levels = list(LEVELS)[:-1]
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
                        row['tasks'] = [name for name in names if chosen[name]==level]
                    else:
                        row['indices'] = list(range(30))
                    doc = copy.deepcopy(document)
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


def commands(manifest, model):
    """Keep cache_lane's resource flags; let the owner choose names/fleets/ports."""
    argv = cache_lane.launch_args(model, manifest, '${RUN_NAME}', size_count=len(SIZES))
    for flag, value in (('--workers','${WORKERS}'), ('--port','${PORT}'), ('--driver-port','${DRIVER_PORT}')):
        argv[argv.index(flag)+1] = value
    argv += ['--worker-counts','${WORKER_COUNTS}']
    def quote(value):
        return '"'+value+'"' if value.startswith('${') else shlex.quote(value)
    prepare = ' '.join(quote(value) for value in ['bash',str(study.ROOT/'exp/r17_integration/ops/formal_eval.sh'),*argv,'--prepare-only'])
    run = 'bash '+str(study.ROOT/'exp/r17_integration/ops/formal_eval.sh')+' --prepared-run "/data/mw_finetune/eval/${RUN_NAME}" --run'
    return prepare, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    dev = sub.add_parser('dev')
    dev.add_argument('--gaps', type=Path, required=True)
    dev.add_argument('--out', type=Path, required=True)
    test = sub.add_parser('test')
    test.add_argument('--model', type=model_name, required=True)
    test.add_argument('--table', type=Path, required=True)
    test.add_argument('--gaps', type=Path, default=GAPS)
    test.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.out, gaps=args.gaps, model=getattr(args,'model',None), table=getattr(args,'table',None))
    print('STUDY='+str(manifest))
    for model in study.read(manifest)['models']:
        profile = cache_lane.PROFILES[model]
        print(f'{model}: {6000 if args.mode=="test" else 21600} episodes; server={profile["node"]}; '
              f'suggested WORKERS={profile["workers"]} PORT={profile["port"]} DRIVER_PORT={profile["driver"]}; '
              'set RUN_NAME and WORKER_COUNTS (e.g. timan107=N or timan107=N,timan1=M,local=L)')
        print('\n'.join(commands(manifest, model)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
