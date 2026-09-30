"""Emit coordinator arm specs and prefit/reload their exact final-path kwargs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import STORE
from .common import CELLS, DENSE_CELLS, HERE, SCRATCH, VERSION, output_path, sha, write_json


METHOD = 'exp.offline_search.rounds.r07.c3_calls.methods:CallController'


def specs(phase, run='<RUN>', *, dense=False):
    rows = []
    rho = .18 if dense else .30
    for cell in DENSE_CELLS if dense else CELLS:
        model, suite, size = cell.split('_')
        for variant in ('CU', 'CT'):
            name = f'r7_{cell}_{variant}{int(rho * 100)}' + ('_profile' if phase == 'profile' else '')
            kwargs = dict(rho=rho, placement='uniform', tilt=variant == 'CT', cooldown_scope='stall',
                calibration_path=f'{run}/cal/{cell}/{variant}/calibration.json',
                stall_model_path=f'{run}/stall/{cell}', random_seed=26092903,
                randomization_key='R6-C-v2/' + cell)
            manifest = f'{run}/manifests/{model}_{suite}_bval20.json' if phase == 'profile' else f'{run}/manifests/eval500.json'
            args = ['--os-root', str(STORE), '--os-no-shadow-native', '--os-blind',
                    '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only',
                    '--os-fit-artifact', f'{run}/fits/{name}.pkl']
            if phase == 'profile':
                args.extend(['--os-log-inputs', '--os-log-r4'])
            rows.append(dict(name=name, model=model, suite=suite, mode='plugin', full_model=True,
                method=METHOD, kwargs=kwargs, manifest=manifest, cost_ledger=True,
                client_overrides=dict(replan_steps=5, resize_size=256 if model == 'groot' else 224),
                plugin_args=args))
    return rows


def emit_dense(out, run='<RUN>'):
    """Emit §9.3 additions without writing any sparse package file."""
    out = output_path(out)
    commands = []
    prefix = ("taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
              "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python")
    for phase in ('profile', 'eval500'):
        rows = specs(phase, run, dense=True)
        write_json(out / f'arms_dense_{phase}.json', rows)
        for row in rows:
            commands.append(f'{prefix} -m exp.offline_search.rounds.r07.c3_calls.package prefit '
                            f'--spec {out}/arms_dense_{phase}.json --name {row["name"]} '
                            '--out /tmp/r7_C3/prefits_dense')
    output_path(out / 'prefit_commands_dense.sh').write_text('\n'.join(commands) + '\n')
    write_json(out / 'package_manifest_dense.json', dict(version=VERSION,
        target=.18, cells=list(DENSE_CELLS), variants=['CU', 'CT'],
        profile_arms=8, eval500_arms=8, profile_episodes_per_arm=20,
        eval500_episodes_per_arm=500, stage_source=str(Path('/tmp/r7_C1/stages')),
        calibration_root='/tmp/r7_C3/cal; copy each complete dense cell directory to <RUN>/cal',
        stall_source='/home/weiland/trace_runs/os_closed_loop/r06_c_cal/stall; canonicalize sp to spatial',
        planned_arms=[r['name'] for phase in ('profile', 'eval500') for r in specs(phase, run, dense=True)]))
    print(json.dumps(dict(out=str(out), dense_rows=16)), flush=True)


def emit(out, run='<RUN>'):
    out = output_path(out)
    for phase in ('profile', 'eval500'):
        write_json(out / f'arms_{phase}.json', specs(phase, run))
    rows = specs('profile', run) + specs('eval500', run)
    write_json(out / 'package_manifest.json', dict(version=VERSION, variants=['CU', 'CT'],
        target=.30, cells=list(CELLS), profile_arms=8, eval500_arms=8,
        profile_episodes_per_arm=20, eval500_episodes_per_arm=500,
        profiling_manifest_owner='C4; bind the verified non-test 20-state manifests before launch',
        calibration_root='/tmp/r7_C3/cal; copy each complete cell directory to <RUN>/cal',
        stall_source='/home/weiland/trace_runs/os_closed_loop/r06_c_cal/stall; canonicalize sp to spatial',
        planned_arms=[r['name'] for r in rows]))
    commands = []
    prefix = ("taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
              "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python")
    for phase in ('profile', 'eval500'):
        for row in specs(phase, run):
            commands.append(f'{prefix} -m exp.offline_search.rounds.r07.c3_calls.package prefit '
                            f'--spec {out}/arms_{phase}.json --name {row["name"]} --out /tmp/r7_C3/prefits')
    output_path(out / 'prefit_commands.sh').write_text('\n'.join(commands) + '\n')
    print(json.dumps(dict(out=str(out), rows=len(rows))), flush=True)


def prefit(spec_path, name, out):
    row = next(r for r in json.loads(Path(spec_path).read_text()) if r['name'] == name)
    if '<RUN>' in json.dumps(row):
        raise ValueError('render final RUN paths before prefit')
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name + '.pkl')
    if path.exists():
        raise FileExistsError(path)
    cls, source = load_method_class(row['method'])
    method = cls(**row['kwargs'])
    cell = f'{row["model"]}_{row["suite"]}_cache'
    ctx = api.Context(root=STORE, cell=cell, seed=0, scratch=out / 'scratch' / name)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    library = store.LibraryView(STORE, store.lib_key(cell), 'current')
    start = time.monotonic()
    method.fit(library, ctx)
    api.check_method_attrs(method)
    method.prof = api.NULL_PROFILER
    blob = dict(method=method, registered=ctx.registered, spec=row['method'], kwargs=row['kwargs'],
                cell=cell, fit_s=time.monotonic() - start,
                provenance=dict(builder=__file__, method_source=source, source_sha256=sha(source)))
    with path.open('xb') as f:
        pickle.dump(blob, f, protocol=4)
    with path.open('rb') as f:
        loaded = pickle.load(f)
    assert loaded['kwargs'] == row['kwargs'] and loaded['spec'] == row['method']
    assert loaded['method'].lambda_ == method.lambda_
    note = dict(name=name, artifact=str(path), sha256=sha(path), bytes=path.stat().st_size,
                fit_info=method.fit_info, reload='PASS', kwargs=row['kwargs'])
    write_json(path.with_suffix('.json'), note)
    print(json.dumps(dict(name=name, bytes=note['bytes'], reload='PASS')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    emit_parser = sub.add_parser('emit')
    emit_parser.add_argument('--out', type=Path, default=HERE)
    emit_parser.add_argument('--run-root', default='<RUN>')
    emit_parser.add_argument('--dense', action='store_true')
    fit_parser = sub.add_parser('prefit')
    fit_parser.add_argument('--spec', type=Path, required=True)
    fit_parser.add_argument('--name', required=True)
    fit_parser.add_argument('--out', type=Path, default=SCRATCH / 'prefits')
    args = parser.parse_args()
    if args.command == 'emit':
        (emit_dense if args.dense else emit)(args.out, args.run_root)
    else:
        prefit(args.spec, args.name, args.out)


if __name__ == '__main__':
    main()
