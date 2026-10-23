"""Build paired B descent roots; emit and install separate frozen test-A roots."""
from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import io
import json
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from .recipe import TaskDescentKnob, LEVELS, validate_levels

HERE = Path(__file__).resolve().parent
R15 = HERE.parent
ROUNDS = R15.parent
R14 = ROUNDS / "r14"
REPO = HERE.parents[4]
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
from .fits import CELLS
SPEC = 'exp.offline_search.rounds.r15.descent.recipe:TaskDescentKnob'
SEED = 2026101401
DEV_ORDER = ('ir45', 'ir40', 'ir32', 'ir25', 'off')


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write_json(path, data):
    atomic(path, (json.dumps(data, indent=2, allow_nan=False) + '\n').encode())


def frozen(cell, level, *, test=False):
    if cell not in CELLS or level not in LEVELS:
        raise ValueError((cell, level))
    rec = json.loads((HERE / 'fits.json').read_text())[cell][level]
    if test and rec.get('serving_wrapper'):
        rec = dict(rec, path=rec['source_path'], sha256=rec['source_sha256'])
        blob = pickle.loads(Path(rec['path']).read_bytes())
        rec.update(spec=blob['spec'], kwargs=blob['kwargs'])
    return rec


def level_cap(cell, level):
    if level not in LEVELS:
        raise ValueError(level)
    if level in ('off', 'c10'):
        return 0.
    rec = frozen(cell, level)
    return float(pickle.loads(Path(rec['path']).read_bytes())['method'].knob_settings['setting'])


def descent_blob(cell, levels):
    levels = validate_levels(levels)
    caps = [level_cap(cell, levels[str(t)]) for t in range(10)]
    source = frozen(cell, 'off', test=True)
    blob = pickle.loads(Path(source['path']).read_bytes())
    method = blob['method']
    method.__class__ = TaskDescentKnob
    method.knob_method = 'task_periodic'
    method.target = None
    for name in ('scope', 'metric', 'head', 'continuity', 'guard'):
        setattr(method, 'r14_' + name, 'task')
    method.r14_fit = {}; method.r14_diagnostics = None
    method._accepted_prefix = ((), ()); method._fallback_counts = {}
    method.leave_own_episode_out = False
    method.set_table(levels, caps)
    method.r14_fit = dict(cell=cell, variant='task_descent', source_sha256=source['sha256'],
                         levels=levels, K=caps, no_refitting=True, lower_layers_byte_preserved=True)
    blob['r14_manifest'] = method.r14_fit
    kwargs = {k: blob['kwargs'][k] for k in ('library', 'episode_subset')}
    kwargs.update(task_levels=levels, task_caps=caps)
    blob.update(spec=SPEC, kwargs=kwargs, descent_manifest=method.r14_fit)
    return blob


def arm_name(cell, level, *, test=False):
    prefix = 'r15dt' if test else 'r15d'
    suffix = 'task_periodic_' + level[5:] if level.startswith('task_ir') else (
        'periodic_' + level if level.startswith('ir') else 'pure_c10' if level == 'c10' else level)
    return f'{prefix}_{cell}_{suffix}'


def arm_spec(cell, name, fit, manifest):
    model, suite, _ = cell.split('_')
    return dict(name=name, model=model, suite=suite, mode='plugin', full_model=True,
                cost_ledger=True, client_overrides=dict(replan_steps=5, **({'resize_size': 256} if model == 'groot' else {})),
                manifest=str(manifest), method=fit['spec'], kwargs=fit['kwargs'],
                plugin_args=['--os-root', str(devset.STORE), '--os-no-shadow-native', '--os-tokens', 'off',
                             '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
                             '--os-judge', 'guard_only', '--os-fit-artifact', fit['path'],
                             '--os-rollout-seed', str(SEED), '--os-r14-audit'])


def seal(root, *, final_root, pool, cell):
    rows = json.loads((root / 'arms.json').read_text())
    fits = {r['plugin_args'][r['plugin_args'].index('--os-fit-artifact') + 1] for r in rows}
    record = dict(root=str(root), final_root=str(final_root), cell=cell, arms=[r['arm'] for r in rows],
                  arm_count=len(rows), episodes_per_arm=300 if pool == 'B' else 500,
                  init_pool=pool, staged=root != final_root, recording=False, launches=0,
                  intended_host='local' if cell.startswith('groot') else 'h100',
                  definitions_sha256={str(p.relative_to(root)): devset.sha(p) for p in sorted(root.rglob('*'))
                                      if p.is_file() and p.name != 'ROOT.json'},
                  fit_sha256={p: devset.sha(p) for p in sorted(fits)})
    write_json(root / 'ROOT.json', record)
    return record


def emit(root, cell, specs, levels, *, pool, final_root):
    write_json(root / 'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root / 'arms_in.json')
    rows = json.loads((root / 'arms.json').read_text())
    for row, level in zip(rows, levels, strict=True):
        row.update(dev=pool == 'B', init_pool=pool, r10_size=int(cell.split("_")[-1]), r14_descent_level=level, r15_leave_own_episode_out=bool(pool == 'B' and cell.endswith('_500')),
                   intended_host='local' if cell.startswith('groot') else 'h100',
                   recording=False, rollout_replicate_seed=SEED, knob_coin_seed=0,
                   episodes=300 if pool == 'B' else 500, fleet_exclusive=True)
        if 'server_host' in row:
            raise ValueError('root must not bind a fleet')
    write_json(root / 'arms.json', rows)
    contracts = {}
    for row in rows:
        contract = devset.arm_contract(root, row, load_manifest(row['manifest']))
        assert not look_recording.validate_arm(row, contract)
        if contract is not None:
            contracts[row['arm']] = contract
    if pool == 'B':
        write_json(root / 'dev_contracts.json', contracts)
    return seal(root, final_root=final_root, pool=pool, cell=cell)


def prior_b_manifests(suite):
    """Identity metadata only, including previously reserved B selections."""
    sources = {}
    for root in sorted(TRACE.iterdir()):
        if not root.is_dir() or not root.name.startswith(('r11_', 'r12_', 'r13_', 'r14_pilot_', 'r14_descent_')):
            continue
        arms = root / 'arms.json'
        if not arms.is_file():
            continue
        for row in json.loads(arms.read_text()):
            if row.get('dev') is not True or devset.suite_short(row['suite']) != suite:
                continue
            path = row.get('manifest')
            if path and Path(path).is_file() and path not in sources:
                manifest = load_manifest(path)
                devset.check_manifest_pool(manifest, 'B')
                sources[path] = dict(sha256=devset.sha(path), pairs=sorted(manifest['selected']))
    return sources


def selection():
    result = dict(seed=SEED, per_task=30, independent_validation=False, model_suites={})
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            subset = devset.r10_subset(model, suite, 200)
            pairs = [(t, i) for t in range(10) for i in range(50)
                     if i not in subset['excluded_inits_by_task'][str(t)]]
            assert len(pairs) == 300
            prior = prior_b_manifests(suite)
            selected = set(pairs)
            result['model_suites'][f'{model}_{suite}'] = dict(pairs=pairs, selection_subset_200=subset,
                served_subsets={str(n):devset.r10_subset(model,suite,n) for n in (100,200,500)},
                prior_sources={path:dict(sha256=r['sha256'], reused_pairs=sorted(selected & set(map(tuple,r['pairs']))))
                               for path,r in prior.items()},
                reused_prior_pairs=sorted(selected & set().union(*(set(map(tuple,r['pairs'])) for r in prior.values()))),
                r14_sources={path:dict(sha256=r['sha256'],reused_pairs=sorted(selected & set(map(tuple,r['pairs']))))
                             for path,r in prior.items() if '/r14_descent_' in path})
    return result


def queue(path, records):
    atomic(path, ''.join(f'@{r["intended_host"]} {r["final_root"]}|{" ".join(r["arms"])}\n'
                         for r in records).encode())


def build_dev():
    selected = selection()
    write_json(HERE / 'selection.json', selected)
    verdict = json.loads((HERE / 'c10_proof.json').read_text())
    if not verdict['library_independent']:
        raise ValueError('this arm layout requires successful pure C10 proof')
    records, shared, used_fits = [], [], {}
    jobs = [(f'{m}_{s}_100', TRACE/f'r15_descent_c10_{m}_{s}', ('c10',))
            for m in ('pi05','groot') for s in ('l10','spatial')]
    jobs += [(cell, TRACE/f'r15_descent_{cell}', DEV_ORDER) for cell in CELLS]
    for cell, root, order in jobs:
        if root.exists(): raise FileExistsError(root)
        model, suite, size = cell.split('_')
        sel = selected['model_suites'][f'{model}_{suite}']
        manifest = root / 'dev_selection.json'
        write_json(manifest, dict(dev=True, init_pool='B', model=model, suite=suite, seed=SEED,
            per_task=30, pairs=sel['pairs'], subset=sel['selection_subset_200'],
            independent_validation=False, selection_sha256=devset.sha(HERE/'selection.json')))
        specs=[]
        for level in order:
            rec=frozen(cell,level);used_fits[f'{cell}:{level}']=rec
            name = f'r15d_{model}_{suite}_pure_c10' if order == ('c10',) else arm_name(cell,level)
            specs.append(arm_spec(cell,name,rec,manifest))
        write_json(root/'protocol.json',dict(dev=True,init_pool='B',paired=True,per_task=30,
            episodes_per_arm=300,recording=False,decision_logs=True,independent_validation=False,
            order=list(order),pure_reference=str(TRACE/f'r15_descent_c10_{model}_{suite}'),
            primary_tolerance_pp=5,first_failure_stop=True,leave_own_episode_out=size=='500',
            fits_include_own_episode=size=='500',evaluation_launched=False))
        rec=emit(root,cell,specs,order,pool='B',final_root=root)
        (shared if order==('c10',) else records).append(rec)
        print('DEV_READY',root,len(order),'arms x 300',flush=True)
    write_json(HERE/'dev_roots.json',records);write_json(HERE/'c10_roots.json',shared)
    write_json(HERE/'dev_fits.json',used_fits)
    # Global descending levels, with every size advancing together.
    lines=[f'@{r["intended_host"]} {r["final_root"]}|{" ".join(r["arms"])}\n' for r in shared]
    lines += [f'@{r["intended_host"]} {r["final_root"]}|{arm_name(r["cell"],level)}\n'
              for level in DEV_ORDER for r in records]
    atomic(HERE/'queue_dev.txt',''.join(lines).encode())
    caps={c:dict(periodic_cap_by_level={l:level_cap(c,l) for l in LEVELS},
                fit_sha256_by_level={l:frozen(c,l)['sha256'] for l in LEVELS},
                test_fit_sha256_by_level={l:frozen(c,l,test=True)['sha256'] for l in LEVELS},
                leave_own_episode_out=c.endswith('_500')) for c in CELLS}
    write_json(HERE/'cap_tables.json',dict(cells=caps))
    return records


def test_reference(cell):
    model,suite,_=cell.split('_')
    return json.loads((HERE/'c10_proof.json').read_text())['test_references'][f'{model}_{suite}']


def emit_test_a(levels, uniform, output):
    if not isinstance(levels, dict) or not levels or set(levels) - set(CELLS):
        raise ValueError('levels must map supported cells to complete task-level tables')
    if not isinstance(uniform, dict) or set(uniform) != set(levels) or any(v not in LEVELS for v in uniform.values()):
        raise ValueError('uniform must map exactly the same cells to allowed levels')
    levels = {c: validate_levels(v) for c, v in levels.items()}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('fresh staging directory required')
    templates = [TRACE / f'r11_knob_{i}/eval500.json' for i in range(1, 5)]
    if len({p.read_bytes() for p in templates}) != 1:
        raise ValueError('R11 eval500 templates differ')
    selected = load_manifest(templates[0])
    devset.check_manifest_pool(selected, 'A')
    if set(selected['selected']) != {(t, i) for t in range(10) for i in range(50)}:
        raise ValueError('R11 eval500 must contain exactly the 500 A pairs')
    records = []
    write_json(output / 'input.json', dict(levels=levels, uniform=uniform))
    for cell in CELLS:
        if cell not in levels:
            continue
        root = output / f'r15_descent_test_{cell}'
        manifest = root / 'eval500.json'
        atomic(manifest, templates[0].read_bytes())
        blob = descent_blob(cell, levels[cell])
        path = output / 'artifacts' / f'{cell}_task_descent.pkl'
        atomic(path, pickle.dumps(blob, protocol=4))
        fit = dict(path=str(path), sha256=devset.sha(path), spec=SPEC, kwargs=blob['kwargs'])
        specs = [arm_spec(cell, f'r15dt_{cell}_uniform_{uniform[cell]}', frozen(cell, uniform[cell], test=True), manifest),
                 arm_spec(cell, f'r15dt_{cell}_task_descent', fit, manifest)]
        write_json(root / 'descent_table.json', dict(levels=levels[cell], uniform=uniform[cell], caps=blob['method'].knob_settings['K'],
            off_tasks=[t for t in range(10) if levels[cell][str(t)] == 'off'], fit=fit,
            template_sha256=devset.sha(templates[0]), shared_c10_reference=test_reference(cell),
            source_artifacts={l: frozen(cell, l, test=True)['sha256'] for l in set(levels[cell].values())}))
        records.append(emit(root, cell, specs, [uniform[cell], 'task_descent'], pool='A', final_root=TRACE / root.name))
        print('STAGED', cell, '2 arms x 500 episodes', flush=True)
    write_json(output / 'test_a_roots.json', records)
    queue(output / 'queue_test_a.txt', records)
    return records


def check_seal(root):
    root = Path(root)
    rec = json.loads((root / 'ROOT.json').read_text())
    return rec


def replace_owned(value, source, target):
    if isinstance(value, str):
        return str(target) + value[len(str(source)):] if value == str(source) or value.startswith(str(source) + '/') else value
    if isinstance(value, dict):
        return {k: replace_owned(v, source, target) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_owned(v, source, target) for v in value]
    return value


def install_test_a(output, destination=TRACE):
    output = Path(output).resolve()
    records = json.loads((output / 'test_a_roots.json').read_text())
    targets = [Path(destination).resolve() / Path(r['final_root']).name for r in records]
    for rec, target in zip(records, targets, strict=True):
        if target.exists():
            raise FileExistsError(f'fresh destination required: {target}')
        if not target.name.startswith('r15_descent_'):
            raise ValueError('invalid destination')
        if check_seal(rec['root']) != rec:
            raise ValueError('staging index differs from sealed root')
    for rec, target in zip(records, targets, strict=True):
        source = Path(rec['root'])
        target.mkdir(parents=True)
        for path in sorted(source.rglob('*')):
            if not path.is_file() and not path.is_dir():
                raise ValueError('regular files only')
            if not path.is_file() or path.name == 'ROOT.json':
                continue
            new = target / path.relative_to(source)
            if path.suffix == '.json':
                write_json(new, replace_owned(json.loads(path.read_text()), source, target))
            else:
                atomic(new, path.read_bytes())
        # Restore byte-identical A identities, including the original formatting.
        atomic(target / 'eval500.json', (source / 'eval500.json').read_bytes())
        seal(target, final_root=target, pool='A', cell=rec['cell'])
    return [str(t) for t in targets]


def owned_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE) or path == HERE:
        raise ValueError('staging output must be below rounds/r15/descent')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('build-dev')
    emit_ap = sub.add_parser('emit-test-a')
    emit_ap.add_argument('--levels', type=Path, required=True)
    emit_ap.add_argument('--uniform', type=Path, required=True)
    emit_ap.add_argument('--out', type=owned_output, required=True)
    install_ap = sub.add_parser('install-test-a')
    install_ap.add_argument('--out', type=owned_output, required=True)
    args = ap.parse_args()
    if args.action == 'build-dev':
        build_dev()
    elif args.action == 'emit-test-a':
        emit_test_a(json.loads(args.levels.read_text()), json.loads(args.uniform.read_text()), args.out)
    else:
        print(json.dumps(install_test_a(args.out)))


if __name__ == '__main__':
    main()
