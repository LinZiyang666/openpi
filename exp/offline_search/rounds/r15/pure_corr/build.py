"""Assemble 24 frozen artifacts, emit standard roots, and verify CPU-only plans."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import yaml

from exp.offline_search.rounds.r09.recipe.recipe import load_head
from exp.offline_search.rounds.r10.data import CELLS, HERE as R10, RUNS, SIZES, sha
from .runtime import PureCorrector

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SPEC = 'exp.offline_search.rounds.r15.pure_corr.runtime:PureCorrector'
EXPECTED = [[t, i] for t in range(10) for i in range(50)]


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json(path, obj):
    atomic(path, (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def load(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def fit_path(row):
    return Path(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1])


def root_for(model, suite):
    return RUNS / f'r15_purecorr_{model}_{suite}'


def reference(model, suite, size, variant):
    if variant in ('A', 'G'):
        root = RUNS / f'r10_size_{model}'
    else:
        suffix = '_b' if model == 'groot' and (suite == 'spatial' or size >= 400) else ''
        root = RUNS / f'r10_corr3_{model}{suffix}'
    name = f'r10_{model}_{suite}_{size}_{variant}'
    row, = [r for r in json.loads((root / 'arms.json').read_text()) if r['arm'] == name]
    arm = root / 'runs' / name
    summary = arm / 'summary.json'
    # Summary metadata only: no success, SR, journal, trajectory or decision read.
    if json.loads(summary.read_text())['complete'] != 500:
        raise ValueError(f'incomplete R10 reference: {arm}')
    files = dict(arms=root / 'arms.json', fit=fit_path(row), summary=summary,
                 manifest=Path(row['manifest']), run_manifest=arm / 'manifest.json',
                 yaml=Path(row['yaml']), matrix=Path(row['matrix']))
    for key in ('manifest', 'run_manifest'):
        if json.loads(files[key].read_text()) != EXPECTED:
            raise ValueError(f'wrong R10 A identities: {files[key]}')
    return dict(root=str(root), arm=name, arm_path=str(arm), row=row,
                files={k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()})


def equal(a, b):
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('frozen array mismatch')
    elif isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError('frozen keys mismatch')
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('frozen sequence mismatch')
        for x, y in zip(a, b):
            equal(x, y)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        raise ValueError(f'frozen value mismatch: {a!r} vs {b!r}')


def audit_source(model, suite, size, refs):
    a, g, gc = (load(refs[v]['files']['fit']['path']) for v in ('A', 'G', 'GC_dist'))
    for v, blob in zip(('A', 'G', 'GC_dist'), (a, g, gc)):
        row = refs[v]['row']
        equal({k: blob[k] for k in ('spec', 'kwargs', 'cell')},
              dict(spec=row['method'], kwargs=row['kwargs'], cell=f'{model}_{suite}_cache'))
        if blob['method'].size != size or blob['method'].variant != v:
            raise ValueError('cross-cell/size source substitution')
    kw = refs['GC_dist']['row']['kwargs']
    paths = dict(head=Path(kw['head_path']), calibration=Path(kw['calibration_path']),
                 metric_fit=Path(kw['source_fit']), subset=R10 / 'subsets' / f'{model}_{suite}_{size}.json',
                 rows=R10 / 'subsets' / f'{model}_{suite}_{size}.npy')
    expected_paths = dict(head=R10 / 'heads2b' / f'{model}_{suite}_{size}_loeo.npz',
                          calibration=R10 / 'astra/calibration' / f'{model}_{suite}_{size}.json',
                          metric_fit=R10 / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl')
    for k, path in expected_paths.items():
        if paths[k] != path:
            raise ValueError(f'wrong model/suite/size corrector path: {k}')
    rows = np.load(paths['rows'], allow_pickle=False)
    subset = json.loads(paths['subset'].read_text())
    base = gc['method'].inner.base
    heads, meta = load_head(paths['head'])
    equal(meta, base.head_meta)
    equal(heads, base.heads)
    if meta['cell'] != f'{model}_{suite}_cache' or meta['size'] != size or meta['fit_pool'] != 'B' or meta['variant'] != 'loeo':
        raise ValueError('wrong size-specific head metadata')
    metric_sha = sha(paths['metric_fit'])
    cal = json.loads(paths['calibration'].read_text())
    if meta['source_fit_sha256'] != metric_sha or cal['source_fit_sha256'] != metric_sha:
        raise ValueError('corrector/metric SHA binding mismatch')
    if gc['method'].fit_info['distance_calibration_sha256'] != sha(paths['calibration']):
        raise ValueError('GC_dist calibration SHA binding mismatch')
    equal(base.distance_scale, cal['scale'])
    equal(base.distance_rule, cal['rule'])
    if cal['rule'] != dict(peak=.5, plateau=.75, cutoff=2.) or cal['fit_pool'] != 'B':
        raise ValueError('unexpected frozen distance rule or fitting pool')
    source_g = load(paths['metric_fit'])['method']
    for wrapper in (a['method'], g['method'], gc['method'], source_g):
        equal(wrapper.row_subset, rows)
        if wrapper.size != size or not wrapper.fit_info.get('library_only'):
            raise ValueError('wrong library size or provenance')
    for other in (a['method'].inner, g['method'].inner.base, source_g.inner.base):
        # All fitted A cache values, including every task metric, must equal GC_dist.
        for key, val in vars(a['method'].inner).items():
            if key in ('name', 'prof', '_anchor', 'last_blind_extras', '_last_g'):
                continue
            value = getattr(other, key)
            target = getattr(base, key)
            if key == 'tasks':
                equal({t: vars(v) for t, v in value.items()}, {t: vars(v) for t, v in target.items()})
            elif key == 'act':
                equal(value.__getstate__(), target.__getstate__())
            else:
                equal(value, target)
    library = Path(a['method'].fit_info['parent'])
    ep = np.load(library / 'episode.npy', mmap_mode='r', allow_pickle=False)
    task = np.load(library / 'task_id.npy', mmap_mode='r', allow_pickle=False)
    selected_ids = {str(t): sorted(set(map(int, ep[rows[task[rows] == t]]))) for t in range(10)}
    expected_ids = {str(t): sorted(ids) for t, ids in subset['episode_ids_by_task'].items()}
    equal(selected_ids, expected_ids)
    if sum(map(len, selected_ids.values())) != size:
        raise ValueError('wrong subset episode count')
    audit = dict(model=model, suite=suite, size=size, references=refs,
                 corrector_artifacts={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
                 distance_scale=base.distance_scale, distance_rule=base.distance_rule,
                 subset_episode_ids=selected_ids, subset_row_count=len(rows),
                 same_size_head_calibration_metric=True, A_G_GC_cache_arrays_byte_identical=True,
                 subset_episode_ids_and_rows_identical=True, refitted=False)
    return a, gc, audit


def build():
    from exp.offline_search.closed_loop.ops.emit_arms import main as emit
    audits = []
    for model, suite in CELLS:
        root = root_for(model, suite)
        if (root / 'arms.json').exists():
            raise ValueError(f'root already emitted; use audit/plans: {root}')
        specs = []
        for size in SIZES:
            refs = {v: reference(model, suite, size, v) for v in ('A', 'G', 'GC_dist')}
            a, gc, record = audit_source(model, suite, size, refs)
            name = f'r15_purecorr_{model}_{suite}_{size}_AC_dist'
            kwargs = dict(size=size, corrector_enabled=True, guard_enabled=False)
            method = PureCorrector.assemble(a['method'], gc['method'])
            method.fit_info['frozen_sources'] = record['corrector_artifacts']
            fit = root / 'fits' / f'{name}.pkl'
            atomic(fit, pickle.dumps(dict(method=method, registered=a['registered'], spec=SPEC,
                                         kwargs=kwargs, cell=a['cell'], fit_s=0.), protocol=4))
            ar = refs['A']['row']
            flags = list(ar['plugin_args'])
            flags[flags.index('--os-fit-artifact') + 1] = str(fit)
            specs.append(dict(name=name, model=model, suite=suite, mode=ar['mode'], method=SPEC,
                              kwargs=kwargs, full_model=ar['full_model'], cost_ledger=True,
                              manifest=str(root / 'eval500.json'), client_overrides=ar['client_overrides'],
                              plugin_args=flags, **{k: ar[k] for k in ('yaml_patch', 'server_env', 'server_seed') if k in ar}))
            atomic(root / 'eval500.json', Path(refs['A']['files']['manifest']['path']).read_bytes())
            record.update(arm=name, root=str(root), artifact=dict(path=str(fit), sha256=sha(fit)))
            audits.append(record)
            print(f'ASSEMBLED {name}', flush=True)
        write_json(root / 'arms_in.json', specs)
        emit(['--run-root', str(root), '--spec', str(root / 'arms_in.json')])
        rows = json.loads((root / 'arms.json').read_text())
        for row in rows:
            record = next(x for x in audits if x['arm'] == row['arm'])
            frozen_yaml = Path(record['references']['A']['files']['yaml']['path'])
            # Preserve the actual R10 A rollout/cache template bytes, not a regenerated modern template.
            atomic(row['yaml'], frozen_yaml.read_bytes())
            row['yaml_sha256'] = sha(row['yaml'])
            row.update(init_pool='A', dev=False, episodes=500, recording=False,
                       rollout_replicate_seed=7, r15_pure_corr=True)
        write_json(root / 'arms.json', rows)
        write_json(root / 'protocol.json', dict(init_pool='A', episodes_per_arm=500, sizes=list(SIZES),
            corrector='same model/suite/size frozen R10 GC_dist only', guard=False, refitted=False,
            strength='.5 * clip((2-r)/1.25, 0, 1); step 0 zero', recording=False,
            rollout_seed=7, seed_evidence='examples/libero/main.py Args.seed=7; R10 WorkerSpec omitted seed',
            evaluation_launched=False, measure_only=True, selection=False))
        definitions = {str(p.relative_to(root)): sha(p) for p in [root / 'arms.json', root / 'arms_in.json',
                        root / 'eval500.json', root / 'protocol.json', *sorted((root / 'config').glob('*.yaml'))]}
        write_json(root / 'ROOT.json', dict(init_pool='A', episodes_per_arm=500, arm_count=6,
                   definitions_sha256=definitions, fit_sha256={x['artifact']['path']: x['artifact']['sha256']
                   for x in audits if x['root'] == str(root)}, fleet_binding=False))
    write_json(HERE / 'audit.json', dict(PASS=True, cells=24, refitted=False, records=audits))
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['new_arm', 'R10_GC_dist_arm', 'R10_GC_dist_fit', 'GC_fit_sha256',
                     'head', 'head_sha256', 'calibration', 'calibration_sha256', 'metric_fit',
                     'metric_fit_sha256', 'subset', 'subset_sha256', 'rows', 'rows_sha256', 'subset_equal'])
    for r in audits:
        ref = r['references']['GC_dist']
        art = r['corrector_artifacts']
        writer.writerow([r['arm'], ref['arm'], ref['files']['fit']['path'], ref['files']['fit']['sha256'],
                         *[v for k in ('head', 'calibration', 'metric_fit', 'subset', 'rows')
                           for v in (art[k]['path'], art[k]['sha256'])], True])
    atomic(HERE / 'corrector_bindings.csv', table.getvalue().encode())
    queue = ''.join(f'@h100 {root_for(m, s)}|r15_purecorr_{m}_{s}_{n}_AC_dist\n'
                    for n in SIZES for m, s in CELLS)
    atomic(HERE / 'queue.txt', queue.encode())
    audit()


def audit():
    records = json.loads((HERE / 'audit.json').read_text())['records']
    if len(records) != 24 or len({r['arm'] for r in records}) != 24:
        raise ValueError('expected 24 unique same-size bindings')
    for r in records:
        for ref in r['references'].values():
            for file in ref['files'].values():
                if sha(file['path']) != file['sha256']:
                    raise ValueError(f'changed R10 reference {file["path"]}')
        for file in [r['artifact'], *r['corrector_artifacts'].values()]:
            if sha(file['path']) != file['sha256']:
                raise ValueError(f'changed frozen source {file["path"]}')
        _, gc, fresh = audit_source(r['model'], r['suite'], r['size'], r['references'])
        new = load(r['artifact']['path'])['method']
        if not new.corrector_enabled or new.guard_enabled or new.size != r['size']:
            raise ValueError('wrong ablation switches')
        for k in ('heads', 'head_meta', 'sig_head', 'distance_scale', 'distance_rule', 'blend', 'chans'):
            equal(getattr(new.inner, k), getattr(gc['method'].inner.base, k))
        equal(new.row_subset, np.load(r['corrector_artifacts']['rows']['path'], allow_pickle=False))
        equal(fresh['subset_episode_ids'], r['subset_episode_ids'])
        root = Path(r['root'])
        seal = json.loads((root / 'ROOT.json').read_text())
        for file, digest in seal['definitions_sha256'].items():
            if sha(root / file) != digest:
                raise ValueError(f'changed emitted definition {file}')
    print('AUDIT PASS: 24 exact cell/size heads, calibrations, metrics, subsets and sealed roots', flush=True)


def plans():
    if os.environ.get('SERVER_HOST') != 'h100' or os.environ.get('WORKER_HOST') != 'timan107':
        raise ValueError('plans require SERVER_HOST=h100 WORKER_HOST=timan107')
    from exp.offline_search.closed_loop.ops.h100 import assets, control
    cache = {}
    def cached_sha(path):
        p = Path(path)
        stat = p.stat()
        key = (str(p), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in cache:
            cache[key] = sha(p)
        return cache[key]
    reports = []
    with patch.object(assets, 'sha', cached_sha), patch.object(control, 'sha', cached_sha):
        for model, suite in CELLS:
            root = root_for(model, suite)
            names = [r['arm'] for r in json.loads((root / 'arms.json').read_text())]
            plan = control.sync(root, names, plan_only=True)
            if plan['worker_host'] != 'timan107' or control.server_host() != 'h100':
                raise ValueError('wrong plan topology')
            plan['server_host'] = 'h100'
            write_json(root / 'h100_sync/plan.json', plan)
            reports.append(dict(root=str(root), server_host='h100', worker_host='timan107',
                                arms=names, files=len(plan['files']), bytes=plan['bytes'],
                                plan_sha256=sha(root / 'h100_sync/plan.json')))
    write_json(HERE / 'plans.json', dict(PASS=True, SERVER_HOST='h100', WORKER_HOST='timan107',
                                       fleet_count=1, root_fleet_binding=False, records=reports))
    print('PLANS PASS: all 24 arms @h100, workers timan107; no sync or launch', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'audit', 'plans'])
    args = parser.parse_args()
    globals()[args.action]()


if __name__ == '__main__':
    main()
