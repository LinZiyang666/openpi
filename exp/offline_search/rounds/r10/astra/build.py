"""Freeze-checked assembly; standard emitter, CPU plugin tests, standard plan only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import pickle
import subprocess
import time
import numpy as np

from .boundary import HERE, RUNS, NEW, install
from .offline import now
from exp.offline_search.rounds.r10 import data

REPO = data.HERE.parents[3]
SPEC = 'exp.offline_search.rounds.r10.astra.method:DistanceController'
PREFIX = ['taskset', '-c', '10-21,54-65', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'PYTHONPATH=.:src', 'PYTHONDONTWRITEBYTECODE=1', 'CUDA_VISIBLE_DEVICES=',
          str(REPO / '.venv/bin/python'), '-m', 'exp.offline_search.rounds.r10.astra.run_tool']


def run_cmd(mode, args, name):
    log = HERE / 'validation' / f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        result = subprocess.run([*PREFIX, mode, *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'CPU preparation failure; see {log}')


def input_paths():
    paths = list(data.HERE.glob('*.py'))
    for model, suite in data.CELLS:
        for size in data.SIZES:
            paths.extend([data.HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl',
                data.HERE / 'heads2b' / f'{model}_{suite}_{size}_loeo.npz',
                RUNS / f'r10_corr2_{model}' / 'fits' / f'r10_{model}_{suite}_{size}_GC_loeo.pkl',
                data.HERE / 'subsets' / f'{model}_{suite}_{size}.json',
                data.HERE / 'subsets' / f'{model}_{suite}_{size}.npy'])
    for model in ('pi05', 'groot'):
        paths.extend([RUNS / f'r10_corr2_{model}' / x for x in ('arms.json', 'eval500.json')])
    return paths


def snapshot():
    path = HERE / 'input_hashes.json'
    current = {str(p): data.sha(p) for p in input_paths()}
    if path.exists():
        assert current == json.loads(path.read_text()), 'protected input changed'
    else:
        data.write_json(path, current)
    return len(current)


def specs():
    freeze = json.loads((HERE / 'freeze.json').read_text())
    for path, digest in freeze['hashes'].items():
        assert data.sha(path) == digest, f'frozen input changed: {path}'
    for model in ('pi05', 'groot'):
        run = RUNS / f'r10_corr3_{model}'
        prior = RUNS / f'r10_corr2_{model}'
        manifest_bytes = (prior / 'eval500.json').read_bytes()
        assert json.loads(manifest_bytes) == [[t, i] for t in range(10) for i in range(50)]
        run.mkdir(exist_ok=False)
        (run / 'eval500.json').write_bytes(manifest_bytes)
        rows = []
        for row in json.loads((prior / 'arms.json').read_text()):
            if row['kwargs']['variant'] != 'GC_loeo':
                continue
            name = row['arm'].replace('GC_loeo', 'GC_dist')
            kw = copy.deepcopy(row['kwargs'])
            kw['variant'] = 'GC_dist'
            kw['calibration_path'] = str(HERE / 'calibration' / f'{model}_{row["suite_short"]}_{kw["size"]}.json')
            flags = list(row['plugin_args'])
            flags[flags.index('--os-fit-artifact') + 1] = str(run / 'fits' / f'{name}.pkl')
            rows.append(dict(name=name, model=model, suite=row['suite_short'], mode='plugin', method=SPEC,
                kwargs=kw, full_model=row['full_model'], cost_ledger=True,
                manifest=str(run / 'eval500.json'), client_overrides=copy.deepcopy(row['client_overrides']),
                plugin_args=flags))
        assert len(rows) == 12
        data.write_json(run / 'arms_in.json', rows)
        data.write_json(run / 'protocol.json', dict(fit_pool='B libraries only', eval_pool='official A, 500 pairs',
            init_states='standard LIBERO init_files; no override', variant='GC_dist',
            same_G_and_stage2b_LOEO_heads=True, new_task_indexed_parameters=False,
            freeze_timestamp=freeze['timestamp_utc'], prediction_sha256=freeze['hashes'][str(HERE / 'PREDICTION.md')],
            rule=freeze['rule'], no_closed_loop_results_read=True, evaluation_launched=False))


def fitted_equal(old, new):
    """Compare immutable G/LOEO numerical state, permitting only the new scalar gate."""
    if isinstance(old, np.ndarray):
        np.testing.assert_array_equal(old, new); return
    if isinstance(old, dict):
        assert set(old) <= set(new)
        for k, v in old.items():
            fitted_equal(v, new[k])
    elif isinstance(old, (str, int, float, bool, type(None))):
        assert old == new or (isinstance(old, float) and np.isnan(old) and np.isnan(new))
    elif isinstance(old, (tuple, list)):
        assert len(old) == len(new)
        for a, b in zip(old, new): fitted_equal(a, b)
    elif hasattr(old, '__dict__'):
        fitted_equal(vars(old), vars(new))
    else:
        assert type(old) is type(new)


def audit_fits():
    records = []
    for model in ('pi05', 'groot'):
        run = RUNS / f'r10_corr3_{model}'
        for row in json.loads((run / 'arms_in.json').read_text()):
            oldpath = RUNS / f'r10_corr2_{model}' / 'fits' / (row['name'].replace('GC_dist', 'GC_loeo') + '.pkl')
            path = run / 'fits' / (row['name'] + '.pkl')
            with oldpath.open('rb') as f: old = pickle.load(f)['method']
            with path.open('rb') as f: new = pickle.load(f)['method']
            fitted_equal(old.inner, new.inner)
            np.testing.assert_array_equal(old.row_subset, new.row_subset)
            assert new.inner.base.distance_scale > 0
            assert set(vars(new.inner.base)) - set(vars(old.inner.base)) == {'distance_scale', 'distance_rule'}
            records.append(dict(arm=row['name'], identical_G_and_head=True, sha256=data.sha(path),
                                bytes=path.stat().st_size))
    data.write_json(HERE / 'fit_audit.json', records)


def build(workers):
    snapshot()
    specs()
    jobs = [(run, row) for run in NEW for row in json.loads((run / 'arms_in.json').read_text())]
    def fit(job):
        run, row = job
        run_cmd('prefit', ['--os-method', SPEC, '--os-kwargs', json.dumps(row['kwargs']),
            '--os-cell', f'{row["model"]}_{row["suite"]}_cache',
            '--os-log-dir', str(HERE / 'prefit_logs' / row['name']), '--os-tag', row['name'],
            *row['plugin_args']], row['name'] + '_prefit')
        print('FIT', row['name'], flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fit, jobs))
    audit_fits()
    for run in NEW:
        run_cmd('emit', ['--run-root', str(run), '--spec', str(run / 'arms_in.json')], run.name + '_emit')
        data.write_json(run / 'emission_receipt.json', dict(timestamp_utc=now(), arms=12,
            prediction_sha256=data.sha(HERE / 'PREDICTION.md'), freeze_sha256=data.sha(HERE / 'freeze.json')))
    validate(workers)


def validate(workers):
    jobs = []
    for run in NEW:
        testroot = HERE / 'test_runs' / run.name
        # Reuse sol's real-plugin selftest, with all synthetic telemetry outside os_closed_loop.
        rows = json.loads((run / 'arms.json').read_text())
        data.write_json(testroot / 'arms.json', rows)
        jobs.extend((testroot, row['arm']) for row in rows)
    def test(job):
        root, name = job
        run_cmd('selftest', [str(root), name], name + '_selftest')
        print('SELFTEST', name, flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(test, jobs))
    plans = []
    for run in NEW:
        names = [r['arm'] for r in json.loads((run / 'arms.json').read_text())]
        run_cmd('plan', [str(run), *names], run.name + '_plan')
        plans.append(json.loads((run / 'h100_sync/plan.json').read_text()))
        print('PLAN', run.name, flush=True)
    unique = {f['rel']: f for plan in plans for f in plan['files']}
    new = [f for f in unique.values() if any(f'/runs/{r.name}/' in '/' + f['rel'] for r in NEW)
           or '/r10/astra/' in f['rel']]
    data.write_json(HERE / 'deployment.json', dict(extra_h100_store_bytes=0,
        incremental_run_and_calibration_bytes=sum(f['size'] for f in new),
        dependency_union_bytes=sum(f['size'] for f in unique.values()),
        full_plans=[dict(run=p['run'], bytes=p['bytes'], files=len(p['files'])) for p in plans],
        new_dependencies=new, protected_inputs_unchanged=snapshot()))
    reports = [json.loads((root / 'selftest' / name / 'selftest_report.json').read_text()) for root, name in jobs]
    data.write_json(HERE / 'selftests.json', reports)


if __name__ == '__main__':
    install()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['build', 'validate', 'snapshot'])
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.action == 'build': build(a.workers)
    elif a.action == 'validate': validate(a.workers)
    else: print('protected inputs:', snapshot())
