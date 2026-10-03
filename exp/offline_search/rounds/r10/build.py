"""Build, prefit, emit, selftest and plan locally. Never sync or launch."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import time

from .data import HERE, STORE, RUNS, SIZES, CELLS, make_subsets, write_json, sha

REPO = HERE.parents[3]
SPEC = 'exp.offline_search.rounds.r10.method:SizeController'
PREFIX = ['taskset', '-c', '22-37,66-81', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=.:src', 'PYTHONDONTWRITEBYTECODE=1',
          str(REPO / '.venv/bin/python')]


def run_root(stage, model):
    stem = {1: 'size', 2: 'corr', '2b': 'corr2'}[stage]
    return RUNS / f'r10_{stem}_{model}'


def stage_label(stage):
    return str(stage).upper()


def run_cmd(args, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w') as f:
        p = subprocess.run(PREFIX + args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'CPU command failed ({p.returncode}): {logfile}')


def specs(stage):
    # Explicit exceptions: read ONLY constructor specs and A evaluation pair identities.
    # Never open any R1-R9 fit, recording, result, or derived data.
    arows = {r['arm']: r for r in json.loads((RUNS / 'r08_main/arms.json').read_text())}
    grows = {r['arm']: r for r in json.loads((RUNS / 'r08_abl/arms.json').read_text())}
    manifest = json.loads((RUNS / 'r09_recipe_full_p/eval500.json').read_text())
    assert manifest == [[t, i] for t in range(10) for i in range(50)]
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        write_json(run / 'eval500.json', manifest)
        rows = []
        for suite in ('l10', 'spatial'):
            for size in SIZES:
                refsize = 50 if size == 50 else 500
                a = arows[f'r8_{model}_{suite}_{refsize}_A']
                g = grows[f'r8abl_onlynp_{"p" if model == "pi05" else "g"}_{"sp" if suite == "spatial" else suite}_{refsize}']
                assert a['kwargs'] == g['kwargs']['base_kwargs']
                bkw = copy.deepcopy(a['kwargs'])
                gkw = copy.deepcopy(g['kwargs'])
                gkw.pop('base_kwargs')
                assert bkw == dict(lib='current' if size == 50 else 'big', kref=5 if size == 50 else 8,
                                   serving='anchor_tail', budget=1, gates='budget_only')
                for variant in (('A', 'G') if stage == 1 else ('GC_loeo', 'GC_pair')):
                    name = f'r10_{model}_{suite}_{size}_{variant}'
                    kw = dict(size=size, variant=variant, base_kwargs=bkw, guard_kwargs=gkw)
                    if stage != 1:
                        kw.update(source_fit=str(HERE / 'artifacts' / f'r10_{model}_{suite}_{size}_G.pkl'),
                                  head_path=str(HERE / ('heads2b' if stage == '2b' else 'heads') /
                                                f'{model}_{suite}_{size}_{variant[3:]}.npz'))
                    flags = ['--os-root', str(STORE), '--os-no-shadow-native', '--os-tokens', 'off', '--os-blind']
                    if variant != 'A':
                        flags += ['--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only']
                    flags += ['--os-fit-artifact', str(run / 'fits' / f'{name}.pkl')]
                    rows.append(dict(name=name, model=model, suite=suite, mode='plugin', method=SPEC, kwargs=kw,
                        full_model=variant != 'A', cost_ledger=True, manifest=str(run / 'eval500.json'),
                        client_overrides=copy.deepcopy((a if variant == 'A' else g)['client_overrides']), plugin_args=flags))
        write_json(run / 'arms_in.json', rows)
        write_json(run / 'protocol.json', dict(fit_pool='B; selected parent library rows ONLY', eval_pool='official A',
            init_states='LIBERO init_files (standard runner; no init_states override)', sizes=list(SIZES),
            kref_rule='5 for 50 episodes; 8 for 100-500 episodes', escalation=False, Bval_recordings_used=[],
            metadata_only_sources=['r08_main/arms.json', 'r08_abl/arms.json', 'r09_recipe_full_p/eval500.json']))
        if stage == '2b':
            from .train import REVISION
            protocol = json.loads((run / 'protocol.json').read_text())
            protocol.update(revision=REVISION, prev_hit=True, pair_cap=16,
                            pair_radius='per-task median LOEO 16-NN distance',
                            weight_mass='number of represented anchor rows; equal episode weights',
                            guard_fit='unchanged accepted Stage 1 canonical G')
            write_json(run / 'protocol.json', protocol)
        print(f'SPECS {run}: {len(rows)} arms', flush=True)


def build(stage, workers=4):
    specs(stage)
    jobs = []
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        for row in json.loads((run / 'arms_in.json').read_text()):
            jobs.append((run, row))
    def one(job):
        run, row = job
        fit = run / 'fits' / f'{row["name"]}.pkl'
        if stage != 1:
            # Training publishes the B-only heads while this preparation runs.
            import numpy as np
            head = Path(row['kwargs']['head_path'])
            source = Path(row['kwargs']['source_fit'])
            while True:
                ready = False
                if head.exists() and source.exists() and time.time() - head.stat().st_mtime > 2:
                    try:
                        with np.load(head, allow_pickle=False) as z:
                            meta = json.loads(str(z['meta_json']))
                            ready = meta.get('source_fit_sha256') == sha(source)
                            if stage == '2b':
                                from .train import REVISION
                                ready &= meta.get('revision') == REVISION
                    except (OSError, ValueError, EOFError):
                        pass
                if ready: break
                time.sleep(2)
        args = ['-m', 'exp.offline_search.rounds.r10.prefit', '--os-method', SPEC, '--os-kwargs', json.dumps(row['kwargs']),
                '--os-cell', f'{row["model"]}_{row["suite"]}_cache', '--os-log-dir', str(run / 'prefit_logs' / row['name']),
                '--os-tag', row['name'], *row['plugin_args']]
        if fit.exists():
            with fit.open('rb') as f:
                blob = pickle.load(f)
            want = dict(spec=SPEC, kwargs=row['kwargs'], cell=f'{row["model"]}_{row["suite"]}_cache')
            if any(blob.get(k) != v for k, v in want.items()):
                raise ValueError(f'existing R10 artifact metadata mismatch: {fit}')
            if blob['method'].fit_info.get('fit_version') != 2:
                # Replace only our own obsolete R10 output, never an input corpus.
                fit.unlink()
                run_cmd(args, run / 'prefit_logs' / f'{row["name"]}.log')
        else:
            run_cmd(args, run / 'prefit_logs' / f'{row["name"]}.log')
        canonical = HERE / ('artifacts2b' if stage == '2b' else 'artifacts') / fit.name
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temporary = canonical.with_suffix('.pkl.tmp')
        shutil.copyfile(fit, temporary)
        temporary.replace(canonical)
        print(f'FIT {row["name"]} {fit.stat().st_size} bytes', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        run_cmd(['-m', 'exp.offline_search.closed_loop.ops.emit_arms', '--run-root', str(run), '--spec', str(run / 'arms_in.json')],
                run / 'validation/emitter.log')
    validate(stage, workers)


def validate(stage, workers=4):
    jobs = []
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        for row in json.loads((run / 'arms.json').read_text()):
            jobs.append((run, row))
    def one(job):
        run, row = job
        run_cmd(['-m', 'exp.offline_search.rounds.r10.selftest', '--run', str(run), '--arm', row['arm']],
                run / 'validation' / f'{row["arm"]}.log')
        print('SELFTEST', row['arm'], flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    plans = []
    for model in ('pi05', 'groot'):
        run = run_root(stage, model)
        names = [r['arm'] for r in json.loads((run / 'arms.json').read_text())]
        run_cmd(['-m', 'exp.offline_search.closed_loop.ops.h100.control', 'plan', str(run), *names], run / 'validation/control_plan.log')
        plan = json.loads((run / 'h100_sync/plan.json').read_text())
        plans.append(plan)
        write_json(run / 'validation/artifact_shas.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                    for p in sorted((run / 'fits').glob('*.pkl'))])
    sources = [p for p in HERE.glob('*.py')] + [HERE / 'HANDBACK.md']
    (HERE / f'STAGE{stage_label(stage)}_SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in sorted(sources) if p.exists()))
    write_json(HERE / f'stage{stage}_deployment.json', dict(extra_h100_store_bytes=0,
        new_store_arrays=[], total_plan_bytes=sum(p['bytes'] for p in plans),
        artifact_bytes=sum(f['size'] for p in plans for f in p['files'] if 'fitted method' in ' '.join(f['reasons'])),
        new_source_files=[str(p.relative_to(REPO)) for p in sorted(sources) if p.exists()]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['subsets', 'stage1', 'stage2', 'stage2b', 'validate1', 'validate2', 'validate2b'])
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.action == 'subsets': make_subsets()
    elif a.action.startswith('validate'): validate('2b' if a.action.endswith('2b') else int(a.action[-1]), a.workers)
    else: build('2b' if a.action.endswith('2b') else int(a.action[-1]), a.workers)


if __name__ == '__main__':
    main()
