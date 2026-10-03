"""Build one artifact, prepare 28 libraries, emit four current arms; no launches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import subprocess
import time

from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import HERE as R10, STORE, RUNS, CELLS, SIZES, PARENT, sha, write_json
from .fitting import fit_path
from .recipe import R10Recipe

HERE = Path(__file__).resolve().parent
REPO = R10.parents[3]
SPEC = 'exp.offline_search.rounds.r10.recipe.recipe:R10Recipe'
CURRENT = RUNS / 'r10_recipe_current'
PREFIX = ['taskset', '-c', '22-37,66-81', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1',
          'CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=.:src', 'PYTHONDONTWRITEBYTECODE=1', str(REPO / '.venv/bin/python')]


def subset_kwargs(model, suite, path, root=STORE, library=None):
    path = fit_path(path, root)
    spec = json.loads(path.read_text())
    if spec.get('model', model) != model or spec.get('suite', suite) != suite:
        raise ValueError('episode subset cell mismatch')
    parent = spec.get('parent')
    inferred = spec.get('library') or (Path(parent).name if parent else None)
    library = library or inferred
    if library is None:
        raise ValueError('subset spec needs library or parent, or pass --library')
    directory = Path(root) / 'library' / f'{model}_{suite}' / library
    fit_path(directory, root)
    if parent and Path(parent).resolve() != directory.resolve():
        raise ValueError('subset parent must match the selected library')
    if 'episodes_sha256' in spec and spec['episodes_sha256'] != sha(directory / 'episodes.json'):
        raise ValueError('parent episode metadata changed')
    ids = spec.get('episode_ids_by_task')
    if ids is None:
        raise ValueError('subset spec must contain episode_ids_by_task')
    # Inline the spec: serving/planning has no dependency on this input JSON.
    return dict(library=library, episode_subset=ids)


def artifact_path(model, suite, size=None):
    if size is None:
        return CURRENT / 'fits' / f'r10_recipe_{model}_{suite}_current.pkl'
    return HERE / 'artifacts' / f'r10_recipe_{model}_{suite}_{size}.pkl'


def fit_one(model, suite, kwargs, output, root=STORE):
    output = Path(output)
    cell = f'{model}_{suite}_cache'
    want = dict(spec=SPEC, kwargs=kwargs, cell=cell)
    if output.exists():
        raise FileExistsError(f'artifact already exists; choose a new --output: {output}')
    scratch = HERE / 'scratch' / output.stem
    ctx = api.Context(root=root, cell=cell, seed=0, scratch=scratch)
    method = R10Recipe(**kwargs)
    started = time.monotonic()
    method.fit(None, ctx)
    api.check_method_attrs(method)
    blob = dict(method=method, registered={}, fit_s=time.monotonic() - started, **want)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('wb') as f:
        pickle.dump(blob, f, protocol=4)
    temporary.replace(output)
    print(json.dumps(dict(artifact=str(output), sha256=sha(output), bytes=output.stat().st_size,
                          episodes=method.size, kref=method.inner.base.kref,
                          distance_scale=method.inner.base.distance_scale, seconds=blob['fit_s'])), flush=True)


def run_cpu(args, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w') as f:
        result = subprocess.run(PREFIX + args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'CPU preparation failed ({result.returncode}): {logfile}')


def prepare(workers, current_only=False):
    jobs = [(m, s, n) for m, s in CELLS for n in SIZES] if not current_only else []
    jobs += [(m, s, None) for m, s in CELLS]
    def one(job):
        model, suite, size = job
        kwargs = dict(library='current') if size is None else subset_kwargs(
            model, suite, R10 / 'subsets' / f'{model}_{suite}_{size}.json')
        output = artifact_path(model, suite, size)
        if output.exists():
            # Idempotent preparation validates only OUR completed output.
            with output.open('rb') as f:
                blob = pickle.load(f)
            want = dict(spec=SPEC, kwargs=kwargs, cell=f'{model}_{suite}_cache')
            if any(blob.get(k) != v for k, v in want.items()) or not blob['method'].fit_info['library_only']:
                raise ValueError(f'own artifact metadata mismatch: {output}')
            print(f'READY {output.name}', flush=True)
            return
        args = ['-m', 'exp.offline_search.rounds.r10.recipe.build', 'fit', '--model', model, '--suite', suite,
                '--output', str(output)]
        if size is None:
            args += ['--library', 'current']
        else:
            args += ['--r10-size', str(size)]
        run_cpu(args, HERE / 'validation' / f'{output.stem}_fit.log')
        print(f'FIT {output.name}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    emit_current()


def emit_current():
    sources = [RUNS / f'r10_size_{m}' / 'eval500.json' for m in ('pi05', 'groot')]
    content = sources[0].read_bytes()
    if sources[1].read_bytes() != content or json.loads(content) != [[t, i] for t in range(10) for i in range(50)]:
        raise ValueError('R10 size manifests must agree on the 500 official A pairs')
    CURRENT.mkdir(exist_ok=True)
    (CURRENT / 'eval500.json').write_bytes(content)
    rows = []
    library_counts = {}
    for model, suite in CELLS:
        name = f'r10_recipe_{model}_{suite}_current'
        output = artifact_path(model, suite)
        with output.open('rb') as f:
            blob = pickle.load(f)
        method = blob['method']
        counts = {t: len(es) for t, es in method.fit_info['episode_ids_by_task'].items()}
        expected_kref = 5 if all(n == 5 for n in counts.values()) else 8
        if method.library != 'current' or method.inner.base.kref != expected_kref:
            raise ValueError('current fit must follow the five-episodes-per-task kref rule')
        library_counts[f'{model}_{suite}'] = dict(episodes=method.size, by_task=counts, kref=expected_kref)
        kwargs = dict(library='current')
        flags = ['--os-root', str(STORE), '--os-no-shadow-native', '--os-tokens', 'off', '--os-blind',
                 '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only',
                 '--os-fit-artifact', str(output)]
        rows.append(dict(name=name, model=model, suite=suite, mode='plugin', method=SPEC, kwargs=kwargs,
            full_model=True, cost_ledger=True, manifest=str(CURRENT / 'eval500.json'),
            client_overrides=dict(replan_steps=5, **(dict(resize_size=256) if model == 'groot' else {})),
            plugin_args=flags))
    write_json(CURRENT / 'arms_in.json', rows)
    write_json(CURRENT / 'protocol.json', dict(layers=['R4 cache, ten-control commit', 'R8 only-no-progress',
        'in-library Stage2b LOEO head with GC_dist'], fit_inputs='selected current library only',
        distance_strength='.5*clip((2-r)/1.25,0,1); step 0 uncorrected',
        kref_rule='5 at five episodes per task, 8 otherwise', escalation=False,
        actual_current_libraries=library_counts,
        eval_manifest='A 500 pairs from r10_size_*/eval500.json; evaluation only',
        manifest_sha256=sha(CURRENT / 'eval500.json'), init_states='standard LIBERO init_files',
        evaluation_launched=False))
    run_cpu(['-m', 'exp.offline_search.closed_loop.ops.emit_arms', '--run-root', str(CURRENT),
             '--spec', str(CURRENT / 'arms_in.json')], HERE / 'validation' / 'current_emit.log')
    print(f'EMITTED {CURRENT}: 4 arms', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('fit', help='fit one library and write one standard plugin artifact')
    f.add_argument('--model', choices=['pi05', 'groot'], required=True)
    f.add_argument('--suite', choices=['l10', 'spatial'], required=True)
    f.add_argument('--library')
    choice = f.add_mutually_exclusive_group()
    choice.add_argument('--subset', type=Path, help='JSON with library and episode_ids_by_task (or an R10 subset JSON)')
    choice.add_argument('--r10-size', type=int, choices=SIZES)
    f.add_argument('--root', type=Path, default=STORE)
    f.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser('prepare', help='fit 24 subsets and four current libraries, then emit current arms')
    prep.add_argument('--workers', type=int, default=4)
    prep.add_argument('--current-only', action='store_true')
    sub.add_parser('emit-current')
    a = p.parse_args()
    if a.action == 'fit':
        path = a.subset or (R10 / 'subsets' / f'{a.model}_{a.suite}_{a.r10_size}.json' if a.r10_size else None)
        kwargs = subset_kwargs(a.model, a.suite, path, a.root, a.library) if path else dict(library=a.library or 'current')
        fit_one(a.model, a.suite, kwargs, a.output, a.root)
    elif a.action == 'prepare':
        if not 1 <= a.workers <= 16:
            p.error('--workers must be 1..16')
        prepare(a.workers, a.current_only)
    else:
        emit_current()


if __name__ == '__main__':
    main()
