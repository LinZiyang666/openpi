"""CPU comparison of unfiltered retrieval-only fits with original Cache fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audit_counts import HERE, read


def compare(left, right, path='', *, records=None):
    """Compare typed arrays byte for byte, including values nested in task tables."""
    records = [] if records is None else records
    if isinstance(left, np.ndarray):
        same = (isinstance(right, np.ndarray) and left.shape == right.shape and left.dtype == right.dtype
                and left.tobytes(order='C') == right.tobytes(order='C'))
        record = dict(path=path, equal=same, shape=list(left.shape), dtype=str(left.dtype),
                      original_sha256=hashlib.sha256(left.tobytes(order='C')).hexdigest(),
                      refit_sha256=hashlib.sha256(right.tobytes(order='C')).hexdigest())
        if not same and left.shape == right.shape and np.issubdtype(left.dtype, np.number):
            record['max_abs_difference'] = float(np.max(np.abs(left.astype(float) - right.astype(float))))
        records.append(record)
    elif hasattr(left, '__dict__'):
        compare(vars(left), vars(right), path, records=records)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            records.append(dict(path=path, equal=False, original_keys=list(left), refit_keys=list(right)))
        for key in left.keys() & right.keys():
            compare(left[key], right[key], path + '/' + str(key), records=records)
    elif isinstance(left, (tuple, list)):
        records.append(dict(path=path + '/length', equal=len(left) == len(right)))
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path + '/' + str(i), records=records)
    else:
        records.append(dict(path=path, equal=bool(left == right)))
    return records


def r17(pipeline, output):
    from exp.r17_integration.ops import cache_study as study
    from exp.r17_integration.search.fit import fit_recipe, load_fitted_recipe, save_fitted_recipe
    from exp.r17_integration.search.teacher_library import IndexedArray

    source, bundle = (
        (Path('/data/r17_library/mt50/studies/pi_k2/STUDY.json'), 'pi05_mw')
        if pipeline == 'r17_mt50' else
        (study.ROOT / 'exp/r17_integration/ops/cache_study/sizes/STUDY.json', 'cosmos_libero')
    )
    row = next(r for r in read(source)['models'][bundle] if r['group'] == 'c')
    library = Path(row['library'])
    parent = read(library / 'READY.json')
    view = study.view_path(row)
    selected = read(view / 'VIEW.json')
    indices = np.load(view / 'row_indices.npy', allow_pickle=False)
    arrays = {key: IndexedArray(np.load(library / entry['file'], mmap_mode='r', allow_pickle=False), indices)
              for key, entry in parent['arrays'].items()
              if key.startswith('key_') or key in ('state', 'normalized_actions')}
    store = SimpleNamespace(manifest=selected, rows=tuple(parent['rows'][i] for i in indices),
                            fields=tuple(arrays), array=lambda key: arrays[key])
    old = load_fitted_recipe(row['fit'], expected_binding=parent['binding'])
    ready = read(Path(row['fit']) / 'READY.json')
    settings = dict(ready['receipt']['config'])
    names = settings.pop('task_names')
    settings['guard'] = False  # Loader likewise discards this retired historical fit flag.
    print(f'{pipeline}: refitting {len(indices)} unfiltered library rows, corrector=False', flush=True)
    new, receipt = fit_recipe(store, task_names=names, corrector=False, compact_loeo=True, **settings)
    records = []
    for field in ('action', 'state', 'task', 'episode', 'step', 'fields', 'pca', 'projected', 'sigma', 'tasks'):
        compare(getattr(old, field), getattr(new, field), field, records=records)
    destination = output / (pipeline + '_retrieval_only')
    if not destination.exists():
        save_fitted_recipe(new, receipt, destination)
    result = dict(pipeline=pipeline, source_study=str(source), source_row=row, corrector=False,
                  source_fit_digest=ready['digest'], refit=str(destination),
                  status='PASS' if all(r['equal'] for r in records) else 'FAIL',
                  compared=records, ignored_fields=['heads', 'distance_scale', 'fit_evidence'])
    return result


def r10(output):
    """R10 A already fits and serves BlindAWM with no corrector/guard wrapper."""
    import pickle
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    from exp.offline_search.rounds.r10 import method
    from exp.offline_search.rounds.r10.data import SubsetLibrary
    from exp.offline_search.harness.api import Context
    from unittest.mock import patch

    source = HERE.parents[1] / 'r10/artifacts/r10_pi05_spatial_50_A.pkl'
    with source.open('rb') as stream:
        blob = pickle.load(stream)
    old = blob['method']
    assert old.variant == 'A' and type(old.inner) is BlindAWM
    target = output / 'r10_fresh_pca'
    (target / 'subsets').mkdir(parents=True, exist_ok=True)
    original_selection = method.HERE / 'subsets/pi05_spatial_50.npy'
    (target / 'subsets/pi05_spatial_50.npy').write_bytes(original_selection.read_bytes())
    ctx = Context(root='/home/weiland/trace_runs/offline_search_store', cell='pi05_spatial_cache',
                  seed=0, scratch=output / 'r10_scratch')
    lib = SubsetLibrary(ctx.root, ctx.lib_key, 50)
    new = method.SizeController(size=50, variant='A', base_kwargs=old.base_kwargs, guard_kwargs=old.guard_kwargs)
    print('r10_vla: refitting unfiltered pi05 spatial N50 using the original A/BlindAWM pipeline', flush=True)
    with patch.object(method, 'HERE', target):
        new.fit(lib, ctx)
    # A has no correction fit to remove: compare the whole fitted method state.
    # Profiler and the lazy action mmap wrapper are runtime helpers, not learned arrays.
    records = []
    for field, value in vars(old.inner).items():
        if field in ('prof', 'act'):
            continue
        other = getattr(new.inner, field)
        if hasattr(value, '__dict__') and not isinstance(value, SimpleNamespace):
            value, other = vars(value), vars(other)
        compare(value, other, field, records=records)
    return dict(pipeline='r10_vla', source_fit=str(source), variant=old.variant,
                inner_class=type(old.inner).__name__, corrector=False,
                status='PASS' if all(r['equal'] for r in records) else 'FAIL', compared=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pipeline', choices=('r17_mt50', 'r17_libero', 'r10_vla'))
    p.add_argument('--output', type=Path, default=HERE / 'proofs')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = r10(args.output) if args.pipeline == 'r10_vla' else r17(args.pipeline, args.output)
    path = args.output / (args.pipeline + '.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    differences = [r for r in result['compared'] if not r['equal']]
    print(json.dumps(dict(pipeline=args.pipeline, status=result['status'], output=str(path),
                          compared=len(result['compared']), differences=differences[:10]), indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
