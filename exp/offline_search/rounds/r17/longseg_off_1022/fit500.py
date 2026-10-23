"""LIBERO size-500 C5 fits and their C5 Q3 stall artifacts (DP, Fast-WAM; libero_10, libero_spatial).

The body of one cell is exp.r17_integration.ops.cache_study.fit_inputs for commit 5: fit_recipe(k=16,
kref=8, early=True, corrector=True, guard=False, periodic_gap=None, compact_loeo=True) on the library's
size_500 view, the same receipt fields, then StallModel.fit on that fit (H, exec_steps = the library
layout; commit_controls = min(H, 2*exec_steps)) and the same r17.cache_stall.v1 receipt. No IR or gap
calibration. `check` refits an existing cell (DP libero_10 n50) into a scratch directory and compares
digests with the C5 fit/stall the C5 gated cache study served.

    nice -n 19 python -m exp.offline_search.rounds.r17.longseg_off_1022.fit500 cell --bundle dp_libero --suite libero_10
    python -m exp.offline_search.rounds.r17.longseg_off_1022.fit500 manifest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp.offline_search.rounds.r17.longseg_1017.audit import EVAL

LIBRARY = Path('/data/r17_library')
# New cache_inputs roots (real directories on /nvme, /data path a symlink), staged to the H100 like any root.
ROOTS = {'dp_libero': EVAL/'c5n500_dp_libero', 'fastwam_libero': EVAL/'c5n500_fastwam_libero'}
N = 500
COMMIT = 5


def inputs(bundle):
    return ROOTS[bundle]/'assets/cache_inputs'


def cell_dir(bundle, suite, n=N):
    return inputs(bundle)/suite/f'n{n}_C{COMMIT}'


def selected_view(view_dir):
    """(store descriptor, fit view) of one selected library view: key/state/action arrays only."""
    import numpy as np
    from exp.r17_integration.ops.cache_study import read
    from exp.r17_integration.search.cache_recipe import sealed_store_descriptor
    from exp.r17_integration.search.teacher_library import IndexedArray
    descriptor = sealed_store_descriptor(view_dir, read(Path(view_dir)/'VIEW.json')['content_digest'])
    parent = descriptor.parent.manifest
    indices = np.load(descriptor.directory/'row_indices.npy', allow_pickle=False)
    arrays = {}
    for key, entry in parent['arrays'].items():
        if key.startswith('key_') or key in ('state', 'normalized_actions'):
            arrays[key] = IndexedArray(np.load(descriptor.parent.directory/entry['file'], mmap_mode='r', allow_pickle=False), indices)
    view = type('SelectedFitView', (), {'manifest': descriptor.manifest, 'rows': descriptor.rows,
                'fields': tuple(arrays), 'array': lambda self, key: arrays[key]})()
    return descriptor, view


def fit_cell(bundle, suite, n, out, *, view_dir=None, commit=COMMIT):
    """view_dir: a selected view other than the library's own subsets/ (seeds_1023); an existing out/fit is kept."""
    import numpy as np
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.ops.cache_study import atomic_json, file_sha, profiles, read, subset_dir
    from exp.r17_integration.search.fit import fit_recipe, load_fitted_recipe, save_fitted_recipe
    from exp.r17_integration.search.cache_recipe import metric_mapping
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    library = LIBRARY/(bundle+'_'+suite)/'bpool'
    descriptor, view = selected_view(view_dir or library/'subsets'/subset_dir(library, n))
    parent = descriptor.parent.manifest
    _, serving = profiles(bundle, parent)
    out = Path(out)
    fit_path, stall_path = out/'fit', out/'stall'
    print(f'B-only fit/calibration {bundle} {suite} N={n} C={commit}', flush=True)
    if not (fit_path/'READY.json').exists():
        fitted, receipt = fit_recipe(view, task_names=serving.ordered_tasks, corrector=True,
            compact_loeo=True, commit_horizon=commit,
            allow_partial_commit=bool(commit % parent['binding']['model']['layout']['replan_steps']),
            k=16, kref=8, early=True, guard=False, periodic_gap=None)
        receipt.update(library_size=n, selection_digest=descriptor.manifest['content_digest'],
            parent_digest=parent['content_digest'], calibration_scope='B library only; C-specific head')
        receipt['digest'] = digest({k: v for k, v in receipt.items() if k != 'digest'})
        save_fitted_recipe(fitted, receipt, fit_path)
    fitted = load_fitted_recipe(fit_path, expected_binding=parent['binding'])
    if not (stall_path/'READY.json').exists():
        data = dict(manifest=dict(H=fitted.layout.horizon, exec_steps=fitted.layout.replan_steps),
            task_id=fitted.task, episode=fitted.episode, step=fitted.step,
            success=np.array([r['success'] for r in descriptor.rows], bool))
        model = StallModel.fit(data, metric=metric_mapping(fitted),
            commit_controls=min(fitted.layout.horizon, 2*fitted.layout.replan_steps))
        model.save(stall_path)
        receipt = dict(schema='r17.cache_stall.v1', store_digest=descriptor.manifest['content_digest'],
            fitted_digest=read(fit_path/'READY.json')['digest'],
            calibration_partition='B_library_successful_LOEO',
            calibration_source=f'selected teacher B{n}; Q3 successful-episode LOEO empirical residuals',
            library_size=n, commit_horizon=commit, independent_B_validation_recording=False,
            files={p.name: file_sha(p) for p in stall_path.iterdir() if p.is_file()},
            unavailable_tasks=[k for k, v in model.tasks.items() if len(v['references']) < 4])
        receipt['digest'] = digest(receipt)
        atomic_json(stall_path/'READY.json', receipt)
    result = dict(bundle=bundle, suite=suite, n=n, fit=str(fit_path), fit_digest=read(fit_path/'READY.json')['digest'],
                  stall=str(stall_path), stall_digest=read(stall_path/'READY.json')['digest'],
                  store_digest=descriptor.manifest['content_digest'])
    print(json.dumps(result), flush=True)
    return result


def manifest(roots=None, purpose='LIBERO size-500 C5 fits and C5 Q3 stalls'):
    """manifest.json of each new cache_inputs root (snapshot staging needs one; a byte inventory only)."""
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.ops.cache_study import atomic_json, file_sha
    for bundle, root in (roots or {b: inputs(b) for b in ROOTS}).items():
        root = Path(root)
        files = {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and p.name not in ('manifest.json', '.eval.lock')}
        value = dict(schema='r17.cache_inputs.v1', files=files, purpose=purpose)
        value['digest'] = digest(value)
        atomic_json(root/'manifest.json', value)
        print(bundle, len(files), value['digest'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('cell', 'check', 'manifest'))
    p.add_argument('--bundle', choices=tuple(ROOTS))
    p.add_argument('--suite', choices=('libero_10', 'libero_spatial'))
    p.add_argument('--out')
    a = p.parse_args()
    if a.action == 'cell':
        from exp.r17_integration.ops.immutable_release import nvme_dir
        if not ROOTS[a.bundle].exists():
            nvme_dir(ROOTS[a.bundle], exist_ok=True)
        fit_cell(a.bundle, a.suite, N, cell_dir(a.bundle, a.suite))
    elif a.action == 'check':
        fit_cell('dp_libero', 'libero_10', 50, a.out)
    else:
        manifest()


if __name__ == '__main__':
    main()
