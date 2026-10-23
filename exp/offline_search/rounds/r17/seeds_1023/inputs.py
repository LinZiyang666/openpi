"""New library seeds of the success-filtered knob descent: views, fits, Q3, gaps, selection (CPU only).

One cache_inputs root per (bundle, seed): /data/mw_finetune/eval/sd<tag>_<bundle>/assets/cache_inputs (the real
directory on /nvme, the /data path a symlink, as longseg_off_1022/fit500.py). <tag> is x<seed % 100> for LIBERO
(x3, x4; old x2) and x<seed> for MT50 (x9, x10; old x8). Per suite/size N, each produced by the procedure that
made the old seed's artifact:

  <suite>/parent/READY.json, parent/subsets/size_<N>/        the library view (LIBERO: teacher_library.nested_views
                                                             with default_rng(seed+task) per task, as r10/data.py;
                                                             MT50: the seed library's own subsets/, W3)
  <suite>/n<N>_C<L>/fit                                      the library's own fit, segment L = C10 (DP, Fast-WAM,
                                                             GR00T), C16 (Cosmos), C5 (pi0.5 MT50): MT50 copies the
                                                             seed library's fits/ (W3 finish); LIBERO fits as
                                                             teacher_library.fit_nested_libraries (a library fit of
                                                             this very view, i.e. the old seed, is copied instead)
  <suite>/n<N>_C<L>/stall                                    its Q3 (cache_study.fit_inputs procedure, fit500.fit_cell)
  <suite>/n<N>_C10/stall_decision1018                        segment-head Q3 (longseg_1017.recalibrate.fit_q3)
  <suite>/n<N>_C5/{fit,stall}                                C5 fit and Q3 (fit500.fit_cell), long-segment models only
  <suite>/n<N>_C*/selection/<fit digest>.json                kept = view episodes that succeeded (audit_counts.summarize);
                                                             filter_failed_knob.runtime reads it beside the fit
  manifest.json                                              byte inventory (filter_failed.stage / longseg staging)

Gaps (outside cache_inputs, in products/<suite>_<N>.json with every path above): C10 from the decision-clock
replay over successful episodes (recalibrate.episodes/DecisionSchedule/solve, the :filtered: arm); C5 reuses its
seed's C10 gap (owner 10-22); Cosmos and pi0.5 from cache_knob.schedule_levels (the knob calibration).

    OMP_NUM_THREADS=1 python -m exp.offline_search.rounds.r17.seeds_1023.inputs views --bundle dp_libero --seed 20261003
    OMP_NUM_THREADS=1 python -m exp.offline_search.rounds.r17.seeds_1023.inputs cell --bundle dp_libero --seed 20261003 [--suite S --size N]
    python -m exp.offline_search.rounds.r17.seeds_1023.inputs manifest --bundle dp_libero --seed 20261003
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

EVAL = Path('/data/mw_finetune/eval')
LIBRARY = Path('/data/r17_library')
HERE = Path(__file__).resolve().parent
SEEDS = {'libero': (20261003, 20261004), 'mt50': (9, 10)}
OLD = {'libero': 20261002, 'mt50': 8}
SIZES = {'libero': (50, 100, 200, 300, 400), 'mt50': (250, 500, 1000, 1500, 2000, 2500)}
BUNDLES = ('dp_libero', 'fastwam_libero', 'cosmos_libero', 'dp_mw', 'fastwam_mw', 'groot_mw', 'pi05_mw')
LONG = ('dp_libero', 'fastwam_libero', 'dp_mw', 'fastwam_mw', 'groot_mw')  # also run C5 (long segment off)
LEVELS = {'k50': .5, 'k40': .4, 'k30': .3, 'k20': .2, 'k10': .1}


def read(path):
    return json.loads(Path(path).read_text())


def benchmark(bundle):
    return 'mt50' if bundle.endswith('_mw') else 'libero'


def model(bundle):
    return bundle.rsplit('_', 1)[0]


def suites(bundle):
    return ('mt50',) if benchmark(bundle) == 'mt50' else ('libero_spatial', 'libero_10')


def tag(seed):
    return 'x'+str(seed % 100 if seed > 1000 else seed)


def library(bundle, suite, seed):
    if benchmark(bundle) == 'libero':
        return LIBRARY/f'{bundle}_{suite}'/'bpool'
    # Keeps the _mt50/bpool suffix cache_study.subset_dir/view_size key on (250 = size_5).
    return (LIBRARY if seed == OLD['mt50'] else LIBRARY/f'seed{seed}')/f'{bundle}_mt50'/'bpool'


def labels(bundle, suite, seed):
    """The library's episode outcomes as this host's campaigns read them (/data/r17_library)."""
    return str(Path('/data/r17_library')/library(bundle, suite, seed).relative_to(LIBRARY)/'READY.json')+'#metadata.episodes'


def root(bundle, seed):
    return EVAL/f'sd{tag(seed)}_{bundle}'


def inputs(bundle, seed):
    return root(bundle, seed)/'assets/cache_inputs'


def view_dir(bundle, suite, seed, size):
    """The library-side view (its parent arrays beside it)."""
    lib = library(bundle, suite, seed)
    if benchmark(bundle) == 'mt50':
        return lib/'subsets'/f'size_{size//50}'
    return lib/f'subsets_s{seed}'/f'size_{size}'


def product_path(bundle, seed, suite, size):
    return root(bundle, seed)/'products'/f'{suite}_{size}.json'


def products(bundle, seed):
    """(suite, size) -> product of every cell; a missing cell is an error."""
    result = {}
    for suite in suites(bundle):
        for size in SIZES[benchmark(bundle)]:
            path = product_path(bundle, seed, suite, size)
            if not path.exists():
                raise ValueError(f'{bundle} seed {seed}: inputs cell {suite}/{size} not made ({path})')
            result[suite, size] = read(path)
    if not (inputs(bundle, seed)/'manifest.json').exists():
        raise ValueError(f'{bundle} seed {seed}: run `inputs manifest` after all cells')
    return result


def cells(seed):
    """filter_failed_knob.common.cells() of a seeds_1023 seed: (model, suite, size) -> kept episodes."""
    result = {}
    for path in sorted(EVAL.glob(f'sd{tag(seed)}_*/products/*.json')):
        p = read(path)
        if p['seed'] != seed:
            continue
        result[model(p['bundle']), p['suite'], p['size']] = dict(
            model=model(p['bundle']), suite=p['suite'], size=p['size'], seed=seed, source='seeds_1023',
            parent=str(library(p['bundle'], p['suite'], seed)), selection=str(view_dir(p['bundle'], p['suite'], seed, p['size'])/'VIEW.json'),
            labels=labels(p['bundle'], p['suite'], seed), candidate_view=None, **p['summary'])
    return result


def alignment(lib, seed):
    """The library's own B-pool alignment receipt with the R10 subset rule for `seed`.

    Same selection rule as r10/data.py make_subsets: default_rng(seed+task).permutation of the task's 50 B
    episodes, first N/10, nested. For the old seed this is the receipt itself (same digest)."""
    import numpy as np
    from exp.r17_integration.debug.validate import digest  # library_alignment's
    from exp.r17_integration.ops.library_alignment import load_alignment
    want = read(lib/'READY.json')['metadata']['alignment_digest']
    base = next(doc for doc in map(load_alignment, sorted(lib.parent.glob('campaigns/*/ALIGNMENT.json')))
                if doc['digest'] == want)
    order = {str(t): np.random.default_rng(seed+t).permutation(np.arange(t*50, (t+1)*50)).tolist() for t in range(10)}
    doc = dict(base, subset_seed=seed, permutation_by_task=order,
               selections={str(n): {str(t): order[str(t)][:n//10] for t in range(10)} for n in (50, 100, 200, 300, 400, 500)})
    if seed != base['subset_seed']:
        doc['subset_hashes'] = {}  # the R10 subset files exist for the original seed only
    doc.pop('digest')
    doc['digest'] = digest(doc)
    return doc


def views(bundle, seed):
    """LIBERO nested views of a new subset seed, beside the library's own subsets/ (same parent_path)."""
    from exp.r17_integration.search.teacher_library import nested_views
    for suite in suites(bundle):
        lib = library(bundle, suite, seed)
        manifest = read(lib/'READY.json')
        parent = SimpleNamespace(directory=lib, manifest=manifest, rows=tuple(manifest['rows']))
        nested_views(parent, alignment(lib, seed), sizes=SIZES['libero'], output_root=lib/f'subsets_s{seed}')
        print('VIEWS', bundle, suite, seed, flush=True)


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    part = target.with_name(f'{target.name}.part{os.getpid()}')
    shutil.copyfile(source, part)
    part.replace(target)


def native_commit(parent):
    layout = parent['binding']['model']['layout']
    return max(layout['replan_steps'], min(10, layout['horizon']))  # teacher_library.fit_nested_libraries


def fit_library_view(bundle, size, view, fit_path, commit):
    """teacher_library.fit_nested_libraries on one view: the same settings and receipt."""
    from exp.r17_integration.common.layers import recipe as recipe_module
    from exp.r17_integration.debug.validate import digest, file_sha
    from exp.r17_integration.ops.cache_study import profiles
    from exp.r17_integration.search.fit import fit_recipe, save_fitted_recipe
    from exp.offline_search.rounds.r17.longseg_off_1022.fit500 import selected_view
    descriptor, selected = selected_view(view)
    parent = descriptor.parent.manifest
    _, serving = profiles(bundle, parent)
    recipe, receipt = fit_recipe(selected, task_names=serving.ordered_tasks, compact_loeo=True,
                                 commit_horizon=commit, corrector=True, early=True, k=16, kref=8)
    receipt.update(library_size=size, selected_parent_rows=len(descriptor.rows),
        selection_digest=descriptor.manifest['content_digest'], parent_digest=parent['content_digest'],
        serving_loeo_required_on_B=size == 500,
        serving_loeo='B query task/local-init must exclude library_episode_id; A queries have no B own episode',
        shared_products=['PCA', 'Mahalanobis', 'early_metric', 'LOEO_corrector', 'distance_calibration',
                         'continuity_scale', 'corrector_caps', 'periodic_knob'])
    receipt['caps_knobs'] = dict(shared_recipe_sha256=file_sha(recipe_module.__file__),
        corrector='unchanged P1 distance ramp, motion-only0.5 cap, fitted sigma/distance_scale',
        stall_gate='Q3 calibrated stall trigger (old no-progress guard retired)',
        periodic_gap=recipe.periodic_gap,
        owner_IR_status='requires selected-library schedule calibration and measured native cost profile')
    receipt['digest'] = digest({k: v for k, v in receipt.items() if k != 'digest'})
    save_fitted_recipe(recipe, receipt, fit_path)


def native_fit(bundle, suite, seed, size, view, parent):
    """(segment, library fit dir or None): the library's own fit when it is of this very view (MT50: required,
    W3 finish); otherwise LIBERO fits it as teacher_library does."""
    lib = library(bundle, suite, seed)
    store = read(view/'VIEW.json')['content_digest']
    source = lib/'fits'/f'size_{size//50 if benchmark(bundle) == "mt50" else size}'
    ready = read(source/'READY.json') if (source/'READY.json').exists() else None
    if ready and ready['receipt']['store_digest'] == store:
        return ready['receipt']['config']['commit_horizon'], source
    if benchmark(bundle) == 'mt50':
        raise ValueError(f'{source}: no library fit of view {store[:12]}; finish the seed {seed} library first (W3)')
    return native_commit(parent), None


def write_selection(fit, kept, meta):
    """filter_failed_knob.runtime.selection_file of this fit."""
    from exp.offline_search.rounds.r17.filter_failed.build import write
    digest = read(fit/'READY.json')['digest']
    path = fit.parent/'selection'/(digest+'.json')
    write(path, dict(meta, fit_digest=digest, kept_episode_ids=kept))
    return digest


def cell(bundle, seed, suite, size):
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.ops.cache_study import profiles
    from exp.r17_integration.search.fit import load_fitted_recipe
    from exp.offline_search.rounds.r17.filter_failed.audit_counts import summarize
    from exp.offline_search.rounds.r17.filter_failed.build import write
    from exp.offline_search.rounds.r17.longseg_1017 import recalibrate as rc
    from exp.offline_search.rounds.r17.longseg_off_1022.fit500 import fit_cell, selected_view
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    from exp.r17_integration.ops.cache_knob import schedule_levels
    lib, view = library(bundle, suite, seed), view_dir(bundle, suite, seed, size)
    if not (view/'VIEW.json').exists():
        raise ValueError(f'{view}: no library view (LIBERO: run `inputs views`; MT50: finish the seed library, W3)')
    base = inputs(bundle, seed)
    if not root(bundle, seed).exists():
        from exp.r17_integration.ops.immutable_release import nvme_dir
        try:
            nvme_dir(root(bundle, seed), exist_ok=True)
        except FileExistsError:  # a parallel cell made it
            pass
    # The view as cache_study.stage_inputs lays it out: VIEW.json's parent_path ../.. is <suite>/parent.
    parent = read(lib/'READY.json')
    target = base/suite/'parent/subsets'/f'size_{size}'
    copy(lib/'READY.json', base/suite/'parent/READY.json')
    for name in ('VIEW.json', 'row_indices.npy'):
        copy(view/name, target/name)
    manifest = read(view/'VIEW.json')
    summary = summarize(parent['metadata']['episodes'], manifest['episode_ids'], id_key='library_episode_id', task_key='task_name')
    kept = summary['kept_episode_ids']
    meta = dict(bundle=bundle, seed=seed, suite=suite, size=size, store_digest=manifest['content_digest'],
                labels=labels(bundle, suite, seed))
    # Native segment: fit, Q3, (C10) segment-head Q3 and gaps.
    layout = parent['binding']['model']['layout']
    commit, source = native_fit(bundle, suite, seed, size, view, parent)
    cdir = base/suite/f'n{size}_C{commit}'
    if not (cdir/'fit/READY.json').exists():
        if source:
            for name in ('fit.npz', 'READY.json'):
                copy(source/name, cdir/'fit'/name)
        else:
            fit_library_view(bundle, size, view, cdir/'fit', commit)
    fit_cell(bundle, suite, size, cdir, view_dir=view, commit=commit)
    native = dict(commit=commit, fit=f'{suite}/n{size}_C{commit}/fit', fit_digest=write_selection(cdir/'fit', kept, meta),
                  stall=f'{suite}/n{size}_C{commit}/stall', stall_digest=read(cdir/'stall/READY.json')['digest'])
    if bundle in LONG:
        record = dict(model=model(bundle), suite=suite, size=size, seed=seed, commit=commit, replan=layout['replan_steps'],
                      fit=str(cdir/'fit'), fit_digest=native['fit_digest'], old_stall=str(cdir/'stall'),
                      old_stall_digest=native['stall_digest'])
        data = rc.labels(record, kept)
        q3 = rc.fit_q3(record, data, work=root(bundle, seed)/'recalibration')
        model_q3 = StallModel.load(q3['path'])
        replay = rc.DecisionSchedule([e for e in rc.episodes(record, data, model_q3) if e['success']])
        solutions = {level: rc.solve(target, replay) for level, target in LEVELS.items()}
        native.update(source_stall=native['stall'], source_stall_digest=native['stall_digest'],
                      stall=f'{suite}/n{size}_C{commit}/stall_decision1018', stall_digest=q3['digest'],
                      gaps={level: s['new_gap'] for level, s in solutions.items()}, solutions=solutions,
                      gap_source='longseg_1017.recalibrate decision-clock replay, :filtered: arm (successful episodes)')
        print('GAPS', bundle, seed, suite, size, [round(s['predicted_ir'], 4) for s in solutions.values()], flush=True)
    else:
        descriptor, selected = selected_view(view)
        _, serving = profiles(bundle, parent)
        fit = load_fitted_recipe(cdir/'fit', expected_binding=parent['binding'])
        levels = schedule_levels(fit, selected, dict(benchmark_protocol=serving.identity, allow_partial_commit=False),
                                 serving.ordered_tasks)
        native.update(gaps={level: levels[str(target)]['gap'] for level, target in LEVELS.items()}, solutions=levels,
                      gap_source='cache_knob.schedule_levels (knob calibration, all library episodes)')
    result = dict(bundle=bundle, model=model(bundle), seed=seed, tag=tag(seed), suite=suite, size=size,
                  view=f'{suite}/parent/subsets/size_{size}', store_digest=manifest['content_digest'],
                  teacher_sampler_identity=digest(parent['binding'].get('sampler', parent['binding']['model']['default_sampler'])),
                  summary=summary, native=native)
    if bundle in LONG:
        c5 = base/suite/f'n{size}_C5'
        fit_cell(bundle, suite, size, c5, view_dir=view, commit=5)
        result['c5'] = dict(commit=5, fit=f'{suite}/n{size}_C5/fit', fit_digest=write_selection(c5/'fit', kept, meta),
                            stall=f'{suite}/n{size}_C5/stall', stall_digest=read(c5/'stall/READY.json')['digest'],
                            gaps=native['gaps'], gap_source='its seed\'s C10 gaps (owner 10-22: long segment off is not recalibrated)')
    write(product_path(bundle, seed, suite, size), result)
    print('CELL', bundle, seed, suite, size, 'kept', summary['kept'], '/', summary['collected'], flush=True)
    return result


def main():
    global EVAL, LIBRARY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('views', 'cell', 'manifest'))
    p.add_argument('--bundle', required=True, choices=BUNDLES)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--suite')
    p.add_argument('--size', type=int)
    p.add_argument('--library-root', type=Path, default=LIBRARY, help='another host\'s copy of /data/r17_library')
    p.add_argument('--out-root', type=Path, default=EVAL, help='where sd<tag>_<bundle> roots go (sync back to /data/mw_finetune/eval)')
    a = p.parse_args()
    EVAL, LIBRARY = a.out_root, a.library_root
    if a.action == 'views':
        if benchmark(a.bundle) != 'libero':
            raise ValueError('MT50 views come with the seed library (W3 finish)')
        views(a.bundle, a.seed)
    elif a.action == 'cell':
        for suite in ([a.suite] if a.suite else suites(a.bundle)):
            for size in ([a.size] if a.size else SIZES[benchmark(a.bundle)]):
                cell(a.bundle, a.seed, suite, size)
    else:
        from exp.offline_search.rounds.r17.longseg_off_1022.fit500 import manifest
        for suite in suites(a.bundle):
            for size in SIZES[benchmark(a.bundle)]:
                if not product_path(a.bundle, a.seed, suite, size).exists():
                    raise ValueError(f'cell {suite}/{size} not made')
        manifest({a.bundle: inputs(a.bundle, a.seed)}, purpose=f'seeds_1023 {a.bundle} library seed {a.seed}')


if __name__ == '__main__':
    main()
