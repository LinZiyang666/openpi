"""Regression of the new-seed inputs on the old seeds (LIBERO 20261002, MT50 8): CPU only.

Run `inputs views/cell/manifest` with the old seed first (roots sdx2_<bundle>, sdx8_<bundle>), then
`regress refit` for LIBERO and `regress compare`. Every artifact must equal, byte for byte, the one the old seed's batches served:
  views          <suite>/parent/subsets/size_N/{VIEW.json,row_indices.npy}  = library subsets/
  native fit     n<N>_C<L>/fit/{fit.npz,READY.json}                          = library fits/ (copied for this view)
  native Q3      n<N>_C<L>/stall/{READY.json,stall.json,stall.pkl,stall.loeo.csv}
                 = longseg_1017 RECALIBRATION.json Q3 source (C10) / the old campaign row's stall (Cosmos, pi0.5)
  segment Q3     n<N>_C10/stall_decision1018/(same files)                    = RECALIBRATION.json Q3 path
  gaps           C10: products solutions[level] = RECALIBRATION solutions[<fit>:filtered:<level>];
                 Cosmos/pi0.5: products gaps = the old knob YAMLs' periodic_gap of that cell and level
  C5 fit, Q3     n<N>_C5/{fit,stall}/*                                       = longseg_off_1022.build.c5_refs
  selection      kept episodes = filter_failed/CELLS.json, and the old campaigns' selection of that fit
`regress refit` fits LIBERO native fits afresh (the path new seeds take) into <old-seed root>/refits, or --out,
and compares fit.npz bytes and READY.json except the retired guard setting, recipe.py sha/guard description,
and their digests. `compare` requires these independent refits for every LIBERO cell; copied fits alone cannot
prove the new fitting path. MT50 native fits are W3 inputs, so compare proves their copying, not W3's fitter.
Both commands exit nonzero on failure; no missing reference or empty check can pass.

    OMP_NUM_THREADS=1 python -m exp.offline_search.rounds.r17.seeds_1023.regress refit --bundle dp_libero
    python -m exp.offline_search.rounds.r17.seeds_1023.regress compare --bundle dp_libero
"""
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path

from . import inputs as ip
from .build import HERE, deployment

STALL = ('READY.json', 'stall.json', 'stall.pkl', 'stall.loeo.csv')
FIT = ('fit.npz', 'READY.json')


def files(a, b, names):
    a, b = Path(a), Path(b)
    return dict(new=str(a), old=str(b), equal={n: (a/n).is_file() and (b/n).is_file() and
                                             filecmp.cmp(a/n, b/n, shallow=False) for n in names})


def compare(bundle, refits=None):
    from exp.offline_search.rounds.r17.longseg_1017.audit import HERE as LS
    from exp.offline_search.rounds.r17.longseg_off_1022.build import c5_refs
    from exp.offline_search.rounds.r17.filter_failed_knob.common import cells
    seed, bench, model = ip.OLD[ip.benchmark(bundle)], ip.benchmark(bundle), ip.model(bundle)
    recal = ip.read(LS/'RECALIBRATION.json') if bundle in ip.LONG else None
    c5 = c5_refs(model, bench) if bundle in ip.LONG else None
    old_cells = cells()
    rows = {}
    for mode in ('reduced', 'full'):
        templates = deployment(bundle, mode)[3]
        study = ip.read(templates/'assets/CACHE_STUDY.json')
        selection = ip.read(templates/'campaign.json')['filter_failed_knob']['selection']
        for g in study['groups']:
            rows.setdefault(g['fit_digest'], []).append(dict(g, mode=mode, campaign=str(templates), selection=selection.get(g['fit_digest'])))
    report = []
    for (suite, size), p in sorted(ip.products(bundle, seed).items()):
        base, lib, n = ip.inputs(bundle, seed), ip.library(bundle, suite, seed), p['native']
        sub = f'size_{size//50 if bench == "mt50" else size}'
        checks = dict(view=files(base/p['view'], lib/'subsets'/sub, ('VIEW.json', 'row_indices.npy')),
                      native_fit=files(base/n['fit'], lib/'fits'/sub, FIT))
        if bench == 'libero':
            checks['native_refit'] = refit_check(bundle, suite, size, refits or ip.root(bundle, seed)/'refits')
        old = rows.get(n['fit_digest'], [])
        kept = sorted(p['summary']['kept_episode_ids'])
        checks['selection'] = dict(cells=kept == sorted(old_cells[model, suite, size]['kept_episode_ids']),
            campaigns={mode: bool(matched := [r for r in old if r['mode'] == mode]) and
                       all(r['selection'] is not None and sorted(r['selection']) == kept for r in matched)
                       for mode in ('reduced', 'full')},
            file=sorted(ip.read((base/n['fit']).parent/'selection'/(n['fit_digest']+'.json'))['kept_episode_ids']) == kept)
        if bundle in ip.LONG:
            q3 = next((v for v in recal['Q3'].values() if v['fit_digest'] == n['fit_digest']), None)
            checks['native_stall'] = files(base/n['source_stall'], q3['source'], STALL) if q3 else 'MISSING old Q3'
            checks['segment_q3'] = files(base/n['stall'], q3['path'], STALL) if q3 else 'MISSING old Q3'
            checks['gaps'] = {level: n['solutions'][level] == recal['solutions'].get(f'{n["fit_digest"]}:filtered:{level}')
                              for level in ip.LEVELS}
            ref = c5[p['store_digest']]
            checks['c5_fit'] = files(base/p['c5']['fit'], ref['fit'], FIT)
            checks['c5_stall'] = files(base/p['c5']['stall'], ref['stall'], STALL)
            checks['c5_selection'] = sorted(ip.read((base/p['c5']['fit']).parent/'selection'/(p['c5']['fit_digest']+'.json'))['kept_episode_ids']) == kept
            checks['c5_gaps'] = p['c5']['gaps'] == n['gaps']
        else:
            stall = next((r['stall'] for r in old if r.get('stall')), None)
            checks['native_stall'] = files(base/n['stall'], stall, STALL) if stall else 'MISSING old stall'
            import yaml
            checks['gaps'] = {f'{mode}/{level}': bool(matched := [r for r in old if r['mode'] == mode and
                              r['group'].rsplit('_', 1)[-1] == level]) and all(
                              n['gaps'].get(level) == r['periodic_gap'] == yaml.safe_load(
                                  (Path(r['campaign'])/'assets'/r['file']).read_text())['recipe']['periodic_gap']
                              for r in matched) for mode in ('reduced', 'full') for level in ip.LEVELS}
        report.append(dict(suite=suite, size=size, fit_digest=n['fit_digest'], checks=checks))
    def ok(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return False
        if isinstance(value, list):
            return bool(value)
        return bool(value) and all(ok(v) for k, v in value.items() if k not in ('new', 'old'))
    result = dict(bundle=bundle, seed=seed, passed=all(ok(r['checks']) for r in report), cells=report)
    out = HERE/'regression'/f'{bundle}_{seed}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(bundle, seed, 'PASS' if result['passed'] else 'FAIL', out, flush=True)
    return result


def refit_check(bundle, suite, size, out):
    seed = ip.OLD['libero']
    lib = ip.library(bundle, suite, seed)
    target = Path(out)/f'{bundle}_{suite}_{size}'
    if not (target/'READY.json').is_file():
        return dict(independent_refit=False)
    new, old = ip.read(target/'READY.json'), ip.read(lib/'fits'/f'size_{size}'/'READY.json')
    for doc in (new, old):
        doc.pop('digest')
        doc['receipt'].pop('digest')
        for key in ('shared_recipe_sha256', 'guard_threshold_source', 'stall_gate'):
            doc['receipt']['caps_knobs'].pop(key, None)
        doc['receipt']['config'].pop('guard', None)
    result = dict(fit_npz=filecmp.cmp(target/'fit.npz', lib/'fits'/f'size_{size}'/'fit.npz', shallow=False),
                  ready_except_guard_provenance=new == old)
    return result


def refit(bundle, suite, size, out):
    if ip.benchmark(bundle) != 'libero':
        raise ValueError('MT50 native fits are produced by W3; refit is for LIBERO only')
    seed = ip.OLD['libero']
    lib, view = ip.library(bundle, suite, seed), ip.view_dir(bundle, suite, seed, size)
    target = Path(out)/f'{bundle}_{suite}_{size}'
    ip.fit_library_view(bundle, size, view, target, ip.native_commit(ip.read(lib/'READY.json')))
    result = refit_check(bundle, suite, size, out)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('compare', 'refit'))
    p.add_argument('--bundle', required=True, choices=ip.BUNDLES)
    p.add_argument('--suite')
    p.add_argument('--size', type=int)
    p.add_argument('--out')
    p.add_argument('--library-root', type=Path, default=ip.LIBRARY)
    p.add_argument('--out-root', type=Path, default=ip.EVAL, help='as given to inputs')
    a = p.parse_args()
    ip.EVAL, ip.LIBRARY = a.out_root, a.library_root
    if a.action == 'compare':
        result = compare(a.bundle, a.out)
        raise SystemExit(0 if result['passed'] else 1)
    else:
        out = a.out or ip.root(a.bundle, ip.OLD['libero'])/'refits'
        passed = True
        for suite in ([a.suite] if a.suite else ip.suites(a.bundle)):
            for size in ([a.size] if a.size else ip.SIZES['libero']):
                passed = all(refit(a.bundle, suite, size, out).values()) and passed
        raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
