"""Regression of seeds_1023/vla on the old subset seed 20261002 (CPU; run before the new seeds' fits are used).

fit.prepare for seed 20261002 on pi05_l10_50 and groot_spatial_200 under a fresh OUT/regress/run_* directory, then:
  subsets   s20261002/subsets/<cell>.{npy,json} == r10/subsets (bytes, all 12 cells of sizes 50/100/200)
  R10 fit   cs/<cell>_cr.pkl: vars(inner) and row_subset == the cache study's ..._cr fit (pickle bytes)
  schedule  row_episodes == old schedule_episodes for ep/task/n_slots/even steps
  Q3        q3/<cell> payload == /nvme/weiland/openpi/r17_q3_port_1020/q3/<cell> (aside from provenance's base_fit
            path); Q3 refitted on the cache-study base itself: stall.pkl bytes and RECEIPT periods equal 1020's;
            PERIODS.json[cell] == 1020 PERIODS.json[cell]
  knob off  fits/<cell>_off.pkl vs 1020 fits/filtered/<cell>_off.pkl: kwargs, levels, caps, filtered cache, rows, Q3
Prints one line per comparison, then REGRESSION_OK or REGRESSION_FAILED (exit 1).

smoke [--seeds ...]: for every new-seed cell, query the knob-off fit along even-step library trajectories
(q3_port_1020/smoke.LibQ, no policy or server): one failed episode's Q3 calls must equal fit's library mask.
Empty-task cells fail: forced extras alone do not prove no retrieval or a policy call at every decision.
SMOKE_OK or SMOKE_FAILED. This is Q3-mask parity, not an end-to-end plugin/policy-tail test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np

from . import fit as F
from .fit import q3, read

OLD = 20261002
CELLS = ('pi05_l10_50', 'groot_spatial_200')
R10_SUBSETS = Path(__file__).resolve().parents[3]/'r10/subsets'
FILTERED_1020 = Path('/home/weiland/projects/openpi_ext/r17_q3_port_1020/fits/filtered')


def same(a, b):
    return pickle.dumps(a, protocol=4) == pickle.dumps(b, protocol=4)


def differing(a, b):
    return sorted(k for k in set(vars(a)) | set(vars(b)) if not same(vars(a).get(k), vars(b).get(k)))


def payload(cell, out):
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    p = StallModel.load(q3.q3_path(cell, out))._payload()
    p['provenance'] = {k: v for k, v in p['provenance'].items() if k != 'q3_port_1020'}
    return p


def report(ok, name, passed, detail=''):
    ok.append(bool(passed)); print('PASS' if passed else 'FAIL', name, detail if not passed else '', flush=True)


def trajectory(method, arr, rows, ep, task):
    from exp.offline_search.harness import api
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.smoke import LibQ
    episode = api.EpisodeView(f'A:smoke:{ep}', '', task, ep % 50, 0, 0)
    method.reset(episode)
    return [method.query(LibQ(arr, rows, i, episode, task)).extras for i in range(0, len(rows), 2)]


def smoke(seeds):
    ok = []
    for seed in seeds:
        d = F.seed_dir(seed)
        for cell in F.cells():
            try:
                F.cell_record(d, cell)
            except ValueError as e:
                report(ok, f'smoke s{seed} {cell}: empty-task fallback unresolved', False, str(e))
                continue
            source = pickle.loads(F.cs_path(d, cell).read_bytes())
            lib, parent = q3.library(cell, source)
            rows_all = np.asarray(source['method'].row_subset)
            arr = {k: np.load(parent/f'{k}.npy', mmap_mode='r') for k in ('key_v0', 'key_v1', 'rs', 'action')}
            forced, _ = q3.forced_mask(q3.stall_model(cell, d), lib)
            method = pickle.loads(F.fit_path(d, cell, 'off').read_bytes())['method']
            failed = sorted(set(np.unique(lib['episode'][~lib['success'].astype(bool)]).tolist()))
            checks = [max(failed, key=lambda e: (sum(v for k, v in forced.items() if k[0] == e), -e))] if failed else []
            for ep in checks:
                local = np.flatnonzero(lib['episode'] == ep); local = local[np.argsort(lib['step'][local])]
                task = int(lib['task_id'][local[0]])
                extras = trajectory(method, arr, rows_all[local], ep, task)
                expected = [forced[(ep, int(s))] for s in lib['step'][local] if s % 2 == 0]
                calls = [bool(x['os_q3_call']) for x in extras]
                report(ok, f'smoke s{seed} {cell} episode {ep}: Q3 calls == library mask ({sum(expected)})', calls == expected)
    print('SMOKE_OK' if all(ok) else 'SMOKE_FAILED', f'{sum(ok)}/{len(ok)}', flush=True)
    return all(ok)


def regress(workers):
    from exp.offline_search.rounds.r15.descent.fits import row_episodes, schedule_episodes
    parent = F.OUT/'regress'; parent.mkdir(parents=True, exist_ok=True)
    # Reusing an earlier fit would let a changed fitter pass without running it.
    root = Path(tempfile.mkdtemp(prefix='run_', dir=parent)); d = root/f's{OLD}'
    print('REGRESSION_ROOT', root, flush=True)
    F.prepare([OLD], workers, CELLS, root)
    ok = []
    check = lambda *a: report(ok, *a)
    for key in ('pi05_l10', 'pi05_spatial', 'groot_l10', 'groot_spatial'):
        for n in F.SIZES:
            for ext in ('npy', 'json'):
                name = f'{key}_{n}.{ext}'
                check(f'subset {name}', (d/'subsets'/name).read_bytes() == (R10_SUBSETS/name).read_bytes())
    for cell in CELLS:
        new, old = pickle.loads(F.cs_path(d, cell).read_bytes()), q3.load_cs(cell, True)
        check(f'R10 fit {cell}: vars(inner) == cache study _cr', same(vars(new['method'].inner), vars(old['method'].inner)),
               differing(new['method'].inner, old['method'].inner))
        check(f'R10 fit {cell}: row_subset, kwargs', np.array_equal(new['method'].row_subset, old['method'].row_subset)
               and new['kwargs'] == old['kwargs'])
        lib, _ = q3.library(cell, new)
        rows = row_episodes(lib['episode'], lib['task_id'], lib['step'])
        anchors, _ = schedule_episodes(cell)
        check(f'ScheduleReplay {cell}: ep/task/n_slots/even steps == old anchors', len(rows) == len(anchors) and all(
            (r.ep, r.task, r.n_slots) == (a.ep, a.task, a.n_slots) and np.array_equal(r.steps, a.steps)
            for r, a in zip(rows, anchors)))
        check(f'Q3 {cell}: payload == 1020', same(payload(cell, d), payload(cell, q3.OUT)))
        on_cs = root/'q3_on_cache_study_base'
        q3.fit_cell(cell, out=on_cs)
        check(f'Q3 {cell} on the cache-study base: stall.pkl bytes == 1020',
               (q3.q3_path(cell, on_cs)/'stall.pkl').read_bytes() == (q3.q3_path(cell)/'stall.pkl').read_bytes())
        check(f'Q3 {cell} on the cache-study base: receipt periods == 1020',
               read(q3.q3_path(cell, on_cs)/'RECEIPT.json')['periods']['q3'] == read(q3.q3_path(cell)/'RECEIPT.json')['periods']['q3'])
        check(f'PERIODS {cell} == 1020', read(d/'PERIODS.json')[cell] == read(q3.OUT/'PERIODS.json')[cell],
               (read(d/'PERIODS.json')[cell], read(q3.OUT/'PERIODS.json')[cell]))
        new, old = pickle.loads(F.fit_path(d, cell, 'off').read_bytes()), pickle.loads((FILTERED_1020/f'{cell}_off.pkl').read_bytes())
        nm, om = new['method'], old['method']
        check(f'knob off {cell}: kwargs, levels, caps', new['kwargs'] == old['kwargs'] and nm.task_levels == om.task_levels
               and same(nm.knob_settings, om.knob_settings))
        check(f'knob off {cell}: filtered cache vars(inner.base), row_subset', same(vars(nm.inner.base), vars(om.inner.base))
               and np.array_equal(nm.row_subset, om.row_subset), differing(nm.inner.base, om.inner.base))
        check(f'knob off {cell}: Q3 metric and tasks', same(nm.inner.stall_model.metric, om.inner.stall_model.metric)
               and same(nm.inner.stall_model.tasks, om.inner.stall_model.tasks))
    print('REGRESSION_OK' if all(ok) else 'REGRESSION_FAILED', f'{sum(ok)}/{len(ok)}', flush=True)
    return all(ok)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('regress', 'smoke'))
    p.add_argument('--workers', type=int, default=4); p.add_argument('--seeds', type=int, nargs='+', default=list(F.SEEDS))
    a = p.parse_args()
    sys.exit(0 if (regress(a.workers) if a.action == 'regress' else smoke(a.seeds)) else 1)


if __name__ == '__main__':
    main()
