"""Q3 port (10-20), CPU only: the 16 Q3 fits and the Q3-mask re-solve of the R15/R16 level periods.

One Q3 per (model, suite, library size), shared by the original and success-filtered library and by
full and reduced steps: Q3 uses only successful library episodes, the filter removes only failed
episodes' candidate rows, and full/reduced fits share one base. The base is the cache study's
loadable no-gate corrected cache (fa_cs_*_r ..._cr/fit.pkl). StallModel.fit is the frozen R6 recipe
(successful-episode LOEO, q90 residual / q10 advance, L=min(H,2R)=10, library step*5 controls).
Periods: the selected library rows' episodes and even-step anchors (R15 fits.row_episodes) with the Q3
LOEO forced-call mask and R6-C-v2's one-anchor cooldown (longseg_1017/recalibrate.episodes) as the guard;
R11 ScheduleReplay.solve unchanged. Writes /nvme/weiland/openpi/r17_q3_port_1020 (fit_cell/summarize take
another output root and base fit, e.g. the new subset seeds of seeds_1023/vla).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parent
KNOB = HERE.parent
REPO = HERE.parents[5]
OUT = Path('/nvme/weiland/openpi/r17_q3_port_1020')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
CS = KNOB/'cache_study_1019/runs'
CS_NAMES = {'pi05': 'fa_cs_pi_r', 'groot': 'fa_cs_gr_r'}
CELLS = tuple(f'{m}_{s}_{n}' for m in ('pi05', 'groot') for s in ('l10', 'spatial') for n in (50, 100, 200, 500))
LEVELS = {'ir45': .45, 'ir40': .40, 'ir32': .32, 'ir25': .25}
ESTIMATE = ('delta_hat', 'phase_hat', 'distance_hat', 'spread_hat')


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=float) + '\n'); tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def short(cell):
    _, suite, size = cell.split('_')
    return ('l' if suite == 'l10' else 's') + size


def cs_fit(cell, corrector):
    name = CS_NAMES[cell.split('_')[0]]
    return CS/name/'definitions'/f'{name}_{short(cell)}_{"cr" if corrector else "c"}'/'fit.pkl'


def load_cs(cell, corrector=True):
    return pickle.loads(cs_fit(cell, corrector).read_bytes())


def library(cell, blob):
    """Parent-library labels of the cell's selected rows, in the fit's local row order."""
    model, suite, _ = cell.split('_')
    parent = STORE/'library'/f'{model}_{suite}'/blob['kwargs']['library']
    rows = np.asarray(blob['method'].row_subset)
    out = {k: np.asarray(np.load(parent/f'{k}.npy', mmap_mode='r')[rows]) for k in ('task_id', 'episode', 'step', 'success')}
    assert np.array_equal(out['episode'], blob['method'].inner.lib_ep), f'{cell}: row/episode order differs from the fit'
    out['manifest'] = read(parent/'manifest.json')
    return out, parent


def q3_path(cell, out=OUT):
    return Path(out)/'q3'/cell


def stall_model(cell, out=OUT):
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    return StallModel.load(q3_path(cell, out))


def forced_mask(model, lib):
    """Q3 LOEO forced calls at every even-step anchor of every selected episode (lottery 0, one-anchor cooldown)."""
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_bridge import call_probability
    ep, step = lib['episode'], lib['step']
    forced, states = {}, {}
    for key, table in model.metric['tasks'].items():
        pos = {int(r): i for i, r in enumerate(table['rows'])}
        td = model.tasks[key]
        known = {(int(r['episode']), int(r['end_control'])): r for r in td['calibration']}
        for e in np.unique(ep[table['rows']]).tolist():
            rows = np.flatnonzero(ep == e)
            rows = rows[np.argsort(step[rows], kind='stable')]
            window, cool = [], False
            for row in rows[step[rows] % 2 == 0]:
                s, state = int(step[row]), 'inactive'
                if int(row) not in pos:
                    window = []
                else:
                    regime = 'early' if s == 0 and 'early' in table['codes'] else 'main'
                    window.append(((np.asarray(table['codes'][regime][pos[int(row)]], np.float64), regime), s*5))
                    if len(td['references']) >= 4 and len(window) >= td['W'] + 1:
                        win = window[-td['W']-1:]; span = win[-1][1] - win[0][1]
                        est = known.get((e, s*5))
                        if est is None or est['window_span'] != span:
                            est = model._estimate(key, [w for w, _ in win], span, exclude=e)
                        state = model.calibrated_status(key, {n: est[n] for n in ESTIMATE}, span)['state']
                call = call_probability(state, cool, 0.) == 1.
                forced[(e, s)], states[(e, s)] = call, state
                cool = call
    return forced, states


def solve_periods(model_name, lib, forced):
    from exp.offline_search.rounds.r15.descent.fits import row_episodes
    from exp.offline_search.rounds.r11.knob.calibration import ScheduleReplay
    eps = [NS(**vars(e), guard=np.array([forced[(e.ep, int(s))] for s in e.steps], bool))
           for e in row_episodes(lib['episode'], lib['task_id'], lib['step'])]
    sched = ScheduleReplay(model_name, eps)
    row = dict(N=int(sched.N), V=int(sched.V), G=int(sched.G), forced_per_anchor=float(sched.G/sched.V),
               off_pred_ir=float(sched.ir('off', 0.)))
    for level, target in LEVELS.items():
        x, ir, status = sched.solve('periodic', target)
        row[level] = dict(setting=round(x, 4), unrounded_setting=float(x), solve_status=status, pred_ir=float(ir),
                          pred_ir_rounded=float(sched.ir('periodic', round(x, 4))))
    return row


def fit_cell(cell, out=OUT, base_fit=None):
    """Q3 + periods of one cell. base_fit: an R10Recipe whose inner is the fitted cache (default the cache
    study's ..._cr fit); out: the output root (q3/<cell>/). An existing stall.pkl under out is reused."""
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    started = time.time()
    model_name = cell.split('_')[0]
    base_fit = Path(base_fit or cs_fit(cell, True))
    blob = pickle.loads(base_fit.read_bytes())
    lib, parent = library(cell, blob)
    target = q3_path(cell, out)
    if not (target/'stall.pkl').exists():
        fitted = StallModel.fit(lib, metric=blob['method'].inner, commit_controls=min(int(lib['manifest']['H']), 2*int(lib['manifest']['exec_steps'])))
        fitted.provenance.update(q3_port_1020=dict(cell=cell, base_fit=str(base_fit), parent_library=str(parent),
                                 clock='plugin decision step * 5 controls; one anchor per C10 segment head'))
        from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import _digest
        fitted.fingerprint = _digest(fitted._payload())
        fitted.save(target)
    fit_s = time.time() - started
    model = StallModel.load(target)
    forced, states = forced_mask(model, lib)
    tasks = {}
    task_of = dict(zip(lib['episode'].tolist(), lib['task_id'].tolist()))
    for key, td in sorted(model.tasks.items(), key=lambda kv: int(kv[0])):
        anchors = [k for k in forced if int(task_of[k[0]]) == int(key)]
        tasks[key] = dict(E=td['E'], K=td['K'], W=td['W'], T_med=td['T_med'], reference_episodes=len(td['references']),
                          active=len(td['references']) >= 4, calibration_windows=len(td['calibration']),
                          anchors=len(anchors), forced=int(sum(forced[k] for k in anchors)),
                          confirmed_states=int(sum(states[k] == 'slow_confirmed' for k in anchors)),
                          ambiguous_states=int(sum(states[k] == 'slow_ambiguous' for k in anchors)))
    success = {int(e) for e in np.unique(lib['episode'][lib['success'].astype(bool)])}
    anchors = len(forced); calls = int(sum(forced.values()))
    succ_anchors = [k for k in forced if k[0] in success]
    receipt = dict(cell=cell, q3=str(target), fingerprint=model.fingerprint, base_fit=str(base_fit),
                   parent_library=str(parent), selected_rows=len(lib['episode']), episodes=len(np.unique(lib['episode'])),
                   successful_episodes=len(success), commit_controls=model.commit_controls, exec_steps=model.exec_steps,
                   inactive_tasks=[k for k, v in tasks.items() if not v['active']], tasks=tasks,
                   anchors=anchors, forced_calls=calls, forced_per_anchor=calls/anchors,
                   forced_per_anchor_successful_episodes=sum(forced[k] for k in succ_anchors)/max(len(succ_anchors), 1),
                   forced_per_anchor_failed_episodes=sum(forced[k] for k in forced if k[0] not in success)/max(anchors-len(succ_anchors), 1),
                   periods=dict(q3=solve_periods(model_name, lib, forced)), fit_seconds=fit_s, total_seconds=time.time()-started)
    dump(target/'RECEIPT.json', receipt)
    print('Q3_FIT', cell, f'rate={receipt["forced_per_anchor"]:.4f}', 'inactive', receipt['inactive_tasks'],
          f'{receipt["total_seconds"]:.0f}s', flush=True)
    return receipt


def summarize(receipts, out=OUT):
    """Q3_FITS.json and PERIODS.json under out for any set of cells; other cells already there are kept."""
    out = Path(out)
    fits = read(out/'Q3_FITS.json') if (out/'Q3_FITS.json').exists() else dict(cells=[], receipts={})
    periods = read(out/'PERIODS.json') if (out/'PERIODS.json').exists() else {}
    done = {r['cell'] for r in receipts}
    rows = [row for row in fits['cells'] if row['cell'] not in done]
    for r in receipts:
        q = r['periods']['q3']
        rows.append(dict(cell=r['cell'], forced_per_anchor=r['forced_per_anchor'], inactive_tasks=r['inactive_tasks'],
                         reference_episodes={k: v['reference_episodes'] for k, v in r['tasks'].items()},
                         off_pred_ir_q3=q['off_pred_ir'],
                         levels={l: dict(q3=q[l]['setting'], q3_status=q[l]['solve_status'], q3_pred_ir=q[l]['pred_ir_rounded'])
                                 for l in LEVELS}))
        periods[r['cell']] = {l: q[l]['setting'] for l in LEVELS}
    dump(out/'Q3_FITS.json', dict(cells=rows, receipts={**fits['receipts'], **{r['cell']: str(q3_path(r['cell'], out)/'RECEIPT.json') for r in receipts}}))
    dump(out/'PERIODS.json', periods)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', nargs='*', default=list(CELLS)); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(a.workers, len(a.cells))) as pool:
        receipts = list(pool.map(fit_cell, a.cells))
    summarize(receipts)


if __name__ == '__main__':
    main()
