"""Stuck predicates on successful demos and frozen recorded decision paths.

All four predicates see identical states. B0 sees every stored visual key. For
tail / phase2 masks, Q6's actual blind_step/query schedules looks, but recorded
executions and HIT/MISS history remain fixed (no counterfactual rollout). Stock
is an omnivision reference projected onto this common mask; K7 and Q6 only see
real anchor keys. Predicted verdicts never alter the recorded trajectory.
"""
import argparse
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge as K7
from exp.offline_search.rounds.r04.k7_guard.evidence import MaskedView
from exp.offline_search.rounds.r04.k3_cost.method import WristView
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindResult
from .common import HERE, ROOT, method, write_json

NAMES = ('stock_two_camera', 'k1_dense', 'k7_two_camera', 'q6_wrist')

def baselines(m, lib):
    stock = MixedJudge(); stock.model = 'pi05'; stock._fit_thresholds(lib, m.C)
    k7 = K7(); k7.model = 'pi05'
    for name in ('M0', 'M1', 'm_thr', 'c_thr'):
        setattr(k7, name, getattr(stock, name))
    assert m.m_thr == stock.m_thr
    return stock, k7

def allvision(q, m, stock, k7, sc, dc, wc):
    motion, cosine = stock._self_change(q)
    sc = sc+1 if q.step and motion < stock.m_thr and cosine >= stock.c_thr else 0
    md = m.base.dense_motion(q)
    dc = dc+1 if len(md) and md[-1] < m.base.motion10[q.task_id] else 0
    wm, wcos = m._self_change(WristView(q))
    wc = wc+1 if q.step and wm < m.m_thr and wcos >= m.c_thr else 0
    kc = k7.confirmed_stuck(q)
    assert kc == sc and m.confirmed_stuck(q) == wc
    return sc, dc, kc, wc

def summarize(source, suite, scale, counts, mask, episodes, **extra):
    n = len(mask); nv = int(mask.sum())
    result = dict(source=source, suite=suite, scale=scale, decisions=n, episodes=episodes,
                  vision=nv, blind=n-nv, **extra)
    for j, name in enumerate(NAMES):
        fired = (counts[:, j] >= 2) & mask
        result[name] = dict(fires=int(fired.sum()), per_decision=float(fired.mean()),
                           per_vision=float(fired.sum()/nv))
    return result

def run(suite, scale):
    m = method(suite, scale)
    lib = store.LibraryView(ROOT, 'pi05_'+suite, m.libname)
    stock, k7 = baselines(m, lib)
    demo = np.zeros((lib.L, 4), np.int32)
    for task in lib.tasks():
        for ep in np.unique(lib.episode[lib.rows_of_task(task)]):
            rows = np.flatnonzero(lib.episode == ep)
            assert np.all(np.diff(rows) == 1)
            lo, hi = int(rows[0]), int(rows[-1])+1
            sc = dc = wc = 0
            for step, i in enumerate(rows):
                q = NS(step=step, task_id=int(task), rs=lib.rs[i], hist_rs=lib.rs[lo:i],
                    key_v0=lib.key_v0[i], key_v1=lib.key_v1[i],
                    hist_key_v0=lib.key_v0[lo:i], hist_key_v1=lib.key_v1[lo:i])
                sc, dc, kc, wc = allvision(q, m, stock, k7, sc, dc, wc)
                demo[i] = sc, dc, kc, wc
    success = np.asarray(lib.success, bool)
    reports = [summarize('deployed_library_all', suite, scale, demo, np.ones(lib.L, bool),
                        len(np.unique(lib.episode)), schedule='all_vision'),
               summarize('successful_library', suite, scale, demo[success], np.ones(int(success.sum()), bool),
                        len(np.unique(lib.episode[success])), schedule='all_vision')]
    np.savez_compressed(HERE/f'results/counts_library_{suite}_{scale}.npz', counts=demo, success=success)
    for arm in ('inf', 'cache'):
        cell = f'pi05_{suite}_{arm}'; qc = store.QueryCell(ROOT, cell); arrays = api.QueryArrays(qc)
        counts = np.zeros((qc.N, 4), np.int32)
        for e in qc.episodes:
            sc = dc = wc = 0
            for i in range(e['start'], e['end']):
                q = view(qc, arrays, i)
                sc, dc, kc, wc = allvision(q, m, stock, k7, sc, dc, wc)
                counts[i] = sc, dc, kc, wc
        reports.append(summarize(cell, suite, scale, counts, np.ones(qc.N, bool), len(qc.episodes),
                                 schedule='all_vision'))
        saved = dict(allvision=counts)
        for variant in ('tail', 'phase2'):
            m = method(suite, scale, variant)
            masked_counts = counts.copy(); mask = np.zeros(qc.N, bool); reasons = {}
            if arm == 'inf':
                # Every recorded previous outcome is a MISS. Verify lifecycle
                # on each decision; no retrieval rerun is needed for this mask.
                for e in qc.episodes:
                    m.reset(view(qc, arrays, e['start']).episode)
                    for i in range(e['start'], e['end']):
                        q = view(qc, arrays, i)
                        r = m.blind_step(blind_view(q))
                        assert not isinstance(r, BlindResult) and r.code == 6
                        reasons[r.name] = reasons.get(r.name, 0)+1
                mask[:] = True
            else:
                for ei, e in enumerate(qc.episodes):
                    q0 = view(qc, arrays, e['start']); m.reset(q0.episode)
                    size = e['end']-e['start']; hv = np.zeros(size, bool); age = 0
                    keys = [np.full((size, len(q0.key_v1)), np.nan, np.float32) for _ in range(2)]
                    for i in range(e['start'], e['end']):
                        q = view(qc, arrays, i); step = q.step
                        r = m.blind_step(blind_view(q, age=age, hist_has_vision=hv[:step]))
                        if isinstance(r, BlindResult):
                            age += 1
                            continue
                        reasons[r.name] = reasons.get(r.name, 0)+1
                        mask[i] = hv[step] = True; age = 0
                        keys[0][step], keys[1][step] = q.key_v0, q.key_v1
                        mq = MaskedView(q, hv[:step], keys)
                        actual = m.query(mq)
                        masked_counts[i, 2] = k7.confirmed_stuck(mq)
                        masked_counts[i, 3] = m.confirmed_stuck(mq)
                        assert actual.extras['stuck_n'] == masked_counts[i, 3]
                    if (ei+1) % 100 == 0:
                        print(cell, scale, variant, ei+1, flush=True)
            reports.append(summarize(cell, suite, scale, masked_counts, mask, len(qc.episodes),
                schedule=variant, look_reasons=reasons, frozen_recorded_outcomes=True,
                stock_is_omnivision_projection=True))
            saved[variant+'_counts'] = masked_counts; saved[variant+'_mask'] = mask
        np.savez_compressed(HERE/f'results/counts_{cell}_{scale}.npz', **saved)
    footprint = dict(library_path=str(lib.dir), library_rows=lib.L,
        library_episodes=len(np.unique(lib.episode)), library_npy_bytes=sum(p.stat().st_size for p in lib.dir.glob('*.npy')),
        successful_rows=int(success.sum()), successful_episodes=len(np.unique(lib.episode[success])),
        c_thr_wrist=m.c_thr, c_thr_two_camera=stock.c_thr, m_thr=m.m_thr,
        fit_library=m.fitname, deployed_library=m.libname,
        k1_motion10={str(k):v for k,v in m.base.motion10.items()},
        wrist_task_means_from_same_library=True, no_larger_library_borrowing=True)
    result = dict(PASS=True, footprint=footprint, rows=reports)
    write_json(HERE/f'results/evidence_{suite}_{scale}.json', result)
    print(result, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--suite', choices=['l10','spatial'], required=True)
    p.add_argument('--scale', type=int, choices=[50,500], required=True); a = p.parse_args()
    run(a.suite, a.scale)
