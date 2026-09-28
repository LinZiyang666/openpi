"""Same frozen QueryViews: exact scalar B=0 parity and full-cell stuck diagnostics.

GT/recorded execution and precomputed ideation anchors are evaluator-only. No
predicted verdict changes the recorded path. B=2 keeps recorded HIT/MISS history.
"""
import argparse
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm.awm import AWM
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge as K1
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view, equal
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindResult
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge as K7

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/offline_search_store')
FIT = Path('/tmp/k7_guard_fits')
IDEA = HERE.parent/'ideation_A'


def method(key, scale):
    name = f"r4k7_p_{'sp' if key.endswith('spatial') else 'l10'}_{scale}_ph2g.pkl"
    path = FIT/name
    if not path.exists():
        path = FIT/f'verification_{key}_{scale}.pkl'
    with path.open('rb') as f:
        m = pickle.load(f)['method']
    return m


def nested_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: nested_equal(a[k], b[k])
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): nested_equal(x, y)
    else:
        assert a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b)), (a, b)


def bit_equal(a, b):
    equal(a, b)
    for field in ('topk', 'scores', 'action'):
        x, y = getattr(a, field), getattr(b, field)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), field
    assert np.float64(a.confidence).tobytes() == np.float64(b.confidence).tobytes()
    for name in a.extras:
        assert np.float64(a.extras[name]).tobytes() == np.float64(b.extras[name]).tobytes(), name


class MaskedView:
    def __init__(self, q, mask, keys):
        self.q, self.hist_has_vision, self.keys = q, mask, keys
    def __getattr__(self, name):
        return getattr(self.q, name)
    @property
    def hist_key_v0(self): return self.keys[0][:self.step]
    @property
    def hist_key_v1(self): return self.keys[1][:self.step]


def parity(key, scale, episodes):
    m = method(key, scale)
    with open(f'/tmp/k1_blind_fits/mixed_{key}_{scale}.pkl', 'rb') as f:
        reference = pickle.load(f)['method']
    for field in ('cal', 'm_thr', 'c_thr', 'M0', 'M1', 'med_len'):
        nested_equal(getattr(m, field), getattr(reference, field))
    reference.__class__ = MixedJudge
    reference.base.__class__ = AWM
    # Exercise all optional guard/event/burst fields too, using the same fit.
    reports = []
    for mode in ('guard_only', 'events_burst'):
        for x in (m, reference):
            x.events = () if mode == 'guard_only' else ('disp', 'grip')
            x.burst = 0 if mode == 'guard_only' else 3
            x.base.budget = 0
        for arm in ('inf', 'cache'):
            qc = store.QueryCell(ROOT, key+'_'+arm); arrays = api.QueryArrays(qc)
            n = 0; fields = set(); flags = {}; regimes = {}
            for ei in np.linspace(0, len(qc.episodes)-1, episodes, dtype=int):
                e = qc.episodes[ei]; q0 = view(qc, arrays, e['start'])
                for x in (m, reference): x.reset(q0.episode)
                for i in range(e['start'], e['end']):
                    q = view(qc, arrays, i)
                    got = m.query(q); want = reference.query(q)
                    bit_equal(want, got)
                    assert m._s == reference._s
                    assert m._noprog_span == want.extras['noprog_n']
                    assert m.confirmed_stuck(q) == want.extras['stuck_n']
                    n += 1; fields.update(want.extras)
                    for target, field in ((flags, 'os_flags'), (regimes, 'regime')):
                        v = int(want.extras[field]); target[v] = target.get(v, 0)+1
            reports.append(dict(cell=key+'_'+arm, scale=scale, mode=mode, episodes=episodes,
                decisions=n, mismatches=0, compared_fields=sorted(fields), flag_histogram=flags,
                regime_histogram=regimes, full_action_scores_topk_extras_confidence_state_equal=True))
    result = dict(PASS=True, calibration_bit_equal=True, rows=reports)
    (HERE/f'results/parity_{key}_{scale}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def rates(key, scale):
    m = method(key, scale); m.base.budget = 2
    report = []
    for arm in ('inf', 'cache'):
        cell = key+'_'+arm; qc = store.QueryCell(ROOT, cell); arrays = api.QueryArrays(qc)
        with np.load(IDEA/f'anchors_{cell}_{scale}.npz') as z:
            rows, weights, actions = z['rows'], z['weights'], z['action']
        stock = np.zeros(qc.N, np.int32); dense = stock.copy(); fixed = stock.copy()
        vision = np.zeros(qc.N, bool); gap_fixed = stock.copy()
        reasons = {}
        for ei, e in enumerate(qc.episodes):
            q0 = view(qc, arrays, e['start']); m.reset(q0.episode)
            size = e['end']-e['start']; hv = np.zeros(size, bool)
            keys = [np.full((size, len(q0.key_v0)), np.nan, np.float32) for _ in range(2)]
            sc = dc = age = 0
            for i in range(e['start'], e['end']):
                q = view(qc, arrays, i); step = q.step
                motion, cosine = MixedJudge._self_change(m, q)
                sc = sc+1 if step and motion < m.m_thr and cosine >= m.c_thr else 0
                md = m.base.dense_motion(q)
                dc = dc+1 if len(md) and md[-1] < m.base.motion10[q.task_id] else 0
                stock[i], dense[i] = sc, dc
                fixed[i] = m.confirmed_stuck(q)
                assert fixed[i] == sc, (cell, i, fixed[i], sc)
                bq = blind_view(q, age=age, hist_has_vision=hv[:step])
                result = m.blind_step(bq)
                if isinstance(result, BlindResult):
                    age += 1
                else:
                    reasons[result.name] = reasons.get(result.name, 0)+1
                    vision[i] = hv[step] = True; age = 0
                    keys[0][step], keys[1][step] = q.key_v0, q.key_v1
                    masked = MaskedView(q, hv[:step], keys)
                    gap_fixed[i] = m.confirmed_stuck(masked)
                    action = np.zeros((m.base.H, 32), np.float32); action[:, :7] = actions[i]
                    m.base._remember_anchor(q, rows[i], weights[i], action)
                    m._progress(q, int(rows[i, 0]))
            if (ei+1) % 100 == 0: print(cell, scale, ei+1, 'episodes', flush=True)
        for budget, mask, new in ((0, np.ones(qc.N, bool), fixed), (2, vision, gap_fixed)):
            r = dict(cell=cell, scale=scale, budget=budget, episodes=len(qc.episodes), decisions=qc.N,
                     vision=int(mask.sum()), blind=int((~mask).sum()), reasons=reasons if budget else {})
            for name, counts in (('stock', stock), ('k1_dense', dense), ('k7', new)):
                fire = (counts >= m.stuck_thr) & mask
                r[name] = dict(stuck_fires=int(fire.sum()), share_all=float(fire.mean()),
                               share_vision=float(fire.sum()/mask.sum()))
            report.append(r)
        np.savez_compressed(HERE/f'results/counts_{cell}_{scale}.npz', stock=stock, dense=dense,
                            k7=fixed, vision_b2=vision, k7_b2=gap_fixed)
    (HERE/f'results/rates_{key}_{scale}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['parity', 'rates'])
    p.add_argument('--key', required=True); p.add_argument('--scale', type=int, required=True)
    p.add_argument('--episodes', type=int, default=20); a=p.parse_args()
    parity(a.key, a.scale, a.episodes) if a.mode == 'parity' else rates(a.key, a.scale)
