"""Synthetic boundary tests and real-fit calibration, parity, lifecycle checks."""
import copy
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge, core
from exp.offline_search.rounds.r04.k3_cost.method import WristAWM, WristMixedJudge
from exp.offline_search.rounds.r04.k1_blind.wrist import BlindWristMixedJudge
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view, equal
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindResult
from exp.offline_search.rounds.r04.k7_guard.evidence import bit_equal, nested_equal, MaskedView
from .judge import WristVisionConfirmedBlindJudge as Q6
from .common import HERE, ROOT, method, write_json

class NoBaseCamera:
    def __init__(self, q): self.q = q
    def __getattr__(self, name):
        if name in ('key_v0', 'hist_key_v0'):
            raise AssertionError('missing base camera accessed: '+name)
        return getattr(self.q, name)

def synthetic():
    checks = []
    m = Q6(); m.model = 'pi05'; m.m_thr = 1.; m.c_thr = .95
    m.M0 = m.M1 = {0: np.array([3., 4.], np.float32)}
    def trial(name, mask, states, keys, expected):
        n = len(mask); rs = np.zeros((n+1, 32), np.float32); rs[:, 0] = states
        k = np.asarray(keys, np.float32)+m.M1[0]
        hist = k[:-1].copy(); hist[~np.asarray(mask, bool)] = np.nan
        q = NS(step=n, task_id=0, rs=rs[-1], hist_rs=rs[:-1], key_v1=k[-1],
               hist_key_v1=hist, hist_has_vision=np.asarray(mask, bool))
        assert m.confirmed_stuck(NoBaseCamera(q)) == expected, name
        hist[~np.asarray(mask, bool)] = 9999
        assert m.confirmed_stuck(NoBaseCamera(q)) == expected, name
        checks.append(name)
        return q
    trial('step_zero', [], [0], [[1, 0]], 0)
    trial('no_left_anchor', [False]*2, [0]*3, [[1, 0]]*3, 0)
    trial('closed_gap', [True, False, False], [0]*4, [[1, 0]]*4, 3)
    trial('wrist_motion_veto', [True, False], [0]*3, [[1, 0], [1, 0], [0, 1]], 0)
    trial('dense_motion_break_inside_gap', [True, False, False], [0, 2, 2, 2], [[1, 0]]*4, 2)
    trial('strict_motion_threshold', [True], [0, 1], [[1, 0]]*2, 0)
    trial('multiple_intervals', [True, False, True, False], [0]*5, [[1, 0]]*5, 4)
    trial('unbounded_prefix_excluded', [False, False, True, False], [0]*5, [[1, 0]]*5, 2)
    trial('previous_visual_break', [True, False, True, False], [0]*5,
          [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]], 2)
    m.c_thr = 1.
    q = trial('inclusive_cosine_threshold_and_task_centering', [True], [0]*2, [[1, 0]]*2, 1)
    q.hist_has_vision = np.array([], bool)
    try: m.confirmed_stuck(q)
    except api.ContractError: checks.append('misaligned_history_refused')
    else: raise AssertionError('bad history accepted')
    for kw in ({'stuck_guard':'dense'}, {'progress_guard':'noprog_n'}, {'memo_reset_after_miss':True},
               {'m_pct':20}, {'c_pct':90}, {'base':'other'}, {'base_kwargs':{'fit_data':'big'}}):
        try: Q6(**kw)
        except ValueError: checks.append('invalid_config_'+str(kw))
        else: raise AssertionError(kw)
    try: Q6().fit(None, NS(model='groot'))
    except api.SkipCell: checks.append('groot_refused')
    else: raise AssertionError('groot accepted')
    assert (m.base.serving, m.base.budget, m.base.gates) == ('anchor_tail', 1, 'budget_only')
    checks.append('default_tail_budget1_budget_only')
    return checks

def real():
    results = []
    for suite in ('l10', 'spatial'):
        for scale in (50, 500):
            m = method(suite, scale)
            lib = store.LibraryView(ROOT, 'pi05_'+suite, m.libname)
            # Independent refit through unchanged K3 threshold implementation.
            reference = WristMixedJudge()
            reference.model = 'pi05'
            reference._fit_thresholds(NoBaseCamera(lib), m.C)
            for field in ('m_thr', 'c_thr', 'M0', 'M1', 'thr_stats'):
                nested_equal(getattr(m, field), getattr(reference, field))
            stock = MixedJudge(); stock.model='pi05'; stock._fit_thresholds(lib, m.C)
            assert m.m_thr == stock.m_thr and m.fitname == m.libname
            # Frozen calibrated K3 with genuine WristAWM, for B=0 bit parity.
            ref = copy.deepcopy(m); ref.__class__ = WristMixedJudge; ref.base.__class__ = WristAWM
            dense = copy.deepcopy(m); dense.__class__ = BlindWristMixedJudge
            parity = gaps = 0
            for arm in ('inf', 'cache'):
                qc = store.QueryCell(ROOT, f'pi05_{suite}_{arm}'); arrays = api.QueryArrays(qc)
                for ei in np.linspace(0, len(qc.episodes)-1, 10, dtype=int):
                    e = qc.episodes[ei]; q0 = view(qc, arrays, e['start'])
                    m.reset(q0.episode); ref.reset(q0.episode)
                    for i in range(e['start'], e['end']):
                        q = NoBaseCamera(view(qc, arrays, i))
                        a = m.query(q); b = ref.query(q)
                        bit_equal(a, b)
                        assert m._s == ref._s and m._noprog_span == b.extras['noprog_n']
                        assert m.confirmed_stuck(q) == a.extras['stuck_n']
                        parity += 1
                for ei in (0, 499):
                    e = qc.episodes[ei]; q0 = view(qc, arrays, e['start'])
                    m.reset(q0.episode); dense.reset(q0.episode)
                    hv = np.zeros(e['end']-e['start'], bool)
                    keys = [np.full((len(hv), len(q0.key_v1)), np.nan, np.float32) for _ in range(2)]
                    for i in range(e['start'], e['end'], 3):
                        q = view(qc, arrays, i); hv[q.step] = True; keys[1][q.step] = q.key_v1
                        q = NoBaseCamera(MaskedView(q, hv[:q.step], keys))
                        a = m.query(q); b = dense.query(q)
                        equal(a, b, extras=False, confidence=False)
                        assert int(a.extras['os_flags']) & ~5 == int(b.extras['os_flags']) & ~5
                        assert m._noprog_span == dense._noprog_span
                        assert m.confirmed_stuck(q) == a.extras['stuck_n']
                        if q.step:
                            captured = []; original = m._features
                            def capture(*args, **kw):
                                f = original(*args, **kw); captured.append(f); return f
                            m._features = capture
                            again = m.query(q)
                            del m._features
                            assert again.confidence == a.confidence
                            assert captured[-1]['stuck'] == min(m.confirmed_stuck(q), 5)
                        gaps += 1
            qc = store.QueryCell(ROOT, f'pi05_{suite}_cache'); arrays = api.QueryArrays(qc)
            q0 = view(qc, arrays, qc.episodes[0]['start']); q1 = view(qc, arrays, qc.episodes[0]['start']+1)
            m.reset(q0.episode)
            assert m.blind_step(blind_view(q0)).code == 6
            r = m.query(q0)
            b = m.blind_step(blind_view(q1, prev_hit=True))
            assert isinstance(b, BlindResult)
            assert np.array_equal(b.action[:5, :7], r.action[5:10, :7])
            q2 = view(qc, arrays, qc.episodes[0]['start']+2)
            assert m.blind_step(blind_view(q2, prev_hit=True, age=1)).code == 1
            m.query(q0)
            assert m.blind_step(blind_view(q1, prev_hit=False)).code == 6
            assert m.base._anchor is None
            other = next(e for e in qc.episodes if e['task_id'] != q0.task_id)
            changed = view(qc, arrays, other['start'])
            got = m.query(changed); ref.reset(changed.episode); bit_equal(got, ref.query(changed))
            # Both explicitly refitted variants retain their exact metadata/defaults.
            phase = method(suite, scale, 'phase2')
            assert (phase.base.serving, phase.base.budget, phase.base.gates) == ('phase_particles', 2, 'all')
            assert m.c_thr == phase.c_thr and m.m_thr == phase.m_thr
            results.append(dict(suite=suite, scale=scale, parity_queries=parity, gap_queries=gaps,
                calibration_exact=True, base_camera_forbidden=True, tail_exact=True, lifecycle=True,
                c_thr=m.c_thr, stock_c_thr=stock.c_thr, m_thr=m.m_thr))
    return results

def main():
    result = dict(PASS=True, synthetic_checks=synthetic(), real=real())
    write_json(HERE/'results/unit.json', result)
    print(result)

if __name__ == '__main__': main()
