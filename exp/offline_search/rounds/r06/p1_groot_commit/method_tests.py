"""Method-level checks of P1 GrootCommitJudge on the store (CPU, fixed recorded QueryViews).

  mirror      GR00T cache cells, all-vision stale regime: C10's own code (the fitted P1 object re-classed to
              CommitJudge) fed the GR00T history with the gripper column mirrored (-x: GR00T openness -> pi0.5
              positive-close) returns exactly P1's Result on the original history, except gexec (negated).
              Also counts what stock-sign C10 would do on the unmirrored GR00T history.
  lifecycle   GR00T policy_tail_step: exact 16-row tail of the MISS chunk (rows 5..9 executed next), one use,
              bypasses cache gates, vetoes on every lifecycle break.
  refusals    constructor / fit contract.
  fits        P1 fitted on pi05 == deployed C10 fits (calibration, thresholds, base state); GR00T thresholds.
  resign      unit cases of the terminal-bit re-derivation.
"""
import copy
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view
from exp.offline_search.rounds.r04.k7_guard.evidence import bit_equal, nested_equal
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge, CLOSED_SIGN

HERE = Path(__file__).resolve().parent
ROOT = '/home/weiland/trace_runs/offline_search_store'
FITS = Path('/home/weiland/trace_runs/os_closed_loop/r06_paper/fits')
C10_FITS = Path('/home/weiland/trace_runs/os_closed_loop/r05_q1/fits')
CELLS = (('l10', 50), ('l10', 500), ('spatial', 50), ('spatial', 500))
SHORT = {'l10': 'l10', 'spatial': 'sp'}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def p1(suite, scale):
    return load(FITS / f'r6p1_c10_g_{SHORT[suite]}_{scale}.pkl')['method']


class Mirrored:
    """QueryView facade: executed gripper column mirrored; every other attribute is the original."""
    def __init__(self, q):
        self._q = q
        pa, ha = q.prev_a_exec, q.hist_a_exec
        self.prev_a_exec = None if pa is None else np.array(pa, copy=True)
        self.hist_a_exec = np.array(ha, copy=True)
        if pa is not None:
            self.prev_a_exec[:, 6] *= -1
        self.hist_a_exec[..., 6] *= -1

    def __getattr__(self, name):
        return getattr(self._q, name)


def mirror(episodes=100):
    reports = []
    for suite, scale in CELLS:
        m = p1(suite, scale)
        c10 = copy.deepcopy(m)
        c10.__class__ = CommitJudge                  # C10's code on the same fitted state
        stock = copy.deepcopy(c10)                   # C10 code, unmirrored GR00T history (wrong sign)
        qc = store.QueryCell(ROOT, f'groot_{suite}_cache')
        A = api.QueryArrays(qc)
        n = changed = fires_p1 = fires_stock = 0
        for ei in np.linspace(0, len(qc.episodes) - 1, episodes, dtype=int):
            e = qc.episodes[ei]
            q0 = view(qc, A, e['start'])
            for x in (m, c10, stock):
                x.reset(q0.episode)
            for i in range(e['start'], e['end']):
                q = view(qc, A, i)
                assert q.prev_hit in (None, True)    # recorded cache cell: stale regime after step 0
                got, ref, wrong = m.query(q), c10.query(Mirrored(q)), stock.query(q)
                assert got.extras['gexec'] == -ref.extras['gexec']
                ref.extras = {**ref.extras, 'gexec': got.extras['gexec']}
                bit_equal(ref, got)                  # topk, scores, action, confidence, every extras value
                nested_equal(m._s, c10._s)
                assert m._noprog_span == c10._noprog_span
                fires_p1 += int(int(got.extras['os_flags']) & 2 != 0)
                fires_stock += int(int(wrong.extras['os_flags']) & 2 != 0)
                changed += int(got.extras['os_force_miss'] != wrong.extras['os_force_miss'])
                n += 1
        reports.append(dict(suite=suite, scale=scale, episodes=episodes, decisions=n, mismatches=0,
                            terminal_fires_p1=fires_p1, terminal_fires_stock_sign=fires_stock,
                            force_miss_verdicts_changed_by_fix=changed))
        print('mirror', json.dumps(reports[-1]), flush=True)
    return reports


def lifecycle():
    reports = []
    for suite, scale in CELLS:
        m = p1(suite, scale)
        qc = store.QueryCell(ROOT, f'groot_{suite}_inf')
        A = api.QueryArrays(qc)
        counts = dict(served=0, one_use=0, vetoes=0)
        for ei in (0, 123, 250, 499):
            e = qc.episodes[ei]
            m.reset(view(qc, A, e['start']).episode)
            for off in range(min(20, e['end'] - e['start'] - 1)):
                q = view(qc, A, e['start'] + off)
                m.query(q)
                nxt = view(qc, A, e['start'] + off + 1)
                chunk = np.asarray(qc.a_inf[e['start'] + off])          # the policy chunk executed on the MISS
                hist = np.array(nxt.hist_hit, copy=True); hist[-1] = 0
                acts = np.array(nxt.hist_a_exec, copy=True); acts[-1] = chunk
                bq = BlindQueryView(**vars(blind_view(nxt, prev_hit=False, hist_hit=hist, hist_a_exec=acts,
                                                      prev_a_exec=acts[-1])))
                m.invalidate_anchor()                                    # the plugin path after a MISS
                keep = (m._noprog_span, m.base.budget, m.base.gates)
                m._noprog_span, m.base.budget, m.base.gates = 100, 0, 'all'   # cache gates would all veto
                z = m.policy_tail_step(bq)
                assert isinstance(z, BlindResult), z
                assert z.action.shape == (16, 32) and z.action.dtype == chunk.dtype
                assert z.action.tobytes() == policy_tail_chunk(chunk, 5).tobytes()
                assert z.action[:5].tobytes() == chunk[5:10].tobytes()      # all 32 columns, rows 5..9
                assert z.extras['policy_tail'] == 1. and z.extras['policy_anchor_step'] == off
                assert m.base._anchor is None                            # cache source stays invalid
                assert m.policy_tail_step(bq) == LookReason(6, 'policy_tail_lifecycle')
                m._noprog_span, m.base.budget, m.base.gates = keep
                counts['served'] += 1
                counts['one_use'] += 1
        q0, q1 = (view(qc, A, qc.episodes[0]['start'] + k) for k in (0, 1))
        chunk = np.asarray(qc.a_inf[qc.episodes[0]['start']])

        def prime():
            m.reset(q0.episode)
            m.query(q0)
            m.invalidate_anchor()
            acts = np.array(q1.hist_a_exec, copy=True); acts[-1] = chunk
            return blind_view(q1, prev_hit=False, hist_hit=np.array([0]), hist_a_exec=acts, prev_a_exec=acts[-1])
        changes = [dict(step=0), dict(task_id=999), dict(episode=NS(uid='other')), dict(prev_hit=True),
                   dict(blind_age=1), dict(hist_has_vision=np.array([False])), dict(hist_hit=np.array([1])),
                   dict(executed_steps=4), dict(executed_steps=10), dict(rs=np.full(8, np.nan)),
                   dict(raw_state=np.full(8, np.inf)), dict(prev_a_exec=np.zeros((9, 32), np.float32)),
                   dict(hist_a_exec=np.empty((0, 16, 32))), dict(prev_a_exec=np.full((16, 32), np.nan))]
        for change in changes:
            bq = prime()
            bq.__dict__.update(change)
            assert m.policy_tail_step(bq).code == 6, change
            counts['vetoes'] += 1
        bq = prime()
        m.reset(q0.episode)                           # MISS at the end of an episode, same identity reset
        assert m.policy_tail_step(bq).code == 6
        counts['vetoes'] += 1
        reports.append(dict(suite=suite, scale=scale, **counts))
        print('lifecycle', json.dumps(reports[-1]), flush=True)
    return reports


def refusals():
    base = {'base_kwargs': {'lib': 'current', 'kref': 5, 'serving': 'anchor_tail', 'budget': 1,
                            'gates': 'budget_only'}, 'policy_tail_gate': 'lifecycle'}
    checks = []
    for bad in (dict(burst=2), dict(policy_tail_gate='inherited'), dict(monitor='loeo_xyz99'),
                dict(stuck_guard='dense'), dict(progress_guard='noprog_n'),
                dict(base_kwargs={**base['base_kwargs'], 'budget': 2}),
                dict(base_kwargs={**base['base_kwargs'], 'serving': 'phase_particles'})):
        try:
            GrootCommitJudge(**{**base, **bad})
        except ValueError:
            checks.append(f'refused {bad}')
        else:
            raise AssertionError(bad)
    m = GrootCommitJudge(**base)
    try:
        m.fit(None, NS(model='other'))
    except api.SkipCell:
        checks.append('unknown model skipped before fit')
    else:
        raise AssertionError('unknown model accepted')
    for cls in (CommitJudge,):
        try:
            cls(**base).fit(None, NS(model='groot'))
        except api.SkipCell:
            checks.append('C10 CommitJudge still refuses GR00T (unchanged)')
    assert CLOSED_SIGN == {'pi05': 1.0, 'groot': -1.0}
    return checks


def fits():
    reports = []
    ctx = lambda cell: api.Context(root=Path(ROOT), cell=cell, seed=0, scratch=Path('/tmp/p1_method_scratch') / cell)
    for suite, scale in CELLS:
        name = f'r5q1_c10_p_{SHORT[suite]}_{scale}'
        blob = load(C10_FITS / f'{name}.pkl')
        c10 = blob['method']
        m = GrootCommitJudge(**blob['kwargs'])
        cell = f'pi05_{suite}_cache'
        c = ctx(cell)
        c.scratch.mkdir(parents=True, exist_ok=True)
        m.prof = api.NULL_PROFILER
        m.fit(store.LibraryView(ROOT, f'pi05_{suite}', 'current'), c)
        for field in ('cal', 'm_thr', 'c_thr', 'M0', 'M1', 'med_len', 'disp_thr', 'k', 'T'):
            nested_equal(getattr(m, field), getattr(c10, field))
        for field in ('blind_next', 'blind_rs', 'blind_event', 'blind_terminal', 'motion10', 'state_scale_by_task'):
            nested_equal(getattr(m.base, field), getattr(c10.base, field))
        assert m.name == 'P1__' + c10.name and m.closed_sign == 1.
        g = p1(suite, scale)
        reports.append(dict(suite=suite, scale=scale, pi05_fit_equal_to_deployed_c10=True,
                            pi05=dict(m_thr=c10.m_thr, c_thr=c10.c_thr, lib=c10.base.cand_name, L=int(len(c10.C.nxt))),
                            groot=dict(m_thr=g.m_thr, c_thr=g.c_thr, lib=g.base.cand_name, L=int(len(g.C.nxt)),
                                       closed_sign=g.closed_sign, H=int(g.base.H),
                                       med_len={str(k): v for k, v in g.med_len.items()},
                                       motion10={str(k): v for k, v in g.base.motion10.items()},
                                       calibration={k: dict(n=v.get('n'), corr_z_err=v.get('corr_z_err'),
                                                            borrowed_stale=v.get('borrowed_stale'))
                                                    for k, v in g.fit_info['calibration'].items() if k in ('0', '1', '2')},
                                       thr_stats=g.fit_info['thr_stats'])))
        print('fits', json.dumps(reports[-1]['groot'] | {'suite': suite, 'scale': scale}), flush=True)
    return reports


def resign():
    m = p1('spatial', 50)
    ep = NS(uid='unit', task_id=0)
    cases = []
    for flags, term1, gexec, guards, want_flags in ((2, 1., 1., True, 0), (0, 1., -1., True, 2), (3, 1., 1., True, 1),
                                                    (8, 1., -1., True, 10), (0, 1., -1., False, 0),
                                                    (2, 0., 1., True, 0), (1, 0., -1., True, 1)):
        m.reset(ep)
        m._s['flag'] = [int(flags != 0)]
        m.guards = guards
        ex = dict(os_force_miss=float(flags != 0), os_reason=float((flags & -flags).bit_length() if flags else 0),
                  os_flags=float(flags), os_phase=0., term1=term1, gexec=gexec, burst_left=0.)
        res = api.Result(np.array([0]), np.array([1.]), .5, action=np.zeros((16, 32), np.float32), extras=ex)
        out = m._resign_terminal(res)
        reason = (want_flags & -want_flags).bit_length() if want_flags else 0
        assert out.extras['os_flags'] == want_flags and out.extras['os_reason'] == reason
        assert out.extras['os_force_miss'] == float(reason != 0) and m._s['flag'][-1] == int(want_flags != 0)
        assert out.confidence == .5 and list(out.extras) == list(ex)
        cases.append(dict(flags=flags, term1=term1, gexec=gexec, guards=guards, out=want_flags))
    m.guards = True
    m.reset(ep)
    m._s['burst_end'] = 5
    try:
        m._resign_terminal(api.Result(np.array([0]), np.array([1.]), .5, extras=dict(os_flags=0., term1=1., gexec=-1.)))
    except api.ContractError:
        cases.append('burst state refused')
    else:
        raise AssertionError('burst state accepted')
    return cases


if __name__ == '__main__':
    which = sys.argv[1:] or ['refusals', 'resign', 'lifecycle', 'fits', 'mirror']
    result = {}
    for name in which:
        result[name] = globals()[name]()
        print(name, 'PASS', flush=True)
    out = HERE / 'results' / f"method_tests_{'_'.join(which)}.json"
    out.write_text(json.dumps(dict(PASS=True, **result), indent=1, default=float) + '\n')
