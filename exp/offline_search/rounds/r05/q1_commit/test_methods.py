"""Final-file replay parity, C10 lifecycle/monitor and D1 historical rule tests."""
import copy
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view
from exp.offline_search.rounds.r04.k7_guard.evidence import bit_equal, nested_equal
from exp.offline_search.rounds.r04.k7_guard.judge import VisionConfirmedBlindMixedJudge as K7
from exp.offline_search.rounds.r04.k10_policy_tail.judge import PolicyTailJudge
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge, GraspCheckJudge, contact_alarm, fit_contact, fit_monitor

HERE = Path(__file__).resolve().parent
ROOT = '/home/weiland/trace_runs/offline_search_store'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def fitted(name):
    with open('/tmp/q1_fits/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)['method']


def parity_lifecycle():
    reports = []
    for suite in ('l10', 'spatial'):
        for scale in (50, 500):
            tag = f'r5q1_c10_p_{"sp" if suite == "spatial" else suite}_{scale}'
            life = fitted(tag)
            old, inherited = copy.deepcopy(life), copy.deepcopy(life)
            old.__class__ = PolicyTailJudge
            inherited.policy_tail_gate = 'inherited'
            qc = store.QueryCell(ROOT, f'pi05_{suite}_cache')
            A = api.QueryArrays(qc)
            counts = dict(vision=0, hit_blind=0, inherited_policy=0, eligible=0, served=0, lifecycle_veto=0)
            for ei in (0, 49, 250, 499):
                e = qc.episodes[ei]
                for m in (old, inherited, life):
                    m.reset(view(qc, A, e['start']).episode)
                for off in range(min(24, e['end'] - e['start'] - 1)):
                    q = view(qc, A, e['start'] + off)
                    bit_equal(old.query(q), inherited.query(q))
                    life.query(q)
                    counts['vision'] += 1
                    nxt = view(qc, A, e['start'] + off + 1)
                    bq = blind_view(nxt, prev_hit=True)
                    # Test unchanged ordinary HIT behavior on the same stream.
                    x, y = old.blind_step(bq), inherited.blind_step(bq)
                    nested_equal(vars(x) if hasattr(x, '__dict__') else
                                 [getattr(x, k) for k in x.__slots__],
                                 vars(y) if hasattr(y, '__dict__') else [getattr(y, k) for k in y.__slots__])
                    counts['hit_blind'] += 1
                    # Actual policy MISS at this anchor, dense history retains MISS.
                    hist = np.array(nxt.hist_hit, copy=True)
                    hist[-1] = 0
                    acts = np.array(nxt.hist_a_exec, copy=True)
                    acts[-1] = qc.a_inf[e['start'] + off]
                    bq = blind_view(nxt, prev_hit=False, hist_hit=hist, hist_a_exec=acts, prev_a_exec=acts[-1])
                    for m in (old, inherited, life):
                        m.invalidate_anchor()
                    facade = BlindQueryView(**vars(bq))
                    x, y = old.policy_tail_step(facade), inherited.policy_tail_step(facade)
                    assert type(x) is type(y)
                    nested_equal([getattr(x, k) for k in x.__slots__], [getattr(y, k) for k in y.__slots__])
                    counts['inherited_policy'] += 1
                    life._noprog_span = 100  # cache progress, budget and event gates all reject.
                    life.base.budget = 0
                    life.base.gates = 'all'
                    before = copy.deepcopy((life._vision_progress, life._s, hist, bq.hist_has_vision))
                    z = life.policy_tail_step(bq)
                    counts['eligible'] += 1
                    assert isinstance(z, BlindResult), z
                    assert z.action.tobytes() == policy_tail_chunk(acts[-1]).tobytes()
                    assert z.action[:5].tobytes() == acts[-1, 5:10].tobytes()
                    nested_equal(before, (life._vision_progress, life._s, hist, bq.hist_has_vision))
                    assert life._noprog_span == 100 and life.base._anchor is None
                    assert life.policy_tail_step(bq).code == 6
                    counts['served'] += 1
            q0, q1 = (view(qc, A, qc.episodes[0]['start'] + n) for n in (0, 1))
            def prime():
                life.reset(q0.episode)
                life.query(q0)
                life.invalidate_anchor()
                acts = np.array(q1.hist_a_exec, copy=True)
                acts[-1] = qc.a_inf[qc.episodes[0]['start']]
                return blind_view(q1, prev_hit=False, hist_hit=np.array([0]), hist_a_exec=acts, prev_a_exec=acts[-1])
            changes = [dict(step=0), dict(task_id=999), dict(episode=NS(uid='other')),
                       dict(prev_hit=True), dict(blind_age=1), dict(hist_has_vision=np.array([False])),
                       dict(hist_hit=np.array([1])), dict(executed_steps=4), dict(executed_steps=10),
                       dict(rs=np.full(32, np.nan)), dict(raw_state=np.full(8, np.inf)),
                       dict(prev_a_exec=np.zeros((5, 32), np.float32)), dict(hist_a_exec=np.empty((0, 10, 32))),
                       dict(prev_a_exec=np.full((10, 32), np.nan))]
            for change in changes:
                bq = prime()
                bq.__dict__.update(change)
                assert life.policy_tail_step(bq).code == 6, change
                counts['lifecycle_veto'] += 1
            bq = prime()
            life.reset(q0.episode)  # terminal MISS, reset same external identity
            assert life.policy_tail_step(bq).code == 6
            counts['lifecycle_veto'] += 1
            reports.append(dict(suite=suite, scale=scale, **counts))
    return reports


def guard_gap():
    """Real fitted query after a MISS/tail gap; both guards still force a MISS."""
    m = fitted('r5q1_c10_p_l10_50')
    qc = store.QueryCell(ROOT, 'pi05_l10_cache')
    A = api.QueryArrays(qc)
    q0 = view(qc, A, qc.episodes[0]['start'])
    m.reset(q0.episode)
    r = m.query(q0)
    m.invalidate_anchor()
    acts = np.repeat(qc.a_inf[0][None], 2, axis=0)
    rs = np.repeat(q0.rs[None], 2, axis=0)
    keys = [np.stack((q0.key_v0, np.full_like(q0.key_v0, np.nan))),
            np.stack((q0.key_v1, np.full_like(q0.key_v1, np.nan)))]
    class Gap:
        def __getattr__(self, key):
            return getattr(q0, key)
    q = Gap()
    q.step, q.hist_rs, q.hist_a_exec, q.prev_a_exec = 2, rs, acts, acts[-1]
    q.hist_hit, q.prev_hit, q.hist_has_vision = np.array([0, 1]), True, np.array([True, False])
    q.hist_key_v0, q.hist_key_v1 = keys
    q.hist_exec_hit_flag = q.hist_hit
    m.m_thr, m.c_thr, m.stuck_thr = 1., .95, 2
    assert m.confirmed_stuck(q) == 2
    # Give the prior real anchor the same progress as the next retrieved row.
    z = m.query(q)
    m._vision_progress = [(0, float(m.C.prog[z.topk[0]]), max(int(m.C.ep_len[z.topk[0]]) - 1, 1))]
    m.noprog_n = 3
    z = m.query(q)
    assert z.extras['stuck_n'] == 2 and z.extras['noprog_span'] == 2
    assert int(z.extras['os_flags']) & 9 == 9 and z.extras['os_force_miss'] == 1
    assert q.hist_hit.tolist() == [0, 1] and np.isnan(q.hist_key_v0[1]).all()
    return dict(stuck_n=2, noprog_span=2, flags=int(z.extras['os_flags']))


def historical():
    D = HERE.parent / 'ideation_D'
    sys.path.insert(0, str(D))
    import contact_rule as reference
    reports = []
    for scale, expected_count, expected_episodes in ((50, 223, 73), (500, 47, 24)):
        m = fitted(f'r5q1_d1_p_l10_{scale}')
        lib = store.LibraryView(ROOT, 'pi05_l10', 'current' if scale == 50 else 'bpool_cs')
        ref_table = reference.fit_contact(lib.rs, lib.action, lib.next)
        nested_equal(m.contact, ref_table)
        with np.load(D / f'r4k7_p_l10_{scale}_tail1ug.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        expected = np.load(D / f'contact_r4k7_p_l10_{scale}_tail1ug.npz')['mask']
        raw = np.zeros(len(expected), bool)
        for j in np.flatnonzero(~data['vision']):
            if data['step'][j] < 2:
                continue
            args = (data['state'][j], data['action'][j-2:j], data['rows'][j], data['weights'][j])
            raw[j] = contact_alarm(m.contact, *args)
            assert raw[j] == reference.contact_alarm(ref_table, 'pi05', *args)
        assert np.array_equal(raw, expected) and int(raw.sum()) == expected_count
        # Reconstruct exact historical eligible anchors. Replay actual heads/state
        # through inherited K7 blind_step; mock only new vision retrieval because
        # these R4 JSONL logs contain no full camera keys. Commit an intervention
        # as a real-vision MISS in the replay history, never consume at proposal.
        uncapped, capped, repeats = [], [], 0
        for ei in np.unique(data['ep']):
            ix = np.flatnonzero(data['ep'] == ei)
            ep = NS(uid=f'historical:{ei}', task_id=int(data['task'][ix[0]]))
            m.reset(ep)
            hits = data['hit'][ix].astype(np.int8).copy()
            hv = data['vision'][ix].copy()
            actions = np.zeros((len(ix), 10, 32), np.float32)
            actions[:, :5, :7] = data['action'][ix]
            states = np.zeros((len(ix), 32), np.float32)
            states[:, :8] = data['state'][ix]
            for t, j in enumerate(ix):
                bq = NS(step=t, episode=ep, task_id=ep.task_id, rs=states[t], raw_state=states[t],
                        prev_hit=bool(hits[t-1]) if t else None, prev_a_exec=actions[t-1] if t else None,
                        hist_hit=hits[:t], hist_has_vision=hv[:t], hist_a_exec=actions[:t], hist_rs=states[:t], blind_age=0)
                m._sync_grasp(bq)
                if data['vision'][j]:
                    continue
                assert t >= 1 and data['vision'][j-1] and data['hit'][j-1]
                def anchor():
                    aq = NS(step=t-1, task_id=ep.task_id, episode=ep, rs=states[t-1])
                    m.base._remember_anchor(aq, data['rows'][j], data['weights'][j], actions[t-1])
                    m._noprog_span = 0
                anchor()
                # Independent uncapped method call; cap state restored afterwards.
                saved = m._grasp_used, m._grasp_pending, m._grasp_issued
                m._grasp_used, m._grasp_pending, m._grasp_issued = False, None, None
                u = m.blind_step(bq)
                if isinstance(u, LookReason) and u.code == m.LOOK_CODE:
                    uncapped.append(j)
                m._grasp_used, m._grasp_pending, m._grasp_issued = saved
                anchor()
                z = m.blind_step(bq)
                if isinstance(z, LookReason) and z.code == m.LOOK_CODE:
                    capped.append(j)
                    assert not m._grasp_used
                    assert m.blind_step(bq) == z  # duplicate proposal before vision
                    fake = lambda *_: api.Result(np.array([0]), np.array([1.]), 1., extras={'os_reason': 4.})
                    with patch.object(K7, 'query', fake):
                        for _ in range(2):
                            res = m.query(bq)
                            assert res.extras['os_force_miss'] == 1 and res.extras['os_reason'] == m.MISS_CODE
                            assert not m._grasp_used
                            repeats += 1
                    hits[t], hv[t] = 0, True
            assert sum(data['ep'][j] == ei for j in capped) <= 1
        assert np.array_equal(np.array(uncapped), np.flatnonzero(expected))
        assert len(capped) == expected_episodes
        reports.append(dict(scale=scale, decisions=len(expected), eligible=int((~data['vision']).sum()),
                            alarms=len(uncapped), disagreements=0, capped=len(capped), repeated_query_checks=repeats,
                            lo=float(m.contact['lo']), hi=float(m.contact['hi'])))
    return reports


def monitor():
    references = json.loads((HERE.parent / 'ideation_A/dynamics_fits.json').read_text())
    reports = []
    for suite in ('l10', 'spatial'):
        for scale in (50, 500):
            ln = 'current' if scale == 50 else 'bpool_cs'
            lib = store.LibraryView(ROOT, f'pi05_{suite}', ln)
            mon = fit_monitor(lib)
            ref = next(x for x in references if (x['model'], x['suite'], x['library']) == ('pi05', suite, ln))
            assert np.isclose(mon['q99'], ref['q99'], rtol=1e-6)
            assert np.allclose(mon['B'], ref['B'], rtol=1e-5, atol=1e-7)
            m = fitted(f'r5q1_c10_p_{"sp" if suite == "spatial" else suite}_{scale}')
            m.monitor, m.policy_monitor = 'loeo_xyz99', mon
            qc = store.QueryCell(ROOT, f'pi05_{suite}_cache')
            A = api.QueryArrays(qc)
            q0, q1 = view(qc, A, 0), view(qc, A, 1)
            def prime(offset):
                m.reset(q0.episode)
                m.query(q0)
                m.invalidate_anchor()
                action = np.array(qc.a_inf[0], copy=True)
                bq = blind_view(q1, prev_hit=False, hist_hit=np.array([0]),
                                hist_a_exec=action[None], prev_a_exec=action)
                x = np.r_[1., np.asarray(bq.hist_rs[-1, :8], float), action[:5, :7].mean(0)]
                bq.rs = np.array(bq.rs, copy=True)
                bq.rs[:3] = bq.hist_rs[-1, :3] + x @ mon['B'] + offset
                return bq
            z = m.policy_tail_step(prime(0))
            assert isinstance(z, BlindResult) and z.extras['policy_xyz_residual'] < 1e-4
            assert m.policy_tail_step(prime(100)).code == 5
            reports.append(dict(suite=suite, scale=scale, q99=float(mon['q99']), bytes=208))
    return reports


def edges():
    checks = []
    heads = np.ones((2, 5, 7), np.float32)
    state = np.zeros(8)
    table = dict(lo=0., hi=1., width=np.array([1., 0.]),
                 pattern=np.array([31, 31]), next=np.array([0, 1]))
    assert contact_alarm(table, state, heads, np.array([0, 1]), np.array([.75, .25]))
    assert not contact_alarm(table, state, heads, np.array([0, 1]), np.array([.749, .251]))
    state[6] = .1  # width == .05 must not alarm
    assert not contact_alarm(table, state, heads, np.array([0]), np.array([1.]))
    state[6] = 0
    table['width'][0] = .25
    assert not contact_alarm(table, state, heads, np.array([0]), np.array([1.]))
    table['width'][0], table['pattern'][1] = 1., 0
    assert contact_alarm(table, state, heads, np.array([0, 1]), np.array([.5, .5]))
    assert not contact_alarm(table, state, heads, np.array([0, 1]), np.array([.499, .501]))
    heads[0, 0, 6] = 0
    assert not contact_alarm(table, state, heads, np.array([0]), np.array([1.]))
    checks.append('D1 exact strict/inclusive aperture, mean, mass, consensus and close-sign boundaries')
    for kwargs in ({'policy_tail_gate': 'bad'}, {'monitor': 'bad'}, {'monitor': 'loeo_xyz99'}):
        try:
            CommitJudge(**kwargs)
        except ValueError:
            checks.append(str(kwargs))
        else:
            raise AssertionError(kwargs)
    for cls in (CommitJudge, GraspCheckJudge):
        kw = {'base_kwargs': {'serving': 'anchor_tail', 'budget': 1}}
        try:
            cls(**kw).fit(None, NS(model='groot'))
        except api.SkipCell:
            checks.append(cls.__name__ + ' refuses GR00T')
        else:
            raise AssertionError('GR00T accepted')
    m = fitted('r5q1_d1_p_l10_50')
    ep = NS(uid='test', task_id=0)
    m.reset(ep)
    m._grasp_pending = m._grasp_issued = 2
    q = NS(step=3, episode=ep, task_id=0, hist_hit=np.ones(3), hist_has_vision=np.ones(3, bool),
           hist_a_exec=np.zeros((3, 10, 32)))
    m._sync_grasp(q)
    assert not m._grasp_used and m._grasp_pending is None  # forced proposal did not execute
    checks.append('unexecuted D1 proposal does not consume allowance')
    m._grasp_pending = m._grasp_issued = 2
    q.hist_hit[2] = 0
    m._sync_grasp(q)
    assert m._grasp_used
    checks.append('executed D1 MISS consumes allowance')
    m.reset(ep)
    assert not m._grasp_used and m._grasp_pending is None
    m._grasp_used = True
    q.episode, q.task_id = NS(uid='test', task_id=1), 1
    m._sync_grasp(q)
    assert not m._grasp_used
    checks.append('same uid reset and task change clear D1 state')
    return checks


if __name__ == '__main__':
    result = {}
    for name in ('parity_lifecycle', 'guard_gap', 'historical', 'monitor', 'edges'):
        result[name] = globals()[name]()
        print(name, json.dumps(result[name]), flush=True)
    result['PASS'] = True
    (OUT / 'methods.json').write_text(json.dumps(result, indent=2) + '\n')
