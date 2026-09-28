"""Checks over the replay matrix (run_replays.py): nesting A == B-off, pi05 P1 == C10, GR00T B selftests + ledger.

Every assertion is exact (bytes / integers). Fixed-observation replay: counts are decisions on recorded observation
streams, not rollout outcomes.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp.offline_search.closed_loop.blind import policy_tail_chunk  # noqa: E402
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger  # noqa: E402

CELLS = ('l10_50', 'l10_500', 'sp_50', 'sp_500')
TIMING = {'ts', 'q_us', 'search_us', 'native_us', 'infer_ms', 'pre_ms', 'post_ms', 's1_ms', 's23_ms',
          'blind_prepare_ms', 'blind_output_ms'}
LABEL = {'method', 'tag'}          # arm label / method display name (P1 prefixes "P1__")
BITS = {1: 'stuck', 2: 'terminal_closed', 3: 'overtime', 4: 'no_progress', 5: 'dispersion', 6: 'grip', 7: 'burst'}


def load(run):
    rows = [json.loads(line) for line in (run / f'decisions_{run.name}.jsonl').read_text().splitlines()]
    decs = [r for r in rows if r['ev'] == 'dec']
    starts = [r for r in rows if r['ev'] == 'startup']
    z = dict(np.load(run / 'served.npz'))
    rep = json.loads((run / 'report.json').read_text())
    assert len(decs) == len(z['row']) == rep['decisions']
    return decs, starts, z, rep


def differing(x, y):
    return (x['served'] != y['served']).reshape(len(x['served']), -1).any(1)


def tie_audit(A, model, suite, idx):
    """Every A-vs-B anchor difference is an exact float32 distance tie across the 16th kernel slot."""
    import pickle
    from exp.offline_search.harness import api, store
    from exp.offline_search.rounds.r04.k1_blind.checks import view
    with open(A[3]['fit_artifact'], 'rb') as f:
        m = pickle.load(f)['method']
    qc = store.QueryCell('/home/weiland/trace_runs/offline_search_store', f'{model}_{suite}_cache')
    arrays = api.QueryArrays(qc)
    out = []
    for i, rows_b in idx:
        q = view(qc, arrays, int(A[2]['row'][i]))
        assert q.step == A[2]['step'][i] and q.prev_hit in (None, True)   # A's regimes: step 0 / after HIT only
        T, *_, dt = m._dist(q)
        pos = {int(r): k for k, r in enumerate(T.rows)}
        ra, rb = set(A[0][i]['rows']), set(rows_b)
        a_only, b_only = sorted(ra - rb), sorted(rb - ra)
        assert len(a_only) == len(b_only) >= 1
        tied = {float(dt[pos[r]]) for r in a_only + b_only}
        kth = float(np.sort(dt)[m.k - 1])
        assert len(tied) == 1 and tied == {kth}, (i, tied, kth)
        assert all(b < a for a, b in zip(a_only, b_only))                 # B keeps the lowest rows among ties
        out.append(dict(decision=int(i), step=int(q.step), A_only=a_only, B_only=b_only, tied_distance=kth))
    return out


def nesting(root):
    out = []
    for model in ('pi05', 'groot'):
        for cell in CELLS:
            A, S, B = (load(root / n) for n in (f'N_{model}_{cell}_A', f'T_{model}_{cell}_Astable',
                                                f'N_{model}_{cell}_Boff'))
            za, zs, zb = A[2], S[2], B[2]
            for x in (za, zs):
                for key in ('row', 'ep', 'step', 'vision', 'hit'):
                    assert np.array_equal(x[key], zb[key]), (model, cell, key)
            assert za['served'].dtype == zs['served'].dtype == zb['served'].dtype == np.float32
            # (1) A with the wrapper's tie rule == B with the MISS trigger off: every served chunk, bit for bit.
            assert zs['served'].tobytes() == zb['served'].tobytes(), (model, cell, 'served')
            assert [d['src'] for d in A[0]] == [d['src'] for d in S[0]] == [d['src'] for d in B[0]]
            assert not (~zb['hit']).any() and all(d['src'] in ('cache', 'cache_blind') for d in B[0])
            # (2) the deployed A differs from B-off exactly where it differs from the tie-rule A.
            d_ab, d_as = differing(za, zb), differing(za, zs)
            assert np.array_equal(d_ab, d_as)
            anchors = np.flatnonzero(d_ab & za['vision'])
            blind = np.flatnonzero(d_ab & ~za['vision'])
            assert set(blind.tolist()) <= set((anchors + 1).tolist())          # only the anchor's own blind tail
            suite = 'l10' if cell.startswith('l10') else 'spatial'
            audit = tie_audit(A, model, suite, [(i, B[0][i]['rows']) for i in anchors])
            out.append(dict(model=model, cell=cell, episodes=A[3]['episodes'], decisions=len(za['row']),
                            vision=int(za['vision'].sum()), blind=int((~za['vision']).sum()),
                            A_method=A[3]['method'], B_off_method=B[3]['method'], A_fit=A[3]['fit_artifact'],
                            served_sha256_B_off=B[3]['served_sha256'],
                            tie_rule_A_equals_B_off_bytes=True, vision_mask_equal=True, src_equal=True,
                            deployed_A_differing_decisions=int(d_ab.sum()), deployed_A_differing_anchors=len(anchors),
                            max_abs_difference=float(np.abs(za['served'] - zb['served']).max()),
                            tie_audit=audit))
    return out


def comparable(d, tags):
    x = {k: v for k, v in d.items() if k not in TIMING and k not in LABEL}
    x['winner'] = x['winner'].replace(tags[0], '<tag>').replace(tags[1], '<tag>')
    return x


def parity(root):
    out = []
    for cell in CELLS:
        for arm in ('cache', 'inf'):
            names = [f'P_pi05_{cell}_{arm}_{v}' for v in ('C10', 'P1')]
            C, P = (load(root / n) for n in names)
            assert C[2]['served'].tobytes() == P[2]['served'].tobytes()
            for key in ('row', 'vision', 'hit'):
                assert np.array_equal(C[2][key], P[2][key])
            fields = 0
            for c, p in zip(C[0], P[0]):
                cc, pp = comparable(c, names), comparable(p, names)
                assert cc == pp, (cell, arm, c['uid'], c['step'],
                                  {k: (cc.get(k), pp.get(k)) for k in set(cc) | set(pp) if cc.get(k) != pp.get(k)})
                fields += len(cc)
            src = collections.Counter(d['src'] for d in C[0])
            out.append(dict(cell=cell, replay=f'pi05_{cell.split("_")[0]}_{arm}', decisions=len(C[0]),
                            compared_row_fields=fields, served_sha256=C[3]['served_sha256'], src=dict(src),
                            misses=C[3]['misses'], C10_fit=C[3]['fit_artifact'], P1_method=P[3]['method'],
                            all_nontiming_fields_equal=True))
    return out


def recompute_flags(ex, closed_sign):
    """Independent guard bits from the logged extras (burst 0, events none)."""
    f = 0
    stuck = ex.get('stuck_n', 0.)
    if stuck >= 2:
        f |= 1
    if ex.get('term1', 0.) == 1. and ex.get('gexec', 0.) * closed_sign > 0:
        f |= 2
    ot, lag = ex.get('overtime', float('nan')), ex.get('lag', float('nan'))
    if ot > 1 and lag > 5 and stuck >= 1:
        f |= 4
    if ex.get('noprog_span', ex.get('noprog_n', 0.)) >= 2:
        f |= 8
    return f


def b_checks(decs, starts, z, rep, model, suite, closed_sign):
    """Selftests of one B replay (GR00T B or pi05 C10); returns counts, asserts invariants."""
    n = len(decs)
    served, vision, hit, step, ep = z['served'], z['vision'], z['hit'], z['step'], z['ep']
    src = np.array([d['src'] for d in decs])
    last = np.r_[step[1:] == 0, True]                       # last decision of its episode
    c = collections.Counter()
    reasons, flags_bits, looks = collections.Counter(), collections.Counter(), collections.Counter()
    max_blind = run = 0
    for i, d in enumerate(decs):
        ex = d.get('extras') or {}
        assert d['vision'] == vision[i] and d['hit'] == hit[i] and d['step'] == step[i]
        if step[i] == 0:
            assert vision[i], 'first decision of an episode must be a vision anchor'
            run = 0
        run = 0 if vision[i] else run + 1
        max_blind = max(max_blind, run)
        if vision[i]:
            looks[d['look_reason']] += 1
            flags = int(ex['os_flags'])
            assert flags == recompute_flags(ex, closed_sign), (i, flags, ex)
            for b in range(7):
                if flags >> b & 1:
                    flags_bits[BITS[b + 1]] += 1
            forced = ex['os_force_miss'] == 1.
            assert forced == (flags != 0) and int(ex['os_reason']) == ((flags & -flags).bit_length() if flags else 0)
            # No MISS without a guard, and every fired guard is a MISS (guard_only, no cap/burst/periodic).
            assert (not hit[i]) == forced, (i, d['judge'], ex)
            if not hit[i]:
                assert d['judge'] == f"force:{int(ex['os_reason'])}" and d['src'] == 'policy'
                assert d['miss_k'] is not None and d['s23_ms'] is not None
                reasons[BITS[int(ex['os_reason'])]] += 1
                c['miss'] += 1
                if ex.get('term1') == 1.:
                    c['terminal_row_miss'] += 1
                if last[i]:
                    c['miss_at_episode_end'] += 1
                else:
                    # Committed policy rescue: next decision is the policy tail (rows 5..9 executed), then vision.
                    assert src[i + 1] == 'policy_tail' and not vision[i + 1] and hit[i + 1], (i, src[i + 1])
                    assert served[i + 1].tobytes() == policy_tail_chunk(served[i], 5).tobytes()
                    assert served[i + 1][:5].tobytes() == served[i][5:10].tobytes()
                    c['policy_tail_after_miss'] += 1
            else:
                assert d['judge'] == 'guard_only' and d['src'] == 'cache' and d['miss_k'] is None
            # Terminal guard with each sign, on the same decisions (what the sign fix changes).
            if ex.get('term1') == 1.:
                c['terminal_rows'] += 1
                c['terminal_closed_model_sign'] += int(ex.get('gexec', 0.) * closed_sign > 0)
                c['terminal_closed_opposite_sign'] += int(ex.get('gexec', 0.) * closed_sign < 0)
            branch = 'gap_branch' if ex.get('vision_confirmed_guard') == 1. else 'all_vision_branch'
            c[branch] += 1
            if branch == 'all_vision_branch' and ex.get('term1') == 1. and ex.get('gexec', 0.) != 0.:
                # Stock evaluates terminal-closed as gexec > 0; the model sign flips that bit on these rows.
                c['stock_branch_terminal_bit_' + ('added' if ex['gexec'] * closed_sign > 0 else 'removed')
                  + ('_by_sign' if closed_sign < 0 else '_none')] += 1
        else:
            assert i > 0 and vision[i - 1] and step[i] == step[i - 1] + 1, 'blind must follow a vision anchor'
            assert d['s1_ms'] is None and d['s23_ms'] is None and d['miss_k'] is None and d['hit']
            if src[i] == 'policy_tail':
                assert not hit[i - 1]
                c['policy_tail'] += 1
            else:
                assert src[i] == 'cache_blind' and hit[i - 1], (i, src[i])
                # anchor_tail: this decision executes rows 5..9 of the vision anchor's served chunk.
                assert served[i][:5, :7].tobytes() == served[i - 1][5:10, :7].tobytes()
                c['cache_blind'] += 1
    assert c['policy_tail'] == c['policy_tail_after_miss'] == c['miss'] - c['miss_at_episode_end']
    assert max_blind <= 1
    meta = dict(model=model, suite=suite, cost_ledger=True, client_overrides={'replan_steps': 5, 'resize_size': 256})
    L = ledger(meta, decs, starts)
    nv, nm = int(vision.sum()), int((~hit).sum())
    assert L['vision_decisions'] == nv == rep['stage1_calls'] and L['misses'] == nm and L['decisions'] == n
    tails = [d for d in decs if d['src'] == 'policy_tail']
    T = ledger(meta, tails, starts) if tails else None
    assert T is None or (T['vision_decisions'] == 0 and T['misses'] == 0 and T['total_cost'] == 0)
    owner = (.152 * nv + .848 * nm) / n
    episodes = int((step == 0).sum())
    return dict(decisions=n, episodes=episodes, vision=nv, cache_anchor_hit=int((vision & hit).sum()),
                cache_blind=c['cache_blind'], miss=nm, policy_tail=c['policy_tail'],
                miss_at_episode_end=c['miss_at_episode_end'], max_blind_run=max_blind,
                vision_share=nv / n, miss_share=nm / n, miss_per_vision=nm / nv,
                miss_reason=dict(reasons), vision_flag_bits=dict(flags_bits),
                vision_look_reason={str(k): v for k, v in sorted(looks.items(), key=lambda x: str(x[0]))},
                terminal_rows=c['terminal_rows'], terminal_closed_model_sign=c['terminal_closed_model_sign'],
                terminal_closed_opposite_sign=c['terminal_closed_opposite_sign'],
                terminal_row_miss=c['terminal_row_miss'], all_vision_branch=c['all_vision_branch'],
                gap_branch=c['gap_branch'],
                stock_branch_terminal_bit_changed_by_sign=dict(
                    added=c['stock_branch_terminal_bit_added_by_sign'],
                    removed=c['stock_branch_terminal_bit_removed_by_sign']) if closed_sign < 0 else None,
                owner_ir_per_five_controls=owner, controls_per_source=10,
                k4_ledger=dict(vision_decisions=L['vision_decisions'], misses=L['misses'], v=L['v'], m=L['m'],
                               cost_source=L['cost_source'], ir_per_five_controls=L['ir_per_five_controls'],
                               k_per_miss=L['k_per_miss']),
                k4_tail_ledger=None if T is None else dict(decisions=T['decisions'], vision=T['vision_decisions'],
                                                           misses=T['misses'], total_cost=T['total_cost']))


def groot_b(root):
    out = []
    for cell in CELLS:
        suite = 'l10' if cell.startswith('l10') else 'spatial'
        for arm in ('cache', 'inf'):
            decs, starts, z, rep = load(root / f'G_groot_{cell}_{arm}')
            assert starts[0]['policy_tail_blocks'] == 1 and starts[0]['H'] == 16 and starts[0]['miss_steps'] == 8
            r = b_checks(decs, starts, z, rep, 'groot', suite, -1.)
            out.append(dict(cell=cell, replay=f'groot_{suite}_{arm}', fit=rep['fit_artifact'], **r))
    return out


def pi05_b(root):
    out = []
    for cell in CELLS:
        suite = 'l10' if cell.startswith('l10') else 'spatial'
        for arm in ('cache', 'inf'):
            decs, starts, z, rep = load(root / f'P_pi05_{cell}_{arm}_C10')
            r = b_checks(decs, starts, z, rep, 'pi05', suite, 1.)
            out.append(dict(cell=cell, replay=f'pi05_{suite}_{arm}', fit=rep['fit_artifact'], **r))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE / 'results' / 'replay_checks.json')
    a = p.parse_args()
    result = dict(root=str(a.root), jobs=json.loads((a.root / 'done.json').read_text()))
    assert all(j['rc'] == 0 for j in result['jobs'])
    for name in ('nesting', 'parity', 'groot_b', 'pi05_b'):
        result[name] = globals()[name](a.root)
        print(name, 'PASS', len(result[name]), flush=True)
    result['PASS'] = True
    a.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
