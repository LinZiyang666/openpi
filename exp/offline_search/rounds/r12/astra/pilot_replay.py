"""Fixed-state replay of accepted B off recordings; no rollout or refitting.

Recomputes legal camera proposals from the existing pilot artifact, propagates
camera memory and guard progress, but keeps recorded states and look slots.
"""
import argparse
import json
import pickle
from types import SimpleNamespace as NS

import numpy as np

from pilot_analysis import CELLS, HERE, PILOTS, read_arm, sha
from exp.offline_search.closed_loop.lookhalf_math import atlas_plan, correct

EXECUTION_ASSUMPTION = ('Off runs disable audit; successful recorded five-control '
                       'blocks assumed acknowledged for optimistic shadow only.')


def decisions_for(cell):
    run = PILOTS / f'r12_lookhalf_fix1_{cell}' / 'runs' / f'r12_lh_{cell}_off'
    accepted = {}
    for line in (run / 'client/journal.jsonl').read_text().splitlines():
        a = json.loads(line)
        if a.get('accepted') and a.get('status') in ('done', 'failed'):
            accepted[a['task_uid']] = a
    decisions = {}
    for p in sorted(run.glob('server_*/decisions_*.jsonl')):
        for line in p.open():
            d = json.loads(line)
            if d.get('ev') == 'dec' and d.get('uid') in accepted and d.get('attempt', 1) == accepted[d['uid']].get('attempt', 1):
                key = (d['uid'], d['step'])
                assert key not in decisions
                decisions[key] = d
    return decisions


def eligibility(d, pp, prev):
    tail = bool(prev and not prev['hit'] and d['step'] == prev['step']+2
                and pp.get('source', pp.get('src')) == 'policy_tail'
                and pp.get('extras', {}).get('policy_tail_lifecycle') == 1
                and pp.get('exec_ok') and pp.get('previous_executed_steps') == 5)
    # Off recordings disable execution_audit. This is an optimistic shadow
    # conditioned on otherwise-successful recorded blocks being acknowledged;
    # it does not waive the real adapter's required audit.
    return bool(d.get('exec_ok') and d.get('previous_executed_steps') == 5
                and (d['look_reason'] not in (6, 7, 8) or (d['look_reason'] == 6 and tail)))


def proposal(m, r, prev):
    b, fit, C = m.inner.base, m.half_fit, m.inner.C
    task, step, sig = r['task'], r['step'], fit['sigma']
    q = NS(task_id=task, step=step)
    x = np.r_[prev['p0'], r['p1'], r['rs'], r['rs']-prev['rs']].astype(np.float32)
    h = m._query_metric('astra_h', task, x, q)
    ha, _ = correct(h, fit['heads']['astra_h'][task], fit['scales']['astra_h'], x, step, task, sig)
    w = m._query_metric('astra_w', task, x[64:], q)
    wa, _ = correct(w, fit['heads']['astra_w'][task], fit['scales']['astra_w'], x[64:], step, task, sig)
    p, past = fit['P'], fit['past2']
    delta = np.einsum('k,kd->d', h['weights'], p[h['rows'], :64]-p[past[h['rows']], :64])
    xf = np.r_[prev['p0']+delta, r['p1'], r['rs']].astype(np.float32)
    f = m._query_metric('astra_f', task, xf, q)
    fa, _ = correct(f, b.heads[str(task)], b.distance_scale, xf, step, task, b.sig_head)
    rms = [float(np.sqrt(np.mean(((a[:10, :6]-ha[:10, :6])/sig[:6])**2))) for a in (wa, fa)]
    grip = [bool(np.any((a[:10, 6] >= 0) != (ha[:10, 6] >= 0))) for a in (wa, fa)]
    ht = int(h['rows'][0])
    pp = [abs(float(C.prog[ht])-float(C.prog[t]))*max(int(C.ep_len[ht])-1, int(C.ep_len[t])-1, 1)
          for t in (w['rows'][0], f['rows'][0])]
    adv = (float(C.prog[ht])-float(C.prog[prev['rows'][0]]))*max(int(C.ep_len[ht])-1, 1)
    out = dict(top=ht, rms=rms, grip=grip, progress=pp, advance=adv)
    # Hidden full observation/action used only here, for evaluation, never gate.
    fulltop = int(r['rows'][0])
    out['full_progress_error'] = abs(float(C.prog[ht])-float(C.prog[fulltop]))*max(int(C.ep_len[ht])-1, int(C.ep_len[fulltop])-1, 1)
    if r['hit']:
        out['full_rms'] = float(np.sqrt(np.mean(((ha[:10, :6]-r['action'][:10, :6])/sig[:6])**2)))
        out['full_grip'] = bool(np.any((ha[:10, 6] >= 0) != (r['action'][:10, 6] >= 0)))
    return out


def extract(cell):
    rows, summary, provenance = read_arm(cell, 'off')
    artifact = HERE.parent / 'lookhalf' / 'artifacts' / cell / 'astra_consensus2.pkl'
    with artifact.open('rb') as f:
        payload = pickle.load(f)
    m = payload['method']
    assert m.lookhalf_method == 'astra_consensus2'
    assert payload['kwargs']['episode_subset'] == summary['kwargs']['episode_subset']
    provenance['artifact_sha256'] = sha(artifact)
    decisions = decisions_for(cell)
    assert len(decisions) == summary['client_decisions']
    C, b = m.inner.C, m.inner.base
    previous, out = {}, []
    for i, r in enumerate(rows):
        with np.load(r['path'], allow_pickle=False) as z:
            assert z['camera_availability'].all()
            r.update(p0=b.B0T @ z['key_v0']-b.muB0,
                     p1=b.B1T @ z['key_v1']-b.muB1, rs=z['rs'][:8],
                     rows=np.searchsorted(m.row_subset, z['donor_rows']),
                     weights=z['donor_weights'], action=z['action'])
            np.testing.assert_array_equal(m.row_subset[r['rows']], z['donor_rows'])
            g = json.loads(str(z['guard_state']))
            guard = next(v for v in g['method'].values() if '_noprog_span' in v)
            threshold = guard['noprog_n']-1
        uid, s = r['uid'], r['step']
        d = decisions[(uid, s)]
        prev = previous.get(uid)
        pp = decisions.get((uid, s-1), {})
        legal = eligibility(d, pp, prev)
        record = dict(uid=uid, step=s, task=r['task'], init=r['init'], hit=r['hit'],
                      top=int(r['rows'][0]), threshold=threshold, legal=legal, span=r['span'],
                      next_vision=decisions.get((uid, s+1), {}).get('vision'),
                      progress=float(C.prog[r['rows'][0]]), length=int(C.ep_len[r['rows'][0]]))
        if prev and s >= 4 and s == prev['step']+2:
            mem = dict(step=prev['step'], rows=prev['rows'], weights=prev['weights'], rs=prev['rs'])
            # Quantify pre-look checks independently of actual prior guard span;
            # replay will impose its own evolving span and memory age.
            plan = atlas_plan(step=s, memory=mem, rs=r['rs'], table=m.half_fit['atlas'],
                              lifecycle=True, due=False, span=0)
            record['pre'] = plan
            record['candidate'] = proposal(m, r, prev)
            h = record['candidate']['top']
            record['candidate'].update(progress_value=float(C.prog[h]), length=int(C.ep_len[h]))
        out.append(record)
        previous[uid] = r
        if i % 500 == 0:
            print('offline proposals', cell, i, '/', len(rows), flush=True)
    (HERE / f'pilot_off_candidates_{cell}.json').write_text(json.dumps(dict(
        rows=out, N=summary['client_decisions'], V=len(rows), provenance=provenance,
        execution_assumption=EXECUTION_ASSUMPTION), separators=(',', ':'))+'\n')
    return out, summary['client_decisions']


def replay(rows, n, cell, *, tau=.1, mass=.75, witnesses=2):
    accepted, noncalls, missed, extra, divergent = [], [], 0, 0, set()
    last_uid, prev, span, lastfull = None, None, 0, -1
    for r in rows:
        if r['uid'] != last_uid:
            last_uid, prev, span, lastfull = r['uid'], None, 0, -1
        c, p = r.get('candidate'), r.get('pre')
        half = bool(c and r['legal'] and span == 0 and r['step']-lastfull == 2
                    and p['support'] >= .99 and p['safe_mass'] >= mass and p['state_error'] <= .2)
        if half and witnesses:
            half = bool(max(c['rms'][:witnesses]) <= tau and not any(c['grip'][:witnesses])
                        and max(c['progress'][:witnesses]) <= .5 and c['advance'] > .5)
        progress, length = (c['progress_value'], c['length']) if half else (r['progress'], r['length'])
        if prev is not None:
            span = span+r['step']-prev[0] if (progress-prev[1])*max(length-1, 1) <= .5 else 0
        call = span >= r['threshold']
        if call != (not r['hit']):
            divergent.add(r['uid'])
            missed += int(not call)
            extra += int(call)
        if r['next_vision'] is not None and bool(span > 0 and not call) != r['next_vision']:
            divergent.add(r['uid'])
        if half:
            accepted.append(r)
            if r['hit']:
                noncalls.append(r)
            if call:
                lastfull = -1
        else:
            lastfull = r['step']
        prev = (r['step'], progress)
        if witnesses == -1:
            assert not half
            assert span == r['span'] and call == (not r['hit'])
    err = [r['candidate']['full_rms'] for r in noncalls]
    return dict(tau=tau if np.isfinite(tau) else 'inf', mass=mass, witnesses=witnesses,
                V=len(rows), H=len(accepted), fixed_call_one_camera=len(noncalls),
                share=len(noncalls)/len(rows), saving=(.076 if cell.startswith('pi05') else .074)*len(noncalls)/n,
                missed=missed, extra=extra, divergent_episodes=len(divergent),
                full_grip_disagreement=sum(r['candidate']['full_grip'] for r in noncalls),
                full_progress_disagreement=sum(r['candidate']['full_progress_error'] > .5 for r in noncalls),
                full_motion_rms_median=float(np.median(err)) if err else None,
                full_motion_rms_p95=float(np.quantile(err, .95)) if err else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', choices=CELLS, default=CELLS)
    ap.add_argument('--cached', action='store_true')
    args = ap.parse_args()
    for cell in args.cells:
        if args.cached:
            cache = json.loads((HERE / f'pilot_off_candidates_{cell}.json').read_text())
            rows, n = cache['rows'], cache['N']
            decisions, previous = decisions_for(cell), {}
            for r in rows:
                r['legal'] = eligibility(decisions[(r['uid'], r['step'])],
                    decisions.get((r['uid'], r['step']-1), {}), previous.get(r['uid']))
                previous[r['uid']] = r
            cache['execution_assumption'] = EXECUTION_ASSUMPTION
            (HERE / f'pilot_off_candidates_{cell}.json').write_text(json.dumps(cache, separators=(',', ':'))+'\n')
        else:
            rows, n = extract(cell)
        # Full-only control must exactly reconstruct no-progress spans/calls.
        control = replay(rows, n, cell, mass=2., witnesses=-1)
        results = [replay(rows, n, cell, tau=tau, mass=mass) for mass in (.75, .5, 0.)
                   for tau in (.1, .15, .2, .3, float('inf'))]
        results += [replay(rows, n, cell, witnesses=k) for k in (0, 1)]
        (HERE / f'pilot_replay_{cell}.json').write_text(json.dumps(dict(cell=cell, control=control, results=results), indent=2)+'\n')
        if cell == 'pi05_spatial_200':
            finer = [replay(rows, n, cell, tau=tau, mass=mass)
                     for mass in (.75, .7, .65, .6, .5) for tau in (.15, .2, .25, .3)]
            (HERE / 'pilot_replay_spatial200_followup.json').write_text(json.dumps(finer, indent=2)+'\n')
        print('RESULT', cell, json.dumps([{k: r[k] for k in ('tau', 'mass', 'witnesses', 'share', 'missed', 'extra', 'full_grip_disagreement')} for r in results]), flush=True)
