#!/usr/bin/env python3
"""Configuration C: calibration, smoke, decision-log audit, SELECTION section-4 rules and post-hoc contrasts (ANALYSIS.md 5).

Inputs (read-only): r06_c_cal/cal/<cell>/calibrated/calibration.json, r06_c_cal journals, r06_c_smoke and
r06_c_validation arms (summary, accepted journal, server decisions_*.jsonl), A/B replicates, pure references, and
frontier_final/noninferiority.csv for the L5 DUAL references that live outside os_closed_loop.
The section-4 verdict tables themselves are regenerated separately with ops/c_validation.py (unchanged).
Writes analysis_r6/c_*.json and c_*.md.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

V = 'r06_c_validation'
SPARSE = [('pi05', 'l10_50'), ('pi05', 'sp_50'), ('groot', 'l10_50'), ('groot', 'sp_50')]
DENSE = [('pi05', 'l10_500'), ('pi05', 'sp_500'), ('groot', 'l10_500'), ('groot', 'sp_500')]
RHO = {'U30': .30, 'R30': .30, 'C30': .30, 'C45': .45, 'U18': .18, 'R18': .18, 'C18': .18}


def varm(model, cell, cfg):
    name = cm.c_cell(model, cell)
    if name == 'groot_l10_50' and cfg == 'C45':
        return f'{V}:r6c_groot_l10_50_Cmax'
    return f'{V}:r6c_{name}_{cfg}'


# ---------------------------------------------------------------- calibration
def calibration():
    rows = []
    for model, cell in SPARSE + DENSE:
        name = cm.c_cell(model, cell)
        c = json.load(open(f'{cm.ROOT}/r06_c_cal/cal/{name}/calibrated/calibration.json'))
        sol = c['solutions']
        rec = cm.journal(f'r06_c_cal:r6c_cal_{name}')
        r = dict(cell=name, a=c['intercept'], b=c['slope'], anchors=c['calibration_anchors'], baseline_E=c['baseline_E'],
                 episodes=len(c['calibration_episodes']), rec_success=sum(rec.values()), rec_n=len(rec),
                 status=c['status'], version=c['controller_version'], ambiguous_rule=c['ambiguous_rule'],
                 cooldown_scope=c['cooldown_scope'], cadence_nodes=c.get('cadence_nodes'))
        for mode in ('no_stall', 'stall'):
            for pl in ('uniform', 'R'):
                s = sol[mode][pl]
                any_ = next(iter(s.values()))
                r[f'{mode}_{pl}_floor'] = any_['floor']
                r[f'{mode}_{pl}_ceiling'] = any_['ceiling']
                for rho in ('0.18', '0.3', '0.45'):
                    if rho in s:
                        r[f'{mode}_{pl}_{rho}_feasible'] = s[rho]['feasible']
                        r[f'{mode}_{pl}_{rho}_param'] = s[rho].get('parameter')
        rows.append(r)
    cm.dump('c_calibration.json', rows)
    return rows


# ---------------------------------------------------------------- decision logs
def audit_arm(spec):
    """Call composition and C lifecycle invariants on the accepted attempt of every episode."""
    run, arm = spec.split(':')
    d = f'{cm.ROOT}/{run}/runs/{arm}'
    acc = {}
    for line in open(f'{d}/client/journal.jsonl'):
        r = json.loads(line)
        if r.get('status') in ('done', 'failed') and ':eval:' in r.get('task_uid', ''):
            acc[r['task_uid']] = r['attempt']
    eps = {}
    for f in glob.glob(f'{d}/server_*/decisions_*.jsonl'):
        for line in open(f):
            if '"ev": "dec"' not in line[:40]:
                continue
            r = json.loads(line)
            if acc.get(r.get('uid')) != r.get('attempt'):
                continue
            e = (r.get('extras') if r.get('vision') else r.get('blind_extras')) or {}
            eps.setdefault(r['uid'], []).append((r['step'], bool(r.get('vision')), r.get('src'), r.get('judge') or '',
                                                 r.get('look_reason'), e.get('os_reason'), e.get('os_c_fresh'),
                                                 e.get('os_c_call'), e.get('os_c_stall_state'),
                                                 e.get('os_c_extra_look'), e.get('os_c_cooldown'), e.get('policy_tail'),
                                                 bool(r.get('shadow_available')), e.get('os_c_p')))
    c = Counter()
    viol = Counter()
    for uid, seq in eps.items():
        seq.sort()
        steps = [x[0] for x in seq]
        if len(set(steps)) != len(steps):
            viol['duplicate_step'] += 1
        for i, (st, vis, src, judge, lr, reason, fresh, call, sstate, xlook, cool, ptail, shadow, p) in enumerate(seq):
            c['decisions'] += 1
            c['vision'] += vis
            c['shadow'] += shadow
            is_call = judge.startswith('force:')
            c['miss'] += is_call
            c['policy_tail'] += src == 'policy_tail'
            if is_call:
                c['call_' + judge.split(':')[1]] += 1
            if vis and fresh == 1.0:
                c['fresh'] += 1
                c[f'state_{int(sstate) if sstate is not None else -1}'] += 1
                c['extra_look'] += xlook == 1.0
                c['cooldown_anchor'] += cool == 1.0
                if cool == 1.0 and is_call and judge != 'force:63':
                    viol['call_in_cooldown'] += 1
                if xlook == 1.0 and (sstate != 3.0 or is_call):
                    viol['extra_look_not_ambiguous_nocall'] += 1
                if sstate == 3.0 and is_call:
                    c['ambiguous_call'] += 1
                if sstate == 2.0 and not is_call and cool != 1.0:
                    viol['confirmed_without_call'] += 1
            if lr == 8:
                c['look_reason_8'] += 1
            if is_call:
                if i + 1 < len(seq):
                    nxt = seq[i + 1]
                    if nxt[2] != 'policy_tail':
                        viol['call_not_followed_by_tail'] += 1
                    elif i + 2 < len(seq) and not seq[i + 2][1]:
                        viol['tail_not_followed_by_vision'] += 1
                    if judge == 'force:63' and i + 2 < len(seq):
                        after = seq[i + 2]
                        if after[1] and after[6] == 1.0 and after[10] != 1.0:
                            viol['stall_call_without_cooldown'] += 1
                    if judge == 'force:62' and i + 2 < len(seq):
                        after = seq[i + 2]
                        if after[1] and after[10] == 1.0:
                            viol['lottery_call_followed_by_cooldown'] += 1
                else:
                    c['terminal_call'] += 1
    N, Vd, M, _ = cm.ledger(spec)
    return dict(spec=spec, episodes=len(eps), counts=dict(c), violations=dict(viol),
                ledger=dict(N=N, V=Vd, M=M), ledger_match=(c['decisions'], c['vision'], c['miss']) == (N, Vd, M))


# ---------------------------------------------------------------- contrasts
def dual_ni():
    rows = {}
    with open(os.path.join(cm.R6, 'frontier_final', 'noninferiority.csv')) as f:
        for r in csv.DictReader(f):
            rows[(r['id'], r['reference_L'])] = r
    return rows


def main():
    os.makedirs(cm.OUT, exist_ok=True)
    cal = calibration()
    specs = [varm(m, c, cfg) for m, c in SPARSE for cfg in ('R30', 'U30', 'C30', 'C45')] + \
            [varm(m, c, cfg) for m, c in DENSE for cfg in ('R18', 'U18', 'C18')] + \
            ['r06_c_smoke:r6c_pi05_l10_500_C18', 'r06_c_smoke:r6c_groot_l10_500_C18']
    with Pool(4) as p:
        audits = p.map(audit_arm, specs, chunksize=1)
    cm.dump('c_decision_audit.json', audits)
    aud = {a['spec']: a for a in audits}
    ni = dual_ni()
    arms, md = [], []
    md.append('| Cell | cfg | SR @ owner IR | IR − ρ | v | m | lottery calls | stall calls | stall share | extra LOOKs | − A mean pp [95% CI] | − B mean pp [95% CI] | NI lower vs L10 / L5 (pp) |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for model, cell in SPARSE + DENSE:
        A, B = cm.ab_arms(model, cell)
        ma, mb = cm.avg(A), cm.avg(B)
        for cfg in (('R30', 'U30', 'C30', 'C45') if (model, cell) in SPARSE else ('R18', 'U18', 'C18')):
            spec = varm(model, cell, cfg)
            o = cm.journal(spec)
            assert set(o) == cm.EXPECTED and cm.summary(spec)['success'] == sum(o.values())
            of = {k: float(v) for k, v in o.items()}
            ir = cm.owner_ir(spec, model)
            rho = .440704194944 if spec.endswith('Cmax') else RHO[cfg]
            da = cm.boot_ci(of, ma)
            db = cm.boot_ci(of, mb)
            N, Vd, M, _ = cm.ledger(spec)
            jm = cm.summary(spec).get('mixed', {}).get('judge_mix', {})
            a = aud[spec]['counts']
            fid = spec.replace(':', '/')
            n10 = ni.get((fid, '10'), {}).get('paired_exact_lower95')
            n5 = ni.get((fid, '5'), {}).get('paired_exact_lower95')
            s10 = ni.get((fid, '10'), {}).get('paired_exact_lower_simultaneous')
            s5 = ni.get((fid, '5'), {}).get('paired_exact_lower_simultaneous')
            # own recomputation vs L10 for cross-check
            own10 = cm.ni_lower(o, cm.journal(cm.l10_ref(model, cell)))[0]
            r = dict(model=model, cell=cell, cfg='Cmax' if spec.endswith('Cmax') else cfg, spec=spec, sr=cm.sr(o), ir=ir,
                     rho=rho, v=Vd / N, m=M / N, calls62=a.get('call_62', 0) + a.get('call_61', 0), calls63=a.get('call_63', 0),
                     judge_mix=jm, extra_look=a.get('extra_look', 0), fresh=a.get('fresh', 0),
                     states={k: v for k, v in a.items() if k.startswith('state_')}, ambiguous_calls=a.get('ambiguous_call', 0),
                     d_a=da[:3], d_b=db[:3], ni10=float(n10) if n10 else None, ni5=float(n5) if n5 else None,
                     sim10=float(s10) if s10 else None, sim5=float(s5) if s5 else None, own_ni10=own10)
            arms.append(r)
            calls = r['calls62'] + r['calls63']
            share = f"{r['calls63'] / calls:.2f}" if calls else '—'
            md.append(f"| {cm.cname(model, cell)} | {r['cfg']} | {r['sr']:.3f} @ {ir:.3f} | {ir - rho:+.3f} | {r['v']:.3f} | {r['m']:.3f} | "
                      f"{r['calls62']} | {r['calls63']} | {share} | {r['extra_look']} | {cm.pp(da[0])} [{cm.pp(da[1])}, {cm.pp(da[2])}] | "
                      f"{cm.pp(db[0])} [{cm.pp(db[1])}, {cm.pp(db[2])}] | "
                      f"{100 * r['ni10']:+.1f} / {100 * r['ni5']:+.1f} |")
    # post-hoc pooled contrasts (four cells, each resampled independently; per-contrast seed)
    def pooled(pairs):
        pts, sims = [], []
        for x, y in pairs:
            p, s = cm.boot({k: float(v) for k, v in cm.journal(x).items()}, {k: float(v) for k, v in cm.journal(y).items()})
            pts.append(p)
            sims.append(s)
        ps = np.mean(sims, axis=0)
        lo, hi = cm.ci(ps)
        return float(np.mean(pts)), lo, hi, pts
    post = {}
    post['C-R_all8'] = pooled([(varm(m, c, 'C30'), varm(m, c, 'R30')) for m, c in SPARSE] +
                              [(varm(m, c, 'C18'), varm(m, c, 'R18')) for m, c in DENSE])
    post['R-U_all8'] = pooled([(varm(m, c, 'R30'), varm(m, c, 'U30')) for m, c in SPARSE] +
                              [(varm(m, c, 'R18'), varm(m, c, 'U18')) for m, c in DENSE])
    for name, grp, x, y in (('R30-U30', SPARSE, 'R30', 'U30'), ('C30-R30', SPARSE, 'C30', 'R30'), ('C30-U30', SPARSE, 'C30', 'U30'),
                            ('C45-C30', SPARSE, 'C45', 'C30'),
                            ('R18-U18', DENSE, 'R18', 'U18'), ('C18-R18', DENSE, 'C18', 'R18'), ('C18-U18', DENSE, 'C18', 'U18')):
        post[name] = pooled([(varm(m, c, x), varm(m, c, y)) for m, c in grp])
        post[name + '_cells'] = []
        for m, c in grp:
            xo, yo = cm.journal(varm(m, c, x)), cm.journal(varm(m, c, y))
            w, l, _ = cm.wl(xo, yo)
            p, lo, hi, _ = cm.boot_ci({k: float(v) for k, v in xo.items()}, {k: float(v) for k, v in yo.items()})
            post[name + '_cells'].append(dict(cell=cm.c_cell(m, c), d=p, lo=lo, hi=hi, w=w, l=l, p=cm.mcnemar(w, l),
                                             dIR=cm.owner_ir(varm(m, c, x), m) - cm.owner_ir(varm(m, c, y), m)))
    # C at high budget vs the task-level risk lottery at the same target
    lot = {}
    for m, c, lot_arm in (('pi05', 'l10_50', 'r06_frontier:r6q2_pi05_l10_50_risk_rho0p45'),
                          ('pi05', 'sp_50', 'r06_frontier:r6q2_pi05_spatial_50_risk_rho0p45'),
                          ('groot', 'l10_50', 'r06_frontier:r6q2_groot_l10_50_risk_rho0p45')):
        x, y = cm.journal(varm(m, c, 'C45')), cm.journal(lot_arm)
        w, l, _ = cm.wl(x, y)
        p, lo, hi, _ = cm.boot_ci({k: float(v) for k, v in x.items()}, {k: float(v) for k, v in y.items()})
        lot[cm.c_cell(m, c)] = dict(c=varm(m, c, 'C45'), lottery=lot_arm, c_sr=cm.sr(x), lot_sr=cm.sr(y),
                                    c_ir=cm.owner_ir(varm(m, c, 'C45'), m), lot_ir=cm.owner_ir(lot_arm, m),
                                    w=w, l=l, p=cm.mcnemar(w, l), d=p, lo=lo, hi=hi)
    # stall component on GR00T Sp-50 against B-without-NP and Bmech (single-run pairs)
    extra = {}
    for tag, ref in (('noNP', 'r06_abl:r6p2_no_progress_g_sp_50'), ('Bmech', 'r06_method:r6p5_bmech_g_sp_50')):
        x, y = cm.journal(varm('groot', 'sp_50', 'C30')), cm.journal(ref)
        w, l, _ = cm.wl(x, y)
        extra[tag] = dict(w=w, l=l, p=cm.mcnemar(w, l), sr_ref=cm.sr(y), ir_ref=cm.owner_ir(ref, 'groot'))
    cm.dump('c_arms.json', dict(arms=arms, posthoc=post, lottery=lot, gsp50=extra))
    with open(os.path.join(cm.OUT, 'c_arms.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(json.dumps(dict(posthoc={k: v for k, v in post.items()}, lottery=lot, gsp50=extra), indent=1, default=float))
    print(json.dumps([dict(spec=a['spec'], ledger_match=a['ledger_match'], violations=a['violations'],
                           episodes=a['episodes']) for a in audits], indent=0))
    print(json.dumps(cal, indent=0, default=float)[:6000])


if __name__ == '__main__':
    main()
