#!/usr/bin/env python3
"""A / B three-replicate headline, replicate spread, and the paper ablations with paired intervals (ANALYSIS.md 2-3).

Recomputed from each arm's summary.json (cost_ledger) and accepted journal records. Intervals: task-stratified init
bootstrap (10,000 draws, seed 20260929), replicates averaged within each (task, init) pair. Per-replicate tests: exact
McNemar. Writes analysis_r6/headline.json, ablations.json and markdown tables headline.md / ablations.md.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

P = {'pi05': 'p', 'groot': 'g'}


def headline():
    rows, md = [], []
    md.append('| Cell | A SR (3 runs) | A mean @ IR | B SR (3 runs) | B mean @ IR | B − A pp [95% CI] | per-rep McNemar p (A_k vs B_k) | pooled +/− | B − pure L10 pp [95% CI] |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    spread = []
    for model, cell in cm.CELLS:
        A, B = cm.ab_arms(model, cell)
        ja, jb = [cm.journal(s) for s in A], [cm.journal(s) for s in B]
        for j, s in zip(ja + jb, A + B):
            assert set(j) == cm.EXPECTED, s
            assert cm.summary(s)['success'] == sum(j.values()), s
        sa, sb = [cm.sr(j) for j in ja], [cm.sr(j) for j in jb]
        ira, irb = [cm.owner_ir(s, model) for s in A], [cm.owner_ir(s, model) for s in B]
        ma, mb = cm.avg(A), cm.avg(B)
        p, lo, hi, _ = cm.boot_ci(mb, ma)
        per = []
        plus = minus = 0
        for x, y in zip(ja, jb):
            w, l, _ = cm.wl(y, x)
            plus += w
            minus += l
            per.append(cm.mcnemar(w, l))
        l10 = cm.journal(cm.l10_ref(model, cell))
        q, qlo, qhi, _ = cm.boot_ci(mb, {k: float(v) for k, v in l10.items()})
        # pooled decision-weighted IR
        def pooled(specs):
            N = V = M = 0
            for s in specs:
                n, v, m, _ = cm.ledger(s)
                N, V, M = N + n, V + v, M + m
            return cm.C1[model] * V / N + (1 - cm.C1[model]) * M / N
        r = dict(model=model, cell=cell, A=A, B=B, A_sr=sa, B_sr=sb, A_ir=ira, B_ir=irb,
                 A_mean=float(np.mean(sa)), B_mean=float(np.mean(sb)), A_ir_mean=float(np.mean(ira)),
                 B_ir_mean=float(np.mean(irb)), A_ir_pooled=pooled(A), B_ir_pooled=pooled(B),
                 d=p, lo=lo, hi=hi, per_rep_p=per, pooled_plus=plus, pooled_minus=minus,
                 pooled_p_not_independent=cm.mcnemar(plus, minus),
                 l10=cm.sr(l10), d_l10=q, d_l10_lo=qlo, d_l10_hi=qhi)
        rows.append(r)
        md.append(f"| {cm.cname(model, cell)} | {' / '.join(cm.f3(x) for x in sa)} | {r['A_mean']:.3f} @ {r['A_ir_mean']:.3f} | "
                  f"{' / '.join(cm.f3(x) for x in sb)} | {r['B_mean']:.3f} @ {r['B_ir_mean']:.3f} | {cm.pp(p)} [{cm.pp(lo)}, {cm.pp(hi)}] | "
                  f"{' / '.join(f'{x:.2g}' for x in per)} | +{plus}/−{minus} | {cm.pp(q)} [{cm.pp(qlo)}, {cm.pp(qhi)}] |")
        for name, js, specs in (('A', ja, A), ('B', jb, B)):
            srs = [cm.sr(j) for j in js]
            pairs = []
            for (i, x), (k, y) in itertools.combinations(enumerate(js), 2):
                w, l, n = cm.wl(x, y)
                pairs.append(dict(i=i + 1, k=k + 1, w=w, l=l, disc=(w + l) / n, p=cm.mcnemar(w, l)))
            spread.append(dict(model=model, cell=cell, cfg=name, sr=srs, sd_pp=100 * float(np.std(srs, ddof=1)),
                               range_pp=100 * (max(srs) - min(srs)), pairs=pairs,
                               min_p=min(x['p'] for x in pairs)))
    md.append('')
    md.append('| Cell | cfg | replicate SRs | sample SD pp | range pp | pairwise discordance | min pairwise McNemar p |')
    md.append('|---|---|---|---|---|---|---|')
    for s in spread:
        disc = ', '.join('%.1f%%' % (100 * x['disc']) for x in s['pairs'])
        md.append(f"| {cm.cname(s['model'], s['cell'])} | {s['cfg']} | {' / '.join(cm.f3(x) for x in s['sr'])} | {s['sd_pp']:.2f} | "
                  f"{s['range_pp']:.1f} | {disc} | {s['min_p']:.2g} |")
    cm.dump('headline.json', dict(rows=rows, spread=spread))
    return md


def ablations():
    md, out = [], {}
    cells = [(m, c) for m in ('pi05', 'groot') for c in ('l10_50', 'l10_500', 'sp_50', 'sp_500')]
    for kind, title in (('identity', 'Identity metric (z-scored Euclidean) − A'), ('direct', 'Direct token PCA − A')):
        md.append(f'**{title}**\n')
        md.append('| Cell | A mean | ablation SR @ IR | Δ vs A mean pp [95% CI] | McNemar p vs A reps (min–max) |')
        md.append('|---|---|---|---|---|')
        out[kind] = []
        for model, cell in cells:
            A, _ = cm.ab_arms(model, cell)
            spec = f'r06_abl:r6p2_{kind}_{P[model]}_{cell}'
            o = cm.journal(spec)
            assert set(o) == cm.EXPECTED
            p, lo, hi, _ = cm.boot_ci({k: float(v) for k, v in o.items()}, cm.avg(A))
            ps = [cm.mcnemar(*cm.wl(o, cm.journal(a))[:2]) for a in A]
            out[kind].append(dict(model=model, cell=cell, spec=spec, sr=cm.sr(o), ir=cm.owner_ir(spec, model),
                                  a_mean=float(np.mean([cm.sr(cm.journal(a)) for a in A])), d=p, lo=lo, hi=hi, p=ps))
            r = out[kind][-1]
            md.append(f"| {cm.cname(model, cell)} | {r['a_mean']:.3f} | {r['sr']:.3f} @ {r['ir']:.3f} | {cm.pp(p)} [{cm.pp(lo)}, {cm.pp(hi)}] | "
                      f"{min(ps):.2g}–{max(ps):.2g} |")
        md.append('')
    md.append('**Trigger leave-one-out (each guard removed alone) and Bmech (no-progress MISS bit off, LOOK veto kept)**\n')
    md.append('| Cell | arm | SR @ IR | − B mean pp [95% CI] | McNemar p vs B reps | − A mean pp [95% CI] | McNemar p vs A reps |')
    md.append('|---|---|---|---|---|---|---|')
    out['loo'] = []
    for model in ('pi05', 'groot'):
        for cell in ('l10_50', 'sp_50'):
            A, B = cm.ab_arms(model, cell)
            ma, mb = cm.avg(A), cm.avg(B)
            bir = float(np.mean([cm.owner_ir(s, model) for s in B]))
            md.append(f"| {cm.cname(model, cell)} | B (3-run mean) | {cm.sr(mb):.3f} @ {bir:.3f} | — | — | "
                      f"{cm.pp(cm.boot_ci(mb, ma)[0])} | — |")
            for g in ('stuck', 'terminal', 'overtime', 'no_progress', 'BMECH'):
                spec = (f'r06_method:r6p5_bmech_{P[model]}_{cell}' if g == 'BMECH'
                        else f'r06_abl:r6p2_{g}_{P[model]}_{cell}')
                o = cm.journal(spec)
                assert set(o) == cm.EXPECTED
                of = {k: float(v) for k, v in o.items()}
                pb, lob, hib, _ = cm.boot_ci(of, mb)
                pa, loa, hia, _ = cm.boot_ci(of, ma)
                psb = [cm.mcnemar(*cm.wl(o, cm.journal(b))[:2]) for b in B]
                psa = [cm.mcnemar(*cm.wl(o, cm.journal(a))[:2]) for a in A]
                r = dict(model=model, cell=cell, guard=g, spec=spec, sr=cm.sr(o), ir=cm.owner_ir(spec, model),
                         d_b=pb, lo_b=lob, hi_b=hib, p_b=psb, d_a=pa, lo_a=loa, hi_a=hia, p_a=psa)
                if g == 'BMECH':
                    nonp = cm.journal(f'r06_abl:r6p2_no_progress_{P[model]}_{cell}')
                    w, l, _ = cm.wl(o, nonp)
                    q, qlo, qhi, _ = cm.boot_ci(of, {k: float(v) for k, v in nonp.items()})
                    r.update(vs_nonp=dict(w=w, l=l, p=cm.mcnemar(w, l), d=q, lo=qlo, hi=qhi))
                out['loo'].append(r)
                name = 'Bmech' if g == 'BMECH' else g.replace('_', '-') + ' removed'
                md.append(f"| {cm.cname(model, cell)} | {name} | {r['sr']:.3f} @ {r['ir']:.3f} | {cm.pp(pb)} [{cm.pp(lob)}, {cm.pp(hib)}] | "
                          f"{min(psb):.2g}–{max(psb):.2g} | {cm.pp(pa)} [{cm.pp(loa)}, {cm.pp(hia)}] | {min(psa):.2g}–{max(psa):.2g} |")
    md.append('')
    md.append('| Cell | Bmech − (no-progress removed): W/L; exact p; Δ pp [95% CI] |')
    md.append('|---|---|')
    for r in out['loo']:
        if r['guard'] == 'BMECH':
            v = r['vs_nonp']
            md.append(f"| {cm.cname(r['model'], r['cell'])} | {v['w']}/{v['l']}; p={v['p']:.2g}; {cm.pp(v['d'])} [{cm.pp(v['lo'])}, {cm.pp(v['hi'])}] |")
    cm.dump('ablations.json', out)
    return md


def main():
    h = headline()
    a = ablations()
    with open(os.path.join(cm.OUT, 'headline.md'), 'w') as f:
        f.write('\n'.join(h) + '\n')
    with open(os.path.join(cm.OUT, 'ablations.md'), 'w') as f:
        f.write('\n'.join(a) + '\n')
    print('\n'.join(h))
    print()
    print('\n'.join(a))


if __name__ == '__main__':
    main()
