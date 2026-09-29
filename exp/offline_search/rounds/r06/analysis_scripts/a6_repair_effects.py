#!/usr/bin/env python3
"""Before / after values of the conclusions touched by the exception-episode repair (ANALYSIS.md 1).

Pre-repair outcome of a purged (task, init) = its purged journal record (always success=False, see a1); every other pair
keeps its current accepted outcome, which the repair did not touch. Writes analysis_r6/repair_effects.json / .md.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

BA = {f"{r['run']}:{r['arm']}": r for r in json.load(open(os.path.join(cm.OUT, 'repair_scan_arms.json')))
      if 'purged_records' in r}


def outcome(spec, before):
    o = dict(cm.journal(spec))
    if before and spec in BA:
        for u in BA[spec]['purged_uids']:
            o[tuple(int(x) for x in u.split(':eval:')[1].split(':')[:2])] = False
    return o


def avg(specs, before):
    js = [outcome(s, before) for s in specs]
    return {k: sum(j[k] for j in js) / len(js) for k in js[0]}


def main():
    md, out = [], {}
    # A / B
    md.append('| Cell | A mean before → after | B mean before → after | B − A before → after (pp) | max replicate range before → after (pp) |')
    md.append('|---|---|---|---|---|')
    out['ab'] = []
    for model, cell in cm.CELLS:
        A, B = cm.ab_arms(model, cell)
        if not any(s in BA for s in A + B):
            continue
        r = {}
        for tag, before in (('before', True), ('after', False)):
            sa = [cm.sr(outcome(s, before)) for s in A]
            sb = [cm.sr(outcome(s, before)) for s in B]
            r[tag] = dict(A=float(np.mean(sa)), B=float(np.mean(sb)), d=float(np.mean(sb) - np.mean(sa)),
                          rng=100 * max(max(sa) - min(sa), max(sb) - min(sb)))
        out['ab'].append(dict(model=model, cell=cell, **r))
        b, a = r['before'], r['after']
        md.append(f"| {cm.cname(model, cell)} | {b['A']:.3f} → {a['A']:.3f} | {b['B']:.3f} → {a['B']:.3f} | {cm.pp(b['d'])} → {cm.pp(a['d'])} | "
                  f"{b['rng']:.1f} → {a['rng']:.1f} |")
    md.append('')
    # ablations
    md.append('| Ablation arm | SR before → after | Δ vs reference mean before → after (pp) [after 95% CI] | reference |')
    md.append('|---|---|---|---|')
    out['abl'] = []
    for spec in sorted(s for s in BA if s.startswith('r06_abl:')):
        arm = spec.split(':')[1]
        parts = arm.split('_')
        model = 'pi05' if parts[-3] == 'p' else 'groot'
        cell = '_'.join(parts[-2:])
        A, B = cm.ab_arms(model, cell)
        kind = arm[len('r6p2_'):].rsplit('_', 3)[0]
        ref, refname = (B, 'B mean') if kind in ('stuck', 'terminal', 'overtime', 'no_progress') else (A, 'A mean')
        rb, ra = avg(ref, True), avg(ref, False)
        ob, oa = outcome(spec, True), outcome(spec, False)
        db = cm.sr(ob) - cm.sr(rb)
        p, lo, hi, _ = cm.boot_ci({k: float(v) for k, v in oa.items()}, ra)
        out['abl'].append(dict(spec=spec, sr_before=cm.sr(ob), sr_after=cm.sr(oa), d_before=db, d_after=p, lo=lo, hi=hi, ref=refname))
        md.append(f"| `{arm}` | {cm.sr(ob):.3f} → {cm.sr(oa):.3f} | {cm.pp(db)} → {cm.pp(p)} [{cm.pp(lo)}, {cm.pp(hi)}] | {refname} (after repair of the references too) |")
    md.append('')
    # frontier arms
    md.append('| Frontier arm | SR before → after | NI lower vs pure L10 before → after (pp) | − B mean after (pp) [95% CI] |')
    md.append('|---|---|---|---|')
    out['frontier'] = []
    for spec in sorted(s for s in BA if s.startswith('r06_frontier:')):
        arm = spec.split(':')[1]
        model = 'pi05' if '_pi05_' in arm else 'groot'
        suite = 'l10' if '_l10_' in arm else 'sp'
        size = '500' if f'_{"l10" if suite == "l10" else "spatial"}_500_' in arm else '50'
        cell = f'{suite}_{size}'
        l10 = cm.journal(cm.l10_ref(model, cell))
        ob, oa = outcome(spec, True), outcome(spec, False)
        nb, na = cm.ni_lower(ob, l10)[0], cm.ni_lower(oa, l10)[0]
        A, B = cm.ab_arms(model, cell)
        p, lo, hi, _ = cm.boot_ci({k: float(v) for k, v in oa.items()}, cm.avg(B))
        out['frontier'].append(dict(spec=spec, sr_before=cm.sr(ob), sr_after=cm.sr(oa), ni_before=nb, ni_after=na, d_b=p, lo=lo, hi=hi,
                                    ir=cm.owner_ir(spec, model)))
        md.append(f"| `{arm}` @ IR {cm.owner_ir(spec, model):.3f} | {cm.sr(ob):.3f} → {cm.sr(oa):.3f} | {100 * nb:+.1f} → {100 * na:+.1f} | "
                  f"{cm.pp(p)} [{cm.pp(lo)}, {cm.pp(hi)}] |")
    cm.dump('repair_effects.json', out)
    with open(os.path.join(cm.OUT, 'repair_effects.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
