"""R7 A2 sparse-versus-dense follow mechanism (SF1 / UF1 against A replicate 1 of the same cell).

Per anchor (accepted episodes, eval run):
  kernel diversity   distinct library episodes among the 16 members (weight > 1e-3), weight-effective number of
                     episodes 1/sum_e w_e^2, and the step spread of members that share the top member's episode;
  valve drift        SF1 blind-check displacement statistic D (os_sf_delta) by check age (1 = A's own blind block,
                     2 = the extension block), in state-scale units, next to the cell's radius;
  look after a cycle for every LOOK: whether the new top-1 row continues the previous anchor's top-1 demonstration
                     (same library episode, library step within +-1 of the elapsed decisions) and the new anchor's
                     retrieval distance d1 (log), split by what the previous cycle was: 'ext' (an extension block was
                     served) or 'A' (A cadence). A replicate 1 provides the A-only reference on the same pairs.
Outputs <out>/a2_mech_<run>.json (per arm and per reference).

    <A2 prefix> -m exp.offline_search.rounds.r07.analysis_scripts.a2_mech [--jobs 4]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C


def q(x, ps=(10, 50, 90)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
                **{f'p{p}': float(np.percentile(x, p)) for p in ps} if len(x) else {})


def diversity(table, rows, w):
    rows, w = np.asarray(rows, int), np.asarray(w, float)
    w = w / w.sum()
    ep = table.episode[rows]
    keep = w > 1e-3
    mass = defaultdict(float)
    for e, ww in zip(ep, w):
        mass[int(e)] += ww
    eff = 1.0 / sum(m * m for m in mass.values())
    same = rows[(ep == ep[0]) & keep]
    spread = int(table.step[same].max() - table.step[same].min()) if len(same) else 0
    return len(set(ep[keep].tolist())), eff, spread, int(((ep == ep[0]) & keep).sum())


def analyze(run_root, arm, cell, label):
    A = C.load_arm(run_root, arm)
    t = C.stage_table(cell)
    out = defaultdict(list)
    for e in A['episodes']:
        ds = e['decisions']
        anchors = [i for i, d in enumerate(ds) if d.get('vision')]
        prev = None
        for k, i in enumerate(anchors):
            d = ds[i]
            j = anchors[k + 1] if k + 1 < len(anchors) else len(ds)
            body = ds[i + 1:j]
            ext = any(C.bx(b, 'os_sf_extension') == 1 for b in body)
            rows, w = d.get('rows'), d.get('weights')
            if not rows or max(rows) >= len(t.mode):
                prev = None
                continue
            n_ep, eff, spread, same_n = diversity(t, rows, w)
            granted = bool(C.ex(d, 'os_sf_granted'))
            tag = 'granted' if granted else 'not_granted'
            out[f'eps_distinct_{tag}'].append(n_ep)
            out[f'eps_eff_{tag}'].append(eff)
            out[f'same_ep_members_{tag}'].append(same_n)
            out[f'same_ep_step_spread_{tag}'].append(spread)
            if ext:
                out['eps_eff_extended'].append(eff)
                out['same_ep_members_extended'].append(same_n)
            for b in body:
                if C.bx(b, 'os_sf_valve_checked') == 1:
                    out[f"delta_age{int(C.bx(b, 'os_sf_age'))}"].append(C.bx(b, 'os_sf_delta'))
            if prev is not None:
                pd, pext = prev
                gap = int(d['step']) - int(pd['step'])
                r0, r1 = int(pd['rows'][0]), int(rows[0])
                cont = (t.episode[r0] == t.episode[r1]) and abs(int(t.step[r1]) - int(t.step[r0]) - gap) <= 1
                typ = 'after_ext' if pext else 'after_A'
                out[f'continue_{typ}'].append(float(cont))
                d1 = C.ex(d, 'd1')
                if d1 is not None and d1 > 0:
                    out[f'logd1_{typ}'].append(float(np.log(d1)))
                out[f'Y_{typ}'].append(e['Y'])
            prev = (d, ext)
    res = dict(arm=arm, label=label, cell=cell, episodes=len(A['episodes']))
    for k, v in out.items():
        if k.startswith(('continue_', 'Y_')):
            res[k] = dict(n=len(v), mean=float(np.mean(v)) if v else None)
        else:
            res[k] = q(v)
    res['radius'] = float(t.valve_radius)
    return res


def jobs_for(run_root):
    from exp.offline_search.rounds.r06.analysis_scripts import common as R6
    jobs = []
    for cell in C.CELLS8:
        for v in ('SF1', 'UF1'):
            arm = f'r7_{cell}_{v}'
            if C.arm_status(run_root, arm) == 'complete':
                jobs.append((run_root, arm, cell, v))
        info = C.parse_arm(f'r7_{cell}_SF1')
        a1 = R6.ab_arms(info['model'], info['r6cell'])[0][0]
        run, arm = a1.split(':')
        jobs.append((C.ROOT / run, arm, cell, 'A_rep1'))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='eval')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--out', type=Path, default=C.OUT)
    a = p.parse_args()
    run_root = C.EVAL if a.run == 'eval' else Path(a.run)
    jobs = jobs_for(run_root)
    with Pool(a.jobs) as pool:
        res = pool.starmap(analyze, jobs)
    C.write_json(a.out / f'a2_mech_{run_root.name}.json', dict(schema='r7.a2.mech.v1', results=res))
    for r in res:
        print(r['cell'], r['label'], 'effEp(granted)', round((r.get('eps_eff_granted') or {}).get('p50') or float('nan'), 2),
              'sameEpMembers', (r.get('same_ep_members_granted') or {}).get('p50'),
              'cont afterA', round(r.get('continue_after_A', {}).get('mean') or float('nan'), 3),
              'afterExt', round((r.get('continue_after_ext') or {}).get('mean') or float('nan'), 3),
              'logd1 A/ext', round((r.get('logd1_after_A') or {}).get('p50') or float('nan'), 3),
              round((r.get('logd1_after_ext') or {}).get('p50') or float('nan'), 3),
              'delta a1/a2 p50', round((r.get('delta_age1') or {}).get('p50') or float('nan'), 3),
              round((r.get('delta_age2') or {}).get('p50') or float('nan'), 3), 'radius', round(r['radius'], 3), flush=True)


if __name__ == '__main__':
    main()
