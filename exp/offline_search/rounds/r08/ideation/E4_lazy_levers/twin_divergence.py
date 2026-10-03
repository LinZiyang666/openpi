"""R8 E4 preliminary evidence: GR00T twin divergence between a lever arm and A replicate 1.

GR00T A is (almost) bit-deterministic, so an SF1/UF1 episode and A replicate 1 on the same (task, init) share
their whole history up to the first decision where the lever acted. From that decision on we measure:
  * d0: action discrepancy (arm dims, library action-sigma RMS) between the lever's block and A's fresh-look block
    at the first divergent decision (same observation);
  * the growth of 8-d robot-state distance (library state-sigma RMS) between the twins, k decisions later;
  * whether the outcome flips (A rep1 vs lever arm).
Read-only; writes twin_summary.json here and per-pair CSV in /tmp/r8_E4_lazy_levers/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C
from exp.offline_search.rounds.r08.ideation.E4_lazy_levers.follow_vs_look import STORE, LIBNAME, SCRATCH, auroc

HERE = Path(__file__).resolve().parent
KS = (1, 2, 4, 8, 16, 32)


def stream(run, arm):
    A = C.load_arm(C.ROOT / run, arm, keep_heavy=True)
    return {(e['task'], e['init']): e for e in A['episodes']}


def main():
    out = {}
    for cell in ['groot_l10_50', 'groot_l10_500', 'groot_spatial_50', 'groot_spatial_500']:
        info = C.parse_arm(f'r7_{cell}_SF1')
        lib = STORE / f"groot_{info['suite']}" / LIBNAME[('groot', info['size'])]
        act = np.load(lib / 'action.npy', mmap_mode='r')
        rs = np.load(lib / 'rs.npy', mmap_mode='r')
        succ = np.load(lib / 'success.npy').astype(bool)
        good = np.flatnonzero(succ)
        asig = act[good][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
        ssig = np.asarray(rs[good][:, :8], np.float64).std(0)
        ssig[ssig < 1e-6] = 1.0
        from exp.offline_search.rounds.r06.analysis_scripts.common import ab_arms
        a_specs, _ = ab_arms('groot', info['r6cell'])
        run, arm = a_specs[0].split(':')
        A = stream(run, arm)
        res = {}
        for lever in ('SF1', 'UF1'):
            L = stream('r07_main', f'r7_{cell}_{lever}')
            rows = []
            for k, ea in A.items():
                el = L.get(k)
                if el is None:
                    continue
                da = {int(d['step']): d for d in ea['decisions']}
                dl = {int(d['step']): d for d in el['decisions']}
                s_star = None
                for s in sorted(set(da) & set(dl)):
                    ha, hl = da[s].get('served_head'), dl[s].get('served_head')
                    if ha is None or hl is None:
                        continue
                    if np.abs(np.asarray(ha) - np.asarray(hl)).max() > 1e-6 or bool(da[s].get('vision')) != bool(dl[s].get('vision')):
                        s_star = s
                        break
                rec = dict(task=k[0], init=k[1], YA=ea['Y'], YL=el['Y'], s_star=s_star)
                if s_star is not None:
                    ha = np.asarray(da[s_star]['served_head'], float)[:5, :7]
                    hl = np.asarray(dl[s_star]['served_head'], float)[:5, :7]
                    z = (hl - ha) / asig
                    rec['d0'] = float(np.sqrt(np.mean(z[:, :6] ** 2)))
                    rec['grip0'] = bool(np.sign(ha[:, 6].mean()) != np.sign(hl[:, 6].mean()))
                    rec['lever_blind'] = not bool(dl[s_star].get('vision'))
                    rec['a_vision'] = bool(da[s_star].get('vision'))
                    for kk in KS:
                        s2 = s_star + kk
                        if s2 in da and s2 in dl and da[s2].get('robot_state') and dl[s2].get('robot_state'):
                            xa = np.asarray(da[s2]['robot_state'], float)[:8]
                            xl = np.asarray(dl[s2]['robot_state'], float)[:8]
                            rec[f'sd{kk}'] = float(np.sqrt(np.mean(((xa - xl) / ssig) ** 2)))
                rows.append(rec)
            C.write_csv(SCRATCH / f'twin_{cell}_{lever}.csv.gz', rows)
            div = [r for r in rows if r['s_star'] is not None]
            flip = [r for r in div if r['YA'] != r['YL']]
            same = [r for r in div if r['YA'] == r['YL']]
            summ = dict(pairs=len(rows), diverged=len(div), first_div_is_lever_blind=float(np.mean([r['lever_blind'] for r in div])) if div else None,
                        s_star_p50=float(np.median([r['s_star'] for r in div])) if div else None,
                        outcome_flips=len(flip), lossA_to_lever=sum(1 for r in flip if r['YA'] == 1),
                        d0_p50=float(np.median([r['d0'] for r in div])) if div else None,
                        grip0_share=float(np.mean([r['grip0'] for r in div])) if div else None,
                        auroc_d0_flip=auroc([r['d0'] for r in flip], [r['d0'] for r in same]))
            for kk in KS:
                v_all = [r[f'sd{kk}'] for r in div if f'sd{kk}' in r]
                summ[f'state_div_k{kk}_p50'] = float(np.median(v_all)) if v_all else None
                summ[f'state_div_k{kk}_p90'] = float(np.quantile(v_all, .9)) if v_all else None
                summ[f'auroc_sd{kk}_flip'] = auroc([r[f'sd{kk}'] for r in flip if f'sd{kk}' in r], [r[f'sd{kk}'] for r in same if f'sd{kk}' in r])
            res[lever] = summ
            print(cell, lever, json.dumps(summ), flush=True)
        out[cell] = res
        C.write_json(HERE / 'twin_summary.json', out)


if __name__ == '__main__':
    main()
