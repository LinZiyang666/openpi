"""Descriptive camera deletion with refitted 72-D action metrics, no rollouts.

Same-camera cached PCA; source episode excluded from retrieval only. This is
explicitly not fully refitted LOEO and cannot establish safety or SR.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from analyze_lazy import load_bank, OUT
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric, _kernel_w, PCA_CUR, PCA_BIG


def main(model, cell):
    t0 = time.time()
    awm, lib, rs, scale, succ, libpath, fitpath = load_bank(model, cell)
    key = model + '_' + ('spatial' if cell.startswith('sp_') else 'l10')
    root = PCA_CUR / key if awm.cand_name == 'current' else PCA_BIG / key / awm.cand_name
    p = [np.load(root / f'v{i}' / 'proj.npy')[:, :64] for i in range(2)]
    assert all(len(x) == len(rs) for x in p)
    actions = np.asarray(lib['action'][:, :, :7], float)
    H = (actions[:, :5]/awm.sig).reshape(len(rs), -1)
    future_motion = np.zeros(len(rs))
    valid = succ[1] >= 0
    future_motion[valid] = np.sqrt(np.mean(((rs[succ[1, valid]] - rs[valid])/scale)**2, axis=1))
    cut = float(np.median(future_motion[valid & lib['success']]))
    sign = actions[:, :, 6] >= 0
    transitions = (sign != sign[:, :1]).any(axis=1)
    prev = lib['prev']
    transitions |= (prev >= 0) & (sign[:, 0] != sign[np.maximum(prev, 0), 4])
    stages = np.where(transitions, 'gripper_change', np.where(future_motion <= cut, 'low_motion',
                  np.where(sign[:, 0], 'moving_positive_grip', 'moving_negative_grip')))
    all_errors, eps, labels, masks = [], [], [], []
    for task, T in awm.tasks.items():
        rows = np.asarray(T.rows)
        ep = lib['episode'][rows]
        selected = np.flatnonzero((lib['step'][rows] > 0) & (lib['step'][rows] % 2 == 0) & lib['success'][rows])
        Zs = [np.asarray(T.Z, float)]
        for cam in [0, 1]:
            x = np.concatenate([p[cam][rows], rs[rows]], axis=1)
            mu, sd, w = fit_metric(x, H[rows], ep, nn=awm.nn, lam=awm.lam, rank=awm.codes)
            Zs.append(((x-mu)/sd) @ w)
        err = np.zeros((len(selected), 3, 3))
        for m, z in enumerate(Zs):
            for j, i in enumerate(selected):
                d = np.linalg.norm(z-z[i], axis=1)
                d[ep == ep[i]] = np.inf
                ix = np.argpartition(d, 15)[:16]
                ix = ix[np.lexsort((rows[ix], d[ix]))]
                w = _kernel_w(d[ix]-d[ix[0]], awm.kref); w /= w.sum()
                mix = np.einsum('n,nhd->hd', w, actions[rows[ix]])
                err[j, m, 0] = np.sqrt(np.mean(((mix[:5]-actions[rows[i], :5])/awm.sig)**2))
                err[j, m, 1] = np.sqrt(np.mean(((mix-actions[rows[i]])/awm.sig)**2))
                err[j, m, 2] = np.mean((mix[:, 6] >= 0) != sign[rows[i]])
        all_errors.append(err); eps.append(ep[selected]); labels.append(stages[rows[selected]])
        masks.append(rows[selected])
    errs, eps, labels = np.concatenate(all_errors), np.concatenate(eps), np.concatenate(labels)
    result = dict(cell=model+'_'+cell, library=libpath, fit=fitpath, rows=len(errs),
                  caveat='frozen PCA/metric; candidate-only LOEO; library action imitation, not SR',
                  motion_median=cut, metrics=['head5_sigma_rms', 'whole_chunk_sigma_rms', 'gripper_sign_mismatch'],
                  modes=['both', 'third_person', 'wrist'], stages={})
    for stage in ['all'] + sorted(set(labels)):
        mask = np.ones(len(labels), bool) if stage == 'all' else labels == stage
        out = dict(rows=int(mask.sum()), episodes=len(np.unique(eps[mask])), row_mean=errs[mask].mean(axis=0).tolist())
        means = np.array([errs[mask & (eps == e)].mean(axis=0) for e in np.unique(eps[mask])])
        out['episode_mean'] = means.mean(axis=0).tolist()
        for m, cam in [(1, 'third_person'), (2, 'wrist')]:
            diff = means[:, m, 0]-means[:, 0, 0]
            out[cam+'_head_delta_mean'] = float(diff.mean())
            out[cam+'_head_delta_se'] = float(diff.std(ddof=1)/np.sqrt(len(diff)))
        result['stages'][stage] = out
    result['elapsed_s'] = time.time()-t0
    (OUT/f'cameras_{model}_{cell}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cell=result['cell'], elapsed_s=result['elapsed_s'], all=result['stages']['all'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cells', nargs='+', default=['l10_50', 'l10_500', 'sp_50', 'sp_500'])
    a = ap.parse_args()
    for cell in a.cells:
        main(a.model, cell)
