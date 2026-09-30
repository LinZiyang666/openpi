"""R7 E3 descriptive replay. No simulator, deployment changes, or SR estimator.

Frozen A metric; leave source episode out of candidate retrieval, not PCA/metric
fitting. All future states are recorded under their original controller.
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r06.p3_profiling.campaign import make_kwargs

STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
OUT = Path(__file__).parent
SCRATCH = Path('/tmp/r7_E3_lazy_levers')
AGES = np.array([1, 2, 3, 4, 6, 8, 9, 12])


def quant(x, qs=(.5, .9, .95)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.quantile(x, qs).tolist() if len(x) else [None] * len(qs)


def load_bank(model, cell):
    kw = make_kwargs(model, cell)
    with open(kw['base_fit'], 'rb') as f:
        awm = pickle.load(f)['method']
    suite = 'spatial' if cell.startswith('sp_') else 'l10'
    root = STORE / f'{model}_{suite}' / awm.cand_name
    lib = {k: np.load(root / (k + '.npy'), mmap_mode='r') for k in
           ['rs', 'action', 'next', 'prev', 'episode', 'step', 'task_id', 'success', 'ep_len']}
    rs = np.asarray(lib['rs'][:, :8], float)
    scale = np.std(rs[np.asarray(lib['success'], bool)], axis=0)
    active = scale > np.finfo(float).eps * max(scale.max(), 1)
    scale[~active] = 1
    nex = np.asarray(lib['next']).copy()
    i = np.flatnonzero(nex >= 0)
    j = nex[i]
    good = ((lib['episode'][i] == lib['episode'][j]) &
            (lib['task_id'][i] == lib['task_id'][j]) & (lib['step'][j] == lib['step'][i] + 1))
    nex[i[~good]] = -1
    succ = [np.arange(len(rs))]
    for _ in range(int(AGES.max())):
        old = succ[-1]
        succ.append(np.where(old >= 0, nex[np.maximum(old, 0)], -1))
    return awm, lib, rs, scale, np.asarray(succ), str(root), str(kw['base_fit'])


def loeo_neighbours(awm, lib):
    starts, members, weights = [], [], []
    for task, t in awm.tasks.items():
        candidates = np.asarray(t.rows)
        eligible = np.flatnonzero((lib['step'][candidates] % 2 == 0) & lib['success'][candidates])
        z = np.asarray(t.Z, float)
        z2 = np.sum(z*z, axis=1)
        ep = lib['episode'][candidates]
        for i in eligible:
            if lib['step'][candidates[i]] == 0 and awm.early:
                zz = np.asarray(t.Z0, float) if t.Z0 is not None else z @ t.A0
                if t.Z0 is None and t.As0 is not None:
                    zz = zz + np.asarray(t.RS, float) @ t.As0
                dist = np.linalg.norm(zz - zz[i], axis=1)
            else:
                dist = np.sqrt(np.maximum(z2 + z2[i] - 2*z @ z[i], 0))
            dist[ep == ep[i]] = np.inf
            ix = np.argpartition(dist, 15)[:16]
            ix = ix[np.lexsort((candidates[ix], dist[ix]))]
            w = _kernel_w(dist[ix] - dist[ix[0]], awm.kref)
            starts.append(candidates[i]); members.append(candidates[ix]); weights.append(w / w.sum())
    return np.asarray(starts), np.asarray(members), np.asarray(weights)


def residuals(states, members, weights, rs, scale, succ):
    """states[N, 13, 8]; NaN future observations stay censored, never terminal-pad."""
    n = len(states)
    delta = np.full((n, len(succ)), np.nan)
    absolute = delta.copy()
    supports = np.zeros_like(delta, bool)
    mean0 = np.einsum('nk,nkd->nd', weights, rs[members])
    for h in range(len(succ)):
        future = succ[h, members]
        ok = (future >= 0).all(axis=1) & np.isfinite(states[:, h]).all(axis=1)
        supports[:, h] = ok
        mu = np.einsum('nk,nkd->nd', weights[ok], rs[future[ok]])
        absolute[ok, h] = np.sqrt(np.mean(((states[ok, h] - mu) / scale)**2, axis=1))
        delta[ok, h] = np.sqrt(np.mean(((states[ok, h] - states[ok, 0] - mu + mean0[ok])/scale)**2, axis=1))
    return delta, absolute, supports


def rows_for(root, table, p3=False):
    """Tables are arm-contiguous. P3 A_r0 is first: read only that arm."""
    seen = False
    with (root / (table + '.csv')).open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if p3 and not r['arm'].endswith('_A_r0'):
                if seen:
                    break
                continue
            seen = True
            yield r


def recordings(model, cell, p3=False):
    name = model + '_' + (cell if p3 else cell.replace('sp_', 'spatial_'))
    run = 'r06_p3_pilot' if p3 else 'r06_c_cal'
    root = RUNS / run / 'tables' / name
    cache = SCRATCH / (run + '_' + name + '.npz')
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    dec = {}
    for r in rows_for(root, 'decisions', p3):
        with np.load(r['absolute_input_archive']) as z:
            dec[r['uid'], int(r['step'])] = np.array(z['robot_state'][:8], float)
    states, members, weights, uid, step, success, task, raw = [], [], [], [], [], [], [], []
    for r in rows_for(root, 'anchors', p3):
        key, s = r['uid'], int(r['step'])
        assert r['treatment'] in ('CACHE', 'cache', '') if 'treatment' in r else True
        states.append([dec.get((key, s+h), np.full(8, np.nan)) for h in range(13)])
        members.append(json.loads(r['retrieval.rows']))
        weights.append(json.loads(r['retrieval.weights']))
        uid.append(key); step.append(s); success.append(int(r['Y'])); task.append(int(r['task_id']))
        raw.append(json.loads(r['state.raw']))
    result = {k: np.asarray(v) for k, v in dict(states=states, members=members, weights=weights,
              uid=uid, step=step, success=success, task=task, raw=raw).items()}
    np.savez_compressed(cache, **result)
    return result


def report_stream(data, delta, absolute, supports, qrow, qep):
    uid = data['uid']
    out = dict(episodes=len(np.unique(uid)), anchors=len(uid),
               successful_episodes=len(np.unique(uid[data['success'] == 1])), by_age={})
    for h in AGES:
        valid = supports[:, h]
        for k, threshold in [('row95', qrow), ('episode95', qep)]:
            alerts = np.any(delta[:, 1:h+1] > threshold, axis=1)
            if k == 'row95':
                out['by_age'][str(5*h)] = dict(n=int(valid.sum()), supported_share=float(valid.mean()),
                    delta_q=quant(delta[:, h]), absolute_q=quant(absolute[:, h]))
            o = out['by_age'][str(5*h)]
            o[k+'_cumulative_alert'] = float(alerts[valid].mean()) if valid.any() else None
            o[k+'_alert_success'] = float(alerts[valid & (data['success'] == 1)].mean()) if (valid & (data['success'] == 1)).any() else None
            o[k+'_alert_failure'] = float(alerts[valid & (data['success'] == 0)].mean()) if (valid & (data['success'] == 0)).any() else None
    # Renewal replay: A's anchors only; real fresh-anchor kernel resets at forced/horizon looks.
    # Forced look at a normally blind half-chunk would lack an executed A kernel,
    # so report this as a chunk-boundary-only schedule, not the proposed full valve.
    out['chunk_boundary_replay'] = {}
    for cap in [2, 4, 6]:
        nlooks, alerts, exhausted, ndec = 0, 0, 0, 0
        for ep in np.unique(uid):
            ix = np.flatnonzero(uid == ep)
            ix = ix[np.argsort(data['step'][ix])]
            anchor = None
            for i in ix:
                ndec += 2
                if anchor is None:
                    anchor = i; nlooks += 1; continue
                h = int(data['step'][i] - data['step'][anchor])
                stop = h >= cap or h >= delta.shape[1] or not supports[anchor, h]
                breach = h < delta.shape[1] and delta[anchor, h] > qrow
                if stop or breach:
                    nlooks += 1; alerts += bool(breach and not stop); exhausted += bool(stop)
                    anchor = i
        out['chunk_boundary_replay'][str(cap*5)] = dict(vision_share=nlooks/ndec,
             alert_looks=alerts, cap_or_end_looks=exhausted, looks=nlooks,
             assumed_full_decisions=ndec, caveat='two decisions per anchor, terminal partial control not corrected')
    return out


def run_cell(model, cell):
    begin = time.time()
    awm, lib, rs, scale, succ, libpath, fitpath = load_bank(model, cell)
    starts, members, weights = loeo_neighbours(awm, lib)
    future = succ[:, starts].T
    states = rs[np.maximum(future, 0)].copy()
    states[future < 0] = np.nan
    delta, absolute, supports = residuals(states, members, weights, rs, scale, succ)
    # Same threshold for all future ages, anchored to today's ten-control A commitment.
    finite = supports[:, 1:3].all(axis=1)
    maxima = np.max(delta[finite, 1:3], axis=1)
    qrow = float(np.quantile(maxima, .95, method='higher'))
    eps = lib['episode'][starts[finite]]
    emax = np.array([maxima[eps == e].max() for e in np.unique(eps)])
    qep = float(np.quantile(emax, .95, method='higher'))
    epmeta = {int(e): int(lib['success'][np.flatnonzero(lib['episode'] == e)[0]]) for e in np.unique(lib['episode'])}
    report = dict(model=model, cell=cell, library_path=libpath, fit=fitpath, horizon=awm.H,
        library_rows=len(rs), library_episodes=len(epmeta), successful_library_episodes=sum(epmeta.values()),
        state_scale=scale.tolist(), calibration=dict(q_row95=qrow, q_episode95=qep,
        anchors=int(finite.sum()), episodes=len(emax), convention='frozen-fit candidate-LOEO; max 5/10-control displacement error, pooled across tasks'),
        library={})
    for h in AGES:
        report['library'][str(5*h)] = dict(supported_share=float(supports[:, h].mean()),
              n=int(supports[:, h].sum()), delta_q=quant(delta[:, h]), absolute_q=quant(absolute[:, h]),
              row95_cumulative_alert=float(np.any(delta[:, 1:h+1] > qrow, axis=1)[supports[:, h]].mean()))
    # Original predicted chunk versus successor rows actually executed by the library.
    report['tail_successor_mismatch'] = {}
    for h in range(1, awm.H // 5):
        valid = succ[h] >= 0
        tail = lib['action'][valid, 5*h:5*(h+1), :7]
        head = lib['action'][succ[h, valid], :5, :7]
        rms = np.sqrt(np.mean(((tail-head)/awm.sig)**2, axis=(1, 2)))
        report['tail_successor_mismatch'][str(h*5)] = dict(rms_q=quant(rms), n=len(rms),
              gripper_sign_mismatch=float(np.mean((tail[:, :, 6] >= 0) != (head[:, :, 6] >= 0))))
    for name, p3 in [('bval', False), ('p3_A_r0', True)]:
        data = recordings(model, cell, p3)
        d, a, s = residuals(data['states'], data['members'], data['weights'], rs, scale, succ)
        report[name] = report_stream(data, d, a, s, qrow, qep)
        np.savez_compressed(SCRATCH / f'{model}_{cell}_{name}_residuals.npz',
                            delta=d, absolute=a, support=s, **data)
    report['elapsed_s'] = time.time()-begin
    (OUT / f'evidence_{model}_{cell}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cell=model+'_'+cell, elapsed_s=report['elapsed_s'], calibration=report['calibration'],
                         bval=report['bval']['by_age']['20'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['pi05', 'groot'], required=True)
    ap.add_argument('--cells', nargs='+', default=['l10_50', 'l10_500', 'sp_50', 'sp_500'])
    args = ap.parse_args()
    for cell in args.cells:
        run_cell(args.model, cell)
