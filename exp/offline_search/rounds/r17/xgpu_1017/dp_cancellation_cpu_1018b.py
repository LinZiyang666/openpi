"""CPU-only DP early-metric cancellation diagnostic (1018b). No GPU, no model.

Uses (a) the sealed DP C10 fit shared by the 4090 B1 replay and the longseg
concurrent recipe (fit.npz sha b8bb9bc2...), (b) real 4090 B1 stage1 query
features of the three inputs_dp3 frames, and (c) the 51 worst 4090-vs-H100
cache pairs from DP_CACHE_LOCALISATION. It re-runs the frozen first-head
formula d2 = n20 - 2 Z (A0 y) + y.y in native float32 order, in float64, and in
the residual-preserving cancellation-free form, and measures:
  * summand magnitudes and the float32 ulp they imply for d2 and distance;
  * whether the observed cross-card d2 differences are integer ulp steps;
  * how weights move under realistic tiny feature perturbations in float32 vs
    float64 (the float64/stable formula cannot remove drift that is in y).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIT = Path('/data/mw_finetune/eval/ffk1015_dp_libero_full_formal_A_original/assets/cache_inputs/libero_spatial/n50_C10/fit')
B1 = HERE / 'results/4090_native_20261018T045343Z'
K, KREF = 16, 8


def load_fit(path):
    m = json.loads((path / 'READY.json').read_text())
    with np.load(path / 'fit.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}

    def dec(n):
        if 'array' in n:
            return np.array(arrays[n['array']], copy=True, order=n.get('order', 'C'))
        if 'namespace' in n:
            return dec(n['namespace'])
        if 'mapping' in n:
            return {k: dec(v) for k, v in n['mapping']}
        if 'sequence' in n:
            return [dec(v) for v in n['sequence']]
        return n['value']
    return dec(m['tree'])


def kernel(d, kref=KREF):
    d = np.asarray(d, np.float64)
    rel = d - d[0]
    w = np.exp(-(rel / max(rel[min(kref, len(rel)) - 1], 1e-6)) ** 2)
    return (w / w.sum()).astype(np.float32)


def project(fit, feats, state, dtype):
    cols = []
    for name in fit['fields']:
        B, shift = fit['pca'][name]
        cols.append(B.astype(dtype) @ np.asarray(feats[name], dtype) - shift.astype(dtype))
    return np.concatenate(cols + [np.asarray(state, dtype)])


def distances(fit, task, x, mode):
    T = fit['tasks'][task]
    E = T['early']
    if mode == 'native32':
        x = x.astype(np.float32)
        y = x @ E['W0f'] - E['c0']
        d2 = E['n20'] - 2.0 * (T['Z'] @ (E['A0'] @ y)) + float(y @ y)
    else:
        x = x.astype(np.float64)
        y = x @ E['W0f'].astype(np.float64) - E['c0'].astype(np.float64)
        Z, A, n20 = T['Z'].astype(np.float64), E['A0'].astype(np.float64), E['n20'].astype(np.float64)
        if mode == 'float64':
            d2 = n20 - 2.0 * (Z @ (A @ y)) + float(y @ y)
        else:  # stable: same metric incl. stored-n20 rounding residual, no cancellation
            u = Z @ A
            d2 = np.sum((u - y[None]) ** 2, 1) + (n20 - np.sum(u * u, 1))
    return np.sqrt(np.maximum(d2, 0.0)), d2, y


def choose(d, rows):
    order = np.lexsort((np.arange(len(d)), d))[:K]
    return rows[order], d[order]


def b1_rows(directory, fit):
    rows = {}
    for p in sorted(Path(directory).glob('row_*_repeat_*.json')):
        tree = json.loads(p.read_text())['tree']
        with np.load(p.with_suffix('.npz'), allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        qry = tree['stages']['query'] if 'query' in tree['stages'] else tree['query']
        feats = {n: a[qry[n]['$array']].reshape(-1).astype(np.float32) for n in fit['fields']}
        state = a[qry['robot_state']['$array']].reshape(-1)
        q = tree['cache']
        rows[p.stem] = dict(feats=feats, state=state, rec_d=a[q['distances']['$array']].astype(np.float64),
                            rec_rows=a[q['neighbours']['$array']], rec_w=a[q['weights']['$array']].astype(np.float64))
    return rows


def cross_b1(fit, task, left, right):
    """4090 vs A100 B1 (same tool, inputs, recipe, fit): weights per arithmetic mode."""
    T = fit['tasks'][task]
    rows = T['rows']
    L, R = b1_rows(left, fit), b1_rows(right, fit)
    out = []
    for key in sorted(set(L) & set(R)):
        l, r = L[key], R[key]
        frel = max(float(np.max(np.abs(l['feats'][n] - r['feats'][n])) / float(np.max(np.abs(l['feats'][n])))) for n in fit['fields'])
        row = dict(row=key, feature_rel_max_diff=frel, feature_bit_equal=all(np.array_equal(l['feats'][n], r['feats'][n]) for n in fit['fields']),
                   recorded_rows_equal=bool(np.array_equal(l['rec_rows'], r['rec_rows'])),
                   recorded_d_max_diff=float(np.max(np.abs(l['rec_d'] - r['rec_d']))),
                   recorded_w_max_diff=float(np.max(np.abs(l['rec_w'] - r['rec_w']))))
        for side, x in (('left', l), ('right', r)):
            d, _, _ = distances(fit, task, project(fit, x['feats'], x['state'], np.float32), 'native32')
            rr, dk = choose(d, rows)
            row[side + '_local_native32_reproduces_recorded'] = bool(np.array_equal(rr, x['rec_rows']) and np.array_equal(dk, x['rec_d']))
        for mode in ('native32', 'float64', 'stable'):
            res = []
            for x in (l, r):
                d, _, _ = distances(fit, task, project(fit, x['feats'], x['state'], np.float32), mode)
                res.append(choose(d, rows))
            (r0, d0), (r1, d1) = res
            w0 = dict(zip(r0.tolist(), kernel(d0).tolist())); w1 = dict(zip(r1.tolist(), kernel(d1).tolist()))
            row[mode] = dict(same_donors=bool(np.array_equal(r0, r1)), max_d_diff=float(np.max(np.abs(d0 - d1))),
                             max_w_diff=max(abs(w0.get(i, 0) - w1.get(i, 0)) for i in set(w0) | set(w1)))
        out.append(row)
    return out


def main():
    fit = load_fit(FIT)
    task = 0
    T = fit['tasks'][task]
    rows = T['rows']
    out = dict(schema='xgpu.dp.cancellation_cpu.v1', fit=str(FIT), task=task, k=K, kref=KREF, b1=[], magnitudes={},
               cross_card_quantisation={}, perturbation={})
    # (b) real 4090 B1 features: reproduce recorded first-head distances.
    vals = []
    for p in sorted(B1.glob('row_*_repeat_*.json')):
        tree = json.loads(p.read_text())['tree']
        with np.load(p.with_suffix('.npz'), allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        q = tree['cache']
        rec_d = a[q['distances']['$array']]
        rec_rows = a[q['neighbours']['$array']]
        qry = tree['stages']['query'] if 'query' in tree['stages'] else tree['query']
        feats = {n: a[qry[n]['$array']].reshape(-1) for n in fit['fields']}
        state = a[qry['robot_state']['$array']].reshape(-1)
        x = project(fit, feats, state, np.float32)
        res = {}
        for mode in ('native32', 'float64', 'stable'):
            d, d2, y = distances(fit, task, x, mode)
            r, dk = choose(d, rows)
            res[mode] = dict(rows=r.tolist(), d=dk.tolist(), w=kernel(dk).tolist())
        y64 = distances(fit, task, x, 'float64')[2]
        n20 = T['early']['n20'].astype(np.float64)
        u = T['Z'].astype(np.float64) @ T['early']['A0'].astype(np.float64)
        cross = 2.0 * (u @ y64)
        vals.append(dict(n20_max=float(n20.max()), yy=float(y64 @ y64), cross_max=float(np.abs(cross).max()),
                         d2_min=float(np.min(res['float64']['d'])) ** 2))
        out['b1'].append(dict(file=p.name, input=json.loads(p.read_text())['tree']['request'].get('decision_seq'),
                              recorded_rows=rec_rows.tolist(), recorded_d=rec_d.tolist(),
                              native32_reproduces_recorded=bool(np.array_equal(np.asarray(res['native32']['rows']), rec_rows)
                                                                and np.allclose(res['native32']['d'], rec_d, rtol=0, atol=0)),
                              native32_vs_recorded_max_abs=float(np.max(np.abs(np.asarray(res['native32']['d']) - rec_d))),
                              float64_vs_native32_d_max=float(np.max(np.abs(np.asarray(res['float64']['d']) - np.asarray(res['native32']['d'])))),
                              stable_vs_float64_d_max=float(np.max(np.abs(np.asarray(res['stable']['d']) - np.asarray(res['float64']['d'])))),
                              float64_vs_native32_w_max=float(np.max(np.abs(np.asarray(res['float64']['w']) - np.asarray(res['native32']['w'])))),
                              same_donors_f64_vs_f32=res['float64']['rows'] == res['native32']['rows'], modes=res))
    mags = dict(n20_max=max(v['n20_max'] for v in vals), yy=max(v['yy'] for v in vals),
                cross_max=max(v['cross_max'] for v in vals), d2_min=min(v['d2_min'] for v in vals))
    big = max(mags['n20_max'], mags['yy'], mags['cross_max'])
    ulp = float(np.spacing(np.float32(big)))
    mags.update(largest_summand=big, float32_ulp_of_largest=ulp, d=float(np.sqrt(mags['d2_min'])),
                distance_step_per_ulp=ulp / (2 * np.sqrt(mags['d2_min'])),
                cancellation_ratio=big / mags['d2_min'])
    out['magnitudes'] = mags
    # (c) observed 4090 vs H100 worst pairs: are d2 differences ulp steps?
    loc = json.loads((HERE / 'evidence/dp_cache_localisation_1018.json').read_text())
    steps = []
    for r in loc['rows']:
        w = r['worst_pair']
        L = np.asarray(w['left_distances'], np.float32).astype(np.float64)
        R = np.asarray(w['right_distances'], np.float32).astype(np.float64)
        dd2 = R ** 2 - L ** 2
        # resolution of reconstructing d2 from float32-rounded distances
        tol = 2 * np.maximum(L, R) * np.spacing(np.float32(46.0))
        best = None
        for e in range(-3, -10, -1):  # largest quantum first
            q = 2.0 ** e
            frac = np.abs(dd2 / q - np.round(dd2 / q)) * q
            if np.all(frac <= tol + 1e-12):
                best = q
                break
        steps.append(dict(key=r['key'], max_abs_d_diff=float(np.max(np.abs(R - L))), quantum=best,
                          multiples=(np.round(dd2 / best).astype(int).tolist() if best else None)))
    qs = [s['quantum'] for s in steps]
    out['cross_card_quantisation'] = dict(pairs=len(steps), all_integer_ulp_steps=all(q is not None for q in qs),
                                          quanta=sorted({q for q in qs if q}), rows=steps)
    # Perturbation: how much of weight drift is float32 cancellation vs y drift.
    p0 = out['b1'][0]
    tree = json.loads((B1 / p0['file']).read_text())['tree']
    with np.load((B1 / p0['file']).with_suffix('.npz'), allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    qry = tree['stages']['query'] if 'query' in tree['stages'] else tree['query']
    feats = {n: a[qry[n]['$array']].reshape(-1).astype(np.float32) for n in fit['fields']}
    state = a[qry['robot_state']['$array']].reshape(-1)
    rng = np.random.default_rng(0)
    ref = {m: choose(distances(fit, task, project(fit, feats, state, np.float32), m)[0], rows) for m in ('native32', 'float64')}
    for rel in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5):
        drift = {'native32': [], 'float64': []}
        flips = {'native32': 0, 'float64': 0}
        for _ in range(200):
            pf = {n: (v * (1 + rel * rng.standard_normal(v.shape))).astype(np.float32) for n, v in feats.items()}
            x = project(fit, pf, state, np.float32)
            for m in drift:
                r, dk = choose(distances(fit, task, x, m)[0], rows)
                r0, d0 = ref[m]
                if not np.array_equal(r, r0):
                    flips[m] += 1
                    continue
                drift[m].append(float(np.max(np.abs(kernel(dk) - kernel(d0)))))
        out['perturbation'][str(rel)] = {m: dict(max_weight_drift=max(v) if v else None,
                                                 median_weight_drift=float(np.median(v)) if v else None,
                                                 donor_set_changes=flips[m]) for m, v in drift.items()}
    a100 = sorted((HERE / 'a100_closedloop_1018/collected/b1_dp_a100').glob('a100_native_*'))
    if a100:
        out['cross_card_b1_4090_vs_a100'] = dict(left=str(B1), right=str(a100[-1]), rows=cross_b1(fit, task, B1, a100[-1]))
    target = HERE / 'evidence/dp_cancellation_cpu_1018b.json'
    target.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(magnitudes=mags, b1=[{k: v for k, v in b.items() if k not in ('modes', 'recorded_rows', 'recorded_d')}
                                               for b in out['b1']],
                          quantisation={k: v for k, v in out['cross_card_quantisation'].items() if k != 'rows'},
                          perturbation=out['perturbation'], cross_card_b1=out.get('cross_card_b1_4090_vs_a100',{}).get('rows')), indent=1))


if __name__ == '__main__':
    main()
