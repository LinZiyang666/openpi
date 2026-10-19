"""CPU-only replay of A100 closed-loop DP first-head captures (1018b).

Input: dp_distance_heads/*.npz|json from the two A100 cache cold starts (the
native formal observer of dp_distance_capture_1018.py). For every first head:
re-derive y from the captured stage1 features, re-run the frozen early-metric
distance in native float32, float64 and residual-preserving cancellation-free
form, and pair cold1/cold2 heads of the same episode (same task/init/seed and
first observation). Reports feature/y drift versus distance and kernel-weight
drift per arithmetic mode. Diagnosis only; the formal recipe is unchanged.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np  # noqa: E402


def kernel(d, kref):
    d = np.asarray(d, np.float64)
    rel = d - d[0]
    w = np.exp(-(rel / max(rel[min(kref, len(rel)) - 1], 1e-6)) ** 2)
    return (w / w.sum()).astype(np.float32)


def load(directory):
    heads = []
    for p in sorted(Path(directory).rglob('*.json')):
        m = json.loads(p.read_text())
        if 'npz_sha256' not in m:
            continue
        npz = p.with_suffix('.npz')
        assert hashlib.sha256(npz.read_bytes()).hexdigest() == m['npz_sha256']
        with np.load(npz, allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
        heads.append((m, z))
    return heads


def analyse_head(m, z):
    k, kref = m['k'], m['kref']
    cols = []
    for i, _ in enumerate(m['feature_fields']):
        cols.append(z[f'pca_B_{i}'] @ np.asarray(z[f'feature_{i}'], np.float32) - z[f'pca_shift_{i}'])
    rs = np.asarray(z['state'], np.float32)[z['valid_state_indices']]
    x32 = np.concatenate(cols + [rs])
    y32 = x32 @ z['W0f'] - z['c0']
    Z, A, n, y = z['Z'], z['A0'], z['n20'], z['y']
    native = n - 2.0 * (Z @ (A @ y)) + float(y @ y)
    x64 = np.concatenate([z[f'pca_B_{i}'].astype(np.float64) @ z[f'feature_{i}'].astype(np.float64)
                          - z[f'pca_shift_{i}'].astype(np.float64) for i, _ in enumerate(m['feature_fields'])]
                         + [rs.astype(np.float64)])
    y64 = x64 @ z['W0f'].astype(np.float64) - z['c0'].astype(np.float64)
    zz, aa, nn = Z.astype(np.float64), A.astype(np.float64), n.astype(np.float64)
    u = zz @ aa
    f64 = nn - 2.0 * (u @ y64) + float(y64 @ y64)
    stable = np.sum((u - y64[None]) ** 2, 1) + (nn - np.sum(u * u, 1))
    pos = z['pos']
    out = {}
    for name, d2 in (('native32', native), ('float64', f64), ('stable', stable)):
        d = np.sqrt(np.maximum(d2, 0))
        order = pos[np.lexsort((pos, d[pos]))[:k]]
        out[name] = dict(rows=z['rows'][order].tolist(), d=d[order].astype(np.float64).tolist(),
                         w=kernel(d[order], kref).tolist())
    big = float(max(np.abs(nn).max(), float(y64 @ y64), np.abs(2 * (u @ y64)).max()))
    return dict(key=[m['request'].get(q) for q in ('task_id', 'task_name', 'original_init_idx', 'seed', 'decision_seq', 'explicit_seed')],
                y_captured_reproduced=bool(np.array_equal(y32, y)),
                native32_reproduces_recorded_d2=bool(np.array_equal(native, z['d2'])),
                recorded_rows=z['rows'][z['chosen']].tolist(), y64=y64, feats=[np.asarray(z[f'feature_{i}']) for i, _ in enumerate(m['feature_fields'])],
                largest_summand=big, d2_min=float(np.min(f64[pos])), modes=out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cold1')
    p.add_argument('cold2')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sides = {}
    for side, d in (('cold1', a.cold1), ('cold2', a.cold2)):
        sides[side] = {}
        for m, z in load(d):
            h = analyse_head(m, z)
            sides[side].setdefault(json.dumps(h['key']), []).append(h)
    per = []
    for key in sorted(set(sides['cold1']) & set(sides['cold2'])):
        L, R = sides['cold1'][key], sides['cold2'][key]
        row = dict(key=json.loads(key), heads=[len(L), len(R)])
        fd = max(float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(x))), 1e-12))
                 for l, r in itertools.product(L, R) for x, y in zip(l['feats'], r['feats']))
        yd = max(float(np.max(np.abs(l['y64'] - r['y64']))) for l, r in itertools.product(L, R))
        row.update(feature_rel_max_drift=fd, y_max_abs_drift=yd)
        for mode in ('native32', 'float64', 'stable'):
            wd, dd, flips = [], [], 0
            for l, r in itertools.product(L, R):
                lm, rm = l['modes'][mode], r['modes'][mode]
                if lm['rows'] != rm['rows']:
                    flips += 1
                lw = dict(zip(lm['rows'], lm['w'])); rw = dict(zip(rm['rows'], rm['w']))
                wd.append(max(abs(lw.get(i, 0) - rw.get(i, 0)) for i in set(lw) | set(rw)))
                ld = dict(zip(lm['rows'], lm['d'])); rd = dict(zip(rm['rows'], rm['d']))
                common = set(ld) & set(rd)
                dd.append(max(abs(ld[i] - rd[i]) for i in common) if common else None)
            row[mode] = dict(max_weight_drift=max(wd), max_distance_drift=max(x for x in dd if x is not None),
                             donor_order_changes=flips)
        per.append(row)
    allh = [h for s in sides.values() for hs in s.values() for h in hs]

    def mx(mode, field):
        return max(r[mode][field] for r in per) if per else None
    report = dict(schema='xgpu.dp.capture_analysis.v1', heads=dict(cold1=sum(map(len, sides['cold1'].values())),
                  cold2=sum(map(len, sides['cold2'].values()))), paired_episodes=len(per),
                  all_y_reproduced=all(h['y_captured_reproduced'] for h in allh),
                  all_native32_reproduce_recorded_d2=all(h['native32_reproduces_recorded_d2'] for h in allh),
                  largest_summand_max=max(h['largest_summand'] for h in allh) if allh else None,
                  d2_min_min=min(h['d2_min'] for h in allh) if allh else None,
                  heads_with_nonzero_feature_drift=sum(1 for r in per if r['feature_rel_max_drift'] > 0),
                  feature_rel_max_drift=max((r['feature_rel_max_drift'] for r in per), default=None),
                  y_max_abs_drift=max((r['y_max_abs_drift'] for r in per), default=None),
                  summary={m: dict(max_weight_drift=mx(m, 'max_weight_drift'), max_distance_drift=mx(m, 'max_distance_drift'),
                                   episodes_with_donor_order_change=sum(1 for r in per if r[m]['donor_order_changes']))
                           for m in ('native32', 'float64', 'stable')},
                  per_episode=per, formal_recipe_changed=False,
                  scope='A100 same-card, closed-loop first heads of the C5 ff1015 DP recipe; not the 4090/H100 longseg arm')
    Path(a.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_episode'}, indent=1))


if __name__ == '__main__':
    main()
