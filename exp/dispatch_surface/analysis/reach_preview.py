"""Offline reachability preview: which costs can a quantile ladder actually hit?

For a family (s-only or (s,v)) on a calibration table, sweep quantiles, export
the deployed boundaries and predict the per-decision cost with the frozen unit
costs. Reports the distinct operating points so a densification run can pick
quantiles by TARGET COST instead of by quantile value.
"""
import argparse
import numpy as np
from exp.dispatch_surface.fit_surface import (GRID_LADDER_S_ONLY, GRID_LADDER_SV,
                                              export_boundaries, final_fit, load_table)
from openpi.cache.components.surface_judge import surface_verdict

S1, S2, S3 = 10.260266, 27.686469, 29.571860
FULL, WARM, MISS = S1, S1 + S2 + 0.3 * S3, S1 + S2 + S3

ap = argparse.ArgumentParser()
ap.add_argument('--table', required=True)
ap.add_argument('--ref-mode', default='fresh')
ap.add_argument('--s-only', action='store_true')
ap.add_argument('--s-bins', type=int, default=0)
ap.add_argument('--qmin', type=float, default=0.50)
ap.add_argument('--qmax', type=float, default=0.99)
ap.add_argument('--steps', type=int, default=99)
ap.add_argument('--targets', default='',
                help='comma list of target cost percentages; print the quantile whose predicted '
                     'cost is closest to each target (and the achievable gap if none is close)')
ap.add_argument('--emit-quantiles', action='store_true',
                help='print only a comma list of the selected quantiles, ready for --quantiles')
a = ap.parse_args()

t = load_table(a.table, ref_mode=a.ref_mode)
dev = np.ones(len(t.s), dtype=bool)
if a.s_only:
    ladder = ((a.s_bins, 1), (12, 1)) if a.s_bins else GRID_LADDER_S_ONLY
else:
    ladder = GRID_LADDER_SV
ff = final_fit(t, dev, alpha=0.05, ladder=ladder)
if ff is None:
    raise SystemExit('grid ladder exhausted')
y10 = np.asarray(t.y10, dtype=np.float64)
v_edges = ff.v_edges
print(f"rows={len(t.s)} s_bins={len(ff.s_edges)-1} v_bins={len(v_edges)-1} ref={a.ref_mode}")
seen = {}
for q in np.linspace(a.qmin, a.qmax, a.steps):
    d = float(np.percentile(y10, 100 * q, method='linear'))
    full, warm = export_boundaries(ff.q_hat, ff.s_edges, d)
    costs = np.empty(len(t.s))
    for i in range(len(t.s)):
        verdict = surface_verdict(float(t.s[i]), float(t.v[i]), v_edges, full, warm,
                                  uses_disagreement=not a.s_only)
        costs[i] = FULL if verdict == 'full' else (WARM if verdict == 'warm' else MISS)
    pct = 100 * costs.mean() / MISS
    key = (tuple(np.round(full, 6)), tuple(np.round(warm, 6)))
    if key not in seen:
        seen[key] = (round(float(q), 4), pct)
points = sorted(seen.values(), key=lambda kv: -kv[1])
if a.targets:
    targets = [float(x) for x in a.targets.split(',')]
    picked, rows = [], []
    for tgt in targets:
        q, pct = min(points, key=lambda kv: abs(kv[1] - tgt))
        rows.append((tgt, q, pct))
        if q not in picked:
            picked.append(q)
    if a.emit_quantiles:
        print(','.join(f'{q:g}' for q in sorted(picked)))
    else:
        print(f"{'target%':>8} {'q':>8} {'pred%':>7} {'miss':>6}")
        for tgt, q, pct in rows:
            print(f"{tgt:>8.1f} {q:>8} {pct:>7.1f} {pct-tgt:>+6.1f}")
        print('\nselected quantiles:', ','.join(f'{q:g}' for q in sorted(picked)))
else:
    print(f"{'q':>8} {'pred cost%':>11}")
    for q, pct in points:
        print(f"{q:>8} {pct:>11.1f}")
