"""Summarize B-only tuning CV, timestamp predictions, and seal pre-emission inputs."""
import argparse
import json
import numpy as np
from .boundary import HERE, install
from .offline import now, strength, weights, BINS
from exp.offline_search.rounds.r10.data import write_json, sha


def interval(a, rule):
    b = strength(a['d1'] / a['scale'], a['step'], rule)
    errors = np.stack([a['m0'], a['m0'] - a['yp'] + .25 * a['pp'],
                      a['m0'] - 2*b*a['yp'] + b*b*a['pp']], 1)
    eps = np.unique(a['ep'])
    e = np.array([errors[a['ep'] == ep].mean(0) for ep in eps])
    tasks = np.array([a['task'][np.flatnonzero(a['ep'] == ep)[0]] for ep in eps])
    rng = np.random.default_rng(20261002)
    boot = []
    for _ in range(2000):
        chosen = np.concatenate([rng.choice(np.flatnonzero(tasks == t), (tasks == t).sum(), replace=True)
                                 for t in range(10)])
        v = e[chosen].mean(0)
        boot.append(100 * (v[2] / v[1] - 1))
    return np.percentile(boot, [2.5, 97.5]).tolist()


def offline_report():
    selection = json.loads((HERE / 'selection.json').read_text())
    summaries = selection['summaries']; rule = selection['selected']['rule']
    for s in summaries:
        with np.load(HERE / 'cv' / s['cell_size'] / 'rows.npz') as z:
            a = {k: np.array(z[k]) for k in z.files}
        s['dist_vs_loeo_conditional_95CI_pct'] = interval(a, rule)
        s['RMSE'] = {v: float(np.sqrt(s[v])) for v in ('none', 'GC_loeo', 'GC_dist')}
    pooled = []
    names = ['step0'] + [f'[{lo:g},{hi:g})' for lo, hi in zip(BINS[:-1], BINS[1:])]
    for name in names:
        bins = [b for s in summaries for b in s['by_distance'] if b['bin'] == name]
        pooled.append(dict(bin=name, contributing_cell_sizes=len(bins), rows=sum(b['rows'] for b in bins),
            loeo_change_pct=float(np.mean([b['loeo_change_pct'] for b in bins])),
            dist_change_pct=float(np.mean([b['dist_change_pct'] for b in bins])),
            dist_vs_loeo_pct=float(np.mean([b['dist_vs_loeo_pct'] for b in bins]))))
    agg = dict(loeo_change_pct=float(np.mean([s['loeo_change_pct'] for s in summaries])),
               dist_change_pct=float(np.mean([s['dist_change_pct'] for s in summaries])),
               dist_vs_loeo_pct=float(np.mean([s['dist_vs_loeo_pct'] for s in summaries])),
               cells_improved=sum(s['GC_dist'] < s['GC_loeo'] for s in summaries),
               rows=sum(s['rows'] for s in summaries))
    write_json(HERE / 'offline_comparison.json', dict(timestamp_utc=now(), aggregate=agg,
        pooled_bins=pooled, cells=summaries, caveat='tuning CV; fixed full-subset PCA; CIs conditional on selected rule'))
    lines = ['# B-library episode-fold comparison', '',
        'Five outer episode folds; four inner episode folds calibrate each outer training set.',
        'Metric, sigma, residual head and distance reference exclude each outer validation episode.',
        'PCA is fixed to the deployed B-subset basis. Failed episodes are retained.',
        'No-corrector = same G cache synthesis without a residual; guard interventions are not simulated.',
        'MSE = episode-balanced mean normalized 10-step × 6-motion squared residual.',
        'Distance bins use held-out query d1 / inner-calibrated scalar; step 0 is separate.',
        'CV chooses the gate and reports it on those same folds. This is tuning CV, not independent evidence.',
        'Intervals are 2,000 task-stratified episode bootstraps, conditional on the chosen rule.', '',
        f"Selected rule: `{json.dumps(rule)}`. All 24 arms use this shape.",
        f"Equal-cell-size mean changes vs none: LOEO {agg['loeo_change_pct']:+.3f}%; dist {agg['dist_change_pct']:+.3f}%.",
        f"Dist improves over LOEO in {agg['cells_improved']}/24 cell-sizes. {agg['rows']:,} validation rows in total (nested sizes reuse episodes).", '',
        '| Cell-size | none MSE | LOEO MSE | dist MSE | LOEO vs none | dist vs none | dist vs LOEO [95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        ci = s['dist_vs_loeo_conditional_95CI_pct']
        lines.append(f"| {s['cell_size']} | {s['none']:.6f} | {s['GC_loeo']:.6f} | {s['GC_dist']:.6f} | "
            f"{s['loeo_change_pct']:+.2f}% | {s['dist_change_pct']:+.2f}% | {s['dist_vs_loeo_pct']:+.2f}% [{ci[0]:+.2f},{ci[1]:+.2f}] |")
    lines += ['', '## Distance bins, equal-cell-size means of relative changes', '',
              '| Calibrated-distance bin | Rows | Cell-sizes | LOEO vs none | dist vs none | dist vs LOEO |',
              '|---|---:|---:|---:|---:|---:|']
    for b in pooled:
        lines.append(f"| {b['bin']} | {b['rows']} | {b['contributing_cell_sizes']} | {b['loeo_change_pct']:+.2f}% | "
                     f"{b['dist_change_pct']:+.2f}% | {b['dist_vs_loeo_pct']:+.2f}% |")
    lines += ['', 'All per-cell-size bins: `distance_bins.csv`; row sufficient statistics: `cv/*/rows.npz`.',
        'Those statistics retain episode, original parent row, task, step, fold, raw d1, training-only scale,',
        'mean y², mean y·prediction and mean prediction², sufficient to reproduce every candidate MSE.',
        'Calibration artifacts contain one scalar per cell-size and no task-specific gate parameter.',
        'Scales use smaller held-out-fold donor libraries (outer training 80%; nested calibration 60% of full);',
        'no size extrapolation is applied. This tends to soften the gate relative to a full-library reference.',
        'The raw scale and fractional attenuation can differ between final serving and outer validation.',
        'The cutoff is in this held-out-episode reference, not in Opus’s in-library LOEO units.',
        'Vectorized float32 offline arithmetic is numerically close, not byte-identical, to single-query serving.',
        'Serving reuses the exact G retrieval and exact original head; selftests check its correction bytes.',
        'Action imitation error is not closed-loop success; no A or closed-loop trajectory was used.', '']
    (HERE / 'OFFLINE.md').write_text('\n'.join(lines))
    return selection, agg


def freeze():
    selection, agg = offline_report()
    timestamp = now()
    rule = selection['selected']['rule']
    text = f'''# Pre-emission prediction — GC_dist

UTC timestamp: {timestamp}

Frozen before emitting either r10_corr3 run. No closed-loop results, clients,
journals, summaries or server logs were read. Inputs are B libraries and their
nested subsets plus the permitted G/Stage-2b LOEO fits and arm specifications.

One variant, one global shape for all 24 cell-sizes:

`r = min geometric distance to the actual retrieved rows / held-out-episode scale`

`strength = 0.5 * clip((2.0 - r) / 1.25, 0, 1)` for step > 0.

Step 0 strength is zero. Thus strength is .5 at r <= .75, .4 at r=1,
.2 at r=1.5, and zero at r >= 2 or nonfinite distance. No task gate or
task-specific scale is added. Each cell-size has one equal-episode-weighted
median d1 from five folds with held-out episodes excluded from metric fitting
and candidates. Final G and Stage 2b LOEO head arrays remain unchanged.
The corrected anchor continues to supply the existing blind tail.

This shape won the 18 predeclared candidates by mean relative MSE across the
24 B-only tuning-CV panels. Mean change vs no corrector: LOEO
{agg['loeo_change_pct']:+.3f}%, GC_dist {agg['dist_change_pct']:+.3f}%; GC_dist
beats LOEO in {agg['cells_improved']}/24 panels. These are action errors,
not success rates. Candidate selection and these comparisons use the same CV;
PCA is fixed to the deployed B-subset basis. Calibration donor sets are smaller
than the final library, without a fitted size extrapolation. See OFFLINE.md.

Prediction fixed now: GC_dist should reduce overcorrection on distant looks and
is most likely to help or tie LOEO for GR00T and the smaller libraries. A large
closed-loop success gain is unlikely. It may lose a little of LOEO's useful bias
correction on the largest pi05 libraries; offline pi05 size-500 already shows
that tradeoff. At very large calibrated distances it returns exactly to the raw
G cache action. This does not make the entire trajectory equal to G, because
earlier corrections can change later states and no-progress guard timing.

No numerical closed-loop success lift or policy-call saving is claimed.
Drift augmentation is not used, preserving a clean strength-only ablation.
No evaluation is launched by this task.
'''
    with (HERE / 'PREDICTION.md').open('x') as f:
        f.write(text)
    paths = [HERE / 'PROTOCOL.md', HERE / 'PREDICTION.md', HERE / 'selection.json',
             HERE / 'offline_comparison.json', *sorted((HERE / 'calibration').glob('*.json')),
             HERE / 'method.py', HERE / 'offline.py']
    write_json(HERE / 'freeze.json', dict(timestamp_utc=timestamp, rule=rule,
               hashes={str(p): sha(p) for p in paths}))
    print('FROZEN', timestamp, rule)


if __name__ == '__main__':
    install()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['offline', 'freeze'])
    a = p.parse_args()
    freeze() if a.action == 'freeze' else offline_report()
