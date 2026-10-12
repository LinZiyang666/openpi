"""Read-only CPU analysis for PURE_LEVEL_NOISE.md; results go to stdout only.

Run from the repository root: .venv/bin/python -B exp/offline_search/rounds/r17/knob/noise_analysis.py
No model/simulator imports, no writes, no experiment launches. Cost evidence is
already in the final test.json; skip per_step.jsonl when loading binary outcomes.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from math import ceil, floor
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom, binomtest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r17.knob import common as c

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/mw_finetune/eval')
SEED, DRAWS = 20261012, 100_000
SPECS = {
    'dp': ('dp_r3', ['knobtestdp', 'knobtestdp_r1', 'knobtestdp_r2'], 'hs5dp_ctl_142812'),
    'fw': ('fw_r2', ['knobtestfw4', 'knobtestfw_r1'], 'hs6fw_res_152931'),
    'dp_red': ('dp_red_r7', ['knobtestdp_red'] + [f'knobtestdp_red_r{i}' for i in range(1, 7)], 'hs5dp_ddim10s0'),
}


class OutcomeSources(c.cache.Sources):
    def obj(self, path):
        # Calibration bundles affect descriptive cost metadata, never outcomes.
        if Path(path).name == 'CACHE_STUDY.json':
            return {}
        return super().obj(path)

    def rows(self, path):
        if Path(path).name != 'per_step.jsonl':
            yield from super().rows(path)


def load(lane):
    p = ROOT / lane
    return c.cache.load_lane(p, c.cache.receipt_snapshot(p), OutcomeSources(), c.cache.SIZES)


def holm(ps):
    order = np.argsort(ps)
    adjusted = np.maximum.accumulate(np.asarray(ps)[order] * np.arange(len(ps), 0, -1))
    out = np.ones(len(ps))
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()


def pair_stats(y, z):
    gains = int(((y == 1) & (z == 0)).sum())
    losses = int(((y == 0) & (z == 1)).sum())
    m = gains + losses
    return dict(gains=gains, losses=losses, discordant=m, deficit=losses-gains,
                p_lower=float(binom.sf(losses-1, m, .5)),
                p_two=float(binomtest(losses, m, .5).pvalue) if m else 1.,
                prob_3_7=float(binom.cdf(floor((m+7)/2), m, .5) - binom.cdf(ceil((m+3)/2)-1, m, .5)))


def convolve_sparse(a, b):
    out = defaultdict(float)
    for x, px in a.items():
        for y, py in b.items():
            out[x+y] += px*py
    return dict(out)


def summarize_dist(dist, observed):
    keys = sorted(dist)
    cdf = np.cumsum([dist[k] for k in keys])
    return dict(observed=int(observed), mean=sum(k*v for k, v in dist.items()),
                ci95=[int(keys[np.searchsorted(cdf, p)]) for p in [.025, .975]],
                p_ge=sum(v for k, v in dist.items() if k >= observed))


def whole_run(tasks):
    """Exact relabeling of the shared control, independently by distinct task."""
    count, deficit = {0: 1.}, {0: 1.}
    obs_count = obs_deficit = 0
    for task in tasks:
        totals = task['matrix'].sum(axis=0).astype(int)
        ds = len(totals)*totals - totals.sum()
        ns = [int(((totals[j]-np.delete(totals, j)) >= 3).sum()) for j in range(len(totals))]
        count = convolve_sparse(count, {k:v/len(ns) for k, v in Counter(ns).items()})
        deficit = convolve_sparse(deficit, {int(k):v/len(ds) for k, v in Counter(ds).items()})
        obs_count += ns[0]
        obs_deficit += int(ds[0])
    return dict(count=summarize_dist(count, obs_count), deficit=summarize_dist(deficit, obs_deficit)), count, deficit


def init_randomization(tasks, seed):
    """Conditional exchangeability within each fixed init; shared control retained."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(DRAWS, dtype=np.int16)
    deficits = np.zeros(DRAWS, dtype=np.int16)
    results = []
    for task in tasks:
        x = task['matrix']
        ncols = x.shape[1]
        totals = np.zeros((DRAWS, ncols), dtype=np.int16)
        for s, frequency in Counter(x.sum(axis=1)).items():
            patterns = np.zeros((1, ncols), dtype=np.int16) if s == 0 else np.array([
                [int(j in indices) for j in range(ncols)] for indices in combinations(range(ncols), int(s))
            ], dtype=np.int16)
            for _ in range(frequency):
                totals += patterns[rng.integers(len(patterns), size=DRAWS)]
        d = totals[:, :1]-totals[:, 1:]
        counts += (d >= 3).sum(axis=1).astype(np.int16)
        deficits += d.sum(axis=1).astype(np.int16)
        obs = x[:, 0].sum() - x[:, 1:].sum(axis=0)
        # Conditional on each init's total, the control is Bernoulli(s / R).
        # Its Poisson-binomial tail is exact; no binomial equal-difficulty assumption.
        control_pmf = np.array([1.])
        for p in x.mean(axis=1):
            control_pmf = np.convolve(control_pmf, [1-p, p])
        results.append(dict(task=task['task'], sizes=task['sizes'], totals=x.sum(axis=0).tolist(),
                            p_mean_lower=float(control_pmf[int(x[:, 0].sum()):].sum()),
                            variable_inits=int(((x.sum(axis=1)>0) & (x.sum(axis=1)<ncols)).sum()),
                            control_null_sd=float(np.sqrt(sum(p*(1-p) for p in x.mean(axis=1)))),
                            init_success_histogram={str(k):int(v) for k,v in sorted(Counter(x.sum(axis=1)).items())},
                            p_all_at_least_observed_min=float((1+(d.min(axis=1) >= obs.min()).sum())/(DRAWS+1))))
    obs_count = sum(int(((t['matrix'][:, :1].sum(axis=0)-t['matrix'][:, 1:].sum(axis=0)) >= 3).sum()) for t in tasks)
    obs_deficit = sum(int((t['matrix'][:, :1].sum(axis=0)-t['matrix'][:, 1:].sum(axis=0)).sum()) for t in tasks)
    def stat(a, observed):
        return dict(observed=observed, mean=float(a.mean()), ci95=np.quantile(a, [.025, .975]).tolist(),
                    p_ge=float((1+(a >= observed).sum())/(DRAWS+1)))
    return dict(count=stat(counts, obs_count), deficit=stat(deficits, obs_deficit), tasks=results), counts, deficits


def main():
    groups, all_rows = {}, []
    combined_run_count, combined_run_deficit = {0: 1.}, {0: 1.}
    combined_init_count = np.zeros(DRAWS, dtype=int)
    combined_init_deficit = np.zeros(DRAWS, dtype=int)
    for gi, (g, (final, lanes, control)) in enumerate(SPECS.items()):
        final_cells = json.loads((HERE / f'test_results_{final}/test.json').read_text())['cells']
        first_cells = json.loads((HERE / f'test_results_{g}_r1/test.json').read_text())['cells']
        initial_pure = {(cell['suite'], cell['size'], name) for cell in first_cells
                        for name, t in cell['tasks'].items() if t['level'] == 'pure'}
        data = defaultdict(dict)
        for lane in lanes:
            for (model, size, group, suite), result in load(lane).items():
                c.merge(data[suite, size, group.rsplit('_', 1)[-1]], result['data'])
        refs = {}
        for (_, _, group, suite), result in load(control).items():
            assert group == 'control'
            refs.update(result['data'])
        tasks, rows = {}, []
        verified = 0
        for cell in final_cells:
            suite, size = cell['suite'], cell['size']
            for name, t in cell['tasks'].items():
                keys = [(suite, name, i) for i in range(50)]
                arm = data[suite, size, t['level']]
                assert all(k in arm and k in refs for k in keys), (g, suite, size, name)
                y = np.array([int(arm[k]['success']) for k in keys], dtype=np.int8)
                z = np.array([int(refs[k]['success']) for k in keys], dtype=np.int8)
                assert t['paired_n'] == 50 and not t['missing_inits'] and not t['missing_control_inits']
                assert int(y.sum()) == t['successes'] and int(z.sum()) == t['control']['successes'], (g, size, name)
                verified += 1
                if t['level'] != 'pure':
                    continue
                assert t['ir_proxy'] == 1 and t['library'] == 0
                row = dict(group=g, task=name, size=size, successes=int(y.sum()), control=int(z.sum()),
                           initial=(suite, size, name) in initial_pure, **pair_stats(y,z))
                rows.append(row)
                task = tasks.setdefault(name, dict(task=name, sizes=[], columns=[z], initial=[]))
                task['sizes'].append(size)
                task['columns'].append(y)
                task['initial'].append(row['initial'])
        tasklist = []
        initial_tasks = []
        repeat_pairs = []
        repeat_tasks = []
        for task in tasks.values():
            task['matrix'] = np.stack(task.pop('columns'), axis=1)
            tasklist.append(task)
            mask = [True]+task['initial']
            if sum(mask)>1:
                initial_tasks.append(dict(task=task['task'], sizes=[s for s,b in zip(task['sizes'], task['initial']) if b],
                                          matrix=task['matrix'][:,mask]))
            totals = task['matrix'][:, 1:].sum(axis=0)
            if len(totals) > 1:
                repeat_tasks.append(dict(task=task['task'], sizes=task['sizes'], successes=totals.tolist(),
                                         sd=float(totals.std(ddof=1)), control=int(task['matrix'][:, 0].sum())))
            for a,b in combinations(range(1, task['matrix'].shape[1]), 2):
                repeat_pairs.append(pair_stats(task['matrix'][:,a], task['matrix'][:,b]))
        wr, distc, distd = whole_run(tasklist)
        ir, ic, idef = init_randomization(tasklist, SEED+gi)
        iw, _, _ = whole_run(initial_tasks)
        ii, _, _ = init_randomization(initial_tasks, SEED+100+gi)
        combined_run_count = convolve_sparse(combined_run_count, distc)
        combined_run_deficit = convolve_sparse(combined_run_deficit, distd)
        combined_init_count += ic
        combined_init_deficit += idef
        task_p = [t['p_mean_lower'] for t in ir['tasks']]
        for t,p in zip(ir['tasks'], holm(task_p)):
            t['holm_within_group_tasks'] = p
        for row,p in zip(rows, holm([r['p_lower'] for r in rows])):
            row['holm_within_group_cells'] = p
        groups[g] = dict(verified_cells=verified, pure_cells=len(rows), unique_tasks=len(tasks),
                         initial_pure=sum(r['initial'] for r in rows),
                         flagged_initial=sum(r['initial'] and r['deficit'] >=3 for r in rows),
                         repeat_tasks=repeat_tasks, repeat_pairs=repeat_pairs,
                         whole_run=wr, init_randomization=ir,
                         initial_whole_run=iw, initial_init_randomization=ii)
        all_rows.extend(rows)
    for r,p in zip(all_rows, holm([r['p_lower'] for r in all_rows])):
        r['holm_all_pure'] = p
    oc = sum(r['deficit'] >=3 for r in all_rows)
    od = sum(r['deficit'] for r in all_rows)
    out = dict(seed=SEED, draws=DRAWS, groups=groups, rows=all_rows,
               combined_whole_run=dict(count=summarize_dist(combined_run_count,oc), deficit=summarize_dist(combined_run_deficit,od)),
               combined_init=dict(count=dict(observed=oc, mean=float(combined_init_count.mean()), ci95=np.quantile(combined_init_count,[.025,.975]).tolist(), p_ge=float((1+(combined_init_count>=oc).sum())/(DRAWS+1))),
                                  deficit=dict(observed=od, mean=float(combined_init_deficit.mean()), ci95=np.quantile(combined_init_deficit,[.025,.975]).tolist(), p_ge=float((1+(combined_init_deficit>=od).sum())/(DRAWS+1)))))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
