"""Shared helpers for the R6 round analysis (ANALYSIS.md). CPU only, read-only on run roots.

Conventions (rounds/r06/ANALYSIS_BRIEF.md):
- an arm is `run:arm` -> /home/weiland/trace_runs/os_closed_loop/<run>/runs/<arm>/{summary.json, client/journal.jsonl};
- outcome of a (task, init) pair = the accepted terminal journal record (status done/failed, no error);
- owner IR = c1*v + (1-c1)*m, c1 = .152 (pi0.5) / .148 (GR00T), from summary.cost_ledger counts; pure L10 = .5, L5 = 1;
- intervals: task-stratified init bootstrap, 10,000 draws, seed 20260929 (a fresh generator per contrast, so every
  contrast is reproducible on its own); replicates are averaged within each (task, init) pair before resampling;
- single-run pairs: exact two-sided McNemar; NI: conservative Clopper-Pearson paired bound
  CP_lower(wins/n; a/2) - CP_upper(losses/n; a/2), a = .05, margin .02 (Q2 / frontier definition).
"""
import json
import math
import os
from functools import lru_cache

import numpy as np
from scipy.stats import beta

ROOT = '/home/weiland/trace_runs/os_closed_loop'
R6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R6, 'analysis_r6')
C1 = {'pi05': .152, 'groot': .148}
SEED = 20260929
DRAWS = 10000
EXPECTED = {(t, i) for t in range(10) for i in range(50)}


def arm_dir(spec):
    run, arm = spec.split(':')
    return f'{ROOT}/{run}/runs/{arm}'


@lru_cache(None)
def journal(spec):
    """{(task, init): success} from accepted terminal records; raises on conflicting accepted duplicates."""
    out = {}
    with open(f'{arm_dir(spec)}/client/journal.jsonl') as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            uid = r.get('task_uid') or ''
            if ':eval:' not in uid or r.get('status') not in ('done', 'failed') or r.get('error') not in (None, ''):
                continue
            if r.get('accepted') is False:
                continue
            key = tuple(int(x) for x in uid.split(':eval:')[1].split(':')[:2])
            val = bool(r['success'])
            if key in out and out[key] != val:
                raise ValueError(f'conflicting accepted records {spec} {key}')
            out[key] = val
    return out


@lru_cache(None)
def summary(spec):
    with open(f'{arm_dir(spec)}/summary.json') as f:
        return json.load(f)


def model_of(spec):
    s = summary(spec)
    return s.get('model') or ('groot' if ':r5x_g' in spec or '_g_' in spec else 'pi05')


def ledger(spec):
    c = summary(spec)['cost_ledger']
    return int(c['decisions']), int(c['vision_decisions']), int(c['misses']), float(c['l_per_request']['mean'])


def owner_ir(spec, model=None):
    model = model or model_of(spec)
    N, V, M, L = ledger(spec)
    return (C1[model] * V / N + (1 - C1[model]) * M / N) * 5 / L


def sr(o):
    return sum(o.values()) / len(o)


def avg(specs):
    """Per-pair mean outcome over replicate runs (intersection of keys)."""
    js = [journal(s) for s in specs]
    keys = set.intersection(*[set(j) for j in js])
    return {k: sum(j[k] for j in js) / len(js) for k in keys}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wl(x, y):
    keys = set(x) & set(y)
    w = sum(1 for k in keys if x[k] and not y[k])
    l = sum(1 for k in keys if y[k] and not x[k])
    return w, l, len(keys)


def boot(x, y, seed=SEED, draws=DRAWS):
    """Point and bootstrap draws of mean(x - y) over common pairs, resampling inits within each task."""
    keys = sorted(set(x) & set(y))
    tasks = sorted({k[0] for k in keys})
    d = {t: np.array([x[k] - y[k] for k in keys if k[0] == t], dtype=float) for t in tasks}
    n = sum(len(v) for v in d.values())
    rng = np.random.default_rng(seed)
    sims = np.zeros(draws)
    for t in tasks:
        v = d[t]
        idx = rng.integers(0, len(v), size=(draws, len(v)))
        sims += v[idx].sum(axis=1)
    sims /= n
    return float(np.mean(np.concatenate([d[t] for t in tasks]))), sims


def ci(sims):
    return float(np.quantile(sims, .025)), float(np.quantile(sims, .975))


def boot_ci(x, y):
    p, s = boot(x, y)
    lo, hi = ci(s)
    return p, lo, hi, s


def ni_lower(x, y, alpha=.05):
    """Conservative exact paired risk-difference lower bound for binary single-run pairs."""
    w, l, n = wl(x, y)
    lo = beta.ppf(alpha / 2, w, n - w + 1) if w > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, l + 1, n - l) if l < n else 1.0
    return float(lo - hi), w, l, n


def pp(v, nd=1):
    return f'{100 * v:+.{nd}f}'


def f3(v):
    return f'{v:.3f}'


def dump(name, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(data, f, indent=1, default=float)
        f.write('\n')


# ---- A / B replicate identities (identical to ops/paper_ab.py:arms) ----
CELLS = [('pi05', 'l10_50'), ('pi05', 'l10_500'), ('pi05', 'sp_50'), ('pi05', 'sp_500'),
         ('groot', 'l10_50'), ('groot', 'l10_500'), ('groot', 'sp_50'), ('groot', 'sp_500')]
CELLNAME = {'l10_50': 'L10-50', 'l10_500': 'L10-500', 'sp_50': 'Sp-50', 'sp_500': 'Sp-500'}
MODELNAME = {'pi05': 'π0.5', 'groot': 'GR00T'}


def ab_arms(model, cell):
    if model == 'groot':
        a = [f'r05_x:r5x_g_{cell}_tail1u'] + [f'r06_paper:r5x_g_{cell}_tail1u_rep{k}' for k in (2, 3)]
        b = [f'r06_paper:r6p1_c10_g_{cell}'] + [f'r06_paper:r6p1_c10_g_{cell}_rep{k}' for k in (2, 3)]
    else:
        a1 = 'r04_blind:r4b3_p_sp_500_tail1uc' if cell == 'sp_500' else f'r05_ptail:r5t_p_{cell}_tail1uc'
        stem = a1.split(':')[1]
        a = [a1] + [f'r06_paper:{stem}_rep{k}' for k in (2, 3)]
        b = [f'r05_q1:r5q1_c10_p_{cell}'] + [f'r06_paper:r5q1_c10_p_{cell}_rep{k}' for k in (2, 3)]
    return a, b


def l10_ref(model, cell):
    s = 'l10' if cell.startswith('l10') else 'sp'
    return f'r04_cost:r4f_p_{s}_inf_k10_L10' if model == 'pi05' else \
        f"r05_q2:r5q2_g_{'l10' if s == 'l10' else 'spatial'}_policy_L10"


def l5_refs(model, cell):
    """In-root L=5 references (pi0.5 only; DUAL references live outside os_closed_loop)."""
    s = 'l10' if cell.startswith('l10') else 'sp'
    return [f'r04_cost:r4f_p_{s}_inf_s1001', f'r04_cost:r4f_p_{s}_inf_s2001'] if model == 'pi05' else []


def cname(model, cell):
    return f'{MODELNAME[model]} {CELLNAME[cell]}'


def c_cell(model, cell):
    """Configuration-C naming: pi05_l10_50, groot_spatial_500, ..."""
    return f"{model}_{cell.replace('sp_', 'spatial_')}"
