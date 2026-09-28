"""Read-only P3 v2 Q2 analysis. See PREREG.md; never launches collection.

Inputs are audited v2 CSVs, or explicitly finished arms passed to read_v2.
All outputs (including reader tables and matplotlib cache) stay beside this file.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
DOSES = dict(A=0., dose125=.125, dose25=.25, dose50=.5, P10=1.)
CORE = list(DOSES) + ['B']
COHORTS = CORE + ['factorial', 'window', 'dose_mix']
RHOS = [.10, .15, .20, .25]
C1 = dict(pi05=.152, groot=.148)
ALPHA = .05 / 296
EPSILON = .02
SEED = 26092802
METRICS = ['SR', 'IR', 'm', 'anchor_call_rate', 'm5', 'calls', 'anchors',
           'requests', 'active_5_controls', 'IR_request']
ARM_RE = re.compile(r'r6p3v2_(pi05|groot)_(l10|sp)_(50|500)_(A|dose125|dose25|dose50|P10|B|factorial|window|dose_mix)_r([012])(?:_.*)?$')


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            r = clean(row)
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def boolean(v):
    if v in (True, 'True', 'true', '1', 1):
        return True
    if v in (False, 'False', 'false', '0', 0):
        return False
    raise ValueError('missing/invalid boolean: ' + repr(v))


def owned(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError('Outputs must be inside ' + str(HERE))
    return p


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(r):
    return r['arm'], r.get('uid', r.get('task_uid')), int(r['attempt'])


def table_dirs(path):
    return [path] if (path/'episodes.csv').exists() else sorted(p.parent for p in path.glob('*/episodes.csv'))


def table_rows(path, name):
    for directory in table_dirs(path):
        with (directory/(name+'.csv')).open() as f:
            yield from csv.DictReader(f)


def arm_info(arm):
    m = ARM_RE.fullmatch(arm)
    if not m:
        raise ValueError('Unrecognized v2 arm ID: ' + arm)
    model, suite, size, cohort, block = m.groups()
    return dict(cell=f'{model}_{suite}_{size}', model=model, suite=suite,
                size=int(size), cohort=cohort, block=int(block))


def load_tables(path, arms=None, cell=None):
    """Stream large tables; do not load raw controls, images, or archives."""
    path = Path(path)
    audit = {str(p): json.loads((p/'audit.json').read_text()) for p in table_dirs(path)}
    episodes = {}
    for r in table_rows(path, 'episodes'):
        if arms and r['arm'] not in arms:
            continue
        meta = arm_info(r['arm'])
        if cell and meta['cell'] != cell:
            continue
        key = identity(r)
        if key in episodes:
            raise ValueError('Duplicate accepted episode')
        for field in ['controls_verified', 'environment_seed_verified', 'physical_transitions_available']:
            if not boolean(r[field]):
                raise ValueError('Q2 requires verified client controls/environment: ' + field)
        if r['collection_mode'] != 'client_verified':
            raise ValueError('Server-only IR is not accepted')
        task, init = int(r['task_id']), int(r['init'])
        if int(r['provenance.run_block']) != meta['block']:
            raise ValueError('Arm/block mismatch')
        split = 'calibration' if init % 5 == 0 else 'validation'
        if r['provenance.split'] != split:
            raise ValueError('Split mismatch')
        if int(r['client.orig_init_state_idx']) != init or int(r['client.task_id']) != task:
            raise ValueError('Original client identity mismatch')
        y, n, v, m, c = (int(r[f]) for f in ['Y', 'decisions', 'anchors', 'misses', 'active_controls'])
        if y not in (0, 1) or not (0 <= m <= v <= n) or not (0 < c <= 5*n):
            raise ValueError('Invalid outcome/counts')
        if c <= 5*(n-1):
            raise ValueError('More than terminal partial request')
        c1 = C1[meta['model']]
        ir = (c1*v + (1-c1)*m)/(c/5)
        if not np.isclose(ir, float(r['deployment_IR_per_actual_5_controls']), atol=1e-10):
            raise ValueError('Reader/owner cost mismatch')
        dose = float(r['episode_assignment.dose'])
        prob = float(r['episode_assignment.probability'])
        random = boolean(r['episode_assignment.random'])
        if meta['cohort'] == 'dose_mix':
            if dose not in DOSES.values() or not np.isclose(prob, .2) or not random:
                raise ValueError('Unexpected episode randomization; version protocol before using')
        elif not np.isclose(prob, 1) or random:
            raise ValueError('Fixed cohorts have propensity 1, not 1/9')
        if meta['cohort'] in DOSES and dose != DOSES[meta['cohort']]:
            raise ValueError('Wrong fixed episode dose')
        episodes[key] = dict(**meta, arm=r['arm'], uid=r['uid'], attempt=int(r['attempt']),
            task=task, init=init, split=split, env_seed=int(r['client_environment_seed']),
            dose=dose, assignment_probability=prob, random_assignment=random,
            catalog=r['provenance.catalog_sha256'], z=np.array([y, v, m, n, c/5.]),
            forwards=int(r['full_policy_forwards']), extra=int(r['extra_stage3']))
    if not episodes:
        raise ValueError('No selected episodes')
    if arms and set(arms) != {r['arm'] for r in episodes.values()}:
        raise ValueError('Requested arm missing from episode table')

    accepted = set()
    selected_arms = {e['arm'] for e in episodes.values()}
    for r in table_rows(path, 'attempts'):
        if r['arm'] not in selected_arms:
            continue
        if r.get('accepted') in ('', None):
            continue
        if boolean(r['accepted']) and r['status'] in ('done', 'failed') and not r.get('error'):
            accepted.add(identity(r))
    if accepted != set(episodes):
        raise ValueError('Accepted journal/table mismatch')

    dcounts = defaultdict(Counter)
    dsteps = defaultdict(set)
    decision_keys = set()
    dsources = {}
    for r in table_rows(path, 'decisions'):
        key = identity(r)
        if key not in episodes:
            continue
        step = int(r['step'])
        dk = (*key, step)
        if dk in decision_keys:
            raise ValueError('Duplicate decision')
        decision_keys.add(dk)
        dsteps[key].add(step)
        source, vision = r['source'], boolean(r['vision'])
        if source not in ('cache', 'policy', 'policy_tail', 'cache_blind'):
            raise ValueError('Unknown source: ' + source)
        if source == 'policy' and not vision:
            raise ValueError('Policy MISS without anchor')
        actual = int(r['actual_controls'])
        if not 1 <= actual <= 5:
            raise ValueError('Invalid active controls')
        for stage in (1, 2):
            if int(r[f'stage_invocations.stage{stage}']) != 1:
                raise ValueError('Missing shadow stage dispatch')
        stage3 = int(r['stage_invocations.stage3'])
        if stage3 not in (1, 4):
            raise ValueError('Unexpected resample design')
        dcounts[key].update(N=1, V=int(vision), M=int(source == 'policy'), C=actual, F=1, X=stage3-1)
        dsources[dk] = (source, vision)

    acounts = defaultdict(Counter)
    support = defaultdict(Counter)
    anchor_keys = set()
    design_hash = hashlib.sha256()
    for r in table_rows(path, 'anchors'):
        key = identity(r)
        if key not in episodes:
            continue
        e = episodes[key]
        dk = (*key, int(r['step']))
        if dk in anchor_keys:
            raise ValueError('Duplicate anchor')
        anchor_keys.add(dk)
        call = boolean(r['assignment.executed_policy'])
        p = float(r['assignment.actual_propensity'])
        nominal = float(r['assignment.propensity'])
        if not 0 <= p <= 1 or not 0 <= nominal <= 1:
            raise ValueError('Invalid logged propensity')
        if (p == 0 and call) or (p == 1 and not call):
            raise ValueError('Deterministic assignment violated')
        if not np.isclose(float(r['assignment.treatment_probability']), p if call else 1-p):
            raise ValueError('Treatment probability mismatch')
        if float(r['assignment.cohort_probability']) != 1:
            raise ValueError('Wrong complete-block propensity')
        if r['assignment.cohort'] != e['cohort']:
            raise ValueError('Cohort mismatch')
        if dsources.get(dk) != ('policy' if call else 'cache', True):
            raise ValueError('Assignment/source mismatch')
        if e['cohort'] in DOSES or e['cohort'] == 'dose_mix':
            if not np.isclose(p, e['dose']) or nominal != e['dose'] or r['assignment.override'] != 'coin':
                raise ValueError('Unexpected fixed-dose override: cannot interpret as planned controller')
            if int(r['assignment.commit_controls']) != 10:
                raise ValueError('Fixed-dose commitment changed')
        acounts[key].update(V=1, M=int(call))
        group = e['cell'], e['cohort'], e['task'], r['assignment.override'], p
        support[group].update(anchors=1, calls=int(call), supported=int(0 < p < 1))
        small = {k: r[k] for k in r if k.startswith('assignment.') or k in ('arm', 'uid', 'attempt', 'step')}
        design_hash.update(json.dumps(small, sort_keys=True).encode())
    for key, e in episodes.items():
        _, v, m, n, c5 = e['z']
        d = dcounts[key]
        if [d[k] for k in ['N', 'V', 'M', 'C', 'F', 'X']] != [n, v, m, c5*5, e['forwards'], e['extra']]:
            raise ValueError('Episode/decision totals disagree: ' + str(key))
        if [acounts[key][k] for k in ['V', 'M']] != [v, m]:
            raise ValueError('Episode/anchor totals disagree')
        if dsteps[key] != set(range(int(n))):
            raise ValueError('Missing decision index')
    expected_anchor_keys = {k for k, (_, vision) in dsources.items() if vision}
    if anchor_keys != expected_anchor_keys:
        raise ValueError('Anchor/vision key mismatch')
    return list(episodes.values()), dict(source_audit=audit, selected_episodes=len(episodes),
        selected_arms=len({e['arm'] for e in episodes.values()}), decisions=len(decision_keys),
        anchors=len(anchor_keys), accounting_verified=True, assignment_projection_sha256=design_hash.hexdigest(),
        table_hashes={str(p): dict(episodes=sha(p/'episodes.csv'), audit=sha(p/'audit.json')) for p in table_dirs(path)},
        physical_controls_reused_from_reader=True,
        snapshot_restore_certified=False), [dict(cell=g[0], cohort=g[1], task=g[2], override=g[3],
            actual_propensity=g[4], **v) for g, v in sorted(support.items())]


def quality_rows(path=None):
    if path:
        rows = list(csv.DictReader(Path(path).open()))
        result = {(r['cell'], int(r['task'])): dict(risk=float(r['risk']),
            h=float(r['mean_decisions']), a=float(r['mean_anchors']), provenance=r['provenance']) for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate Q1 quality rows')
    else:
        path = HERE / 'library_quality.json'
        result = {}
        for r in json.loads(path.read_text()):
            if r['lib'] not in ('50', '500'):
                continue
            cell = r['cell'].replace('_spatial', '_sp') + '_' + r['lib']
            result[cell, int(r['task'])] = dict(risk=r['state_loeo_distance'], h=r['mean_decisions'],
                a=r['mean_anchors_b2'], provenance=r['library'] + '; library_proxy.py')
    for r in result.values():
        if not all(np.isfinite(r[k]) for k in ('risk', 'h', 'a')) or r['risk'] < 0 or not 0 < r['a'] <= r['h']:
            raise ValueError('Invalid library quality or lengths')
        if not r['provenance']:
            raise ValueError('Quality provenance required')
    return result, dict(path=str(Path(path).resolve()), sha256=sha(Path(path)))


def allocation(quality, cell, rho, uniform=False, traffic=None):
    tasks = sorted(t for c, t in quality if c == cell)
    if not tasks:
        return None
    q = np.array([quality[cell, t]['risk'] for t in tasks])
    h = np.array([quality[cell, t]['h'] for t in tasks])
    a = np.array([quality[cell, t]['a'] for t in tasks])
    pi = np.array([traffic.get((cell, t), 0.) for t in tasks]) if traffic else np.ones(len(tasks))
    if np.any(pi <= 0):
        raise ValueError('Traffic weights must be positive and cover all library tasks')
    pi /= pi.sum()
    q = np.ones_like(q) if uniform or not q.max() else np.maximum(q, np.finfo(float).eps*q.max())
    w = q / np.dot(pi, q)
    c1 = C1[cell.split('_')[0]]
    cost = lambda p: np.dot(pi, a*(c1+(1-c1)*p))/np.dot(pi, h)
    floor, ceiling = cost(np.zeros_like(w)), cost(np.ones_like(w))
    target = np.clip(rho, floor, ceiling)
    lo, hi = 0., 1/w.min()
    for _ in range(80):
        mid = (lo+hi)/2
        if cost(np.minimum(1., mid*w)) < target:
            lo = mid
        else:
            hi = mid
    p = np.minimum(1., (lo+hi)/2*w)
    if rho <= floor:
        p[:] = 0
    if rho >= ceiling:
        p[:] = 1
    weights = {}
    knots = np.array(list(DOSES.values()))
    for t, v in zip(tasks, p):
        if v >= 1:
            mix = np.array([0, 0, 0, 0, 1.])
        else:
            j = max(0, np.searchsorted(knots, v, side='right')-1)
            u = (v-knots[j])/(knots[j+1]-knots[j])
            mix = np.zeros(5)
            mix[j:j+2] = [1-u, u]
        weights[t] = dict(zip(DOSES, mix))
    return dict(rho=rho, uniform=uniform, predicted_IR=cost(p), floor=floor, ceiling=ceiling,
        clamped=bool(rho < floor or rho > ceiling), weights=weights,
        tasks=[dict(task=t, risk=quality[cell, t]['risk'], weight=w[i], p=p[i],
            predicted_calls=a[i]*p[i], library_decisions=h[i], library_anchors=a[i], traffic=pi[i]) for i, t in enumerate(tasks)])


def metric_values(z, c1):
    y, v, m, n, c = np.moveaxis(z, -1, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack([y, (c1*v+(1-c1)*m)/c, m/n, m/v, m/c, m, v, n, c,
                         (c1*v+(1-c1)*m)/n], axis=-1)


def metric_influence(z, point_z, c1):
    """Ratio delta-method cluster scores; constants vanish within task."""
    ans = np.empty(z.shape[:-1] + (len(METRICS),))
    point = metric_values(point_z, c1)
    ans[..., 0] = z[..., 0]
    num = c1*z[..., 1] + (1-c1)*z[..., 2]
    ans[..., 1] = (num-point[..., 1]*z[..., 4])/point_z[..., 4]
    for k, numcol, dencol in [(2, 2, 3), (3, 2, 1), (4, 2, 4)]:
        ans[..., k] = (z[..., numcol]-point[..., k]*z[..., dencol])/point_z[..., dencol]
    ans[..., 5:9] = z[..., [2, 1, 3, 4]]
    ans[..., 9] = (num-point[..., 9]*z[..., 3])/point_z[..., 3]
    return ans


def variance(scores, strata):
    tasks = sorted(set(strata))
    groups = [np.flatnonzero(np.array(strata) == t) for t in tasks]
    if any(len(g) < 2 for g in groups):
        return np.full(scores.shape[1:], np.nan), 0
    var = sum(np.var(scores[g], axis=0, ddof=1)/len(g) for g in groups)/len(groups)**2
    return var, sum(len(g)-1 for g in groups)


class Engine:
    def __init__(self, maps, c1, draws, context):
        self.names = sorted(maps)
        self.units = sorted(set.intersection(*(set(maps[n]) for n in self.names)))
        self.excluded = {n: len(maps[n])-len(self.units) for n in self.names}
        if not self.units:
            raise ValueError('No common matched episodes')
        self.raw = np.array([[maps[n][u] for n in self.names] for u in self.units])
        self.clusters = sorted({u[:2] for u in self.units})
        self.z = np.array([self.raw[[i for i, u in enumerate(self.units) if u[:2] == k]].mean(0) for k in self.clusters])
        self.strata = [k[0] for k in self.clusters]
        tasks = sorted(set(self.strata))
        groups = [np.flatnonzero(np.array(self.strata) == t) for t in tasks]
        wt = np.array([1/(len(tasks)*self.strata.count(t)) for t in self.strata])
        self.point_z = np.einsum('k,kpj->pj', wt, self.z)
        self.point = metric_values(self.point_z, c1)
        self.influence = metric_influence(self.z, self.point_z, c1)
        self.var, self.df = variance(self.influence, self.strata)
        self.valid_ci = all(len(g) >= 2 for g in groups)
        self.reason = '' if self.valid_ci else 'fewer than two distinct inits in a task; variance unavailable'
        self.boot = None
        if self.valid_ci and draws:
            h = int.from_bytes(hashlib.sha256(context.encode()).digest()[:4], 'little')
            rng = np.random.Generator(np.random.PCG64(SEED+h))
            bw = np.zeros((draws, len(self.clusters)))
            for g in groups:
                bw[:, g] = rng.multinomial(len(g), np.full(len(g), 1/len(g)), size=draws)/(len(tasks)*len(g))
            self.boot = metric_values(np.einsum('bk,kpj->bpj', bw, self.z), c1)

    def estimate(self, name):
        i = self.names.index(name)
        out = dict(controller=name, episodes=len(self.units), clusters=len(self.clusters), tasks=len(set(self.strata)),
                   inference_reason=self.reason, cluster_df=self.df)
        for k, metric in enumerate(METRICS):
            out[metric] = self.point[i, k]
            vals = self.boot[:, i, k] if self.boot is not None else None
            finite = float(np.isfinite(vals).mean()) if vals is not None else 0.
            lo, hi = np.quantile(vals, [.025, .975]) if vals is not None and finite == 1 else [np.nan]*2
            out[metric+'_lo'], out[metric+'_hi'] = lo, hi
            out[metric+'_bootstrap_finite_fraction'] = finite
        return out

    def contrast(self, a, b):
        i, j = self.names.index(a), self.names.index(b)
        delta = self.point[i]-self.point[j]
        scores = self.influence[:, i]-self.influence[:, j]
        var, df = variance(scores, self.strata)
        se = np.sqrt(var)
        q = stats.t.ppf(1-ALPHA, df) if df else np.nan
        out = dict(candidate=a, reference=b, episodes=len(self.units), clusters=len(self.clusters),
                   cluster_df=df, inference_reason=self.reason)
        for k, metric in enumerate(METRICS):
            boot = self.boot[:, i, k]-self.boot[:, j, k] if self.boot is not None else None
            lo, hi = np.quantile(boot, [.025, .975]) if boot is not None and np.isfinite(boot).all() else [np.nan]*2
            # No equality claim from a finite sample with no observed variation.
            lower, upper = (delta[k]-q*se[k], delta[k]+q*se[k]) if se[k] > 0 else (np.nan, np.nan)
            out.update({metric+'_delta': delta[k], metric+'_lo': lo, metric+'_hi': hi,
                        metric+'_se': se[k], metric+'_simul_lower': lower, metric+'_simul_upper': upper})
        # Different seed blocks of the same init remain together for ICC too.
        diff = self.raw[:, i, 0]-self.raw[:, j, 0]
        groups = [diff[[k for k, u in enumerate(self.units) if u[:2] == c]] for c in self.clusters]
        from exp.offline_search.rounds.r06.p3_profiling.pilot_stats import icc
        out['paired_SR_ICC'] = icc(groups, self.strata)
        return out

    def slope(self, low, high):
        d = self.contrast(high, low)
        i, j = self.names.index(high), self.names.index(low)
        out = dict(low=low, high=high, delta_SR=d['SR_delta'], delta_m=d['m_delta'], delta_IR=d['IR_delta'])
        for cost in ['m', 'IR']:
            k = METRICS.index(cost)
            den = d[cost+'_delta']
            supported = bool(self.valid_ci and d[cost+'_lo'] > 0)
            out[cost+'_slope_resolved'] = supported
            out['SR_per_'+cost] = d['SR_delta']/den if supported else None
            if supported:
                db = self.boot[:, i]-self.boot[:, j]
                with np.errstate(divide='ignore', invalid='ignore'):
                    vals = db[:, 0]/db[:, k]
                if np.isfinite(vals).all() and np.all(db[:, k] > 0):
                    out['SR_per_'+cost+'_lo'], out['SR_per_'+cost+'_hi'] = np.quantile(vals, [.025, .975])
                else:
                    out[cost+'_slope_resolved'] = False
                    out['SR_per_'+cost] = None
                    out[cost+'_reason'] = 'nonpositive/undefined bootstrap denominator; no filtered ratio interval'
        return out


def expected_units(stage):
    ranges = {'pilot': [range(2)]*3, 'full': [range(25), range(15), range(10)],
              'continuation': [range(2, 25), range(2, 15), range(2, 10)]}[stage]
    return {(t, i, b) for b, indices in enumerate(ranges) for t in range(10) for i in indices}


def analyze(episodes, quality, draws, stage, traffic=None):
    estimates, contrasts, slopes, gaps, allocrows, qualitygaps, doseout, decisions, completeness = ([] for _ in range(9))
    diminishing, mix_contrasts = [], []
    for cell in sorted({e['cell'] for e in episodes}):
        records = [e for e in episodes if e['cell'] == cell]
        c1 = C1[records[0]['model']]
        by = defaultdict(dict)
        seeds = {}
        for r in records:
            u = r['task'], r['init'], r['block']
            if u in by[r['cohort']]:
                raise ValueError('Duplicate cohort/task/init/block')
            by[r['cohort']][u] = r['z']
            if u in seeds and seeds[u] != r['env_seed']:
                raise ValueError('Environment seed differs in paired block')
            seeds[u] = r['env_seed']
        exp_units = expected_units(stage)
        missing = {c: len(exp_units-set(by[c])) for c in COHORTS}
        excess = {c: len(set(by[c])-exp_units) for c in COHORTS}
        complete = all(v == 0 for v in missing.values()) and all(v == 0 for v in excess.values())
        core_complete = all(not missing[c] and not excess[c] for c in CORE)
        completeness.append(dict(cell=cell, stage=stage, complete=complete, core_complete=core_complete,
                                 missing_slots=missing, extra_slots=excess))
        maps = {c: d for c, d in by.items() if d}
        allocation_metadata = {}
        if all(by[c] for c in DOSES):
            common = set.intersection(*(set(by[c]) for c in DOSES))
            for rho in RHOS:
                for uniform in (False, True):
                    rule = allocation(quality, cell, rho, uniform, traffic)
                    if rule is None or any(u[0] not in rule['weights'] for u in common):
                        continue
                    name = ('uniform' if uniform else 'risk') + f'_rho{rho:.2f}'
                    maps[name] = {u: sum(rule['weights'][u[0]][c]*by[c][u] for c in DOSES) for u in common}
                    allocation_metadata[name] = rule
                    for r in rule['tasks']:
                        allocrows.append(dict(cell=cell, controller=name, **r, rho=rho,
                            library_predicted_IR=rule['predicted_IR'], clamped=rule['clamped'],
                            dose_weights=rule['weights'][r['task']]))
        # Do not let missing experimental packages remove support of fixed-dose comparisons.
        primary_maps = {k: v for k, v in maps.items() if k in CORE or k.startswith(('risk_', 'uniform_'))}
        for split in ['all', 'calibration', 'validation']:
            subset = lambda u: split == 'all' or ('calibration' if u[1] % 5 == 0 else 'validation') == split
            for task in [None] + sorted({r['task'] for r in records}):
                selected = {k: {u: z for u, z in v.items() if subset(u) and (task is None or u[0] == task)}
                            for k, v in primary_maps.items()}
                selected = {k: v for k, v in selected.items() if v}
                if not selected or not set.intersection(*(set(v) for v in selected.values())):
                    continue
                label = dict(cell=cell, split=split, task='ALL' if task is None else task)
                eng = Engine(selected, c1, draws, f'{cell}/{split}/{task}')
                for name in eng.names:
                    row = dict(**label, **eng.estimate(name), excluded_unmatched=eng.excluded,
                               nominal_dose=DOSES.get(name), scope='fixed-dose or episode-lottery')
                    if name in allocation_metadata:
                        rule = allocation_metadata[name]
                        tr = next((r for r in rule['tasks'] if r['task'] == task), None)
                        pred = rule['predicted_IR'] if task is None else tr['library_anchors']*(c1+(1-c1)*tr['p'])/tr['library_decisions']
                        row.update(library_predicted_IR=pred, IR_prediction_error=row['IR']-pred,
                                   library_prediction_scope='declared library traffic' if task is None else 'task library lengths')
                    estimates.append(row)
                pairset = {(n, ref) for n in eng.names for ref in ['P10', 'B'] if ref in eng.names and n != ref}
                pairset |= {(f'risk_rho{r:.2f}', f'uniform_rho{r:.2f}') for r in RHOS
                            if f'risk_rho{r:.2f}' in eng.names and f'uniform_rho{r:.2f}' in eng.names}
                pairrows = {}
                for a, b in sorted(pairset):
                    row = dict(**label, **eng.contrast(a, b))
                    contrasts.append(row)
                    pairrows[a, b] = row
                ds = list(DOSES)
                for low, high in zip(ds[:-1], ds[1:]):
                    if low in eng.names and high in eng.names:
                        slopes.append(dict(**label, **eng.slope(low, high)))
                for a, b, c in zip(ds[:-2], ds[1:-1], ds[2:]):
                    if not all(n in eng.names for n in (a, b, c)):
                        continue
                    s1, s2 = eng.slope(a, b), eng.slope(b, c)
                    for cost in ['m', 'IR']:
                        resolved = s1[cost+'_slope_resolved'] and s2[cost+'_slope_resolved']
                        row = dict(**label, lower_doses=f'{a}->{b}', upper_doses=f'{b}->{c}', cost=cost,
                                   conclusion='unresolved')
                        if resolved:
                            ia, ib, ic = (eng.names.index(n) for n in (a, b, c))
                            k = METRICS.index(cost)
                            d1, d2 = eng.boot[:, ib]-eng.boot[:, ia], eng.boot[:, ic]-eng.boot[:, ib]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                change = d2[:, 0]/d2[:, k]-d1[:, 0]/d1[:, k]
                            if np.isfinite(change).all():
                                lo, hi = np.quantile(change, [.025, .975])
                                row.update(slope_change=s2['SR_per_'+cost]-s1['SR_per_'+cost], lo=lo, hi=hi,
                                    conclusion='diminishing' if hi < 0 else 'increasing' if lo > 0 else 'unresolved')
                        diminishing.append(row)
                if 'A' in eng.names and 'P10' in eng.names:
                    gap = eng.contrast('P10', 'A')
                    ia, ip = eng.names.index('A'), eng.names.index('P10')
                    for name in eng.names:
                        k = eng.names.index(name)
                        resolved = bool(eng.valid_ci and gap['SR_lo'] > 0)
                        if resolved:
                            den = eng.boot[:, ip, 0]-eng.boot[:, ia, 0]
                            resolved = bool(np.all(den > 0))
                        fraction = (eng.point[k, 0]-eng.point[ia, 0])/gap['SR_delta'] if resolved else None
                        lo = hi = None
                        if resolved:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                vals = (eng.boot[:, k, 0]-eng.boot[:, ia, 0])/den
                            lo, hi = np.quantile(vals, [.025, .975])
                        gaps.append(dict(**label, controller=name, A_to_P10_gap=gap['SR_delta'],
                            gap_lo=gap['SR_lo'], gap_hi=gap['SR_hi'], gap_resolved=resolved,
                            fraction_closed=fraction, fraction_lo=lo, fraction_hi=hi))
                    if task is not None and (cell, task) in quality:
                        qualitygaps.append(dict(**label, risk=quality[cell, task]['risk'],
                            gap=gap['SR_delta'], gap_lo=gap['SR_lo'], gap_hi=gap['SR_hi']))
                if split == 'validation' and task is None:
                    eligible = []
                    for name in eng.names:
                        if name == 'P10':
                            continue
                        pp = pairrows.get((name, 'P10'), {})
                        pb = pairrows.get((name, 'B'), {})
                        ok = core_complete and eng.valid_ci and pp.get('SR_simul_lower', np.nan) > -EPSILON
                        if name != 'B':
                            ok = ok and pb.get('SR_simul_lower', np.nan) > -EPSILON and pb.get('IR_simul_upper', np.nan) < 0
                        if name.startswith('risk'):
                            pair = pairrows.get((name, name.replace('risk', 'uniform')), {})
                            ok = ok and pair.get('SR_delta', np.nan) >= 0 and pair.get('IR_delta', np.nan) <= 0
                        if ok:
                            eligible.append((eng.point[eng.names.index(name), 1], not name.startswith('uniform'), name))
                    decisions.append(dict(cell=cell, recommendation=min(eligible)[2] if eligible else 'P10_reference_inconclusive',
                        eligible=[x[2] for x in sorted(eligible)], core_complete=core_complete,
                        uncertainty_available=eng.valid_ci, reason=eng.reason or ('simultaneous SR/cost gates applied; see contrasts'),
                        alpha_per_test=ALPHA, epsilon=EPSILON))
            for name in ['factorial', 'window']:
                selected = {u: z for u, z in by[name].items() if subset(u)}
                if selected:
                    eng = Engine({name: selected}, c1, draws, f'{cell}/{split}/{name}')
                    estimates.append(dict(cell=cell, split=split, task='ALL', **eng.estimate(name),
                        nominal_dose=None, scope='separate intervention package; not a dose point'))
            mix = [r for r in records if r['cohort'] == 'dose_mix' and subset((r['task'], r['init'], r['block']))]
            for task in [None] + sorted({r['task'] for r in mix}):
                rr = [r for r in mix if task is None or r['task'] == task]
                if not rr:
                    continue
                for dose in DOSES.values():
                    scores = {(r['task'], r['init'], r['block']): r['z']*(r['dose'] == dose)/r['assignment_probability'] for r in rr}
                    assigned = [r for r in rr if r['dose'] == dose]
                    if not assigned:
                        doseout.append(dict(cell=cell, split=split, task='ALL' if task is None else task,
                            dose=dose, supported=False, assigned_episodes=0, reason='zero assigned support; SR unavailable'))
                        continue
                    eng = Engine({'HT': scores}, c1, draws, f'{cell}/{split}/{task}/mix')
                    vals = np.array([1/r['assignment_probability'] for r in assigned])
                    row = eng.estimate('HT')
                    # Task/init/block standardized ratio of HT Y to HT treatment mass.
                    masses = {(r['task'], r['init'], r['block']): np.array([float(r['dose'] == dose)/r['assignment_probability'], 1, 0, 1, 1]) for r in rr}
                    mass = Engine({'mass': masses}, c1, 0, 'mass').point[0, 0]
                    row.update(cell=cell, split=split, task='ALL' if task is None else task, dose=dose,
                        supported=True, assigned_episodes=len(assigned), assigned_clusters=len({(r['task'], r['init']) for r in assigned}),
                        kish_weight_ESS=vals.sum()**2/(vals@vals), Hajek_SR=row['SR']/mass,
                        assigned_tasks=len({r['task'] for r in assigned}))
                    doseout.append(row)
                    fixed_name = next(k for k, v in DOSES.items() if v == dose)
                    fixed = {u: z for u, z in by[fixed_name].items() if u in scores}
                    if fixed:
                        comparison = Engine({'HT': scores, 'fixed': fixed}, c1, draws, f'{cell}/{split}/{task}/mix_fixed')
                        mix_contrasts.append(dict(cell=cell, split=split, task='ALL' if task is None else task,
                            dose=dose, **comparison.contrast('HT', 'fixed'), excluded_unmatched=comparison.excluded))
    associations = []
    for cell in sorted({r['cell'] for r in qualitygaps}):
        for split in ['all', 'calibration', 'validation']:
            rows = [r for r in qualitygaps if r['cell'] == cell and r['split'] == split]
            varying = len(rows) > 2 and np.ptp([r['risk'] for r in rows]) > 0 and np.ptp([r['gap'] for r in rows]) > 0
            corr = stats.spearmanr([r['risk'] for r in rows], [r['gap'] for r in rows]).statistic if varying else None
            associations.append(dict(cell=cell, split=split, tasks=len(rows), spearman=corr,
                                     scope='exploratory; shared bank; no outcome-tuned proxy choice'))
    return dict(estimates=estimates, contrasts=contrasts, marginal_slopes=slopes, gap_closure=gaps,
        allocations=allocrows, quality_gaps=qualitygaps, quality_associations=associations,
        dose_mix=doseout, dose_mix_vs_fixed=mix_contrasts, diminishing_returns=diminishing,
        recommendations=decisions, completeness=completeness)


def power_table():
    rows = []
    for stage, repeats in [('pilot', [3]*20), ('full', [3]*100+[2]*50+[1]*100)]:
        for icc in [0., .3, 1.]:
            r = np.array(repeats)
            episode_eff = r.sum()/(1+icc*np.sum(r*(r-1))/r.sum())
            init_eff = len(r)**2/np.sum((1+(r-1)*icc)/r)
            for estimator, eff in [('campaign_episode_weighted', episode_eff), ('Q2_equal_init', init_eff)]:
                rows.append(dict(stage=stage, estimator=estimator, ICC_assumption=icc, n_eff=eff,
                    halfwidth95=1.96*math.sqrt(.2/eff), MDE80=2.801621*math.sqrt(.2/eff),
                    scope='all data planning only; validation split and multiplicity reduce precision'))
    counts = dict(variance_assumption=.2, effective_for_halfwidth_05=math.ceil(1.96**2*.2/.05**2),
        effective_for_halfwidth_02=math.ceil(1.96**2*.2/.02**2),
        effective_for_MDE_05=math.ceil((stats.norm.ppf(.975)+stats.norm.ppf(.8))**2*.2/.05**2),
        effective_for_NI_02_single=math.ceil((stats.norm.ppf(.95)+stats.norm.ppf(.8))**2*.2/.02**2),
        effective_for_NI_02_family296=math.ceil((stats.norm.ppf(1-ALPHA)+stats.norm.ppf(.8))**2*.2/.02**2))
    return dict(sensitivity=rows, requirements=counts)


def plots(result, out, smoke):
    os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for cell in sorted({r['cell'] for r in result['estimates']}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        rows = [r for r in result['estimates'] if r['cell'] == cell and r['split'] == 'all' and r['task'] == 'ALL']
        fixed = sorted([r for r in rows if r['controller'] in DOSES], key=lambda r: DOSES[r['controller']])
        for ax, x in zip(axes, ['anchor_call_rate', 'm', 'IR']):
            ax.plot([r[x] for r in fixed], [r['SR'] for r in fixed], '-o', label='fixed dose')
            for r in rows:
                if r['controller'].startswith(('risk', 'uniform')):
                    continue
                ax.scatter(r[x], r['SR'], marker='o' if r['controller'] in DOSES else 'x')
                ax.annotate(r['controller'], (r[x], r['SR']), fontsize=7, xytext=(3, 3), textcoords='offset points')
                if np.isfinite(r['SR_lo']) and np.isfinite(r[x+'_lo']):
                    ax.plot([r[x+'_lo'], r[x+'_hi']], [r['SR'], r['SR']], color='gray', alpha=.4)
                    ax.plot([r[x], r[x]], [r['SR_lo'], r['SR_hi']], color='gray', alpha=.4)
            ax.set(xlabel=x, ylabel='Success rate', ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
        fig.suptitle(cell + (' | SMOKE: PLUMBING ONLY' if smoke else ' | descriptive all-init data'))
        fig.tight_layout()
        target = out / f'curves_{cell}.png'
        fig.savefig(target, dpi=150)
        plt.close(fig)
        paths.append(target.name)
    return paths


def reader_input(args, out):
    if args.tables:
        return args.tables
    if not args.run_root or not (args.arms or args.arms_file or args.cell):
        raise ValueError('Use --tables, or --run-root plus --arms/--arms-file/--cell')
    arms = args.arms or []
    if args.arms_file:
        arms += [s.strip() for s in args.arms_file.read_text().splitlines() if s.strip() and not s.startswith('#')]
    if not arms:
        arms = [f'r6p3v2_{args.cell}_{c}_r{b}' for c in COHORTS for b in range(3)]
    if len(set(arms)) != len(arms):
        raise ValueError('Duplicate requested arms')
    for arm in arms:
        arm_info(arm)
        saved = args.run_root / 'runs' / arm / 'manifest.json'
        if not saved.exists():
            raise ValueError('Missing finished-arm saved manifest: ' + arm)
        from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
        manifest_sha = load_manifest(saved)['sha256']
        marker = args.run_root / 'state' / f'{arm}.manifest_{manifest_sha}.DONE'
        # This campaign is manifest scoped; an old phase's generic DONE is insufficient.
        if not marker.is_file():
            raise ValueError('Missing current-manifest DONE: ' + str(marker))
    target = out / 'reader_tables'
    target.mkdir()
    commands = []
    for arm in arms:
        # read_v2 holds expanded controls in RAM: process ONE finished arm at a time.
        # Its client root is restricted to that same arm, never active neighbors.
        cmd = ['taskset', '-c', '18-21,62-65', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
               'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
               str(ROOT / '.venv/bin/python'), '-m', 'exp.offline_search.rounds.r06.p3_profiling.read_v2',
               '--run-root', str(args.run_root), '--arms', arm, '--client-root', str(args.run_root/'runs'/arm),
               '--require-stage-counts', '--require-snapshots', '--out', str(target/arm)]
        commands.append(cmd)
        dump(out/'reader_command.json', commands)
        with (out/(arm+'.reader.log')).open('w') as f:
            subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True, stdout=f, stderr=subprocess.STDOUT)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tables', type=Path)
    ap.add_argument('--run-root', type=Path)
    ap.add_argument('--arms', nargs='+')
    ap.add_argument('--arms-file', type=Path)
    ap.add_argument('--cell', help='e.g. pi05_l10_50; expands to all 27 campaign arm IDs')
    ap.add_argument('--quality', type=Path, help='Frozen Q1 risk CSV; otherwise existing library-only Q2 proxy')
    ap.add_argument('--traffic', type=Path, help='CSV cell,task,weight; budget allocation only, SR still equal task')
    ap.add_argument('--stage', choices=['pilot', 'full', 'continuation'], default='pilot')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true', help='Mark every artifact as plumbing; no scientific recommendation')
    ap.add_argument('--bootstrap', type=int, default=4000)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    if args.bootstrap != 4000 and not args.smoke:
        ap.error('Production preregistration fixes --bootstrap 4000')
    if args.tables and args.run_root:
        ap.error('--tables and --run-root are alternatives')
    if args.arms and args.arms_file:
        ap.error('Choose --arms or --arms-file, not both')
    if args.traffic:
        ap.error('Nonuniform task traffic needs a separately versioned population estimand; default is equal task traffic')
    out = owned(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tables = reader_input(args, out)
    arms = args.arms
    if args.arms_file and args.tables:
        arms = [s.strip() for s in args.arms_file.read_text().splitlines() if s.strip() and not s.startswith('#')]
    episodes, audit, support = load_tables(tables, arms, args.cell)
    quality, qsource = quality_rows(args.quality)
    traffic = None
    if args.traffic:
        traffic = {(r['cell'], int(r['task'])): float(r['weight']) for r in csv.DictReader(args.traffic.open())}
    result = analyze(episodes, quality, args.bootstrap, args.stage, traffic)
    if args.smoke:
        for r in result['recommendations']:
            r.update(recommendation='SMOKE_ONLY_NO_SCIENTIFIC_RECOMMENDATION')
    for name, rows in result.items():
        write_csv(out/(name+'.csv'), [dict(dataset_scope='SMOKE_ONLY' if args.smoke else 'P3', **r) for r in rows])
    write_csv(out/'assignment_support.csv', support)
    write_csv(out/'episode_costs.csv', [{k: v for k, v in e.items() if k != 'z'} | dict(zip(['Y', 'V', 'M', 'N', 'C5'], e['z'])) for e in episodes])
    dump(out/'estimates.json', result)
    dump(out/'power.json', power_table())
    figure_paths = [] if args.no_plots else plots(result, out, args.smoke)
    audit.update(dataset_scope='SMOKE_ONLY' if args.smoke else 'P3', input_tables=str(Path(tables).resolve()),
        quality=qsource, script_sha256=sha(Path(__file__)), prereg_sha256=sha(HERE/'PREREG.md'),
        bootstrap=args.bootstrap, bootstrap_seed=SEED, alpha_per_test=ALPHA, figures=figure_paths,
        output_row_counts={k: len(v) for k, v in result.items()},
        intervals_are_approximate=True, native_pure_inference_comparison_available=False)
    dump(out/'audit_q2.json', audit)
    (out/'README.md').write_text(('SMOKE PLUMBING ONLY. Do not interpret SR/cost rankings.\n\n' if args.smoke else '')+
        'See PREREG.md for estimands. Curves are descriptive all-init data; recommendations use reserved inits.\n'
        'Empty cells/intervals mean unsupported or unestimable, never zero risk.\n'
        'Owner IR excludes shadow/resample profiling work and is not observed runtime.\n'
        'All stage/accounting checks passed; raw controls and snapshot restores were not re-audited.\n')
    print(json.dumps(dict(scope=audit['dataset_scope'], arms=audit['selected_arms'], episodes=len(episodes),
        decisions=audit['decisions'], anchors=audit['anchors'], output=str(out),
        rows=audit['output_row_counts'], figures=figure_paths)))


if __name__ == '__main__':
    main()
