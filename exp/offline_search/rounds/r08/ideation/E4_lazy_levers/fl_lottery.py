"""R8 callback E4: all-anchor analysis of the follow lottery (FL) arms.

Every fresh look of an FL arm with lottery support draws E in {0,1,2} (p = 1/3 each over its support) before any
extension. Stage at assignment is read from the anchor's own 16 neighbours and the frozen R7 catalog labels
(pre-assignment by construction). We report, on DISCOVERY inits 0-29 only (holdout 30-49 untouched):
  * per-anchor randomized local contrasts (E1-E0, E2-E1): next-look retrieval distance log-ratio, next-look absolute
    state deviation (shadow valve D_abs), next-look kernel class;
  * summed-anchor Horvitz-Thompson scores for final success, by stage class and by the SF gate flag
    (sum over the episode's anchors in the stratum of [1(E=a)/p_a - 1(E=b)/p_b] * (Y - task mean)):
    the first-order effect of switching every anchor in the stratum from E=b to E=a.
Intervals: 2000 bootstrap draws of episodes within task. Outputs /tmp/r8cb_E4_lazy_levers/fl_*.json/csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader

RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E4_lazy_levers')
LIB = {('pi05', 50): 'current', ('pi05', 500): 'bpool_cs', ('groot', 50): 'current', ('groot', 500): 'bpool_all'}
RNG = np.random.default_rng(20261001)


def catalog(model, suite, size):
    c = pd.read_parquet(RUN / 'catalog' / f"{model}_{suite}_{LIB[(model, size)]}" / 'rows.parquet')
    c = c.sort_values('row')
    return c['mode'].to_numpy(), c['event_near'].to_numpy(bool), c['stage_run'].to_numpy(), c['success'].to_numpy(bool)


def klass(rows, w, mode, ev, run):
    rows = np.asarray(rows, int)
    m = mode[rows]
    if (m < 0).any():
        c = 'unknown'
    elif len(set(m.tolist())) > 1:
        c = 'mixed'
    elif ev[rows].any():
        c = 'event'
    else:
        c = 'interior'
    w = np.asarray(w, float)
    ok = m >= 0
    macro = None
    if ok.any():
        r = run[rows[ok]]
        vals, inv = np.unique(r, return_inverse=True)
        mass = np.bincount(inv, weights=w[ok])
        macro = int(vals[mass.argmax()])
    return c, macro


def boot(ep, col, reps=2000):
    """ep: DataFrame with task and value col (per-episode). Mean with within-task bootstrap."""
    v = ep[col].to_numpy(float)
    groups = [np.flatnonzero(ep.task.to_numpy() == t) for t in sorted(ep.task.unique())]
    est = v.mean()
    bs = np.empty(reps)
    for b in range(reps):
        idx = np.concatenate([g[RNG.integers(0, len(g), len(g))] for g in groups])
        bs[b] = v[idx].mean()
    return dict(est=float(est), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(v)))


def analyze(arm):
    a = reader.open_arm(RUN, arm)
    a.cache_enabled = False
    spec = a.manifest.get('arm_spec', {})
    model, suite = spec['model'], spec['suite_short']
    size = int(spec['r8']['library_size'])
    mode, ev, run, _ = catalog(model, 'l10' if suite == 'l10' else 'spatial', size)
    d = a.decisions(columns=['episode_key', 'decision_seq', 'task_id', 'init', 'vision', 'eligible', 'drawn_e', 'support',
                             'propensities', 'rows', 'weights', 'd1', 'diag', 'journal_success', 'src', 'n_applied'])
    d['task'] = d.task_id.astype(int)
    d['init'] = d['init'].astype(int)
    d['vision'] = d.vision.astype(str).eq('True')
    d = d[d.init < 30].copy()          # discovery only
    d['Y'] = d.journal_success.astype(str).eq('True').astype(float)
    d['decision_seq'] = d.decision_seq.astype(int)
    d = d.sort_values(['episode_key', 'decision_seq'])
    recs = []
    for ek, g in d.groupby('episode_key', sort=False):
        looks = g[g.vision]
        lk = looks.to_dict('records')
        for i, r in enumerate(lk):
            diag = r['diag'] if isinstance(r['diag'], dict) else {}
            if str(r['eligible']) != 'True' or r['drawn_e'] is None or (isinstance(r['drawn_e'], float) and np.isnan(r['drawn_e'])):
                continue
            sup = list(r['support']) if r['support'] is not None else []
            if not isinstance(r['rows'], (list, np.ndarray)) or len(r['rows']) != 16:
                continue
            c, macro = klass(r['rows'], r['weights'], mode, ev, run)
            sok = diag.get('stage_ok_by_e') or [None, None, None]
            nxt = lk[i + 1] if i + 1 < len(lk) else None
            rec = dict(episode_key=ek, task=r['task'], init=r['init'], Y=r['Y'], seq=r['decision_seq'], E=int(float(r['drawn_e'])),
                       support=tuple(int(s) for s in sup), p=list(r['propensities']), cls=c, macro=macro,
                       sf1_ok=bool(sok[1]) if len(sok) > 1 and sok[1] is not None else None,
                       d1=float(r['d1']) if r['d1'] is not None else np.nan,
                       dabs=float(diag.get('shadow_absolute', np.nan) or np.nan))
            if nxt is not None:
                nd = nxt['diag'] if isinstance(nxt['diag'], dict) else {}
                rec['next_gap_dec'] = int(nxt['decision_seq']) - int(r['decision_seq'])
                rec['next_d1_logratio'] = float(np.log(float(nxt['d1']) / rec['d1'])) if nxt['d1'] and rec['d1'] > 0 else np.nan
                rec['next_dabs'] = float(nd.get('shadow_absolute', np.nan) or np.nan)
                if isinstance(nxt['rows'], (list, np.ndarray)) and len(nxt['rows']) == 16:
                    rec['next_cls'] = klass(nxt['rows'], nxt['weights'], mode, ev, run)[0]
            recs.append(rec)
    A = pd.DataFrame(recs)
    A.to_csv(OUT / f'fl_anchors_{arm}.csv.gz', index=False)
    res = dict(arm=arm, episodes_discovery=int(d.episode_key.nunique()), anchors=int(len(A)),
               support_full_share=float(np.mean([s == (0, 1, 2) for s in A.support])),
               class_share={k: float(v) for k, v in A.cls.value_counts(normalize=True).items()},
               sf1_ok_share=float(A.sf1_ok.mean()))
    # local randomized contrasts (anchors with full support), by class
    full = A[A.support.map(lambda s: s == (0, 1, 2))]
    loc = {}
    for stratum, sub in [('all', full)] + [(c, full[full.cls == c]) for c in ('interior', 'event', 'mixed', 'unknown')] + \
            [('sf1_ok', full[full.sf1_ok == True]), ('sf1_refused', full[full.sf1_ok == False])]:
        if len(sub) < 60:
            continue
        out = dict(n=int(len(sub)))
        for col in ('next_d1_logratio', 'next_dabs'):
            if col not in sub:
                continue
            m = sub.groupby('E')[col].mean()
            out[col] = {f'E{e}': float(m.get(e, np.nan)) for e in (0, 1, 2)}
        if 'next_cls' in sub:
            hard = sub.next_cls.isin(['mixed', 'unknown'])
            out['next_hard_share'] = {f'E{e}': float(hard[sub.E == e].mean()) for e in (0, 1, 2)}
        # cluster bootstrap for E1-E0 and E2-E0 of next_d1_logratio
        for col in ('next_d1_logratio', 'next_dabs'):
            if col not in sub:
                continue
            per = sub.dropna(subset=[col]).groupby(['task', 'episode_key', 'E'])[col].agg(['sum', 'count']).reset_index()
            for a_, b_ in ((1, 0), (2, 0)):
                eps = per.pivot_table(index=['task', 'episode_key'], columns='E', values=['sum', 'count'], fill_value=0)
                tasks = eps.index.get_level_values(0).to_numpy()
                sa, ca = eps[('sum', a_)].to_numpy(), eps[('count', a_)].to_numpy()
                sb, cb = eps[('sum', b_)].to_numpy(), eps[('count', b_)].to_numpy()
                groups = [np.flatnonzero(tasks == t) for t in np.unique(tasks)]
                est = sa.sum() / ca.sum() - sb.sum() / cb.sum()
                bs = []
                for _ in range(1000):
                    idx = np.concatenate([g[RNG.integers(0, len(g), len(g))] for g in groups])
                    bs.append(sa[idx].sum() / ca[idx].sum() - sb[idx].sum() / cb[idx].sum())
                out[f'{col}_E{a_}mE{b_}'] = dict(est=float(est), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
        loc[stratum] = out
    res['local'] = loc
    # summed-anchor HT success scores, by stratum
    ep = A.groupby('episode_key').agg(task=('task', 'first'), Y=('Y', 'first')).reset_index()
    ep['Yc'] = ep.Y - ep.groupby('task').Y.transform('mean')
    ht = {}
    strata = {'all': A.index == A.index, **{c: (A.cls == c).to_numpy() for c in ('interior', 'event', 'mixed', 'unknown')},
              'sf1_ok': (A.sf1_ok == True).to_numpy(), 'sf1_refused': (A.sf1_ok == False).to_numpy()}
    for name, mask in strata.items():
        sub = A[mask & A.support.map(lambda s: s == (0, 1, 2)).to_numpy()]
        if len(sub) < 60:
            continue
        for a_, b_ in ((1, 0), (2, 1), (2, 0)):
            s = sub.assign(sc=(sub.E == a_) * 3.0 - (sub.E == b_) * 3.0).groupby('episode_key').sc.sum()
            e2 = ep.merge(s.rename('sc'), on='episode_key', how='left').fillna({'sc': 0.0})
            e2['v'] = e2.Yc * e2.sc
            r_ = boot(e2, 'v')
            r_['anchors_per_episode'] = float(len(sub) / len(ep))
            ht[f'{name}:E{a_}-E{b_}'] = r_
    res['ht_success_sum'] = ht
    res['SR_discovery'] = float(ep.Y.mean())
    return res


def main():
    arms = sys.argv[1:]
    out = {}
    p = OUT / 'fl_lottery.json'
    if p.exists():
        out = json.loads(p.read_text())
    for arm in arms:
        out[arm] = analyze(arm)
        p.write_text(json.dumps(out, indent=1))
        print(arm, json.dumps({k: out[arm][k] for k in ('anchors', 'support_full_share', 'sf1_ok_share', 'class_share')}), flush=True)


if __name__ == '__main__':
    main()
