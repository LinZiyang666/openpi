"""Q1 preregistered P3-v2 analysis. CPU/read-only inputs; outputs confined here.

Run from the repository root with the taskset/environment command in PREREG.md.
Smoke estimates are explicitly non-scientific. This module never launches a rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
PRIMARY = ['C', 'D', 'R', 'Qrisk']
SECONDARY = ['dispersion', 'V7', 'negative_neff']
COHORTS = ['A', 'dose125', 'dose25', 'dose50', 'P10', 'B', 'factorial', 'window', 'dose_mix']
EKEY = ['arm', 'uid', 'attempt']
AKEY = EKEY + ['step']
CLUSTER = ['suite', 'task_id', 'init']
SEED = 60601


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    p.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def truth(s):
    return s.astype(str).str.lower().isin(['true', '1'])


def arr(x):
    if isinstance(x, str) and x: return np.asarray(json.loads(x))
    return np.array([])


def mean(s):
    a = np.asarray(s, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def matrix_rho(x, y):
    """Row-wise Spearman, including bootstrap repeated task points."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    xr=rankdata(np.where(ok,x,np.nan),axis=1,nan_policy='omit')
    yr=rankdata(np.where(ok,y,np.nan),axis=1,nan_policy='omit')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        xr=xr-np.nanmean(xr,axis=1)[:,None];yr=yr-np.nanmean(yr,axis=1)[:,None]
    den=np.sqrt(np.nansum(xr*xr,axis=1)*np.nansum(yr*yr,axis=1))
    return np.divide(np.nansum(xr*yr,axis=1),den,out=np.full(len(x),np.nan),where=(den>0)&(ok.sum(1)>=3))


def residual_rank_rho(x,y,z):
    """Descriptive partial Spearman after size and model/suite fixed effects."""
    x,y=np.atleast_2d(x),np.atleast_2d(y)
    if len(z)-np.linalg.matrix_rank(z)<3:return np.full(len(x),np.nan)
    proj=np.eye(len(z))-z@np.linalg.pinv(z)
    rx=rankdata(x,axis=1)@proj;ry=rankdata(y,axis=1)@proj
    den=np.sqrt(np.sum(rx*rx,axis=1)*np.sum(ry*ry,axis=1))
    return np.divide(np.sum(rx*ry,axis=1),den,out=np.full(len(x),np.nan),where=den>1e-12)


def weighted_rho(x,y,w):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    keep=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    x,y,w=x[keep],y[keep],w[keep]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    def wrank(v):
        _,ix=np.unique(v,return_inverse=True);mass=np.bincount(ix,weights=w)
        return (np.cumsum(mass)-mass/2)[ix]
    x,y=wrank(x),wrank(y);x-=np.average(x,weights=w);y-=np.average(y,weights=w)
    return float(np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y)))


def hierarchy_weights(d):
    """Equal cell/task/init/repeat/anchor; each row is one anchor."""
    g = d.groupby(['cell', 'task_id', 'init'], observed=True)
    na = d.groupby(EKEY, observed=True)['step'].transform('size').to_numpy(float)
    nr = g['uid'].transform('nunique').to_numpy(float)
    ni = d.groupby(['cell', 'task_id'], observed=True)['init'].transform('nunique').to_numpy(float)
    nt = d.groupby('cell', observed=True)['task_id'].transform('nunique').to_numpy(float)
    return 1 / (na * nr * ni * nt * d.cell.nunique())


def hierarchical_draws(index, B, fixed_tasks=False):
    """Shared suite/task/init multiplicities preserve all cell/cohort pairing."""
    index = index.drop_duplicates(CLUSTER).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    mult = np.zeros((B, len(index)), dtype=np.int16)
    for _, sg in index.groupby('suite', sort=True):
        tasks = sorted(sg.task_id.unique())
        for b in range(B):
            for t in (tasks if fixed_tasks else rng.choice(tasks, size=len(tasks), replace=True)):
                ids = sg.index[sg.task_id == t].to_numpy()
                # divide by number of inits later; this draw has fixed n within task
                for j in rng.choice(ids, size=len(ids), replace=True): mult[b, j] += 1
    return index, mult


class Inference:
    def __init__(self, episodes, B, fixed_tasks=False):
        self.index, self.mult = hierarchical_draws(episodes, B, fixed_tasks)
        self.lookup = {tuple(r): i for i, r in enumerate(self.index[CLUSTER].itertuples(index=False, name=None))}

    def aggregate(self, d, columns):
        """d: episode or cluster rows; return equal-cell/task/init means and draws."""
        g = d.groupby(['cell'] + CLUSTER, observed=True)[columns].mean().reset_index()
        if not len(g): return np.full(len(columns), np.nan), np.full((len(self.mult), len(columns)), np.nan)
        def divide(a, b): return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        points=[];boot=[]
        for _,cg in g.groupby('cell',sort=True):
            tp=[];tb=[];tm=[]
            for _,tg in cg.groupby(['suite','task_id'],sort=True):
                v=tg[columns].to_numpy(float);good=np.isfinite(v);v=np.nan_to_num(v)
                ix=[self.lookup[tuple(r)] for r in tg[CLUSTER].itertuples(index=False,name=None)]
                wt=self.mult[:,ix]
                tp.append(divide(v.sum(0),good.sum(0)))
                tb.append(divide(wt@v,wt@good))
                tm.append(wt.sum(1)/len(tg))
            pv=np.stack(tp);bv=np.stack(tb,axis=1);tw=np.stack(tm,axis=1)[:,:,None]
            points.append(divide(np.nansum(pv,axis=0),np.isfinite(pv).sum(0)))
            boot.append(divide(np.sum(tw*np.nan_to_num(bv),axis=1),np.sum(tw*np.isfinite(bv),axis=1)))
        p=np.stack(points);b=np.stack(boot,axis=1)
        return divide(np.nansum(p,axis=0),np.isfinite(p).sum(0)),divide(np.nansum(b,axis=1),np.isfinite(b).sum(1))


def interval(v, alpha=.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v): return [np.nan, np.nan]
    return np.quantile(v, [alpha / 2, 1 - alpha / 2]).tolist()


class FitUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('osm_'):
            for mod in ['exp.offline_search.rounds.r04.k1_blind.blind_awm',
                        'exp.offline_search.rounds.r06.p1_groot_commit.judge',
                        'exp.offline_search.rounds.r05.q1_commit.judge',
                        'exp.offline_search.rounds.r02.g1_awm.awm',
                        'exp.offline_search.rounds.r05.q4_growth.demo_method',
                        'exp.offline_search.rounds.r05.q4_growth.method']:
                m = importlib.import_module(mod)
                if hasattr(m, name): return getattr(m, name)
        return super().find_class(module, name)


def retrieval_fingerprint(path):
    with Path(path).open('rb') as f: m = FitUnpickler(f).load()['method']
    from exp.offline_search.rounds.r06.p3_profiling.method import awm_of
    m = awm_of(m)
    h = hashlib.sha256()
    def add(k, v):
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        else: h.update(repr(v).encode())
    for k in ['k', 'kref', 'early', 'features', 'feat0', 'lam_c', 'norm_cap', 'hyst', 'act', 'sig', 'lib_ep', 'lib_step', 'B0T', 'B1T', 'muB0', 'muB1']:
        add(k, getattr(m, k, None))
    for t, T in sorted(m.tasks.items()):
        for k in ['rows', 'Wf', 'shift', 'Z', 'z2', 'W0f', 'c0', 'A0', 'As0', 'n20', 'Z0']:
            add(str(t) + ':' + k, getattr(T, k, None))
    return h.hexdigest()


def catalogs(decisions):
    """Read startup only in servers explicitly referenced by accepted table rows."""
    found, paths = {}, []
    servers = {Path(p).parents[2] for p in decisions.absolute_input_archive}
    for server in sorted(servers):
        for p in sorted(server.glob('decisions_*.jsonl')):
            with p.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get('ev') == 'p3_v2_startup':
                        c = r['catalog']
                        digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
                        if digest != r['catalog_sha256']: raise ValueError('catalog hash mismatch')
                        found[digest] = c
                        paths.append(str(p))
                        # one startup per server connection; a file may contain more connections
                    elif r.get('ev') == 'p3_anchor' and set(decisions.loc[decisions.absolute_input_archive.str.startswith(str(server)), 'catalog_sha256']).issubset(found):
                        break
    if not set(decisions.catalog_sha256).issubset(found): raise ValueError('missing catalog')
    return found, paths


def calibration(cat, fingerprints):
    c = cat
    suite = c['cell'].split('_')[1]
    tag = f"{c['model']}_{suite}_{c['candidate_library']}_refit"
    p = HERE / (tag + '.json')
    meta = json.loads(p.read_text())
    n = np.load(HERE / (tag + '_loeo.npz'), allow_pickle=False)
    lib = Path(c['library_manifest']['path']).parent
    oldhash = {r['path']: r['sha256'] for r in json.loads((HERE / 'input_hashes.json').read_text())}
    manifest_ok = sha(lib / 'manifest.json') == c['library_manifest']['sha256'] == oldhash[str(lib / 'manifest.json')]
    rows_ok = all(np.array_equal(n[key], np.load(lib / (file + '.npy'), mmap_mode='r'))
                  for key, file in [('ep', 'episode'), ('task', 'task_id'), ('step', 'step')])
    def fp(path):
        if path not in fingerprints: fingerprints[path] = retrieval_fingerprint(path)
        return fingerprints[path]
    oldfit, newfit = meta['job']['fit'], c['base_fit']['path']
    fit_bytes_ok = sha(newfit) == c['base_fit']['sha256'] and sha(oldfit) == oldhash[oldfit]
    metric_ok = fp(oldfit) == fp(newfit)
    ok = manifest_ok and rows_ok and fit_bytes_ok and metric_ok
    audit = dict(tag=tag, manifest_match=manifest_ok, row_match=rows_ok, fit_bytes_match=fit_bytes_ok,
                 metric_kernel_fingerprint_match=metric_ok, enabled=ok,
                 old_fingerprint=fp(oldfit), deployed_fingerprint=fp(newfit),
                 residual_sha256=sha(HERE / (tag + '_loeo.npz')), meta_sha256=sha(p))
    return meta, n, audit


def table_read(path, name):
    # Do not materialize privileged physics/controller arrays: Q1 has no use for
    # them. This also keeps a complete eight-cell run's memory footprint modest.
    if name=='anchors':
        prefixes=('assignment.','retrieval.','resampling.','provenance.','distance.')
        base=set(AKEY+['model','suite','lib','task_id','init','Y','client_environment_seed','catalog_sha256',
                      'guards.inputs_outputs.pred_err','commit_truncated'])
        exclude=('retrieval.modalities.','retrieval.metric_code')
        use=lambda k:k in base or (k.startswith(prefixes) and not k.startswith(exclude))
    elif name=='decisions':
        use=lambda k:k in AKEY+['absolute_input_archive','catalog_sha256']
    elif name=='episodes':
        use=lambda k:not k.startswith('client.') and not k.startswith('provenance.chunk_')
    else:use=None
    return pd.read_csv(path / (name + '.csv'), low_memory=False,usecols=use)


def validate_tables(ep, a, d, action, attempts):
    for frame, key in [(ep, EKEY), (a, AKEY), (d, AKEY), (action, AKEY + ['chunk_step'])]:
        if frame.duplicated(key).any(): raise ValueError('duplicate table identity: ' + str(key))
    if ep.uid.duplicated().any(): raise ValueError('multiple accepted attempts')
    ekeys = set(map(tuple, ep[EKEY].to_numpy()))
    if any(tuple(x) not in ekeys for x in a[EKEY].to_numpy()): raise ValueError('unjoined anchor')
    if not truth(ep.controls_verified).all() or not truth(ep.environment_seed_verified).all(): raise ValueError('unverified client data')
    if set(ep.collection_mode) != {'client_verified'}: raise ValueError('server-only tables unsupported')
    if not (ep['provenance.split'] == np.where(ep.init % 5 == 0, 'calibration', 'validation')).all(): raise ValueError('split mismatch')
    if not np.array_equal(ep.client_environment_seed.to_numpy(),603+ep['provenance.run_block'].to_numpy()):raise ValueError('environment block mismatch')
    joined=a.merge(ep[EKEY+['Y','task_id','init','client_environment_seed']],on=EKEY,suffixes=('','_episode'),validate='many_to_one')
    for key in ['Y','task_id','init','client_environment_seed']:
        if not np.array_equal(joined[key],joined[key+'_episode']):raise ValueError('anchor/episode identity mismatch: '+key)
    accepted = attempts[truth(attempts.accepted) & attempts.status.isin(['done', 'failed'])]
    aks = set(map(tuple, accepted[['arm', 'task_uid', 'attempt']].to_numpy()))
    if not ekeys.issubset(aks): raise ValueError('episode not accepted in journal')
    if not np.array_equal(a.groupby(EKEY).size().reindex(pd.MultiIndex.from_frame(ep[EKEY])).values, ep.anchors): raise ValueError('anchor count mismatch')
    for key in ['assignment.actual_propensity', 'resampling.selection_p']:
        if not a[key].between(0, 1).all(): raise ValueError('invalid probability')
    if not np.allclose(a['assignment.cohort_probability'], 1): raise ValueError('not fixed matched design')
    return dict(episodes=len(ep), anchors=len(a), decisions=len(d), action_steps=len(action), attempts=len(attempts))


def build_anchors(a, actions, cats, calibrations):
    action_groups = actions.groupby(AKEY, sort=False)
    rows = []
    for _, r in a.iterrows():
        k = tuple(r[x] for x in AKEY)
        ac = action_groups.get_group(k).sort_values('chunk_step')
        c = cats[r.catalog_sha256]
        meta, libcal, bridge = calibrations[r.catalog_sha256]
        dims = c['interface']['action_valid_indices']
        block = int(c['interface']['R']); horizon = min(2 * block, int(c['interface']['H']))
        sigma = np.asarray(meta['action_sigma_own'])
        cache = ac[['cache.' + str(j) for j in dims]].to_numpy(float)
        policy = ac[['policy.' + str(j) for j in dims]].to_numpy(float)
        if len(cache) < horizon or sigma.shape != (len(dims),): raise ValueError('action interface mismatch')
        err = (cache - policy) / sigma
        E = np.sqrt(np.mean(err[:horizon] ** 2))
        raw = np.sqrt(np.mean((cache[:horizon] - policy[:horizon]) ** 2))
        if not np.isclose(raw, r['distance.commit10_rms'], atol=2e-6): raise ValueError('shadow label parity')
        ix, w = arr(r['retrieval.rows']).astype(int), arr(r['retrieval.weights']).astype(float)
        if not len(ix) or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=2e-5): raise ValueError('invalid kernel')
        w /= w.sum()
        t = int(r.task_id)
        if np.any(libcal['task'][ix] != t): raise ValueError('neighbour task mismatch')
        reference=np.asarray(c['loeo_distance_tables'][f"{t}:{r['retrieval.metric']}"])
        q=(np.searchsorted(reference,r['retrieval.d1'],'left')+np.searchsorted(reference,r['retrieval.d1'],'right'))/(2*len(reference))
        if not np.isclose(q,r['retrieval.d1_loeo_quantile'],atol=1e-10):raise ValueError('LOEO CDF parity')
        pairs = meta['scales'][str(t)]['pair']
        D = r['retrieval.d1'] / pairs
        R = w @ libcal['err10'][ix]
        succ = libcal['succ2'][ix]
        S = w @ np.where(np.isfinite(succ), succ, meta['calibration'][str(t)]['succ2_mean'])
        d = {key: r[key] for key in EKEY + ['step', 'model', 'suite', 'lib', 'task_id', 'init', 'Y', 'client_environment_seed']}
        d.update(cell=f'{r.model}_{r.suite}_{r.lib}', cohort=r['assignment.cohort'],
                 split=r['provenance.split'], block=r['provenance.run_block'], tag=bridge['tag'],
                 C=r['retrieval.d1_loeo_quantile'], D=D if bridge['enabled'] else np.nan,
                 R=R if bridge['enabled'] else np.nan, Qrisk=D+R+S if bridge['enabled'] else np.nan,
                 Q=1/(1+D+R+S) if bridge['enabled'] else np.nan, S=S,
                 successor_observed_mass=w[np.isfinite(succ)].sum(),
                 dispersion=r['retrieval.dispersion_rms'], V7=r['guards.inputs_outputs.pred_err'],
                 negative_neff=-1/(w@w), E=E, E_head=np.sqrt(np.mean(err[:block]**2)),
                 E_full=np.sqrt(np.mean(err**2)), E_raw=raw,
                 progress=float(w @ arr(r['retrieval.progress'])),
                 p=r['assignment.actual_propensity'], call=bool(truth(pd.Series([r['assignment.executed_policy']])).iloc[0]),
                 override=r['assignment.override'], pre_guard=bool(truth(pd.Series([r['assignment.pre_guard_randomization']])).iloc[0]),
                 duration=r['assignment.duration_choice'], hold=r['assignment.hold_choice'],
                 delay=r['assignment.delay_choice'], pd=r['assignment.duration_probability'],
                 ph=r['assignment.hold_probability'], pl=r['assignment.delay_probability'],
                 sample_p=r['resampling.selection_p'], selected=bool(truth(pd.Series([r['resampling.selected']])).iloc[0]),
                 truncated=bool(truth(pd.Series([r['commit_truncated']])).iloc[0]),
                 catalog_sha256=r.catalog_sha256)
        d['bridge_enabled']=bridge['enabled']
        d.update(V=np.nan, L=np.nan, excess=np.nan, mean_policy_error=np.nan)
        if d['selected']:
            extra = arr(r['resampling.extra_chunks'])
            if extra.shape[0] != 3: raise ValueError('K4 requires exactly three extra draws')
            draws = np.concatenate([policy[None, :horizon], extra[:, :horizon, dims]], axis=0) / sigma
            cc = cache[:horizon] / sigma
            V = np.mean(np.var(draws, axis=0, ddof=1))
            L = np.mean((cc[None] - draws) ** 2)
            d.update(V=V, L=L, excess=L-V, mean_policy_error=np.mean((cc-draws.mean(0))**2))
        d['phase_bin'] = int(np.clip(np.floor(d['progress']*5), 0, 4))
        rows.append(d)
    return pd.DataFrame(rows)


def refit(a):
    params = []; out = a.copy()
    for cell, g in a.groupby('cell', sort=True):
        train = g[(g.cohort == 'A') & (g.split == 'calibration')]
        if not len(train):
            for k in PRIMARY + SECONDARY: out.loc[g.index, 'pred_' + k] = np.nan
            out.loc[g.index, 'baseline'] = np.nan
            continue
        weights = hierarchy_weights(train)
        baseline = np.average(train.E, weights=weights)
        out.loc[g.index, 'baseline'] = baseline
        for k in PRIMARY + SECONDARY:
            good = np.isfinite(train[k]) & np.isfinite(train.E)
            if good.sum() < 2:
                out.loc[g.index, 'pred_' + k] = np.nan
                continue
            x = train.loc[good, k].to_numpy(); y = train.loc[good, 'E'].to_numpy()
            wt = np.sqrt(weights[good])
            beta = nnls(np.column_stack([np.ones(len(x)), x])*wt[:, None], y*wt)[0]
            out.loc[g.index, 'pred_' + k] = beta[0] + beta[1]*g[k]
            params.append(dict(cell=cell, candidate=k, intercept=beta[0], slope=beta[1], baseline=baseline,
                               calibration_episodes=sorted(train.uid.unique()), calibration_clusters=len(train[CLUSTER].drop_duplicates())))
    return out, params


def episode_metrics(a):
    rows = []
    for _, g in a.groupby(EKEY, sort=False):
        r = g.iloc[0]
        row = {k: r[k] for k in EKEY + ['cell', 'suite', 'task_id', 'init', 'cohort', 'block', 'split', 'Y', 'tag', 'model', 'lib','bridge_enabled']}
        row.update(n_anchors=len(g), E=g.E.mean(), baseline_mae=np.mean(abs(g.E-g.baseline)))
        for k in PRIMARY + SECONDARY:
            row.update({k: g[k].mean(), k+'_first':g.sort_values('step')[k].iloc[0],
                        'rho_'+k:rho(g[k], g.E), 'mae_'+k:mean(abs(g['pred_'+k]-g.E))})
        for k in ['V', 'L', 'excess', 'mean_policy_error']:
            # HT with every eligible anchor in denominator, zeros for unselected.
            row[k+'_HT'] = (g[k].fillna(0)/g.sample_p).sum()/len(g)
        w = np.where(g.selected, 1/g.sample_p, 0)
        row.update(k4_selected=int(g.selected.sum()), k4_ess=w.sum()**2/(w@w) if w@w else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def state_estimates(em, infer, cohort='A'):
    results = []; comparisons = []; decision = {}
    val = em[(em.split == 'validation') & (em.cohort == cohort)]
    scopes = [('pooled', val)] + list(val.groupby('cell'))
    for scope, g in scopes:
        for k in PRIMARY + SECONDARY:
            cols = ['rho_'+k, 'mae_'+k, 'baseline_mae']
            # Loss contrasts always use the same finite episode set.
            gg = g[np.isfinite(g['mae_'+k]) & np.isfinite(g.baseline_mae)]
            point, draws = infer.aggregate(gg, cols)
            gain = 1-point[1]/point[2] if point[2] > 0 else np.nan
            with np.errstate(invalid='ignore', divide='ignore'): bgain = 1-draws[:, 1]/draws[:, 2]
            alpha = .05/8 if k in PRIMARY else .05
            row = dict(scope=scope, cohort=cohort, candidate=k, episodes=len(gg), clusters=len(gg[CLUSTER].drop_duplicates()),
                       rho=point[0], rho_defined=int(gg['rho_'+k].notna().sum()), rho_fraction=gg['rho_'+k].notna().sum()/len(g) if len(g) else 0,
                       rho_lo=interval(draws[:, 0])[0], rho_hi=interval(draws[:, 0])[1],
                       rho_sim_lo=interval(draws[:, 0], alpha)[0], rho_sim_hi=interval(draws[:, 0], alpha)[1],
                       mae=point[1], baseline_mae=point[2], mae_gain=gain,
                       gain_lo=interval(bgain)[0], gain_hi=interval(bgain)[1],
                       gain_sim_lo=interval(bgain, alpha)[0], gain_sim_hi=interval(bgain, alpha)[1])
            results.append(row)
        for i, ka in enumerate(PRIMARY):
            for kb in PRIMARY[i+1:]:
                gg = g.dropna(subset=['mae_'+ka, 'mae_'+kb])
                p, b = infer.aggregate(gg, ['mae_'+ka, 'mae_'+kb, 'rho_'+ka, 'rho_'+kb])
                comparisons.append(dict(scope=scope, cohort=cohort, simpler=ka, complex=kb,
                    mae_improvement=p[0]-p[1], lo=interval(b[:,0]-b[:,1], .05/6)[0], hi=interval(b[:,0]-b[:,1], .05/6)[1],
                    rho_difference=p[3]-p[2], rho_lo=interval(b[:,3]-b[:,2])[0], rho_hi=interval(b[:,3]-b[:,2])[1]))
    return pd.DataFrame(results), pd.DataFrame(comparisons)


def cohorts_and_contrasts(ep, infer):
    means = []; contrasts = []
    cols = ['Y', 'deployment_IR_per_request', 'deployment_IR_per_actual_5_controls']
    for (cell, cohort), g in ep[ep.split == 'validation'].groupby(['cell', 'cohort']):
        p, b = infer.aggregate(g, cols)
        means.append(dict(cell=cell, cohort=cohort, episodes=len(g), clusters=len(g[CLUSTER].drop_duplicates()),
                          **{k: p[i] for i,k in enumerate(cols)}))
    for cell, cg in ep[ep.split == 'validation'].groupby('cell'):
        keys = ['cell'] + CLUSTER + ['client_environment_seed']
        for ref in ['A', 'P10']:
            rr = cg[cg.cohort == ref]
            for cohort in COHORTS:
                if cohort == ref: continue
                g = cg[cg.cohort == cohort].merge(rr, on=keys, suffixes=('', '_ref'), validate='one_to_one')
                if not len(g): continue
                for k in cols: g['delta_'+k] = g[k]-g[k+'_ref']
                cc = ['delta_'+k for k in cols]
                p,b = infer.aggregate(g, cc)
                for i,k in enumerate(cols):
                    ci=interval(b[:,i]); simultaneous=interval(b[:,i], .05/5)
                    contrasts.append(dict(cell=cell, cohort=cohort, reference=ref, outcome=k,
                        paired_episodes=len(g), clusters=len(g[CLUSTER].drop_duplicates()), difference=p[i],
                        lo=ci[0], hi=ci[1], screening_lo=simultaneous[0], screening_hi=simultaneous[1]))
    return pd.DataFrame(means), pd.DataFrame(contrasts)


def support_and_excursions(a, infer):
    support = []
    for keys, g in a.groupby(['cell', 'cohort', 'split', 'override'], dropna=False):
        eligible = (g.p > 0) & (g.p < 1)
        w = np.where(eligible & g.call, 1/g.p.clip(lower=1e-12), np.where(eligible & ~g.call, 1/(1-g.p).clip(lower=1e-12), 0))
        cw = pd.DataFrame({'task_id':g.task_id, 'init':g.init, 'w':w}).groupby(['task_id','init']).w.sum().to_numpy()
        support.append(dict(zip(['cell','cohort','split','override'],keys), anchors=len(g), supported=int(eligible.sum()),
            p_min=g.p.min(), p_max=g.p.max(), call=int(g.call.sum()), unique_clusters=len(g[CLUSTER].drop_duplicates()),
            cluster_weight_ess=cw.sum()**2/(cw@cw) if cw@cw else 0, pd_min=g.pd.min(), ph_min=g.ph.min(), pl_min=g.pl.min()))
    g = a[(a.cohort=='factorial') & (a.split=='validation') & a.pre_guard & (a.override=='coin') & (a.p>0) & (a.p<1)].copy()
    effects=[]
    if len(g):
        g['stratum']=np.searchsorted([1/3,2/3], g.C, side='right')
        for (cell, stratum), s in g.groupby(['cell','stratum']):
            for duration, hold in [(0,0)]+[(d,h) for d in [5,10] for h in [1,2,3]]:
                active=s.call if duration==0 else s.call & (s.duration==duration) & (s.hold==hold)
                prob=s.p if duration==0 else s.p*s.pd*s.ph
                score=np.where(active,s.Y/prob,0)-np.where(~s.call,s.Y/(1-s.p),0)
                x=s[EKEY+['cell']+CLUSTER].copy();x['ht_sum']=score;x['opportunities']=1
                x=x.groupby(EKEY+['cell']+CLUSTER)[['ht_sum','opportunities']].sum().reset_index()
                # Zero-opportunity episodes remain in the population. Never divide
                # each episode's HT sum by its future treatment-dependent length.
                all_ep=a[(a.cell==cell)&(a.cohort=='factorial')&(a.split=='validation')][EKEY+['cell']+CLUSTER].drop_duplicates()
                x=all_ep.merge(x,how='left',on=EKEY+['cell']+CLUSTER).fillna({'ht_sum':0,'opportunities':0})
                p,b=infer.aggregate(x,['ht_sum','opportunities'])
                bd=np.divide(b[:,0],b[:,1],out=np.full(len(b),np.nan),where=b[:,1]>0);ci=interval(bd)
                effects.append(dict(cell=cell,stratum=stratum,duration=duration,hold=hold,estimate=p[0]/p[1],lo=ci[0],hi=ci[1],
                    opportunities=len(s),episodes=len(x),clusters=len(x[CLUSTER].drop_duplicates()),
                    treated_opportunities=int(active.sum()),cache_opportunities=int((~s.call).sum()),
                    estimand='ratio of macro expected HT sum to expected opportunities; package excursion; descriptive'))
    return pd.DataFrame(support),pd.DataFrame(effects)


def old_bank_ranks(ep, em, infer):
    """Fixed pre-pilot scores; episode bootstrap shared across bank/cohort pairs."""
    old = pd.read_csv(HERE/'task_scores.csv')
    # Normalize historical task column spelling in this report's artifact.
    if 'task_id' not in old and 'task' in old: old=old.rename(columns={'task':'task_id'})
    rows=[]; ranks=[]
    for cell,g in ep[(ep.split=='validation') & ep.cohort.isin(['A','B','P10'])].groupby('cell'):
        if not em[em.cell==cell].bridge_enabled.all():continue
        tag=em[em.cell==cell].tag.iloc[0]
        for task,tg in g.groupby('task_id'):
            pvt=tg.pivot(index=CLUSTER+['client_environment_seed'],columns='cohort',values='Y').reset_index()
            pvt['cell']=cell
            for target in ['failure_A','gap_P10_A','gain_B_A']:
                if 'A' not in pvt: continue
                if target=='failure_A':pvt['target']=1-pvt.A
                elif target=='gap_P10_A' and 'P10' in pvt:pvt['target']=pvt.P10-pvt.A
                elif target=='gain_B_A' and 'B' in pvt:pvt['target']=pvt.B-pvt.A
                else:continue
                p,b=infer.aggregate(pvt.dropna(subset=['target']),['target'])
                for _,o in old[(old.tag==tag)&(old.task_id==task)].iterrows():
                    for score, value in [('D',o.d_pair),('R',o.err10),('Qrisk',1-o.q),('C95_failure',1-o.covered)]:
                        rows.append(dict(cell=cell,suite=tg.suite.iloc[0],tag=tag,task_id=task,source=o.source,candidate=score,score=value,target=target,value=p[0],draw=b[:,0]))
    if not rows:return pd.DataFrame(),pd.DataFrame()
    frame=pd.DataFrame(rows)
    for (source,candidate,target),g in frame.groupby(['source','candidate','target']):
        for level in ['task','global','within_model_suite_size','size_adjusted_global']:
            gg=g.copy()
            if level=='within_model_suite_size':
                gg['stratum']=gg.cell.str.rsplit('_',n=1).str[0]
                # bpool_all/cs suffix makes textual splitting ambiguous; use tag model/suite prefix.
                gg['stratum']=gg.tag.str.extract(r'^(pi05|groot)_(l10|spatial)')[0]+'_'+gg.tag.str.extract(r'^(pi05|groot)_(l10|spatial)')[1]+'_'+gg.task_id.astype(str)
                dd=[]
                for st,sg in gg.groupby('stratum'):
                    if len(sg)!=2:continue
                    sg=sg.assign(small=sg.cell.str.endswith('_current')).sort_values('small',ascending=False)
                    dd.append(dict(score=sg.score.iloc[1]-sg.score.iloc[0],value=sg.value.iloc[1]-sg.value.iloc[0],draw=sg.draw.iloc[1]-sg.draw.iloc[0],
                                   cell=sg.cell.iloc[0],suite=sg.suite.iloc[0],task_id=sg.task_id.iloc[0]))
                gg=pd.DataFrame(dd)
            if not len(gg):continue
            # Task multiplicity is shared across cells, and conditional init
            # resampling supplies each cell/task outcome. Absent tasks have zero
            # multiplicity; do not average their NaNs into a global bank mean.
            count=[]
            for _,r in gg.iterrows():
                ix=infer.index.index[(infer.index.suite==r.suite)&(infer.index.task_id==r.task_id)]
                count.append((infer.mult[:,ix].sum(1)/len(ix)).astype(int))
            count=np.stack(count).T
            x=gg.score.to_numpy();y=gg.value.to_numpy();bs=np.stack(gg.draw).T
            if level in ['global','size_adjusted_global']:
                px=[];py=[];bx=[];by=[]
                for cell in sorted(gg.cell.unique()):
                    ix=np.flatnonzero(gg.cell.to_numpy()==cell);wt=count[:,ix]
                    px.append(mean(x[ix]));py.append(mean(y[ix]))
                    den=wt.sum(1)
                    bx.append(np.divide(wt@x[ix],den,out=np.full(len(bs),np.nan),where=den>0))
                    by.append(np.divide(np.sum(wt*np.nan_to_num(bs[:,ix]),axis=1),den,out=np.full(len(bs),np.nan),where=den>0))
                x,y=np.array(px),np.array(py);xb=np.stack(bx).T;yb=np.stack(by).T
            else:
                N=int(count.sum(1).max());xb=np.full((len(bs),N),np.nan);yb=xb.copy()
                for b in range(len(bs)):
                    ix=np.repeat(np.arange(len(gg)),count[b]);xb[b,:len(ix)]=x[ix];yb[b,:len(ix)]=bs[b,ix]
            if level=='size_adjusted_global':
                z=[];cells=sorted(gg.cell.unique());strata=[];sizes=[]
                for cell in cells:
                    tag=gg[gg.cell==cell].tag.iloc[0]
                    meta=json.loads((HERE/(tag+'.json')).read_text())
                    strata.append(meta['job']['cell']);sizes.append(np.log(meta['episodes']))
                z=np.column_stack([np.ones(len(cells)),np.asarray(sizes),pd.get_dummies(strata,drop_first=True).to_numpy(float)])
                br=residual_rank_rho(xb,yb,z);point=residual_rank_rho(x,y,z)[0]
            else:br=matrix_rho(xb,yb);point=rho(x,y)
            ci=interval(br)
            ranks.append(dict(source=source,candidate=candidate,target=target,level=level,points=len(x),banks=g.cell.nunique(),rho=point,lo=ci[0],hi=ci[1],
                              scope='fixed banks; hierarchical task/init uncertainty; seeds clustered'))
    return frame.drop(columns='draw'),pd.DataFrame(ranks)


def complete_cells(ep, smoke):
    result={}
    for cell,g in ep.groupby('cell'):
        expected={(cohort,rep,t,i) for cohort in COHORTS for rep in range(3) for t in range(10) for i in range(2)}
        actual=set(map(tuple,g[['cohort','block','task_id','init']].to_numpy()))
        result[cell]=dict(pilot_complete=expected.issubset(actual) and not smoke,
                          episodes=len(g), missing_pilot_slots=len(expected-actual),cohorts=sorted(g.cohort.unique()))
    return result


def decision_rule(state, comparisons, completeness, smoke):
    complete=len(completeness)==8 and all(v['pilot_complete'] for v in completeness.values())
    decisions={}
    for k in PRIMARY:
        p=state[(state.scope=='pooled')&(state.candidate==k)]
        cs=state[(state.scope!='pooled')&(state.candidate==k)]
        if not len(p):continue
        r=p.iloc[0]
        passes=(r.rho>=.20 and r.rho_sim_lo>0 and r.mae_gain>=.05 and r.gain_sim_lo>0
                and (cs.rho>=0).all() and (cs.mae_gain>=0).all() and (cs.rho_fraction>=.8).all())
        rejected=r.rho_sim_hi<.20 or r.gain_sim_hi<.05
        decisions[k]='SMOKE_ONLY' if smoke else ('PARTIAL_NO_SELECTION' if not complete else ('survives' if passes else 'practical_utility_rejected' if rejected else 'inconclusive'))
    choice=None
    for k in PRIMARY:
        if decisions.get(k)!='survives':continue
        if choice is None:choice=k
        else:
            c=comparisons[(comparisons.scope=='pooled')&(comparisons.simpler==choice)&(comparisons.complex==k)]
            if len(c) and c.iloc[0].lo>0:choice=k
    return dict(complete_eight_cells=complete,scientific_use=not smoke,status=decisions,selected=choice,
                permitted_use='state disagreement diagnostic only; no validated SR probability or call utility')


def secondary_tables(x, em, ep, infer, out):
    """Diagnostics kept out of the primary survival decision."""
    all_states=[]
    for cohort in sorted(set(em.cohort)-{'A'}):
        s,_=state_estimates(em,infer,cohort);all_states.append(s)
    pd.concat(all_states,ignore_index=True).to_csv(out/'state_by_other_cohort.csv',index=False) if all_states else None
    # Task estimates cannot acquire an init CI from repeated seeds of a single init.
    per_task=[];noise=[];occupancy=[];label_sensitivity=[]
    val=em[(em.cohort=='A')&(em.split=='validation')]
    eligible_cells=[cell for cell,g in val.groupby('cell') if (g.groupby('task_id').init.nunique()>=2).all()]
    dump(out/'fixed_task_scope.json',dict(eligible_cells=eligible_cells,
         excluded_reason='Need two or more distinct validation inits per task; seed repeats do not supply this contrast.'))
    if eligible_cells:
        fem=em[em.cell.isin(eligible_cells)]
        fixed=Inference(fem[fem.split=='validation'],len(infer.mult),fixed_tasks=True)
        fs,_=state_estimates(fem,fixed);fs.to_csv(out/'fixed_task_state_sensitivity.csv',index=False)
    for (cell,task),g in val.groupby(['cell','task_id']):
        for k in PRIMARY:
            p,b=infer.aggregate(g,['rho_'+k,'mae_'+k,'baseline_mae'])
            ci=interval(b[:,0]) if g.init.nunique()>=2 else [np.nan,np.nan]
            per_task.append(dict(cell=cell,task_id=task,candidate=k,episodes=len(g),distinct_inits=g.init.nunique(),rho=p[0],
                rho_lo=ci[0],rho_hi=ci[1],mae=p[1],baseline_mae=p[2],
                uncertainty='unavailable: one validation init' if g.init.nunique()<2 else 'hierarchical conditional-training interval'))
    for cell,g in val.groupby('cell'):
        ac=x[(x.cell==cell)&(x.cohort=='A')&(x.split=='validation')]
        aw=hierarchy_weights(ac)*np.where(ac.selected,1/ac.sample_p,0)
        mass=ac[EKEY+CLUSTER].copy();mass['weight']=aw
        ew=mass.groupby(EKEY).weight.sum().to_numpy();cw=mass.groupby(CLUSTER).weight.sum().to_numpy()
        episode_ess=ew.sum()**2/(ew@ew) if ew@ew else 0
        cluster_ess=cw.sum()**2/(cw@cw) if cw@cw else 0
        for label in ['V_HT','L_HT','excess_HT','mean_policy_error_HT']:
            p,b=infer.aggregate(g,[label]);ci=interval(b[:,0])
            noise.append(dict(cell=cell,label=label,estimate=p[0],lo=ci[0],hi=ci[1],episodes=len(g),
                              selected_anchors=int(g.k4_selected.sum()),zero_selection_episodes=int((g.k4_selected==0).sum()),
                              episode_selection_weight_ess=episode_ess,cluster_selection_weight_ess=cluster_ess))
        # Outcome association is explicitly descriptive, with no SR fit.
        cg=g.groupby(['cell']+CLUSTER)[['Y']+PRIMARY+[k+'_first' for k in PRIMARY]].mean().reset_index()
        pos=[infer.lookup[tuple(r)] for r in cg[CLUSTER].itertuples(index=False,name=None)]
        mw=infer.mult[:,pos]
        for k in PRIMARY+[k+'_first' for k in PRIMARY]:
            n=int(mw.sum(1).max());xx=np.full((len(mw),n),np.nan);yy=xx.copy()
            for b in range(len(mw)):
                ix=np.repeat(np.arange(len(cg)),mw[b]);xx[b,:len(ix)]=cg[k].to_numpy()[ix];yy[b,:len(ix)]=1-cg.Y.to_numpy()[ix]
            ci=interval(matrix_rho(xx,yy))
            occupancy.append(dict(cell=cell,candidate=k,rho=rho(cg[k],1-cg.Y),lo=ci[0],hi=ci[1],clusters=len(cg),
                                  status='first anchor association' if k.endswith('_first') else 'retrospective occupancy; not an online prediction'))
    # Horizon and K4 rank tests: no recalibration/no substitution into primary rule.
    xa=x[(x.cohort=='A')&(x.split=='validation')]
    er=[]
    for _,g in xa.groupby(EKEY):
        base={k:g.iloc[0][k] for k in EKEY+['cell']+CLUSTER}
        for label in ['E_head','E_full','E_raw','excess']:
            sel=g[g.selected] if label=='excess' else g
            for k in PRIMARY:
                rr=weighted_rho(sel[k],sel[label],1/sel.sample_p) if label=='excess' else rho(sel[k],sel[label])
                er.append(dict(**base,label=label,candidate=k,rho=rr))
    er=pd.DataFrame(er)
    for (cell,label,k),g in er.groupby(['cell','label','candidate']):
        p,b=infer.aggregate(g,['rho']);ci=interval(b[:,0])
        label_sensitivity.append(dict(cell=cell,label=label,candidate=k,rho=p[0],lo=ci[0],hi=ci[1],
                                      episodes_with_rank=int(g.rho.notna().sum()),episodes=len(g)))
    for name,rows in [('per_task_state_estimates',per_task),('policy_noise_estimates',noise),
                      ('occupancy_outcome_associations',occupancy),('label_sensitivity',label_sensitivity)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    # Whole-dose screening is a flag for Q2, never a new policy decision.
    from exp.offline_search.rounds.r06.p3_profiling.pilot_stats import icc
    precision=[]
    joined=ep.merge(em[EKEY+['E']],on=EKEY,validate='one_to_one')
    for (cell,cohort),g in joined.groupby(['cell','cohort']):
        for value in ['Y','E','deployment_IR_per_request']:
            groups=list(g.groupby(['task_id','init']))
            stats=icc([a[value].to_numpy() for _,a in groups],[key[0] for key,_ in groups])
            precision.append(dict(cell=cell,cohort=cohort,value=value,split='all; precision planning only',**stats))
        for ref in ['A','P10']:
            if cohort==ref:continue
            cg=joined[(joined.cell==cell)&(joined.cohort==ref)]
            paired=g.merge(cg,on=['cell']+CLUSTER+['client_environment_seed'],suffixes=('','_ref'),validate='one_to_one')
            if not len(paired):continue
            paired['delta']=paired.Y-paired.Y_ref
            groups=list(paired.groupby(['task_id','init']))
            stats=icc([a.delta.to_numpy() for _,a in groups],[key[0] for key,_ in groups])
            precision.append(dict(cell=cell,cohort=cohort,value='Y_minus_'+ref,split='all; precision planning only',
                                   discordance=mean(paired.delta**2),paired_variance=float(paired.delta.var(ddof=1)),**stats))
    dump(out/'precision_icc.json',precision)
    # Transport stress test: train coefficients on the other cells' calibration
    # trajectories; never use this target cell's calibration or validation labels.
    loco=x.copy();params=[]
    for cell,g in x.groupby('cell'):
        train=x[(x.cell!=cell)&(x.cohort=='A')&(x.split=='calibration')]
        if not len(train):continue
        wt=hierarchy_weights(train);base=np.average(train.E,weights=wt)
        loco.loc[g.index,'baseline']=base
        for k in PRIMARY+SECONDARY:
            good=np.isfinite(train[k])
            if good.sum()<2:loco.loc[g.index,'pred_'+k]=np.nan;continue
            xx=train.loc[good,k].to_numpy();sw=np.sqrt(wt[good]);yy=train.loc[good,'E'].to_numpy()
            beta=nnls(np.column_stack([np.ones(len(xx)),xx])*sw[:,None],yy*sw)[0]
            loco.loc[g.index,'pred_'+k]=beta[0]+beta[1]*g[k]
            params.append(dict(held_cell=cell,candidate=k,intercept=beta[0],slope=beta[1],training_cells=sorted(train.cell.unique())))
    if params:
        ls,_=state_estimates(episode_metrics(loco),infer);ls.to_csv(out/'leave_cell_out_state_estimates.csv',index=False)
    dump(out/'leave_cell_out_parameters.json',params)


def budget_screen(contrasts, smoke, completeness):
    if contrasts.empty:return []
    rows=[]
    for cell,g in contrasts[contrasts.reference=='P10'].groupby('cell'):
        for cohort in ['A','dose125','dose25','dose50','B']:
            sr=g[(g.cohort==cohort)&(g.outcome=='Y')]
            ir=g[(g.cohort==cohort)&(g.outcome=='deployment_IR_per_request')]
            if not len(sr) or not len(ir):continue
            supported=sr.iloc[0].screening_lo>=-.05 and ir.iloc[0].screening_hi<0
            rows.append(dict(cell=cell,cohort=cohort,passes_descriptive_screen=bool(supported),
                             eligible_for_discussion=bool(supported and not smoke and completeness[cell]['pilot_complete']),
                             note='five-point screening margin only; independent near-pure-inference SR confirmation needed'))
    return rows


def plots(a, state, epmeans, out, smoke):
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    s=state[(state.scope=='pooled')&state.candidate.isin(PRIMARY)]
    axs[0].bar(s.candidate,s.rho)
    axs[0].set(ylabel='Mean within-episode Spearman',title='A validation state disagreement')
    for cell,g in epmeans.groupby('cell'):
        axs[1].scatter(g.deployment_IR_per_request,g.Y,label=cell)
        for _,r in g.iterrows():axs[1].annotate(r.cohort,(r.deployment_IR_per_request,r.Y),fontsize=6)
    axs[1].set(xlabel='Deployment IR / request',ylabel='Episode success',title='Matched cohorts (descriptive)')
    axs[1].legend(fontsize=6)
    fig.suptitle('SMOKE: PLUMBING ONLY, NO SCIENTIFIC CONCLUSIONS' if smoke else 'Q1 preregistered analysis')
    fig.tight_layout();fig.savefig(out/'q1_overview.png',dpi=150);fig.savefig(out/'q1_overview.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for cell,g in a[(a.cohort=='A')&(a.split=='validation')].groupby('cell'):
        v=g.groupby(EKEY+['phase_bin']).E.mean().groupby('phase_bin').mean()
        ax.plot((v.index+.5)/5,v.values,marker='o',label=cell)
    ax.set(xlabel='Retrieved library progress (five bins)',ylabel='Standardized cache–policy RMS',title='SMOKE ONLY' if smoke else 'A occupancy by retrieved phase')
    ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/'q1_phase.png',dpi=150);plt.close(fig)


def resolve_tables(args, out):
    if args.tables:
        return Path(args.tables)
    if not args.run_root:raise ValueError('give --tables or --run-root with --arms / --cell')
    run=Path(args.run_root)
    arms=args.arms
    if not arms and args.cell:
        arms=[f'r6p3v2_{args.cell}_{c}_r{r}' for c in COHORTS for r in range(3)]
    if not arms:raise ValueError('explicit finished arms or cell required')
    # No queue polling: one check, fail immediately on any incomplete marker.
    for arm in arms:
        markers=list((run/'state').glob(arm+'.DONE'))+list((run/'state').glob(arm+'.manifest_*.DONE'))
        if not markers:raise ValueError('arm not marked finished: '+arm)
    # read_v2 recursively indexes client-root. Run per selected arm so it never
    # traverses unfinished/unselected arms, then concatenate audited products.
    dest=out/'tables';dest.mkdir()
    audit={}
    for arm in arms:
        sub=out/'reader'/arm
        cmd=['taskset','-c','14-17,58-61','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
             'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',str(ROOT/'.venv/bin/python'),'-m',
             'exp.offline_search.rounds.r06.p3_profiling.read_v2','--run-root',str(run),'--arms',arm,
             '--client-root',str(run/'runs'/arm),'--require-stage-counts','--require-snapshots','--out',str(sub)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        audit[arm]=json.loads((sub/'audit.json').read_text())
    for name in ['anchors','decisions','episodes','action_steps','attempts','neighbours']:
        frames=[table_read(out/'reader'/arm,name) for arm in arms]
        pd.concat(frames,ignore_index=True).to_csv(dest/(name+'.csv'),index=False)
    dump(dest/'audit.json',audit)
    return dest


def main():
    source_sha=sha(Path(__file__))
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--tables',type=Path)
    source.add_argument('--run-root',type=Path)
    ap.add_argument('--arms',nargs='+')
    ap.add_argument('--cell',help='e.g. pi05_l10_50; requires all 27 pilot arms finished')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=10000)
    args=ap.parse_args()
    out=args.out.resolve()
    if HERE not in out.parents:raise ValueError('output must be under ideation_Q1')
    if args.bootstrap<10000 and not args.smoke:raise ValueError('preregistered inference needs 10000 draws')
    out.mkdir(parents=True,exist_ok=False)
    tables=resolve_tables(args,out)
    if 'smoke' in str(tables).lower() and not args.smoke:raise ValueError('smoke tables require --smoke')
    ep,a,d,action,attempts=[table_read(tables,n) for n in ['episodes','anchors','decisions','action_steps','attempts']]
    if args.arms and args.tables:
        ep,a,d,action,attempts=[f[f.arm.isin(args.arms)].copy() for f in [ep,a,d,action,attempts]]
    counts=validate_tables(ep,a,d,action,attempts)
    cats,catpaths=catalogs(d)
    fps={};cals={};bridges=[]
    for digest,c in cats.items():
        cal=calibration(c,fps);cals[digest]=cal;bridges.append(dict(catalog_sha256=digest,**cal[2]))
    dump(out/'provenance_bridges.json',bridges)
    x=build_anchors(a,action,cats,cals)
    if not (x.cohort=='A').any():raise ValueError('Q1 requires at least one finished A reference arm with calibration and validation inits')
    x,params=refit(x)
    em=episode_metrics(x)
    ep=ep.merge(em[EKEY+['cell','cohort','block','split','tag']],on=EKEY,validate='one_to_one')
    if ep.duplicated(['cell','cohort','block','task_id','init']).any():raise ValueError('duplicate design slot')
    inf=Inference(ep[ep.split=='validation'],args.bootstrap)
    state,comparison=state_estimates(em,inf)
    cm,contrasts=cohorts_and_contrasts(ep,inf)
    support,effects=support_and_excursions(x,inf)
    bank,ranks=old_bank_ranks(ep,em,inf)
    completeness=complete_cells(ep,args.smoke)
    decision=decision_rule(state,comparison,completeness,args.smoke)
    for name,frame in [('anchor_scores',x),('episode_scores',em),('episode_outcomes',ep),('state_estimates',state),('candidate_comparisons',comparison),
                       ('cohort_means',cm),('paired_contrasts',contrasts),('support',support),('factorial_excursions',effects),
                       ('bank_task_scores_outcomes',bank),('bank_rank_correlations',ranks)]:frame.to_csv(out/(name+'.csv'),index=False)
    phase=x.groupby(['cell','cohort','split','task_id','phase_bin'])[PRIMARY+['E','E_head','E_raw']].agg(['mean','count'])
    phase.to_csv(out/'phase_descriptive.csv')
    task=em.groupby(['cell','cohort','split','task_id'])[['E','Y']+['rho_'+k for k in PRIMARY]].mean()
    task.to_csv(out/'per_task_descriptive.csv')
    dump(out/'calibration_parameters.json',params);dump(out/'completeness.json',completeness);dump(out/'decision.json',decision)
    secondary_tables(x,em,ep,inf,out)
    dump(out/'q2_budget_screen.json',budget_screen(contrasts,args.smoke,completeness))
    plots(x,state,cm,out,args.smoke)
    manifest=dict(smoke=args.smoke,scientific_conclusions_allowed=not args.smoke,bootstrap=args.bootstrap,seed=SEED,
                  counts=counts,calibration_clusters=len(ep[ep.split=='calibration'][['cell']+CLUSTER].drop_duplicates()),
                  validation_clusters=len(ep[ep.split=='validation'][['cell']+CLUSTER].drop_duplicates()),
                  bridge_passes=sum(b['enabled'] for b in bridges),bridges=len(bridges),
                  input_tables={str(tables/(n+'.csv')):sha(tables/(n+'.csv')) for n in ['episodes','anchors','decisions','action_steps','attempts']},
                  catalog_startup_paths=catpaths,script_sha256=source_sha,prereg_sha256=sha(HERE/'PREREG.md'),
                  restrictions='All pilot outcome inference remains unrun at implementation time; smoke plumbing only.')
    if sha(Path(__file__))!=source_sha:raise ValueError('analysis source changed while running; rerun into a fresh output directory')
    dump(out/'run_manifest.json',manifest)
    print(json.dumps(dict(output=str(out),smoke=args.smoke,counts=counts,bridge_passes=manifest['bridge_passes'],bridges=len(bridges),selection=decision['selected'])))


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='invalid value encountered in cast',category=RuntimeWarning)
    main()
