"""Shadow labels and library proxy diagnostics; no A outcomes or design input."""
from common import *
fence()
import numpy as np
from scipy.stats import rankdata, spearmanr

def stats(d,r,ep,task):
    good=np.isfinite(d)&np.isfinite(r)
    d,r,ep,task=[np.asarray(x)[good] for x in (d,r,ep,task)]
    if len(d)<10: return None
    _,iv,ct=np.unique(ep,return_inverse=True,return_counts=True)
    w=1/ct[iv]; w=w/w.sum()
    u=rankdata(d,method='average')/len(d)
    top=u>.75
    er=float(np.sum(w*r)); topmass=float(w[top].sum())
    frac=float(np.sum(w[top]*r[top])/np.sum(w*r))
    corrs=[]
    for e in np.unique(ep):
        q=ep==e
        if q.sum()>=5 and np.std(d[q])>0 and np.std(r[q])>0:
            corrs.append(float(spearmanr(d[q],r[q]).statistic))
    # Demean ranks within task to distinguish pooled task scale from placement.
    rd,rr=rankdata(d),rankdata(r)
    for t in np.unique(task):
        q=task==t; rd[q]-=rd[q].mean(); rr[q]-=rr[q].mean()
    adjusted=float(np.corrcoef(rd,rr)[0,1])
    return dict(looks=len(d),episodes=len(np.unique(ep)),distance_median=float(np.median(d)),
        error_episode_mean=er,pooled_spearman=float(spearmanr(d,r).statistic),
        task_demeaned_rank_correlation=adjusted,
        within_episode_spearman_mean=float(np.mean(corrs)),
        top_quartile_error_capture=frac,top_quartile_episode_mass=topmass,
        top_quartile_lift=frac/topmass)

def main():
    rows=[]; audit=[]; pathrows=[]
    for cell in CELLS:
        with np.load(R11/'astra/data'/cell/'signals.npz',allow_pickle=False) as z:
            q=z['step']%2==0
            out=stats(z['distance'][q],z['risk'][q],z['ep'][q],z['task'][q])
            rows.append(dict(cell=cell,source='library_whole_episode_heldout',population='all_even_anchors',**out))
    for mp in sorted((HERE/'data').glob('*.json')):
        meta=read(mp)
        with np.load(mp.with_suffix('.npz'),allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        q=a['vision'].astype(bool)
        if np.isfinite(a['distance'][q]).any():
            pathrows.append(dict(cell=meta['cell'],source=meta['root'],method=meta['method'],
                episodes=meta['n'],ir=meta['ir'],sr=meta['sr'],
                median_distance=float(np.nanmedian(a['distance'][q])),
                q90_distance=float(np.nanquantile(a['distance'][q],.9)),
                guard_per_look=float(a['guard'][q].mean())))
        if not np.isfinite(a['risk'][q]).any(): continue
        for pop,mask in [('all_looks',q),('guard_eligible',q&~a['guard'].astype(bool)),('guard_calls',q&a['guard'].astype(bool))]:
            out=stats(a['distance'][mask],a['risk'][mask],a['ep'][mask],a['task'][mask])
            if out: rows.append(dict(cell=meta['cell'],source=meta['root'],population=pop,**out))
        audit.append({k:meta[k] for k in ('cell','looks','replay_hit_max_abs_error','recorded_kernel_min_zero','distance_log_reconstruction_max')})
    csvout(HERE/'signal_quality.csv',rows); csvout(HERE/'distance_paths.csv',pathrows);dump(HERE/'shadow_audit.json',audit)
    dump(HERE/'signal_quality_summary.json',rows)
    provenance('signal')
    print(table(['cell','source','population','Spearman','within episode','top-quartile lift'],
        [[r['cell'],r['source'],r['population'],f"{r['pooled_spearman']:.3f}",f"{r['within_episode_spearman_mean']:.3f}",f"{r['top_quartile_lift']:.3f}"] for r in rows]))

if __name__=='__main__': main()
