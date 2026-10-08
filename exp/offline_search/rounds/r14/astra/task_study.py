from common import *
fence()
import numpy as np
from scipy.stats import spearmanr

fits=read(HERE/'allocation_fits.json');out=[];ranks=[]
metas=[read(p) for p in sorted((HERE/'data').glob('*.json'))]
for cell in CELLS:
    lib=next(f['priority'] for f in fits if f['cell']==cell)
    cache={}
    for m in metas:
        if m['cell']!=cell or m['method'] not in ('off','periodic'):continue
        with np.load(HERE/'data'/f'{m["root"]}__{m["arm"]}.npz') as z:d={k:z[k] for k in z.files}
        vals=[]
        for t in range(10):
            k=d['keys'][:,0]==t;rows=np.isin(d['ep'],np.flatnonzero(k));look=rows&d['vision'].astype(bool)
            risks=[]
            for j in np.flatnonzero(k):
                rr=(d['ep']==j)&look&np.isfinite(d['risk'])&~d['guard'].astype(bool)
                if rr.any():risks.append(float(np.log1p(d['risk'][rr].mean())))
            gap=float(np.mean(risks)) if risks else None;vals.append(gap)
            cv,cm=COSTS[m['model']]
            out.append(dict(cell=cell,root=m['root'],method=m['method'],task=t,episodes=int(k.sum()),
                 sr=float(d['success'][k].mean()),IR=float((cv*d['epV'][k].sum()+cm*d['epM'][k].sum())/d['epN'][k].sum()),
                 library_log_gap=lib['raw'][t],dev_log_gap=gap,labelled_episodes=len(risks)))
        if all(x is not None for x in vals):
            ranks.append(dict(cell=cell,root=m['root'],method=m['method'],tasks=len(vals),
                         spearman_library_dev=float(spearmanr(lib['raw'],vals).statistic)))
csvout(HERE/'task_descriptives.csv',out);dump(HERE/'task_rank_agreement.json',ranks)
provenance('task_study');print(ranks)
