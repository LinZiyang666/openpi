"""Whole-episode task routing: identifiable offline value, honest init cross-fitting.

All actions are existing COMPLETE controllers, selected before an episode starts.
This estimator never splices trajectories. Its value is the expected outcome over
the router's episode lottery conditional on the observed controller realizations.
The optimizer minimizes a ratio of additive work/decision counts, not mean IR.
"""
import json
import numpy as np
from scipy.optimize import linprog
from .common import HERE, DERIVED, dump, load_compact, mixture_ir, paired_bootstrap


def optimize(y, c, n, prior=12, tolerance=.01, weights=None):
    """[task,train_init,arm], arm 0 pure policy. Shrink paired SR differences."""
    nt, ni, na = y.shape
    weights=np.ones(y.shape[:2]) if weights is None else np.asarray(weights)
    mass=weights.sum(1)[:,None]
    if np.any(mass<=0):raise ValueError('each task needs positive training mass')
    mean=lambda value:(value*weights[:,:,None]).sum(1)/mass
    delta = mean(y - y[:, :, :1])
    delta = (mass * delta + prior * delta.mean(0, keepdims=True)) / (mass + prior)
    c, n = mean(c), mean(n)
    eq = np.zeros((nt, nt*na))
    for t in range(nt):
        eq[t, t*na:(t+1)*na] = 1
    rate = .5
    for _ in range(40):
        sol = linprog((c-rate*n).ravel(), A_ub=-delta.reshape(1, -1), b_ub=[nt*tolerance],
                      A_eq=eq, b_eq=np.ones(nt), bounds=(0, 1), method='highs')
        if not sol.success:
            raise RuntimeError(sol.message)
        p = sol.x.reshape(nt, na)
        new = np.sum(p*c)/np.sum(p*n)
        if abs(new-rate) < 1e-10:
            break
        rate = new
    return p, delta


def evaluate(y, c, n, p):
    v = (y*p).sum(-1)
    d = v-y[:, :, 0]
    return dict(sr=float(v.mean()), ir=mixture_ir(c, n, p),
                sr_difference=paired_bootstrap(d),
                reference_sr=float(y[:, :, 0].mean()),
                reference_ir=float(c[:, :, 0].sum()/n[:, :, 0].sum()),
                pure_policy_share=float(p[:, :, 0].mean()))


def run():
    meta = [json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')]
    results = []
    for model in ['pi05','groot']:
        for suite in ['l10','spatial']:
            for size in [50,500]:
                names = {m['variant']:m['arm'] for m in meta if m['model']==model and m['suite']==suite and
                         (m['library_size']==size or m['variant']=='P10')}
                for menu, variants in [('binary',['P10','A']),
                                       ('perception',['P10','A','SW','W10','SF1']),
                                       ('calls',['P10','A','CU'])]:
                    variants = [v for v in variants if v in names]
                    if 'P10' not in variants or 'A' not in variants:
                        continue
                    data = [load_compact(names[v]) for v in variants]
                    y,c,n = [np.stack([z[k] for z in data],axis=-1) for k in ['success','cost','decisions']]
                    for prior in [0,12,30]:
                        for tolerance in [0., .01, .02]:
                            p_oof = np.zeros_like(y)
                            folds = []
                            for fold in range(5):
                                test = np.arange(30)%5 == fold
                                p, _ = optimize(y[:,~test],c[:,~test],n[:,~test],prior,tolerance)
                                p_oof[:,test] = p[:,None,:]
                                folds.append(p.tolist())
                            full, delta = optimize(y,c,n,prior,tolerance)
                            rec = dict(cell=f'{model}_{suite}_{size}',menu=menu,variants=variants,arms=[names[v] for v in variants],
                                       prior=prior,tolerance=tolerance,oof=evaluate(y,c,n,p_oof),
                                       in_sample=evaluate(y,c,n,np.broadcast_to(full[:,None,:],y.shape)),
                                       full_probabilities=full.tolist(),fold_probabilities=folds,
                                       task_sr=y.mean(1).tolist(),task_delta_shrunk=delta.tolist())
                            results.append(rec)
                            if prior==12 and tolerance==.01:
                                print(rec['cell'],menu,json.dumps(rec['oof']),flush=True)
    dump(HERE/'results/routing.json',results)


if __name__=='__main__':
    run()
