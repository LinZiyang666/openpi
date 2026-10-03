import numpy as np, pandas as pd
from exp.offline_search.rounds.r09.explore_fable.tools import paired, common
eps = paired.load_episodes()
disc = list(common.DISCOVERY_INITS)
folds3 = [disc[0::3], disc[1::3], disc[2::3]]
for model, suite, lib in common.CELLS:
    M = paired.cell_matrix(eps, model, suite, lib)
    Md = M[M.index.get_level_values('init').isin(disc)]
    variants = paired.variants_of(M)
    cv = [v for v in ("A","O5b","O5a","IP","CU","CT","P10") if v in variants]
    print(f"\n=== {model} {suite} {lib}  (discovery inits, n={len(Md)})  target P10={common.P10_SR[(model,suite)]}")
    arms = {v: paired.summarize(Md, pd.Series(v, index=Md.index), n_boot=0) for v in cv}
    print("  arms:", "  ".join(f"{v}:{a['sr']:.3f}@{a['ir']:.3f}" for v,a in arms.items()))
    oracle = paired.hull(paired.task_frontier(M, cv, disc, disc, n_boot=0), 'eval_sr', 'eval_ir')
    print("  oracle hull:", "  ".join(f"{p['eval_sr']:.3f}@{p['eval_ir']:.3f}" for p in oracle))
    cvh = paired.hull(paired.cv_frontier(M, cv, folds3, n_boot=0))
    print("  3-fold CV hull:", "  ".join(f"{p['sr']:.3f}@{p['ir']:.3f}" for p in cvh))
    xs = sorted((a['ir'], a['sr']) for a in arms.values())
    ux, uy = [x for x,_ in xs], [y for _,y in xs]
    gains = [(p['ir'], p['sr'] - float(np.interp(p['ir'], ux, uy))) for p in cvh]
    print("  CV gain vs uniform-arm interpolation (pp):", "  ".join(f"{ir:.3f}:{100*g:+.1f}" for ir,g in gains))
