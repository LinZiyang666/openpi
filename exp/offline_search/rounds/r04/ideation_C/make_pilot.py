"""Concrete nested init manifests. Does not launch workers or modify ops."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
import numpy as np
from exp.offline_search.rounds.r04.ideation_C.pilot_diagnostic import ys,alloc,OUT

for scale in [50,500]:
    for model,l in [('pi05','p'),('groot','g')]:
        for suite,s in [('spatial','sp'),('l10','l10')]:
            run=f'r02_g{scale}';prefix=f'oscl{scale}_{l}_{s}_cl'
            y0=ys(run,prefix+'2');cal=[ys(run,prefix+str(i)) for i in [0,1,3]]
            probabilities={}
            for outcome in [0,1]:
                obs=np.concatenate([y[y0==outcome] for y in cal]);p=(obs.sum()+1)/(len(obs)+2)
                probabilities[outcome]=p
            groups=[(t,b,np.flatnonzero(y0[t]==b)) for t in range(10) for b in [0,1] if np.any(y0[t]==b)]
            Nh=np.array([len(g) for t,b,g in groups]);sd=np.array([np.sqrt(probabilities[b]*(1-probabilities[b])) for t,b,g in groups])
            rng=np.random.default_rng(20260927)
            perm=[rng.permutation(g) for t,b,g in groups]
            for n in [100,200,250]:
                nh=alloc(n,Nh,sd);selected=[];strata=[]
                for h,(t,b,g) in enumerate(groups):
                    pick=perm[h][:nh[h]]
                    selected.extend(dict(task=int(t),init=int(i),stratum=h,inclusion_probability=float(nh[h]/Nh[h])) for i in pick)
                    strata.append(dict(stratum=h,task=t,reference_success=b,N=int(Nh[h]),n=int(nh[h]),population_weight=float(Nh[h]/500)))
                obj=dict(model=model,suite=suite,scale=scale,reference=f'{run}:{prefix}2',seed=20260927,n=n,
                         calibration='borrowed big-library information: historical evaluation outcomes; no deployed policy fit',
                         strata=strata,selected=selected)
                (OUT/f'pilot_manifest_{model}_{suite}_{scale}_n{n}.json').write_text(json.dumps(obj,indent=2))
print('24 nested manifests written; no closed-loop workers launched.')
