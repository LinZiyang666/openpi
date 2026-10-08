"""Own-episode exclusion before V7 diagnostic calibration reference selection."""
import copy
import numpy as np


def donor_scores(original, audit):
    def score(base,q):
        if not hasattr(q,'_r') or not hasattr(q,'_L'):
            return original(base,q)
        episode=int(q._L.episode[q._r])
        task=int(q.task_id);T=base.tasks[task]
        mask=base.lib_ep[T.rows]!=episode
        if not mask.any():
            raise ValueError('no cross-episode diagnostic calibration donors')
        selected=copy.copy(T)
        for name in ('rows','Z','z2','HD','h2','V0','V1','RS','rs2','Z0','n20'):
            a=getattr(T,name,None)
            if a is not None:setattr(selected,name,a[mask])
        base.tasks[task]=selected
        try:
            rows,S,aux=original(base,q)
            assert np.all(base.lib_ep[rows]!=episode)
            audit['queries']+=1
            audit['removed_rows']+=int(np.count_nonzero(~mask))
            return rows,S,aux
        finally:
            base.tasks[task]=T
    return score
