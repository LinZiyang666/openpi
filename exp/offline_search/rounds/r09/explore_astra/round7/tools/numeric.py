"""Serving math: observation, action and retrieval statistics only."""
import numpy as np
from exp.offline_search.rounds.r09.explore_astra.round6.tools.numeric import dense_predict


def features(x, enriched):
    if not enriched: return x
    a=x[:,136:206].reshape(-1,10,7)
    closed=(a[:,:,6]<0).astype(np.float32)  # GR00T normalized -1 closes.
    # Smooth motion/closure interactions and chunk events; no task-derived feature.
    return np.c_[x, closed.mean(1),closed[:,0],closed[:,-1],
        np.abs(np.diff(closed,axis=1)).sum(1),a[:,:,:6].mean(1),a[:,5:,:6].mean(1)-a[:,:5,:6].mean(1),
        a[:,:,:6].mean(1)*closed.mean(1)[:,None],x[:,134:136]*closed.mean(1)[:,None]].astype(np.float32)


def confidence(actions, rows, weights, scale):
    """Weighted retrieved motion disagreement, independent of row numbering."""
    w=np.asarray(weights,np.float64); w=w/w.sum(1,keepdims=True)
    a=actions[np.asarray(rows),:10,:6].reshape(len(rows),rows.shape[1],60)
    mu=np.einsum('nk,nko->no',w,a)
    variance=np.maximum(0,np.einsum('nk,nko->n',w,a*a)/60-(mu*mu).mean(1))
    return 1/(1+variance/max(float(scale),1e-8))


def predict(head, table, x, rows, weights, actions, meta):
    w=np.asarray(weights,np.float64); w=w/w.sum(1,keepdims=True)
    shared=dense_predict(head,features(x,meta['enriched']))
    local=np.einsum('nk,nko->no',w,table[np.asarray(rows)])
    strength=.5+meta.get('confidence_gain',0.)*confidence(actions,rows,w,meta['dispersion_scale'])
    return (strength[:,None]*(shared+local)).reshape(-1,10,6)
