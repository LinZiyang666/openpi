"""Weighted binary ranking metrics, grouped ties, NumPy only."""
import numpy as np


def ranking(y,score,sample_weight=None):
    y=np.asarray(y,dtype=float);s=np.asarray(score,dtype=float)
    w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,dtype=float)
    if not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(s).all() or np.any(w<0):
        raise ValueError('invalid ranking inputs')
    ix=np.argsort(-s,kind='stable');y,s,w=y[ix],s[ix],w[ix]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*(1-y))[ends]
    pos,neg=tp[-1],fp[-1]
    ap=float(np.sum(np.diff(np.r_[0,tp/max(pos,1e-15)])*tp/np.maximum(tp+fp,1e-15)))
    auc=float(np.sum(np.diff(np.r_[0,fp/max(neg,1e-15)])*(np.r_[0,tp[:-1]]+tp)/(2*max(pos,1e-15))))
    return ap,auc


def average_precision_score(y,score,sample_weight=None):return ranking(y,score,sample_weight)[0]
def roc_auc_score(y,score,sample_weight=None):return ranking(y,score,sample_weight)[1]
