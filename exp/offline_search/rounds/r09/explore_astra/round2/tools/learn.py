"""Weighted shared ridge/RFF head; task identity never enters the fit."""
import numpy as np
from scipy.linalg import solve
from .data import SEED, balanced_weights


def fit(x, y, episode, random_features=768, alpha=100.):
    weights = balanced_weights(episode)
    mean = np.average(x, axis=0, weights=weights)
    std = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)), 1e-4)
    xx = np.clip((x-mean)/std, -8, 8)
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(x.shape[1],random_features))/np.sqrt(x.shape[1])
    bias = rng.uniform(0,2*np.pi,random_features)
    xx = np.c_[xx, np.cos(xx@w+bias)*np.sqrt(2)]
    fm = np.average(xx, axis=0, weights=weights)
    ym = np.average(y, axis=0, weights=weights)
    centered = xx-fm
    coef = solve(centered.T@(centered*weights[:,None])+alpha*np.eye(xx.shape[1]),
                 centered.T@((y-ym)*weights[:,None]), assume_a='pos')
    return {k: np.asarray(v,np.float32) for k,v in dict(mean=mean,std=std,w=w,bias=bias,
        coef=coef,intercept=ym-fm@coef).items()}
