"""NumPy-only, task-agnostic shared-head inference and exact temporal features."""
import numpy as np


def design(model, x):
    xx = np.clip((x-model['mean'])/model['std'], -8, 8)
    if model['w'].shape[1]:
        xx = np.c_[xx, np.cos(xx@model['w']+model['bias'])*np.sqrt(2)]
    return xx


def predict(model, x):
    return design(model, x)@model['coef'] + model['intercept']


def inputs(visual, state, action, step, previous=None, history=False):
    state = np.asarray(state)[:8]
    action = np.asarray(action)[:10,:7]
    x = np.r_[visual, state, action.ravel(), min(step,120)/120]
    if history:
        delta, prev, gap = np.zeros(8), np.zeros(7), 0.
        if previous is not None:
            delta = state-previous['state']
            prev = previous['action'].mean(0)
            gap = min(step-previous['step'],12)/12
        x = np.r_[x, delta, prev, gap]
    return x[None]


def correct(model, x, action, blend=.5, gripper=False, threshold=.8):
    target = predict(model, x).reshape(-1,10,7)
    out = np.array(action, copy=True)
    out[:,:10,:6] += blend*target[:,:,:6]
    if gripper:
        score = target[:,:,6]
        change = (np.abs(score) >= threshold) & ((score >= 0) != (out[:,:10,6] >= 0))
        out[:,:10,6] = np.where(change, np.where(score>=0,1.,-1.), out[:,:10,6])
    return out
