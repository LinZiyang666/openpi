"""NumPy-only student head; safe to import in either policy server environment."""
import numpy as np


def predict(model,x):
    xx=np.clip((x-model['mean'])/model['std'],-8,8)
    if model['w'] is not None:
        xx=np.c_[xx,np.cos(xx@model['w']+model['bias'])*np.sqrt(2)]
    return (xx@model['coef'].T+model['intercept']).reshape(len(x),10,6)
