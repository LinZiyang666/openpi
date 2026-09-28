"""OAS spherical shrinkage, NumPy only; pair dependence violates its IID model.

Formula: alpha=mean(S**2), mu=trace(S)/p,
gamma=min(1,(alpha+mu**2)/((n+1)*(alpha-mu**2/p))).
It is a numerical regularizer here, not an optimality claim under temporal data.
"""
import numpy as np

def oas(X, assume_centered=True):
    if not assume_centered:
        X = X-X.mean(0)
    n,p=X.shape
    S=X.T@X/n
    mu=np.trace(S)/p
    alpha=np.mean(S*S)
    den=(n+1)*(alpha-mu*mu/p)
    gamma=1. if den<=0 else min(1.,(alpha+mu*mu)/den)
    return (1-gamma)*S+gamma*mu*np.eye(p),gamma
