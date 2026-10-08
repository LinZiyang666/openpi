"""Frozen R12 arithmetic. Serving has no exploration imports or fit-file reads."""
from __future__ import annotations

import numpy as np
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r10.recipe.recipe import predict


def freeze_fit(value, seen=None):
    """Restore fitted-array immutability after numpy pickle deserialization."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        return value.nbytes
    if isinstance(value, dict):
        return sum(freeze_fit(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(freeze_fit(v, seen) for v in value)
    if isinstance(value, Metric):
        return freeze_fit(vars(value), seen)
    return 0


class Metric:
    """Astra float32 affine metric or Opus float64 standardized metric."""
    def __init__(self, X, actions, ep, sigma, *, precision='astra', early=False, step=None):
        from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
        X = np.asarray(X, np.float64)
        self.precision = precision
        heads = (actions[:, :5, :7] / sigma).reshape(len(X), 35).astype(np.float64)
        self.mean, self.std, self.transform = fit_metric(X, heads, ep, nn=3, lam=.1, rank=0)
        self.W = (self.transform / self.std[:, None]).astype(np.float32)
        self.shift = ((self.mean / self.std) @ self.transform).astype(np.float32)
        self.Z = self.code(X)
        self.z2 = (self.Z * self.Z).sum(1)
        self.early = None
        if early:
            keep = np.asarray(step) <= 2
            self.early = Metric(X[keep], actions[keep], ep[keep], sigma, precision=precision)
            # Early metric must search ALL current rows, not just early rows.
            self.early.Z = self.early.code(X)
            self.early.z2 = (self.early.Z * self.early.Z).sum(1)

    def code(self, X):
        if self.precision == 'opus':
            return ((np.asarray(X, np.float64) - self.mean) / self.std) @ self.transform
        # Candidate codes use the unrounded fitting transform (explorer path).
        return (((np.asarray(X, np.float64)-self.mean)/self.std) @ self.transform).astype(np.float32)

    def dist(self, X, *, initial=False):
        m = self.early if initial and self.early is not None else self
        if m.precision == 'opus':
            z = m.code(X)
        else:
            z = np.asarray(X, np.float32) @ m.W - m.shift
        return np.sqrt(np.maximum((z*z).sum(1)[:, None] + m.z2[None] - 2*z @ m.Z.T, 0))


def features(X, action, step, task, sigma):
    X = np.atleast_2d(X)
    action = np.asarray(action)[:, :10, :7]
    return np.c_[X, (action/sigma).reshape(len(X), 70),
                 np.minimum(np.broadcast_to(step, len(X)), 120)/120,
                 np.eye(10)[np.broadcast_to(task, len(X))]].astype(np.float32)


def retrieve(metric, rows, X, actions, kref, *, initial=False, distances=None):
    d = metric.dist(np.atleast_2d(X), initial=initial)[0] if distances is None else distances
    ix = np.argsort(d, kind='stable')[:16]
    dk = d[ix].astype(np.float64)
    w = _kernel_w(dk-dk[0], kref)
    w = (w/w.sum()).astype(np.float32)
    chosen = rows[ix]
    raw = np.tensordot(w, actions[chosen], 1).astype(np.float32)
    ref = max(float(dk[min(kref, len(dk))-1]-dk[0]), 1e-6)
    scores = -((dk-dk[0])/ref)**2
    all_scores = -((d.astype(np.float64)-dk[0])/ref)**2
    return dict(rows=chosen, weights=w, distance=float(dk[0]), raw=raw,
                scores=scores, all_scores=all_scores, all_distances=d)


def correct(proposal, head, scale, X, step, task, sigma):
    raw = proposal['raw']
    residual = predict(head, features(X, raw[None], step, task, sigma))[0].reshape(10, 6)*sigma[:6]
    factor = np.float32(np.clip((2-proposal['distance']/scale)/1.25, 0, 1))
    action = raw.copy()
    action[:10, :6] += np.float32(.5)*(residual*factor)
    return action, float(factor)


def balanced_median(x, ep):
    _, inv, count = np.unique(ep, return_inverse=True, return_counts=True)
    w = 1./count[inv]
    order = np.argsort(x, kind='stable')
    return float(x[order[np.searchsorted(np.cumsum(w[order]), w.sum()/2)]])


def atlas_plan(*, step, memory, rs, table, lifecycle, due, span):
    if step < 4 or memory is None or step-memory['step'] != 2 or span > 0 or due or not lifecycle:
        return dict(half=False, reason='lifecycle_or_due')
    rows, w = memory['rows'], memory['weights'].copy()
    w /= w.sum()
    nxt = table['next2'][rows]
    ok, idx = nxt >= 0, np.maximum(nxt, 0)
    support = float((w*ok).sum())
    mass = float((w*ok*table['safe'][idx]).sum())
    expected = np.einsum('k,kd->d', w*ok/max(support, 1e-9), table['rs'][idx]-table['rs'][rows])
    error = float(np.sqrt(np.mean(((rs-memory['rs']-expected)/table['state_scale'])**2)))
    return dict(half=bool(support >= .99 and mass >= .75 and error <= .2),
                reason='atlas_or_state', support=support, safe_mass=mass, state_error=error)
