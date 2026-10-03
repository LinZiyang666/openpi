"""Paired / bootstrap helpers for the R11 analysis (episode-level pairing on test set A: (task, init))."""
from __future__ import annotations

import math

import numpy as np


def mcnemar_p(b, c):
    """Exact two-sided binomial McNemar p (normal approximation with continuity correction above 400)."""
    n = b + c
    if n == 0:
        return 1.0
    if n > 400:
        z = (abs(b - c) - 1) / math.sqrt(n)
        return math.erfc(max(z, 0) / math.sqrt(2))
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (float("nan"), float("nan"))
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi)))


def two_sided_p_boot(a):
    """Bootstrap two-sided p for 'statistic = 0' (share of replicates on the other side of 0, doubled)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan")
    return float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean())))


class CellBoot:
    """Stratified (by task) episode bootstrap shared by every arm of one cell-size, so all contrasts stay paired."""

    def __init__(self, keys, B, seed):
        self.keys = sorted(keys)
        self.pos = {k: i for i, k in enumerate(self.keys)}
        rng = np.random.default_rng(seed)
        tasks = np.array([k[0] for k in self.keys])
        cols = []
        for t in np.unique(tasks):
            I = np.flatnonzero(tasks == t)
            cols.append(I[rng.integers(0, len(I), size=(B, len(I)))])
        self.idx = np.concatenate(cols, axis=1)
        self.B = B
        self.arms = {}

    def add(self, name, ep, model=None, costs=None, ir_const=None):
        n = len(self.keys)
        s = np.zeros(n)
        m = np.zeros(n)
        cost = np.zeros(n)
        N = np.zeros(n)
        for e in ep:
            j = self.pos.get((int(e["task"]), int(e["init"])))
            if j is None:
                continue
            s[j], m[j] = e["success"], 1.0
            if costs is not None and e["N"] > 0:
                cv, cm = costs
                cost[j], N[j] = cv * e["V"] + cm * e["M"], e["N"]
        sr = s.sum() / m.sum()
        sr_b = (s[self.idx]).sum(1) / np.maximum(m[self.idx].sum(1), 1)
        if ir_const is not None:
            ir, ir_b = ir_const, np.full(self.B, ir_const)
        else:
            ir = cost.sum() / N.sum()
            ir_b = cost[self.idx].sum(1) / np.maximum(N[self.idx].sum(1), 1)
        self.arms[name] = dict(s=s, m=m, sr=sr, sr_b=sr_b, ir=ir, ir_b=ir_b)
        return self.arms[name]

    def add_pool(self, name, eps_list, costs):
        """Reference that averages several runs of the same configuration per episode (success share, summed cost
        and slots), to damp batch-to-batch noise of a single knob-off run."""
        n = len(self.keys)
        s = np.zeros(n)
        m = np.zeros(n)
        cost = np.zeros(n)
        N = np.zeros(n)
        cv, cm = costs
        for ep in eps_list:
            for e in ep:
                j = self.pos.get((int(e["task"]), int(e["init"])))
                if j is None:
                    continue
                s[j] += e["success"]
                m[j] += 1
                cost[j] += cv * e["V"] + cm * e["M"]
                N[j] += e["N"]
        sm = np.where(m > 0, s / np.maximum(m, 1), 0.0)
        w = (m > 0).astype(float)
        sr = sm.sum() / w.sum()
        sr_b = sm[self.idx].sum(1) / np.maximum(w[self.idx].sum(1), 1)
        ir = cost.sum() / N.sum()
        ir_b = cost[self.idx].sum(1) / np.maximum(N[self.idx].sum(1), 1)
        self.arms[name] = dict(s=sm, m=w, sr=sr, sr_b=sr_b, ir=ir, ir_b=ir_b)
        return self.arms[name]

    def paired(self, a, b):
        A, Bm = self.arms[a], self.arms[b]
        both = (A["m"] > 0) & (Bm["m"] > 0)
        bb = int(((A["s"] == 1) & (Bm["s"] == 0) & both).sum())
        cc = int(((A["s"] == 0) & (Bm["s"] == 1) & both).sum())
        return bb, cc, int(both.sum())


TOL = 0.01   # allowed linear extrapolation beyond a curve's end points, in IR units


def _interp(x, xs, ys, tol):
    if x < xs[0] - tol or x > xs[-1] + tol or len(xs) < 2:
        return float("nan")
    if x < xs[0]:
        return float(ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9))
    if x > xs[-1]:
        return float(ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9))
    return float(np.interp(x, xs, ys))


def interp_curve(x, xs, ys, tol=TOL):
    """Piecewise-linear SR at IR x along a method curve; linear extrapolation up to `tol` IR beyond the ends,
    NaN further out."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    o = np.argsort(xs)
    return _interp(x, xs[o], ys[o], tol)


def interp_curve_boot(xb, xsb, ysb, tol=TOL):
    """Replicate-wise interpolation: xb [B], xsb/ysb [P, B]."""
    B = len(xb)
    out = np.full(B, np.nan)
    for i in range(B):
        xs, ys = xsb[:, i], ysb[:, i]
        o = np.argsort(xs)
        out[i] = _interp(xb[i], xs[o], ys[o], tol)
    return out


def shapley(f, x0, x1):
    """Shapley attribution of f(x1) - f(x0) over the coordinates (exact, all orderings)."""
    import itertools
    n = len(x0)
    contrib = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for p in perms:
        x = list(x0)
        prev = f(*x)
        for i in p:
            x[i] = x1[i]
            cur = f(*x)
            contrib[i] += cur - prev
            prev = cur
    return contrib / len(perms)


def since_last(flag, ep):
    """Per anchor (rows sorted by episode then step): anchors since the last flagged anchor earlier in the same
    episode (1 = the previous anchor); a large number when none."""
    n = len(flag)
    pos = np.arange(n)
    start = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
    ep_start = np.repeat(start, np.diff(np.r_[start, n]))
    lastpos = np.where(flag, pos, -1)
    acc = np.maximum.accumulate(lastpos)
    prev = np.r_[-1, acc[:-1]]
    out = np.where(prev >= ep_start, pos - prev, 10 ** 6)
    return out


def runs_without(flag_call, ep):
    """Lengths of maximal stretches of consecutive non-call anchors, per episode: returns (episode ids, lengths)."""
    n = len(flag_call)
    eps, lens = [], []
    if not n:
        return np.zeros(0, int), np.zeros(0, int)
    cur, run = ep[0], 0
    for i in range(n):
        if ep[i] != cur:
            if run:
                eps.append(cur); lens.append(run)
            cur, run = ep[i], 0
        if flag_call[i]:
            if run:
                eps.append(cur); lens.append(run)
            run = 0
        else:
            run += 1
    if run:
        eps.append(cur); lens.append(run)
    return np.array(eps), np.array(lens)
