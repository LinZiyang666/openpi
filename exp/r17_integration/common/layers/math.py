"""Frozen numerical equations ported from R2/R10, parameterized by arrays.
Sources: rounds/r02/g1_awm/awm.py, rounds/r10/train.py, rounds/r10/recipe/recipe.py.
No historical artifact paths or harness imports.
"""
import numpy as np
PDIM=64
N_RFF=384
ALPHA=100.

def pca_fit(X, keep=PDIM, K=128, over=32, power=3, seed=0, chunk=2048):
    """Centred randomized SVD of the rows of X (memmap [n, D]) with the r01 m7_pca_fit recipe (k=128, oversampling
    32, 3 power iterations, seed 0); returns mean f32 [D], basis f32 [D, keep] (leading PCs), proj f32 [n, keep]."""
    n, D = X.shape
    cuts = [(lo, min(n, lo + chunk)) for lo in range(0, n, chunk)]
    mu = np.zeros(D, np.float64)
    for lo, hi in cuts:
        mu += np.asarray(X[lo:hi], np.float64).sum(0)
    mu = (mu / n).astype(np.float32)

    def xmul(M):
        out = np.empty((n, M.shape[1]), np.float32)
        muM = mu @ M
        for lo, hi in cuts:
            out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ M - muM
        return out

    def xtmul(Y):
        out = np.zeros((D, Y.shape[1]), np.float32)
        for lo, hi in cuts:
            out += np.asarray(X[lo:hi], np.float32).T @ Y[lo:hi]
        out -= np.outer(mu, Y.sum(0))
        return out

    r = min(K + over, n)
    Om = np.random.default_rng(seed).standard_normal((D, r)).astype(np.float32)
    Q, _ = np.linalg.qr(xmul(Om))
    for _ in range(power):
        Z, _ = np.linalg.qr(xtmul(Q))
        Q, _ = np.linalg.qr(xmul(Z))
    _, _, Vt = np.linalg.svd(xtmul(Q).T, full_matrices=False)
    V = np.ascontiguousarray(Vt[:keep].T.astype(np.float32))
    return mu, V, xmul(V)


def fit_metric(X, H, ep, nn=3, lam=0.1, rank=0):
    """Ideation A's fit_transform (p3_white.py), float64. X [n, d] features, H [n, 35] sigma-scaled heads, ep [n]
    episode ids. Returns (mean, std, W) with code = ((x - mean) / std) @ W."""
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xn = (X - mean) / std
    h2 = (H * H).sum(1)
    DH = h2[:, None] - 2 * H @ H.T + h2[None, :]
    DH[ep[:, None] == ep[None, :]] = np.inf
    k = min(nn, DH.shape[1] - 1)
    nn_idx = np.argpartition(DH, k, axis=1)[:, :k]
    okm = np.isfinite(np.take_along_axis(DH, nn_idx, 1))
    diffs = (Xn[:, None, :] - Xn[nn_idx])[okm]
    Sw = diffs.T @ diffs / max(len(diffs), 1)
    d = Sw.shape[0]
    scatter = np.trace(Sw)
    if not d or not np.isfinite(scatter) or scatter <= 0:
        raise ValueError("zero-scatter library metric cannot be fitted; provide distinct finite cross-episode state/key rows")
    Minv = np.linalg.inv(Sw + lam * np.trace(Sw) / d * np.eye(d))
    A = np.linalg.cholesky(Minv)
    if not rank:
        return mean, std, A
    St = np.cov(Xn.T) + 1e-6 * np.eye(d)
    Ms = A.T @ St @ A
    ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
    return mean, std, A @ V[:, np.argsort(-ev)[:rank]]


def _kernel_w(rel, kref):
    """rel [.., k] = d~ - d~_1 (sorted ascending); w = exp(-(rel / rel[kref-1])^2)."""
    k = rel.shape[-1]
    ref = np.maximum(rel[..., min(kref, k) - 1:min(kref, k)], 1e-6)
    return np.exp(-(rel / ref) ** 2)


def anchor_weights(rows, episodes):
    """Equal episode mass; an anchor divides its row weight among its pairs.

    Total mass is the number of represented anchors, never the pair count.
    Radius-empty anchors have no samples and are excluded from that count.
    """
    anchors, first, inverse, counts = np.unique(rows, return_index=True,
                                               return_inverse=True, return_counts=True)
    anchor_eps = episodes[first]
    eps, ep_inverse, ep_counts = np.unique(anchor_eps, return_inverse=True, return_counts=True)
    per_anchor = len(anchors) / (len(eps) * ep_counts[ep_inverse])
    weights = per_anchor[inverse] / counts[inverse]
    assert np.isclose(weights.sum(), len(anchors))
    return weights


def fit_head(X, Y, weights, seed=0, batch=4096):
    """R9's standardized-input + 384 RFF weighted ridge, alpha 100.

    Accumulate sufficient statistics in float64 batches so pair training is
    bounded in memory. The equation/intercept match R9 fit_head, preserving
    the supplied anchor mass so PAIR multiplicity cannot weaken alpha.
    """
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    W = (rng.standard_normal((X.shape[1], N_RFF)) / np.sqrt(X.shape[1])).astype(np.float32)
    bias = rng.uniform(0, 2 * np.pi, N_RFF).astype(np.float32)
    dim = X.shape[1] + N_RFF
    A, B = np.zeros((dim, dim)), np.zeros((dim, Y.shape[1]))
    sf, sy, total = np.zeros(dim), np.zeros(Y.shape[1]), 0.
    weights = np.asarray(weights, np.float64)
    assert weights.shape == (len(X),) and np.all(np.isfinite(weights)) and np.all(weights > 0)
    for lo in range(0, len(X), batch):
        hi = min(len(X), lo + batch)
        xn = np.clip((X[lo:hi] - mean) / std, -8, 8)
        F = np.concatenate([xn, np.cos(xn @ W + bias) * np.sqrt(2)], 1).astype(np.float64)
        w = weights[lo:hi]
        yf = Y[lo:hi].astype(np.float64)
        A += F.T @ (F * w[:, None])
        B += F.T @ (yf * w[:, None])
        sf += F.T @ w; sy += yf.T @ w; total += w.sum()
    A -= np.outer(sf, sf) / total
    B -= np.outer(sf, sy) / total
    A.flat[::dim + 1] += ALPHA
    coef = np.linalg.solve(A, B).T
    intercept = sy / total - coef @ (sf / total)
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), w=W, bias=bias,
                coef=coef.astype(np.float32), intercept=intercept.astype(np.float32))


def predict(head, X):
    # Frozen R9 equation, also used by R10 Stage 2b; no exploration imports.
    Xn = np.clip((X - head['mean']) / head['std'], -8, 8)
    F = np.concatenate([Xn, np.cos(Xn @ head['w'] + head['bias']) * np.sqrt(2)], 1)
    return F @ head['coef'].T + head['intercept']

