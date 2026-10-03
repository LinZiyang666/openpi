"""R10 offline analysis (opus): in-library corrector training schemes, k-fold-by-episode, library data only.

Data rule: every array is read from offline_search_store/library/<model>_<suite>/<bpool_cs|bpool_all>; nothing under
trace_runs/os_closed_loop (test set A) or any R8-derived table is opened (``assert_library_path`` refuses it).

What is replicated from serving (R4 BlindAWM = R2 AWM, the cache the R8 judge and the R9 corrector wrap):
  * features x = [PCA-64 key_v0, PCA-64 key_v1, rs[:8]] (joint), per-task metric ``awm.fit_metric`` (nn 3, lam .1),
    early metric (rows with step <= 2) for step-0 queries, top-16 by code distance (ties by row), kernel
    w = exp(-((d - d1) / (d_kref - d1))^2), served chunk = weighted mean of the members' library chunks.
    Regime 2 (no continuity term) for step >= 1, i.e. a fresh look after a cache HIT; regime 1 (after a policy
    MISS) is not modelled.
  * PCA basis: refitted on the (training-fold) subset with ``awm.pca_fit`` (the recipe AWM uses for a "current"
    library) for sizes < 500; the stored r01 bpool basis (same recipe on all 500 episodes) for size 500.
  * corrector head = fable round-2 / R9 recipe: inputs [PCA-128, rs8, served[:10,:7]/sigma, min(step,120)/120,
    task one-hot], standardized + clip 8, 384 cos RFF (seed 0), weighted ridge alpha 100, one head per task,
    target = motion residual (channels 0-5, 10 steps) / sigma, applied as served[:10,:6] += blend * corr.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w, fit_metric, pca_fit

STORE = Path("/home/weiland/trace_runs/offline_search_store")
FORBIDDEN = Path("/home/weiland/trace_runs/os_closed_loop")
PARENT = {"pi05": "bpool_cs", "groot": "bpool_all"}
SEED = 20261002            # sol's subset seed (rounds/r10/data.py): default_rng(SEED + task).permutation(episode ids)
K = 16
N_RFF = 384
ALPHA = 100.0
EARLY_MAX_STEP = 2
NFOLD = 5
H10 = 10
MOT = 6


def assert_library_path(p):
    p = Path(p).resolve()
    if p == FORBIDDEN or FORBIDDEN in p.parents or ("derived" in p.parts and "r08" in p.parts):
        raise ValueError(f"test-set / R8-derived path refused: {p}")
    if (STORE / "library") not in p.parents and (STORE / "derived" / "r01") not in p.parents:
        raise ValueError(f"only offline_search_store library (and its r01 PCA basis) may be read: {p}")
    return p


# ------------------------------------------------------------------------------------------------------- loading
def lib_dir(model, suite):
    return assert_library_path(STORE / "library" / f"{model}_{suite}" / PARENT[model])


def load_cell(model, suite):
    d = lib_dir(model, suite)
    act = np.load(d / "action.npy", mmap_mode="r")
    out = dict(model=model, suite=suite, dir=str(d),
               act=np.ascontiguousarray(act[:, :H10, :7], np.float32),
               rs=np.ascontiguousarray(np.load(d / "rs.npy")[:, :8], np.float64),
               task=np.load(d / "task_id.npy").astype(np.int64), ep=np.load(d / "episode.npy").astype(np.int64),
               step=np.load(d / "step.npy").astype(np.int64), ep_len=np.load(d / "ep_len.npy").astype(np.int64),
               success=np.load(d / "success.npy").astype(bool))
    return out


def permutations(C, seed=SEED):
    perms = {}
    for t in range(10):
        ids = np.unique(C["ep"][C["task"] == t])
        if len(ids) != 50:
            raise ValueError(f"task {t}: {len(ids)} episodes")
        perms[t] = np.random.default_rng(seed + t).permutation(ids)
    return perms


def subset_and_folds(C, size, seed=SEED):
    """Nested subset (first size/10 episodes of each task's permutation) and fold = position % 5."""
    perms = permutations(C, seed)
    n = size // 10
    fold_of_ep = {}
    for t, p in perms.items():
        for pos, e in enumerate(p[:n]):
            fold_of_ep[int(e)] = pos % NFOLD
    rows = np.flatnonzero(np.isin(C["ep"], np.array(sorted(fold_of_ep))))
    fold = np.array([fold_of_ep[int(e)] for e in C["ep"][rows]], np.int64)
    return rows, fold


def stored_pca(model, suite):
    """r01 basis of the 500-episode library (AWM's lib=big PCA): projections of every row (f32, [L, 64] x 2)."""
    base = STORE / "derived" / "r01" / "f4_vision" / "pca" / f"{model}_{suite}" / PARENT[model]
    out = []
    for f in ("v0", "v1"):
        p = assert_library_path(base / f / "proj.npy")
        out.append(np.ascontiguousarray(np.load(p, mmap_mode="r")[:, :64], np.float32))
    return out


def gather_keys(model, suite, rows):
    d = lib_dir(model, suite)
    out = []
    for f in ("v0", "v1"):
        X = np.load(d / f"key_{f}.npy", mmap_mode="r")
        out.append(np.asarray(X[np.sort(rows)], np.float32))
    return out


# --------------------------------------------------------------------------------------------------- cache metric
class TaskMetric:
    """AWM's per-task joint metric (main + early) fitted on one set of rows."""

    def __init__(self, X, heads, ep, step):
        self.mean, self.std, self.W = fit_metric(X, heads, ep, nn=3, lam=0.1, rank=0)
        m0 = step <= EARLY_MAX_STEP
        if m0.sum() < 2 or len(np.unique(ep[m0])) < 2:
            raise ValueError("too few early rows")
        self.mean0, self.std0, self.W0 = fit_metric(X[m0], heads[m0], ep[m0], nn=3, lam=0.1, rank=0)

    def code(self, X):
        return ((X - self.mean) / self.std) @ self.W

    def code0(self, X):
        return ((X - self.mean0) / self.std0) @ self.W0


def pdist(Q, R):
    d2 = (Q * Q).sum(1)[:, None] - 2 * Q @ R.T + (R * R).sum(1)[None]
    return np.sqrt(np.maximum(d2, 0.0))


def topk_kernel(D, kref, k=K):
    """Per row of D (inf = excluded): AWM's stable top-k (ties by column) and kernel weights (normalized)."""
    n = D.shape[1]
    kk = min(k, int(np.isfinite(D).sum(1).min()))
    if kk < 1:
        raise ValueError("no candidates")
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    Dk = np.take_along_axis(D, idx, 1)
    # stable order: by distance, ties by column index
    order = np.argsort(Dk + 0.0, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, 1)
    Dk = np.take_along_axis(Dk, order, 1)
    # break exact ties by column (argpartition does not keep column order)
    for r in np.flatnonzero((np.diff(Dk, axis=1) == 0).any(1)):
        o2 = np.lexsort((idx[r], Dk[r]))
        idx[r], Dk[r] = idx[r][o2], Dk[r][o2]
    w = _kernel_w(Dk - Dk[:, :1], kref)
    w = w / w.sum(1, keepdims=True)
    return idx, Dk, w


def serve(qX, qstep, qep, met, cX, cstep_unused, cep, cact, kref, exclude_own=True, cand_mask=None, chunk=1024,
          return_d=False):
    """Served chunks for queries qX against candidates cX of one task (main metric; early metric for step 0).

    Returns served [nq, 10, 7], d1 [nq] (main-metric distance to the nearest admissible candidate), d16 [nq]
    (main metric, 16th admissible neighbour), members [nq, 16] (candidate positions), weights [nq, 16]."""
    Zc, Zc0 = met.code(cX), met.code0(cX)
    Zq, Zq0 = met.code(qX), met.code0(qX)
    nq = len(qX)
    served = np.empty((nq, H10, 7), np.float32)
    d1 = np.empty(nq)
    d16 = np.empty(nq)
    mem = np.full((nq, K), -1, np.int64)
    wts = np.zeros((nq, K))
    for lo in range(0, nq, chunk):
        hi = min(nq, lo + chunk)
        D = pdist(Zq[lo:hi], Zc)
        bad = np.zeros_like(D, bool)
        if exclude_own:
            bad |= qep[lo:hi, None] == cep[None, :]
        if cand_mask is not None:
            bad |= ~cand_mask[None, :] if cand_mask.ndim == 1 else ~cand_mask[lo:hi]
        D[bad] = np.inf
        Ds = np.sort(D, axis=1)
        d1[lo:hi] = Ds[:, 0]
        d16[lo:hi] = Ds[:, min(K, D.shape[1]) - 1]
        s0 = qstep[lo:hi] == 0
        Duse = D.copy()
        if s0.any():
            D0 = pdist(Zq0[lo:hi][s0], Zc0)
            D0[bad[s0]] = np.inf
            Duse[s0] = D0
        idx, Dk, w = topk_kernel(Duse, kref)
        served[lo:hi] = np.einsum("qk,qktc->qtc", w, cact[idx]).astype(np.float32)
        mem[lo:hi, :idx.shape[1]] = idx
        wts[lo:hi, :idx.shape[1]] = w
    return served, d1, d16, mem, wts


# ------------------------------------------------------------------------------------------------- corrector head
def features(P, rs8, chunk, step, task, sig):
    n = len(P)
    cols = [P, rs8, (chunk[:, :H10, :7] / sig).reshape(n, -1), (np.minimum(step, 120) / 120.0)[:, None],
            np.eye(10)[task]]
    return np.concatenate(cols, 1).astype(np.float64)


class Head:
    """fable's ``fit_head`` (round2/tools/corrector.py) with chunked normal equations, so that very large pair sets fit
    in memory. ``total`` rescales the (normalized) weights so that sum(w) = total; fable's own choice is total = n
    (the number of training samples), which ties the effective ridge strength to the sample count."""

    def __init__(self, X, Y, weights, total=None, seed=0, n_rff=N_RFF, alpha=ALPHA, chunk=20000):
        rng = np.random.default_rng(seed)
        n, dx = X.shape
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        self.w = (rng.standard_normal((dx, n_rff)) / np.sqrt(dx)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, n_rff).astype(np.float32)
        wt = np.asarray(weights, np.float64)
        wt = wt / wt.sum() * (n if total is None else float(total))
        nf = dx + n_rff
        SFF = np.zeros((nf, nf))
        SFY = np.zeros((nf, Y.shape[1]))
        sF = np.zeros(nf)
        sY = np.zeros(Y.shape[1])
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            F = self.feats(X[lo:hi])
            ww = wt[lo:hi]
            Fw = F * ww[:, None]
            SFF += F.T @ Fw
            SFY += Fw.T @ Y[lo:hi]
            sF += Fw.sum(0)
            sY += ww @ Y[lo:hi]
        W = wt.sum()
        mu_f, mu_y = sF / W, sY / W
        A = SFF - W * np.outer(mu_f, mu_f) + alpha * np.eye(nf)
        B = SFY - W * np.outer(mu_f, mu_y)
        self.coef = np.linalg.solve(A, B).T
        self.intercept = mu_y - self.coef @ mu_f

    def feats(self, X):
        Xn = np.clip((X - self.mean) / self.std, -8, 8).astype(np.float32)
        return np.concatenate([Xn, np.cos(Xn @ self.w + self.b) * np.sqrt(2)], 1).astype(np.float64)

    def predict(self, X):
        return self.feats(X) @ self.coef.T + self.intercept


def residual(target, chunk, sig):
    return ((target[:, :H10, :MOT] - chunk[:, :H10, :MOT]) / sig[:MOT]).reshape(len(target), -1)


def motion_err(a, b, sig, steps=H10):
    return np.sqrt(np.mean(((a[:, :steps, :MOT] - b[:, :steps, :MOT]) / sig[:MOT]) ** 2, axis=(1, 2)))


def episode_weights(ep):
    _, inv, cnt = np.unique(ep, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def to_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")
