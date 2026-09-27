"""G4 (R2) shared machinery: AWM features / PCA / whitening fit, T2 pair building, kernel synthesis.

The AWM parts follow ideation A-P1 exactly (rounds/r02/NOTES_ideation_A.md, diag_A/p3_white.py):
  features x = [PCA-64 of key_v0, PCA-64 of key_v1, rs[:8]] (136-d), z-scored per task on the fit rows;
  heads h = action[:5, :7] / sigma_d (35-d);  per task: for every fit row its `nn` nearest rows from OTHER episodes
  in head space -> S_w = mean(diff diff^T) of the feature differences; M = (S_w + lam*tr(S_w)/d*I)^-1;
  W = chol(M) (x' = x_n @ W); rank-r version = the top-r generalized-eigen directions (LDA-like) of A^T S_t A.
  Kernel synthesis: top-16 by score, w_i = exp(-((d_i - d_1)/(d_kref - d_1))^2), kref = 5.

Everything numeric runs on the valid dims only (harness/dims.py): actions [:5, :7], rs[:8].
Heavy, input-only artefacts (current-library PCA, projections, trained T2 weights) are cached under
DERIVED (flock-protected, atomic writes) so every job / server of a batch reuses one canonical copy.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import time

import numpy as np

from exp.offline_search.harness import dims

DERIVED = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g4_t2")
PCA_R01 = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
PDIM = 64                                   # PCA dims per camera (AWM default)
NH = dims.EXEC_STEPS * dims.ACT_DIMS        # 35 = executed head, valid dims
FEAT_DIM = 2 * PDIM + 8                     # 136
CHUNK = 2048


# ------------------------------------------------------------------------------------------ io helpers
@contextlib.contextmanager
def file_lock(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def atomic_save(path: pathlib.Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def file_sig(p: pathlib.Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}"


# ------------------------------------------------------------------------------------------------ PCA
def project(X, mu: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(X - mu) @ B for a [n, 32768] (memmap) array, f32, chunked; same formula as the r01 proj.npy."""
    out = np.empty((X.shape[0], B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, X.shape[0], CHUNK):
        hi = min(X.shape[0], lo + CHUNK)
        out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ B - muB
    return out


def _fit_pca_gram(X, k: int):
    """Exact PCA of a small library (n <= ~3k rows) through the n x n Gram matrix. X [n, D] memmap."""
    n, D = X.shape
    mu = np.zeros(D, np.float64)
    for lo in range(0, n, CHUNK):
        mu += np.asarray(X[lo:lo + CHUNK], np.float64).sum(0)
    mu = (mu / n).astype(np.float32)
    Xc = np.asarray(X, np.float32) - mu
    G = (Xc @ Xc.T).astype(np.float64)
    ev, U = np.linalg.eigh((G + G.T) / 2)
    idx = np.argsort(-ev)[:k]
    ev, U = np.maximum(ev[idx], 1e-12), U[:, idx]
    V = (Xc.T @ U.astype(np.float32)) / np.sqrt(ev).astype(np.float32)[None]   # [D, k] unit columns
    V /= np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12)
    sgn = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(k)])                # deterministic sign
    V = np.ascontiguousarray(V * sgn[None], np.float32)
    tot = float((Xc.astype(np.float64) ** 2).sum() / n)
    return mu, V, (ev / n), tot


def current_pca_dir(ms: str, field: str) -> pathlib.Path:
    return DERIVED / "pca" / ms / "current" / field


def ensure_current_pca(root, ms: str) -> None:
    """PCA-64 per camera fitted on library current (task-agnostic, all rows); cached."""
    for f in ("v0", "v1"):
        d = current_pca_dir(ms, f)
        if (d / "meta.json").exists():
            continue
        with file_lock(DERIVED / "locks" / f"pca_{ms}_{f}.lock"):
            if (d / "meta.json").exists():
                continue
            t0 = time.time()
            src = pathlib.Path(root) / "library" / ms / "current" / f"key_{f}.npy"
            X = np.load(src, mmap_mode="r")
            mu, V, eig, tot = _fit_pca_gram(X, PDIM)
            atomic_save(d / "mean.npy", mu)
            atomic_save(d / "basis.npy", V)
            atomic_save(d / "proj.npy", project(X, mu, V))
            atomic_json(d / "meta.json", {"ms": ms, "lib": "current", "field": f, "n": int(X.shape[0]), "k": PDIM,
                                          "explained": float(eig.sum() / tot), "src": file_sig(src),
                                          "method": "exact Gram eigendecomposition", "fit_s": time.time() - t0})


def pca_basis(root, ms: str, basis: str):
    """{field: (mu f32[32768], B f32[32768, 64])} for basis in {"big" (r01, fitted on the 10x library), "current"}."""
    out, meta = {}, {}
    model = ms.split("_")[0]
    if basis == "current":
        ensure_current_pca(root, ms)
    for f in ("v0", "v1"):
        d = (PCA_R01 / ms / BIG[model] / f) if basis == "big" else current_pca_dir(ms, f)
        mu = np.load(d / "mean.npy").astype(np.float32)
        B = np.ascontiguousarray(np.load(d / "basis.npy", mmap_mode="r")[:, :PDIM], np.float32)
        out[f] = (mu, B)
        m = json.loads((d / "meta.json").read_text())
        meta[f] = {"n": m.get("n"), "dir": str(d)}
    return out, meta


def lib_proj(root, ms: str, libname: str, basis: str) -> dict:
    """{field: f32[L, 64]} projections of a library's pooled keys onto a basis (cached)."""
    model = ms.split("_")[0]
    libdir = pathlib.Path(root) / "library" / ms / libname
    L = int(np.load(libdir / "task_id.npy", mmap_mode="r").shape[0])
    out = {}
    for f in ("v0", "v1"):
        if basis == "big" and libname == BIG[model]:
            d = PCA_R01 / ms / libname / f
            meta = json.loads((d / "meta.json").read_text())
            if meta["n"] != L:
                raise RuntimeError(f"r01 PCA cache {d} fitted on {meta['n']} rows, library has {L}")
            out[f] = np.ascontiguousarray(np.load(d / "proj.npy", mmap_mode="r")[:, :PDIM], np.float32)
            continue
        if basis == "current" and libname == "current":
            ensure_current_pca(root, ms)
            out[f] = np.ascontiguousarray(np.load(current_pca_dir(ms, f) / "proj.npy"), np.float32)
            continue
        src = libdir / f"key_{f}.npy"
        p = DERIVED / "proj" / ms / f"{libname}__b{basis}_{f}.npy"
        pm = p.with_suffix(".json")
        with file_lock(DERIVED / "locks" / f"proj_{ms}_{libname}_{basis}_{f}.lock"):
            if not (p.exists() and pm.exists() and json.loads(pm.read_text()).get("src") == file_sig(src)):
                bs, _ = pca_basis(root, ms, basis)
                mu, B = bs[f]
                atomic_save(p, project(np.load(src, mmap_mode="r"), mu, B))
                atomic_json(pm, {"src": file_sig(src), "basis": basis, "L": L})
        out[f] = np.ascontiguousarray(np.load(p), np.float32)
    return out


# ---------------------------------------------------------------------------------------------- AWM fit
def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.einsum("ij,ij->i", A, A)
    b2 = np.einsum("ij,ij->i", B, B)
    return np.maximum(a2[:, None] - 2.0 * (A @ B.T) + b2[None, :], 0.0)


def awm_fit(X: np.ndarray, H: np.ndarray, ep: np.ndarray, nn: int = 3, lam: float = 0.1):
    """X [n, d] f64 raw features, H [n, 35] sigma-scaled heads, ep [n] episode ids.
    Returns (mean, std, W [d, d], Minv [d, d]); codes = ((x - mean) / std) @ W (ideation p3 fit_transform)."""
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xn = (X - mean) / std
    DH = sqdist(H, H)
    DH[ep[:, None] == ep[None, :]] = np.inf
    k = min(nn, DH.shape[1] - 1)
    nn_idx = np.argpartition(DH, k, axis=1)[:, :k]
    okm = np.isfinite(np.take_along_axis(DH, nn_idx, 1))
    diffs = (Xn[:, None, :] - Xn[nn_idx])[okm]
    Sw = diffs.T @ diffs / max(len(diffs), 1)
    d = Sw.shape[0]
    Minv = np.linalg.inv(Sw + lam * np.trace(Sw) / d * np.eye(d))
    W = np.linalg.cholesky(Minv)
    return mean, std, W, Minv


def rank_w(Xn: np.ndarray, Minv: np.ndarray, rank: int) -> np.ndarray:
    """Top-`rank` directions of the whitened total covariance (ideation p3 `rank` variant): W_r = A @ V[:, top]."""
    d = Minv.shape[0]
    A = np.linalg.cholesky(Minv)
    St = np.cov(Xn.T) + 1e-6 * np.eye(d)
    Ms = A.T @ St @ A
    ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
    idx = np.argsort(-ev)[:rank]
    return A @ V[:, idx]


# ------------------------------------------------------------------------------------------ synthesis
def topk_kernel(s: np.ndarray, k: int = 16, kref: int = 5):
    """s [C] scores (higher = better). Returns (o [k] positions best first, ties by position; w [k] kernel weights)."""
    C = s.shape[0]
    k = min(k, C)
    part = np.argpartition(-s, k - 1)[:k] if k < C else np.arange(C)
    o = part[np.lexsort((part, -s[part]))]
    dd = -s[o].astype(np.float64)
    dd = dd - dd[0]
    ref = max(float(dd[min(kref, k) - 1]), 1e-6)
    w = np.exp(-(dd / ref) ** 2)
    return o, w


def topk_kernel_batch(S: np.ndarray, k: int = 16, kref: int = 5):
    """Row-wise topk_kernel for S [n, C] (diagnostics; same ordering rule)."""
    n, C = S.shape
    k = min(k, C)
    part = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < C else np.tile(np.arange(C), (n, 1))
    ps = np.take_along_axis(S, part, 1)
    # sort by (-score, position): lexsort per row via a composite key
    order = np.lexsort((part, -ps), axis=1)
    o = np.take_along_axis(part, order, 1)
    dd = -np.take_along_axis(S, o, 1).astype(np.float64)
    dd = dd - dd[:, :1]
    ref = np.maximum(dd[:, min(kref, k) - 1:min(kref, k)], 1e-6)
    w = np.exp(-(dd / ref) ** 2)
    return o, w


# ------------------------------------------------------------------------------------------- T2 pairs
def build_pairs(Xn: np.ndarray, Wl0: np.ndarray, H: np.ndarray, ep: np.ndarray, n_nn: int, n_rand: int, seed: int,
                n_act: int = 0):
    """Per task: each row with its n_nn nearest rows (other episodes) in the initial metric Xn @ Wl0, n_rand random
    other-episode rows and (optionally) its n_act nearest other-episode rows in head (action) space. Returns local
    (i, j), head-RMS targets r (sigma units), the initial distances d0 and kind (0 metric-NN, 1 random, 2 action-NN)."""
    n = Xn.shape[0]
    Z = Xn @ Wl0
    same = ep[:, None] == ep[None, :]
    ii, jj, kk = [], [], []

    def knn(D, k, code):
        D = D.copy()
        D[same] = np.inf
        k = min(k, n - 1)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < n else np.argsort(D, axis=1)[:, :k]
        ok = np.isfinite(np.take_along_axis(D, idx, 1)).ravel()
        ii.append(np.repeat(np.arange(n), k)[ok])
        jj.append(idx.ravel()[ok])
        kk.append(np.full(int(ok.sum()), code, np.int8))

    if n_nn > 0:
        knn(sqdist(Z, Z), n_nn, 0)
    if n_act > 0:
        knn(sqdist(H, H), n_act, 2)
    rng = np.random.default_rng(seed)
    if n_rand > 0:
        u = rng.integers(0, n, size=(n, 4 * n_rand))
        val = ep[u] != ep[:, None]
        for i in range(n):
            c = u[i][val[i]][:n_rand]
            ii.append(np.full(c.shape[0], i))
            jj.append(c)
            kk.append(np.full(c.shape[0], 1, np.int8))
    i = np.concatenate(ii).astype(np.int64)
    j = np.concatenate(jj).astype(np.int64)
    r = np.sqrt(np.mean((H[i] - H[j]) ** 2, axis=1))
    d0 = np.sqrt(np.sum((Z[i] - Z[j]) ** 2, axis=1))
    return i, j, r.astype(np.float32), d0, np.concatenate(kk)


def lsq_scale(d0: np.ndarray, r: np.ndarray, intercept: bool):
    """r ~ alpha * d0 (+ b): least squares; alpha > 0, b >= 0 (falls back to no intercept otherwise)."""
    d0 = np.asarray(d0, np.float64)
    r = np.asarray(r, np.float64)
    if intercept:
        dc, rc = d0 - d0.mean(), r - r.mean()
        alpha = float((dc * rc).sum() / max((dc * dc).sum(), 1e-12))
        b = float(r.mean() - alpha * d0.mean())
        if alpha > 0 and b >= 0:
            return alpha, b
    return float((d0 * r).sum() / max((d0 * d0).sum(), 1e-12)), 0.0


def fingerprint(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
