"""M7 precompute (T1): task-agnostic PCA per vision field on the BIG library of each model x suite
(pi0.5: bpool_cs, GR00T: bpool_all). Centered randomized SVD (k=128, oversampling 32, 3 power iterations) over all
rows of the library's pooled 32768-d keys. Run once; ONE process with 16 BLAS threads inside the F4 CPU allotment:

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 18-25,62-69 \
        .venv/bin/python exp/offline_search/rounds/r01/f4_vision/m7_pca_fit.py

Writes <DERIVED>/pca/<m>_<s>/<lib>/<field>/{basis.npy [32768,128] f32 (columns = PCs, descending), mean.npy [32768]
f32, proj.npy [L,128] f32 (centered library keys projected), eigval.npy [128] (variance per PC), meta.json}.
The method (m7_cascade.py) loads these; the leading k columns give PCA-k for k in {32, 64, 128}.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")
DERIVED = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
K, OVER, POWER, CHUNK = 128, 32, 3, 2048


def _rowchunks(n):
    return [(lo, min(n, lo + CHUNK)) for lo in range(0, n, CHUNK)]


def _xmul(X, mu, M):
    """(X - mu) @ M, X [n, D] memmap, M [D, r] -> [n, r] float64-accumulated in float32 chunks."""
    out = np.empty((X.shape[0], M.shape[1]), np.float32)
    muM = mu @ M
    for lo, hi in _rowchunks(X.shape[0]):
        out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ M - muM
    return out


def _xtmul(X, mu, Y):
    """(X - mu)^T @ Y, Y [n, r] -> [D, r]."""
    out = np.zeros((X.shape[1], Y.shape[1]), np.float32)
    for lo, hi in _rowchunks(X.shape[0]):
        out += np.asarray(X[lo:hi], np.float32).T @ Y[lo:hi]
    out -= np.outer(mu, Y.sum(0))
    return out


def fit_one(key, field):
    m = key.split("_")[0]
    lib = BIG[m]
    X = np.load(ROOT / "library" / key / lib / f"key_{field}.npy", mmap_mode="r")
    n, D = X.shape
    t0 = time.time()
    mu = np.zeros(D, np.float64)
    for lo, hi in _rowchunks(n):
        mu += np.asarray(X[lo:hi], np.float64).sum(0)
    mu = (mu / n).astype(np.float32)
    rng = np.random.default_rng(0)
    Om = rng.standard_normal((D, K + OVER)).astype(np.float32)
    Y = _xmul(X, mu, Om)
    Q, _ = np.linalg.qr(Y)
    for _ in range(POWER):
        Z = _xtmul(X, mu, Q)
        Z, _ = np.linalg.qr(Z)
        Y = _xmul(X, mu, Z)
        Q, _ = np.linalg.qr(Y)
    B = _xtmul(X, mu, Q).T            # [r, D] = Q^T (X - mu)
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = np.ascontiguousarray(Vt[:K].T.astype(np.float32))   # [D, K]
    proj = _xmul(X, mu, V)                                  # [n, K]
    eig = (S[:K] ** 2 / n).astype(np.float64)
    tot = 0.0
    for lo, hi in _rowchunks(n):
        xc = np.asarray(X[lo:hi], np.float64) - mu
        tot += float((xc * xc).sum())
    tot /= n
    d = DERIVED / "pca" / key / lib / field
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "basis.npy", V)
    np.save(d / "mean.npy", mu)
    np.save(d / "proj.npy", proj.astype(np.float32))
    np.save(d / "eigval.npy", eig)
    meta = {"key": key, "lib": lib, "field": field, "n": int(n), "D": int(D), "k": K, "oversample": OVER,
            "power_iter": POWER, "seed": 0, "total_var": tot,
            "explained": {str(k): float(eig[:k].sum() / tot) for k in (8, 16, 32, 64, 128)},
            "participation_ratio_topk": float(eig.sum() ** 2 / (eig ** 2).sum()), "fit_s": time.time() - t0}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(key, field, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()
                                  if k in ("n", "explained", "participation_ratio_topk", "fit_s")}), flush=True)


def main():
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
    for key in keys:
        for field in ("v1", "v0"):
            fit_one(key, field)
    (DERIVED / "pca" / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
