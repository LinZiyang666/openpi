"""G2 (R2 cosine family) precompute: PCA bases of the pooled vision keys + library codes, per model x suite.

Two fit sources (protocol §9 effect decomposition, `fit_data`):
  fit "big"  basis fitted on the 10x library (pi0.5 bpool_cs / GR00T bpool_all): the r01 F4 randomized-SVD bases
             (derived/r01/f4_vision/pca/<key>/<big>/<f>/{mean,basis,proj}.npy, k=128) are COPIED here so the family
             does not depend on another family's files at run time. Codes of the big library = the r01 proj.npy;
             codes of the current library are computed here in the same basis.
  fit "cur"  basis fitted on the CURRENT library only (<= 2,645 rows): exact centred PCA via the float64 Gram matrix
             (eigh), top 128 components, column sign fixed (largest |loading| positive). Codes of the current library.

Output (arrays: data, not in the repo):
  DERIVED/pca/<key>/fit_<big|cur>/<f>/{mean.npy f32[D], basis.npy f32[D,128] (columns = PCs, descending), meta.json}
  DERIVED/proj/<key>/fit_<big|cur>/<lib>_<f>.npy   f32[L,128]  = key @ basis - mean @ basis  (row chunks of 2048)
  DERIVED/proj/<key>/fit_<big|cur>/<lib>_meta.json L, source key file size / mtime (checked by the method's fit)
One process, BLAS threads = the CPU range (one-off):
  OMP_NUM_THREADS=18 OPENBLAS_NUM_THREADS=18 taskset -c 9-17,53-61 .venv/bin/python \
      exp/offline_search/rounds/r02/g2_pcacos/precompute.py [keys]
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time

import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")
R01_PCA = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
DERIVED = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
KMAX = 128
CHUNK = 2048
FIELDS = ("v0", "v1")


def log(*a):
    print(*a, flush=True)


def project(X, mu, B):
    """key @ B - mu @ B in float32 row chunks."""
    out = np.empty((X.shape[0], B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, X.shape[0], CHUNK):
        out[lo:lo + CHUNK] = np.asarray(X[lo:lo + CHUNK], np.float32) @ B - muB
    return out


def src_meta(key, lib, f):
    p = ROOT / "library" / key / lib / f"key_{f}.npy"
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime": float(st.st_mtime)}


def save_proj(key, fit, lib, f, P):
    d = DERIVED / "proj" / key / f"fit_{fit}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{lib}_{f}.tmp.npy"
    np.save(tmp, np.ascontiguousarray(P, np.float32))
    tmp.replace(d / f"{lib}_{f}.npy")
    m = {"L": int(P.shape[0]), "k": int(P.shape[1]), "src": src_meta(key, lib, f)}
    (d / f"{lib}_{f}_meta.json").write_text(json.dumps(m, indent=1))


def fit_big(key):
    model = key.split("_")[0]
    big = BIG[model]
    for f in FIELDS:
        t0 = time.time()
        sd = R01_PCA / key / big / f
        dd = DERIVED / "pca" / key / "fit_big" / f
        dd.mkdir(parents=True, exist_ok=True)
        mu = np.load(sd / "mean.npy").astype(np.float32)
        B = np.ascontiguousarray(np.load(sd / "basis.npy").astype(np.float32))
        np.save(dd / "mean.npy", mu)
        np.save(dd / "basis.npy", B)
        meta = json.loads((sd / "meta.json").read_text())
        meta.update({"copied_from": str(sd), "fit_lib": big})
        (dd / "meta.json").write_text(json.dumps(meta, indent=1))
        # big library codes: the r01 projection (same basis, same formula)
        P = np.load(sd / "proj.npy").astype(np.float32)
        Lb = np.load(ROOT / "library" / key / big / "task_id.npy").shape[0]
        assert P.shape == (Lb, KMAX), (P.shape, Lb)
        save_proj(key, "big", big, f, P)
        # current library codes in the big basis
        X = np.load(ROOT / "library" / key / "current" / f"key_{f}.npy", mmap_mode="r")
        save_proj(key, "big", "current", f, project(X, mu, B))
        log(f"[{key}] fit_big {f}: copied basis, big L={Lb}, current L={X.shape[0]} projected ({time.time() - t0:.1f}s)")


def fit_cur(key):
    for f in FIELDS:
        t0 = time.time()
        X = np.asarray(np.load(ROOT / "library" / key / "current" / f"key_{f}.npy", mmap_mode="r"), np.float32)
        n, D = X.shape
        mu64 = X.astype(np.float64).mean(0)
        mu = mu64.astype(np.float32)
        Xc = (X.astype(np.float64) - mu64)
        G = Xc @ Xc.T                                        # [n, n] float64
        ev, U = np.linalg.eigh(G)
        idx = np.argsort(ev)[::-1][:KMAX]
        ev, U = ev[idx], U[:, idx]
        V = (Xc.T @ U) / np.sqrt(np.maximum(ev, 1e-30))       # [D, K] orthonormal columns
        sgn = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
        sgn[sgn == 0] = 1.0
        V = np.ascontiguousarray((V * sgn).astype(np.float32))
        tot = float((Xc * Xc).sum() / n)
        eig = ev / n
        dd = DERIVED / "pca" / key / "fit_cur" / f
        dd.mkdir(parents=True, exist_ok=True)
        np.save(dd / "mean.npy", mu)
        np.save(dd / "basis.npy", V)
        meta = {"key": key, "fit_lib": "current", "field": f, "n": int(n), "D": int(D), "k": KMAX,
                "method": "exact centred PCA via float64 Gram eigh", "total_var": tot,
                "explained": {str(k): float(eig[:k].sum() / tot) for k in (8, 16, 32, 64, 128)},
                "src": src_meta(key, "current", f), "fit_s": time.time() - t0}
        (dd / "meta.json").write_text(json.dumps(meta, indent=1))
        save_proj(key, "cur", "current", f, project(X, mu, V))
        log(f"[{key}] fit_cur {f}: n={n} explained32={meta['explained']['32']:.3f} "
            f"explained64={meta['explained']['64']:.3f} ({time.time() - t0:.1f}s)")


def main():
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
    for key in keys:
        fit_big(key)
        fit_cur(key)
    (DERIVED / "DONE_precompute").write_text(json.dumps({"keys": keys, "t": time.time()}) + "\n")


if __name__ == "__main__":
    main()
