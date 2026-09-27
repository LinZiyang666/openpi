"""One-off precompute for the g4_t2 family (ONE process, BLAS threads = the agent's CPU range):
  1. PCA-64 per camera on library current (exact Gram PCA)       -> DERIVED/pca/<ms>/current/<f>/
  2. current-library keys projected on the big (r01) basis         -> DERIVED/proj/<ms>/current__bbig_<f>.npy
  3. (--queries) query keys of every cell on both bases (diag only) -> DERIVED/qproj/<cell>__b{big,current}_<f>.npy

    OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 MKL_NUM_THREADS=14 taskset -c 27-33,71-77 \
        .venv/bin/python exp/offline_search/rounds/r02/g4_t2/precompute.py [--queries]
The method recomputes 1-2 itself (single-threaded, flock-protected) when they are missing.
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))
import g4t2_core as core  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
MS = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]


def main():
    for ms in MS:
        t0 = time.time()
        core.ensure_current_pca(ROOT, ms)
        core.lib_proj(ROOT, ms, "current", "big")
        m = {f: json.loads((core.current_pca_dir(ms, f) / "meta.json").read_text()) for f in ("v0", "v1")}
        print(ms, "current PCA explained", {f: round(v["explained"], 4) for f, v in m.items()}, f"{time.time() - t0:.1f}s", flush=True)
    if "--queries" not in sys.argv:
        return
    for ms in MS:
        bases = {b: core.pca_basis(ROOT, ms, b)[0] for b in ("big", "current")}
        for arm in ("inf", "cache"):
            cell = f"{ms}_{arm}"
            for f in ("v0", "v1"):
                outs = {b: core.DERIVED / "qproj" / f"{cell}__b{b}_{f}.npy" for b in bases}
                if all(p.exists() for p in outs.values()):
                    continue
                t0 = time.time()
                X = np.load(pathlib.Path(ROOT) / "queries" / cell / f"key_{f}.npy", mmap_mode="r")
                mu = np.stack([bases[b][f][0] for b in bases])                   # [2, D]
                B = np.concatenate([bases[b][f][1] for b in bases], 1)            # [D, 128]
                muB = np.concatenate([mu[i] @ bases[b][f][1] for i, b in enumerate(bases)])
                out = np.empty((X.shape[0], B.shape[1]), np.float32)
                for lo in range(0, X.shape[0], core.CHUNK):
                    out[lo:lo + core.CHUNK] = np.asarray(X[lo:lo + core.CHUNK], np.float32) @ B - muB
                for i, b in enumerate(bases):
                    core.atomic_save(outs[b], np.ascontiguousarray(out[:, i * core.PDIM:(i + 1) * core.PDIM]))
                print(cell, f, X.shape, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
