"""Project pooled keys (queries of all 8 cells + library current) into the big-library PCA-128 bases (r01 F4).
One process, BLAS threads = range size (one-off). Output: <SCR>/proj/<cell>_{v0,v1}.npy [N,128] f32 (centred, NOT normalised),
<SCR>/proj/lib_<ms>_current_{v0,v1}.npy. Big-library projections are the r01 proj.npy files (centred, not normalised).
"""
import pathlib, sys, time, json
import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")
PCA = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
SCR = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r02_ideation_A")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
CH = 2048


def proj(X, mu, B):
    out = np.empty((X.shape[0], B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, X.shape[0], CH):
        hi = min(X.shape[0], lo + CH)
        out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ B - muB
    return out


def main():
    (SCR / "proj").mkdir(parents=True, exist_ok=True)
    for ms in ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]:
        m = ms.split("_")[0]
        for f in ("v0", "v1"):
            d = PCA / ms / BIG[m] / f
            mu = np.load(d / "mean.npy").astype(np.float32)
            B = np.ascontiguousarray(np.load(d / "basis.npy"), np.float32)
            t0 = time.time()
            for arm in ("inf", "cache"):
                cell = f"{ms}_{arm}"
                out = SCR / "proj" / f"{cell}_{f}.npy"
                if out.exists():
                    continue
                X = np.load(ROOT / "queries" / cell / f"key_{f}.npy", mmap_mode="r")
                np.save(out, proj(X, mu, B))
            out = SCR / "proj" / f"lib_{ms}_current_{f}.npy"
            if not out.exists():
                X = np.load(ROOT / "library" / ms / "current" / f"key_{f}.npy", mmap_mode="r")
                np.save(out, proj(X, mu, B))
            print(ms, f, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
