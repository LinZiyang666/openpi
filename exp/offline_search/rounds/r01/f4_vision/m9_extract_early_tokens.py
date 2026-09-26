"""M9: extract the full vision tokens (and, for pi0.5, the model-input images) of decision steps 0..2 of EVERY query
episode from the trace h5 files (read-only). The store's tok/ subsample only holds 5 inits per task.

Output (big arrays, outside the repo):
  <out>/<cell>/v0.npy, v1.npy    float16 [n_eps, 3, 256, 2048]   (vision_0 agentview, vision_1 wrist)
  <out>/<cell>/img0.npy, img1.npy uint8 [n_eps, 3, 224, 224, 3]  pi0.5 only: input_images/base_0_rgb, left_wrist_0_rgb
                                                                 (the same image source as library tok/img0|img1)
  <out>/<cell>/rows.npy          int64 [n_eps, 3]  query row of (episode, step) (-1 when the episode is shorter)
  <out>/<cell>/DONE
Checks: the 4x4 pool of the extracted tokens equals the stored query key_v0/key_v1 up to the online key's reduced-
precision pooling (max relative L2 diff, printed; measured 1.7e-3, cosine 0.999998).

    taskset -c 18-25,62-69 .venv/bin/python exp/offline_search/rounds/r01/f4_vision/m9_extract_early_tokens.py --procs 16
"""
import argparse
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path("/dev/shm/offline_search_store")
OUT = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/early_tokens")
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
NS = 3


def _alloc(cell):
    eps = json.loads((ROOT / "queries" / cell / "episodes.json").read_text())
    d = OUT / cell
    d.mkdir(parents=True, exist_ok=True)
    n = len(eps)
    rows = np.full((n, NS), -1, np.int64)
    for i, e in enumerate(eps):
        for s in range(min(NS, e["end"] - e["start"])):
            rows[i, s] = e["start"] + s
    np.save(d / "rows.npy", rows)
    for f in ("v0", "v1"):
        np.lib.format.open_memmap(d / f"{f}.npy", mode="w+", dtype=np.float16, shape=(n, NS, 256, 2048)).flush()
    if cell.startswith("pi05"):
        for f in ("img0", "img1"):
            np.lib.format.open_memmap(d / f"{f}.npy", mode="w+", dtype=np.uint8, shape=(n, NS, 224, 224, 3)).flush()
    return eps


def _work(args):
    cell, lo, hi = args
    eps = json.loads((ROOT / "queries" / cell / "episodes.json").read_text())
    d = OUT / cell
    V = {f: np.load(d / f"{f}.npy", mmap_mode="r+") for f in ("v0", "v1")}
    pi = cell.startswith("pi05")
    if pi:
        V.update({f: np.load(d / f"{f}.npy", mmap_mode="r+") for f in ("img0", "img1")})
    for i in range(lo, hi):
        e = eps[i]
        with h5py.File(e["file"], "r") as h:
            for s in range(min(NS, e["end"] - e["start"])):
                g = h[f"step_{s:04d}"]
                V["v0"][i, s] = g["vision_0"][()]
                V["v1"][i, s] = g["vision_1"][()]
                if pi:
                    V["img0"][i, s] = g["input_images/base_0_rgb"][()]
                    V["img1"][i, s] = g["input_images/left_wrist_0_rgb"][()]
    for a in V.values():
        a.flush()
    return cell, lo, hi


def _check(cell):
    d = OUT / cell
    rows = np.load(d / "rows.npy")
    out = {}
    for f, k in (("v0", "key_v0"), ("v1", "key_v1")):
        T = np.load(d / f"{f}.npy", mmap_mode="r")
        K = np.load(ROOT / "queries" / cell / f"{k}.npy", mmap_mode="r")
        mx = 0.0
        for i in range(0, rows.shape[0], 37):
            for s in range(NS):
                if rows[i, s] < 0:
                    continue
                p = T[i, s].astype(np.float32).reshape(4, 4, 4, 4, 2048).mean(axis=(1, 3)).reshape(-1)
                k = np.asarray(K[rows[i, s]], np.float32)
                mx = max(mx, float(np.linalg.norm(p - k) / np.linalg.norm(k)))
        out[f] = mx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--cells", default=",".join(CELLS))
    a = ap.parse_args()
    cells = a.cells.split(",")
    jobs = []
    for c in cells:
        eps = _alloc(c)
        n = len(eps)
        step = 25
        jobs += [(c, lo, min(n, lo + step)) for lo in range(0, n, step)]
    with ProcessPoolExecutor(a.procs) as ex:
        for k, (c, lo, hi) in enumerate(ex.map(_work, jobs)):
            if k % 20 == 0:
                print("done", k, "/", len(jobs), c, lo, hi, flush=True)
    for c in cells:
        chk = _check(c)
        print(c, "pool-vs-key max rel L2", chk, flush=True)
        (OUT / c / "DONE").write_text(json.dumps(chk))


if __name__ == "__main__":
    sys.exit(main())
