"""R12 opus: does a different pooling of the stored visual tokens make one-camera retrieval closer to two-camera?

Library only (B-pool 50-episode R10 subsets, k-fold by episode as replay.py). The deployed key is the 4x4 mean pool
of the 16x16 token grid (verified bit-exact against key_v1). For the wrist camera we try other poolings of the same
tokens (no new encoder work online: the tokens already exist after the wrist encode):
  p4   4x4 pool (16 tokens x 2048)      = deployed key (reference W)
  p8   8x8 pool (64 x 2048)
  p2   2x2 pool (4 x 2048)
  p1   global mean (2048)
  p4n  4x4 pool of per-token L2-normalised tokens
each -> PCA64 on the training fold -> [PCA64, rs8] with its own AWM metric. Reported: e10 vs the policy chunk and the
paired difference to F (two-camera deployed features) on held-out look rows.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import pca_fit
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm


def pooled(tok, kind):
    n = tok.shape[0]
    x = tok.astype(np.float32).reshape(n, 16, 16, 2048)
    if kind == "p4n":
        x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
        return x.reshape(n, 4, 4, 4, 4, 2048).mean((2, 4)).reshape(n, -1)
    g = {"p8": 8, "p4": 4, "p2": 2, "p1": 1}[kind]
    b = 16 // g
    return x.reshape(n, g, b, g, b, 2048).mean((2, 4)).reshape(n, -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="pi05:l10:50")
    ap.add_argument("--view", default="v1")
    a = ap.parse_args(argv)
    model, suite, size = a.cell.split(":")
    size = int(size)
    C = cm.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    d = core.lib_dir(model, suite)
    tr_rows = np.load(d / "tok" / "rows.npy")
    if not np.array_equal(tr_rows, np.arange(len(tr_rows))):
        raise ValueError("tok rows are not the identity map")
    T = np.load(d / "tok" / f"{a.view}.npy", mmap_mode="r")
    kinds = ("p4", "p8", "p2", "p1", "p4n")
    reps = {k: [] for k in kinds}
    for lo in range(0, len(rows), 256):
        blk = np.asarray(T[rows[lo:lo + 256]])
        for k in kinds:
            reps[k].append(pooled(blk, k))
    reps = {k: np.concatenate(v) for k, v in reps.items()}
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    kref = cm.kref_for(size)
    res = {k: [] for k in kinds + ("F",)}
    for fold in range(5):
        tr = fold_of_row != fold
        P0, P1, _ = cm.projections(model, suite, size, fold, rows, tr, keep=128)
        sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
        heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
        Pk = {}
        for k in kinds:
            mu, V, _ = pca_fit(reps[k][tr], keep=64)
            Pk[k] = (reps[k] @ V - mu @ V).astype(np.float64)
        XF = np.concatenate([P0[:, :64], P1[:, :64], rs], 1).astype(np.float64)
        for t in range(10):
            itr = np.flatnonzero(tr & (task == t))
            ite = np.flatnonzero(~tr & (task == t) & (step % 2 == 0) & (step > 0))
            if not len(ite):
                continue
            for name in kinds + ("F",):
                X = XF if name == "F" else np.concatenate([Pk[name], rs], 1)
                met = core.TaskMetric(X[itr], heads[itr], ep[itr], step[itr])
                s, *_ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                   exclude_own=False)
                res[name].append(core.motion_err(act[ite], s, sig_r))
        print(f"fold {fold} done", flush=True)
    out = {}
    eF = np.concatenate(res["F"])
    for k in kinds + ("F",):
        e = np.concatenate(res[k])
        out[k] = dict(e10=float(e.mean()), d_vs_F=float((e - eF).mean()), se=float((e - eF).std() / np.sqrt(len(e))))
        print(f"{a.cell} {a.view} {k:4s} e10 {e.mean():.4f} vs F {(e - eF).mean():+.4f} (se {out[k]['se']:.4f})")
    cm.save_json(out, cm.OUT / f"tok_repr_{model}_{suite}_{size}_{a.view}.json")


if __name__ == "__main__":
    main()
