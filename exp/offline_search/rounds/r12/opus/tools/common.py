"""R12 opus (look-half) shared library-only helpers.

Data rule: arrays come only from offline_search_store/library/<model>_<suite>/<bpool_cs|bpool_all> (the B pool) and
its r01 PCA basis, through R10 opus's ``core`` path guard. Nothing under trace_runs/os_closed_loop and nothing under
offline_search_store/queries is opened here.

Folds: R10's nested subsets (``core.subset_and_folds``): the first size/10 episodes of each task's seeded permutation;
fold = permutation position % 5. Held-out rows are served by the training-fold cache. For 50-episode libraries a
fold holds out exactly one episode per task (= leave-one-episode-out); for 500 it holds out 10 per task.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w, fit_metric, pca_fit
from exp.offline_search.rounds.r10.analysis_opus.tools import core

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"
CACHE = HERE / "cache"                       # disposable projection cache (library-derived only)
K = 16
COSTS = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}
PRICE = dict(full_owner=0.152, full_req=0.1656, wrist=0.0646, completion=0.0502)


def kref_for(size):
    return 5 if size <= 50 else 8


def load_cell(model, suite):
    C = core.load_cell(model, suite)
    d = Path(C["dir"])
    C["prog"] = np.load(core.assert_library_path(d / "progress.npy")).astype(np.float64)
    C["act_full"] = None
    return C


def lag_index(C, rows, lag=2):
    """Position (within ``rows``) of the same episode's row ``lag`` steps earlier; self when step < lag."""
    pos = {int(r): i for i, r in enumerate(rows)}
    out = np.arange(len(rows))
    st = C["step"][rows]
    for i, r in enumerate(rows):
        if st[i] >= lag:
            j = pos.get(int(r) - lag)
            if j is None or C["ep"][rows[j]] != C["ep"][r] or C["step"][rows[j]] != st[i] - lag:
                raise ValueError("non-contiguous episode rows")
            out[i] = j
    return out


def next_index(C, rows, n=1):
    """Position of the same episode's row n steps later (clamped at the episode end)."""
    pos = {int(r): i for i, r in enumerate(rows)}
    out = np.arange(len(rows))
    for i, r in enumerate(rows):
        j = i
        for _ in range(n):
            jj = pos.get(int(rows[j]) + 1)
            if jj is None or C["ep"][rows[jj]] != C["ep"][rows[j]]:
                break
            j = jj
        out[i] = j
    return out


def projections(model, suite, size, fold, rows, train_mask, keep=128):
    """Per-view PCA projections [n, keep] for the fold (refit on the training fold for size < 500; the stored r01
    bpool basis for 500, as R10/R11 opus). Cached under r12/opus/cache (library-derived, disposable)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"proj_{model}_{suite}_{size}_f{fold}_k{keep}.npz"
    if p.exists():
        z = np.load(p)
        return z["P0"], z["P1"], str(z["src"])
    if size == 500:
        base = core.STORE / "derived" / "r01" / "f4_vision" / "pca" / f"{model}_{suite}" / core.PARENT[model]
        Ps = []
        for f in ("v0", "v1"):
            q = core.assert_library_path(base / f / "proj.npy")
            Ps.append(np.ascontiguousarray(np.load(q, mmap_mode="r")[:, :keep], np.float32)[rows])
        src = "stored_r01_bpool"
    else:
        K0, K1 = core.gather_keys(model, suite, rows)
        Ps = []
        for Kx in (K0, K1):
            mu, V, _ = pca_fit(Kx[train_mask], keep=keep)
            Ps.append((Kx @ V - mu @ V).astype(np.float32))
        del K0, K1
        src = "refit_trainfold"
    np.savez(p, P0=Ps[0], P1=Ps[1], src=np.array(src))
    return Ps[0], Ps[1], src


class Metric:
    """AWM per-task Mahalanobis metric (main + early) on arbitrary features, fitted on training rows."""

    def __init__(self, X, heads, ep, step, lam=0.1, nn=3):
        self.mean, self.std, self.W = fit_metric(X, heads, ep, nn=nn, lam=lam, rank=0)
        m0 = step <= 2
        self.mean0, self.std0, self.W0 = fit_metric(X[m0], heads[m0], ep[m0], nn=nn, lam=lam, rank=0)

    def code(self, X):
        return ((X - self.mean) / self.std) @ self.W

    def code0(self, X):
        return ((X - self.mean0) / self.std0) @ self.W0


def topk(D, kref, k=K):
    return core.topk_kernel(D, kref, k)


def motion_err(a, b, sig, steps=10):
    return np.sqrt(np.mean(((a[..., :steps, :6] - b[..., :steps, :6]) / sig[:6]) ** 2, axis=(-1, -2)))


def grip_mis(a, b, steps=10):
    return np.mean(np.sign(a[..., :steps, 6]) != np.sign(b[..., :steps, 6]), axis=-1)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def stage_arrays(model, suite):
    """R7 StageTable segmentation on the whole parent library (successful rows; failed rows unknown):
    mode (-1 unknown / 0 / 1), event_near (gripper change within one row), next (successor, -1 none)."""
    d = Path(core.lib_dir(model, suite))
    act = np.load(d / "action.npy", mmap_mode="r")
    command = np.median(np.asarray(act[:, :5, 6], np.float64), axis=1)
    task = np.load(d / "task_id.npy").astype(np.int64)
    ep = np.load(d / "episode.npy").astype(np.int64)
    step = np.load(d / "step.npy").astype(np.int64)
    succ = np.load(d / "success.npy").astype(bool)
    nxt = np.load(d / "next.npy").astype(np.int64)
    r = np.flatnonzero(nxt >= 0)
    good = (ep[r] == ep[nxt[r]]) & (step[nxt[r]] == step[r] + 1)
    nxt[r[~good]] = -1
    x = command[succ]
    lo, hi = np.quantile(x, [.25, .75])
    c = np.array([lo, hi])
    for _ in range(100):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        new = np.array([x[lab == k].mean() if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    thr = float(np.sort(c).mean())
    mode = np.full(len(task), -1, np.int8)
    event = np.zeros(len(task), bool)
    for e in np.unique(ep[succ]):
        rows = np.flatnonzero((ep == e) & succ)
        rows = rows[np.argsort(step[rows], kind="stable")]
        m = command[rows] > thr
        if len(m) > 2:
            m = np.r_[m[0], (m[:-2].astype(int) + m[1:-1] + m[2:]) >= 2, m[-1]]
        ev = np.flatnonzero(np.diff(m)) + 1
        mode[rows] = m
        if len(ev):
            pos = np.arange(len(rows))
            event[rows] = np.abs(pos[:, None] - ev).min(axis=1) <= 1
    return dict(mode=mode, event=event, next=nxt, succ=succ, thr=thr)


def advance(nxt, rows, n):
    out = np.asarray(rows, np.int64).copy()
    for _ in range(n):
        ok = out >= 0
        new = np.full(out.shape, -1, np.int64)
        new[ok] = nxt[out[ok]]
        out = new
    return out
