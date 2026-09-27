"""Vectorized whole-cell diagnostic of AWMT2 variants (NOT the harness; same fitted arrays, batch arithmetic).

Per variant x cell: fit the method exactly as the harness does (library current view + Context), then score every
decision of the cell with AWMT2.batch_eval (query projections from precompute.py --queries; prev_hit by arm: the
store is 100 % HIT in cache arms / 100 % MISS in inf arms). For T2 variants the SAME run also scores the full AWM
codes of the same fit (use_awm=True) -> a paired in-run control. Reports per regime (step 0 / stale / fresh):
err, gripper-sign-faithful err, grip_mis, AURC (harness definition), n.

    taskset -c 27-33,71-77 .venv/bin/python exp/offline_search/rounds/r02/g4_t2/diag_eval.py \
        --variants '[{"phi":"mlp","lib":"big"}]' --cells all --procs 8 --out <json>
"""
import argparse
import json
import os
import pathlib
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[4]))
from exp.offline_search.harness import api, store  # noqa: E402
import g4t2_core as core  # noqa: E402
from method import AWMT2  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]


def aurc(conf, err):
    o = np.lexsort((np.arange(len(conf)), -np.asarray(conf, np.float64)))
    e = np.asarray(err, np.float64)[o]
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def fit_method(kw, cell, root=ROOT):
    m = AWMT2(**kw)
    ctx = api.Context(root=root, cell=cell, seed=0, scratch=pathlib.Path("/tmp") / f"g4t2_scr_{os.getpid()}")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    lib = store.LibraryView(root, ctx.lib_key, "current")
    t0 = time.time()
    m.fit(lib, ctx)
    return m, ctx, time.time() - t0


def eval_cell(job):
    kw, cell = job
    m, ctx, fit_s = fit_method(kw, cell)
    qd = pathlib.Path(ROOT) / "queries" / cell
    eps = json.loads((qd / "episodes.json").read_text())
    ep = np.load(qd / "ep.npy")
    step = np.load(qd / "step.npy").astype(np.int64)
    task = np.array([e["task_id"] for e in eps])[ep]
    succ = np.array([bool(e["success"]) for e in eps])[ep]
    X = np.concatenate([np.load(core.DERIVED / "qproj" / f"{cell}__b{m.basis}_{f}.npy") for f in ("v0", "v1")]
                       + [np.load(qd / "rs.npy")[:, :8].astype(np.float32)], 1).astype(np.float32)
    rs8 = np.ascontiguousarray(X[:, -8:])
    ainf = np.asarray(np.load(qd / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float64)
    aex = np.load(qd / "a_exec.npy", mmap_mode="r")
    tails = np.zeros((len(step), core.NH), np.float32)
    s1 = np.flatnonzero(step >= 1)
    tails[s1] = (np.asarray(aex[s1 - 1][:, 5:10, :7], np.float32) / m.sig32).reshape(len(s1), core.NH)
    ph = np.full(len(step), 1 if cell.endswith("_cache") else 0, np.int8)
    sig = m.sig32.astype(np.float64)
    res = {"cell": cell, "name": m.name, "kw": kw, "fit_s": fit_s, "fit_info": {k: m.fit_info.get(k) for k in ("train", "train_cache_hit", "fit_s")}}
    arms = [("method", False)] + ([("awm_ctrl", True)] if m.phi in ("lin", "mlp") else [])
    per = {}
    for arm_name, use_awm in arms:
        err = np.full(len(step), np.nan)
        esf = np.full(len(step), np.nan)
        gm = np.full(len(step), np.nan)
        conf = np.full(len(step), np.nan)
        top1 = np.full(len(step), -1)
        for t in np.unique(task):
            r = np.flatnonzero(task == t)
            for lo in range(0, len(r), 2048):
                rr = r[lo:lo + 2048]
                o = m.batch_eval(int(t), X[rr], step[rr], ph[rr], tails[rr], rs8[rr], use_awm=use_awm)
                a5 = o["a5"].astype(np.float64)
                err[rr] = np.sqrt(np.mean(((a5 - ainf[rr]) / sig) ** 2, axis=(1, 2)))
                a5s = a5.copy()
                a5s[..., 6] = np.sign(a5s[..., 6])
                esf[rr] = np.sqrt(np.mean(((a5s - ainf[rr]) / sig) ** 2, axis=(1, 2)))
                gm[rr] = np.mean((a5[..., 6] >= 0) != (ainf[rr][..., 6] >= 0), axis=1)
                conf[rr] = o["conf"]
                top1[rr] = o["top1"]
        per[arm_name] = dict(err=err, esf=esf, conf=conf, top1=top1)
        R = {}
        for reg, msk in (("step0", step == 0), ("stale" if cell.endswith("_cache") else "fresh", step >= 1)):
            R[reg] = {"n": int(msk.sum()), "err": float(err[msk].mean()), "err_sf": float(esf[msk].mean()),
                      "grip_mis": float(gm[msk].mean()), "aurc": aurc(conf[msk], err[msk]),
                      "err_succ_eps": float(err[msk & succ].mean()),
                      "err_fail_eps": float(err[msk & ~succ].mean()) if (msk & ~succ).any() else float("nan")}
        res[arm_name] = R
    if "awm_ctrl" in per:
        a, b = per["method"], per["awm_ctrl"]
        s = step >= 1
        d = a["err"][s] - b["err"][s]
        # episode bootstrap CI of the paired difference (stale / fresh rows)
        ue, inv = np.unique(ep[s], return_inverse=True)
        sm = np.bincount(inv, weights=d)
        cn = np.bincount(inv).astype(np.float64)
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)) for _ in range(1000)]).astype(np.float64)
        reps = (W @ sm) / (W @ cn)
        res["paired_step_ge1"] = {"d": float(d.mean()), "lo": float(np.quantile(reps, .025)), "hi": float(np.quantile(reps, .975)),
                                  "win": float((d < -1e-9).mean()), "loss": float((d > 1e-9).mean()),
                                  "top1_same": float((a["top1"][s] == b["top1"][s]).mean())}
    print(json.dumps({"cell": cell, "name": m.name, **{k: res[k] for k in res if k in ("method", "awm_ctrl", "paired_step_ge1")}}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON list of AWMT2 kwargs")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit-only", action="store_true", help="fit (train) every variant x model-suite sequentially "
                    "in this process (run with CUDA visible to train on the GPU), print the train logs, exit")
    a = ap.parse_args()
    variants = json.loads(a.variants)
    cells = CELLS if a.cells == "all" else a.cells.split(",")
    if a.fit_only:
        seen = set()
        for kw in variants:
            for c in cells:
                ms = c.rsplit("_", 1)[0]
                if (json.dumps(kw, sort_keys=True), ms) in seen:
                    continue
                seen.add((json.dumps(kw, sort_keys=True), ms))
                m, _ctx, fs = fit_method(kw, f"{ms}_cache")
                tr = m.fit_info.get("train") or {}
                print(json.dumps({"ms": ms, "name": m.name, "fit_s": round(fs, 1), "cache_hit": m.fit_info.get("train_cache_hit"),
                                  **{k: tr.get(k) for k in ("device", "train_s", "steps", "P", "loss_init", "loss_final",
                                                            "val_loss_init", "val_loss_final", "dead_units",
                                                            "residual_norm_ratio")}}), flush=True)
        return
    jobs = [(kw, c) for kw in variants for c in cells]
    # biggest cells first
    jobs.sort(key=lambda j: -np.load(pathlib.Path(ROOT) / "queries" / j[1] / "step.npy", mmap_mode="r").shape[0])
    procs = max(1, min(a.procs, len(jobs), len(os.sched_getaffinity(0))))
    if procs == 1:
        res = [eval_cell(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(procs) as pool:
            res = pool.map(eval_cell, jobs, chunksize=1)
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
