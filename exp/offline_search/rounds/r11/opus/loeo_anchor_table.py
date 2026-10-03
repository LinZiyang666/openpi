"""R11 opus: whole-episode-out (5-fold) anchor table for the IR model and the schedule-knob analysis. Library only.

For every R10 nested cell-size and fold f (episodes at permutation position % 5 == f are held out), exactly as
R10 opus's k-fold pipeline (rounds/r10/analysis_opus/tools/run_offline.py, ``pca='auto'``):
  * PCA refitted on the training fold for sizes < 500, the stored r01 bpool basis for 500;
  * per-task AWM metric (main + early) fitted on training-fold rows; kref 5 at 5 episodes/task, else 8;
  * every held-out row is served by the training-fold cache (the held-out episode is a "new episode").
Per held-out row we keep what the schedule / IR analysis needs and R10 opus did not store:
  * top1 = the cache's nearest admissible row (parent row id) and its library progress / episode length, which
    drive the frozen R8 only-no-progress guard (span rule, noprog_n 3, prog_eps .5);
  * e10 (uncorrected motion error vs the row's own stored policy chunk, sigma units, R10's reporting yardstick),
    gripper sign disagreement over 10 controls, d1 / d1norm (distance to the nearest training row).
The R10 opus k-fold npz of the same (cell, size, fold) is joined by row id to bring in the LOEO-corrector
dot/cc terms, so a GC_dist-strength corrected error can be reconstructed per row (see ir_model.corrected_err).

Data rule: arrays come only from offline_search_store/library (and its r01 PCA basis) through R10 opus's
``core`` (its path guard refuses trace_runs/os_closed_loop), plus R10 opus's own library-only k-fold outputs.

Run (opus CPUs):
  taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
    CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r11.opus.loeo_anchor_table --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r10.analysis_opus.tools.run_offline import kref_for, projections

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "anchor"
R10_KFOLD = HERE.parents[1] / "r10" / "analysis_opus" / "out" / "kfold"
JOIN_KEYS = ("full|none|e10", "full|loeo|dot", "full|loeo|cc", "d1norm")
CELL_SIZES = [("pi05", "l10", 50), ("pi05", "l10", 200), ("pi05", "l10", 500), ("pi05", "spatial", 50),
              ("groot", "l10", 50), ("groot", "l10", 200), ("groot", "l10", 500), ("groot", "spatial", 50),
              # outside the R11 grid; cheap context for "strong" Spatial libraries
              ("pi05", "spatial", 200), ("pi05", "spatial", 500), ("groot", "spatial", 200), ("groot", "spatial", 500)]


def run_job(job):
    model, suite, size, fold = job
    out_path = OUT / f"{model}_{suite}_{size}_f{fold}.npz"
    if out_path.exists():
        return str(out_path), 0.0
    t0 = time.time()
    C = core.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    tr_m = fold_of_row != fold
    P, pca_src = projections(model, suite, rows, tr_m, size, "auto")
    act = C["act"][rows]
    rs = C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    prog_all = np.load(Path(C["dir"]) / "progress.npy").astype(np.float64)
    sig_m = act[tr_m][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    kref = kref_for(size)
    X = np.concatenate([P, rs], 1)
    heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
    rec = {k: [] for k in ("row", "task", "ep", "step", "ep_len", "success", "top1", "top1_prog", "top1_eplen",
                           "top1_ep", "d1", "e10", "e5", "grip_mis")}
    for t in range(10):
        itr = np.flatnonzero(tr_m & (task == t))
        ite = np.flatnonzero(~tr_m & (task == t))
        if not len(ite):
            continue
        met = core.TaskMetric(X[itr], heads[itr], ep[itr], step[itr])
        s_te, d1_te, _, mem, _ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                            exclude_own=False)
        top1 = rows[itr][mem[:, 0]]                      # parent row id of the nearest admissible training row
        rec["row"].append(rows[ite]); rec["task"].append(task[ite]); rec["ep"].append(ep[ite])
        rec["step"].append(step[ite]); rec["ep_len"].append(C["ep_len"][rows][ite])
        rec["success"].append(C["success"][rows][ite])
        rec["top1"].append(top1); rec["top1_prog"].append(prog_all[top1]); rec["top1_eplen"].append(C["ep_len"][top1])
        rec["top1_ep"].append(C["ep"][top1])
        rec["d1"].append(d1_te)
        rec["e10"].append(core.motion_err(act[ite], s_te, sig_r))
        rec["e5"].append(core.motion_err(act[ite], s_te, sig_r, steps=5))
        rec["grip_mis"].append(np.mean(np.sign(act[ite][:, :10, 6]) != np.sign(s_te[:, :10, 6]), axis=1))
    arrays = {k: np.concatenate(v) for k, v in rec.items()}
    # join R10 opus's library-only k-fold outputs (same cell, size, fold, PCA rule) by row id
    src = R10_KFOLD / f"{model}_{suite}_{size}_f{fold}_auto.npz"
    join = dict(source=str(src), matched=False)
    if src.exists():
        z = np.load(src, allow_pickle=False)
        pos = {int(r): i for i, r in enumerate(z["row"])}
        idx = np.array([pos.get(int(r), -1) for r in arrays["row"]])
        if (idx >= 0).all():
            for k in JOIN_KEYS:
                arrays["r10|" + k] = z[k][idx]
            dev = np.abs(arrays["r10|full|none|e10"] - arrays["e10"])
            join.update(matched=True, max_abs_e10_dev=float(dev.max()), mean_abs_e10_dev=float(dev.mean()))
    meta = dict(model=model, suite=suite, size=size, fold=fold, pca=pca_src, kref=kref, sig_m=sig_m.tolist(),
                sig_r=sig_r.tolist(), join=join, wall_s=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return str(out_path), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="", help="comma list model:suite:size (default: all CELL_SIZES)")
    a = ap.parse_args(argv)
    cells = CELL_SIZES if not a.only else [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2]))
                                          for s in a.only.split(",")]
    jobs = [(m, s, n, f) for m, s, n in cells for f in range(5)]
    jobs.sort(key=lambda j: -j[2])
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for path, dt in pool.imap_unordered(run_job, jobs):
            print(f"done {path} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
