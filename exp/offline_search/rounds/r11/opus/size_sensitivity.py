"""R11 opus: how the replayed guard rate g depends on the number of library episodes per task (library only).

The whole-episode-out replay of an n-episode-per-task library necessarily serves each held-out episode from
n-1 (50-cells: 4 of 5) or 4n/5 episodes per task, i.e. a sparser cache than the deployed one. This script measures
g(n) on the 500-episode B-pool libraries (stored bpool PCA basis, 5 folds by permutation position) with the
training side subsampled to n in {3, 4, 5, 8, 16, 40} episodes per task (keyed subsample, same for every n-nested
draw), so the bias of the 50-cell estimate (n = 4 instead of 5) can be read off without any closed-loop data.

  taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= \
    .venv/bin/python -m exp.offline_search.rounds.r11.opus.size_sensitivity
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r11.opus.ir_model import guard_flags

OUT = Path(__file__).resolve().parent / "out"
NS = (3, 4, 5, 8, 16, 40)


def job(args):
    model, suite, fold = args
    C = core.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, 500)
    P0, P1 = core.stored_pca(model, suite)
    P = np.concatenate([P0[rows], P1[rows]], 1).astype(np.float64)
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    prog_all = np.load(Path(C["dir"]) / "progress.npy").astype(np.float64)
    X = np.concatenate([P, rs], 1)
    tr_m = fold_of_row != fold
    sig = act[tr_m][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    heads = (act[:, :5, :7] / sig).reshape(len(act), -1)
    out = {n: [0, 0] for n in NS}
    for t in range(10):
        ite = np.flatnonzero(~tr_m & (task == t))
        tr_eps = np.unique(ep[tr_m & (task == t)])
        order = np.random.default_rng([fold, t, 99]).permutation(tr_eps)      # nested subsamples
        for n in NS:
            keep = order[:n]
            itr = np.flatnonzero(tr_m & (task == t) & np.isin(ep, keep))
            met = core.TaskMetric(X[itr], heads[itr], ep[itr], step[itr])
            kref = 5 if n <= 5 else 8
            _, _, _, mem, _ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                         exclude_own=False)
            top1 = rows[itr][mem[:, 0]]
            for e in np.unique(ep[ite]):
                m = np.flatnonzero(ep[ite] == e)
                m = m[np.argsort(step[ite][m])]
                a = m[step[ite][m] % 2 == 0]
                g = guard_flags(step[ite][a], prog_all[top1[a]], C["ep_len"][top1[a]])
                out[n][0] += int(g.sum()); out[n][1] += len(g)
    return model, suite, fold, out


def main():
    jobs = [(m, s, f) for m in ("pi05", "groot") for s in ("l10", "spatial") for f in range(5)]
    with mp.get_context("fork").Pool(16) as pool:
        res = pool.map(job, jobs)
    agg = {}
    for model, suite, fold, out in res:
        a = agg.setdefault(f"{model}_{suite}", {n: [0, 0] for n in NS})
        for n, (gs, an) in out.items():
            a[n][0] += gs; a[n][1] += an
    table = {c: {str(n): v[0] / v[1] for n, v in d.items()} for c, d in agg.items()}
    (OUT / "size_sensitivity.json").write_text(json.dumps(table, indent=1) + "\n")
    L = ["| library (500 B-pool, stored PCA) | " + " | ".join(f"g at n={n}/task" for n in NS) + " |",
         "|---|" + "---|" * len(NS)]
    for c, d in table.items():
        L.append(f"| {c} | " + " | ".join(f"{d[str(n)]:.3f}" for n in NS) + " |")
    (OUT / "size_sensitivity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
