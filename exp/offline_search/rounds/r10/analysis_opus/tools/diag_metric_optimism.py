"""Why are held-out rows farther from the library (d1 / LOEO d1 ~ 1.7 at size 50)?  PCA vs metric in-sample effect.

For one cell/size/fold: held-out d1 (nearest training row, main metric) normalized by the median LOEO d1 of the
training rows, under (a) PCA + metric fitted on the training fold only (deployment-faithful), (b) PCA training-only,
metric fitted on training + held-out rows (metric leak), (c) PCA and metric on all rows (both leak). Also the served
error of the held-out rows vs the LOEO served error of the training rows (in-sample optimism of LOEO targets)."""
import json, sys
import numpy as np
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r10.analysis_opus.tools.run_offline import kref_for

def main(model, suite, size, fold):
    C = core.load_cell(model, suite)
    rows, fr = core.subset_and_folds(C, size)
    tr = fr != fold
    act, rs, task, ep, step = C["act"][rows], C["rs"][rows], C["task"][rows], C["ep"][rows], C["step"][rows]
    sig = act[tr][:, :5, :7].reshape(-1, 7).std(0)
    if size == 500:
        P0, P1 = core.stored_pca(model, suite)
        Pall = np.concatenate([P0[rows], P1[rows]], 1).astype(np.float64)
        Ptr = Pall
    else:
        K0, K1 = core.gather_keys(model, suite, rows)
        Ptr, Pall = [], []
        for Kx in (K0, K1):
            mu, V, _ = core.pca_fit(Kx[tr]); Ptr.append(Kx @ V - mu @ V)
            mu, V, _ = core.pca_fit(Kx); Pall.append(Kx @ V - mu @ V)
        Ptr, Pall = np.concatenate(Ptr, 1).astype(np.float64), np.concatenate(Pall, 1).astype(np.float64)
    heads = (act[:, :5, :7] / sig).reshape(len(act), -1)
    out = {}
    for label, P, metric_rows in (("a_train_only", Ptr, "train"), ("b_metric_leak", Ptr, "all"), ("c_pca_and_metric_leak", Pall, "all")):
        X = np.concatenate([P, rs], 1)
        r_te, r_lo, e_te, e_lo = [], [], [], []
        for t in range(10):
            itr = np.flatnonzero(tr & (task == t)); ite = np.flatnonzero(~tr & (task == t))
            fit = itr if metric_rows == "train" else np.flatnonzero(task == t)
            met = core.TaskMetric(X[fit], heads[fit], ep[fit], step[fit])
            s_lo, d1_lo, *_ = core.serve(X[itr], step[itr], ep[itr], met, X[itr], None, ep[itr], act[itr], kref_for(size))
            s_te, d1_te, *_ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref_for(size), exclude_own=False)
            med = np.median(d1_lo)
            r_te.append(d1_te / med); r_lo.append(d1_lo / med)
            e_te.append(core.motion_err(act[ite], s_te, sig)); e_lo.append(core.motion_err(act[itr], s_lo, sig))
        out[label] = dict(heldout_d1norm_median=float(np.median(np.concatenate(r_te))),
                          heldout_err10=float(np.concatenate(e_te).mean()), loeo_train_err10=float(np.concatenate(e_lo).mean()))
    print(json.dumps(dict(cell=f"{model}_{suite}_{size}_f{fold}", **out), indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
