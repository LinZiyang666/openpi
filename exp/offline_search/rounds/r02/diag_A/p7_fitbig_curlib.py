import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p1_diag import load_lib, load_q, sigma_d, err_of, ROOT, SCR, BIG
from p3_white import l2score, fit_transform, kern_synth
def main(cell):
    ms = cell.rsplit("_", 1)[0]; model = ms.split("_")[0]
    B = load_lib(ms, BIG[model]); C = load_lib(ms, "current"); Q = load_q(cell); sig = sigma_d(ms)
    N = Q["N"]; step = Q["step"]
    hB = (B["a5"] / sig).reshape(B["L"], 35)
    def feats(L, r): return np.concatenate([L["P0"][r, :64], L["P1"][r, :64], L["rs8"][r]], 1).astype(np.float64)
    e_cur_bigfit = np.full(N, np.nan); e_cur_bigfit_early = np.full(N, np.nan); e_big = np.full(N, np.nan); e_b0 = err_of(Q["a_exec"][:, :5, :], Q["a_inf"], sig)
    for t in np.unique(Q["task_id"]):
        qi = np.where(Q["task_id"] == t)[0]; rb = np.where(B["task_id"] == t)[0]; rc = np.where(C["task_id"] == t)[0]
        Xb = feats(B, rb); Xc = feats(C, rc); Xq = feats(Q, qi) if False else np.concatenate([Q["P0"][qi, :64], Q["P1"][qi, :64], Q["rs8"][qi]], 1).astype(np.float64)
        for lam, early, tgt in ((0.1, False, e_cur_bigfit), (0.1, True, e_cur_bigfit_early)):
            mean, std, W = fit_transform(Xb, hB[rb], B["episode"][rb], B["step"][rb], nn=3, lam=lam, rows_mask=(B["step"][rb] <= 2) if early else None)
            S = l2score(((Xq - mean) / std) @ W, ((Xc - mean) / std) @ W)
            a, o, _ = kern_synth(S, C["a5"][rc]); tgt[qi] = err_of(a, Q["a_inf"][qi], sig)
        mean, std, W = fit_transform(Xb, hB[rb], B["episode"][rb], B["step"][rb], nn=3, lam=0.1)
        S = l2score(((Xq - mean) / std) @ W, ((Xb - mean) / std) @ W)
        a, o, _ = kern_synth(S, B["a5"][rb]); e_big[qi] = err_of(a, Q["a_inf"][qi], sig)
    m = step >= 1
    res = {"cell": cell, "stale_cur_bigfit": float(np.nanmean(e_cur_bigfit[m])), "step0_cur_bigfit": float(np.nanmean(e_cur_bigfit[~m])),
           "step0_cur_bigfit_early": float(np.nanmean(e_cur_bigfit_early[~m])), "stale_big": float(np.nanmean(e_big[m])), "stale_b0": float(np.nanmean(e_b0[m]))}
    res["per_task"] = {int(t): {"n": int(((Q["task_id"] == t) & m).sum()), "big": float(np.nanmean(e_big[(Q["task_id"] == t) & m])), "b0": float(np.nanmean(e_b0[(Q["task_id"] == t) & m])),
                                "cur_bigfit": float(np.nanmean(e_cur_bigfit[(Q["task_id"] == t) & m]))} for t in np.unique(Q["task_id"])}
    json.dump(res, open(SCR / "out" / f"fb_{cell}.json", "w"), indent=1); print(cell, flush=True)
if __name__ == "__main__":
    main(sys.argv[1])
