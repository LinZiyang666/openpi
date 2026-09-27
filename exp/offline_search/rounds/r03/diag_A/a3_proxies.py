"""Offline proxies computed on the trace store for the closed-loop arms' methods: which one orders the arms like SR?
Target: sp  B0 .668 < M4 .764 < AWM .800 ; l10  M4 .428 <= B0 .440 << AWM .630."""
import pathlib, json, sys
import numpy as np, pandas as pd

RES = pathlib.Path("/home/weiland/projects/openpi/exp/offline_search/results")
STORE = pathlib.Path("/dev/shm/offline_search_store")
OUT = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.precision", 3)

METHODS = [("B0", "r00/B0_current", "top1"), ("M4k5mean", "r01/M4_b0cons_k5_mean", "mean5"),
           ("AWMkr5", "r02/AWM_joint_cur_fcur_kr5", "kern5"),
           ("M4k8mean", "r01/M4_b0cons_k8_mean", "mean8"), ("M4k5med", "r01/M4_b0cons_k5_med", "mean5"),
           ("M4k3med", "r01/M4_b0cons_k3_med", "mean3"), ("AWMkr8", "r02/AWM_joint_cur_fcur", "kern8"),
           ("AWMins", "r02/AWM_joint_cur_fcur_ins", "kern8"), ("AWMfbig", "r02/AWM_joint_cur_fbig", "kern8"),
           ("AWMvis", "r02/AWM_vis_cur_fcur", "kern8"), ("V4cur", "r02/V4pc_LcurFcur_p32_st1_km8T0p5", "kern8v4")]
SR = {"pi05_spatial_cache": {"B0": .668, "M4k5mean": .764, "AWMkr5": .800},
      "pi05_l10_cache": {"B0": .440, "M4k5mean": .428, "AWMkr5": .630}}


def kernel_w(dt, kref):
    rel = dt - dt[0]; ref = max(rel[min(kref, len(dt)) - 1], 1e-6); return np.exp(-(rel / ref) ** 2)


def weights(kind, scores):
    k = len(scores)
    if kind == "top1": w = np.zeros(k); w[0] = 1
    elif kind.startswith("mean"): n = int(kind[4:]); w = np.zeros(k); w[:n] = 1
    elif kind.startswith("kern"):
        kref = int(kind[4]); w = kernel_w(-scores, kref)
    return w / w.sum()


def load_cell(cell):
    Q = STORE / "queries" / cell
    a_inf = np.asarray(np.load(Q / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float64)
    a_exec = np.asarray(np.load(Q / "a_exec.npy", mmap_mode="r")[:, :5, :7], np.float64)
    rec = np.load(Q / "rec_top1.npy"); ep = np.load(Q / "ep.npy"); step = np.load(Q / "step.npy")
    E = json.load(open(Q / "episodes.json")); succ = np.array([e["success"] for e in E])[ep]; nst = np.array([e["num_steps"] for e in E])[ep]
    m, s = cell.split("_")[:2]
    L = STORE / "library" / f"{m}_{s}" / "current"
    lact = np.asarray(np.load(L / "action.npy", mmap_mode="r")[:, :5, :7], np.float64)
    sig = lact.reshape(-1, 7).std(0)
    lib = dict(act=lact, sig=sig, ep=np.load(L / "episode.npy"), step=np.load(L / "step.npy"), ep_len=np.load(L / "ep_len.npy"),
               prog=np.load(L / "progress.npy"))
    # B0 stuck spells in the recorded trace: rec_top1 equals the previous decision's (same episode)
    prev_same = np.r_[False, (rec[1:] == rec[:-1]) & (ep[1:] == ep[:-1])]
    # teacher gripper transitions: a_inf sign at step t differs from the previous decision's a_inf sign (same episode)
    gs = np.sign(a_inf[:, 0, 6]); gprev = np.r_[0, gs[:-1]]; trans = np.r_[False, (ep[1:] == ep[:-1])] & (gs != gprev)
    intra = (np.sign(a_inf[:, :, 6]).min(1) != np.sign(a_inf[:, :, 6]).max(1))
    return dict(a_inf=a_inf, a_exec=a_exec, rec=rec, ep=ep, step=step, succ=succ, nst=nst, lib=lib, stuck=prev_same,
                trans=trans | intra, late=step >= 0.66 * nst)


def proxies(cell, C, name, path, kind):
    z = np.load(RES / path / f"{cell}.npz")
    n = len(z["err"]); assert n == len(C["ep"]), (n, len(C["ep"]))
    sig = C["lib"]["sig"]
    top = z["topk"]; sc = z["topk_scores"]
    used = bool(z["used_synth"].any())
    synth = z["synth_seg"].astype(np.float64) if used else C["lib"]["act"][z["top1"]]
    # per-decision set weights over the saved top-10
    W = np.zeros_like(sc, dtype=np.float64)
    for i in range(n):
        k = int((top[i] >= 0).sum()); W[i, :k] = weights(kind, sc[i, :k].astype(np.float64))
    rows = np.where(top >= 0, top, 0)
    lib = C["lib"]
    term = (lib["step"][rows] >= lib["ep_len"][rows] - 1)                      # terminal library rows in the set
    w_term = (W * term).sum(1)
    gset = np.sign(lib["act"][rows][:, :, 0, 6])                                # (n,10) first-step gripper sign of members
    gvote = np.abs((W * gset).sum(1))                                           # |weighted vote|, 1 = unanimous
    hs = synth / sig; ts = C["a_inf"] / sig
    sp_ratio = np.linalg.norm(hs[:, :, :6], axis=2).mean(1) / np.maximum(np.linalg.norm(ts[:, :, :6], axis=2).mean(1), 1e-9)
    # member-norm cancellation: |mean| / mean|member| over dims :6 (needs member heads)
    mem = lib["act"][rows] / sig                                                # (n,10,5,7)
    mnorm = np.linalg.norm(mem[:, :, :, :6], axis=3).mean(2)                    # (n,10)
    wmean = np.einsum("nk,nktd->ntd", W, mem)
    cancel = np.linalg.norm(wmean[:, :, :6], axis=2).mean(1) / np.maximum((W * mnorm).sum(1), 1e-9)
    err = z["err"]; gm = z["grip_mis"]
    # action change between consecutive decisions inside B0 stuck spells (escape potential) in sigma units
    dact = np.full(n, np.nan); same = np.r_[False, C["ep"][1:] == C["ep"][:-1]]
    dact[same] = np.sqrt(((hs[1:] - hs[:-1]) ** 2).mean((1, 2)))[same[1:]]
    st, tr, late, fail = C["stuck"], C["trans"], C["late"], ~C["succ"]
    out = dict(err=err.mean(), err_p50=np.median(err), err_p90=np.quantile(err, .9), err_worst10=np.sort(err)[-n // 10:].mean(),
               regret=z["regret"].mean(), grip_mis=gm.mean(), grip_mis_trans=gm[tr].mean(), err_trans=err[tr].mean(),
               err_stuck=err[st].mean(), err_free=err[~st].mean(), err_fail_eps=err[fail].mean(), err_succ_eps=err[~fail].mean(),
               err_late=err[late].mean(), speed=sp_ratio.mean(), speed_p50=np.median(sp_ratio), slow_frac=(sp_ratio < .5).mean(),
               cancel=cancel.mean(), cancel_stuck=cancel[st].mean(), gvote=gvote.mean(), gsplit=(gvote < .5).mean(),
               gsplit_stuck=(gvote[st] < .5).mean(), w_term=w_term.mean(), w_term_late=w_term[late].mean(),
               top1_term=term[:, 0].mean(), dact_stuck=np.nanmean(dact[st]), dact_free=np.nanmean(dact[~st]),
               stay_top1=float(np.mean(np.r_[False, (z["top1"][1:] == z["top1"][:-1]) & same[1:]])),
               conf_auroc_fail=np.nan)
    # per-task err / grip_mis / gsplit for the task-level check
    pt = pd.DataFrame(dict(task=z["task_id"], err=err, gm=gm, gsplit=(gvote < .5).astype(float), cancel=cancel, w_term=w_term,
                           speed=sp_ratio)).groupby("task").mean()
    return out, pt


def main():
    allrows = {}
    pertask = {}
    for cell in ("pi05_spatial_cache", "pi05_l10_cache"):
        C = load_cell(cell)
        print(f"\n===== {cell}: n={len(C['ep'])}, B0-stuck decisions {C['stuck'].mean():.3f}, teacher gripper-transition decisions {C['trans'].mean():.3f}, failed-episode share {(~C['succ']).mean():.3f}")
        rows = {}
        for name, path, kind in METHODS:
            try:
                rows[name], pertask[(cell, name)] = proxies(cell, C, name, path, kind)
            except Exception as e:
                print("skip", name, e)
        df = pd.DataFrame(rows).T
        df.insert(0, "SR", [SR[cell].get(k, np.nan) for k in df.index])
        allrows[cell] = df
        print(df.round(3).to_string())
    # ordering check on the three deployed arms
    print("\n===== proxy ordering check (deployed arms). '+' = proxy ranks the three arms exactly like SR (lower proxy = higher SR unless noted) =====")
    higher_better = {"speed", "speed_p50", "gvote", "dact_stuck", "dact_free", "cancel", "cancel_stuck"}
    res = []
    for col in allrows["pi05_spatial_cache"].columns:
        if col == "SR": continue
        r = dict(proxy=col)
        for cell, df in allrows.items():
            d = df.loc[["B0", "M4k5mean", "AWMkr5"], [col, "SR"]].dropna()
            if len(d) < 3: r[cell[5:-6]] = "na"; continue
            sgn = 1 if col in higher_better else -1
            from scipy.stats import spearmanr
            rho = spearmanr(sgn * d[col], d.SR).correlation
            r[cell[5:-6]] = f"{rho:+.1f}"
            r[cell[5:-6] + "_vals"] = " ".join(f"{v:.3f}" for v in d[col])
        res.append(r)
    print(pd.DataFrame(res).to_string())
    pd.to_pickle((allrows, pertask), OUT / "a3_proxies.pkl")
    # per-task: offline delta vs closed-loop SR delta
    CL = pd.read_pickle(OUT / "tables" / "oscl50_p_sp_cl0_epfeat.pkl")
    print("\n===== per-task offline deltas (method - B0) vs closed-loop SR deltas =====")
    for cell, suite in (("pi05_spatial_cache", "sp"), ("pi05_l10_cache", "l10")):
        sr = {i: pd.read_pickle(OUT / "tables" / f"oscl50_p_{suite}_cl{i}_epfeat.pkl").groupby("task").success.mean() for i in range(3)}
        for name, i in (("M4k5mean", 1), ("AWMkr5", 2)):
            pt = pertask[(cell, name)]; b = pertask[(cell, "B0")]
            d = pd.DataFrame(dict(dSR=sr[i] - sr[0], d_err=pt.err - b.err, d_gm=pt.gm - b.gm, gsplit=pt.gsplit, cancel=pt.cancel,
                                  d_wterm=pt.w_term - b.w_term, speed=pt.speed, err=pt.err, b0_err=b.err))
            from scipy.stats import spearmanr
            print(f"\n{cell} {name}: spearman(dSR, x) over 10 tasks:",
                  {k: round(spearmanr(d.dSR, d[k]).correlation, 2) for k in ["d_err", "d_gm", "gsplit", "cancel", "d_wterm", "speed", "err"]})
            print(d.round(3).T.to_string())


if __name__ == "__main__":
    main()
