"""Closed-loop decision-sequence mining: per-episode features, paired flips B0 -> AWM, success-vs-failure separation."""
import pathlib, sys
import numpy as np, pandas as pd

T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
CAP = {"sp": 44, "l10": 104}


def auroc(x, y):
    """AUROC of feature x for label y (1 = success); NaN-safe (drops NaN)."""
    m = np.isfinite(x); x, y = x[m], y[m]
    if y.sum() == 0 or (1 - y).sum() == 0: return np.nan
    from scipy.stats import rankdata
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ep_features(D, cap):
    out = []
    for uid, g in D.groupby("uid", sort=False):
        g = g.sort_values("step")
        t1 = g.top1.values; le = g.lib_ep.values; ls = g.lib_step.values; n = len(g)
        stay = (t1[1:] == t1[:-1])
        same = (le[1:] == le[:-1])
        adv = np.where(same, ls[1:] - ls[:-1], np.nan)
        # longest run of identical picks
        run, best = 1, 1
        for i in range(1, n):
            run = run + 1 if t1[i] == t1[i - 1] else 1; best = max(best, run)
        g0 = g.g0.values; gflip = (g0[1:] != g0[:-1]).sum()
        # gripper "open->close->open" chatter: count sign changes beyond the first two (a pick&place needs <= 2..4)
        prog = g.lib_prog.values
        f = dict(uid=uid, task=int(g.task.iloc[0]), init=int(g.init.iloc[0]), success=bool(g.success.iloc[0]), n_dec=n,
                 timeout=n >= cap, stay=stay.mean() if n > 1 else 0, same_ep=same.mean() if n > 1 else 0,
                 adv_mean=np.nanmean(adv) if np.isfinite(adv).any() else np.nan,
                 back=(np.nan_to_num(adv, nan=0) < 0).mean() if n > 1 else 0,       # backward jumps within same lib ep
                 maxrun=best, stuck3=(np.convolve(np.r_[0, stay].astype(int), np.ones(2, int), "same") >= 2).mean(),
                 gflips=gflip, gflip_rate=gflip / max(n - 1, 1), gabs=g.gabs.mean(), g_intra=(g.g0 != g.g4).mean(),
                 tnorm=g.tnorm.mean(), rnorm=g.rnorm.mean(), tnorm_first10=g.tnorm.values[:10].mean(),
                 conf=g.conf.mean(), conf_min=g.conf.min(), prog_max=prog.max(), prog_last=prog[-1],
                 prog_p90=np.quantile(prog, .9), n_lib_eps=len(np.unique(le)),
                 lib_end=(ls >= g.lib_eplen.values - 1).mean(),                      # pick is the last row of its lib episode
                 agree=g.agree.astype(float).mean() if g.agree.notna().any() else np.nan,
                 infer_ms=g.infer_ms.median())
        for k in ("x_d1_rel", "x_dst", "x_still", "x_disp5", "x_w_eff", "x_margin", "x_cos_v0", "x_cos_v1", "x_dist_rs"):
            if k in g:
                f[k] = g[k].mean(); f[k + "_max"] = g[k].max()
        if "x_still" in g:
            f["still_hi"] = (g.x_still > 1.98).mean()                                  # both cameras cos > .99
        out.append(f)
    return pd.DataFrame(out).set_index("uid")


def main(suite):
    arms = [f"oscl50_p_{suite}_cl{i}" for i in range(3)]
    F = {}
    for a in arms:
        D = pd.read_pickle(T / f"{a}_dec.pkl")
        F[a] = ep_features(D, CAP[suite]); F[a].to_pickle(T / f"{a}_epfeat.pkl")
    key = lambda df: df.reset_index().set_index(["task", "init"])
    print(f"\n===== {suite}: episode-level summary =====")
    rows = []
    for a in arms:
        f = F[a]
        rows.append(dict(arm=a, SR=f.success.mean(), n_dec_succ=f.n_dec[f.success].mean(), n_dec_fail=f.n_dec[~f.success].mean(),
                         timeout_frac_of_fail=f.timeout[~f.success].mean(), fail_n=(~f.success).sum(),
                         stay=f.stay.mean(), same_ep=f.same_ep.mean(), maxrun=f.maxrun.mean(), gflips=f.gflips.mean(),
                         gabs=f.gabs.mean(), tnorm=f.tnorm.mean(), rnorm=f.rnorm.mean(), prog_max=f.prog_max.mean(),
                         n_lib_eps=f.n_lib_eps.mean(), lib_end=f.lib_end.mean(), agree=f.agree.mean(), back=f.back.mean()))
    print(pd.DataFrame(rows).set_index("arm").T)
    print(f"\n--- {suite}: per-task SR (rows tasks) ---")
    print(pd.DataFrame({a: F[a].groupby("task").success.mean() for a in arms}).round(2).T)
    print(f"\n--- {suite}: paired outcomes by init (rows = CL0 outcome, cols = CLx outcome) ---")
    b0 = key(F[arms[0]])
    for a in arms[1:]:
        x = key(F[a]); j = b0[["success"]].join(x[["success"]], rsuffix="_x")
        print(a, "\n", pd.crosstab(j.success, j.success_x, rownames=["CL0"], colnames=[a[-3:]]))
        print("  per-task flips fail->succ:", j[(~j.success) & j.success_x].groupby(level=0).size().to_dict(),
              " succ->fail:", j[(j.success) & ~j.success_x].groupby(level=0).size().to_dict())
    print(f"\n--- {suite}: success-vs-failure feature means within arm (S / F) and AUROC(feature -> success) ---")
    feats = ["n_dec", "stay", "same_ep", "adv_mean", "back", "maxrun", "gflips", "gflip_rate", "gabs", "g_intra", "tnorm", "rnorm",
             "tnorm_first10", "conf", "conf_min", "prog_max", "prog_last", "n_lib_eps", "lib_end", "agree",
             "x_d1_rel", "x_dst", "x_still", "still_hi", "x_disp5", "x_w_eff", "x_margin", "x_cos_v0", "x_cos_v1", "x_dist_rs"]
    tab = {}
    for a in arms:
        f = F[a]; y = f.success.values.astype(float)
        for k in feats:
            if k in f:
                tab[(a[-3:], k)] = dict(S=f[k][f.success].mean(), F=f[k][~f.success].mean(), auroc=auroc(f[k].values.astype(float), y))
    tab = pd.DataFrame(tab).T
    print(tab.unstack(0).round(3).to_string())
    print(f"\n--- {suite}: paired feature deltas CL2 - CL0 on the same init, split by flip type ---")
    x = key(F[arms[2]]); j = b0.join(x, rsuffix="_2")
    j["flip"] = np.select([(~j.success) & j.success_2, j.success & ~j.success_2, j.success & j.success_2], ["F->S", "S->F", "S->S"], "F->F")
    for k in ["n_dec", "stay", "same_ep", "maxrun", "gflips", "gabs", "tnorm", "rnorm", "prog_max", "n_lib_eps", "lib_end", "back"]:
        j["d_" + k] = j[k + "_2"] - j[k]
    cols = [c for c in j.columns if c.startswith("d_")]
    print(j.groupby("flip")[cols].mean().round(3).T)
    print(j.groupby("flip").size())


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["sp", "l10"]):
        main(s)
