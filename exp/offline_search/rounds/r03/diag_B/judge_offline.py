"""R3 ideation B: HIT/MISS judge analysis on the offline trace (read-only).

For every model x suite, methods B0 (results/r00/B0_current), AWMc (AWM_joint_cur_fcur_kr5), AWMb (AWM_joint_big_fbig):
  * per-decision features (online-legal): regime, motion / still / stuck_n, overtime, lag5, terminal_top1, method extras
  * confidences: own; V7-like fixed-weight z-sum + isotonic (cross-fitted by episode parity); cross-fitted linear +
    isotonic; each with / without drift guards (force MISS)
  * risk at target hit rates 30/50/70 % in the stale regime (cache cells), fresh regime (inf cells) and the mixture
    (after-HIT states weighted h, after-MISS states weighted 1-h; one threshold): accepted err, bad rate, grip_mis,
    fraction of accepted decisions in failed episodes / overtime, catastrophic (err > 1), per-episode bad-HIT counts
  * staleness penalty curve: cache err(step) - inf err(step)
Outputs: judge_<ms>.json + judge_tables.md in this directory.
"""
import json, math, os, sys, pathlib
import numpy as np
from multiprocessing import Pool

REPO = pathlib.Path("/home/weiland/projects/openpi")
RES = REPO / "exp/offline_search/results"
STORE = pathlib.Path("/dev/shm/offline_search_store")
PCA_BIG = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
OUT = pathlib.Path(__file__).resolve().parent
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
METHODS = {"B0": RES / "r00/B0_current", "AWMc": RES / "r02/AWM_joint_cur_fcur_kr5", "AWMb": RES / "r02/AWM_joint_big_fbig"}
MS = [("pi05", "spatial"), ("pi05", "l10"), ("groot", "spatial"), ("groot", "l10")]
HR = [0.3, 0.5, 0.7]
CURVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BIN_NAMES = ["early", "mid", "late"]


def pav_decreasing(x, y):
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    sy, sx, n = [], [], []
    for xi, yi in zip(xs, ys):
        sy.append(yi); sx.append(xi); n.append(1)
        while len(sy) > 1 and sy[-2] / n[-2] < sy[-1] / n[-1]:
            by, bx, bn = sy.pop(), sx.pop(), n.pop()
            sy[-1] += by; sx[-1] += bx; n[-1] += bn
    kx = np.asarray(sx) / np.asarray(n); ky = np.asarray(sy) / np.asarray(n)
    ux, inv = np.unique(kx, return_inverse=True)
    if ux.size != kx.size:
        kn = np.asarray(n, float)
        ky = np.bincount(inv, weights=ky * kn) / np.bincount(inv, weights=kn); kx = ux
    return kx, ky


def iso_crossfit(score, err, ep, min_n=30):
    """Isotonic (non-increasing) map score -> err, fitted on the other episode parity. Returns pred err."""
    pred = np.full(score.shape, np.nan)
    par = ep % 2
    for p in (0, 1):
        fit, app = par != p, par == p
        if fit.sum() < min_n or app.sum() == 0:
            continue
        kx, ky = pav_decreasing(score[fit], err[fit])
        pred[app] = np.interp(score[app], kx, ky)
    return pred


def linfit_crossfit(X, err, ep, ridge=1e-3):
    """Cross-fitted (episode parity) ridge regression of err on X (standardized inside the fit half)."""
    out = np.full(err.shape, np.nan)
    par = ep % 2
    for p in (0, 1):
        fit, app = par != p, par == p
        if fit.sum() < 50:
            continue
        mu = X[fit].mean(0); sd = X[fit].std(0); sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        A = np.c_[Z[fit], np.ones(fit.sum())]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ err[fit])
        out[app] = np.c_[Z[app], np.ones(app.sum())] @ w
    return out


def aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())


def wquantile_threshold(conf, w, h):
    """tau such that sum w[conf >= tau] = h (w sums to 1). -inf confs (guards) never accepted."""
    o = np.argsort(-conf, kind="stable")
    cw = np.cumsum(w[o])
    k = int(np.searchsorted(cw, h, side="left"))
    k = min(k, conf.size - 1)
    return float(conf[o][k])


def load_lib(model, suite, name):
    d = STORE / "library" / f"{model}_{suite}" / name
    L = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("step", "ep_len", "task_id", "episode", "next", "prev", "progress")}
    L["rs"] = np.load(d / "rs.npy", mmap_mode="r")
    L["dir"] = d
    return L


def lib_motion_thr(L, pct=10.0):
    nxt = np.asarray(L["next"]); ok = nxt >= 0
    rs = np.asarray(L["rs"], np.float32)[:, :8]
    m = np.linalg.norm(rs[nxt[ok]] - rs[ok], axis=1)
    return float(np.percentile(m, pct))


def lib_still_thr(L, mu0, mu1, pct=95.0, chunk=1024):
    """95th pct of the library's consecutive-decision 'still' = sum over cameras of cos(key_t - mu, key_prev - mu)
    (AWM's x_still definition: centred at the PCA mean of the fit library)."""
    prev = np.asarray(L["prev"]); ok = np.flatnonzero(prev >= 0)
    vals = np.empty(ok.size, np.float64)
    for f, mu in (("v0", mu0), ("v1", mu1)):
        K = np.load(L["dir"] / f"key_{f}.npy", mmap_mode="r")
        acc = np.empty(ok.size, np.float64)
        for lo in range(0, ok.size, chunk):
            r = ok[lo:lo + chunk]
            a = np.asarray(K[r], np.float32) - mu
            b = np.asarray(K[prev[r]], np.float32) - mu
            acc[lo:lo + chunk] = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        if f == "v0":
            vals[:] = acc
        else:
            vals += acc
    return float(np.percentile(vals, pct))


def pca_mean(model, suite, big, f):
    return np.array(np.load(PCA_BIG / f"{model}_{suite}" / big / f / "mean.npy"), np.float32)


def cur_mean(L, f, chunk=2048):
    K = np.load(L["dir"] / f"key_{f}.npy", mmap_mode="r")
    acc = np.zeros(K.shape[1], np.float64)
    for lo in range(0, K.shape[0], chunk):
        acc += np.asarray(K[lo:lo + chunk], np.float64).sum(0)
    return (acc / K.shape[0]).astype(np.float32)


def load_cell(mdir, cell):
    z = np.load(mdir / f"{cell}.npz", allow_pickle=True)
    j = json.load(open(mdir / f"{cell}.json"))
    return z, j


def build(model, suite):
    key = f"{model}_{suite}"
    out = {"key": key}
    libs = {"current": load_lib(model, suite, "current"), BIG[model]: load_lib(model, suite, BIG[model])}
    m_thr = lib_motion_thr(libs["current"])
    mu_cur = (cur_mean(libs["current"], "v0"), cur_mean(libs["current"], "v1"))
    mu_big = (pca_mean(model, suite, BIG[model], "v0"), pca_mean(model, suite, BIG[model], "v1"))
    c_thr = {"current": lib_still_thr(libs["current"], *mu_cur), BIG[model]: lib_still_thr(libs["current"], *mu_big)}
    out["thr"] = {"m_thr": m_thr, "c_thr": c_thr}
    # per-library tables
    med_len = {}
    for name, L in libs.items():
        t = np.asarray(L["task_id"]); ep = np.asarray(L["episode"]); el = np.asarray(L["ep_len"])
        med_len[name] = {}
        for tt in np.unique(t):
            eps, first = np.unique(ep[t == tt], return_index=True)
            med_len[name][int(tt)] = float(np.median(el[t == tt][first]))
    # queries
    Q = {}
    for arm in ("inf", "cache"):
        cell = f"{key}_{arm}"
        d = STORE / "queries" / cell
        eps = json.load(open(d / "episodes.json"))
        Q[arm] = {"ep": np.asarray(np.load(d / "ep.npy")), "step": np.asarray(np.load(d / "step.npy")),
                  "rs": np.asarray(np.load(d / "rs.npy", mmap_mode="r"))[:, :8].astype(np.float32),
                  "success": np.asarray([e["success"] for e in eps], bool),
                  "num_steps": np.asarray([e["num_steps"] for e in eps], np.int64)}
    # still (visual self-change) per query row from the AWM npz (observation-side; AWMc centred at the current mean,
    # AWMb at the big mean) -- aligned by 'row'
    D = {}
    for mname, mdir in METHODS.items():
        for arm in ("inf", "cache"):
            cell = f"{key}_{arm}"
            z, j = load_cell(mdir, cell)
            q = Q[arm]
            row = z["row"]; assert np.all(np.diff(row) > 0)
            ep, step, task = z["ep"], z["step"].astype(np.int64), z["task_id"]
            assert np.array_equal(ep, q["ep"][row]) and np.array_equal(step, q["step"][row])
            libname = str(z["lib_names"][0]); L = libs[libname]
            lstep = np.asarray(L["step"]); lnext = np.asarray(L["next"])
            top1 = z["top1"]; topk = z["topk"][:, :5]
            tk = np.where(topk >= 0, topk, top1[:, None])
            lag5 = step - lstep[tk].mean(1)
            terminal = lnext[top1] < 0
            ml = med_len[libname]
            overtime = step / np.asarray([ml[int(t)] for t in task])
            # motion / stuck
            rs = q["rs"][row]
            motion = np.full(step.shape, np.nan)
            same = np.r_[False, ep[1:] == ep[:-1]]
            motion[same] = np.linalg.norm(rs[same] - rs[np.flatnonzero(same) - 1], axis=1)
            fp90 = np.asarray([j["floor"]["p90"][b] for b in BIN_NAMES])[z["bin"].astype(np.int64)]
            fmed = np.asarray([j["floor"]["median"][b] for b in BIN_NAMES])[z["bin"].astype(np.int64)]
            rec = dict(row=row, ep=ep, step=step, task=task, err=z["err"], grip=z["grip_mis"], bad=(z["err"] > fp90),
                       indist=(z["err"] <= fmed), conf=z["confidence"], lag5=lag5, terminal=terminal, overtime=overtime,
                       motion=motion, lib=libname, success=q["success"][ep], num_steps=q["num_steps"][ep],
                       regime=np.where(step == 0, 0, 1 if arm == "inf" else 2))
            for k in z.files:
                if k.startswith("x_"):
                    rec[k[2:]] = z[k]
            D[(mname, arm)] = rec
    # still from AWM npz -> shared by all methods of that library family (B0 uses AWMc's = current-mean centring)
    for arm in ("inf", "cache"):
        D[("B0", arm)]["still"] = D[("AWMc", arm)]["still"]
    for (mname, arm), r in D.items():
        cth = c_thr[r["lib"]]
        still = (r["motion"] < m_thr) & (np.nan_to_num(r["still"], nan=-9) >= cth)
        sn = np.zeros(still.size, np.int64)
        c = 0
        for i in range(still.size):
            c = c + 1 if (still[i] and i > 0 and r["ep"][i] == r["ep"][i - 1]) else 0
            sn[i] = c
        r["stuck_n"] = sn
        r["guard"] = (sn >= 2) | ((r["regime"] == 2) & r["terminal"]) | ((r["overtime"] > 1) & (r["lag5"] > 5) & (sn >= 1))
        r["guard_soft"] = (sn >= 2) | ((r["overtime"] > 1) & (r["lag5"] > 5) & (sn >= 1))
    out["med_len"] = {k: v for k, v in med_len.items()}
    return out, D


def feats(r, mname, reg):
    """Feature matrix (higher = more risk sign already applied: each column is a 'risk' feature)."""
    n = r["err"].size
    st = np.minimum(r["stuck_n"], 5).astype(float)
    nanv = np.full(n, np.nan)
    if mname.startswith("AWM"):
        if reg == 1:
            cols = [r.get("c0", nanv), r["disp5"], r["d1_rel"]]
            names = ["c0", "disp5", "d1_rel"]
        else:
            cols = [r["dst"], r["disp5"], r["overtime"], r["d1_rel"], st, np.abs(r["lag5"])]
            names = ["dst", "disp5", "overtime", "d1_rel", "stuck", "abslag"]
    else:
        vis = -(r["cos_v0"] + r["cos_v1"])
        if reg == 1:
            cols = [r["dist_rs"], vis]
            names = ["dist_rs", "-vis"]
        else:
            cols = [r["dist_rs"], vis, r["overtime"], st, np.abs(r["lag5"])]
            names = ["dist_rs", "-vis", "overtime", "stuck", "abslag"]
    X = np.stack([np.asarray(c, float) for c in cols], 1)
    return X, names


def confidences(D, mname):
    """Return dict conf_name -> {arm: conf array} for the method (pooled calibration across both arms per regime)."""
    A = {arm: D[(mname, arm)] for arm in ("inf", "cache")}
    n_inf, n_cache = A["inf"]["err"].size, A["cache"]["err"].size
    reg = np.r_[A["inf"]["regime"], A["cache"]["regime"]]
    err = np.r_[A["inf"]["err"], A["cache"]["err"]]
    ep = np.r_[A["inf"]["ep"], A["cache"]["ep"] + 100000]     # distinct parity groups still by episode index
    own = np.r_[A["inf"]["conf"], A["cache"]["conf"]]
    v7 = np.full(err.shape, np.nan); lin = np.full(err.shape, np.nan)
    v7z = np.full(err.shape, np.nan)
    for rg in (0, 1, 2):
        m = reg == rg
        if m.sum() < 60:
            continue
        # regime rows come from both arms (step 0: both; fresh: inf; stale: cache)
        parts = []
        for arm in ("inf", "cache"):
            mm = A[arm]["regime"] == rg
            X, names = feats(A[arm], mname, rg)
            parts.append(X[mm])
        X = np.concatenate(parts, 0)
        med = np.nanmedian(X, 0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        e = err[m]; ee = ep[m]
        # fixed-weight z-sum (all weights +1 on risk features) -> iso
        mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1
        z = ((X - mu) / sd).sum(1)
        v7z[m] = -z
        v7[m] = -iso_crossfit(-z, e, ee)
        s = linfit_crossfit(X, e, ee)
        lin[m] = -iso_crossfit(-s, e, ee)
    guard = np.r_[A["inf"]["guard"], A["cache"]["guard"]]
    gsoft = np.r_[A["inf"]["guard_soft"], A["cache"]["guard_soft"]]
    C = {"own": own, "v7": v7, "lin": lin, "v7z": v7z,
         "v7+g": np.where(guard, -np.inf, v7), "lin+g": np.where(guard, -np.inf, lin), "own+g": np.where(guard, -np.inf, own),
         "lin+gs": np.where(gsoft, -np.inf, lin)}
    split = lambda a: {"inf": a[:n_inf], "cache": a[n_inf:]}
    return {k: split(v) for k, v in C.items()}


def accepted_stats(r, acc):
    """Stats over accepted decisions of one arm."""
    n = acc.sum()
    if n == 0:
        return {"n": 0}
    e = r["err"][acc]
    st = {"n": int(n), "err": float(e.mean()), "bad": float(r["bad"][acc].mean()), "grip": float(r["grip"][acc].mean()),
          "cat1": float((e > 1.0).mean()), "indist": float(r["indist"][acc].mean()),
          "in_fail": float((~r["success"][acc]).mean()), "overtime": float((r["overtime"][acc] > 1).mean()),
          "stuck2": float((r["stuck_n"][acc] >= 2).mean())}
    # per-episode accepted bad hits
    eps = np.unique(r["ep"])
    cnt = np.bincount(r["ep"][acc & r["bad"]], minlength=eps.max() + 1)[eps]
    st["badhits_per_ep"] = float(cnt.mean()); st["ep_ge3bad"] = float((cnt >= 3).mean())
    return st


def evaluate(D, mname, C):
    res = {}
    A = {arm: D[(mname, arm)] for arm in ("inf", "cache")}
    for cname, cc in C.items():
        rr = {"aurc": {}, "at": {}}
        for arm, regname in (("cache", "stale"), ("inf", "fresh")):
            r = A[arm]; m = r["regime"] > 0
            c = cc[arm][m]; e = r["err"][m]
            fin = np.isfinite(c) | (c == -np.inf)
            rr["aurc"][regname] = aurc(np.where(np.isfinite(c), c, -1e9)[fin], e[fin])
        # regime-wise thresholds at target hit rates
        for h in HR + [x for x in CURVE if x not in HR]:
            row = {}
            for arm, regname in (("cache", "stale"), ("inf", "fresh")):
                r = A[arm]; c = np.where(np.isfinite(cc[arm]), cc[arm], -1e9)
                w = np.full(c.size, 1.0 / c.size)
                tau = wquantile_threshold(c, w, h)
                acc = (c >= tau) & (cc[arm] > -np.inf)
                s = accepted_stats(r, acc); s["tau"] = tau; s["rate"] = float(acc.mean())
                row[regname] = s
            # mixture: after-HIT states weighted h, after-MISS states weighted 1-h; ONE threshold
            c_all = np.r_[np.where(np.isfinite(cc["inf"]), cc["inf"], -1e9), np.where(np.isfinite(cc["cache"]), cc["cache"], -1e9)]
            n_i, n_c = cc["inf"].size, cc["cache"].size
            w = np.r_[np.full(n_i, (1 - h) / n_i), np.full(n_c, h / n_c)]
            tau = wquantile_threshold(c_all, w, h)
            acc_i = (c_all[:n_i] >= tau) & (cc["inf"] > -np.inf)
            acc_c = (c_all[n_i:] >= tau) & (cc["cache"] > -np.inf)
            si, sc = accepted_stats(A["inf"], acc_i), accepted_stats(A["cache"], acc_c)
            mix = {"tau": tau, "rate_fresh": float(acc_i.mean()), "rate_stale": float(acc_c.mean()),
                   "rate": float((1 - h) * acc_i.mean() + h * acc_c.mean())}
            for k in ("err", "bad", "grip", "cat1", "in_fail", "overtime"):
                a, b = si.get(k, np.nan), sc.get(k, np.nan)
                wi, wc = (1 - h) * acc_i.mean(), h * acc_c.mean()
                mix[k] = float((wi * a + wc * b) / max(wi + wc, 1e-12)) if (si["n"] and sc["n"]) else (a if si["n"] else b)
            mix["stale"] = sc; mix["fresh"] = si
            row["mix"] = mix
            rr["at"][f"{h:.1f}"] = row
        res[cname] = rr
    return res


def staleness(D, mname, steps=(1, 2, 3, 5, 8, 12, 16, 20, 30)):
    out = {}
    for s in steps:
        a = D[(mname, "cache")]; b = D[(mname, "inf")]
        ma, mb = a["step"] == s, b["step"] == s
        if ma.sum() > 30 and mb.sum() > 30:
            out[str(s)] = {"cache": float(a["err"][ma].mean()), "inf": float(b["err"][mb].mean()),
                           "n_cache": int(ma.sum()), "n_inf": int(mb.sum()),
                           "cache_succ": float(a["err"][ma & a["success"]].mean()) if (ma & a["success"]).sum() > 10 else None}
    return out


def run_ms(ms):
    model, suite = ms
    meta, D = build(model, suite)
    result = {"meta": meta, "methods": {}}
    for mname in METHODS:
        C = confidences(D, mname)
        result["methods"][mname] = {"eval": evaluate(D, mname, C), "stale_curve": staleness(D, mname),
                                    "n": {arm: int(D[(mname, arm)]["err"].size) for arm in ("inf", "cache")},
                                    "flags": {arm: {"guard": float(D[(mname, arm)]["guard"][D[(mname, arm)]["regime"] > 0].mean()),
                                                    "guard_err": float(D[(mname, arm)]["err"][D[(mname, arm)]["guard"]].mean()) if D[(mname, arm)]["guard"].any() else None,
                                                    "noguard_err": float(D[(mname, arm)]["err"][~D[(mname, arm)]["guard"] & (D[(mname, arm)]["regime"] > 0)].mean()),
                                                    "stuck2": float((D[(mname, arm)]["stuck_n"] >= 2).mean()),
                                                    "terminal": float(D[(mname, arm)]["terminal"][D[(mname, arm)]["regime"] == 2].mean()) if arm == "cache" else None}
                                              for arm in ("inf", "cache")}}
        # save per-decision confidences for the closed-loop threshold table
        np.savez_compressed(OUT / f"conf_{model}_{suite}_{mname}.npz",
                            **{f"{k}_{arm}": v[arm] for k, v in C.items() for arm in ("inf", "cache")},
                            err_inf=D[(mname, "inf")]["err"], err_cache=D[(mname, "cache")]["err"],
                            regime_inf=D[(mname, "inf")]["regime"], regime_cache=D[(mname, "cache")]["regime"])
    json.dump(result, open(OUT / f"judge_{model}_{suite}.json", "w"), indent=1, default=float)
    return f"{model}_{suite}", result


def fmt_tables(all_res):
    lines = []
    order = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
    def cells(fn):
        return " / ".join(fn(all_res[k]) if k in all_res else "-" for k in order)
    lines.append("# Judge analysis (cells: pi05-sp / pi05-l10 / groot-sp / groot-l10)\n")
    lines.append("## AURC by regime (lower better)\n")
    lines.append("| method | conf | stale | fresh |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        for c in ("own", "v7z", "v7", "lin", "lin+g"):
            lines.append(f"| {m} | {c} | " + cells(lambda R: f"{R['methods'][m]['eval'][c]['aurc']['stale']:.3f}") + " | "
                         + cells(lambda R: f"{R['methods'][m]['eval'][c]['aurc']['fresh']:.3f}") + " |")
    for h in HR:
        hk = f"{h:.1f}"
        lines.append(f"\n## Target hit rate {h:.0%}: regime-wise thresholds\n")
        lines.append("| method | conf | stale err | stale bad | stale in_fail | stale overtime | fresh err | fresh bad |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            for c in ("own", "v7", "lin", "lin+g", "own+g"):
                g = lambda k, reg: cells(lambda R: f"{R['methods'][m]['eval'][c]['at'][hk][reg].get(k, float('nan')):.3f}")
                lines.append(f"| {m} | {c} | {g('err','stale')} | {g('bad','stale')} | {g('in_fail','stale')} | {g('overtime','stale')} | {g('err','fresh')} | {g('bad','fresh')} |")
        lines.append(f"\n## Target hit rate {h:.0%}: ONE threshold over the mixture (after-HIT weight h, after-MISS weight 1-h)\n")
        lines.append("| method | conf | rate_stale | rate_fresh | mix err | mix bad | mix grip | mix in_fail | mix overtime | tau |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            for c in ("own", "v7", "lin", "lin+g", "own+g"):
                g = lambda k: cells(lambda R: f"{R['methods'][m]['eval'][c]['at'][hk]['mix'][k]:.3f}")
                lines.append(f"| {m} | {c} | {g('rate_stale')} | {g('rate_fresh')} | {g('err')} | {g('bad')} | {g('grip')} | {g('in_fail')} | {g('overtime')} | {g('tau')} |")
    lines.append("\n## Guards (stale decisions flagged; err flagged vs not)\n")
    lines.append("| method | guard rate stale | err flagged | err unflagged | guard rate fresh | stuck2 stale | terminal stale |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in METHODS:
        F = lambda arm, k: cells(lambda R: (f"{R['methods'][m]['flags'][arm][k]:.3f}" if R['methods'][m]['flags'][arm][k] is not None else "-"))
        lines.append(f"| {m} | {F('cache','guard')} | {F('cache','guard_err')} | {F('cache','noguard_err')} | {F('inf','guard')} | {F('cache','stuck2')} | {F('cache','terminal')} |")
    lines.append("\n## Staleness penalty: err at step s, cache (all / B0-successful episodes) vs inf\n")
    for m in METHODS:
        lines.append(f"\n{m}: step: cache/inf (cache_succ)")
        for k in order:
            if k not in all_res:
                continue
            sc = all_res[k]["methods"][m]["stale_curve"]
            lines.append(f"  {k}: " + "; ".join(f"s{s}: {v['cache']:.2f}/{v['inf']:.2f} ({v['cache_succ']:.2f})" if v['cache_succ'] else f"s{s}: {v['cache']:.2f}/{v['inf']:.2f}" for s, v in sc.items()))
    lines.append("\n## Curves (mixture, lin+g): hit rate -> mix err / bad / in_fail\n")
    for m in METHODS:
        for c in ("own", "lin+g"):
            lines.append(f"\n{m} {c}:")
            for k in order:
                if k not in all_res:
                    continue
                at = all_res[k]["methods"][m]["eval"][c]["at"]
                lines.append(f"  {k}: " + "; ".join(f"h{h:.1f}: {at[f'{h:.1f}']['mix']['err']:.3f}/{at[f'{h:.1f}']['mix']['bad']:.3f}/{at[f'{h:.1f}']['mix']['in_fail']:.3f}" for h in CURVE))
    return "\n".join(lines)


if __name__ == "__main__":
    which = MS if len(sys.argv) < 2 else [tuple(a.split("_")) for a in sys.argv[1:]]
    with Pool(len(which)) as pool:
        results = dict(pool.map(run_ms, which))
    txt = fmt_tables(results)
    (OUT / "judge_tables.md").write_text(txt)
    print(txt)
