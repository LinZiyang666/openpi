"""R3 ideation B: closed-loop decision logs (r02_g50 pure-cache arms) -- does the judge's confidence know which
episodes are failing? For each arm: join 'dec' rows with the 'episode' rows (uid, attempt) of the same server log.
  * would-be MISS rates at matched hit rates (tau = arm's own conf quantile): in failed vs successful episodes
  * AUROC(conf, decision in a failed episode); episode-level AUROC of fraction-below-tau vs failure
  * first would-be MISS position in failed episodes (fraction of the episode elapsed)
  * for AWM arms: the same with a V7-like z-sum of extras (d1_rel, disp5, dst, still, overtime, lag, terminal)
Read-only; outputs cl_logs.json + stdout tables.
"""
import json, glob, pathlib, sys
import numpy as np

ROOT = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r02_g50/runs")
STORE = pathlib.Path("/dev/shm/offline_search_store")
OUT = pathlib.Path(__file__).resolve().parent
ARMS = ["oscl50_p_sp_cl0", "oscl50_p_sp_cl1", "oscl50_p_sp_cl2", "oscl50_p_l10_cl0", "oscl50_p_l10_cl1", "oscl50_p_l10_cl2"]
SUITE = {"sp": "spatial", "l10": "l10"}
HR = [0.3, 0.5, 0.7, 0.9]


def auroc(score, label):
    """AUROC of score for label==True (higher score -> more likely True)."""
    s = np.asarray(score, float); y = np.asarray(label, bool)
    if y.all() or (~y).all():
        return float("nan")
    o = np.argsort(s, kind="stable"); r = np.empty(s.size); r[o] = np.arange(1, s.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_arm(arm):
    model = "pi05" if "_p_" in arm else "groot"
    suite = SUITE[arm.split("_")[2]]
    decs, eps = [], {}
    for f in sorted(glob.glob(str(ROOT / arm / "server_*" / "decisions_*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d.get("ev") == "dec":
                decs.append(d)
            elif d.get("ev") == "episode":
                eps[(d["uid"], d["attempt"])] = d
    # keep decisions of accepted (final) attempts only: the episode row exists
    rows = [d for d in decs if (d["uid"], d["attempt"]) in eps]
    return model, suite, rows, eps


def lib_tables(model, suite):
    d = STORE / "library" / f"{model}_{suite}" / "current"
    L = {k: np.asarray(np.load(d / f"{k}.npy")) for k in ("step", "ep_len", "task_id", "episode", "next")}
    med = {}
    for t in np.unique(L["task_id"]):
        m = L["task_id"] == t
        _, first = np.unique(L["episode"][m], return_index=True)
        med[int(t)] = float(np.median(L["ep_len"][m][first]))
    return L, med


def analyse(arm):
    model, suite, rows, eps = load_arm(arm)
    L, med = lib_tables(model, suite)
    n = len(rows)
    uid = np.asarray([r["uid"] for r in rows]); att = np.asarray([r["attempt"] for r in rows])
    step = np.asarray([r["step"] for r in rows]); task = np.asarray([r["task_id"] for r in rows])
    conf = np.asarray([r["conf"] for r in rows], float)
    nat = np.asarray([r.get("native_score", np.nan) for r in rows], float)
    top1 = np.asarray([r["top1"] for r in rows]); topk = [r["topk"][:5] for r in rows]
    succ = np.asarray([eps[(u, a)]["success"] for u, a in zip(uid, att)], bool)
    ndec = np.asarray([eps[(u, a)]["n_decisions"] for u, a in zip(uid, att)], int)
    ep_id = np.unique(uid, return_inverse=True)[1]
    overtime = step / np.asarray([med[int(t)] for t in task])
    lag5 = step - np.asarray([L["step"][k].mean() for k in topk])
    terminal = L["next"][top1] < 0
    ex = {}
    for k in ("d1_rel", "disp5", "dst", "still", "w_eff", "lib_step"):
        ex[k] = np.asarray([r.get("extras", {}).get(k, np.nan) for r in rows], float)
    out = {"arm": arm, "n_dec": n, "n_ep": int(len(set(uid))), "sr": float(np.mean([e["success"] for e in eps.values()])),
           "dec_per_ep_fail": float(ndec[~succ].mean()) if (~succ).any() else None, "dec_per_ep_succ": float(ndec[succ].mean())}
    judges = {"own": conf}
    if not np.isnan(nat).all():
        judges["b0_native"] = nat
    if np.isfinite(ex["d1_rel"]).any():
        # V7-like z-sum over online features (risk features; higher = more risk) -> conf = -z
        st = np.minimum(np.abs(lag5), 30)
        X = np.stack([ex["d1_rel"], ex["disp5"], ex["dst"], overtime, np.abs(lag5)], 1)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
        z = ((X - X.mean(0)) / np.maximum(X.std(0), 1e-9)).sum(1)
        judges["v7z"] = -z
        guard = terminal | ((overtime > 1) & (lag5 > 5))
        judges["v7z+g"] = np.where(guard, -np.inf, -z)
        out["guard_rate"] = float(guard.mean()); out["guard_in_fail"] = float(guard[~succ].mean()); out["guard_in_succ"] = float(guard[succ].mean())
    out["judges"] = {}
    for jn, c in judges.items():
        J = {"auroc_dec_fail": auroc(-np.where(np.isfinite(c), c, -1e9), ~succ)}
        # per-episode min conf
        epmin = np.full(ep_id.max() + 1, np.inf); np.minimum.at(epmin, ep_id, np.where(np.isfinite(c), c, -1e9))
        epsucc = np.zeros(ep_id.max() + 1, bool); epsucc[ep_id] = succ
        J["auroc_ep_minconf_fail"] = auroc(-epmin, ~epsucc)
        fin = np.isfinite(c)
        J["conf_q"] = {f"q{int(q*100)}": float(np.quantile(c[fin], q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)}
        J["guard_frac"] = float((~fin).mean())
        for h in HR:
            cc = np.where(fin, c, -1e9)
            # guarded decisions are always MISS; the threshold is set so that the TOTAL hit rate is h when possible
            tau = np.quantile(cc, 1 - h)
            miss = (cc < tau) | (~fin)
            # fraction of episode elapsed at the first would-be MISS (failed episodes)
            first = {}
            for i in np.flatnonzero(miss):
                e = ep_id[i]
                if e not in first:
                    first[e] = step[i] / max(ndec[i] - 1, 1)
            ff = [first[e] for e in np.unique(ep_id[~succ]) if e in first]
            fs = [first[e] for e in np.unique(ep_id[succ]) if e in first]
            # per-episode fraction of would-be MISS
            fr = np.bincount(ep_id, weights=miss) / np.bincount(ep_id)
            J[f"h{h:.1f}"] = {"tau": float(tau), "miss_rate": float(miss.mean()), "miss_in_fail": float(miss[~succ].mean()),
                              "miss_in_succ": float(miss[succ].mean()),
                              "fail_eps_with_miss": float(len(ff) / max((~epsucc).sum(), 1)),
                              "first_miss_pos_fail_p50": float(np.median(ff)) if ff else None,
                              "first_miss_pos_succ_p50": float(np.median(fs)) if fs else None,
                              "auroc_ep_missfrac_fail": auroc(fr, ~epsucc),
                              "share_of_misses_in_fail": float(miss[~succ].sum() / max(miss.sum(), 1)),
                              "frac_dec_in_fail": float((~succ).mean())}
        out["judges"][jn] = J
    return out


if __name__ == "__main__":
    arms = ARMS if len(sys.argv) < 2 else sys.argv[1:]
    res = {}
    for a in arms:
        try:
            res[a] = analyse(a)
        except Exception as e:  # noqa
            res[a] = {"error": repr(e)}
            continue
        o = res[a]
        print(f"\n== {a}: n_dec {o['n_dec']} n_ep {o['n_ep']} SR {o['sr']:.3f} dec/ep fail {o['dec_per_ep_fail']} succ {o['dec_per_ep_succ']:.1f}"
              + (f" guard {o['guard_rate']:.3f} (fail {o['guard_in_fail']:.3f} / succ {o['guard_in_succ']:.3f})" if 'guard_rate' in o else ""))
        for jn, J in o["judges"].items():
            print(f"  [{jn}] AUROC dec->fail {J['auroc_dec_fail']:.3f}; ep min-conf->fail {J['auroc_ep_minconf_fail']:.3f}")
            for h in HR:
                x = J[f"h{h:.1f}"]
                print(f"    h={h:.1f}: miss in fail {x['miss_in_fail']:.3f} vs succ {x['miss_in_succ']:.3f}; share of misses in failed eps {x['share_of_misses_in_fail']:.3f} (dec share {x['frac_dec_in_fail']:.3f}); "
                      f"fail eps reached {x['fail_eps_with_miss']:.2f}, first miss at {x['first_miss_pos_fail_p50']}; ep AUROC {x['auroc_ep_missfrac_fail']:.3f}")
    json.dump(res, open(OUT / "cl_logs.json", "w"), indent=1, default=float)
