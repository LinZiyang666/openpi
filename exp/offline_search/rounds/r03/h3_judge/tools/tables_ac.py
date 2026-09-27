"""Tables (a) and (c) from the harness runs of the MixedJudge variants (tools/run_all.sh -> <derived>/runs/<name>/).

(a) per variant x model-suite: stale / fresh / pooled AURC of the confidence; forced-MISS rate per regime and per
    reason; err on forced vs unforced decisions; accepted err at target hit rates h = .5 / .7 with the mixture
    weighting of ideation B section F4 (after-HIT states = cache cell decisions with step >= 1 weighted h, after-MISS
    states = inf cell decisions with step >= 1 weighted 1-h; forced decisions count as -inf confidence, i.e. they
    consume MISS budget; tau = the weighted (1-h) quantile), plus the offline err identity with the base.
(c) threshold-init table: tau0 (confidence units = -pred_err) per cell and h, realized stale / fresh acceptance,
    predicted forced-MISS share (of all decisions, per regime), accepted err / bad / grip_mis at tau0.
Also: how much the log-replay approximation of the V7 confidence (vis -> mean; dnn -> mean) costs in stale AURC.

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h3_judge/tools/tables_ac.py
Writes <derived>/tables/tables_ac.json + tables_ac.md and prints the markdown.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

D = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge")
RUNS = D / "runs"
OUT = D / "tables"
MS = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
HR = (0.5, 0.7)
BIN = ["early", "mid", "late"]
REASON = {0: "none", 1: "stuck", 2: "terminal", 3: "overtime", 4: "noprog", 5: "disp", 6: "grip", 7: "burst"}
VARIANTS = ["MXJ_g0_ev0__AWM_joint_cur_fcur_kr5", "MXJ_g1_ev0__AWM_joint_cur_fcur_kr5",
            "MXJ_g1_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5", "MXJ_g0_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5",
            "MXJ_g1_ev0__AWM_joint_big_fbig", "MXJ_g1_evDG_b2_rm0p1__AWM_joint_big_fbig"]
BASES = {"AWM_joint_cur_fcur_kr5": "AWM_joint_cur_fcur_kr5"}


def aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())


def wq(conf, w, h):
    """Weighted (1-h) quantile: the confidence at which the accepted (>= tau) weight is h."""
    o = np.argsort(-conf, kind="stable")
    cw = np.cumsum(w[o])
    k = min(int(np.searchsorted(cw, h, side="left")), conf.size - 1)
    return float(conf[o][k])


def load(d, cell):
    z = np.load(d / f"{cell}.npz", allow_pickle=True)
    j = json.load(open(d / f"{cell}.json"))
    fl = j.get("floor") or {}
    if fl.get("available") and "p90" in fl:
        p90 = np.asarray([fl["p90"].get(b, fl["p90"].get("all", np.nan)) for b in BIN])[z["bin"].astype(int)]
        bad = z["err"] > p90
    else:
        bad = np.full(z["err"].shape, np.nan)
    g = lambda k: z[f"x_{k}"].astype(np.float64) if f"x_{k}" in z.files else np.full(z["err"].shape, np.nan)
    force = np.nan_to_num(g("os_force_miss"), nan=0.0) > 0
    reason = np.nan_to_num(g("os_reason"), nan=0.0).astype(int)
    return {"conf": z["confidence"].astype(np.float64), "err": z["err"].astype(np.float64), "bad": bad,
            "grip": z["grip_mis"].astype(np.float64), "step": z["step"].astype(int), "force": force, "reason": reason,
            "reg": g("regime"), "phase": np.nan_to_num(g("os_phase"), nan=0.0).astype(int), "zsum": g("zsum"),
            "vis": g("vis"), "dnn": g("dnn"), "pred": g("pred_err"), "conf_raw": g("os_conf_raw"),
            "ms": (j.get("timing") or {}).get("ms_per_query"), "fit_s": (j.get("fit") or {}).get("fit_s"),
            "n": int(z["err"].size), "json": j}


def per_reason(reason, force, mask, err=None):
    tot = max(int(mask.sum()), 1)
    out = {}
    for r in range(1, 8):
        m = (reason == r) & force & mask
        if m.any():
            out[REASON[r]] = {"rate": float(m.sum() / tot), "err": float(err[m].mean()) if err is not None else None}
    return out


def cal_of(v, ms):
    """Calibration mu/sd/w of the stale map from the fit json in the run's scratch dir (for the approx-V7 check)."""
    for p in (RUNS / v / "scratch" / f"{ms}_cache").glob("mxj_fit_*.json"):
        info = json.load(open(p)).get("calibration", {}).get("2")
        if info:
            return info
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tau_rows, approx_rows = [], [], []
    for v in VARIANTS:
        d = RUNS / v
        if not d.exists():
            continue
        for ms in MS:
            try:
                I, C = load(d, f"{ms}_inf"), load(d, f"{ms}_cache")
            except FileNotFoundError:
                continue
            st, fr, s0 = C["reg"] > 0, I["reg"] > 0, I["reg"] == 0
            r = {"variant": v, "ms": ms, "n_cache": C["n"], "n_inf": I["n"], "err_cache": float(C["err"].mean()),
                 "err_inf": float(I["err"].mean()), "aurc_stale": aurc(C["conf"][st], C["err"][st]),
                 "aurc_fresh": aurc(I["conf"][fr], I["err"][fr]), "aurc_step0": aurc(I["conf"][s0], I["err"][s0]),
                 "force_stale": float(C["force"][st].mean()), "force_fresh": float(I["force"][fr].mean()),
                 "force_step0": float(I["force"][s0].mean()),
                 "reasons_stale": per_reason(C["reason"], C["force"], st, C["err"]),
                 "reasons_fresh": per_reason(I["reason"], I["force"], fr, I["err"]),
                 "err_forced_stale": float(C["err"][st & C["force"]].mean()) if (st & C["force"]).any() else None,
                 "err_unforced_stale": float(C["err"][st & ~C["force"]].mean()) if (st & ~C["force"]).any() else None,
                 "err_forced_fresh": float(I["err"][fr & I["force"]].mean()) if (fr & I["force"]).any() else None,
                 "err_unforced_fresh": float(I["err"][fr & ~I["force"]].mean()) if (fr & ~I["force"]).any() else None,
                 "phase_return_stale": float((C["phase"][st] == 2).mean()), "phase_return_fresh": float((I["phase"][fr] == 2).mean()),
                 "ms_per_query": {"cache": C["ms"], "inf": I["ms"]}, "fit_s": {"cache": C["fit_s"], "inf": I["fit_s"]}}
            # pooled one-scale AURC (p = .5)
            pc = np.r_[I["conf"][fr], C["conf"][st]]
            pe = np.r_[I["err"][fr], C["err"][st]]
            r["aurc_pooled"] = aurc(pc, pe)
            # mixture thresholds (F4): forced -> -inf
            ci = np.where(I["force"], -1e9, I["conf"])[fr]
            cc = np.where(C["force"], -1e9, C["conf"])[st]
            ei, ec = I["err"][fr], C["err"][st]
            bi, bc = I["bad"][fr], C["bad"][st]
            gi, gc = I["grip"][fr], C["grip"][st]
            fi, fc = I["force"][fr], C["force"][st]
            for h in HR:
                w = np.r_[np.full(ci.size, (1 - h) / ci.size), np.full(cc.size, h / cc.size)]
                tau = wq(np.r_[ci, cc], w, h)
                ai, ac = (ci >= tau) & ~fi, (cc >= tau) & ~fc
                wi, wc = (1 - h) * ai.mean(), h * ac.mean()
                mix = lambda a, b: float((wi * a[ai].mean() + wc * b[ac].mean()) / max(wi + wc, 1e-12)) if (ai.any() and ac.any()) else None
                r[f"h{h}"] = {"tau0": tau, "tau0_pred_err": -tau, "rate": float(wi + wc), "rate_stale": float(ac.mean()),
                              "rate_fresh": float(ai.mean()), "err": mix(ei, ec), "bad": mix(bi, bc) if np.isfinite(bi).all() else None,
                              "grip": mix(gi, gc), "err_stale": float(ec[ac].mean()) if ac.any() else None,
                              "err_fresh": float(ei[ai].mean()) if ai.any() else None,
                              "forced_share_stale": float(fc.mean()), "forced_share_fresh": float(fi.mean()),
                              "thr_miss_stale": float(((cc < tau) & ~fc).mean()), "thr_miss_fresh": float(((ci < tau) & ~fi).mean())}
                tau_rows.append({"variant": v, "ms": ms, "h": h, **r[f"h{h}"]})
            rows.append(r)
            # approx-V7 cost: stale AURC of z without vis / without vis+dnn (ranking within the regime = z's)
            info = cal_of(v, ms)
            if info and v.startswith("MXJ_g0_ev0"):
                names, mu, sd, wv = info["feats"], np.asarray(info["mu"]), np.asarray(info["sd"]), np.asarray(info["w"])
                z = C["zsum"][st]
                e = C["err"][st]
                def drop(keys):
                    zz = z.copy()
                    for k in keys:
                        i = names.index(k)
                        x = C[k][st] if k in ("vis", "dnn") else None
                        x = np.where(np.isfinite(x), x, mu[i])
                        zz = zz - wv[i] * (x - mu[i]) / sd[i]
                    return zz
                approx_rows.append({"ms": ms, "aurc_full_z": aurc(z, e), "aurc_no_vis": aurc(drop(["vis"]), e),
                                    "aurc_no_vis_dnn": aurc(drop(["vis", "dnn"]), e),
                                    "corr_z_novis": float(np.corrcoef(z, drop(["vis"]))[0, 1]),
                                    "corr_z_novis_nodnn": float(np.corrcoef(z, drop(["vis", "dnn"]))[0, 1])})
    # base err identity
    base_rows = []
    for bname in BASES.values():
        d = RUNS / bname
        if not d.exists():
            continue
        for ms in MS:
            for arm in ("cache", "inf"):
                try:
                    zb = np.load(d / f"{ms}_{arm}.npz")
                    zj = np.load(RUNS / VARIANTS[2] / f"{ms}_{arm}.npz")
                    jb, jj = json.load(open(d / f"{ms}_{arm}.json")), json.load(open(RUNS / VARIANTS[2] / f"{ms}_{arm}.json"))
                except FileNotFoundError:
                    continue
                base_rows.append({"ms": ms, "arm": arm, "err_base": float(zb["err"].mean()), "err_judge": float(zj["err"].mean()),
                                  "err_identical": bool(np.array_equal(zb["err"], zj["err"])),
                                  "synth_identical": bool(np.array_equal(np.nan_to_num(zb["synth_seg"]), np.nan_to_num(zj["synth_seg"]))),
                                  "ms_base": (jb.get("timing") or {}).get("ms_per_query"), "ms_judge": (jj.get("timing") or {}).get("ms_per_query"),
                                  "fit_s_base": (jb.get("fit") or {}).get("fit_s"), "fit_s_judge": (jj.get("fit") or {}).get("fit_s")})
    json.dump({"rows": rows, "tau": tau_rows, "approx": approx_rows, "base": base_rows}, open(OUT / "tables_ac.json", "w"),
              indent=1, default=float)
    md = ["## (a) offline: AURC, forced-MISS rates, accepted err at h = .5 / .7 (F4 mixture; forced = -inf)", "",
          "| variant | cell | AURC stale / fresh / pooled | forced stale / fresh / step0 | err forced vs unforced (stale) | err forced vs unforced (fresh) | h=.5: tau0(pred_err) rate(st/fr) err | h=.7: tau0 rate(st/fr) err | ms/q (cache) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: "-" if x is None else f"{x:.3f}"
        a, b = r["h0.5"], r["h0.7"]
        tp = lambda x: "capped" if abs(x["tau0"]) > 1e8 else f"{x['tau0_pred_err']:.3f}"
        md.append(f"| {r['variant'].replace('__AWM_joint_', ' / ')} | {r['ms']} | {r['aurc_stale']:.3f} / {r['aurc_fresh']:.3f} / {r['aurc_pooled']:.3f} | "
                  f"{r['force_stale']:.3f} / {r['force_fresh']:.3f} / {r['force_step0']:.3f} | {f(r['err_forced_stale'])} vs {f(r['err_unforced_stale'])} | "
                  f"{f(r['err_forced_fresh'])} vs {f(r['err_unforced_fresh'])} | "
                  f"{tp(a)} {a['rate_stale']:.2f}/{a['rate_fresh']:.2f} {f(a['err'])} | {tp(b)} {b['rate_stale']:.2f}/{b['rate_fresh']:.2f} {f(b['err'])} | "
                  f"{r['ms_per_query']['cache'] if r['ms_per_query']['cache'] is None else round(r['ms_per_query']['cache'], 2)} |")
    md += ["", "### forced-MISS share per reason: rate (err of those decisions) -- stale = cache cells step >= 1; fresh = inf cells step >= 1", "",
           "| variant | cell | stale reasons | fresh reasons |", "|---|---|---|---|"]
    for r in rows:
        fmt = lambda d: ", ".join(f"{k} {v['rate']:.3f} ({v['err']:.2f})" for k, v in d.items()) or "-"
        md.append(f"| {r['variant'].replace('__AWM_joint_', ' / ')} | {r['ms']} | {fmt(r['reasons_stale'])} | {fmt(r['reasons_fresh'])} |")
    md += ["", "## (c) threshold-init table: tau0 in confidence units (= -pred_err; the plugin's quantile controller starts here)", "",
           "| variant | cell | h | tau0 (confidence) | realized rate stale / fresh | forced share stale / fresh | threshold-MISS share stale / fresh | accepted err (mix) | accepted bad | accepted grip_mis |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tau_rows:
        f = lambda x: "-" if x is None else f"{x:.3f}"
        t0 = "capped (forced > 1-h)" if abs(t["tau0"]) > 1e8 else f"{t['tau0']:.4f}"
        md.append(f"| {t['variant'].replace('__AWM_joint_', ' / ')} | {t['ms']} | {t['h']} | {t0} | {t['rate_stale']:.2f} / {t['rate_fresh']:.2f} | "
                  f"{t['forced_share_stale']:.3f} / {t['forced_share_fresh']:.3f} | {t['thr_miss_stale']:.3f} / {t['thr_miss_fresh']:.3f} | {f(t['err'])} | {f(t['bad'])} | {f(t['grip'])} |")
    if approx_rows:
        md += ["", "### log-replay approximation cost of the V7 confidence (stale AURC of the z-sum ranking)", "",
               "| cell | full z | z without vis | z without vis, dnn | corr(z, no vis) | corr(z, no vis/dnn) |", "|---|---|---|---|---|---|"]
        for a in approx_rows:
            md.append(f"| {a['ms']} | {a['aurc_full_z']:.3f} | {a['aurc_no_vis']:.3f} | {a['aurc_no_vis_dnn']:.3f} | {a['corr_z_novis']:.3f} | {a['corr_z_novis_nodnn']:.3f} |")
    if base_rows:
        md += ["", "### served action identity with the base (full variant vs AWM kr5 current) and timing", "",
               "| cell | arm | err base | err judge | err identical | synth identical | ms/q base | ms/q judge | fit_s base | fit_s judge |", "|---|---|---|---|---|---|---|---|---|---|"]
        for b in base_rows:
            g = lambda x: "-" if x is None else f"{x:.2f}"
            md.append(f"| {b['ms']} | {b['arm']} | {b['err_base']:.4f} | {b['err_judge']:.4f} | {b['err_identical']} | {b['synth_identical']} | "
                      f"{g(b['ms_base'])} | {g(b['ms_judge'])} | {g(b['fit_s_base'])} | {g(b['fit_s_judge'])} |")
    (OUT / "tables_ac.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
