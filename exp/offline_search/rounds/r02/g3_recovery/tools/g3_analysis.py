"""G3 analysis of harness outputs (read-only; the GT a_inf is read here, never inside a method).

For every base found in --res (a directory of <method>/<cell>.npz, e.g. results/r02 or a subset run) and the G3
wrappers around it (method dirs named '<prefix>__<base name>'):
  A  err per regime (step 0 / fresh = inf cells step >= 1 / stale = cache cells step >= 1) + gripper-sign-faithful err
  B  V6 blend effect: wrapper (blend only) - base on stale decisions, paired, episode-bootstrap 95 % CI; split by
     whether the blend applied, by trace-episode outcome
  C  V6 detector: flag rates on stale decisions and the base's err on flagged / unflagged decisions; recovery levels
  D  V7 confidence: AURC per regime vs the base's own confidence; pooled one-threshold AURC (fresh + stale, weights
     (1-p)/N_f and p/N_s, p = .5) with the method's own confidence vs regime-wise rank normalisation vs opt

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/g3_analysis.py --res <dir>
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import store  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
MS = [f"{m}_{s}" for m in ("pi05", "groot") for s in ("spatial", "l10")]


@functools.lru_cache(maxsize=None)
def qcell(cell):
    return store.QueryCell(ROOT, cell)


@functools.lru_cache(maxsize=None)
def sigma(key):
    return np.asarray(store.action_sigma(ROOT, key), np.float64)


def load(res, m, cell):
    p = pathlib.Path(res) / m / f"{cell}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def gs_err(d, cell):
    """gripper-sign-faithful err: dim 6 of the synthesized executed segment snapped to its sign (LIBERO applies
    sign()); decisions served as a library row keep their err."""
    e = d["err"].astype(np.float64).copy()
    if "synth_seg" not in d:
        return e
    u = d["used_synth"]
    if not u.any():
        return e
    qc = qcell(cell)
    a = np.asarray(qc.a_inf[d["row"][u]][:, :5, :7], np.float64)
    s = d["synth_seg"][u].astype(np.float64)
    s[:, :, 6] = np.where(s[:, :, 6] >= 0, 1.0, -1.0)
    sg = sigma(store.lib_key(cell))
    e[u] = np.sqrt((((s - a) / sg) ** 2).mean(axis=(1, 2)))
    return e


def aurc(err, conf, w=None):
    err = np.asarray(err, np.float64)
    o = np.lexsort((np.arange(len(conf)), -np.asarray(conf, np.float64)))
    e = err[o]
    if w is None:
        return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())
    ww = np.asarray(w, np.float64)[o]
    cw = np.cumsum(ww)
    r = np.cumsum(e * ww) / cw
    return float(np.sum(r * ww) / ww.sum())


def boot(delta, ep, reps=1000, seed=0):
    ue, inv = np.unique(ep, return_inverse=True)
    s = np.bincount(inv, weights=delta)
    n = np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, ue.size, size=(reps, ue.size))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(delta.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ep_success(cell, ep):
    return np.asarray([bool(qcell(cell).episodes[int(i)]["success"]) for i in ep])


def f3(x):
    return "  nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--p", type=float, default=0.5)
    a = ap.parse_args(argv)
    res = pathlib.Path(a.res)
    methods = sorted(p.name for p in res.iterdir() if p.is_dir() and not p.name.startswith("_"))
    bases = [m for m in methods if not any(m.startswith(x) for x in ("V6sr_", "V7dc_", "G3pt_", "G3w_"))
             and any(w.endswith("__" + m) for w in methods)]
    lines = []
    P = lines.append
    summary = {}
    for b in bases:
        wr = [m for m in methods if m.endswith("__" + b)]
        P(f"\n######## base {b}   wrappers: {', '.join(w.split('__')[0] for w in wr)}")
        summary[b] = {}
        # ---------------- A
        P("\n## A. err per regime (mean) | stale gripper-sign-faithful err;  cells " + " / ".join(MS))
        for m in [b] + wr:
            row = {"step0": [], "fresh": [], "stale": [], "stale_gs": []}
            for ms in MS:
                di, dc = load(res, m, f"{ms}_inf"), load(res, m, f"{ms}_cache")
                row["step0"].append(di["err"][di["step"] == 0].mean() if di is not None else np.nan)
                row["fresh"].append(di["err"][di["step"] > 0].mean() if di is not None else np.nan)
                row["stale"].append(dc["err"][dc["step"] > 0].mean() if dc is not None else np.nan)
                row["stale_gs"].append(gs_err(dc, f"{ms}_cache")[dc["step"] > 0].mean() if dc is not None else np.nan)
            summary[b][m] = {k: [float(x) for x in v] for k, v in row.items()}
            P(f"  {m.split('__')[0]:28s} step0 {' '.join(f3(x) for x in row['step0'])} | fresh "
              f"{' '.join(f3(x) for x in row['fresh'])} | stale {' '.join(f3(x) for x in row['stale'])} | stale-gs "
              f"{' '.join(f3(x) for x in row['stale_gs'])}")
        # ---------------- B blend effect
        P("\n## B. V6 blend effect on stale decisions (wrapper - base, paired; mean [95% episode-bootstrap CI])")
        for m in wr:
            if not m.startswith("V6sr_bl1rc0"):
                continue
            P(f"  {m.split('__')[0]}")
            for ms in MS:
                c = f"{ms}_cache"
                db, dw = load(res, b, c), load(res, m, c)
                if db is None or dw is None:
                    continue
                assert np.array_equal(db["row"], dw["row"])
                s = dw["step"] > 0
                dl = (dw["err"] - db["err"])[s]
                ep = dw["ep"][s]
                bl = dw["x_blend"][s] > 0
                src = dw["x_anchor_src"][s]
                succ = ep_success(c, ep)
                m0, lo, hi = boot(dl, ep)
                P(f"    {ms:13s} all {m0:+.4f} [{lo:+.4f},{hi:+.4f}]  blend applied {bl.mean():.2f} "
                  f"(exec-anchor {np.mean(src == 1):.2f}): {dl[bl].mean() if bl.any() else np.nan:+.4f} | "
                  f"B0-succ eps {dl[succ].mean() if succ.any() else np.nan:+.4f} fail eps "
                  f"{dl[~succ].mean() if (~succ).any() else np.nan:+.4f} | nx already in top-k "
                  f"{dw['x_nx_in'][s][bl].mean() if bl.any() else np.nan:.2f} w_nx p50 "
                  f"{np.nanmedian(dw['x_w_nx'][s][bl]) if bl.any() else np.nan:.3f}")
                summary[b].setdefault("blend", {}).setdefault(m, {})[ms] = [m0, lo, hi]
        # ---------------- C detector
        full = [m for m in wr if m.startswith("V6sr_bl1rc1")]
        for m in full:
            P(f"\n## C. V6 detector on stale decisions ({m.split('__')[0]}): frac | base err flagged / unflagged")
            for ms in MS:
                c = f"{ms}_cache"
                db, dw = load(res, b, c), load(res, m, c)
                if db is None or dw is None:
                    continue
                s = dw["step"] > 0
                e = db["err"][s]
                x = {k[2:]: dw[k][s] for k in dw if k.startswith("x_")}
                flags = {"still": x["still"] > 0, "stuck_n>=2": x["stuck_n"] >= 2, "terminal": x["terminal"] > 0,
                         "overtime>1": x["overtime"] > 1, "lag>5": x["lag"] > 5,
                         "overtime&lag": (x["overtime"] > 1) & (x["lag"] > 5), "STUCK": x["stuck"] > 0}
                parts = [f"{k} {v.mean():.3f} ({f3(e[v].mean() if v.any() else np.nan)}/{f3(e[~v].mean())})"
                         for k, v in flags.items()]
                P(f"  {ms:13s} " + "  ".join(parts))
                lv = x["level"]
                ew = dw["err"][s]
                P(f"  {'':13s} recovery level 1/2/3: {np.mean(lv == 1):.3f}/{np.mean(lv == 2):.3f}/{np.mean(lv == 3):.3f}"
                  f"  (offline err wrapper vs base on recovering decisions: {f3(ew[lv > 0].mean() if (lv > 0).any() else np.nan)}"
                  f" vs {f3(e[lv > 0].mean() if (lv > 0).any() else np.nan)}; not meaningful: trace states do not respond)"
                  f"  relaxed {np.mean(x['relaxed'] > 0):.3f}  all-stale err wrapper {ew.mean():.3f} vs base {e.mean():.3f}")
                summary[b].setdefault("flags", {}).setdefault(m, {})[ms] = {k: [float(v.mean()), float(e[v].mean()) if v.any() else None, float(e[~v].mean()) if (~v).any() else None] for k, v in flags.items()}
        # ---------------- D confidence
        P(f"\n## D. confidence AURC (stale = cache step>=1, fresh = inf step>=1, step0 = inf step 0); pooled p={a.p}")
        for m in [b] + [w for w in wr if w.startswith("V7dc_") or w.startswith("V6sr_")]:
            st, fr, s0, po, pr, opt = [], [], [], [], [], []
            for ms in MS:
                di, dc = load(res, m, f"{ms}_inf"), load(res, m, f"{ms}_cache")
                if di is None or dc is None:
                    for L in (st, fr, s0, po, pr, opt):
                        L.append(np.nan)
                    continue
                sc, sf, z0 = dc["step"] > 0, di["step"] > 0, di["step"] == 0
                st.append(aurc(dc["err"][sc], dc["confidence"][sc]))
                fr.append(aurc(di["err"][sf], di["confidence"][sf]))
                s0.append(aurc(di["err"][z0], di["confidence"][z0]))
                ef, es = di["err"][sf], dc["err"][sc]
                cf, cs = di["confidence"][sf], dc["confidence"][sc]
                w = np.r_[np.full(ef.size, (1 - a.p) / ef.size), np.full(es.size, a.p / es.size)]
                e = np.r_[ef, es]
                rf = (np.argsort(np.argsort(-cf)) + 0.5) / cf.size
                rs = (np.argsort(np.argsort(-cs)) + 0.5) / cs.size
                po.append(aurc(e, np.r_[cf, cs], w))
                pr.append(aurc(e, -np.r_[rf, rs], w))
                opt.append(aurc(e, -e, w))
            summary[b].setdefault("aurc", {})[m] = {"stale": st, "fresh": fr, "step0": s0, "pooled_own": po,
                                                    "pooled_rank": pr, "pooled_opt": opt}
            P(f"  {m.split('__')[0]:28s} stale {' '.join(f3(x) for x in st)} | fresh {' '.join(f3(x) for x in fr)} | "
              f"step0 {' '.join(f3(x) for x in s0)} | pooled own {' '.join(f3(x) for x in po)} | rank-norm "
              f"{' '.join(f3(x) for x in pr)} | opt {' '.join(f3(x) for x in opt)}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        pathlib.Path(a.out).write_text(txt + "\n")
        pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
