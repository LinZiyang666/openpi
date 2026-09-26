"""compare -- paired comparison of two runs A vs B on their common cells / decisions.

Per cell: decisions are aligned by query-store `row`; d = err_A - err_B (negative = A better).
  win/tie/loss rates (|d| <= --tie counts as tie), mean d with an episode-level bootstrap 95% CI
  (resample episodes with replacement, --reps 2000, fixed --seed), AURC_A - AURC_B with the same
  bootstrap, indist_A - indist_B with CI, top-1 flip rate (only when both use the same library),
  per-slice mean d / win / loss (breakdown slices), and the most divergent decisions both ways
  (ready-made `explain --row` commands).

  python -m exp.offline_search.profile.compare RUN_A RUN_B [--method-a M] [--method-b M] [--cells all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import breakdown as BD
from . import common as C


def _aurc_boot(conf: np.ndarray, err: np.ndarray, bs: C.EpisodeBootstrap, tie_aware=None, lo: int = 0,
               hi: int | None = None) -> np.ndarray:
    """AURC on bootstrap replicates lo..hi-1 (default all): sort once, expand rows by their episode multiplicity."""
    order = C.conf_order(conf)
    cs = np.where(np.isfinite(conf), conf, -np.inf)[order]
    rs = np.asarray(err, np.float64)[order]
    inv_sorted = bs.inv[order]
    hi = len(bs.W) if hi is None else hi
    out = np.empty(hi - lo)
    for k in range(lo, hi):
        w = bs.W[k][inv_sorted].astype(np.int64)
        _, r = C.sorted_risk_curve(np.repeat(cs, w), np.repeat(rs, w), tie_aware)
        out[k - lo] = r.mean()
    return out


def _boot_chunk(job) -> tuple[np.ndarray, np.ndarray]:
    """Replicates [lo, hi) of the AURC bootstrap of A and B (the draws are regenerated from the same seed, so the
    concatenation of all chunks equals the single-process result exactly)."""
    ca, ea, cb, eb, ep, reps, seed, lo, hi, tie_aware = job
    bs = C.EpisodeBootstrap(ep, reps, seed)
    return _aurc_boot(ca, ea, bs, tie_aware, lo, hi), _aurc_boot(cb, eb, bs, tie_aware, lo, hi)


def _ci(est, reps, alpha=0.05) -> dict:
    return {"est": float(est), "lo": float(np.quantile(reps, alpha / 2)), "hi": float(np.quantile(reps, 1 - alpha / 2)),
            "p_le0": float((reps <= 0).mean())}


def compare_cell(job) -> dict:
    mda, mdb, cell, root, reps, seed, tie, top, floor_mode, tie_aware = job
    A, B = C.RunCell(mda, cell), C.RunCell(mdb, cell)
    qc = C.QueryCell(root, cell)
    rows, ia, ib = np.intersect1d(A.rows, B.rows, return_indices=True)
    ea, eb = A["err"][ia].astype(np.float64), B["err"][ib].astype(np.float64)
    d = ea - eb
    ep = qc.ep[rows]
    bs = C.EpisodeBootstrap(ep, reps, seed)
    res = {"cell": cell, "n": int(rows.size), "n_only_a": int(A.n - rows.size), "n_only_b": int(B.n - rows.size),
           "lib_a": A.library, "lib_b": B.library, "err_a": float(ea.mean()), "err_b": float(eb.mean()),
           "win": float((d < -tie).mean()), "tie": float((np.abs(d) <= tie).mean()), "loss": float((d > tie).mean())}
    res["d_mean"] = bs.mean_ci(d)
    res["d_median"] = float(np.median(d))
    # indistinguishable fraction (same floor thresholds for both, from A's context)
    thr, n_cand, n_good, notes = BD.cell_context(A, qc, root, floor_mode)
    thr = thr[ia]
    res["indist_a"], res["indist_b"] = float((ea <= thr).mean()), float((eb <= thr).mean())
    res["d_indist"] = bs.mean_ci((ea <= thr).astype(float) - (eb <= thr).astype(float))
    if A.has("confidence") and B.has("confidence"):
        ca, cb = A["confidence"][ia].astype(np.float64), B["confidence"][ib].astype(np.float64)
        aa, ab = C.aurc(ca, ea, tie_aware), C.aurc(cb, eb, tie_aware)
        res["aurc_a"], res["aurc_b"] = aa, ab
        res["_boot"] = (ca, ea, cb, eb, ep)      # dAURC bootstrap runs in a second, chunked pool (see run)
        res["conf_spearman_ab"] = float(C.spearman_rows(ca[None], cb[None])[0])
    if A.has("top1") and B.has("top1"):  # a flip = different (library, row) of the top-1
        la = np.array(A.lib_names)[A.lib_code[ia]]
        lb = np.array(B.lib_names)[B.lib_code[ib]]
        res["flip"] = float(((A["top1"][ia] != B["top1"][ib]) | (la != lb)).mean())
        res["lib_differs"] = float((la != lb).mean())
    # slices (labels from the query side; confidence deciles of A)
    lab = C.decision_slices(qc, rows, A["confidence"][ia] if A.has("confidence") else None,
                            n_cand[ia] if n_cand is not None else None, n_good[ia] if n_good is not None else None)
    sl = []
    for name in C.SLICE_ORDER:
        if name not in lab:
            continue
        for v in sorted(np.unique(lab[name]), key=lambda x: C.slice_sort_key(name, x)):
            m = lab[name] == v
            sl.append({"cell": cell, "slice": name, "value": v, "n": int(m.sum()), "err_a": float(ea[m].mean()),
                       "err_b": float(eb[m].mean()), "d": float(d[m].mean()), "win": float((d[m] < -tie).mean()),
                       "loss": float((d[m] > tie).mean())})
    res["slices"] = sl
    # most divergent decisions
    def rec(j):
        return {"row": int(rows[j]), "ep": int(ep[j]), "uid": qc.episodes[int(ep[j])].get("uid"), "step": int(qc.step[rows[j]]),
                "err_a": float(ea[j]), "err_b": float(eb[j]), "d": float(d[j]),
                "top1_a": int(A["top1"][ia[j]]) if A.has("top1") else None,
                "top1_b": int(B["top1"][ib[j]]) if B.has("top1") else None,
                "oracle": float(A["oracle_err"][ia[j]]) if A.has("oracle_err") else None}
    o = np.argsort(d, kind="stable")
    res["a_much_better"] = [rec(j) for j in o[:top] if d[j] < -tie]
    res["a_much_worse"] = [rec(j) for j in o[::-1][:top] if d[j] > tie]
    res["notes"] = notes
    return res


def run(run_a, run_b, method_a=None, method_b=None, root=C.DEFAULT_ROOT, cells="all", reps=2000, seed=0, tie=1e-6,
        top=10, procs=None, floor_mode="third", out=None, quiet=False, tie_aware=False) -> dict:
    root = Path(root)
    ma, mb = C.method_dirs(run_a, method_a), C.method_dirs(run_b, method_b)
    if len(ma) != 1 or len(mb) != 1:
        raise SystemExit(f"need exactly one method on each side (A: {[m.name for m in ma]}, B: {[m.name for m in mb]})")
    mda, mdb = ma[0], mb[0]
    want = C.expand_cells(cells)
    common = [c for c in C.run_cells(mda, want) if c in C.run_cells(mdb, want)]
    jobs = [(str(mda), str(mdb), c, str(root), reps, seed, tie, top, floor_mode, tie_aware) for c in common]
    res = C.pmap(compare_cell, jobs, procs)
    # phase 2: dAURC bootstrap split into replicate chunks over all cores
    P = C.DEFAULT_PROCS if procs is None else max(1, int(procs))
    nch = max(1, min(reps // 50 or 1, -(-P // max(len(res), 1))))
    bounds = np.linspace(0, reps, nch + 1).astype(int)
    bjobs, owner = [], []
    for i, r in enumerate(res):
        if "_boot" in r:
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                bjobs.append((*r["_boot"], reps, seed, int(lo), int(hi), tie_aware))
                owner.append(i)
    bres = C.pmap(_boot_chunk, bjobs, procs)
    for i, r in enumerate(res):
        if "_boot" in r:
            parts = [b for o, b in zip(owner, bres) if o == i]
            ra = np.concatenate([p_[0] for p_ in parts])
            rb = np.concatenate([p_[1] for p_ in parts])
            r["d_aurc"] = _ci(r["aurc_a"] - r["aurc_b"], ra - rb)
            del r["_boot"]
    tag = f"{mda.name}__vs__{mdb.name}"
    L = [f"## compare A={mda.name}  vs  B={mdb.name}   (d = err_A - err_B; negative = A better; "
         f"episode bootstrap {reps} reps, seed {seed})"]
    rows = []
    for r in res:
        dm, di = r["d_mean"], r["d_indist"]
        da = r.get("d_aurc")
        rows.append({"cell": r["cell"], "n": r["n"], "err_a": r["err_a"], "err_b": r["err_b"],
                     "d": f"{dm['est']:+.4f} [{dm['lo']:+.4f},{dm['hi']:+.4f}]",
                     "wtl": f"{r['win']:.2f}/{r['tie']:.2f}/{r['loss']:.2f}",
                     "aurc": f"{r['aurc_a']:.3f}/{r['aurc_b']:.3f}" if da else "-",
                     "d_aurc": f"{da['est']:+.4f} [{da['lo']:+.4f},{da['hi']:+.4f}]" if da else "-",
                     "d_ind": f"{di['est']:+.3f} [{di['lo']:+.3f},{di['hi']:+.3f}]",
                     "flip": r.get("flip")})
    L.append(C.md_table(rows, [("cell", "cell"), ("n", "n"), ("err_a", "err_A"), ("err_b", "err_B"), ("d", "d mean [95% CI]"),
                               ("wtl", "win/tie/loss"), ("aurc", "AURC A/B"), ("d_aurc", "dAURC [95% CI]"),
                               ("d_ind", "d indist [95% CI]"), ("flip", "top1 flip")], "per cell"))
    for r in res:
        L.append("")
        nmin = max(20, int(0.02 * r["n"]))  # summary line: ignore tiny slices and A's own confidence deciles
        best = sorted([s_ for s_ in r["slices"] if s_["slice"] != "conf_dec" and s_["n"] >= nmin], key=lambda s_: s_["d"])
        L.append(C.md_table(r["slices"], [("slice", "slice"), ("value", "value"), ("n", "n"), ("err_a", "err_A"),
                                          ("err_b", "err_B"), ("d", "d", "+.4f"), ("win", "win"), ("loss", "loss")],
                            f"{r['cell']} by slice"))
        if best:
            L.append("> A gains most on: " + ", ".join(f"{s['slice']}={s['value']} ({s['d']:+.3f}, n={s['n']})" for s in best[:3])
                     + " | loses most on: " + ", ".join(f"{s['slice']}={s['value']} ({s['d']:+.3f}, n={s['n']})" for s in best[::-1][:3]))
        for key, title in (("a_much_worse", "A much worse"), ("a_much_better", "A much better")):
            if r[key]:
                L.append(f"{title}: " + "; ".join(f"row {x['row']} (ep {x['ep']} s{x['step']}: {x['err_a']:.3f} vs {x['err_b']:.3f})"
                                                  for x in r[key][:5]))
        if r["a_much_worse"]:
            rr = " ".join(f"--row {x['row']}" for x in r["a_much_worse"][:3])
            L.append(f"  explain: python -m exp.offline_search.profile.explain {mda} --cell {r['cell']} --root {root} {rr}")
        for n_ in r["notes"]:
            L.append(f"> note: {n_}")
    s = "\n".join(L)
    if not quiet:
        print(s)
    rdir = C.report_dir(root, "compare", tag, out)
    C.write_json(rdir / "compare.json", {"a": str(mda), "b": str(mdb), "reps": reps, "seed": seed, "tie": tie, "cells": res})
    C.write_csv(rdir / "compare_cells.csv", [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} |
                                             {"d_mean": r["d_mean"]["est"], "d_lo": r["d_mean"]["lo"], "d_hi": r["d_mean"]["hi"],
                                              "d_aurc": (r.get("d_aurc") or {}).get("est"), "d_aurc_lo": (r.get("d_aurc") or {}).get("lo"),
                                              "d_aurc_hi": (r.get("d_aurc") or {}).get("hi"), "d_indist": r["d_indist"]["est"]}
                                             for r in res])
    C.write_csv(rdir / "compare_slices.csv", [s_ for r in res for s_ in r["slices"]])
    (rdir / "compare.md").write_text(s + "\n")
    if not quiet:
        print(f"\n[compare] {len(res)} cells -> {rdir}")
    return {"cells": res, "report_dir": str(rdir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--method-a", default=None)
    ap.add_argument("--method-b", default=None)
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--cells", default="all")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tie", type=float, default=1e-6)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    ap.add_argument("--floor-mode", default="third", choices=["third", "overall"])
    ap.add_argument("--tie-aware", action="store_true", help="AURC averaged over tie order (default: harness stable order)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.run_a, a.run_b, a.method_a, a.method_b, a.root, a.cells, a.reps, a.seed, a.tie, a.top, a.procs, a.floor_mode,
        a.out, tie_aware=a.tie_aware)


if __name__ == "__main__":
    sys.exit(main())
