"""breakdown -- slice the per-decision metrics of a run.

Slices (per cell; metric-side labels only -- success / num_steps are never method inputs):
  task, third (step-third by p = step/num_steps), outcome (episode success/failure),
  grip (teacher a_inf gripper sign flips inside the executed 5 steps or vs the previous decision),
  conf_dec (confidence decile within the cell, d9 = most confident), cand (task-filtered candidate
  count bucket), good (#good candidates in the library: 0 / 1-4 / 5-19 / >=20, from the coverage cache).
Plus a per-arm pooled table (inf vs cache) across the cells present.

Per slice: n, share, err mean/p50, indist (err <= floor median), regret (err - oracle_err),
orc_hit (top1 == oracle_row), grip_mis, phase_err, mean confidence, AURC (within the slice).
Cross-checks run oracle_err against the coverage cache (flags candidate-set or metric mismatches).

  python -m exp.offline_search.profile.breakdown RUN_OR_METHOD_DIR [--method M] [--cells all] [--slices task,third,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import common as C


def slice_metrics(rc: C.RunCell, m: np.ndarray, thr: np.ndarray, n_all: int, tie_aware=None) -> dict:
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    err = rc["err"][m].astype(np.float64)
    d = {"n": n, "share": n / max(n_all, 1), "err": float(np.nanmean(err)), "err_p50": float(np.nanmedian(err)),
         "indist": float(np.mean(err <= thr[m]))}
    if rc.has("oracle_err"):
        o = rc["oracle_err"][m].astype(np.float64)
        d["oracle"] = float(np.nanmean(o))
        d["regret"] = float(np.nanmean(err - o))
    if rc.has("oracle_row") and rc.has("top1"):
        d["orc_hit"] = float(np.mean(rc["top1"][m] == rc["oracle_row"][m]))
    for f, k in (("grip_mis", "grip"), ("phase_err", "phase")):
        if rc.has(f):
            d[k] = float(np.nanmean(rc[f][m]))
    if rc.has("confidence"):
        c = rc["confidence"][m].astype(np.float64)
        d["conf"] = float(np.nanmean(c)) if np.isfinite(c).any() else np.nan
        d["aurc"] = C.aurc(c, err, tie_aware)
    if rc.has("used_synth"):
        d["synth"] = float(np.mean(rc["used_synth"][m]))
    return d


def cell_context(rc: C.RunCell, qc: C.QueryCell, root, floor_mode="third"):
    """-> thr[n], n_cand[n] | None, n_good[n] | None, notes (all aligned to rc positions).
    thr = harness floor median of the decision's step bin (same as the runner's `indist`). n_cand / n_good come
    from the coverage cache of the run's library (library current for mixed runs); the run's oracle_err (harness:
    min over library current) is cross-checked against coverage_current."""
    rows = rc.rows
    notes = []
    fl = C.load_floor(root, qc.ms)
    thr = fl.thr(qc.third[rows], floor_mode) if fl is not None else np.full(rows.size, np.nan)
    if fl is None:
        notes.append("no floor available: indist is NaN")
    cov_cur = C.load_coverage(root, "current")
    lname = "current" if rc.mixed else rc.library
    cov = cov_cur if lname == "current" else C.load_coverage(root, lname)
    n_cand = n_good = None
    if cov is not None and cov.has(qc.cell):
        n_cand = cov.get(qc.cell, "n_cand")[rows]
        n_good = cov.get(qc.cell, "n_good")[rows]
    else:
        lib = C.open_library(root, qc.ms, lname)
        if lib is not None and lib.has("task_id"):
            cnt = {int(t): lib.task_rows(t).size for t in np.unique(qc.task_id)}
            n_cand = np.array([cnt[int(t)] for t in qc.task_id[rows]])
        notes.append(f"no coverage cache for library {lname!r}: 'good' slice unavailable (run `coverage --lib {lname}`)")
    if rc.mixed:
        notes.append(f"mixed-library run {rc.lib_names}: cand/good slices use library current")
    if cov_cur is not None and cov_cur.has(qc.cell) and rc.has("oracle_err"):
        dlt = rc["oracle_err"].astype(np.float64) - cov_cur.get(qc.cell, "oracle_err")[rows]
        fin = np.isfinite(dlt)
        mx = float(np.abs(dlt[fin]).max()) if fin.any() else 0.0
        if mx > 1e-4:
            notes.append(f"oracle_err differs from coverage_current (max |d|={mx:.3g}; run>cov on "
                         f"{int((dlt[fin] > 1e-4).sum())}, run<cov on {int((dlt[fin] < -1e-4).sum())} decisions): "
                         "metric / sigma / candidate definition mismatch -- check before trusting regret slices")
    return thr, n_cand, n_good, notes


def cell_breakdown(md: Path, cell: str, root, slices=None, floor_mode="third", tie_aware=None) -> dict:
    rc = C.RunCell(md, cell)
    qc = C.QueryCell(root, cell)
    rows = rc.rows
    thr, n_cand, n_good, notes = cell_context(rc, qc, root, floor_mode)
    conf = rc["confidence"] if rc.has("confidence") else None
    lab = C.decision_slices(qc, rows, conf, n_cand, n_good)
    allm = np.ones(rc.n, bool)
    out = [{"cell": cell, "slice": "all", "value": "all", **slice_metrics(rc, allm, thr, rc.n, tie_aware),
            **(C.risk_at(conf, rc["err"], tie_aware=tie_aware) if conf is not None else {})}]
    for name in C.SLICE_ORDER:
        if name not in lab or (slices and name not in slices):
            continue
        vals = sorted(np.unique(lab[name]), key=lambda v: C.slice_sort_key(name, v))
        for v in vals:
            out.append({"cell": cell, "slice": name, "value": v, **slice_metrics(rc, lab[name] == v, thr, rc.n, tie_aware)})
    return {"cell": cell, "library": rc.library, "rows": out, "notes": notes,
            "arm_pool": {"err": rc["err"].astype(np.float64), "ind": rc["err"] <= thr,
                         "regret": (rc["err"] - rc["oracle_err"]).astype(np.float64) if rc.has("oracle_err") else None,
                         "aurc": out[0].get("aurc")}}


COLS = [("slice", "slice"), ("value", "value"), ("n", "n"), ("share", "share", ".2f"), ("err", "err"), ("err_p50", "p50"),
        ("indist", "indist"), ("regret", "regret"), ("orc_hit", "orc_hit"), ("grip", "grip"), ("phase", "phase"),
        ("conf", "conf", ".3g"), ("aurc", "AURC")]
ALL_COLS = [("cell", "cell"), ("n", "n"), ("err", "err"), ("err_p50", "p50"), ("indist", "indist"), ("oracle", "oracle"),
            ("regret", "regret"), ("orc_hit", "orc_hit"), ("grip", "grip"), ("phase", "phase"), ("aurc", "AURC"),
            ("risk@30", "r@30"), ("risk@50", "r@50"), ("risk@70", "r@70"), ("risk@90", "r@90")]


def _job(a):
    return cell_breakdown(*a)


def run(run_path, method=None, root=C.DEFAULT_ROOT, cells="all", slices=None, floor_mode="third", out=None,
        quiet=False, tie_aware=False, procs=None) -> dict:
    root = Path(root)
    res = {}
    slices = slices.split(",") if isinstance(slices, str) else slices
    mds = C.method_dirs(run_path, method)
    jobs = [(md, c, root, slices, floor_mode, tie_aware) for md in mds for c in C.run_cells(md, C.expand_cells(cells))]
    done = C.pmap(_job, jobs, procs)                          # one pool over every (method, cell)
    for md in mds:
        per = [r for (m, *_), r in zip(jobs, done) if m == md]
        txt = [f"## breakdown {md.name}  (store {root})"]
        txt.append(C.md_table([p["rows"][0] for p in per], ALL_COLS, "per cell (all decisions)"))
        pool = []
        for arm in C.ARMS:
            ps = [p for p in per if C.parse_cell(p["cell"])[2] == arm]
            if not ps:
                continue
            e = np.concatenate([p["arm_pool"]["err"] for p in ps])
            ind = np.concatenate([p["arm_pool"]["ind"] for p in ps])
            rg = [p["arm_pool"]["regret"] for p in ps if p["arm_pool"]["regret"] is not None]
            au = [p["arm_pool"]["aurc"] for p in ps if p["arm_pool"]["aurc"] is not None]
            pool.append({"arm": arm, "cells": len(ps), "n": e.size, "err": float(e.mean()), "indist": float(ind.mean()),
                         "regret": float(np.concatenate(rg).mean()) if rg else None,
                         "aurc_mean_of_cells": float(np.mean(au)) if au else None})
        txt.append("")
        txt.append(C.md_table(pool, [("arm", "arm"), ("cells", "cells"), ("n", "n"), ("err", "err"), ("indist", "indist"),
                                     ("regret", "regret"), ("aurc_mean_of_cells", "AURC(mean of cells)")], "pooled by arm"))
        for p in per:
            txt.append("")
            txt.append(C.md_table(p["rows"][1:], COLS, f"{p['cell']}  (library {p['library']})"))
            for n_ in p["notes"]:
                txt.append(f"> note: {n_}")
        s = "\n".join(txt)
        if not quiet:
            print(s)
        rdir = C.report_dir(root, "breakdown", md.name, out, multi=len(mds) > 1)
        flat = [r for p in per for r in p["rows"]]
        C.write_json(rdir / "breakdown.json", {"method": md.name, "arm_pool": pool, "rows": flat,
                                                "notes": {p["cell"]: p["notes"] for p in per}})
        C.write_csv(rdir / "breakdown.csv", flat)
        (rdir / "breakdown.md").write_text(s + "\n")
        if not quiet:
            print(f"\n[breakdown] {md.name}: {len(per)} cells -> {rdir}")
        res[md.name] = {"rows": flat, "arm_pool": pool, "report_dir": str(rdir)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--method", default=None, help="method name(s) under a run root (default: all)")
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--cells", default="all")
    ap.add_argument("--slices", default=None, help="comma subset of " + ",".join(C.SLICE_ORDER))
    ap.add_argument("--floor-mode", default="third", choices=["third", "overall"])
    ap.add_argument("--tie-aware", action="store_true", help="AURC averaged over tie order (default: harness stable order)")
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    ap.add_argument("--out", default=None, help="report dir (one subdir per method when several are reported)")
    a = ap.parse_args(argv)
    run(a.run, a.method, a.root, a.cells, a.slices, a.floor_mode, a.out, tie_aware=a.tie_aware, procs=a.procs)


if __name__ == "__main__":
    sys.exit(main())
