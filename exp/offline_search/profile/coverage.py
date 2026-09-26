"""coverage -- LIBRARY-side ceiling profile, independent of any method.

For every query of every cell, the executed-segment err (protocol §4.2, [:5, :7] / sigma_d) to EVERY
task-filtered candidate of a library is computed (vectorized per task as a GEMM, multiprocess over
cell x library x task x 1024-query block, all cores by default). Reported per cell / task / step-third:
  oracle err distribution, #/fraction of "good" candidates (err <= teacher floor median),
  whether any good candidate exists, rank structure (err at rank 1/2/5/10/50, median candidate),
  and for library `current` where the recorded B0 pick sits in that ranking (rank, regret).
This bounds what ANY search method could gain on this library and shows where coverage is missing.

Cache: <root>/profile_cache/coverage_<lib>.npz, per cell (keys "<cell>__<field>"):
  top_rows int32[N,topn] (-1 pad), top_errs f32[N,topn] (NaN pad)  sorted ascending by err
  n_cand, n_good (err<=thr), n_good2x (err<=2 thr)  int32[N]
  oracle_err, second_err, rec_err f64[N] (exact, == harness oracle_err); med_err, thr f32[N]
  rec_rank int32[N] (1 = oracle; -1 = not a candidate / not library current)
plus "__meta__" (json: floors, sigma, fingerprints, topn). Later tools reuse it via common.load_coverage.

CLI:
  python -m exp.offline_search.profile.coverage [--root R] [--lib auto|current,bpool_all] [--cells all]
         [--procs NCPU] [--topn 50] [--floor-mode third|overall] [--thr X] [--force] [--out DIR] [--brief]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

FIELDS = ("top_rows", "top_errs", "n_cand", "n_good", "n_good2x", "oracle_err", "second_err", "med_err",
          "thr", "rec_err", "rec_rank")
CHUNK = 1024


def _fp(p: Path) -> list | None:
    try:
        st = p.stat()
        return [st.st_size, st.st_mtime_ns]
    except FileNotFoundError:
        return None


def fingerprint(root, cell: str, lib: str) -> dict:
    root = Path(root)
    ms = C.cell_ms(cell)
    qd = root / "queries" / cell
    ld = C.library_dir(root, ms, lib)
    cd = C.library_dir(root, ms, "current")
    return {"q_a_inf": _fp(qd / "a_inf.npy") or _fp(qd / "rows.a_inf.npy"), "q_episodes": _fp(qd / "episodes.json"),
            "lib_action": _fp(ld / "action.npy") or _fp(ld / "rows.action.npy"),
            "lib_task_id": _fp(ld / "task_id.npy") or _fp(ld / "rows.task_id.npy"),
            "cur_action": _fp(cd / "action.npy") or _fp(cd / "rows.action.npy"),
            "floor_step0": _fp(root / "floor" / ms / "step0_pairs.npz"),
            "floor_resample": _fp(root / "floor" / ms / "resample.npz")}


TASK_FIELDS = ("top_rows", "top_errs", "n_good", "n_good2x", "oracle_err", "second_err", "med_err", "rec_err", "rec_rank")


def cell_context(root, cell: str, libname: str, floor_mode="third", thr=None, cache: dict | None = None) -> dict | None:
    """Per (cell, library) inputs shared by its task jobs: sigma (library current), per-query thresholds, tasks."""
    qc = C.QueryCell(root, cell)
    lib = C.open_library(root, qc.ms, libname)
    cache = {} if cache is None else cache
    if qc.ms not in cache:
        cur = C.open_library(root, qc.ms, "current")
        if cur is None:
            return None
        sigma = C.lib_sigma(cur["action"])
        cache[qc.ms] = (sigma, C.load_floor(root, qc.ms, sigma))
    if lib is None:
        return None
    sigma, floor = cache[qc.ms]
    if thr is not None:
        th = np.full(qc.n, float(thr))
    elif floor is not None:
        th = floor.thr(qc.third, floor_mode)
    else:
        th = np.full(qc.n, np.nan)
    lt = lib.task_id
    tasks = [(int(t), np.flatnonzero(qc.task_id == t), int((lt == t).sum())) for t in np.unique(qc.task_id)]
    return {"qc": qc, "lib_rows": lib.n, "sigma": sigma, "floor": floor, "thr": th, "tasks": tasks}


def cover_task(job: dict) -> dict:
    """Coverage arrays of the queries of one task (rows `job['rows']`) against the task's library candidates.
    Same arithmetic and CHUNK boundaries as a whole-cell pass, so the result does not depend on the split."""
    t0 = time.time()
    root, cell, libname, topn, t = job["root"], job["cell"], job["lib"], job["topn"], job["task"]
    qc = C.QueryCell(root, cell)
    lib = C.open_library(root, qc.ms, libname)
    qi_all = np.asarray(job["rows"], np.int64)
    thr = np.asarray(job["thr"], np.float64)
    sig = np.asarray(job["sigma"], np.float64)
    ci = np.flatnonzero(lib.task_id == t)
    nq = qi_all.size
    out = {"top_rows": np.full((nq, topn), -1, np.int32), "top_errs": np.full((nq, topn), np.nan, np.float32),
           "n_good": np.zeros(nq, np.int32), "n_good2x": np.zeros(nq, np.int32),
           "oracle_err": np.full(nq, np.nan, np.float64), "second_err": np.full(nq, np.nan, np.float64),
           "med_err": np.full(nq, np.nan, np.float32), "rec_err": np.full(nq, np.nan, np.float64),
           "rec_rank": np.full(nq, -1, np.int32)}
    res = {"cell": cell, "lib": libname, "task": t, "rows": qi_all, "n_cand": int(ci.size), "arrays": out}
    if ci.size == 0 or nq == 0:
        res["sec"] = time.time() - t0
        return res
    a_q = qc["a_inf"][qi_all]
    a_l = lib["action"][ci]
    Qraw, Lraw = C.exec_block(a_q), C.exec_block(a_l)      # [nq|nc, 5, 7] float64 (exact recompute of the shortlist)
    Q, Lc = C.scaled_seg(a_q, sig), C.scaled_seg(a_l, sig)  # / sigma (GEMM screening of all candidates)

    def exact(ql, cols):  # harness arithmetic: sqrt(mean(((a - c) / sigma)^2)) over [5, 7]
        d = (Lraw[cols] - Qraw[ql][:, None]) / sig
        return np.sqrt(np.mean(d * d, axis=(-2, -1)))
    rec = np.asarray(qc["rec_top1"][qi_all], np.int64) if (libname == "current" and qc.rows.has("rec_top1")) else None
    pos = np.full(lib.n, -1, np.int64)
    pos[ci] = np.arange(ci.size)
    k = min(topn, ci.size)
    kk = min(topn + 8, ci.size)                               # shortlist margin for the exact recompute
    for b in range(0, nq, CHUNK):
        ql = np.arange(b, min(b + CHUNK, nq))
        E = C.err_matrix(Q[ql], Lc)                           # [nq, nc] screening values (GEMM, ~1e-8 rounding)
        if kk < ci.size:
            part = np.argpartition(E, kk - 1, axis=1)[:, :kk]
        else:
            part = np.broadcast_to(np.arange(ci.size), (ql.size, ci.size)).copy()
        pe = exact(ql, part)                                  # exact err of the shortlist (== harness oracle)
        o = np.lexsort((ci[part], pe), axis=1)[:, :k]         # err asc, then library row asc (== harness argmin)
        part = np.take_along_axis(part, o, 1)
        pe = np.take_along_axis(pe, o, 1)
        out["top_rows"][ql, :k] = ci[part]
        out["top_errs"][ql, :k] = pe
        out["oracle_err"][ql] = pe[:, 0]
        if k > 1:
            out["second_err"][ql] = pe[:, 1]
        out["med_err"][ql] = np.median(E, axis=1)
        th = thr[ql][:, None]
        out["n_good"][ql] = (E <= th).sum(1)
        out["n_good2x"][ql] = (E <= 2 * th).sum(1)
        if rec is not None:
            col = pos[rec[ql]]
            ok = col >= 0
            re = np.full(ql.size, np.nan)
            rr = np.full(ql.size, -1, np.int64)
            if ok.any():
                re[ok] = exact(ql[ok], col[ok][:, None])[:, 0]
                self_e = E[np.flatnonzero(ok), col[ok]]           # rank on the screening matrix (same rounding)
                rr[ok] = (E[ok] < self_e[:, None]).sum(1) + 1
            out["rec_err"][ql] = re
            out["rec_rank"][ql] = rr
    res["sec"] = time.time() - t0
    return res


def assemble(ctx: dict, parts: list[dict], topn: int) -> dict:
    """Task results -> the per-query arrays of the whole cell."""
    N = ctx["qc"].n
    out = {"top_rows": np.full((N, topn), -1, np.int32), "top_errs": np.full((N, topn), np.nan, np.float32),
           "n_cand": np.zeros(N, np.int32), "n_good": np.zeros(N, np.int32), "n_good2x": np.zeros(N, np.int32),
           "oracle_err": np.full(N, np.nan, np.float64), "second_err": np.full(N, np.nan, np.float64),
           "med_err": np.full(N, np.nan, np.float32), "thr": ctx["thr"].astype(np.float32),
           "rec_err": np.full(N, np.nan, np.float64), "rec_rank": np.full(N, -1, np.int32)}
    for r in parts:
        rows = r["rows"]
        out["n_cand"][rows] = r["n_cand"]
        for f in TASK_FIELDS:
            out[f][rows] = r["arrays"][f]
    return out


def cover_cell(job: dict) -> dict:
    """Serial whole-cell API (kept for callers): all task jobs of one (cell, library) in this process."""
    t0 = time.time()
    ctx = cell_context(job["root"], job["cell"], job["lib"], job.get("floor_mode", "third"), job.get("thr"))
    if ctx is None:
        return {"cell": job["cell"], "lib": job["lib"], "error": f"library {job['lib']!r} or current missing"}
    parts = [cover_task(dict(root=job["root"], cell=job["cell"], lib=job["lib"], topn=job["topn"], task=t, rows=rows,
                             thr=ctx["thr"][rows], sigma=ctx["sigma"])) for t, rows, _ in ctx["tasks"]]
    return _cell_result(job["root"], job["cell"], job["lib"], ctx, assemble(ctx, parts, job["topn"]), time.time() - t0)


def _cell_result(root, cell, libname, ctx, arrays, sec) -> dict:
    fl = ctx["floor"]
    return {"cell": cell, "lib": libname, "arrays": arrays, "sigma": np.asarray(ctx["sigma"]).tolist(),
            "floor": None if fl is None else {"median": fl.median, "by_third": fl.by_third, "source": fl.source, "n": fl.n},
            "lib_rows": ctx["lib_rows"], "fingerprint": fingerprint(root, cell, libname), "sec": round(sec, 2)}


# -------------------------------------------------------------------------------------------- cache
def read_cache(root, lib: str) -> tuple[dict, dict]:
    """-> (meta, {cell: {field: array}}) of an existing cache file (empty if absent)."""
    p = C.coverage_path(root, lib)
    if not p.exists():
        return {}, {}
    cc = C.CoverageCache(p)
    arrays = {c: {f: cc.get(c, f) for f in cc.fields(c)} for c in cc.cells}
    return cc.meta, arrays


def write_cache(root, lib: str, meta: dict, arrays: dict) -> Path:
    p = C.coverage_path(root, lib)
    p.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{c}__{f}": v for c, d in arrays.items() for f, v in d.items()}
    flat["__meta__"] = np.array(json.dumps(meta, default=C._jsonable))
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, p)
    return p


# ------------------------------------------------------------------------------------------ summary
def slice_stats(a: dict, m: np.ndarray) -> dict:
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    o = a["oracle_err"][m].astype(np.float64)
    thr = a["thr"][m].astype(np.float64)
    nc = a["n_cand"][m]
    ng = a["n_good"][m]
    te = a["top_errs"][m].astype(np.float64)
    d = {"n": n, "n_cand_p50": float(np.median(nc)), "thr_p50": float(np.nanmedian(thr)) if np.isfinite(thr).any() else np.nan,
         "oracle_mean": float(np.nanmean(o)), "oracle_p50": float(np.nanmedian(o)), "oracle_p90": float(np.nanquantile(o, .9)),
         "any_good": float((ng > 0).mean()), "n_good_p50": float(np.median(ng)), "n_good_mean": float(ng.mean()),
         "frac_good": float(np.mean(ng / np.maximum(nc, 1))), "any_good2x": float((a["n_good2x"][m] > 0).mean()),
         "second_mean": float(np.nanmean(a["second_err"][m])) if np.isfinite(a["second_err"][m]).any() else np.nan,
         "med_mean": float(np.nanmean(a["med_err"][m]))}
    for r in (5, 10, 50):
        if te.shape[1] >= r:
            col = te[:, r - 1]
            d[f"e@{r}"] = float(np.nanmean(col)) if np.isfinite(col).any() else np.nan
    rr = a["rec_rank"][m]
    ok = rr > 0
    if ok.any():
        re = a["rec_err"][m][ok].astype(np.float64)
        d.update({"rec_mean": float(re.mean()), "rec_rank_p50": float(np.median(rr[ok])),
                  "rec_top1_is_oracle": float((rr[ok] == 1).mean()), "rec_regret": float((re - o[ok]).mean()),
                  "rec_indist": float((re <= thr[ok]).mean()), "rec_rank_le10": float((rr[ok] <= 10).mean())})
    return d


def summarize(root, lib: str, per_cell: dict, full: bool = False) -> dict:
    rows_cell, rows_task, rows_third, rows_tt = [], [], [], []
    for cell in C.CELLS:
        if cell not in per_cell:
            continue
        a = per_cell[cell]
        qc = C.QueryCell(root, cell)
        allm = np.ones(qc.n, bool)
        rows_cell.append({"cell": cell, **slice_stats(a, allm)})
        for t in np.unique(qc.task_id):
            m = qc.task_id == t
            rows_task.append({"cell": cell, "task": int(t), **slice_stats(a, m)})
            if full:
                for th in range(3):
                    rows_tt.append({"cell": cell, "task": int(t), "third": C.THIRDS[th], **slice_stats(a, m & (qc.third == th))})
        for th in range(3):
            rows_third.append({"cell": cell, "third": C.THIRDS[th], **slice_stats(a, qc.third == th)})
    return {"cell": rows_cell, "task": rows_task, "third": rows_third, "task_third": rows_tt}


COLS = [("n", "n"), ("n_cand_p50", "cand", ".0f"), ("thr_p50", "thr"), ("oracle_mean", "orc_mean"),
        ("oracle_p50", "orc_p50"), ("oracle_p90", "orc_p90"), ("any_good", "any_good"), ("frac_good", "frac_good"),
        ("n_good_p50", "ngood_p50", ".0f"), ("any_good2x", "any_2x"), ("e@5", "e@5"), ("e@10", "e@10"), ("med_mean", "med"),
        ("rec_mean", "B0rec"), ("rec_regret", "B0regret"), ("rec_rank_p50", "B0rank_p50", ".0f"), ("rec_indist", "B0indist")]


def print_report(lib: str, summ: dict, meta: dict, full_tables: bool = True) -> str:
    out = [f"## coverage — library `{lib}`  ({meta.get('created', '')})",
           "err = RMS over [:5,:7] / sigma_d (sigma from library current); good = err <= floor median (thr); "
           "B0* = recorded top-1 (library current only). any_good = fraction of queries with >=1 good candidate.", ""]
    fl = meta.get("floors", {})
    if fl:
        out.append("floors: " + "; ".join(
            f"{ms}: p50={v['median']:.3f}" + (f" by_third={ {k: round(x, 3) for k, x in (v.get('by_third') or {}).items()} }" if v.get("by_third") else "")
            + f" [{v['source']}]" for ms, v in fl.items() if v))
        out.append("")
    out.append(C.md_table(summ["cell"], [("cell", "cell")] + COLS, "per cell"))
    out.append("")
    out.append(C.md_table(summ["third"], [("cell", "cell"), ("third", "third")] + COLS, "per cell x step-third"))
    if full_tables:
        out.append("")
        out.append(C.md_table(summ["task"], [("cell", "cell"), ("task", "task")] + COLS, "per cell x task"))
    txt = "\n".join(out)
    print(txt)
    return txt


# --------------------------------------------------------------------------------------------- main
def available_libs(root, spec: str) -> list[str]:
    if spec != "auto":
        return spec.split(",")
    libs = ["current"]
    if any((Path(root) / "library" / ms / "bpool_all").exists() for ms in C.MODEL_SUITES):
        libs.append("bpool_all")
    return libs


def run(root=C.DEFAULT_ROOT, libs="auto", cells="all", procs=None, topn=50, floor_mode="third", thr=None,
        force=False, out=None, full_tables=True, quiet=False) -> dict:
    root = Path(root)
    cells = C.expand_cells(cells)
    t0 = time.time()
    plan, jobs, fcache = {}, [], {}
    for lib in available_libs(root, libs):
        meta, arrays = read_cache(root, lib)
        stale = []
        for cell in cells:
            ms = C.cell_ms(cell)
            if not (root / "queries" / cell).exists() or not C.library_dir(root, ms, lib).exists():
                continue
            fp_old = (meta.get("fingerprints") or {}).get(cell)
            same = (fp_old == fingerprint(root, cell, lib) and cell in arrays and meta.get("topn") == topn
                    and meta.get("floor_mode") == floor_mode and meta.get("thr_override") == thr)
            if force or not same:
                stale.append(cell)
        ctxs = {}
        for cell in stale:
            ctx = cell_context(root, cell, lib, floor_mode, thr, fcache)
            if ctx is None:
                continue
            ctxs[cell] = ctx
            for t, rows, nc in ctx["tasks"]:
                for b in range(0, max(rows.size, 1), CHUNK):   # one job per CHUNK-row block (== serial chunking)
                    r_ = rows[b:b + CHUNK]
                    jobs.append((r_.size * max(nc, 1), dict(root=str(root), cell=cell, lib=lib, topn=topn, task=t, rows=r_,
                                                            thr=ctx["thr"][r_], sigma=ctx["sigma"])))
        plan[lib] = {"meta": meta, "arrays": arrays, "stale": [c for c in stale if c in ctxs], "ctxs": ctxs,
                     "missing": [c for c in stale if c not in ctxs]}
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i][0])        # largest tasks first (tail balance)
    res = C.pmap(cover_task, [jobs[i][1] for i in order], procs)
    by_cell: dict = {}
    for r in res:
        by_cell.setdefault((r["lib"], r["cell"]), []).append(r)
    t_pool = time.time() - t0
    results = {}
    for lib, P in plan.items():
        meta, arrays, stale = P["meta"], P["arrays"], P["stale"]
        compatible = bool(meta) and meta.get("topn") == topn and meta.get("floor_mode") == floor_mode \
            and meta.get("thr_override") == thr
        if not compatible:  # parameters changed: drop every cached cell (requested ones are all in `stale`)
            meta = {"lib": lib, "topn": topn, "floor_mode": floor_mode, "thr_override": thr, "fingerprints": {},
                    "floors": {}, "sigma": {}, "sec": {}}
            arrays = {}
        errors = [f"library {lib!r} or current missing for {c}" for c in P["missing"]]
        for cell in stale:
            parts = sorted(by_cell.get((lib, cell), []), key=lambda r: (r["task"], int(r["rows"][0]) if len(r["rows"]) else -1))
            r = _cell_result(root, cell, lib, P["ctxs"][cell], assemble(P["ctxs"][cell], parts, topn),
                             sum(x["sec"] for x in parts))
            arrays[cell] = r["arrays"]
            ms = C.cell_ms(cell)
            meta["fingerprints"][cell] = r["fingerprint"]
            meta["floors"][ms] = r["floor"]
            meta["sigma"][ms] = r["sigma"]
            meta["sec"][cell] = r["sec"]
            meta.setdefault("lib_rows", {})[ms] = r["lib_rows"]
        meta.update(created=C.now_local(), root=str(root), dims_source=C.DIMS_SOURCE, metric_source=C.METRIC_SOURCE)
        if stale:
            write_cache(root, lib, meta, arrays)
        if not arrays:
            if not quiet:
                print(f"[coverage] library {lib}: nothing to report ({'; '.join(errors) or 'no cells present'})")
            continue
        summ = summarize(root, lib, {c: arrays[c] for c in cells if c in arrays}, full=True)
        rdir = C.report_dir(root, "coverage", lib, out, multi=len(plan) > 1)
        C.write_json(rdir / f"coverage_{lib}.json", {"meta": meta, "computed_cells": stale, "wall_s": round(time.time() - t0, 1),
                                                    "errors": errors, **summ})
        C.write_csv(rdir / f"coverage_{lib}_cells.csv", summ["cell"])
        C.write_csv(rdir / f"coverage_{lib}_slices.csv",
                    [{"level": "task", **r} for r in summ["task"]] + [{"level": "third", **r} for r in summ["third"]]
                    + [{"level": "task_third", **r} for r in summ["task_third"]])
        if not quiet:
            print_report(lib, summ, meta, full_tables)
            print(f"\n[coverage] {lib}: computed {len(stale)} cells (reused {len([c for c in cells if c in arrays and c not in stale])}); "
                  f"cache {C.coverage_path(root, lib)}; report {rdir}" + (f"; errors: {errors}" if errors else ""))
        results[lib] = {"meta": meta, "summary": summ, "report_dir": str(rdir)}
    if not quiet:
        print(f"[coverage] {len(jobs)} task jobs on {min(C.DEFAULT_PROCS if procs is None else procs, max(len(jobs), 1))} "
              f"procs: pool {t_pool:.1f}s, total {time.time() - t0:.1f}s")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--lib", default="auto", help="auto (current + bpool_all if present) or comma list / library dirs")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--floor-mode", default="third", choices=["third", "overall"])
    ap.add_argument("--thr", type=float, default=None, help="override the good-candidate threshold")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache fingerprint matches")
    ap.add_argument("--out", default=None, help="report dir (default <root>/profile_cache/reports/coverage/<lib>)")
    ap.add_argument("--brief", action="store_true", help="omit the per-task table on stdout")
    a = ap.parse_args(argv)
    run(a.root, a.lib, a.cells, a.procs, a.topn, a.floor_mode, a.thr, a.force, a.out, not a.brief)


if __name__ == "__main__":
    sys.exit(main())
