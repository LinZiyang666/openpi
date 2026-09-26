"""Acceptance gates of the harness (protocol §5.2). Prints PASS / FAIL per gate, writes <out>/gates.json.

    python -m exp.offline_search.harness.gates --root <store root> [--out DIR] [--workers <#CPUs in affinity mask>] [--cells all]

G0 data      gripper channel (action dim 6) bimodal around 0 in library + queries (|g| < 0.5 in < 1 %, both signs
             >= 5 %); query task strings <-> task_id consistent with the library manifest; the executed chunk equals
             a_hit on every row of cache arms and a_inf on every row of inf arms (bitwise on [:, :7]; q.prev_hit)
G1 B0 repro  B0 top-1 == recorded top-1 (rec_top1) on >= 99.9 % of decisions; every disagreement has a fused-score gap
             (B0 score of its top-1 minus B0 score of rec_top1) < 1e-4; |B0 score - rec_score| < 1e-4 where they
             agree; the per-field normalized scores of rec_top1 match rec_perfield (reported)
G2 B3 <= B0  oracle err <= B0 err on every decision (and B3 err == oracle_err)
G3 B2 >> B0  random-within-task mean err >= 1.2 x B0 mean err in every cell
G4 determ.   B0 and B2 rerun with a different worker count give bit-identical per-decision arrays
G5 library   library/<m>_<s>/current vs the pkl export (trace_dual audit/libs): keys, rs, action within 1e-6, same ids
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from . import baselines, dims, metrics, run, store  # noqa: E402

AUDIT_LIBS = pathlib.Path("/home/weiland/trace_runs/dual_20260923/audit/libs")
AGREE_MIN = 0.999
GAP_MAX = 1e-4
SCORE_TOL = 1e-4
B2_RATIO_MIN = 1.2
IGNORE_DETERMINISM = ("t_query_us",)

_R: dict = {}


def _print(g, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {g}: {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ G0
def gate_data(root, cells):
    res = {"ok": True, "cells": {}}
    libs = sorted({store.lib_key(c) for c in cells})
    for k in libs:
        lib = store.LibraryView(root, k, "current")
        g = np.asarray(dims.gripper(lib.action), np.float64).ravel()
        ent = {"lib_grip_frac_neg": float((g < 0).mean()), "lib_grip_frac_abs_lt_0.5": float((np.abs(g) < 0.5).mean()),
               "lib_grip_hist": np.histogram(g, bins=[-10, -1.5, -0.5, 0, 0.5, 1.5, 10])[0].tolist()}
        ok = ent["lib_grip_frac_abs_lt_0.5"] < 0.01 and 0.05 <= ent["lib_grip_frac_neg"] <= 0.95
        tm = lib.meta.get("task_map") or {}
        res["cells"][k] = ent
        res["ok"] &= ok
        for c in [c for c in cells if store.lib_key(c) == k]:
            qc = store.QueryCell(root, c)
            gq = np.asarray(dims.gripper(qc.a_inf), np.float64).ravel()
            e = {"q_grip_frac_neg": float((gq < 0).mean()), "q_grip_frac_abs_lt_0.5": float((np.abs(gq) < 0.5).mean())}
            okq = e["q_grip_frac_abs_lt_0.5"] < 0.01 and 0.05 <= e["q_grip_frac_neg"] <= 0.95
            bad_tasks = []
            if tm:
                bad_tasks = sorted({(ep["task"], ep["task_id"]) for ep in qc.episodes if tm.get(ep["task"]) != ep["task_id"]})
            e["task_map_mismatch"] = [list(x) for x in bad_tasks]
            e["task_map_checked"] = bool(tm)
            # executed chunk == a_hit (HIT) in cache arms, == a_inf (MISS) in inf arms, bitwise on [:, :7]
            e["exec_hit_counts"] = qc.exec_hit_counts()
            want = "hit" if qc.arm == "cache" else "miss"
            allrows = e["exec_hit_counts"]["all_rows"]
            e["exec_hit_consistent"] = allrows[want] == qc.N
            res["cells"][c] = e
            res["ok"] &= okq and not bad_tasks and e["exec_hit_consistent"]
    return res


# ------------------------------------------------------------------------------------------------ G1
def _g1_chunk(args):
    cell, lo, hi = args
    qc, lib, rows, rec = _R[cell]
    r = rows[lo:hi]
    f, pf = baselines.fused_for_rows(qc.model, qc.suite, qc.key_v0[r], qc.key_v1[r], qc.rs[r], lib, rec[lo:hi])
    return lo, f, pf


def gate_b0(root, cells, b0_dir, workers):
    res = {"ok": True, "cells": {}}
    for c in cells:
        z = np.load(b0_dir / f"{c}.npz")
        qc = store.QueryCell(root, c)
        lib = store.LibraryView(root, qc.lib_key, "current")
        rows = z["row"].astype(np.int64)
        rec = np.asarray(qc.rec_top1[rows], np.int64)
        rec_score = np.asarray(qc.rec_score[rows], np.float64)
        rec_pf = np.asarray(qc.rec_perfield[rows], np.float64)
        top1 = z["top1"].astype(np.int64)
        f_b0 = z["topk_scores"][:, 0].astype(np.float64)
        _R.clear()
        _R[c] = (qc, lib, rows, rec)
        n = rows.shape[0]
        step = 1024
        f_rec = np.empty(n)
        pf_rec = np.empty((n, 3))
        jobs = [(c, lo, min(n, lo + step)) for lo in range(0, n, step)]
        if workers > 1 and len(jobs) > 1:
            with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
                outs = pool.map(_g1_chunk, jobs)
        else:
            outs = [_g1_chunk(j) for j in jobs]
        for lo, f, pf in outs:
            f_rec[lo:lo + f.shape[0]] = f
            pf_rec[lo:lo + f.shape[0]] = pf
        agree = top1 == rec
        dis = ~agree
        gap = f_b0 - f_rec
        e = {"n": int(n), "agree": float(agree.mean()) if n else float("nan"), "n_disagree": int(dis.sum()),
             "max_gap_disagree": float(np.abs(gap[dis]).max()) if dis.any() else 0.0,
             "max_absdiff_b0score_vs_rec_score_agree": float(np.abs(f_b0[agree] - rec_score[agree]).max()) if agree.any() else 0.0,
             "max_absdiff_recomputed_rec_score": float(np.abs(f_rec - rec_score).max()) if n else 0.0,
             "max_absdiff_perfield": [float(x) for x in np.abs(pf_rec - rec_pf).max(0)] if n else [0.0] * 3,
             "disagree_rows": rows[dis][:20].tolist()}
        e["ok"] = (e["agree"] >= AGREE_MIN and e["max_gap_disagree"] < GAP_MAX
                   and e["max_absdiff_b0score_vs_rec_score_agree"] < SCORE_TOL)
        res["cells"][c] = e
        res["ok"] &= e["ok"]
    return res


# --------------------------------------------------------------------------------------------- G2/G3
def gate_b3(cells, b0_dir, b3_dir):
    res = {"ok": True, "cells": {}}
    for c in cells:
        a, b = np.load(b0_dir / f"{c}.npz"), np.load(b3_dir / f"{c}.npz")
        assert np.array_equal(a["row"], b["row"])
        viol = b["err"] > a["err"] + 1e-12
        e = {"b0_err": float(a["err"].mean()), "b3_err": float(b["err"].mean()), "n_violations": int(viol.sum()),
             "max_b3_minus_oracle": float(np.abs(b["err"] - b["oracle_err"]).max())}
        e["ok"] = e["n_violations"] == 0 and e["max_b3_minus_oracle"] < 1e-12
        res["cells"][c] = e
        res["ok"] &= e["ok"]
    return res


def gate_b2(cells, b0_dir, b2_dir):
    res = {"ok": True, "cells": {}}
    for c in cells:
        a, b = np.load(b0_dir / f"{c}.npz"), np.load(b2_dir / f"{c}.npz")
        r = float(b["err"].mean() / a["err"].mean())
        e = {"b0_err": float(a["err"].mean()), "b2_err": float(b["err"].mean()), "ratio": r,
             "frac_b2_worse": float((b["err"] > a["err"]).mean())}
        e["ok"] = r >= B2_RATIO_MIN
        res["cells"][c] = e
        res["ok"] &= e["ok"]
    return res


# ------------------------------------------------------------------------------------------------ G4
def _same(a, b):
    if a.dtype.kind in "fc":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a.shape == b.shape and np.array_equal(a, b)


def gate_determinism(cells, pairs):
    res = {"ok": True, "runs": {}}
    for name, (d1, d2) in pairs.items():
        diffs = {}
        for c in cells:
            a, b = np.load(d1 / f"{c}.npz"), np.load(d2 / f"{c}.npz")
            keys = sorted(set(a.files) | set(b.files))
            bad = [k for k in keys if k not in IGNORE_DETERMINISM and (k not in a.files or k not in b.files or not _same(a[k], b[k]))]
            if bad:
                diffs[c] = bad
        res["runs"][name] = {"ok": not diffs, "diffs": diffs}
        res["ok"] &= not diffs
    return res


# ------------------------------------------------------------------------------------------------ G5
def gate_library(root, cells):
    res = {"ok": True, "libs": {}}
    for k in sorted({store.lib_key(c) for c in cells}):
        m, s = k.split("_")
        base = AUDIT_LIBS / f"{m}_{store.SUITE_FULL[s]}"
        if not pathlib.Path(f"{base}.meta.json").exists():
            res["libs"][k] = {"ok": None, "note": f"no pkl export at {base}.*"}
            continue
        lib = store.LibraryView(root, k, "current")
        if lib.meta.get("synthetic"):
            res["libs"][k] = {"ok": None, "note": "synthetic fixture library, no pkl to compare"}
            continue
        ref_ids = json.loads(pathlib.Path(f"{base}.meta.json").read_text())["ids"]
        ids = lib.ids
        e = {"L": lib.L, "L_ref": len(ref_ids)}
        if ids:
            pos = {x: i for i, x in enumerate(ref_ids)}
            miss = [x for x in ids if x not in pos]
            e["ids_missing_in_ref"] = len(miss)
            idx = np.array([pos[x] for x in ids if x in pos], np.int64)
            e["same_order"] = bool(len(ids) == len(ref_ids) and np.array_equal(idx, np.arange(len(ref_ids))))
            mine = np.array([i for i, x in enumerate(ids) if x in pos], np.int64)
        else:
            e["ids_missing_in_ref"] = None
            idx = mine = np.arange(min(lib.L, len(ref_ids)))
        for f_store, f_ref in (("key_v0", "vision_0"), ("key_v1", "vision_1"), ("rs", "robot_state"), ("action", "action")):
            A = getattr(lib, f_store)
            B = np.load(f"{base}.{f_ref}.npy", mmap_mode="r")
            mx = 0.0
            for lo in range(0, idx.size, 256):
                a = np.asarray(A[mine[lo:lo + 256]], np.float64)
                b = np.asarray(B[idx[lo:lo + 256]], np.float64)
                if a.shape[1:] != b.shape[1:]:
                    mx = float("inf")
                    break
                mx = max(mx, float(np.abs(a - b).max()) if a.size else 0.0)
            e[f"max_absdiff_{f_store}"] = mx
        e["ok"] = (e["ids_missing_in_ref"] in (0, None) and all(e[f"max_absdiff_{f}"] <= 1e-6
                                                                 for f in ("key_v0", "key_v1", "rs", "action")))
        res["libs"][k] = e
        res["ok"] &= e["ok"]
    return res


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(store.SMOKE_ROOT))
    ap.add_argument("--out", default=None, help="default: exp/offline_search/results/gates/<root basename>")
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--workers-alt", type=int, default=None, help="worker count of the determinism rerun")
    ap.add_argument("--cells", default="all")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    out = pathlib.Path(a.out) if a.out else store.REPO / "exp" / "offline_search" / "results" / "gates" / root.name
    out.mkdir(parents=True, exist_ok=True)
    cells = store.resolve_cells(a.cells, root)
    w2 = a.workers_alt or max(1, a.workers // 3 - 1)
    print(f"gates: root={root} cells={cells} workers={a.workers}/{w2} out={out}", flush=True)
    t0 = time.time()
    report = {"root": str(root), "cells": cells, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    B = "exp.offline_search.harness.baselines:"
    runs = {}

    def _run(key, cls, wk, sub):
        r = run.run(B + cls, {}, cells, root=root, out=out / sub, workers=wk, scoreboard=out / "scoreboard.csv",
                    round_=f"gates_{sub}", verbose=False, timing=(sub == "runs"))
        if r["errors"]:
            raise RuntimeError(f"{cls} failed on {sorted(r['errors'])}: " + next(iter(r["errors"].values()))[-2000:])
        runs[key] = pathlib.Path(r["dir"])

    gates = [
        ("G0_data", lambda: gate_data(root, cells)),
        ("G5_library", lambda: gate_library(root, cells)),
    ]
    for name, fn in gates:
        try:
            report[name] = fn()
        except Exception as exc:
            report[name] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
    try:
        _run("b0", "B0Current", a.workers, "runs")
        _run("b2", "B2Random", a.workers, "runs")
        _run("b3", "B3Oracle", a.workers, "runs")
        report["G1_b0_repro"] = gate_b0(root, cells, runs["b0"], a.workers)
        report["G2_b3_le_b0"] = gate_b3(cells, runs["b0"], runs["b3"])
        report["G3_b2_gg_b0"] = gate_b2(cells, runs["b0"], runs["b2"])
        _run("b0_alt", "B0Current", w2, "rerun")
        _run("b2_alt", "B2Random", w2, "rerun")
        report["G4_determinism"] = gate_determinism(cells, {"B0": (runs["b0"], runs["b0_alt"]),
                                                            "B2": (runs["b2"], runs["b2_alt"])})
    except Exception as exc:
        report["run_error"] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
    report["seconds"] = round(time.time() - t0, 1)
    (out / "gates.json").write_text(json.dumps(report, indent=1, default=metrics._json_default))

    print()
    for g in ("G0_data", "G1_b0_repro", "G2_b3_le_b0", "G3_b2_gg_b0", "G4_determinism", "G5_library", "run_error"):
        if g not in report:
            continue
        r = report[g]
        if "error" in r:
            _print(g, False, r["error"])
            continue
        if g == "G0_data":
            msg = "; ".join(f"{k}: neg={v.get('lib_grip_frac_neg', v.get('q_grip_frac_neg')):.3f} "
                            f"|g|<.5={v.get('lib_grip_frac_abs_lt_0.5', v.get('q_grip_frac_abs_lt_0.5')):.4f}"
                            + (f" task_mismatch={v['task_map_mismatch']}" if v.get("task_map_mismatch") else "")
                            + (f" exec hit/miss/both/neither={'/'.join(str(x) for x in v['exec_hit_counts']['all_rows'].values())}"
                               if "exec_hit_counts" in v else "")
                            for k, v in r["cells"].items())
        elif g == "G1_b0_repro":
            msg = "; ".join(f"{c}: agree={v['agree']:.5f} ({v['n_disagree']}/{v['n']} differ, max gap "
                            f"{v['max_gap_disagree']:.2e}) |score-rec|max={v['max_absdiff_b0score_vs_rec_score_agree']:.2e} "
                            f"perfield max={max(v['max_absdiff_perfield']):.2e}" for c, v in r["cells"].items())
        elif g == "G2_b3_le_b0":
            msg = "; ".join(f"{c}: B3 {v['b3_err']:.4f} <= B0 {v['b0_err']:.4f} (violations {v['n_violations']})"
                            for c, v in r["cells"].items())
        elif g == "G3_b2_gg_b0":
            msg = "; ".join(f"{c}: B2/B0 = {v['ratio']:.2f} ({v['b2_err']:.3f}/{v['b0_err']:.3f})" for c, v in r["cells"].items())
        elif g == "G4_determinism":
            msg = "; ".join(f"{k}: {'identical' if v['ok'] else 'DIFF ' + json.dumps(v['diffs'])}" for k, v in r["runs"].items())
        elif g == "G5_library":
            msg = "; ".join(f"{k}: " + (v.get("note") or
                                        f"L={v['L']}/{v['L_ref']} ids_missing={v['ids_missing_in_ref']} "
                                        f"maxdiff v0={v['max_absdiff_key_v0']:.1e} v1={v['max_absdiff_key_v1']:.1e} "
                                        f"rs={v['max_absdiff_rs']:.1e} act={v['max_absdiff_action']:.1e}")
                            for k, v in r["libs"].items())
        else:
            msg = ""
        _print(g, bool(r.get("ok")), msg)
    allok = all(bool(report[g].get("ok")) for g in report
                if isinstance(report[g], dict) and "ok" in report[g] and report[g]["ok"] is not None)
    print(f"\nGATES {'PASS' if allok else 'FAIL'}  ({report['seconds']} s)  -> {out / 'gates.json'}", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
