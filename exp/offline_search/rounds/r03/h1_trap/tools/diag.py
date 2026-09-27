"""H1 diagnostics on harness outputs: (1) err by third (early / mid / late) per method -- where does term_guard's
offline cost sit; (2) at teacher gripper transitions (inf cells = clean policy history): why grip_commit held --
|v| < thr at the transition decision, previous vote < thr, dwell < 3; (3) served vs teacher gripper flips per
episode (cache cells: B0's executed history).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h1_trap/tools/diag.py --out <dir> --cell <cell> \
        --methods <name> ... [--thr 0.8]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import dims, store  # noqa: E402

ES, GD = dims.EXEC_STEPS, dims.GRIPPER_DIM


def _sgn(x):
    return np.where(np.asarray(x) >= 0, 1, -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args(argv)
    qc = store.QueryCell(a.root, a.cell)
    rep = {}
    for m in a.methods:
        f = pathlib.Path(a.out) / m / f"{a.cell}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        row = np.asarray(z["row"], np.int64)
        step = np.asarray(z["step"], np.int64)
        ep = np.asarray(z["ep"], np.int64)
        err = np.asarray(z["err"], np.float64)
        bin_ = np.asarray(z["bin"], np.int64)
        r = {"n": int(row.size), "err_by_third": [float(err[bin_ == b].mean()) for b in range(3)],
             "n_by_third": [int((bin_ == b).sum()) for b in range(3)]}
        gt0 = _sgn(np.asarray(qc.a_inf[row][:, 0, GD]))
        m1 = step >= 1
        prev = np.zeros(row.size, np.int64)
        prev[m1] = _sgn(np.asarray(qc.a_exec[row[m1] - 1][:, ES - 1, GD]))
        trans = m1 & (gt0 != prev)
        r["teacher_flips"] = int(trans.sum())
        if "synth_seg" in z.files:
            served = _sgn(np.asarray(z["synth_seg"])[:, 0, GD])
            r["served_flips"] = int((served[m1] != prev[m1]).sum())
            r["served_right_at_teacher_flip"] = float((served[trans] == gt0[trans]).mean()) if trans.any() else None
            # flips per episode
            fe = np.bincount(ep[m1], weights=(served[m1] != prev[m1]).astype(float))
            te = np.bincount(ep[m1], weights=(gt0[m1] != prev[m1]).astype(float))
            eps = np.unique(ep)
            r["served_flips_per_ep"] = float(fe[eps].mean())
            r["teacher_flips_per_ep"] = float(te[eps].mean())
        if "x_gvote" in z.files:
            v = np.asarray(z["x_gvote"], np.float64)
            r["vote_ge_thr_at_teacher_flip"] = float((np.abs(v[trans]) >= a.thr).mean()) if trans.any() else None
            r["vote_sign_eq_teacher_at_flip"] = float((_sgn(v[trans]) == gt0[trans]).mean()) if trans.any() else None
            if "x_gheld" in z.files:
                held = np.asarray(z["x_gheld"], np.float64) == 1
                dw = np.asarray(z["x_gdwell"], np.float64)
                # previous decision's vote (same episode)
                pv = np.full(row.size, np.nan)
                pv[1:] = v[:-1]
                pv[step == 0] = np.nan
                ht = trans & held
                r["held_at_teacher_flip"] = int(ht.sum())
                if ht.any():
                    r["held_reasons_at_teacher_flip"] = {
                        "vote_lt_thr": float((np.abs(v[ht]) < a.thr).mean()),
                        "vote_wrong_sign": float((_sgn(v[ht]) != gt0[ht]).mean()),
                        "prev_vote_lt_thr": float((np.abs(pv[ht]) < a.thr).mean()),
                        "dwell_lt_3": float((dw[ht] < 3).mean())}
                r["held_total"] = int(held[m1].sum())
                r["held_not_at_teacher_flip_and_vote_ge_thr"] = int((held & ~trans & (np.abs(v) >= a.thr)).sum())
        rep[m] = r
    print(json.dumps({"cell": a.cell, **rep}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
