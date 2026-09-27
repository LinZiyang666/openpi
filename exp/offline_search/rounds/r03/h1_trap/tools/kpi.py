"""H1 offline KPIs from harness outputs (<out>/<method>/<cell>.npz + .json) + the store's teacher / executed chunks.

Per (method, cell): n, err (all / step >= 1 / step 0), gripper-sign-faithful err (dim 6 of the synthesized executed
segment snapped to +-1 before scoring), grip_mis, AURC, regret; gripper vote split |gvote| < .5 / < .8 (all rows and
per task, esp. task 6); weight on terminal rows (last 2 library rows) in the late third and overall; teacher gripper
transitions (a_inf sign differs from the previously EXECUTED sign or changes inside the executed segment): their count
and the method's grip_mis on them; served-sign flips vs the executed history (sign(synth[0, 6]) != sign of the
previous executed chunk's step 4) vs the teacher's own flip rate on the same history; grip_commit / term_guard rates.

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h1_trap/tools/kpi.py --out <dir> [--cells ...]
        [--methods ...] [--task 6] [--md <report.md>] [--json <report.json>]
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


def kpis(npz_path: pathlib.Path, json_path: pathlib.Path, qc: store.QueryCell, sigma, task: int | None):
    z = np.load(npz_path)
    J = json.loads(json_path.read_text()) if json_path.exists() else {}
    row = np.asarray(z["row"], np.int64)
    step = np.asarray(z["step"], np.int64)
    tid = np.asarray(z["task_id"], np.int64)
    err = np.asarray(z["err"], np.float64)
    gm = np.asarray(z["grip_mis"], np.float64)
    bin_ = np.asarray(z["bin"], np.int64)
    n = row.size
    out = {"n": int(n), "err": float(err.mean()), "err_s1": float(err[step >= 1].mean()) if (step >= 1).any() else np.nan,
           "err_s0": float(err[step == 0].mean()) if (step == 0).any() else np.nan, "grip_mis": float(gm.mean()),
           "aurc": float(J.get("metrics", {}).get("aurc", np.nan)), "regret": float(np.nanmean(z["regret"])),
           "ms_q": float((J.get("timing") or {}).get("ms_per_query", np.nan)), "fit_s": float(J.get("fit", {}).get("fit_s", np.nan))}
    gt = np.asarray(qc.a_inf[row][:, :ES, :dims.ACT_DIMS], np.float64)      # teacher executed segment
    sig = np.asarray(sigma, np.float64)
    if "synth_seg" in z.files and bool(np.asarray(z["used_synth"]).any()):
        syn = np.asarray(z["synth_seg"], np.float64)                            # (N, 5, 7)
        snap = syn.copy()
        snap[:, :, GD] = _sgn(syn[:, :, GD])
        egs = np.sqrt(np.mean(((snap - gt) / sig) ** 2, axis=(1, 2)))
        out["err_gs"] = float(egs.mean())
        out["err_gs_s1"] = float(egs[step >= 1].mean()) if (step >= 1).any() else np.nan
        served0 = _sgn(syn[:, 0, GD])
    else:
        out["err_gs"] = out["err_gs_s1"] = np.nan
        served0 = _sgn(np.asarray(qc.a_hit[row][:, 0, GD]))                     # library row served as is
    # executed history: previous row's executed chunk (same episode when step >= 1)
    m1 = step >= 1
    prev_exec_sign = np.full(n, 0, np.int64)
    prev_exec_sign[m1] = _sgn(np.asarray(qc.a_exec[row[m1] - 1][:, ES - 1, GD]))
    teach0 = _sgn(gt[:, 0, GD])
    teach_inner = (_sgn(gt[:, :, GD]) != teach0[:, None]).any(1)
    trans = m1 & ((teach0 != prev_exec_sign) | teach_inner)
    out["n_trans"] = int(trans.sum())
    out["grip_mis_trans"] = float(gm[trans].mean()) if trans.any() else np.nan
    out["grip_mis_notrans"] = float(gm[m1 & ~trans].mean()) if (m1 & ~trans).any() else np.nan
    out["flip_served"] = float((served0[m1] != prev_exec_sign[m1]).mean()) if m1.any() else np.nan
    out["flip_teacher"] = float((teach0[m1] != prev_exec_sign[m1]).mean()) if m1.any() else np.nan
    # extras
    def x(k):
        return np.asarray(z[f"x_{k}"], np.float64) if f"x_{k}" in z.files else None
    gv = x("gvote")
    if gv is not None:
        av = np.abs(gv)
        out["vsplit05"], out["vsplit08"] = float((av < .5).mean()), float((av < .8).mean())
        if task is not None and (tid == task).any():
            mt = tid == task
            out[f"vsplit05_t{task}"], out[f"vsplit08_t{task}"] = float((av[mt] < .5).mean()), float((av[mt] < .8).mean())
            out[f"err_t{task}"], out[f"n_t{task}"] = float(err[mt].mean()), int(mt.sum())
        out["vsplit05_by_task"] = {int(t): round(float((av[tid == t] < .5).mean()), 3) for t in np.unique(tid)}
    wt = x("w_term")
    if wt is not None:
        late = bin_ == 2
        out["w_term_late"] = float(np.nanmean(wt[late])) if late.any() else np.nan
        out["w_term_all"] = float(np.nanmean(wt))
    for k in ("gheld", "gflip", "term_open"):
        v = x(k)
        if v is not None:
            out[f"{k}_rate"] = float(np.nanmean(v[m1])) if m1.any() else np.nan
    v = x("gdwell")
    if v is not None:
        out["gdwell_p50"] = float(np.nanmedian(v[m1])) if m1.any() else np.nan
    v = x("term_masked")
    if v is not None:
        out["term_masked_mean"] = float(np.nanmean(v))
    return out


COLS = ["n", "err", "err_s1", "err_s0", "err_gs", "grip_mis", "aurc", "regret", "vsplit05", "vsplit08", "w_term_late",
        "n_trans", "grip_mis_trans", "flip_served", "flip_teacher", "gheld_rate", "gflip_rate", "term_masked_mean",
        "term_open_rate", "ms_q", "fit_s"]


def fmt(v):
    if isinstance(v, float):
        return "" if np.isnan(v) else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir with <method>/<cell>.npz")
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--task", type=int, default=6)
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--cols", default=",".join(COLS))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    cols = a.cols.split(",")
    rows, qcs, sig = [], {}, {}
    for md in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if a.methods and md.name not in a.methods:
            continue
        for npz in sorted(md.glob("*.npz")):
            cell = npz.stem
            if a.cells and cell not in a.cells:
                continue
            if cell not in qcs:
                qcs[cell] = store.QueryCell(a.root, cell)
                sig[cell] = store.action_sigma(a.root, qcs[cell].lib_key)
            k = kpis(npz, npz.with_suffix(".json"), qcs[cell], sig[cell], a.task)
            rows.append({"method": md.name, "cell": cell, **k})
    lines = ["| method | cell | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        lines.append(f"| {r['method']} | {r['cell']} | " + " | ".join(fmt(r.get(c, np.nan)) for c in cols) + " |")
    t6 = [r for r in rows if f"err_t{a.task}" in r]
    if t6:
        lines += ["", f"task {a.task}: | method | cell | n | err | vsplit05 | vsplit08 |", "|---|---|---|---|---|---|"]
        for r in t6:
            lines.append(f"| {r['method']} | {r['cell']} | {r[f'n_t{a.task}']} | {r[f'err_t{a.task}']:.4f} | "
                         f"{r[f'vsplit05_t{a.task}']:.4f} | {r[f'vsplit08_t{a.task}']:.4f} |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        pathlib.Path(a.md).write_text(txt + "\n")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
