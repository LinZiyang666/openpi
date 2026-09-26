"""Aggregate the M9 probe raw jobs (<derived>/m9_raw/*.json) into <family dir>/m9_results.json + m9_tables.md.

    .venv/bin/python exp/offline_search/rounds/r01/f4_vision/m9_tables.py
"""
import collections
import glob
import json
import pathlib

import numpy as np

DER = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision")
FAM = pathlib.Path(__file__).resolve().parent
R00 = FAM.parents[2] / "results" / "r00"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
SHORT = {c: c.replace("spatial", "sp").replace("_inf", "-inf").replace("_cache", "-cache").replace("_", "-")
         for c in CELLS}
STEPS = (0, 1, 2)
PRIV = ("priv_obj", "priv_full", "priv_ooi", "priv_rel")


def load():
    jobs, chk = [], None
    for f in sorted(glob.glob(str(DER / "m9_raw" / "*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        jobs += d["jobs"]
        chk = d.get("init_check") or chk
    return jobs, chk


def aggregate(jobs):
    """(lib, cell, step) -> {rep: (mean err, mean rho, n)}; per-query errs pooled over tasks."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    rho = collections.defaultdict(lambda: collections.defaultdict(list))
    nq = collections.defaultdict(int)
    for j in jobs:
        k = (j["lib"], j["cell"], j["step"])
        nq[k] += j["nq"]
        for r, v in j["err"].items():
            acc[k][r] += v
        for r, v in j["rho"].items():
            rho[k][r].append((v, j["nq"]))
    out = {}
    for k in acc:
        out[k] = {}
        for r, v in acc[k].items():
            rr = rho[k].get(r)
            out[k][r] = {"err": float(np.mean(v)), "n": len(v),
                         "rho": float(sum(a * b for a, b in rr) / sum(b for _, b in rr)) if rr else None}
    return out


def b0_recorded():
    """B0 (online, current library, all candidates) err at steps 0..2 from the r00 run."""
    out = {}
    for c in CELLS:
        p = R00 / "B0_current" / f"{c}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for s in STEPS:
            out[(c, s)] = float(z["err"][z["step"] == s].mean())
    return out


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def main():
    jobs, chk = load()
    agg = aggregate(jobs)
    b0r = b0_recorded()
    big = {"pi05": "bpool_cs", "groot": "bpool_all"}
    cells = [c for c in CELLS if (big[c.split("_")[0]], c, 0) in agg]
    reps = sorted({r for k, v in agg.items() if k[0] != "current" for r in v})
    lines = ["# M9 vision_layout_probe — early decisions (steps 0/1/2), step-aligned candidates", "",
             "err = harness err (RMS over [:5,:7] in σ units) of the picked library chunk vs the query's a_inf; "
             "candidates = the row at the same step of every big-library episode of the task (~50). "
             "Pooled over the 10 tasks (n = 500 queries per cell × step). Lower is better. "
             "`*_all` rows: same picker over ALL big-library rows of the task (not step-aligned). "
             "`B0 rec` = B0's recorded online pick (current library, all candidates, results/r00).", ""]
    main_rows = ["oracle", "oracle_alltask", "random", "state", "priv_obj", "priv_full", "priv_ooi", "priv_rel",
                 "priv_obj+st", "priv_full+st", "priv_ooi+st", "priv_rel+st", "B0", "pool_v0", "pool_v1",
                 "pool_both", "pool_v1+st", "pool_both+st", "res_pool_both", "res_pool_tm_both", "res_pool_tm_v1",
                 "res_pool_tm_both+st", "res_pool_tm_v1+st", "tok_both", "res_tok_both", "res_tok_v1",
                 "topvar16_raw_both", "topvar16_res_both", "topvar32_raw_both", "topvar32_res_both",
                 "maxsim_both", "maxsim_v1", "maxsim_res_both", "chamfer_both", "chamfer_res_both",
                 "maxsim_res_both+st", "pix16_both", "pix32_both", "pix32_rescos_both", "pix16_rescos_v1",
                 "state_all", "B0_all", "pool_v1_all", "pool_both_all", "res_pool_tm_both_all",
                 "pool_v1+st_all", "res_pool_tm_both+st_all"]
    res_json = {"init_check": chk, "cells": {}}
    for s in STEPS:
        lines += [f"## Step {s}", "", "| picker | " + " | ".join(SHORT[c] for c in cells) + " |",
                  "|---|" + "---|" * len(cells)]
        lines.append("| B0 rec (current lib) | " + " | ".join(fmt(b0r.get((c, s))) for c in cells) + " |")
        for r in main_rows:
            vals = [agg.get((big[c.split("_")[0]], c, s), {}).get(r, {}).get("err") for c in cells]
            if all(v is None for v in vals):
                continue
            lines.append(f"| {r} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")
    # full json
    for c in cells:
        L = big[c.split("_")[0]]
        res_json["cells"][c] = {str(s): {r: agg[(L, c, s)][r] for r in agg[(L, c, s)]} for s in STEPS}
        res_json["cells"][c]["B0_recorded"] = {str(s): b0r.get((c, s)) for s in STEPS}
    # recovery: (err_state - err_rep) / (err_state - err_priv*), priv* = best privileged picker of that cell x step
    lines += ["## Recovery of the privileged gain over state-only", "",
              "gain_priv = err(state) − min over {priv_obj, priv_full, priv_ooi, priv_rel, and their +st} (per cell × "
              "step); recovery(rep) = (err(state) − err(rep)) / gain_priv. Shown: mean over the 24 cell × step "
              "slots, the min, and the number of slots with recovery ≥ 0.5; plus mean err gain over state.", ""]
    privs = [p for p in reps if p.startswith("priv")]
    rec = {}
    gains = {}
    for c in cells:
        L = big[c.split("_")[0]]
        for s in STEPS:
            a = agg[(L, c, s)]
            st = a["state"]["err"]
            best = min(a[p]["err"] for p in privs)
            gains[(c, s)] = (st - best, min(privs, key=lambda p: a[p]["err"]))
            for r in reps:
                if r in a and not r.startswith(("priv", "oracle", "random")):
                    g = st - best
                    rec.setdefault(r, []).append(((st - a[r]["err"]) / g if g > 1e-6 else np.nan, st - a[r]["err"]))
    lines += ["| cell | " + " | ".join(f"s{s} gain_priv (best)" for s in STEPS) + " |", "|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {SHORT[c]} | " + " | ".join(f"{gains[(c, s)][0]:.3f} ({gains[(c, s)][1]})" for s in STEPS)
                     + " |")
    lines += ["", "| representation | mean recovery | min | slots ≥ 0.5 | mean err gain vs state | mean rho |",
              "|---|---|---|---|---|---|"]
    order = sorted(rec, key=lambda r: -np.nanmean([x[1] for x in rec[r]]))
    rj = {}
    for r in order:
        v = np.array([x[0] for x in rec[r]], float)
        gv = np.array([x[1] for x in rec[r]], float)
        rhos = [agg[(big[c.split("_")[0]], c, s)].get(r, {}).get("rho") for c in cells for s in STEPS]
        rhos = [x for x in rhos if x is not None]
        rj[r] = {"mean_recovery": float(np.nanmean(v)), "min_recovery": float(np.nanmin(v)),
                 "n_ge_half": int(np.sum(v >= 0.5)), "n": int(np.isfinite(v).sum()),
                 "mean_gain_vs_state": float(gv.mean()), "mean_rho": float(np.mean(rhos)) if rhos else None}
        lines.append(f"| {r} | {rj[r]['mean_recovery']:.2f} | {rj[r]['min_recovery']:.2f} | {rj[r]['n_ge_half']}/"
                     f"{rj[r]['n']} | {rj[r]['mean_gain_vs_state']:+.3f} | {fmt(rj[r]['mean_rho'])} |")
    res_json["recovery"] = rj
    res_json["priv_gain"] = {f"{c}|{s}": list(gains[(c, s)]) for c in cells for s in STEPS}
    # rho table for the main rows (ranking quality)
    lines += ["", "## Ranking quality: mean Spearman(sim, −err) over the ~50 step-aligned candidates (avg of steps 0–2)",
              "", "| picker | " + " | ".join(SHORT[c] for c in cells) + " |", "|---|" + "---|" * len(cells)]
    for r in main_rows:
        vals = []
        for c in cells:
            L = big[c.split("_")[0]]
            x = [agg[(L, c, s)].get(r, {}).get("rho") for s in STEPS]
            x = [y for y in x if y is not None]
            vals.append(float(np.mean(x)) if x else None)
        if all(v is None for v in vals):
            continue
        lines.append(f"| {r} | " + " | ".join(fmt(v) for v in vals) + " |")
    # current library (secondary)
    cur_cells = [c for c in CELLS if ("current", c, 0) in agg]
    if cur_cells:
        lines += ["", "## Secondary: current library (5 episodes per task, step-aligned; pi05_l10 has no init idx)", ""]
        for s in STEPS:
            lines += [f"Step {s}", "", "| picker | " + " | ".join(SHORT[c] for c in cur_cells) + " |",
                      "|---|" + "---|" * len(cur_cells)]
            for r in ["oracle", "random", "state", "priv_obj", "priv_full", "priv_rel", "priv_obj+st", "B0",
                      "pool_v0", "pool_v1", "pool_both", "res_pool_both", "res_pool_tm_both", "pool_v1+st"]:
                vals = [agg[("current", c, s)].get(r, {}).get("err") for c in cur_cells]
                lines.append(f"| {r} | " + " | ".join(fmt(v) for v in vals) + " |")
            lines.append("")
        res_json["current"] = {c: {str(s): agg[("current", c, s)] for s in STEPS} for c in cur_cells}
    if chk:
        lines += ["", "## Init-index mapping check", "",
                  "Spearman between pairwise init arm-joint distance and pairwise step-0 eef-position distance, per task "
                  "(mean over tasks); control = the same with the OTHER pool's rows (should be ≈ 0).", "",
                  "| key | mapped | control |", "|---|---|---|"]
        for k, v in chk.items():
            lines.append(f"| {k} | {v['spearman_joint_vs_eef_dist']:.3f} | "
                         f"{v['same_with_the_other_pool (control)']:.3f} |")
    (FAM / "m9_tables.md").write_text("\n".join(lines) + "\n")
    (FAM / "m9_results.json").write_text(json.dumps(res_json, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
