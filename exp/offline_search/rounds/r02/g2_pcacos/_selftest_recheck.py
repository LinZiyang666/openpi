"""NaN-aware re-check of a closed-loop selftest run (exp/offline_search/closed_loop/selftest.py compares extras with
`==`, so NaN extras -- e.g. the r01 B0Plus continuity extras at step 0 -- count as mismatches).

Loads the plugin's per-episode input logs (<out>/inputs/*.npz), re-runs the offline harness on the same store
episodes, and compares topk / scores / confidence / synthesized segment / scalar extras with NaN == NaN. Episodes whose
online history diverged from the recorded one (a pick other than rec_top1, or a synthesized action) are compared on
every decision when --all-decisions is given (valid for methods that never read prev_a_exec / hist_a_exec), else on
their first decision only (selftest policy).

    taskset -c ... .venv/bin/python exp/offline_search/rounds/r02/g2_pcacos/_selftest_recheck.py --out <selftest out> \
        --cell pi05_spatial_cache --method <spec> --kwargs '<json>' [--all-decisions]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.harness import run, store  # noqa: E402

ROOT = "/dev/shm/offline_search_store"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--all-decisions", action="store_true")
    a = ap.parse_args()
    qc = store.QueryCell(ROOT, a.cell)
    uid2ep = {e["uid"]: i for i, e in enumerate(qc.episodes)}
    online = {}
    for f in sorted(glob.glob(str(pathlib.Path(a.out) / "inputs" / "*.npz"))):
        z = np.load(f, allow_pickle=False)
        online[json.loads(str(z["meta"]))["uid"]] = z
    sel = sorted(uid2ep[u] for u in online)
    cls, _ = run.load_method_class(a.method)
    F = run._fit_cell(cls, json.loads(a.kwargs), a.cell, root=ROOT,
                      out_dir=pathlib.Path(a.out) / "offline_recheck", seed=0, profile=False)
    jobs = run.episode_jobs(qc, None, sel)
    off = run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0, cell=a.cell)
    rec = np.asarray(qc.rec_top1)
    eq = {"topk": 0, "scores": 0, "conf": 0, "synth": 0, "extras": 0}
    n = 0
    diverged = 0
    first = None
    i = 0
    for ei in sel:
        e = qc.episodes[ei]
        z = online[e["uid"]]
        rows = np.arange(e["start"], e["end"])
        same_hist = bool(((z["top1"] == rec[rows]) & (z["lib"] == "current")).all()) and not bool(z["used_synth"].any())
        diverged += int(not same_hist)
        for s in range(rows.size):
            j = i + s
            if not (same_hist or a.all_decisions) and s > 0:
                continue
            n += 1
            k = off["topk"][j].size
            ok_t = np.array_equal(z["topk"][s, :k], off["topk"][j]) and z["lib"][s] == off["lib"][j]
            ok_s = np.array_equal(z["scores"][s, :k], off["scores"][j], equal_nan=True)
            ok_c = z["conf"][s] == off["conf"][j]
            osyn = off["synth"][j]
            ok_y = (osyn is None and not z["used_synth"][s]) or (
                osyn is not None and np.array_equal(z["synth"][s, :5, :7], osyn))
            exo = off["extras"][j] or {}
            bad = [kk for kk, v in exo.items() if np.asarray(v).size == 1 and not
                   np.array_equal(np.float64(z[f"x_{kk}"][s]), np.float64(v), equal_nan=True)]
            ok_e = not bad
            for key, v in (("topk", ok_t), ("scores", ok_s), ("conf", ok_c), ("synth", ok_y), ("extras", ok_e)):
                eq[key] += int(bool(v))
            if first is None and not (ok_t and ok_s and ok_c and ok_y and ok_e):
                first = {"uid": e["uid"], "step": s, "topk": bool(ok_t), "scores": bool(ok_s), "conf": bool(ok_c),
                         "synth": bool(ok_y), "extras_bad": bad[:8]}
        i += rows.size
    rep = {"cell": a.cell, "method": F["method"].name, "episodes": len(sel), "episodes_diverged": diverged,
           "compared": n, "all_decisions": a.all_decisions, "equal_nan_aware": {k: v / max(n, 1) for k, v in eq.items()},
           "first_diff": first}
    print(json.dumps(rep, indent=1))
    (pathlib.Path(a.out) / "recheck_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
