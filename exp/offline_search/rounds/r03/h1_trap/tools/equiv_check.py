"""Bit-for-bit equivalence of two (method, kwargs) on one cell's episodes through the harness' in-process runner:
per decision top-k rows, scores, confidence, synthesized action [:5, :7], library and the extras present in both.

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h1_trap/tools/equiv_check.py \
        --a exp/offline_search/rounds/r02/g1_awm/awm.py:AWM --a-kwargs '{"lib":"current","kref":5}' \
        --b exp/offline_search/rounds/r03/h1_trap/awm3.py:AWM3 --b-kwargs '{}' --cell pi05_spatial_cache --episodes 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import run, store  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", threads = 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-kwargs", default="{}")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-kwargs", default="{}")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    a = ap.parse_args(argv)
    out = pathlib.Path(tempfile.mkdtemp(prefix="h1_equiv_"))
    qc = store.QueryCell(a.root, a.cell)
    n = len(qc.episodes)
    eps = sorted(set(np.linspace(0, n - 1, min(a.episodes, n)).round().astype(int).tolist()))
    jobs = run.episode_jobs(qc, None, eps)
    res, names, fit_s = {}, {}, {}
    for tag, spec, kw in (("a", a.a, json.loads(a.a_kwargs)), ("b", a.b, json.loads(a.b_kwargs))):
        cls, _ = run.load_method_class(spec)
        t0 = time.perf_counter()
        F = run._fit_cell(cls, kw, a.cell, root=a.root, out_dir=out / tag, seed=0, profile=False)
        fit_s[tag] = time.perf_counter() - t0
        names[tag] = F["method"].name
        res[tag] = run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0, cell=a.cell)
    A, B = res["a"], res["b"]
    N = len(A["topk"])
    eq = {"topk": 0, "scores": 0, "conf": 0, "synth": 0, "lib": 0, "extras": 0}
    maxd = 0.0
    ex_keys = set()
    for i in range(N):
        eq["topk"] += int(np.array_equal(np.asarray(A["topk"][i]), np.asarray(B["topk"][i])))
        eq["scores"] += int(np.array_equal(np.asarray(A["scores"][i]), np.asarray(B["scores"][i])))
        eq["conf"] += int(A["conf"][i] == B["conf"][i])
        eq["lib"] += int(A["lib"][i] == B["lib"][i])
        sa, sb = A["synth"][i], B["synth"][i]
        if sa is None and sb is None:
            eq["synth"] += 1
        elif sa is not None and sb is not None:
            d = float(np.max(np.abs(np.asarray(sa, np.float64) - np.asarray(sb, np.float64))))
            maxd = max(maxd, d)
            eq["synth"] += int(d == 0.0)
        ea, eb = A["extras"][i] or {}, B["extras"][i] or {}
        common = [k for k in ea if k in eb and np.asarray(ea[k]).size == 1]
        ex_keys.update(common)
        eq["extras"] += int(all(np.float64(ea[k]) == np.float64(eb[k]) for k in common))
    rep = {"cell": a.cell, "episodes": len(jobs), "decisions": N, "a": names["a"], "b": names["b"],
           "fit_s": fit_s, "equal_frac": {k: v / N for k, v in eq.items()}, "synth_max_abs_diff": maxd,
           "extras_compared": sorted(ex_keys),
           "only_in_b": sorted(set(k for e in B["extras"] for k in (e or {})) - set(k for e in A["extras"] for k in (e or {})))}
    rep["IDENTICAL"] = bool(all(v == 1.0 for v in rep["equal_frac"].values()))
    print(json.dumps(rep, indent=1))
    return 0 if rep["IDENTICAL"] else 1


if __name__ == "__main__":
    sys.exit(main())
