"""Served-action contract check for MixedJudge: on the same episodes the wrapper's top-k (first synth_k rows), library
and synthesized action [:5, :7] must equal the BASE's bit for bit (the judge changes confidence / flags only).
Also reports the confidence agreement with plain V7 (DriftCalibratedConfidence) around the same base outside the
burst / return phases (where MixedJudge lowers the confidence by ret_margin).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h3_judge/tools/contract_check_mx.py \
        --base exp/offline_search/rounds/r02/g1_awm/awm.py:AWM --base-kwargs '{"lib":"current","kref":5}' \
        --judge-kwargs '{"guards": true, "events": "all", "burst": 2}' --cell pi05_spatial_cache --episodes 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import run, store  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]
JUDGE = str(HERE / "judge.py") + ":MixedJudge"
V7 = str(pathlib.Path(__file__).resolve().parents[3] / "r02" / "g3_recovery" / "wrappers.py") + ":DriftCalibratedConfidence"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exp/offline_search/rounds/r02/g1_awm/awm.py:AWM")
    ap.add_argument("--base-kwargs", default='{"lib": "current", "kref": 5}')
    ap.add_argument("--judge-kwargs", default='{"guards": true, "events": "all", "burst": 2}')
    ap.add_argument("--cell", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    bkw = json.loads(a.base_kwargs)
    jkw = dict(json.loads(a.judge_kwargs), base=a.base, base_kwargs=bkw)
    out = pathlib.Path(a.out or tempfile.mkdtemp(prefix="mxj_contract_"))
    qc = store.QueryCell(a.root, a.cell)
    n = len(qc.episodes)
    eps = sorted(set(np.linspace(0, n - 1, min(a.episodes, n)).round().astype(int).tolist()))
    jobs = run.episode_jobs(qc, None, eps)
    res = {}
    for tag, spec, kw in (("base", a.base, bkw), ("judge", JUDGE, jkw), ("v7", V7, {"base": a.base, "base_kwargs": bkw})):
        cls, _ = run.load_method_class(spec)
        F = run._fit_cell(cls, kw, a.cell, root=a.root, out_dir=out / tag, seed=0, profile=False)
        res[tag] = (F["method"], run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0,
                                                         cell=a.cell), F["fit_s"])
    mj = res["judge"][0]
    k = mj.k
    B, J, V = res["base"][1], res["judge"][1], res["v7"][1]
    N = len(B["topk"])
    same_k = sum(np.array_equal(np.asarray(B["topk"][i][:k]), np.asarray(J["topk"][i][:k])) for i in range(N))
    same_lib = sum(B["lib"][i] == J["lib"][i] for i in range(N))
    bitexact, maxd = 0, 0.0
    for i in range(N):
        sb, sj = B["synth"][i], J["synth"][i]
        if sb is None and sj is None:
            bitexact += 1
            continue
        if sb is None or sj is None:
            maxd = float("inf")
            continue
        d = float(np.max(np.abs(np.asarray(sb, np.float64) - np.asarray(sj, np.float64))))
        maxd = max(maxd, d)
        bitexact += int(d == 0.0)
    # confidence vs plain V7 outside the return phase (os_phase == 2 lowers it by ret_margin)
    ph = np.asarray([float((J["extras"][i] or {}).get("os_phase", 0.0)) for i in range(N)])
    raw = np.asarray([float((J["extras"][i] or {}).get("os_conf_raw", np.nan)) for i in range(N)])
    cj = np.asarray(J["conf"], np.float64)
    cv = np.asarray(V["conf"], np.float64)
    conf_raw_eq_v7 = float(np.mean(raw == cv))
    conf_eq_v7_outside_return = float(np.mean(cj[ph != 2] == cv[ph != 2])) if (ph != 2).any() else None
    pen_ok = bool(np.all(np.abs((cv[ph == 2] - cj[ph == 2]) - mj.ret_margin) < 1e-12)) if (ph == 2).any() else True
    force = np.asarray([float((J["extras"][i] or {}).get("os_force_miss", 0.0)) for i in range(N)])
    reason = np.asarray([int((J["extras"][i] or {}).get("os_reason", 0)) for i in range(N)])
    rep = {"base": a.base, "base_kwargs": bkw, "judge_kwargs": jkw, "judge_name": mj.name, "cell": a.cell,
           "episodes": len(jobs), "decisions": N, "synth_k": k, "fit_s": {t: round(res[t][2], 3) for t in res},
           "topk_agree": same_k / N, "library_agree": same_lib / N, "synth_bitexact": bitexact / N,
           "synth_max_abs_diff": maxd, "conf_raw_eq_v7": conf_raw_eq_v7,
           "conf_eq_v7_outside_return": conf_eq_v7_outside_return, "return_penalty_exact": pen_ok,
           "force_rate": float(force.mean()), "reason_counts": {int(r): int((reason == r).sum()) for r in range(8)},
           "phase_counts": {int(p): int((ph == p).sum()) for p in range(3)}}
    rep["PASS"] = bool(rep["topk_agree"] == 1.0 and rep["library_agree"] == 1.0 and maxd == 0.0
                       and conf_raw_eq_v7 == 1.0 and pen_ok)
    print(json.dumps(rep, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
