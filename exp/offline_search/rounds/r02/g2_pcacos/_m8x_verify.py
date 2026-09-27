"""Verify B0BigLibConsFast (M8x) == r01 B0BigLibCons (M8) bit for bit (topk, scores, confidence, synthesized action,
extras) on the decisions of a cell, and time both single-threaded in this process.

    taskset -c <one cpu> .venv/bin/python exp/offline_search/rounds/r02/g2_pcacos/_m8x_verify.py --cell pi05_l10_cache \
        --every 10 [--synth top1|mean --k 5] [--out json]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from exp.offline_search.harness import run  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
HERE = "exp/offline_search/rounds/r02/g2_pcacos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--every", type=int, default=10, help="every k-th episode of the cell")
    ap.add_argument("--synth", default="top1")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kw = {"k": a.k, "synth": a.synth}
    base, _ = run.load_method_class("exp/offline_search/rounds/r01/f3_b0plus/method.py:B0BigLibCons")
    fast, _ = run.load_method_class(f"{HERE}/m8fast.py:B0BigLibConsFast")
    tmp = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/scratch/m8x_verify")
    Fb = run._fit_cell(base, kw, a.cell, root=ROOT, out_dir=tmp, seed=0, profile=False)
    Ff = run._fit_cell(fast, {**kw, "m": a.m}, a.cell, root=ROOT, out_dir=tmp, seed=0, profile=False)
    qc = Fb["qc"]
    eps = list(range(0, len(qc.episodes), a.every))
    jobs = run.episode_jobs(qc, None, eps)
    ob = run.run_jobs_inprocess(Fb["method"], qc, jobs, lib_sizes=Fb["lib_sizes"], seed=0, cell=a.cell)
    of = run.run_jobs_inprocess(Ff["method"], qc, jobs, lib_sizes=Ff["lib_sizes"], seed=0, cell=a.cell)
    n = len(ob["rows"])
    eq = {"topk": 0, "scores": 0, "conf": 0, "synth": 0, "extras": 0, "lib": 0}
    first = None
    for i in range(n):
        s_t = np.array_equal(ob["topk"][i], of["topk"][i])
        s_s = np.array_equal(ob["scores"][i], of["scores"][i])
        s_c = ob["conf"][i] == of["conf"][i]
        sb, sf = ob["synth"][i], of["synth"][i]
        s_y = (sb is None and sf is None) or (sb is not None and sf is not None and np.array_equal(sb, sf))
        eb, ef = ob["extras"][i] or {}, of["extras"][i] or {}
        s_e = set(eb) == set(ef) and all(np.array_equal(np.asarray(eb[k]), np.asarray(ef[k]), equal_nan=True) for k in eb)
        s_l = ob["lib"][i] == of["lib"][i]
        for k_, v in (("topk", s_t), ("scores", s_s), ("conf", s_c), ("synth", s_y), ("extras", s_e), ("lib", s_l)):
            eq[k_] += int(bool(v))
        if first is None and not (s_t and s_s and s_c and s_y and s_e and s_l):
            first = {"i": i, "row": int(ob["rows"][i]), "topk": s_t, "scores": s_s, "conf": s_c, "synth": s_y, "extras": s_e}
    mf = Ff["method"]
    tb = np.asarray(ob["t_ns"], np.float64) / 1e6
    tf = np.asarray(of["t_ns"], np.float64) / 1e6
    cands = np.mean([b.rows.size for b in mf.blocks.values()])
    rep = {"cell": a.cell, "kwargs": kw, "m": a.m, "decisions": n, "episodes": len(jobs),
           "equal_frac": {k_: v / n for k_, v in eq.items()}, "first_diff": first,
           "ms_M8": {"mean": float(tb.mean()), "p50": float(np.median(tb)), "p95": float(np.percentile(tb, 95))},
           "ms_M8x": {"mean": float(tf.mean()), "p50": float(np.median(tf)), "p95": float(np.percentile(tf, 95))},
           "speedup_mean": float(tb.mean() / tf.mean()), "pruned": mf.n_pruned, "full_path": mf.n_full,
           "survivors_mean": mf.sub_total / max(mf.n_pruned, 1), "survivors2_mean": mf.sub2_total / max(mf.n_pruned, 1),
           "cands_per_task_mean": float(cands),
           "fit_s": {"M8": Fb["fit_s"], "M8x": Ff["fit_s"]}}
    print(json.dumps(rep, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
