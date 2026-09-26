"""Full-cell consistency check of M4 / M5 / M6 (diagnostic, not a harness run).

B0's ranking is the only expensive part of these methods and it is already stored: results/r00/B0_current/<cell>.npz
holds B0's top-10 rows + fused scores (bit-identical to FusedKNN.score_all). This script feeds that shortlist into the
REAL method classes (score_all patched to return the stored top-10; everything after it - order, re-rank, synthesis,
extras, confidence, library-fitted scales from the real fit() - is the production code path) and scores the outputs
with the harness metrics on every decision of every cell. Metric-side arrays (a_inf, exec_hit_flag) are used only
to score / to build the QueryView stand-in, exactly as the runner does.

    taskset -c 12-17,56-61 .venv/bin/python exp/offline_search/rounds/r01/f3_b0plus/_replay_check.py [--procs 8]
Writes replay_check.json next to this file and prints a table.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from exp.offline_search.harness import api, metrics, store  # noqa: E402

import method as M  # noqa: E402

ROOT = "/dev/shm/offline_search_store"
R00 = REPO / "exp/offline_search/results/r00/B0_current"
VARIANTS = [
    (M.B0TopkConsensus, {"k": 5, "synth": "mean"}), (M.B0TopkConsensus, {"k": 5, "synth": "kernel"}),
    (M.B0TopkConsensus, {"k": 5, "synth": "med"}), (M.B0TopkConsensus, {"k": 3, "synth": "med"}),
    (M.B0TopkConsensus, {"k": 8, "synth": "mean"}),
    (M.B0ShortlistContRerank, {"lam": 1.0, "k": 5}), (M.B0ShortlistContRerank, {"lam": 2.0, "k": 3}),
    (M.B0ShortlistContRerank, {"lam": 2.0, "k": 5}), (M.B0ShortlistContRerank, {"lam": "inf", "k": 3}),
    (M.ConsistencyConfidence, {"base": "b0", "weights": "equal"}),
    (M.ConsistencyConfidence, {"base": "b0", "weights": "split"}),
    (M.ConsistencyConfidence, {"base": "b0", "weights": "split", "veto": True, "tau": 0.3}),
    (M.ConsistencyConfidence, {"base": "m4k5med", "weights": "split"}),
    (M.ConsistencyConfidence, {"base": "m5l2k3", "weights": "split"}),
]


class Q:
    """QueryView stand-in with the fields the F3 methods read after score_all."""
    __slots__ = ("step", "task_id", "prev_a_exec", "prev_hit", "i")


def run_cell(cell):
    z = np.load(R00 / f"{cell}.npz")
    qc = store.QueryCell(ROOT, cell)
    lib = store.LibraryView(ROOT, qc.lib_key, "current")
    sigma = store.action_sigma(ROOT, qc.lib_key)
    N = z["row"].size
    rowq, step, topk, tsc = z["row"], z["step"].astype(int), z["topk"], z["topk_scores"].astype(np.float32)
    a_exec = qc.a_exec
    flag = qc.exec_hit_flag
    gt = metrics.seg(np.asarray(qc.a_inf[rowq], np.float64))
    trans = None
    out = {}
    zeros = np.zeros(10, np.float32)
    for cls, kw in VARIANTS:
        m = cls(**kw)
        ctx = api.Context(root=ROOT, cell=cell, seed=0, scratch=pathlib.Path("/tmp") / f"f3_replay_{cell}")
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        m.fit(lib, ctx)
        cur = {}

        def score_all(q, _cur=cur):
            i = q.i
            return topk[i].astype(np.int64), tsc[i].copy(), (zeros, zeros, zeros, zeros, zeros, zeros)

        m.score_all = score_all
        err = np.empty(N)
        grip = np.empty(N)
        conf = np.empty(N)
        reg = np.empty(N)
        ex_keep = {k: np.full(N, np.nan) for k in ("cont0", "cont_b0", "cont_act", "flip10", "t_cont", "t_disp", "t_g",
                                                   "veto", "rank_b0")}
        for i in range(N):
            q = Q()
            q.i = i
            q.step = int(step[i])
            q.task_id = int(z["task_id"][i])
            if q.step > 0:
                pr = int(rowq[i]) - 1
                q.prev_a_exec = a_exec[pr]
                f = int(flag[pr])
                q.prev_hit = True if f == 1 else (False if f == 0 else None)
            else:
                q.prev_a_exec = None
                q.prev_hit = None
            r = m.query(q)
            a = r.action if r.action is not None else lib.action[int(r.topk[0])]
            seg = metrics.seg(np.asarray(a, np.float64)[None])
            err[i] = metrics.err_seg(seg, gt[i:i + 1], sigma)[0]
            grip[i] = metrics.grip_mis_seg(seg, gt[i:i + 1])[0]
            conf[i] = r.confidence
            reg[i] = r.extras["regime"]
            for k in ex_keep:
                if k in r.extras:
                    ex_keep[k][i] = r.extras[k]
        rc = metrics.risk_coverage(err, conf)
        res = {"err": float(err.mean()), "grip": float(grip.mean()), "aurc": float(rc["aurc"]),
               "risk_c30": float(rc["risk_c30"]), "aurc_opt": float(metrics.risk_coverage(err, -err)["aurc"])}
        for nm, msk in (("step0", reg == 0), ("miss", reg == 1), ("hit", reg == 2)):
            if msk.sum() >= 50:
                res[f"err_{nm}"] = float(err[msk].mean())
                res[f"aurc_{nm}"] = float(metrics.risk_coverage(err[msk], conf[msk])["aurc"])
        if "t_cont" in r.extras:
            res["veto_frac"] = float(np.nanmean(ex_keep["veto"]))
            res["libstats"] = {k: m.stats[k] for k in ("mu_S", "sd_S", "s_c", "s_a", "bp1_s_c", "bp1_s_a")}
        if isinstance(m, M.B0ShortlistContRerank):
            res["frac_rank_b0_gt0"] = float(np.nanmean(ex_keep["rank_b0"] > 0))
        out[m.name] = res
        print(cell, m.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                                        if k != "libstats"}), flush=True)
    b0 = {"err": float(z["err"].mean()), "aurc": float(metrics.risk_coverage(z["err"], z["confidence"])["aurc"])}
    out["B0_current"] = b0
    return cell, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cells", default=",".join(store.CELLS))
    a = ap.parse_args()
    cells = a.cells.split(",")
    with mp.get_context("fork").Pool(min(a.procs, len(cells))) as pool:
        res = dict(pool.imap_unordered(run_cell, cells))
    (HERE / "replay_check.json").write_text(json.dumps(res, indent=1))
    names = [v for v in res[cells[0]]]
    print(f"\n{'method':28s} " + " ".join(f"{c.replace('_spatial', '_sp').replace('_cache', '_c'):>13s}" for c in cells))
    for key in ("err", "aurc"):
        print(f"-- {key}")
        for n in names:
            print(f"{n:28s} " + " ".join(f"{res[c][n].get(key, float('nan')):13.3f}" for c in cells))


if __name__ == "__main__":
    main()
