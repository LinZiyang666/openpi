"""Base-contract check for the G3 wrappers: Passthrough(base) (every mechanism off) must reproduce base.query().

Fits the base alone and the Passthrough wrapper around a fresh instance on one cell, runs the same episodes through
the harness' in-process runner, and compares per decision: the served top-k rows (first synth_k), the synthesized
action on [:5, :7] (bit-exact and max |diff|), the library, and -- when the base has os_confidence -- the
confidence. A base whose os_score_all / synth_k / synth_T describe its own selection passes with 100 % top-k
agreement and max |diff| ~ 0 (bit-exact when it synthesizes with the same arithmetic as g3_core.synth).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/contract_check.py \
        --base exp/offline_search/rounds/r02/g1_awm/<file>.py:<Class> --base-kwargs '{...}' \
        --cell groot_l10_cache --episodes 10 [--root /dev/shm/offline_search_store]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import run, store  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", threads = 1)

WRAP = str(pathlib.Path(__file__).resolve().parents[1] / "wrappers.py") + ":Passthrough"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-kwargs", default="{}")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    bkw = json.loads(a.base_kwargs)
    out = pathlib.Path(a.out or tempfile.mkdtemp(prefix="g3_contract_"))
    qc = store.QueryCell(a.root, a.cell)
    n = len(qc.episodes)
    eps = sorted(set(np.linspace(0, n - 1, min(a.episodes, n)).round().astype(int).tolist()))
    jobs = run.episode_jobs(qc, None, eps)
    res = {}
    for tag, spec, kw in (("base", a.base, bkw), ("wrap", WRAP, {"base": a.base, "base_kwargs": bkw})):
        cls, _ = run.load_method_class(spec)
        F = run._fit_cell(cls, kw, a.cell, root=a.root, out_dir=out / tag, seed=0, profile=False)
        res[tag] = (F["method"], run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=0,
                                                         cell=a.cell))
    mw = res["wrap"][0]
    k = mw.k
    B, W = res["base"][1], res["wrap"][1]
    N = len(B["topk"])
    same_k = sum(np.array_equal(np.asarray(B["topk"][i][:k]), np.asarray(W["topk"][i][:k])) for i in range(N))
    same_lib = sum(B["lib"][i] == W["lib"][i] for i in range(N))
    bitexact, maxd = 0, 0.0
    for i in range(N):
        sb, sw = B["synth"][i], W["synth"][i]
        if sb is None and sw is None:
            bitexact += 1
            continue
        if sb is None or sw is None:
            maxd = float("inf")
            continue
        d = float(np.max(np.abs(np.asarray(sb, np.float64) - np.asarray(sw, np.float64))))
        maxd = max(maxd, d)
        bitexact += int(d == 0.0)
    has_conf = mw.has_conf
    same_conf = sum(B["conf"][i] == W["conf"][i] for i in range(N)) if has_conf else None
    rep = {"base": a.base, "base_kwargs": bkw, "cell": a.cell, "episodes": len(jobs), "decisions": N,
           "synth_k": k, "synth_T": mw.T, "os_library": mw.libname, "os_fit_library": mw.fitname,
           "has_os_synth": mw.has_synth, "has_os_confidence": has_conf,
           "topk_agree": same_k / N, "library_agree": same_lib / N, "synth_bitexact": bitexact / N,
           "synth_max_abs_diff": maxd, "conf_agree": (same_conf / N) if has_conf else None}
    rep["PASS"] = bool(rep["topk_agree"] == 1.0 and rep["library_agree"] == 1.0 and maxd <= 1e-6
                       and (not has_conf or rep["conf_agree"] == 1.0))
    print(json.dumps(rep, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
