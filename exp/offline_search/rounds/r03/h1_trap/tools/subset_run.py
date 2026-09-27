"""One (method, cell) harness run on an episode subset -- every n-th episode, or all episodes of given tasks -- for
development numbers between a 5-episode smoke and the coordinator's full batch. Same runner code path as the full
run (run.run_cell: fit in this process, forked workers, metrics, <out>/<name>/<cell>.npz / .json).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h1_trap/tools/subset_run.py \
        --method exp/offline_search/rounds/r03/h1_trap/awm3.py:AWM3 --kwargs '{"prior_alpha": 0.5}' \
        --cell pi05_spatial_cache [--every 5 | --tasks 6] --workers 2 --out <dir> [--timing]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import run, store  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", threads = 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--every", type=int, default=0, help="every n-th episode (0 = off)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids: all their episodes")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", action="store_true", help="also run the 300-query single-thread timing pass")
    a = ap.parse_args(argv)
    kw = json.loads(a.kwargs)
    cls, src = run.load_method_class(a.method)
    m = run.build_method(cls, kw)
    qc = store.QueryCell(a.root, a.cell)
    eps = list(range(len(qc.episodes)))
    if a.every:
        eps = eps[::a.every]
    if a.tasks:
        want = {int(t) for t in a.tasks.split(",")}
        eps = [i for i in eps if int(qc.episodes[i]["task_id"]) in want]
    out = pathlib.Path(a.out) / m.name
    out.mkdir(parents=True, exist_ok=True)
    summ = run.run_cell(cls, kw, a.cell, root=a.root, out_dir=out, workers=a.workers, seed=0, profile=a.timing,
                        episodes=eps, progress_path=out / "progress.jsonl", do_timing=a.timing,
                        fam=run.infer_family(m, src), src=src)
    mt = summ["metrics"]
    print(json.dumps({"method": m.name, "cell": a.cell, "episodes": len(eps), "n": summ["n_decisions"],
                      "err_mean": mt["err_mean"], "grip_mis": mt["grip_mis"], "aurc": mt["aurc"],
                      "fit_s": summ["fit"]["fit_s"], "ms_per_query": (summ.get("timing") or {}).get("ms_per_query")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
