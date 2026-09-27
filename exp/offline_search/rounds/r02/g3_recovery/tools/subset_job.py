"""One (method, cell) harness run on an evenly spaced episode subset (every --every-th episode), for development
numbers between a 5-episode smoke and the coordinator's full batch. Same runner code path as the full run
(run.run_cell: fit in this process, forked workers, metrics, <cell>.npz / <cell>.json).

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/subset_job.py \
        --method <file>:<Class> --kwargs '<json>' --cell pi05_spatial_cache --every 5 --workers 4 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from exp.offline_search.harness import run, store  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args(argv)
    kw = json.loads(a.kwargs)
    cls, src = run.load_method_class(a.method)
    m = run.build_method(cls, kw)
    qc = store.QueryCell(a.root, a.cell)
    eps = list(range(0, len(qc.episodes), a.every))
    out = pathlib.Path(a.out) / m.name
    out.mkdir(parents=True, exist_ok=True)
    summ = run.run_cell(cls, kw, a.cell, root=a.root, out_dir=out, workers=a.workers, seed=0, profile=a.timing,
                        episodes=eps, progress_path=out / "progress.jsonl", do_timing=a.timing,
                        fam=run.infer_family(m, src), src=src)
    mt = summ["metrics"]
    print(json.dumps({"method": m.name, "cell": a.cell, "n": summ["n_decisions"], "err_mean": mt["err_mean"],
                      "aurc": mt["aurc"], "fit_s": summ["fit"]["fit_s"],
                      "ms_per_query": (summ.get("timing") or {}).get("ms_per_query")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
