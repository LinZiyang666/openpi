"""Run subset_job.py for every (spec entry x cell) with a fixed number of concurrent jobs (each with W forked
workers), all inside the caller's CPU mask (`taskset -c ...` the launcher; children inherit it).

    taskset -c 18-26,62-70 .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/subset_launch.py \
        --spec <batch-like json> --cells all --every 5 --jobs 4 --workers 4 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[5]
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(pathlib.Path(a.spec).read_text())
    cells = CELLS if a.cells == "all" else a.cells.split(",")
    out = pathlib.Path(a.out)
    (out / "_logs").mkdir(parents=True, exist_ok=True)
    todo = []
    for i, s in enumerate(spec):
        for c in (cells if s.get("cells", "all") == "all" else [x for x in s["cells"] if x in cells]):
            todo.append((i, s, c))
    todo.sort(key=lambda x: ("l10" not in x[2], "cache" not in x[2]))      # biggest cells first
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(REPO))
    running, done, fails = [], 0, []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            i, s, c = todo.pop(0)
            log = open(out / "_logs" / f"{i:02d}_{c}.log", "w")
            cmd = [sys.executable, str(HERE / "subset_job.py"), "--method", s["method"], "--kwargs",
                   json.dumps(s.get("kwargs", {})), "--cell", c, "--every", str(a.every), "--workers", str(a.workers),
                   "--out", str(out)] + (["--timing"] if s.get("timing") else [])
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env), i, c, log))
        time.sleep(2)
        for p in list(running):
            if p[0].poll() is not None:
                running.remove(p)
                p[3].close()
                done += 1
                if p[0].returncode:
                    fails.append((p[1], p[2]))
                print(f"[{time.time() - t0:6.0f}s] done {done} rc={p[0].returncode} spec={p[1]} cell={p[2]}", flush=True)
    print(json.dumps({"done": done, "fails": fails, "wall_s": round(time.time() - t0, 1)}), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
