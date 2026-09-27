"""Run harness smoke for every batch.json variant on 4 cells (pi0.5-sp inf/cache, GR00T-l10 inf/cache), in parallel
(<= --procs processes, each single-threaded). Outputs: rounds/r02/g4_t2/smoke/<name>__<cell>.txt + summary.json.
T2 fits must already be cached (smoke hides CUDA; a missing fit would train on CPU).

    taskset -c 27-33,71-77 .venv/bin/python exp/offline_search/rounds/r02/g4_t2/_smoke_all.py [--procs 12] [--cells ...]
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[4]
CELLS = ["pi05_spatial_inf", "pi05_spatial_cache", "groot_l10_inf", "groot_l10_cache"]


def one(job):
    spec, kw, cell, out = job
    cmd = [sys.executable, "-m", "exp.offline_search.harness.smoke", "--method", spec, "--kwargs", json.dumps(kw),
           "--cell", cell, "--episodes", "5", "--root", "/dev/shm/offline_search_store"]
    cp = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    txt = cp.stdout + ("\n[stderr]\n" + "\n".join(l for l in cp.stderr.splitlines() if "Warn" not in l)[-4000:] if cp.stderr.strip() else "")
    out.write_text(txt)
    return cell, kw, cp.returncode, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--only", default=None, help="comma list of name substrings")
    a = ap.parse_args()
    spec_list = json.loads((HERE / "batch.json").read_text())
    sd = HERE / "smoke"
    sd.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))
    from method import AWMT2
    jobs = []
    for e in spec_list:
        name = AWMT2(**e["kwargs"]).name
        if a.only and not any(s in name for s in a.only.split(",")):
            continue
        for c in a.cells.split(","):
            jobs.append((e["method"], e["kwargs"], c, sd / f"{name}__{c}.txt"))
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(one, jobs))
    summ = []
    for (cell, kw, rc, txt), j in zip(res, jobs):
        fails = re.findall(r"\[FAIL\].*", txt)
        summ.append({"name": j[3].stem.split("__")[0], "cell": cell, "rc": rc, "fails": fails})
        print(f"{j[3].stem:60s} rc={rc} fails={len(fails)}")
    (sd / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
