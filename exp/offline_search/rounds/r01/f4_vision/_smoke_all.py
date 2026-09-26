"""Smoke every F4 harness variant (batch.json) on a few cells, <= 12 concurrent procs; run under
`taskset -c 18-25,62-69`. Writes _smoke/smoke_<name>_<cell>.txt and prints the metric lines.
usage: python _smoke_all.py [episodes] [cells comma] [name-substring filter]"""
import json
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

D = pathlib.Path(__file__).resolve().parent
REPO = D.parents[4]
EP = sys.argv[1] if len(sys.argv) > 1 else "20"
CELLS = (sys.argv[2] if len(sys.argv) > 2 else "pi05_spatial_inf,pi05_spatial_cache,groot_l10_inf,groot_l10_cache").split(",")
FILT = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = D / "_smoke"
OUT.mkdir(exist_ok=True)
batch = json.loads((D / "batch.json").read_text())


def one(job):
    b, cell = job
    tag = f"{b['method'].split(':')[1]}_{abs(hash(json.dumps(b['kwargs'], sort_keys=True))) % 10**6}"
    cmd = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.harness.smoke", "--method", b["method"],
           "--kwargs", json.dumps(b["kwargs"]), "--cell", cell, "--episodes", EP,
           "--root", "/dev/shm/offline_search_store", "--out", "/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/smoke_runs"]
    if b.get("subsample"):
        cmd += ["--subsample", b["subsample"]]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    m = re.search(r"attrs: name=(\S+)", txt)
    name = m.group(1) if m else tag
    (OUT / f"smoke_{name}_{cell}.txt").write_text(txt)
    lines = [l for l in txt.splitlines() if "metrics [" in l or "SMOKE" in l or "FAIL" in l or "fit+query" in l]
    return name, cell, lines


jobs = [(b, c) for b in batch if re.search(FILT, json.dumps(b)) for c in CELLS]
with ThreadPoolExecutor(12) as ex:
    for name, cell, lines in ex.map(one, jobs):
        print(f"== {name} {cell}")
        for l in lines:
            print("   ", l.strip()[:400])
