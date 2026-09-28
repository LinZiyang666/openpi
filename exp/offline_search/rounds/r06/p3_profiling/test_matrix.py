"""CPU real-plugin/store replay matrix. Every child is pinned; no server or git."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np

from .campaign import make_kwargs, source_rows, fitpath, HERE
from .read_logs import load

PREFIX = ["taskset", "-c", "2-5,46-49", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
          "MKL_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.:src", ".venv/bin/python"]


def run_cell(root, model, cell):
    reports = {}
    source = source_rows()
    suite = "spatial" if cell.startswith("sp_") else "l10"
    for variant, baseline, enabled, p in (("A", "A", False, 0), ("off_A", "A", False, 0),
            ("p0", "A", True, 0), ("p30", "A", True, .3), ("p1", "A", True, 1),
            ("B", "B", False, 0), ("off_B", "B", False, 0), ("on_B", "B", True, .3)):
        tag = f"{model}_{cell}_{variant}"
        out = root / "runs" / tag / "server_cpu"
        out.parent.mkdir(parents=True, exist_ok=False)
        if variant in ("A", "B"):
            row = source[model, baseline, cell]["row"]
            method, kwargs, fit = row["method"], row["kwargs"], fitpath(row)
        else:
            method, kwargs, fit = "exp.offline_search.rounds.r06.p3_profiling.method:Profile", make_kwargs(
                model, cell, p=p, enabled=enabled, baseline=baseline), ""
        cmd = PREFIX + ["-m", "exp.offline_search.rounds.r06.p3_profiling.replay",
            "--method", method, "--kwargs", json.dumps(kwargs), "--fit-artifact", fit,
            "--cell", f"{model}_{suite}_cache", "--replay-cell", f"{model}_{suite}_cache",
            "--policy-tail", "--blocks", "1", "--judge", "guard_only", "--episodes", "4",
            "--max-steps", "24", "--tag", tag, "--out", str(out)]
        if enabled:
            cmd.append("--log-inputs")
        (out.parent / "command.json").write_text(json.dumps(cmd))
        with (out.parent / "replay.log").open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        r = json.loads((out / "report.json").read_text())
        ds = [json.loads(line) for line in (out / f"decisions_{tag}.jsonl").read_text().splitlines()]
        if enabled:
            assert r["profile_calls"] == r["vision"]
            client = out.parent / "client"
            client.mkdir()
            # Synthetic journal for strict reader plumbing ONLY; truncated
            # fixed-observation replays are not simulator outcome experiments.
            js = [dict(task_uid=d["uid"], attempt=d["attempt"], success=d["success"],
                       accepted=True, status="done", error=None) for d in ds
                  if d["ev"] == "episode" and d.get("reason") == "episode_end"]
            (client / "journal.jsonl").write_text("".join(json.dumps(j) + "\n" for j in js))
            tables = load(root, tag)
            r["reader_anchors"] = len(tables[0])
            r["reader_neighbours"] = len(tables[1])
            r["reader_action_steps"] = len(tables[2])
            assert len(tables[0]) == r["vision"]
        else:
            assert r["profile_calls"] == 0 and not any(d["ev"].startswith("p3_") for d in ds)
        reports[variant] = r
        print(tag, r["decisions"], r["vision"], r["misses"], flush=True)
    for x, y in (("A", "off_A"), ("A", "p0"), ("B", "off_B")):
        for field in ("served_sha256", "vision_sha256", "misses"):
            assert reports[x][field] == reports[y][field], (model, cell, x, y, field)
    assert reports["p1"]["misses"] == reports["p1"]["vision"]
    assert 0 < reports["p30"]["misses"] < reports["p30"]["vision"]
    return dict(model=model, cell=cell, PASS=True, variants=reports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise ValueError("1..4 children only")
    a.out.mkdir(parents=True, exist_ok=False)
    cases = [(m, c) for m in ("pi05", "groot") for c in ("l10_50", "l10_500", "sp_50", "sp_500")
             if not a.cells or f"{m}_{c}" in a.cells]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda mc: run_cell(a.out, *mc), cases))
    (a.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    name = "matrix.json" if len(cases) == 8 else f"matrix_{a.out.name}.json"
    (HERE / "results" / name).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
