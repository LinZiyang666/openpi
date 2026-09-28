"""V2 real-plugin CPU replay; deterministic recorded policies, no simulator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np

from .campaign import make_kwargs, source_rows, fitpath, HERE
from .test_matrix import PREFIX


VARIANTS = [
    ("A", "A", False, 0., {}), ("offA", "A", False, 0., {}),
    ("p0", "A", True, 0., {}), ("p1", "A", True, 1., {}),
    ("p25", "A", True, .25, {}), ("B", "B", False, 0., {}),
    ("offB", "B", False, 0., {}), ("onB0", "B", True, 0., {}),
    ("factorial", "B", True, .5, dict(pre_guard=True, durations=[5, 10], holds=[1, 2, 3], cooldown=1)),
    ("delay", "A", True, .5, dict(cap=1, delays=[0, 1, 2])),
    ("short", "A", True, 1., dict(durations=[5])),
    ("suppress", "B", True, 0., dict(pre_guard=True)),
    ("dose_mix", "A", True, .25, dict(episode_doses=[0., .125, .25, .5, 1.])),
]


def run_cell(root, model, cell):
    reports = {}
    source = source_rows()
    suite = "spatial" if cell.startswith("sp_") else "l10"
    for variant, baseline, enabled, p, design in VARIANTS:
        tag = f"{model}_{cell}_{variant}"
        out = root / "runs" / tag / "server_cpu"
        out.parent.mkdir(parents=True, exist_ok=False)
        if variant in ("A", "B"):
            row = source[model, baseline, cell]["row"]
            method, kwargs, fit = row["method"], row["kwargs"], fitpath(row)
        else:
            method, fit = "exp.offline_search.rounds.r06.p3_profiling.v2:Profile", ""
            kwargs = make_kwargs(model, cell, p=p, enabled=enabled, baseline=baseline)
            kwargs.update(design=design, resample_p=.25, resample_draws=4)
        cmd = PREFIX + ["-m", "exp.offline_search.rounds.r06.p3_profiling.replay", "--method", method,
            "--kwargs", json.dumps(kwargs), "--fit-artifact", fit, "--cell", f"{model}_{suite}_cache",
            "--replay-cell", f"{model}_{suite}_cache", "--policy-tail", "--blocks", "1", "--judge", "guard_only",
            "--episodes", "4", "--max-steps", "24", "--tag", tag, "--out", str(out)]
        if enabled:
            cmd.append("--log-inputs")
        (out.parent / "command.json").write_text(json.dumps(cmd))
        with (out.parent / "replay.log").open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        report = json.loads((out / "report.json").read_text())
        if enabled:
            from .read_v2 import audit_server
            audit = audit_server(out)
            report["audit"] = audit
            assert audit["decisions"] == report["decisions"]
            assert audit["policy_evaluations"] == report["profile_calls"]
        reports[variant] = report
        print(tag, report["decisions"], report["vision"], report["misses"], flush=True)
    for x, y in (("A", "offA"), ("A", "p0"), ("B", "offB")):
        for field in ("served_sha256", "vision_sha256", "misses"):
            assert reports[x][field] == reports[y][field], (model, cell, x, y, field)
    # B's policy samples are recorded fixtures here, so p0 B parity is testable.
    for field in ("served_sha256", "vision_sha256", "misses"):
        assert reports["B"][field] == reports["onB0"][field]
    assert reports["p1"]["misses"] == reports["p1"]["vision"]
    assert reports["short"]["misses"] == reports["short"]["decisions"]
    assert reports["suppress"]["misses"] == 0
    return dict(model=model, cell=cell, PASS=True, variants=reports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = [(m, c) for m in ("pi05", "groot") for c in ("l10_50", "l10_500", "sp_50", "sp_500")
             if not a.cells or f"{m}_{c}" in a.cells]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda mc: run_cell(a.out, *mc), cases))
    (a.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (HERE / "results" / ("matrix_v2.json" if len(cases) == 8 else "matrix_v2_subset.json")).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
