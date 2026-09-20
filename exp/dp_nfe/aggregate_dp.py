"""Tabulate the Diffusion Policy step ladder: <root>/<task>/<sched><steps>/summary.json (written by eval_dp_steps.py).

usage: python -m exp.dp_nfe.aggregate_dp --root exp/dp_nfe/data/results [--out exp/dp_nfe/data/aggregate_dp.json]

Prints one row per task with the success rate (n_test seeded episodes) and batch-1 predict_action latency of every
(scheduler, steps) cell, and writes the joined JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib

CELLS = ["ddpm100", "ddim10", "ddim4", "ddim2", "ddim1", "ddpm10"]
TASKS = ["square_mh", "can_mh", "tool_hang_ph", "transport_mh", "pusht"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    out: dict = {"policy": "diffusion_policy_cnn_ddpm100 (official real-stanford checkpoints)", "cells": CELLS, "per_task": {}}
    print(f"{'task':14s}" + "".join(f"{c:>16s}" for c in CELLS))
    for t in TASKS:
        row = {}
        for c in CELLS:
            f = root / t / c / "summary.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            row[c] = {"success_rate": d["test_mean_score"], "n_test": d["n_test"], "batch1_latency_ms": d["batch1_latency_ms"],
                      "runner_predict_ms_mean_per_batch": d["runner_predict_ms_mean_per_batch"], "wall_s": d["wall_s"]}
        if not row:
            continue
        out["per_task"][t] = row
        print(f"{t:14s}" + "".join(
            f"{row[c]['success_rate']:6.2f} {row[c]['batch1_latency_ms']:7.0f}ms" if c in row else f"{'-':>16s}" for c in CELLS))
    if out["per_task"]:
        means = {c: [r[c]["success_rate"] for r in out["per_task"].values() if c in r] for c in CELLS}
        out["mean_over_tasks"] = {c: sum(v) / len(v) for c, v in means.items() if v}
        print(f"{'mean':14s}" + "".join(f"{out['mean_over_tasks'][c]:6.2f} ({len(means[c])})   " if means[c] else f"{'-':>16s}" for c in CELLS))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
