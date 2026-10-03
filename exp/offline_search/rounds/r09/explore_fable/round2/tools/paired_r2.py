"""Paired analysis of round-2 screening arms (inits 20-29 only) across run roots.

Arms are given as ``root:arm`` or plain names (default root). Every journal is
filtered to inits < 30 before anything is computed (rule 1); the screening
manifest itself only contains inits 20-29. Owner IR from the cost ledger
(v, m shares). Paired SR differences against a reference arm with exact
McNemar and a task-stratified init bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from exp.offline_search.rounds.r09.explore_fable.tools import common

ROOTS = {"r2": "/home/weiland/trace_runs/os_closed_loop/r09_fable_r2", "grown": "/home/weiland/trace_runs/os_closed_loop/r09_fable_grown"}


def outcomes(root, arm, max_init=30):
    rows = {}
    for line in (Path(root) / "runs" / arm / "client" / "journal.jsonl").open():
        r = json.loads(line)
        if not r.get("accepted") or r.get("status") not in ("done", "failed") or r.get("error"):
            continue
        t, i = int(r["task_uid"].split(":")[-2]), int(r["task_uid"].split(":")[-1])
        if i >= max_init:                      # rule 1: never aggregate inits 30-49
            continue
        rows[(t, i)] = bool(r.get("success"))
    s = pd.Series(rows).sort_index()
    s.index = pd.MultiIndex.from_tuples(s.index, names=["task_id", "init"])
    return s


def owner_ir(root, arm):
    s = json.loads((Path(root) / "runs" / arm / "summary.json").read_text())
    led = s.get("cost_ledger") or {}
    p = common.PRICE["groot" if "groot" in arm else "pi05"]
    if led.get("decisions"):
        return p["look"] * float(led["v"]) + p["call"] * float(led["m"]), dict(decisions=led["decisions"], looks=led.get("vision_decisions"), calls=led.get("misses"))
    return float("nan"), led


def resolve(spec):
    if ":" in spec:
        root, arm = spec.split(":", 1)
        return ROOTS.get(root, root), arm
    return ROOTS["r2"], spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference arm (root:arm)")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rroot, rarm = resolve(a.ref)
    ref = outcomes(rroot, rarm)
    report = {a.ref: dict(sr=float(ref.mean()), n=int(len(ref)), owner_ir=owner_ir(rroot, rarm)[0])}
    print(f"{a.ref}: SR {ref.mean():.3f} n={len(ref)} IR {owner_ir(rroot, rarm)[0]:.4f}")
    for spec in a.arms:
        root, arm = resolve(spec)
        try:
            s = outcomes(root, arm)
        except FileNotFoundError:
            print(f"{spec}: no journal yet")
            continue
        both = pd.concat([s, ref], axis=1, join="inner")
        x, y = both.iloc[:, 0].astype(int).values, both.iloc[:, 1].astype(int).values
        w, l = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        p = stats.binomtest(min(w, l), w + l, 0.5).pvalue if w + l else 1.0
        task = both.index.get_level_values("task_id").values
        boot = common.task_strat_bootstrap(np.column_stack([x, y]), task, n_boot=5000, seed=0)
        lo, hi = common.ci(boot[:, 0] - boot[:, 1])
        ir, led = owner_ir(root, arm)
        per_task = pd.DataFrame(dict(task=task, a=x, b=y)).groupby("task").mean().round(2)
        report[spec] = dict(sr=float(s.mean()), n=int(len(s)), n_paired=int(len(both)), owner_ir=float(ir), ledger=led, delta=float(x.mean() - y.mean()),
                            delta_ci=[float(lo), float(hi)], wins=w, losses=l, p=float(p), per_task=per_task.to_dict("index"))
        print(f"{spec}: SR {s.mean():.3f} @ IR {ir:.4f} (calls {led.get('calls')}/{led.get('decisions')})  vs ref {100*(x.mean()-y.mean()):+.1f} pp [{100*lo:+.1f},{100*hi:+.1f}] +{w}/-{l} p={p:.3f}")
        print("   per task (arm / ref):", " ".join(f"{t}:{r['a']:.1f}/{r['b']:.1f}" for t, r in per_task.iterrows()))
    if a.out:
        common.write_json(a.out, report)


if __name__ == "__main__":
    main()
