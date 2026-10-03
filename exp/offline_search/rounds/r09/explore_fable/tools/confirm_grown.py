"""Paired analysis of the r09_fable_grown confirmation run (same-topology arms on the same pairs).

Reads each arm's accepted journal (one terminal record per (task, init)) and the
cost ledger of its summary.json, pairs arms on (task, init), and reports SR, owner
IR, paired wins/losses, exact McNemar p and a task-stratified bootstrap interval
of the SR difference against a reference arm. Standard-mode run: no debug data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import common


def journal_outcomes(run_root, arm):
    rows = {}
    path = Path(run_root) / "runs" / arm / "client" / "journal.jsonl"
    for line in path.open():
        r = json.loads(line)
        if not r.get("accepted") or r.get("status") not in ("done", "failed") or r.get("error"):
            continue
        uid = r["task_uid"]
        t, i = int(uid.split(":")[-2]), int(uid.split(":")[-1])
        rows[(t, i)] = bool(r.get("success"))
    return pd.Series(rows, name=arm).sort_index()


def owner_ir(run_root, arm):
    s = json.loads((Path(run_root) / "runs" / arm / "summary.json").read_text())
    led = s.get("cost_ledger") or {}
    model = "groot" if "_g_" in arm else "pi05"
    p = common.PRICE[model]
    if led.get("decisions"):
        v, m = float(led["v"]), float(led["m"])
        return p["look"] * v + p["call"] * m, dict(decisions=led["decisions"], v=v, m=m, looks=led.get("vision_decisions"), calls=led.get("misses"))
    return float("nan"), led


def mcnemar(a, b):
    w, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(min(w, l), w + l, 0.5).pvalue if w + l else 1.0
    return w, l, float(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="/home/weiland/trace_runs/os_closed_loop/r09_fable_grown")
    ap.add_argument("--ref", default="r9f_ctrlA_p_l10_50")
    ap.add_argument("arms", nargs="*", default=["r9f_ctrlA_p_l10_50", "r9f_grownA_p_l10_50", "r9f_grownall_p_l10_50", "r9f_P10_p_l10"])
    ap.add_argument("--out", default=str(common.OUT / "confirm_grown.json"))
    a = ap.parse_args(argv)
    outcomes = {}
    report = {}
    for arm in a.arms:
        try:
            outcomes[arm] = journal_outcomes(a.run_root, arm)
        except FileNotFoundError:
            print(f"{arm}: no journal yet")
    ref = outcomes.get(a.ref)
    for arm, s in outcomes.items():
        ir, led = owner_ir(a.run_root, arm)
        row = dict(n=int(len(s)), sr=float(s.mean()), owner_ir=float(ir), ledger=led)
        if ref is not None and arm != a.ref:
            both = pd.concat([s, ref], axis=1, join="inner")
            x, y = both.iloc[:, 0].astype(int).values, both.iloc[:, 1].astype(int).values
            w, l, p = mcnemar(x, y)
            task = np.array([t for t, _ in both.index])
            boot = common.task_strat_bootstrap(np.column_stack([x, y]), task, n_boot=5000, seed=0)
            lo, hi = common.ci(boot[:, 0] - boot[:, 1])
            row.update(ref=a.ref, n_paired=int(len(both)), delta_sr=float(x.mean() - y.mean()), delta_ci=[float(lo), float(hi)],
                       wins=w, losses=l, mcnemar_p=p, per_task=pd.DataFrame(dict(task=task, a=x, b=y)).groupby("task").mean().round(2).to_dict("index"))
        report[arm] = row
        print(arm, json.dumps({k: v for k, v in row.items() if k not in ("ledger", "per_task")}))
    common.write_json(a.out, report)


if __name__ == "__main__":
    main()
