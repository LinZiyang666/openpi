"""Analyse the escalation screen once the coordinator has run it (standard-mode run root; inits 20-29 manifest).

Per arm: SR, owner IR (from the per-decision ledger), escalation rate, escalation time, rescue rate r = success among
escalated episodes, false-alarm rate = escalated among pairs that the paired base arm solved, paired comparison
against the arm's base (exact McNemar + task-stratified bootstrap).  Also the base arm's own *self-recovery*: success
among base episodes whose logged top-1 rows cross the same lag >= 12 rule before decision 80 (pi0.5 only; GR00T
standard logs carry no rows).

RULE 1: the per-step reader drops init >= 30 at parse time; the screen manifest only contains inits 20-29.

    taskset -c 2-9 env OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m \
        exp.offline_search.rounds.r09.explore_opus.round2.tools.screen_analysis --run-root <RUN>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import OUT, PRICE, dump, iter_jsonl_discovery, parse_uid, check_root
from .triggers import catalog

PAIRS = {"esc": "cache", "onlynp_esc": "onlynp"}


def ledger(run, arm):
    path = check_root(Path(run)) / "runs" / arm / "client" / "per_step.jsonl"
    dec, eps = [], {}
    for r in iter_jsonl_discovery(path):
        if r.get("accepted") is False:
            continue
        t, i = parse_uid(r["task_uid"])
        if r.get("_kind") == "client_timing":
            eps[(t, i)] = bool(r.get("success"))
            continue
        if "hit_type" not in r:
            continue
        fo = (r.get("factor_outputs") or {}).get("osplug") or {}
        dec.append(dict(task=t, init=i, step=int(r.get("step_idx", -1)), miss=r.get("hit_type") == "MISS",
                        look=bool(r.get("searched")), judge=str(fo.get("os_judge")),
                        row=int(fo["os_row"]) if fo.get("os_row") is not None else -1,
                        success=bool(r.get("success"))))
    D = pd.DataFrame(dec)
    if D.empty:
        return D, eps
    assert D["init"].max() < 30
    return D.sort_values(["task", "init", "step"]), eps


def arm_summary(run, arm, model, esc_codes=("force:82", "force:91")):
    D, eps = ledger(run, arm)
    v, m = PRICE[model]
    rows = []
    for (t, i), e in D.groupby(["task", "init"]):
        miss = e.miss.values
        look = e.look.values
        esc = e.miss.values & e.judge.isin(esc_codes).values if arm.endswith("_esc") else np.zeros(len(e), bool)
        k = int(e.step.values[np.argmax(esc)] // 5) if esc.any() else None
        rows.append(dict(task=t, init=i, success=eps.get((t, i), bool(e.success.values[-1])), decisions=len(e),
                         cost=float(v * look.sum() + m * miss.sum()), calls=int(miss.sum()), esc_step=k))
    return pd.DataFrame(rows)


def mcnemar(a, b):
    """a, b boolean arrays on the same pairs: returns (+, -, exact two-sided p)."""
    plus, minus = int((a & ~b).sum()), int((~a & b).sum())
    n = plus + minus
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(plus, minus), n, 0.5))
    return plus, minus, p


def boot(diff, task, draws=4000, seed=20261002):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(task == t) for t in np.unique(task)]
    vals = []
    for _ in range(draws):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        vals.append(diff[idx].mean())
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def _server_top1(run, arm):
    """(task, init) -> list of (decision index, top-1 row) at fresh decisions, from the server decision logs
    (last attempt only); RULE 1 filter on the record's uid."""
    best = {}
    for path in sorted((check_root(Path(run)) / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):
            if r.get("ev") != "dec" or not r.get("vision") or r.get("top1") is None:
                continue
            t, i = parse_uid(r["uid"])
            a = int(r.get("attempt", 1))
            cur = best.setdefault((t, i), [a, []])
            if a > cur[0]:
                best[(t, i)] = cur = [a, []]
            if a == cur[0]:
                cur[1].append((int(r.get("step", -1)), int(r["top1"])))
    return {k: sorted(v[1]) for k, v in best.items()}


def self_recovery(run, arm, cell):
    """Base arm: success among episodes whose top-1 rows cross lag >= 12 before decision 80 (the frozen rule)."""
    D, eps = ledger(run, arm)
    if D.empty:
        return None
    st = catalog(cell).step.values
    if (D.row >= 0).any():
        seqs = {k: [(s // 5, int(r)) for s, r, lk in zip(e.step.values, e.row.values, e.look.values) if r >= 0 and lk]
                for k, e in D.groupby(["task", "init"])}
    else:
        seqs = _server_top1(run, arm)
    out = []
    for k, seq in seqs.items():
        if any(d - st[r] >= 12 and d <= 80 for d, r in seq):
            out.append(eps.get(k, None))
    out = [x for x in out if x is not None]
    return dict(n=len(out), recovered=float(np.mean(out)) if out else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="/home/weiland/trace_runs/os_closed_loop/r09_opus_escalation")
    a = ap.parse_args(argv)
    run = Path(a.run_root)
    arms = {x["arm"]: x for x in json.loads((run / "arms.json").read_text())}
    have = {k: v for k, v in arms.items() if (run / "runs" / k / "client" / "per_step.jsonl").exists()}
    S = {k: arm_summary(run, k, v["model"]) for k, v in have.items()}
    report = {}
    for k, df in S.items():
        model = arms[k]["model"]
        r = dict(n=len(df), sr=float(df.success.mean()), ir=float(df.cost.sum() / df.decisions.sum()),
                 calls_per_episode=float(df.calls.mean()))
        if k.endswith("_esc"):
            e = df[df.esc_step.notna()]
            r.update(escalated=float(df.esc_step.notna().mean()), esc_controls_median=float(e.esc_step.median() * 5) if len(e) else None,
                     rescue_rate=float(e.success.mean()) if len(e) else None)
            base = next((k[: -len(s)] + PAIRS[s] for s in sorted(PAIRS, key=len, reverse=True) if k.endswith("_" + s)), None)
            if base in S:
                j = df.merge(S[base], on=["task", "init"], suffixes=("", "_base"))
                plus, minus, p = mcnemar(j.success.values, j.success_base.values)
                lo, hi = boot(j.success.values.astype(float) - j.success_base.values, j.task.values)
                r.update(base=base, paired_n=len(j), delta_pp=100 * float(j.success.mean() - j.success_base.mean()),
                         plus=plus, minus=minus, mcnemar_p=p, delta_ci_pp=[100 * lo, 100 * hi],
                         false_alarm_on_base_successes=float(j[j.success_base].esc_step.notna().mean()))
                cell = (model, "l10", 50 if "_50_" in k else 500)
                r["base_self_recovery_after_rule"] = self_recovery(run, base, cell)
        report[k] = r
        print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()}))
    dump(OUT / "screen_report.json", report)


if __name__ == "__main__":
    main()
