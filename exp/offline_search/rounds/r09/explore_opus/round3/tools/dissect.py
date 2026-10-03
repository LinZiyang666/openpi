"""Dissect the round-2 escalation screen per episode (inits 20-29 only; standard-mode server logs, both models).

For every arm: per episode the success, decisions, owner cost, calls, the online escalation step (from the
``r9o_*`` extras the escalation classes log), and the lag trajectory recomputed from the logged top-1 row (so the
frozen rule can also be replayed on arms that do not escalate, e.g. the base controls).  Pairs each escalation arm
with its same-run base and attributes every discordant pair to (a) an episode where the escalation arm never
escalated -> the controllers were identical for the whole episode, so the flip is run-to-run noise; or (b) an
escalated episode -> escalation effect + noise.

RULE 1: server records are admitted through ``iter_jsonl_discovery(uid_key="uid")`` (init >= 30 dropped before
decoding); holdout roots are refused by ``check_root``; journals likewise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import (
    RUNS, PRICE, dump, iter_jsonl_discovery, parse_uid, check_root)
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"
SCREEN = RUNS / "r09_opus_escalation"


def journal(run, arm):
    out = {}
    for r in iter_jsonl_discovery(check_root(Path(run)) / "runs" / arm / "client" / "journal.jsonl"):
        if r.get("accepted") and r.get("phase") in (None, "eval"):
            t, i = parse_uid(r["task_uid"])
            out[(t, i)] = (bool(r.get("success")), int(r.get("attempt", 1)))
    return out


def server_decisions(run, arm, attempts):
    rows = []
    for path in sorted((check_root(Path(run)) / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):
            if r.get("ev") != "dec":
                continue
            t, i = parse_uid(r["uid"])
            if (t, i) not in attempts or int(r.get("attempt", 1)) != attempts[(t, i)]:
                continue
            ex = r.get("extras") or {}
            rows.append(dict(task=t, init=i, step=int(r.get("step", -1)), vision=bool(r.get("vision")),
                             hit=bool(r.get("hit")), src=r.get("src"), top1=int(r["top1"]) if r.get("top1") is not None else -1,
                             esc=float(ex.get("r9o_escalated", 1.0 if float(ex.get("r9o3_mode", 0) or 0) >= 1 else np.nan)),
                             esc_step=float(ex.get("r9o_esc_step", ex.get("r9o3_t_trig", np.nan))),
                             reason=float(ex.get("os_reason", np.nan)), force=float(ex.get("os_force_miss", np.nan))))
    D = pd.DataFrame(rows)
    if len(D):
        assert D["init"].max() < 30
        D = D.sort_values(["task", "init", "step"]).drop_duplicates(["task", "init", "step"], keep="last")
    return D


def episodes(run, arm, model, cell, lag_thr=12, deadline=80):
    J = journal(run, arm)
    D = server_decisions(run, arm, {k: v[1] for k, v in J.items()})
    st = catalog(cell).step.values
    v, m = PRICE[model]
    out = []
    for (t, i), e in D.groupby(["task", "init"]):
        fresh = e[e.vision & (e.top1 >= 0)]
        lag = fresh.step.values - st[fresh.top1.values]
        rule = np.flatnonzero((lag >= lag_thr) & (fresh.step.values <= deadline))
        online = e[(e.esc == 1)]
        calls = int((e.vision & ~e.hit).sum())
        cost = float(v * e.vision.sum() + m * (e.vision & ~e.hit).sum())
        first_call = e[e.vision & ~e.hit].step.min() if calls else np.nan
        out.append(dict(task=t, init=i, success=J[(t, i)][0], decisions=len(e), cost=cost, calls=calls,
                        rule_step=int(fresh.step.values[rule[0]]) if len(rule) else np.nan,
                        esc_step=float(online.esc_step.iloc[0]) if len(online) else np.nan,
                        first_call=first_call, max_lag=float(lag.max()) if len(lag) else np.nan,
                        calls_before_rule=int(((e.vision & ~e.hit) & (e.step < (fresh.step.values[rule[0]] if len(rule) else 1e9))).sum())))
    return pd.DataFrame(out)


PAIRS = [("pi05", 50, "esc", "cache"), ("pi05", 50, "esc_w24", "cache"), ("pi05", 50, "onlynp_esc", "onlynp"),
         ("groot", 50, "esc", "cache"), ("groot", 50, "esc_w24", "cache"), ("groot", 50, "onlynp_esc", "onlynp"),
         ("pi05", 500, "esc", "cache"), ("groot", 500, "esc", "cache")]


def attribute(E, B):
    j = E.merge(B, on=["task", "init"], suffixes=("", "_b"))
    j["escalated"] = j.esc_step.notna()
    j["flip_up"] = j.success & ~j.success_b
    j["flip_down"] = ~j.success & j.success_b
    res = {}
    for name, sel in (("never_escalated", ~j.escalated), ("escalated", j.escalated)):
        s = j[sel]
        res[name] = dict(n=int(len(s)), up=int(s.flip_up.sum()), down=int(s.flip_down.sum()),
                         sr=float(s.success.mean()) if len(s) else None, base_sr=float(s.success_b.mean()) if len(s) else None)
    s = j[j.escalated]
    res["escalated_base_success"] = dict(n=int(s.success_b.sum()), kept=int((s.success & s.success_b).sum()))
    res["escalated_base_failure"] = dict(n=int((~s.success_b).sum()), rescued=int((s.success & ~s.success_b).sum()))
    res["esc_controls"] = dict(median=float(s.esc_step.median() * 5) if len(s) else None,
                               q25=float(s.esc_step.quantile(.25) * 5) if len(s) else None)
    ok = s[s.success]
    res["rescued_decisions_after_esc"] = dict(median=float((ok.decisions - ok.esc_step).median()) if len(ok) else None,
                                              q75=float((ok.decisions - ok.esc_step).quantile(.75)) if len(ok) else None,
                                              max=float((ok.decisions - ok.esc_step).max()) if len(ok) else None)
    res["by_esc_time"] = {str(k): dict(n=int(len(g)), sr=float(g.success.mean()), base_sr=float(g.success_b.mean()))
                          for k, g in s.groupby(pd.cut(s.esc_step * 5, [0, 200, 250, 300, 400]), observed=True)}
    # attribution cross-tab: both-crossed is the comparable cell; off-diagonals are run-to-run divergence
    te, tb = j.esc_step.notna(), j.rule_step_b.notna()
    res["crosstab"] = {f"esc{x}_base{y}": dict(n=int(((te == x) & (tb == y)).sum()),
                                               sr=float(j[(te == x) & (tb == y)].success.mean()) if ((te == x) & (tb == y)).any() else None,
                                               base_sr=float(j[(te == x) & (tb == y)].success_b.mean()) if ((te == x) & (tb == y)).any() else None,
                                               up=int((j.flip_up & (te == x) & (tb == y)).sum()),
                                               down=int((j.flip_down & (te == x) & (tb == y)).sum()))
                       for x in (0, 1) for y in (0, 1)}
    # the base's own rule-triggered episodes (would have escalated) and their self-recovery
    br = j[j.rule_step_b.notna()]
    res["base_rule_triggered"] = dict(n=int(len(br)), self_recovered=float(br.success_b.mean()) if len(br) else None,
                                      calls_before_rule=float(br.calls_before_rule_b.mean()) if len(br) else None)
    return res, j


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(SCREEN))
    ap.add_argument("--prefix", default="r9o", help="arm name prefix, e.g. r9o3 for the round-3 screen")
    ap.add_argument("--pairs", default="", help="model:lib:arm:base,... (default: the round-2 screen pairs)")
    a = ap.parse_args(argv)
    run = Path(a.run_root)
    pairs = PAIRS if not a.pairs else [(x.split(":")[0], int(x.split(":")[1]), x.split(":")[2], x.split(":")[3])
                                        for x in a.pairs.split(",")]
    cache = {}
    report = {}
    for model, lib, arm_s, base_s in pairs:
        cell = (model, "l10", lib)
        for s in (arm_s, base_s):
            name = f"{a.prefix}_{model}_l10_{lib}_{s}"
            if name not in cache:
                cache[name] = episodes(run, name, model, cell)
        E, B = cache[f"{a.prefix}_{model}_l10_{lib}_{arm_s}"], cache[f"{a.prefix}_{model}_l10_{lib}_{base_s}"]
        res, j = attribute(E, B)
        key = f"{a.prefix}_{model}_l10_{lib}:{arm_s}_vs_{base_s}"
        res.update(sr=float(E.success.mean()), base_sr=float(B.success.mean()),
                   ir=float(E.cost.sum() / E.decisions.sum()), base_ir=float(B.cost.sum() / B.decisions.sum()))
        report[key] = res
        print(key, json.dumps(res))
    for name, df in cache.items():
        df.to_parquet(OUT / f"episodes_{name}.parquet") if OUT.exists() else None
    dump(OUT / ("dissect.json" if a.prefix == "r9o" else f"dissect_{a.prefix}.json"), report)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
