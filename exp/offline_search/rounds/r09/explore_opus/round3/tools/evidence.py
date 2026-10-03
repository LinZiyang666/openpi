"""Round-3 evidence, reproducible (all discovery inits; fitting-type questions on 0-19, screen questions on 20-29).

  crosstab     escalation arm vs its base on the round-2 screen, split by whether each arm's own trajectory crossed the
               frozen rule (pre-treatment-free attribution: only the both-triggered cell isolates the takeover effect)
  proximity    takeover success vs distance of the arm from its start pose at the trigger:
               (a) round-2 screen escalated episodes (inits 20-29, normalized server state),
               (b) R8 call arms after the same rule (inits 0-19, metric eef position from the client controls)
  futility     does library progress under the policy separate rescued from unrescued takeovers early? (screen)
  early        within-task AUROC of initial retrieval distance for always-fail vs always-succeed pairs (0-29 labels)
  b_calls      share of the guard controller's policy calls made while on the demonstration's pace, by pair type
RULE 1 everywhere: readers drop init >= 30 before decoding; holdout roots refused.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from exp.offline_search.rounds.r09.explore_opus.round2.tools.common import RUNS, DERIVED, iter_jsonl_discovery, parse_uid, dump
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import (catalog, signals, first_trigger,
                                                                               episodes_from_r8, load_cell)
from exp.offline_search.rounds.r09.explore_opus.round2.tools.replicates import load as rload, matrix
from .dissect import journal, server_decisions, SCREEN, PAIRS, episodes as screen_episodes

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def crosstab():
    res = {}
    for model, lib, a, b in PAIRS:
        cell = (model, "l10", lib)
        E = screen_episodes(SCREEN, f"r9o_{model}_l10_{lib}_{a}", model, cell)
        B = screen_episodes(SCREEN, f"r9o_{model}_l10_{lib}_{b}", model, cell)
        j = E.merge(B, on=["task", "init"], suffixes=("", "_b"))
        te, tb = j.esc_step.notna(), j.rule_step_b.notna()
        tab = {}
        for x in (0, 1):
            for y in (0, 1):
                s = j[(te == x) & (tb == y)]
                if len(s):
                    tab[f"esc{x}_base{y}"] = dict(n=int(len(s)), sr=float(s.success.mean()), base_sr=float(s.success_b.mean()),
                                                  up=int((s.success & ~s.success_b).sum()), down=int((~s.success & s.success_b).sum()))
        res[f"{model}_{lib}:{a}_vs_{b}"] = dict(total=dict(sr=float(j.success.mean()), base_sr=float(j.success_b.mean())),
                                                 cells=tab)
    return res


def proximity_screen():
    out = {}
    for model, lib, arm in [("pi05", 50, "esc"), ("pi05", 50, "esc_w24"), ("groot", 50, "esc"), ("groot", 50, "esc_w24"),
                            ("pi05", 500, "esc"), ("groot", 500, "esc")]:
        name = f"r9o_{model}_l10_{lib}_{arm}"
        J = journal(SCREEN, name)
        st = {}
        for f in glob.glob(str(SCREEN / "runs" / name / "server_*" / "decisions*.jsonl")):
            for r in iter_jsonl_discovery(f, uid_key="uid"):
                if r.get("ev") != "dec" or r.get("robot_state") is None:
                    continue
                t, i = parse_uid(r["uid"])
                if (t, i) not in J or int(r.get("attempt", 1)) != J[(t, i)][1]:
                    continue
                st.setdefault((t, i), []).append((int(r["step"]), np.asarray(r["robot_state"], float)[:3],
                                                  (r.get("extras") or {}).get("r9o_escalated", 0)))
        rec = []
        for k, v in st.items():
            v.sort(key=lambda x: x[0])
            e = [x for x in v if x[2] == 1]
            if e:
                rec.append(dict(success=J[k][0], dist=float(np.linalg.norm(e[0][1] - v[0][1]))))
        R = pd.DataFrame(rec)
        med = R.dist.median()
        out[name] = dict(n=len(R), rescued=int(R.success.sum()),
                         auroc_failed_farther=auc(R[~R.success].dist, R[R.success].dist),
                         rescue_near_half=float(R[R.dist <= med].success.mean()), rescue_far_half=float(R[R.dist > med].success.mean()),
                         units="normalized model state")
    return out


def proximity_fit():
    O = pd.read_parquet(DERIVED / "catalog" / "outcomes.parquet")
    out = {}
    for model in ("pi05", "groot"):
        cell = (model, "l10", 50)
        cat = catalog(cell)
        for a in ("A", "CU", "CT", "IP"):
            arm = f"r8_{model}_l10_50_{a}"
            o = O[(O.root == "r08_main") & (O.arm == arm)]
            eps = {(e.task, e.init): e for e in episodes_from_r8(DERIVED / "r8ledger" / f"{arm}.parquet",
                   {(int(x.task), int(x.init)): bool(x.success) for x in o.itertuples()})}
            rows = []
            for ep_dir in sorted((RUNS / "r08_main" / "runs" / arm / "debug" / "client").iterdir()):
                meta = json.loads((ep_dir / "episode.json").read_text())
                t, i = parse_uid(meta["task_uid"])
                if i >= 20 or (t, i) not in eps:
                    continue                           # FIT inits only; identity before arrays
                e = eps[(t, i)]
                k = first_trigger(e, signals(e, cat), "lag", 12, 0, 80)
                if k is None:
                    continue
                parts = [np.load(p) for p in sorted(ep_dir.glob("controls_*.npz"))]
                pos = np.concatenate([p["eef_pos"] for p in parts])
                ds = np.concatenate([p["decision_seq"] for p in parts])
                j = np.flatnonzero(ds == k)
                if len(j):
                    rows.append(dict(success=e.success, dist=float(np.linalg.norm(pos[j[0]] - pos[0]))))
            R = pd.DataFrame(rows)
            med = R.dist.median()
            out[f"{model}_{a}"] = dict(n=len(R), recovered=float(R.success.mean()), dist_median_m=float(med),
                                       auroc_failed_farther=auc(R[~R.success].dist, R[R.success].dist),
                                       recovery_near_half=float(R[R.dist <= med].success.mean()),
                                       recovery_far_half=float(R[R.dist > med].success.mean()))
    return out


def futility():
    out = {}
    for model, lib, arm in [("pi05", 50, "esc"), ("groot", 50, "esc"), ("pi05", 500, "esc"), ("groot", 500, "esc")]:
        name = f"r9o_{model}_l10_{lib}_{arm}"
        J = journal(SCREEN, name)
        X = server_decisions(SCREEN, name, {k: v[1] for k, v in J.items()})
        st = catalog((model, "l10", lib)).step.values
        rec = []
        for (t, i), e in X.groupby(["task", "init"]):
            esc = e[e.esc == 1]
            if not len(esc):
                continue
            k = int(esc.esc_step.iloc[0])
            f = e[e.vision & (e.top1 >= 0)]
            s0 = st[f[f.step <= k].top1.values[-1]]
            g = f[(f.step > k) & (f.step <= k + 8)]
            rec.append(dict(success=J[(t, i)][0], adv8=float(st[g.top1.values].max() - s0) if len(g) else np.nan))
        R = pd.DataFrame(rec)
        stop = R.adv8 < 3
        out[name] = dict(escalated=len(R), rescued=int(R.success.sum()),
                         median_advance_rescued=float(R[R.success].adv8.median()),
                         median_advance_failed=float(R[~R.success].adv8.median()),
                         rule_adv8_lt3_stops_rescued=int((stop & R.success).sum()), stops_failed=int((stop & ~R.success).sum()))
    return out


def early():
    arms, outc = rload()
    out = {}
    for model, suite, lib in [("pi05", "l10", 50), ("groot", "l10", 50), ("pi05", "l10", 500), ("groot", "l10", 500)]:
        d = outc[(outc.model == model) & (outc.suite == suite) & (outc.lib == lib)]
        p = matrix(d[d.family == "cache"]).mean(1)
        X = pd.read_parquet(DERIVED / "r8ledger" / f"r8_{model}_{suite}_{lib}_A.parquet")
        rows = []
        for (t, i), e in X.groupby(["task", "init"]):
            f = e[e.vision].sort_values("seq")
            rows.append(dict(task=t, init=i, d1=f.d1.iloc[:3].mean()))
        R = pd.DataFrame(rows)
        R["p"] = p.reindex(pd.MultiIndex.from_frame(R[["task", "init"]])).values
        wt = [auc(g[g.p == 0].d1, g[g.p == 1].d1) for _, g in R.groupby("task")]
        wt = [w for w in wt if w is not None]
        out[f"{model}_{suite}_{lib}"] = dict(always_fail=int((R.p == 0).sum()), always_succeed=int((R.p == 1).sum()),
                                            pooled_auroc=auc(R[R.p == 0].d1, R[R.p == 1].d1),
                                            mean_within_task_auroc=float(np.mean(wt)) if wt else None, tasks=len(wt))
    return out


def b_calls():
    arms, outc = rload()
    out = {}
    for cell in [("pi05", "l10", 50), ("groot", "l10", 50)]:
        st = catalog(cell).step.values
        d = outc[(outc.model == cell[0]) & (outc.suite == cell[1]) & (outc.lib == cell[2])]
        p = matrix(d[d.family == "cache"]).mean(1)
        rec = []
        for e in load_cell(cell, "guards_B"):
            if e.init >= 20:
                continue
            lag = dict(zip(e.seq, e.seq - st[e.rows[:, 0]]))
            for c in np.flatnonzero(e.cost >= 0.9):
                rec.append(dict(p=p.get((e.task, e.init), np.nan), success=e.success, lag=lag.get(c, np.nan)))
        R = pd.DataFrame(rec)
        R["pair"] = pd.cut(R.p, [-.01, .001, .999, 1.01], labels=["always_fail", "mixed", "always_succeed"])
        g = R.groupby(["pair", "success"], observed=True)
        out["_".join(map(str, cell))] = dict(calls=len(R), share_on_pace_lag_lt4=float((R.lag < 4).mean()),
                                             by_pair={f"{a}|{'S' if s else 'F'}": dict(calls=int(len(x)), share_of_calls=float(len(x) / len(R)),
                                                                                    on_pace=float((x.lag < 4).mean()))
                                                      for (a, s), x in g})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(crosstab=crosstab(), proximity_screen=proximity_screen(), proximity_fit=proximity_fit(),
               futility=futility(), early=early(), b_calls=b_calls())
    dump(OUT / "evidence.json", res)
    for k, v in res.items():
        print("==", k)
        for kk, vv in v.items():
            print("  ", kk, json.dumps(vv)[:400])


if __name__ == "__main__":
    main()
