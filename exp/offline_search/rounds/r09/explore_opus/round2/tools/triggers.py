"""Task-agnostic trouble signals computed from the cache's own retrieval, and an exact-prefix escalation simulator.

Signals (computed at every fresh decision from the top-1 library row; no task id, no policy, no simulator truth):
  lag      = decision index - library step of the top-1 neighbour (how far behind the retrieved demo's own pace)
  lagw     = decision index - weighted library step of the top-4 neighbours
  eos      = running count of fresh decisions whose top-1 library progress >= theta ("demo script exhausted")
  np       = no-progress span (decisions) of the top-1 library step, the B guard's statistic with a larger threshold

A *trigger rule* fires at the first fresh decision where its signal crosses a threshold.  The escalation simulator
takes the cache's real episodes (replicates, inits 0-29 only), cuts each at its trigger, keeps the exact cache prefix
(outcome, decisions, owner cost) and replaces the suffix by a policy takeover with an assumed rescue probability r.
Everything before the trigger is real data; ``r`` is the single unknown a closed-loop screen must measure.

RULE 1: inputs come from ``episodes.py`` / ``r8ledger.py`` which already dropped init >= 30; asserted again here.
Thresholds are chosen on FIT inits 0-19 and reported on EVAL inits 20-29.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DERIVED, OUT, RUNS, FIT_INITS, EVAL_INITS, PRICE, dump

CAT = RUNS / "r08_main" / "catalog"
LIBNAME = {("pi05", "l10", 50): "pi05_l10_current", ("pi05", "l10", 500): "pi05_l10_bpool_cs",
           ("pi05", "spatial", 50): "pi05_spatial_current", ("pi05", "spatial", 500): "pi05_spatial_bpool_cs",
           ("groot", "l10", 50): "groot_l10_current", ("groot", "l10", 500): "groot_l10_bpool_all",
           ("groot", "spatial", 50): "groot_spatial_current", ("groot", "spatial", 500): "groot_spatial_bpool_all"}
MAX_DEC = {"l10": 104, "spatial": 44}


def catalog(cell):
    return pd.read_parquet(CAT / LIBNAME[cell] / "rows.parquet").set_index("row")


@dataclass
class Episode:
    source: str
    task: int
    init: int
    success: bool
    n_dec: int            # decisions in the episode
    seq: np.ndarray       # decision index of each fresh look (with a retrieval)
    rows: np.ndarray      # (n_fresh, k) top-k library rows (k>=1), -1 where missing
    w: np.ndarray         # (n_fresh, k) weights (nan where missing)
    cost: np.ndarray      # owner cost of every decision (len n_dec)


def episodes_from_standard(path, model):
    """Standard-mode ledger (episodes.py): top-1 row only; owner cost reconstructed from kinds."""
    X = pd.read_parquet(path)
    assert X["init"].max() < 30
    v, m = PRICE[model]
    out = []
    for (t, i), e in X.groupby(["task", "init"], sort=True):
        e = e.sort_values("step")
        kind = e.kind.values
        cost = np.where(kind == 2, v + m, np.where(kind == 1, v, 0.0))
        fresh = (kind >= 1) & (e.row.values >= 0)
        seq = (e.step.values // 5)[fresh]
        out.append(Episode(str(path), int(t), int(i), bool(e.success.values[-1]), len(e), seq,
                           e.row.values[fresh][:, None].astype(int), np.ones((fresh.sum(), 1)), cost))
    return out


def episodes_from_r8(path, outcomes):
    """R8 debug server ledger (r8ledger.py): top-4 rows and weights; success from the journal outcomes."""
    X = pd.read_parquet(path)
    assert X["init"].max() < 30
    out = []
    for (t, i), e in X.groupby(["task", "init"], sort=True):
        e = e.sort_values("seq")
        fresh = e.vision.values & (e.r0.values >= 0)
        rows = e[["r0", "r1", "r2", "r3"]].values[fresh].astype(int)
        w = e[["w0", "w1", "w2", "w3"]].values[fresh].astype(float)
        out.append(Episode(str(path), int(t), int(i), bool(outcomes[(int(t), int(i))]), len(e), e.seq.values[fresh],
                           rows, w, np.nan_to_num(e.cost.values.astype(float))))
    return out


def signals(ep: Episode, cat: pd.DataFrame, theta=0.9, prog_eps=0.5):
    step = cat.step.values
    prog = cat.progress.values
    eplen = cat.ep_len.values
    r = ep.rows
    top = r[:, 0]
    lag = ep.seq - step[top]
    ww = np.where(r >= 0, np.nan_to_num(ep.w), 0.0)
    ww = ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)
    st = np.where(r >= 0, step[np.maximum(r, 0)], 0)
    lagw = ep.seq - (ww * st).sum(1)
    eos = np.cumsum(prog[top] >= theta)
    # B's no-progress span: accumulates decision gaps while the top-1 step does not advance
    span = np.zeros(len(top))
    for j in range(1, len(top)):
        adv = (prog[top[j]] - prog[top[j - 1]]) * max(eplen[top[j]] - 1, 1)
        span[j] = span[j - 1] + (ep.seq[j] - ep.seq[j - 1]) if adv <= prog_eps else 0
    return dict(lag=lag, lagw=lagw, eos=eos, np=span, prog=prog[top], time=ep.seq.astype(float))


def first_trigger(ep: Episode, sig: dict, rule: str, thr: float, smin: int = 0, deadline: int | None = None):
    """Decision index of the first fresh decision where rule fires (None if never or only after ``deadline``)."""
    x = sig[rule]
    ok = np.flatnonzero((x >= thr) & (ep.seq >= smin))
    if not len(ok):
        return None
    k = int(ep.seq[ok[0]])
    return None if deadline is not None and k > deadline else k


def simulate(eps, trig, r, model, suite, rescue_dec=20, success_r=None, window=None):
    """Exact-prefix escalation simulator.

    eps: list of Episode; trig: list of trigger decision index or None (same order).
    Policy takeover from the trigger: every other decision is a fresh policy call (P10 pattern, cost v+m), the other
    a policy tail (cost 0).  A rescued episode ends ``rescue_dec`` decisions after the trigger (capped by the time
    limit); an unrescued one runs to the time limit.  ``r`` = success probability after a takeover for episodes the
    cache would have failed; ``success_r`` = the same for triggered episodes the cache would still have recovered by
    itself.  Default (conservative): ``success_r = r`` -- a triggered episode is in trouble, so the policy's
    takeover success is not assumed to be the pure policy's success on healthy pairs.
    Returns expected SR and owner IR (pooled cost / pooled decisions).
    """
    v, m = PRICE[model]
    T = MAX_DEC[suite]
    success_r = r if success_r is None else success_r
    sr = cost = dec = 0.0
    n_trig = n_fa = 0
    for ep, k in zip(eps, trig):
        if k is None or k >= ep.n_dec:
            sr += ep.success
            cost += ep.cost.sum()
            dec += ep.n_dec
            continue
        n_trig += 1
        pre_cost = ep.cost[:k].sum()
        p = success_r if ep.success else r
        if ep.success:
            n_fa += 1
        rest_ok = min(rescue_dec, T - k)
        rest_fail = T - k
        for ok, prob in ((True, p), (False, 1 - p)):
            rest = rest_ok if ok else rest_fail
            if window is None:
                c = pre_cost + np.ceil(rest / 2) * (v + m)
            else:   # policy for ``window`` decisions, then the cache (a look every other decision)
                on = min(rest, window)
                c = pre_cost + np.ceil(on / 2) * (v + m) + np.ceil(max(rest - window, 0) / 2) * v
            cost += prob * c
            dec += prob * (k + rest)
            sr += prob * ok
    n = len(eps)
    return dict(sr=sr / n, ir=cost / dec, trig_rate=n_trig / n, false_alarm_rate=n_fa / n)


def evaluate_rules(eps, cat, model, suite, rules, rs=(0.3, 0.5, 0.7, 0.9), deadline=None):
    sigs = [signals(e, cat) for e in eps]
    res = []
    for rule, thr, smin in rules:
        trig = [first_trigger(e, s, rule, thr, smin, deadline) for e, s in zip(eps, sigs)]
        row = dict(rule=rule, thr=thr, smin=smin, deadline=deadline)
        for name, inits in (("fit", FIT_INITS), ("eval", EVAL_INITS)):
            idx = [j for j, e in enumerate(eps) if e.init in inits]
            fail = [j for j in idx if not eps[j].success]
            succ = [j for j in idx if eps[j].success]
            tf = [trig[j] for j in fail if trig[j] is not None]
            ts = [trig[j] for j in succ if trig[j] is not None and trig[j] < eps[j].n_dec]
            row[f"{name}_fail_detect"] = len(tf) / max(len(fail), 1)
            row[f"{name}_fail_t_med_controls"] = float(np.median(tf) * 5) if tf else None
            row[f"{name}_fail_before250"] = float(np.mean([t * 5 <= 250 for t in tf])) * len(tf) / max(len(fail), 1) if tf else 0.0
            row[f"{name}_succ_false_alarm"] = len(ts) / max(len(succ), 1)
            row[f"{name}_n"] = len(idx)
            for r in rs:
                s = simulate([eps[j] for j in idx], [trig[j] for j in idx], r, model, suite)
                row[f"{name}_sr_r{r}"] = s["sr"]
                row[f"{name}_ir_r{r}"] = s["ir"]
        base = simulate(eps, [None] * len(eps), 0, model, suite)
        row["cache_sr"], row["cache_ir"] = base["sr"], base["ir"]
        res.append(row)
    return pd.DataFrame(res)


DEFAULT_RULES = ([("time", T, 0) for T in (45, 50, 55, 60)] +
                 [("lag", L, 0) for L in (8, 10, 12, 15, 20, 25)] +
                 [("lagw", L, 0) for L in (8, 10, 12, 15, 20)] +
                 [("np", n, 0) for n in (4, 6, 8, 10, 12, 15)] +
                 [("eos", k, 0) for k in (3, 4, 6, 8)])


def load_cell(cell, family="cache"):
    """Replicate episodes of a family in a cell (standard ledgers for pi0.5, R8 debug ledger for GR00T)."""
    from .replicates import load as rload
    arms, outc = rload()
    model, suite, lib = cell
    a = arms[(arms.model == model) & (arms.suite == suite) & (arms.lib == lib) & (arms.family == family)]
    eps = []
    for r in a.itertuples():
        f = DERIVED / "episodes" / f"{r.root}__{r.arm}.parquet"
        if f.exists():
            X = pd.read_parquet(f, columns=["row", "kind"])
            if (X[X.kind >= 1].row >= 0).mean() > 0.9:
                eps += episodes_from_standard(f, model)
                continue
        sl = DERIVED / "serverledger" / f"{r.root}__{r.arm}.parquet"
        if sl.exists():
            from .serverledger import episodes as server_episodes
            o = outc[(outc.root == r.root) & (outc.arm == r.arm)]
            eps += server_episodes(sl, {(int(x.task), int(x.init)): bool(x.success) for x in o.itertuples()}, model)
            continue
        if r.root == "r08_main":
            g = DERIVED / "r8ledger" / f"{r.arm}.parquet"
            if g.exists():
                o = outc[(outc.root == r.root) & (outc.arm == r.arm)]
                eps += episodes_from_r8(g, {(int(x.task), int(x.init)): bool(x.success) for x in o.itertuples()})
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,pi05:l10:500,groot:l10:50,groot:l10:500,pi05:spatial:50,groot:spatial:50")
    ap.add_argument("--family", default="cache")
    ap.add_argument("--deadline", type=int, default=None, help="no escalation starts after this decision index")
    a = ap.parse_args(argv)
    allres = {}
    for c in a.cells.split(","):
        m, s, n = c.split(":")
        cell = (m, s, int(n))
        eps = load_cell(cell, a.family)
        if not eps:
            print("no episodes", cell)
            continue
        df = evaluate_rules(eps, catalog(cell), m, s, DEFAULT_RULES, deadline=a.deadline)
        key = f"{m}_{s}_{n}_{a.family}"
        allres[key] = dict(n_episodes=len(eps), sources=sorted({e.source.split('/')[-1] for e in eps}),
                           table=json.loads(df.to_json(orient="records")))
        pd.set_option("display.width", 250)
        cols = ["rule", "thr", "fit_fail_detect", "fit_fail_t_med_controls", "fit_fail_before250", "fit_succ_false_alarm",
                "eval_fail_detect", "eval_fail_t_med_controls", "eval_succ_false_alarm", "eval_sr_r0.5", "eval_ir_r0.5",
                "eval_sr_r0.7", "eval_ir_r0.7", "cache_sr", "cache_ir"]
        print("=====", key, "episodes", len(eps))
        print(df[cols].round(3).to_string())
    tag = f"_dl{a.deadline}" if a.deadline is not None else ""
    dump(OUT / "triggers" / f"triggers_{a.family}{tag}.json", allres)


if __name__ == "__main__":
    main()
