"""Estimator replay on the disagreement table: held-out coverage, FM-0/FM-1, cold start.

Reads the M1 table, restricts everything to the **calibration query set**
(out-of-library trajectories; in-library rows are diagnostics only), splits it
by trajectory into the ``fit`` / ``test`` halves the builder stamped, and
answers four engineering questions before any closed-loop run (plan §4 M1-5):

1. coverage -- curves fitted on the fit half in episode order, frozen, and
   scored on the test half: per tier and per score quartile, the tail rate
   ``E = P(d > q_pre(s))`` over finite-q rows, the finite-q share, the support
   kinds and the pinball loss; the single-layer LP on the same rows as a
   reference;
2. FM-0 versus FM-1 -- from the same initial state, walk the test half through
   the **real** ``ScoreHysteresisGate`` (skip / probe / lock, no-candidate
   decisions kept as MISS with no feedback) and the ladder dispatch at a
   tolerance, feeding only the executed tier (FM-0) or every tier on
   backbone-paid decisions (FM-1); report never-filled knots and cut paths;
3. cold start -- from empty windows through the same gate, how many updates
   until each tier first has a finite cut;
4. release gate -- the plan's numbers, written as PASS / FAIL with reasons.

Public interface: ``candidate_rows``, ``fit_curves``, ``coverage``, ``walk``,
``cold_start``, ``release_gate``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Optional

import numpy as np

from exp.online_rit import ladder3
from exp.online_rit.common import ALPHA, N_MIN, WINDOW, load_jsonl, sha256_file, write_json
from exp.online_rit.ir_replay import GateParams, calibration_rows
from openpi.cache.components.judge import HitType
from openpi.cache.components.online_rit import (
    ContinuationFeedback,
    OnlineRiskCurves,
    SOURCE_EXECUTED,
    SOURCE_SHADOW,
)
from openpi.cache.types import CheckpointID

COVERAGE_MAX_E = 0.10
MIN_FINITE_ROWS = 200
MIN_FINITE_EPISODES = 20
MIN_FINITE_SHARE = 0.80


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (str(r["trajectory_id"]), int(r["decision_id"])))


def candidate_rows(rows: list[dict], split: Optional[str] = None, *, include_library: bool = False) -> list[dict]:
    """Rows with a candidate, in (trajectory, decision) order; calibration set unless asked otherwise."""
    out = rows if include_library else calibration_rows(rows)
    out = [r for r in out if r.get("s") is not None]
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    return ordered(out)


def episode_rows(rows: list[dict]) -> list[list[dict]]:
    """Calibration episodes as ordered row lists, no-candidate rows included."""
    by: dict[str, list[dict]] = {}
    for r in calibration_rows(rows):
        by.setdefault(str(r["trajectory_id"]), []).append(r)
    return [sorted(v, key=lambda r: int(r["decision_id"])) for _, v in sorted(by.items())]


def _feedback(row: dict, tiers, source: str) -> list[ContinuationFeedback]:
    fb = []
    for t in tiers:
        d = row.get(t.d_column)
        if d is not None and math.isfinite(float(d)):
            fb.append(ContinuationFeedback(t.index, float(d), source))
    return fb


def fit_curves(rows: list[dict], knots: list[float], *, alpha=ALPHA, window=WINDOW, n_min=N_MIN, tiers=None, fixed_params=None) -> OnlineRiskCurves:
    tiers = tiers or ladder3.warm_tiers()
    c = OnlineRiskCurves(knots=knots, tier_indices=[t.index for t in tiers], alpha=alpha, window=window, n_min=n_min, fixed_params=fixed_params)
    for r in candidate_rows(rows):
        c.update_batch(float(r["s"]), _feedback(r, tiers, SOURCE_SHADOW), (str(r["trajectory_id"]), int(r["decision_id"])))
    return c


def score_bands(fit_rows: list[dict]) -> list[float]:
    """Fixed quartile edges of the calibration fit half (frozen for every consumer)."""
    s = np.array([float(r["s"]) for r in candidate_rows(fit_rows)])
    if s.size == 0:
        raise SystemExit("no calibration fit rows to derive score bands from")
    return [float(x) for x in np.quantile(s, [0.25, 0.5, 0.75])]


def coverage(curves: OnlineRiskCurves, test_rows: list[dict], *, alpha=ALPHA, tiers=None, bands: Optional[list[float]] = None) -> dict:
    tiers = tiers or ladder3.warm_tiers()
    test = candidate_rows(test_rows)
    qs = bands if bands is not None else score_bands(test_rows)
    out: dict = {"n_test_rows": len(test), "band_edges": qs, "tiers": {}}
    for t in tiers:
        cells: dict[str, dict] = {}
        finite = 0
        exceed = 0
        pinball = 0.0
        kinds: dict[str, int] = {}
        episodes: set[str] = set()
        for r in test:
            d = r.get(t.d_column)
            if d is None:
                continue
            s = float(r["s"])
            q, kind = curves.query(t.index, s)
            kinds[kind] = kinds.get(kind, 0) + 1
            if q is None:
                continue
            finite += 1
            episodes.add(str(r["trajectory_id"]))
            over = float(d) > q
            exceed += over
            resid = float(d) - q
            pinball += (1 - alpha) * max(resid, 0) + alpha * max(-resid, 0)
            band = sum(s > e for e in qs)
            cell = cells.setdefault(f"q{band}|{kind}", {"n": 0, "exceed": 0, "episodes": set()})
            cell["n"] += 1
            cell["exceed"] += over
            cell["episodes"].add(str(r["trajectory_id"]))
        for cell in cells.values():
            n_ep = len(cell.pop("episodes"))
            cell["n_episodes"] = n_ep
            cell["E"] = cell["exceed"] / cell["n"] if cell["n"] >= 30 and n_ep >= 10 else None
        out["tiers"][t.name] = {
            "n_finite": finite,
            "finite_share": finite / len(test) if test else None,
            "n_episodes": len(episodes),
            "E": exceed / finite if finite else None,
            "pinball_mean": pinball / finite if finite else None,
            "support_kinds": kinds,
            "cells": cells,
        }
    return out


def lp_reference(fit_rows: list[dict], test_rows: list[dict], knots, *, alpha=ALPHA, tiers=None) -> dict:
    tiers = tiers or ladder3.warm_tiers()
    fit_c = candidate_rows(fit_rows)
    s = np.array([float(r["s"]) for r in fit_c])
    ys = {t.name: np.array([float(r[t.d_column]) for r in fit_c]) for t in tiers}
    fits = ladder3.fit_independent(s, ys, np.asarray(knots), alpha=alpha, tiers=tiers)
    out = {}
    from exp.rit_pareto import rit_k

    test = candidate_rows(test_rows)
    for t in tiers:
        f = fits[t.name]
        st = np.array([float(r["s"]) for r in test])
        dt = np.array([float(r[t.d_column]) for r in test])
        q = rit_k.predict(f, st, t.name)
        out[t.name] = {"E": float(np.mean(dt > q)) if dt.size else None, "q": [float(x) for x in f.q[t.name]]}
    return out


def walk(start: OnlineRiskCurves, rows: list[dict], *, delta: float, mode: str, gate: GateParams, tiers=None, log_every: int = 200) -> dict:
    """Walk calibration episodes through the real gate from ``start``.

    FM-0 feeds the executed tier only; FM-1 feeds every tier on decisions that
    paid the backbone (executed warm start or candidate MISS). Gate skips and
    no-candidate decisions produce no feedback and count as MISS.
    """
    tiers = tiers or ladder3.warm_tiers()
    c = OnlineRiskCurves.from_snapshot(start.snapshot(), update_enabled=True)
    order = [t.index for t in tiers]
    trajectory: list[dict] = []
    counts = {"skip": 0, "no_candidate": 0, "miss": 0, **{t.name: 0 for t in tiers}}
    g = gate.build()
    n = 0
    for ep in episode_rows(rows):
        g.on_episode_start(str(ep[0]["trajectory_id"]))
        for r in ep:
            n += 1
            searched = g(CheckpointID.CP1, {}, None)
            s = r.get("s")
            if not searched:
                counts["skip"] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=False)
                continue
            if s is None:
                counts["no_candidate"] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=True)
                continue
            s = float(s)
            cuts = c.cuts(delta)
            chosen = ladder3.dispatch(s, [cuts[i] for i in order])
            if chosen is None:
                counts["miss"] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=s, winner_id="c", start_t=None, searched=True)
                fb = _feedback(r, tiers, SOURCE_SHADOW) if mode == "fm1" else []
            else:
                t = tiers[chosen]
                counts[t.name] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.WARM_START, cp1_score=s, winner_id="c", start_t=t.start_t, searched=True)
                if mode == "fm1":
                    fb = _feedback(r, tiers, SOURCE_SHADOW)
                else:
                    fb = [ContinuationFeedback(t.index, float(r[t.d_column]), SOURCE_EXECUTED)] if r.get(t.d_column) is not None else []
            if fb:
                c.update_batch(s, fb, ("walk", mode, n))
            if n % log_every == 0:
                trajectory.append({"n": n, "cuts": {t.name: (None if not math.isfinite(cuts[t.index]) else cuts[t.index]) for t in tiers}})
    empty = {t.name: float(np.mean(~c.valid(t.index))) for t in tiers}
    final = c.cuts(delta)
    return {
        "mode": mode, "delta": delta, "n_decisions": n, "counts": counts, "n_updates": c.n_updates,
        "empty_knot_share": empty, "cut_trajectory": trajectory,
        "final_cuts": {t.name: (None if not math.isfinite(final[t.index]) else final[t.index]) for t in tiers},
    }


def cold_start(rows: list[dict], knots, *, delta: float, gate: GateParams, tiers=None) -> dict:
    """From empty windows through the real gate under FM-1: first finite cut per tier."""
    tiers = tiers or ladder3.warm_tiers()
    empty = OnlineRiskCurves(knots=knots, tier_indices=[t.index for t in tiers], alpha=ALPHA, window=WINDOW, n_min=N_MIN)
    c = OnlineRiskCurves.from_snapshot(empty.snapshot(), update_enabled=True)
    order = [t.index for t in tiers]
    first: dict[str, Optional[int]] = {t.name: None for t in tiers}
    g = gate.build()
    n = 0
    for ep in episode_rows(rows):
        g.on_episode_start(str(ep[0]["trajectory_id"]))
        for r in ep:
            n += 1
            searched = g(CheckpointID.CP1, {}, None)
            s = r.get("s")
            if not searched:
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=False)
                continue
            if s is None:
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=True)
                continue
            s = float(s)
            cuts = c.cuts(delta)
            chosen = ladder3.dispatch(s, [cuts[i] for i in order])
            if chosen is None:
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=s, winner_id="c", start_t=None, searched=True)
            else:
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.WARM_START, cp1_score=s, winner_id="c", start_t=tiers[chosen].start_t, searched=True)
            fb = _feedback(r, tiers, SOURCE_SHADOW)
            if fb:
                c.update_batch(s, fb, ("cold", n))
            for t in tiers:
                if first[t.name] is None and math.isfinite(c.cut(t.index, delta)[0]):
                    first[t.name] = c.n_updates
    return {"first_finite_cut_after_updates": first, "n_decisions": n, "n_updates": c.n_updates, "shadowed": [t.name for t in tiers if t.index in c.shadowed(delta)]}


def release_gate(cov: dict) -> dict:
    reasons = []
    for name, t in cov["tiers"].items():
        if t["n_finite"] < MIN_FINITE_ROWS:
            reasons.append(f"{name}: {t['n_finite']} finite-q test rows < {MIN_FINITE_ROWS}")
        if t["n_episodes"] < MIN_FINITE_EPISODES:
            reasons.append(f"{name}: {t['n_episodes']} episodes < {MIN_FINITE_EPISODES}")
        if t["finite_share"] is None or t["finite_share"] < MIN_FINITE_SHARE:
            reasons.append(f"{name}: finite-q share {t['finite_share']} < {MIN_FINITE_SHARE}")
        if t["E"] is not None and t["E"] > COVERAGE_MAX_E:
            reasons.append(f"{name}: tail rate E={t['E']:.3f} > {COVERAGE_MAX_E}")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--knots", required=True, help="knots.json from fit_init_curves knots")
    ap.add_argument("--delta", type=float, required=True, help="tolerance for the FM walks and cold start")
    ap.add_argument("--gate-theta", type=float, required=True, help="the suite's hysteresis theta (R arm record)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_jsonl(args.table)
    knots_doc = json.loads(pathlib.Path(args.knots).read_text(encoding="utf-8"))
    knots = knots_doc["knots"]
    fit_rows = [r for r in rows if r.get("split") == "fit"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not fit_rows or not test_rows:
        raise SystemExit("table rows carry no fit/test split")
    gate = GateParams(theta=args.gate_theta)
    curves = fit_curves(fit_rows, knots)
    bands = knots_doc.get("score_bands") or score_bands(fit_rows)
    cov = coverage(curves, test_rows, bands=bands)
    result = {
        "table": args.table,
        "table_sha256": sha256_file(args.table),
        "knots": knots,
        "knots_sha256": sha256_file(args.knots),
        "gate": {"theta": args.gate_theta},
        "n_fit_rows": len(candidate_rows(fit_rows)),
        "n_calibration_episodes": len(episode_rows(rows)),
        "coverage": cov,
        "lp_reference": lp_reference(fit_rows, test_rows, knots),
        "walk_fm0": walk(curves, test_rows, delta=args.delta, mode="fm0", gate=gate),
        "walk_fm1": walk(curves, test_rows, delta=args.delta, mode="fm1", gate=gate),
        "cold_start": cold_start(rows, knots, delta=args.delta, gate=gate),
        "release_gate": release_gate(cov),
    }
    write_json(args.out, result)
    print(f"release gate: {result['release_gate']}")


if __name__ == "__main__":
    main()
