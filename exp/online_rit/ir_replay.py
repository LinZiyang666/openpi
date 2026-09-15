"""Full per-episode replay of gate + dispatch + feedback billing, and the delta search.

The row-wise ``delta_for_ir`` of the offline lines prices each shadow row by
its own tier and ignores the gate. The hysteresis gate's next decision depends
on this decision's verdict (its skip / probe / lock state machine), and under
``fm1`` a warm start pays a batch-2 side step while a candidate MISS pays a
batch-3 one, so the inference ratio of a tolerance is a property of whole
episode sequences. This module replays every calibration episode of the
disagreement table through the *real* ``ScoreHysteresisGate`` and the ladder
dispatch, prices every decision through ``CostLedger`` (side steps included),
and searches delta for each target ratio: complete nested uniform grids from
513 through ``MAX_GRID_POINTS``, plus all finite initial q values in the bracket.
IR(delta) is never assumed monotone; a target no evaluated delta reaches
within tolerance is reported as not found, never fabricated.

Feedback modes priced: ``fm1`` (warm -> batch 2, candidate MISS -> batch 3),
``fm0`` (executed tier only, no side step) and ``none`` (a rule without online
feedback, i.e. R's ``threshold`` judge). The R' branch forces ``none``.

Public interface: ``Episode``, ``episodes_from_table``, ``replay_ir``,
``search_delta``, ``cuts_from_state``, ``cuts_from_rprime``, ``GateParams``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
from typing import Callable, Optional

import numpy as np

from exp.online_rit import ladder3
from exp.online_rit.common import (
    IR_TOLERANCE_PP,
    TARGET_IRS,
    TIER_TS,
    CostLedger,
    load_jsonl,
    load_ledger,
    sha256_file,
    write_json,
)
from openpi.cache.components.gate import ScoreHysteresisGate
from openpi.cache.components.judge import HitType
from openpi.cache.components.online_rit import OnlineRiskCurves
from openpi.cache.types import CheckpointID

GRID_POINTS = 513
MAX_GRID_POINTS = 4097
FEEDBACK_MODES = ("fm1", "fm0", "none")


@dataclasses.dataclass(frozen=True)
class GateParams:
    theta: float
    j: int = 3
    probe_interval: int = 3
    L: int = 6
    include_ws: bool = True

    def build(self) -> ScoreHysteresisGate:
        return ScoreHysteresisGate(
            theta_low=self.theta, theta_high=self.theta, j=self.j, probe_interval=self.probe_interval, L=self.L, include_ws=self.include_ws
        )


@dataclasses.dataclass(frozen=True)
class Episode:
    trajectory_id: str
    scores: tuple[Optional[float], ...]  # None = searched but no candidate


def calibration_rows(rows: list[dict]) -> list[dict]:
    """The calibration query set: out-of-library trajectories only (plan §3.5 / B6)."""
    return [r for r in rows if not r.get("in_library")]


def episodes_from_table(rows: list[dict]) -> list[Episode]:
    by: dict[str, list[tuple[int, Optional[float]]]] = {}
    for r in calibration_rows(rows):
        s = r.get("s")
        by.setdefault(str(r["trajectory_id"]), []).append((int(r["decision_id"]), None if s is None else float(s)))
    out = []
    for traj in sorted(by):
        seq = sorted(by[traj])
        out.append(Episode(traj, tuple(s for _, s in seq)))
    return out


CutProvider = Callable[[float], list[float]]


def cuts_from_state(curves: OnlineRiskCurves, tier_indices: list[int]) -> CutProvider:
    def provider(delta: float) -> list[float]:
        c = curves.cuts(delta)
        return [c[i] for i in tier_indices]

    return provider


def cuts_from_rprime(fit, tier_names: list[str]) -> CutProvider:
    def provider(delta: float) -> list[float]:
        c = ladder3.rprime_cuts(fit, delta)
        return [c[n] for n in tier_names]

    return provider


@dataclasses.dataclass
class ReplayResult:
    ir_percent: float
    ir_percent_no_fb: float
    n_decisions: int
    counts: dict[str, int]
    fb_batches: dict[str, int]
    trajectory_sha256: str


def _fb_batch(feedback_mode: str, hit: str) -> int:
    if feedback_mode == "fm1":
        return 2 if hit == "WARM_START" else 3
    return 0


def replay_ir(
    episodes: list[Episode],
    cuts: list[float],
    ledger: CostLedger,
    gate: GateParams,
    *,
    feedback_mode: str = "fm1",
    tier_ts: tuple[float, ...] = TIER_TS,
    update_enabled: bool = True,
) -> ReplayResult:
    """Price one cut vector over every episode with the real gate state machine."""
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")
    total = 0.0
    total_no_fb = 0.0
    n = 0
    counts = {"MISS": 0, "MISS_no_candidate": 0, "SKIP": 0}
    for t in tier_ts:
        counts[f"WARM@{t:g}"] = 0
    batches = {"0": 0, "2": 0, "3": 0}
    trace = hashlib.sha256()
    g = gate.build()
    for ep in episodes:
        g.on_episode_start(ep.trajectory_id)
        for s in ep.scores:
            n += 1
            searched = g(CheckpointID.CP1, {}, None)
            if not searched:
                hit, start_t, batch = "MISS", None, 0
                counts["SKIP"] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=False)
                trace.update(b"S")
            elif s is None:
                hit, start_t, batch = "MISS", None, 0
                counts["MISS_no_candidate"] += 1
                g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=None, winner_id=None, start_t=None, searched=True)
                trace.update(b"N")
            else:
                idx = ladder3.dispatch(s, cuts)
                if idx is None:
                    hit, start_t = "MISS", None
                    batch = _fb_batch(feedback_mode, hit)
                    counts["MISS"] += 1
                    g.record_verdict(CheckpointID.CP1, hit_type=HitType.MISS, cp1_score=s, winner_id="c", start_t=None, searched=True)
                    trace.update(b"M")
                else:
                    hit, start_t = "WARM_START", tier_ts[idx]
                    batch = _fb_batch(feedback_mode, hit)
                    counts[f"WARM@{start_t:g}"] += 1
                    g.record_verdict(CheckpointID.CP1, hit_type=HitType.WARM_START, cp1_score=s, winner_id="c", start_t=start_t, searched=True)
                    trace.update(bytes([idx]))
            pricing = {"online": feedback_mode != "none", "update_enabled": update_enabled,
                       "has_candidate": searched and s is not None}
            base = ledger.decision_ms(hit, start_t, 0, include_feedback=False, **pricing)
            total_no_fb += base
            total += ledger.decision_ms(hit, start_t, batch, **pricing)
            batches[str(batch)] += 1
        total += ledger.episode_end_ms(online=feedback_mode != "none", update_enabled=update_enabled)
    if n == 0:
        raise ValueError("no decisions to replay")
    denom = n * ledger.miss_ms
    return ReplayResult(
        ir_percent=100.0 * total / denom,
        ir_percent_no_fb=100.0 * total_no_fb / denom,
        n_decisions=n,
        counts=counts,
        fb_batches=batches,
        trajectory_sha256=trace.hexdigest(),
    )


def search_delta(
    episodes: list[Episode],
    provider: CutProvider,
    ledger: CostLedger,
    gate: GateParams,
    *,
    delta_lo: float,
    delta_hi: float,
    targets: tuple[float, ...] = TARGET_IRS,
    tol_pp: float = IR_TOLERANCE_PP,
    feedback_mode: str = "fm1",
    extra_deltas: tuple[float, ...] = (),
    evaluate: Optional[Callable[[float], ReplayResult]] = None,
    update_enabled: bool = True,
) -> dict:
    """Complete nested uniform grids plus q knots; distinct dispatch traces only.

    ``evaluate`` (delta -> ReplayResult) defaults to the real replay and is
    injectable so the search logic can be tested on synthetic, non-monotone
    IR(delta) shapes.
    """
    if not (math.isfinite(delta_lo) and math.isfinite(delta_hi) and delta_hi > delta_lo >= 0):
        raise ValueError("delta bracket must be finite with hi > lo >= 0")
    cache: dict[float, ReplayResult] = {}

    def ev(delta: float) -> ReplayResult:
        if delta not in cache:
            cache[delta] = (
                evaluate(delta) if evaluate is not None
                else replay_ir(episodes, provider(delta), ledger, gate, feedback_mode=feedback_mode, update_enabled=update_enabled)
            )
        return cache[delta]

    def grid(n_points: int) -> list[float]:
        pts = set(np.linspace(delta_lo, delta_hi, n_points).tolist())
        pts |= {float(x) for x in extra_deltas if delta_lo <= x <= delta_hi}
        return sorted(pts)

    for d in grid(GRID_POINTS):
        ev(d)

    def best_for(target: float) -> tuple[float, ReplayResult]:
        return min(cache.items(), key=lambda kv: (abs(kv[1].ir_percent - target), kv[0]))

    def hit(target: float) -> bool:
        return abs(best_for(target)[1].ir_percent - target) <= tol_pp

    # Complete each nested grid. Extra q knots never consume the uniform budget.
    n_points = GRID_POINTS
    while not all(hit(target) for target in targets) and n_points < MAX_GRID_POINTS:
        n_points = min(2 * n_points - 1, MAX_GRID_POINTS)
        for d in grid(n_points):
            ev(d)

    found: dict[str, dict] = {}
    for target in targets:
        delta, res = best_for(target)
        found[f"{target:g}"] = {
            "target": target,
            "found": abs(res.ir_percent - target) <= tol_pp,
            "delta": delta,
            "ir_percent": res.ir_percent,
            "ir_percent_no_fb": res.ir_percent_no_fb,
            "counts": res.counts,
            "fb_batches": res.fb_batches,
            "trajectory_sha256": res.trajectory_sha256,
            "cuts": [None if not math.isfinite(c) else c for c in provider(delta)] if evaluate is None else None,
            "duplicate_of": None,
        }
    # de-duplicate only among found targets: identical dispatch trace = one working point
    seen: dict[str, str] = {}
    for key, rec in found.items():
        if not rec["found"]:
            continue
        sig = rec["trajectory_sha256"]
        rec["duplicate_of"] = seen.get(sig)
        seen.setdefault(sig, key)
    irs = [r.ir_percent for r in cache.values()]
    return {
        "delta_bracket": [delta_lo, delta_hi],
        "n_evaluated": len(cache),
        "uniform_grid_points": n_points,
        "extra_deltas": sorted({float(x) for x in extra_deltas if delta_lo <= x <= delta_hi}),
        "envelope": {"min_ir": min(irs), "max_ir": max(irs)},
        "targets": found,
        "tolerance_pp": tol_pp,
        "feedback_mode": feedback_mode,
        "gate": dataclasses.asdict(gate),
        "ledger": ledger.source,
        "evaluations": sorted(({"delta": d, "ir_percent": r.ir_percent} for d, r in cache.items()), key=lambda x: x["delta"]),
    }


def delta_bracket_from_table(rows: list[dict], curves: OnlineRiskCurves) -> tuple[float, float]:
    """Lower bound = max over tiers of the d_self p95; upper = max finite curve value (+ slack)."""
    lo = 0.0
    for col in ("d_self_7", "d_self_6", "d_self_4"):
        vals = np.array([r[col] for r in rows if r.get(col) is not None], dtype=np.float64)
        if vals.size:
            lo = max(lo, float(np.quantile(vals, 0.95)))
    hi = lo
    for i in curves.tier_indices:
        q = curves.q_values(i)
        finite = q[np.isfinite(q)]
        if finite.size:
            hi = max(hi, float(finite.max()))
    hi = hi + max(1e-6, 1e-6 * hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="disagreement table jsonl")
    ap.add_argument("--state", required=True, help="init_state.json (frozen d curves) or rprime_fit.json with --rprime")
    ap.add_argument("--rprime", action="store_true", help="R' (threshold judge): priced with no online feedback")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--gate-theta", type=float, required=True)
    ap.add_argument("--feedback-mode", default="fm1", choices=("fm0", "fm1"))
    ap.add_argument("--frozen", action="store_true", help="price frozen online curves without learning costs")
    ap.add_argument("--targets", default=",".join(f"{t:g}" for t in TARGET_IRS))
    ap.add_argument("--provisional", action="store_true", help="ledger may lack fb_batch_ms (M1 addressing)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.table)
    episodes = episodes_from_table(rows)
    ledger = load_ledger(args.ledger, require_fb=not args.provisional)
    if args.provisional and not ledger.fb_batch_ms:
        # provisional: price a side step as one Euler step at any batch size
        ledger = dataclasses.replace(ledger, fb_batch_ms={2: ledger.stage3_step_ms, 3: ledger.stage3_step_ms})
    gate = GateParams(theta=args.gate_theta)
    state = json.loads(pathlib.Path(args.state).read_text(encoding="utf-8"))
    tiers = ladder3.warm_tiers()
    if args.rprime:
        from exp.online_rit.fit_init_curves import rprime_fit_from_record

        fit = rprime_fit_from_record(state)
        provider = cuts_from_rprime(fit, [t.name for t in tiers])
        q_all = np.concatenate([np.asarray(v) for v in fit.q.values()])
        lo, hi = 0.0, float(q_all.max()) * 1.05 + 1e-6
        feedback_mode = "none"
    else:
        curves = OnlineRiskCurves.from_snapshot(state, update_enabled=False)
        provider = cuts_from_state(curves, [t.index for t in tiers])
        lo, hi = delta_bracket_from_table(rows, curves)
        q_all = np.concatenate([curves.q_values(t.index) for t in tiers])
        feedback_mode = args.feedback_mode
    targets = tuple(float(x) for x in args.targets.split(",") if x.strip())
    result = search_delta(episodes, provider, ledger, gate, delta_lo=lo, delta_hi=hi, targets=targets, feedback_mode=feedback_mode, extra_deltas=tuple(float(q) for q in q_all if np.isfinite(q)), update_enabled=not args.frozen)
    result["provisional"] = bool(args.provisional)
    result["update_enabled"] = not args.frozen
    result["state"] = args.state
    result["state_sha256"] = sha256_file(args.state)
    result["table_sha256"] = sha256_file(args.table)
    result["ledger_sha256"] = sha256_file(args.ledger)
    result["n_episodes"] = len(episodes)
    result["rule"] = "rprime" if args.rprime else "online"
    write_json(args.out, result)
    reach = [k for k, v in result["targets"].items() if v["found"]]
    print(f"envelope {result['envelope']}  reached {reach}  (n_eval={result['n_evaluated']}, feedback={feedback_mode})")


if __name__ == "__main__":
    main()
