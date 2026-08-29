"""Phase 0 exploratory summary: anchor A-4 gate + cost/tier mix per arm.

Outcome policy (plan section 3.5): the anchor's success rate is RECORDED
(it is the A-4 exact-stack baseline input) but never judged; surface arms
emit cost and tier mix only -- no SR key exists in their output. All fields
carry ``posthoc_exploratory=true``.

A-4 is mechanical: every cell accepted, every verdict MISS, every cell's
decision count equal to its client_timing inference count, and the
decision-weighted cost equal to ``unit_cost("MISS")`` at full precision.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.analysis.analytic_cost import (
    cost_matches,
    cost_model_digest,
    cost_model_payload,
    unit_cost,
)
from exp.dispatch_surface.analysis.precheck_io import (
    load_accepted_cells_costonly,
    load_accepted_episodes,
    load_cost_cells_costonly,
)
from exp.dispatch_surface.phase0_roster import FAMILY_ANCHOR
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, NUM_TASKS, official_test_inits


def anchor_gate(cells: dict, verdict_counts: dict, n_cells_expected: int) -> dict:
    """A-4 mechanical judgement for the always-full-inference anchor."""
    problems = []
    if len(cells) != n_cells_expected:
        problems.append(f"{len(cells)} accepted cells, expected {n_cells_expected}")
    counts = dict(verdict_counts)
    if set(counts) - {"MISS"} or counts.get("MISS", 0) <= 0:
        problems.append(f"verdicts are not 100% MISS: {counts}")
    num = sum(c for c, _n in cells.values())
    den = sum(n for _c, n in cells.values())
    realized = num / den if den else float("nan")
    expected = unit_cost("MISS", None)
    if not cost_matches(realized, expected):
        problems.append(f"ratio-of-sums {realized!r} != unit_cost(MISS) {expected!r}")
    return {"passed": not problems, "problems": problems,
            "realized_cost_ms": realized, "expected_cost_ms": expected,
            "n_cells": len(cells), "verdict_counts": counts}


def summarize(args) -> dict:
    ctx = phase0_discipline.validate(args.arm_matrix, args.launch_manifest, args.split_manifest,
                                     trials=args.trials)
    officials = official_test_inits(args.split_manifest, args.trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    if len(grid) != NUM_TASKS * args.trials:
        raise SystemExit("adjudicated grid size != tasks x trials")
    partial = bool(args.executed_only) and not ctx["roster_complete"]
    if not args.executed_only and not ctx["roster_complete"]:
        raise SystemExit("ledger did not execute the full roster; a formal summary needs every arm "
                         "(use --executed-only for a partial, non-adjudicative view)")
    arms = [a for a in ctx["arms"] if a in set(ctx["executed_arms"])] if args.executed_only else ctx["arms"]
    accepted = load_accepted_cells_costonly(args.journal, arms, grid)
    executed = {k: set(v) for k, v in ctx["executed_arms_by_run"].items()}
    phase0_discipline.assert_rows_claimed(accepted, executed, what="phase0 summary")
    cells, cost_summary = load_cost_cells_costonly(args.per_step, arms, accepted, officials)
    out = {
        "protocol": "dispatch_surface_rev2_phase0",
        "posthoc_exploratory": True,
        "partial_nonadjudicative": partial,
        "discipline": {k: v for k, v in ctx.items() if not k.startswith("_")},
        "cost_model": cost_model_payload(),
        "cost_model_digest": cost_model_digest(),
        "per_step_sha256": cost_summary["per_step_sha256"],
        "arms": {},
    }
    for arm in arms:
        family = ctx["families"][arm]
        num = sum(c for c, _n in cells[arm].values())
        den = sum(n for _c, n in cells[arm].values())
        entry = {
            "family": family,
            "quantile": ctx["quantiles"].get(arm),
            "delta": ctx["deltas"].get(arm),
            "realized_cost_ms": num / den,
            "n_decisions": int(den),
            "n_cells": len(cells[arm]),
            "verdict_counts": dict(cost_summary["verdict_counts"][arm]),
            "cells": {f"{t}:{i}": [c, n] for (t, i), (c, n) in sorted(cells[arm].items())},
        }
        if family == FAMILY_ANCHOR:
            gate = anchor_gate(cells[arm], cost_summary["verdict_counts"][arm], len(grid))
            entry["a4"] = gate
            # The anchor SR is recorded (A-4 baseline input), never judged.
            outcomes = load_accepted_episodes(args.journal, [arm], grid)[arm]
            phase0_discipline.assert_rows_claimed({arm: outcomes}, executed, what="anchor outcomes")
            entry["sr_recorded_not_judged"] = sum(1 for r in outcomes.values() if r["success"]) / len(outcomes)
            if not gate["passed"]:
                raise SystemExit(f"anchor arm {arm} fails A-4: {gate['problems']}")
        out["arms"][arm] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--launch-manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True)
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    ap.add_argument("--executed-only", action="store_true",
                    help="summarise only arms the ledger executed (partial runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = summarize(args)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({a: {k: v for k, v in e.items() if k != "cells"} for a, e in out["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
