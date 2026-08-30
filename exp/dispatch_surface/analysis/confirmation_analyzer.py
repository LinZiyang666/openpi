"""Confirmation analyzer for fresh C (confirmation plan 3.6-c).

Without an unseal record only the cost-only view is allowed. With it, the
analyzer re-checks every digest the seal froze, loads outcomes, runs the ONE
H1 implementation (``h1_verdict.evaluate_h1_verdict``) under the sealed
interval and roster, reports H2 / S0-T descriptively (no pass field), the
studentized max-t simultaneous band at B_1 / B_2, the step-envelope
sensitivity, re-checks the anchor (A-4) and refuses to emit any Action Cache
comparison field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface import seal_confirmation
from exp.dispatch_surface.action_cache_decision import assert_no_action_cache_fields
from exp.dispatch_surface.analysis import budget_mixture as bm
from exp.dispatch_surface.analysis.analytic_cost import cost_matches, unit_cost
from exp.dispatch_surface.analysis.confirmation_io import load_accepted_c, load_cost_cells_c
from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest
from exp.dispatch_surface.analysis.h1_verdict import (
    FrozenDesign,
    audit_replicate_indices,
    cell_arrays,
    evaluate_h1_verdict,
    evaluate_hypothesis,
    full_sample_stats,
    stats_for_index,
    verdict_summary,
)
from exp.dispatch_surface.cost_map_api import index_arrays, shared_index
from exp.dispatch_surface.phase0_roster import ANCHOR_ARM, FAMILY_S0, FAMILY_SV, FAMILY_THRESHOLD
from exp.dispatch_surface.run_precheck import NUM_TASKS

PROTOCOL = "dispatch_surface_rev2_confirmation_result"
R = 10000
BOOTSTRAP_SEED = 20260829


def _sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _load_outcomes(journal_path: str, arms: list[str], grid) -> dict[str, dict]:
    """(task, prefix) -> success for accepted rows; the only outcome read in this module."""
    out = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        if row.get("yaml_id") not in out or row.get("accepted") is not True:
            continue
        from exp.dispatch_surface.analysis.precheck_io import parse_task_uid
        _arm, t, i = parse_task_uid(row["task_uid"])
        if row.get("status") not in ("done", "failed") or not isinstance(row.get("success"), bool) \
                or ((row["status"] == "done") != row["success"]):
            raise SystemExit(f"arm {_arm} cell {(t, i)}: terminal status/success schema is invalid")
        out[_arm][(t, i)] = float(row["success"])
    for a in arms:
        if set(out[a]) != set(grid):
            raise SystemExit(f"arm {a}: outcome rows do not cover the sealed grid")
    return out


def cost_only_view(seal_path: str, journal: str, per_step: str) -> dict:
    seal, seal_sha = seal_confirmation.load_seal(seal_path)
    n = int(seal["N"])
    arms = list(seal["roster"]["arms"])
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(n)}
    acc = load_accepted_c(journal, arms, grid)
    cells, summary = load_cost_cells_c(per_step, arms, acc)
    return {"protocol": PROTOCOL, "cost_only": True, "seal_sha256": seal_sha,
            "arms": {a: {"cost_ms": sum(c for c, _n in cells[a].values()) / sum(nn for _c, nn in cells[a].values()),
                         "decisions": int(sum(nn for _c, nn in cells[a].values())),
                         "verdict_counts": summary["verdict_counts"][a]} for a in arms}}


def analyze(seal_path: str, unseal_path: str, discipline_path: str, ledger_path: str, journal: str, per_step: str) -> dict:
    seal, seal_sha = seal_confirmation.load_seal(seal_path)
    unseal = json.loads(pathlib.Path(unseal_path).read_text())
    if unseal.get("seal_sha256") != seal_sha or unseal.get("discipline_sha256") != _sha(discipline_path) \
            or unseal.get("ledger_sha256") != _sha(ledger_path):
        raise SystemExit("unseal record does not bind this seal / discipline / ledger")
    disc = json.loads(pathlib.Path(discipline_path).read_text())
    from exp.dispatch_surface.analysis import confirmation_discipline as cdisc
    if disc.get("protocol") != cdisc.PROTOCOL or disc.get("passed") is not True:
        raise SystemExit("confirmation discipline artifact is not a passing certification")
    # the discipline must certify exactly this seal / ledger / task plan / pool / journal / per-step set (G2R1-B7)
    bound = {"seal_sha256": seal_sha, "ledger_sha256": _sha(ledger_path),
             "task_plan_sha256": seal["confirmation_task_plan_sha256"], "pool_manifest_sha256": seal["pool"]["manifest_sha256"],
             "journal_sha256": _sha(journal), "per_step_sha256": _sha(per_step), "suite": seal["suite"], "N": int(seal["N"]),
             "arms": list(seal["roster"]["arms"]), "roster_complete": True}
    for key, val in bound.items():
        if disc.get(key) != val:
            raise SystemExit(f"confirmation discipline {key} does not certify the current inputs")
    if unseal.get("task_plan_sha256") != bound["task_plan_sha256"] or unseal.get("journal_sha256") != bound["journal_sha256"] \
            or unseal.get("per_step_sha256") != bound["per_step_sha256"]:
        raise SystemExit("unseal record does not bind this task plan / journal / per-step set")
    if seal["estimator_digest"] != budget_mixture_digest():
        raise SystemExit("seal estimator != budget_mixture_v1")
    n = int(seal["N"])
    arms = list(seal["roster"]["arms"])
    fams = seal["roster"]["families"]
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(n)}
    acc = load_accepted_c(journal, arms, grid)
    cost_cells, summary = load_cost_cells_c(per_step, arms, acc)
    succ = _load_outcomes(journal, arms, grid)
    cells = {a: {k: (cost_cells[a][k][0], cost_cells[a][k][1], succ[a][k]) for k in cost_cells[a]} for a in arms}
    # A-4 on the anchor, re-checked on C
    a4 = None
    if ANCHOR_ARM in arms:
        num = sum(c for c, _n in cost_cells[ANCHOR_ARM].values())
        den = sum(nn for _c, nn in cost_cells[ANCHOR_ARM].values())
        counts = summary["verdict_counts"][ANCHOR_ARM]
        a4 = {"passed": bool(set(k for k, v in counts.items() if v) == {"MISS"} and cost_matches(num / den, unit_cost("MISS", None))),
              "verdict_counts": counts, "cost_ms": num / den, "sr_recorded_not_judged": float(np.mean(list(succ[ANCHOR_ARM].values())))}
        if not a4["passed"]:
            raise SystemExit(f"anchor fails A-4 on C: {a4}")
    B_L, B_H = seal["budget_interval"]["B_L"], seal["budget_interval"]["B_H"]
    B_1, B_2 = seal["budget_interval"]["B_1"], seal["budget_interval"]["B_2"]
    roster = {fam: [a for a in arms if fams[a] == fam] for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD)}
    by_task: dict[int, list] = {}
    for k in sorted(grid):
        by_task.setdefault(k[0], []).append(k)
    picks = shared_index(by_task, BOOTSTRAP_SEED, R)
    cell_list, idx = index_arrays(picks, sorted(grid))
    index_sha = hashlib.sha256(json.dumps(picks, separators=(",", ":")).encode()).hexdigest()
    audit_idx = audit_replicate_indices(R, estimator_digest=budget_mixture_digest(), input_digest=_sha(journal))
    fam_arms = roster[FAMILY_SV] + roster[FAMILY_S0] + roster[FAMILY_THRESHOLD]
    arrays = cell_arrays(cells, fam_arms)
    full = full_sample_stats(arrays, fam_arms)
    out = {"protocol": PROTOCOL, "suite": seal["suite"], "N": n, "seal_sha256": seal_sha,
           "unseal_sha256": _sha(unseal_path), "discipline_sha256": _sha(discipline_path),
           "estimator_version": budget_mixture_digest(), "analyzer_version": seal["analyzer_version"],
           "budget_interval": seal["budget_interval"], "bootstrap": {"R": R, "seed": BOOTSTRAP_SEED, "index_sha256": index_sha},
           "audit_replicates": audit_idx, "A4_anchor": a4, "families": {}, "hypotheses": {}}
    for fam in roster:
        st = {a: full[a] for a in roster[fam]}
        block = {"arms": roster[fam], "measured_policies": {a: {"cost": full[a].c, "sr": full[a].s} for a in roster[fam]},
                 "standalone_dominated": bm.standalone_dominance(roster[fam], st), "feasible_at_B_L": bm.feasible(roster[fam], st, B_L)}
        if block["feasible_at_B_L"]:
            block.update(bm.active_basis_union(roster[fam], st, B_L, B_H))
        out["families"][fam] = block
    h1_des = FrozenDesign(family_a=FAMILY_SV, family_b=FAMILY_THRESHOLD, roster=roster, B_L=B_L, B_H=B_H, R=R)
    h1 = evaluate_h1_verdict(cells, h1_des, idx, audit_replicates=audit_idx)
    out["hypotheses"]["H1"] = verdict_summary(h1)
    out["verdict"] = "h1_pass" if h1.passed else ("support_miss" if h1.reason in ("left_support_fail", "joint_miss_exceeds") else "h1_fail")
    for name, (fa, fb) in (("H2", (FAMILY_SV, FAMILY_S0)), ("S0_minus_T", (FAMILY_S0, FAMILY_THRESHOLD))):
        if not roster[fa] or not roster[fb]:
            continue
        v = evaluate_hypothesis(cells, FrozenDesign(family_a=fa, family_b=fb, roster=roster, B_L=B_L, B_H=B_H, R=R), idx,
                                audit_replicates=audit_idx)
        s = verdict_summary(v)
        s.pop("passed", None)
        s["inferential"] = False
        out["hypotheses"][name] = s
    # secondary: studentized max-t simultaneous 95% band for delta V at B_1, B_2 (exploratory)
    deltas = np.full((R, 2), np.nan)
    plug = []
    for j, b in enumerate((B_1, B_2)):
        va = bm.value_at(roster[FAMILY_SV], {a: full[a] for a in roster[FAMILY_SV]}, b)[0]
        vb = bm.value_at(roster[FAMILY_THRESHOLD], {a: full[a] for a in roster[FAMILY_THRESHOLD]}, b)[0]
        plug.append(None if va is None or vb is None else va - vb)
    for r in range(R):
        st = stats_for_index(arrays, roster[FAMILY_SV] + roster[FAMILY_THRESHOLD], idx[r])
        for j, b in enumerate((B_1, B_2)):
            xa = bm.value_at(roster[FAMILY_SV], st, b)[0]
            xb = bm.value_at(roster[FAMILY_THRESHOLD], st, b)[0]
            deltas[r, j] = np.nan if (xa is None or xb is None) else xa - xb
    ok = ~np.isnan(deltas).any(axis=1)
    band = {"points": [B_1, B_2], "plugin": plug, "support_miss_rate": float(1.0 - ok.mean()), "exploratory": True}
    if ok.sum() > 1 and all(p is not None for p in plug):
        d = deltas[ok]
        sd = d.std(axis=0, ddof=1)
        if (sd > 0).all():
            tstat = np.abs((d - np.array(plug)) / sd)
            crit = float(np.quantile(tstat.max(axis=1), 0.95))
            band.update({"sd": sd.tolist(), "critical_max_t": crit,
                         "lower": (np.array(plug) - crit * sd).tolist(), "upper": (np.array(plug) + crit * sd).tolist()})
    out["secondary_band"] = band
    assert_no_action_cache_fields(out, what="confirmation output")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True)
    ap.add_argument("--unseal", default="")
    ap.add_argument("--discipline", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cost_only or not args.unseal:
        if not args.cost_only:
            raise SystemExit("no unseal record: only --cost-only is permitted before unsealing")
        out = cost_only_view(args.seal, args.journal, args.per_step)
    else:
        if not (args.discipline and args.ledger):
            raise SystemExit("--discipline and --ledger are required with --unseal")
        out = analyze(args.seal, args.unseal, args.discipline, args.ledger, args.journal, args.per_step)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps({k: out.get(k) for k in ("verdict", "cost_only")}, indent=2))


if __name__ == "__main__":
    main()
