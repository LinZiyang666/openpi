"""Stage 2 of the budget amendment: outcomes under the frozen interval
(confirmation plan 3.3, estimator ``budget_mixture_v1``).

Runs only after ``budget_cost_map_frozen.json`` exists and its SHA is
supplied; every frozen input digest is re-checked before an outcome row is
read. Produces the development value envelopes of the three families over
ALL measured arms (no pruning, no equal-arm-count rule), the development
H1 / H2 / S0-T quantities through the single ``h1_verdict`` implementation,
A-2 for spatial, leave-one-task-out, per-task descriptives, the step-
envelope sensitivity, the family-agnostic C roster selector (active optimal
bases + bootstrap frequency >= F_MIN, <= M_MAX per family, anchor fixed) and
the dense-baseline stop-loss (development H1 q05 <= 0 => stop_before_C).
Phase 0 exact-cost artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface import tgrid_package as tpkg
from exp.dispatch_surface.analysis import budget_mixture as bm
from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.analysis.analytic_cost import cost_matches
from exp.dispatch_surface.analysis.cost_map import REPLICATES, SEED
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
from exp.dispatch_surface.analysis.precheck_io import load_accepted_episodes, load_analytic_cost
from exp.dispatch_surface.cost_map_api import index_arrays, shared_index_from_map
from exp.dispatch_surface.phase0_roster import (
    ANCHOR_ARM,
    F_MIN,
    FAMILY_S0,
    FAMILY_SV,
    FAMILY_THRESHOLD,
    M_MAX,
)
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, official_test_inits

PROTOCOL = "dispatch_surface_rev2_budget_amendment"
PAIRS = {"H1": (FAMILY_SV, FAMILY_THRESHOLD), "H2": (FAMILY_SV, FAMILY_S0), "S0_minus_T": (FAMILY_S0, FAMILY_THRESHOLD)}


def _sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _cells_with_outcomes(journal: str, per_step: str, arms: list[str], grid, officials, executed: dict) -> dict:
    acc = load_accepted_episodes(journal, arms, grid)
    phase0_discipline.assert_rows_claimed(acc, executed, what="outcome loader")
    cost, _ = load_analytic_cost(per_step, arms, acc, officials)
    return {a: {k: (cost[a][k][0], cost[a][k][1], float(acc[a][k]["success"])) for k in cost[a]} for a in arms}


def load_sources(args, cost_map: dict, trials: int) -> dict:
    """Re-check every frozen digest, then load (cost_sum, n, success) per cell."""
    frozen = cost_map["input_sha256"]
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    if manifest_sha != frozen["rev1"]["rev1_package_manifest"] or manifest_sha != cost_map["rev1_package_manifest_sha256"]:
        raise SystemExit("--rev1-package-manifest is not the package the cost map froze")
    for role, key in (("matrix", "matrix"), ("ledger", "ledger"), ("split_manifest", "split_manifest"),
                      ("journal", "journal"), ("per_step", "per_step")):
        if pkgmod.member_sha(manifest, role) != frozen["rev1"][key]:
            raise SystemExit(f"Rev 1 package member {role} differs from the cost map's frozen digest")
    ctx = phase0_discipline.validate(args.phase0_arm_matrix, args.phase0_launch_manifest, args.split_manifest, trials=trials)
    p0 = frozen["phase0"]
    checks = {"matrix": ctx["arm_matrix_sha256"], "ledger": ctx["launch_manifest_sha256"],
              "split_manifest": ctx["split_manifest_sha256"], "journal": _sha(args.phase0_journal),
              "per_step": _sha(args.phase0_per_step)}
    for key, got in checks.items():
        if got != p0[key]:
            raise SystemExit(f"Phase 0 {key} differs from the cost map's frozen digest")
    if list(ctx["export_record_sha256"]) != list(p0["export_records"]):
        raise SystemExit("Phase 0 export records differ from the cost map's frozen digests")
    tgrid_ctx = None
    if args.tgrid_package_manifest:
        tmanifest, tpkg_dir, tsha = tpkg.load_manifest(args.tgrid_package_manifest)
        tpkg.verify_package(args.tgrid_package_manifest)
        if frozen.get("tgrid") is None or tsha != frozen["tgrid"]["tgrid_package_manifest"]:
            raise SystemExit("--tgrid-package-manifest is not the package the cost map froze")
        tgrid_ctx = phase0_discipline.validate_tgrid(str(tpkg.verify_member(tmanifest, tpkg_dir, "matrix")),
                                                     str(tpkg.verify_member(tmanifest, tpkg_dir, "ledger")),
                                                     str(tpkg.verify_member(tmanifest, tpkg_dir, "split_manifest")), trials=trials)
    elif frozen.get("tgrid") is not None:
        raise SystemExit("the cost map froze a threshold-grid package; supply --tgrid-package-manifest")
    if cost_map.get("suite") != ctx["suite"]:
        raise SystemExit("cost map suite != Phase 0 suite")
    officials = official_test_inits(args.split_manifest, trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    # --- now outcomes may be read ---
    rev1_ledger = pkgmod.load_json_member(manifest, pkg, "ledger")
    rev1_exec = phase0_discipline.executed_arms_by_run(rev1_ledger.get("launches") or [])
    sources = cost_map["sources"]
    rev1_arms = [a for a in cost_map["candidates"] if sources[a]["kind"] == "rev1_package"]
    p0_arms = [a for a in cost_map["candidates"] if sources[a]["kind"] == "phase0"]
    tg_arms = [a for a in cost_map["candidates"] if sources[a]["kind"] == "tgrid_package"]
    cells = {}
    cells.update(_cells_with_outcomes(str(pkgmod.verify_member(manifest, pkg, "journal")),
                                      str(pkgmod.verify_member(manifest, pkg, "per_step")), rev1_arms, grid, officials, rev1_exec))
    cells.update(_cells_with_outcomes(args.phase0_journal, args.phase0_per_step, p0_arms, grid, officials,
                                      {k: set(v) for k, v in ctx["executed_arms_by_run"].items()}))
    if tg_arms:
        cells.update(_cells_with_outcomes(str(tpkg.verify_member(tmanifest, tpkg_dir, "journal")),
                                          str(tpkg.verify_member(tmanifest, tpkg_dir, "per_step")), tg_arms, grid, officials,
                                          {k: set(v) for k, v in tgrid_ctx["executed_arms_by_run"].items()}))
    keys = sorted(grid)
    for a in cost_map["candidates"]:
        num = sum(cells[a][k][0] for k in keys)
        den = sum(cells[a][k][1] for k in keys)
        if not cost_matches(num / den, cost_map["point_cost"][a]) or int(den) != cost_map["decisions"][a]:
            raise SystemExit(f"reloaded cost/decisions for {a} differ from the frozen cost map")
    # anchor from Phase 0 (fixed C member; A-4 already adjudicated by phase0_summary)
    p0_matrix = json.loads(pathlib.Path(args.phase0_arm_matrix).read_text())
    anchor = {"arm": ANCHOR_ARM, "yaml_path": p0_matrix["arms"][ANCHOR_ARM], "yaml_sha256": p0_matrix["arm_yaml_sha256"][ANCHOR_ARM]}
    # yaml / artifact locations for the roster record
    locations = {}
    for a in cost_map["candidates"]:
        s = sources[a]
        if s["kind"] == "rev1_package":
            ypath = pkgmod.verify_member(manifest, pkg, s["role"])
            loc = {"yaml_path": str(ypath), "yaml_sha256": s["yaml_sha256"]}
            if s.get("artifact_role"):
                loc["artifact_path"] = str(pkgmod.verify_member(manifest, pkg, s["artifact_role"]))
                loc["artifact_sha256"] = pkgmod.member_sha(manifest, s["artifact_role"])
            if cost_map["families"][a] == FAMILY_THRESHOLD:
                # Rev 1 threshold arms carry no matrix pair record: read the pair from the archived yaml itself.
                from openpi.cache.config import load_cache_config
                cp1 = load_cache_config(str(ypath)).checkpoints["cp1"]
                tiers = cp1.judge.warm_tiers or []
                loc["threshold_pair"] = [float(cp1.judge.threshold), (float(tiers[0]["threshold"]) if tiers else None)]
        elif s["kind"] == "phase0":
            loc = {"yaml_path": s["yaml_path"], "yaml_sha256": s["yaml_sha256"],
                   "artifact_path": s["artifact_path"], "artifact_sha256": s["artifact_sha256"]}
        else:
            loc = {"yaml_path": str(tpkg.verify_member(tmanifest, tpkg_dir, s["role"])), "yaml_sha256": s["yaml_sha256"],
                   "threshold_pair": s["threshold_pair"], "nominal": s["nominal"]}
        locations[a] = loc
    return {"cells": cells, "grid": grid, "suite": ctx["suite"], "library_sha256": ctx["library_sha256"],
            "anchor": anchor, "locations": locations, "input_sha256": {"rev1": frozen["rev1"], "phase0": checks,
                                                                     "tgrid": frozen.get("tgrid")}}


def family_rosters(cost_map: dict) -> dict[str, list[str]]:
    """Frozen roster order per family = the cost map's family_points order."""
    return {fam: list(fp["arms"]) for fam, fp in cost_map["family_points"].items()}


def design(cost_map: dict, cells: dict, picks: list, *, suite: str, audit_input_digest: str) -> dict:
    B_L, B_H = cost_map["budget_interval"]["B_L"], cost_map["budget_interval"]["B_H"]
    B_1, B_2 = cost_map["budget_interval"]["B_1"], cost_map["budget_interval"]["B_2"]
    rosters = family_rosters(cost_map)
    all_arms = [a for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD) for a in rosters[fam]]
    grid = sorted(next(iter(cells.values())).keys())
    cell_list, idx = index_arrays(picks, grid)
    R = idx.shape[0]
    arrays = cell_arrays(cells, all_arms)
    full = full_sample_stats(arrays, all_arms)
    est = budget_mixture_digest()
    audit_idx = audit_replicate_indices(R, estimator_digest=est, input_digest=audit_input_digest)
    out = {"interval": [B_L, B_H], "B_1": B_1, "B_2": B_2, "families": {}, "hypotheses": {}, "decision_gates": {},
           "audit_replicates": audit_idx}
    # --- development envelopes over ALL measured arms ---
    for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD):
        arms = rosters[fam]
        st = {a: full[a] for a in arms}
        feas = bm.feasible(arms, st, B_L)
        block = {"arms": arms, "measured_policies": {a: {"cost": full[a].c, "sr": full[a].s, "t": full[a].t, "d": full[a].d}
                                                     for a in arms},
                 "standalone_dominated": bm.standalone_dominance(arms, st), "feasible_at_B_L": feas}
        if feas:
            block.update(bm.active_basis_union(arms, st, B_L, B_H))
            block["audit_full_sample"] = bm.audit_family(arms, st, B_L, B_H)
            block["value_at"] = {str(b): bm.value_at(arms, st, b)[0] for b in (B_L, B_1, B_2, B_H)}
        out["families"][fam] = block
    # --- hypotheses through the single implementation ---
    verdicts = {}
    for name, (fa, fb) in PAIRS.items():
        des = FrozenDesign(family_a=fa, family_b=fb, roster=rosters, B_L=B_L, B_H=B_H, R=R)
        v = evaluate_h1_verdict(cells, des, idx, audit_replicates=audit_idx) if name == "H1" else \
            evaluate_hypothesis(cells, des, idx, audit_replicates=audit_idx)
        verdicts[name] = v
        summ = verdict_summary(v)
        # delta V at B_1 / B_2 (plug-in + bootstrap band, descriptive)
        pts = {}
        for label, b in (("B_1", B_1), ("B_2", B_2)):
            va = bm.value_at(rosters[fa], {a: full[a] for a in rosters[fa]}, b)[0]
            vb = bm.value_at(rosters[fb], {a: full[a] for a in rosters[fb]}, b)[0]
            reps = np.full(R, np.nan)
            for r in range(R):
                st = stats_for_index(arrays, rosters[fa] + rosters[fb], idx[r])
                xa = bm.value_at(rosters[fa], st, b)[0]
                xb = bm.value_at(rosters[fb], st, b)[0]
                reps[r] = (xa - xb) if (xa is not None and xb is not None) else np.nan
            ok = ~np.isnan(reps)
            pts[label] = {"plugin": (va - vb) if (va is not None and vb is not None) else None,
                          "q05": float(np.quantile(reps[ok], 0.05)) if ok.any() else None,
                          "q95": float(np.quantile(reps[ok], 0.95)) if ok.any() else None,
                          "infeasible_rate": float(1.0 - ok.mean())}
        summ["delta_value_at"] = pts
        if name != "H1":
            summ.pop("passed", None)   # descriptive pairs carry no pass field
        out["hypotheses"][name] = summ
    # --- A-2 (spatial descriptive) ---
    sv_st = {a: full[a] for a in rosters[FAMILY_SV]}
    t_st = {a: full[a] for a in rosters[FAMILY_THRESHOLD]}
    a2 = {"applies": suite == "libero_spatial", "descriptive": True}
    if bm.feasible(rosters[FAMILY_SV], sv_st, B_L) and bm.feasible(rosters[FAMILY_THRESHOLD], t_st, B_L):
        a2.update(bm.difference_extrema(rosters[FAMILY_SV], sv_st, rosters[FAMILY_THRESHOLD], t_st, B_L, B_H))
        a2["sv_dominated_on_interval"] = a2.pop("a_dominated")
        a2.pop("b_dominated", None)
    out["decision_gates"]["A2"] = a2
    # --- LOTO and per-task descriptives (full-sample plug-ins) ---
    tasks = sorted({k[0] for k in grid})
    loto, per_task = {}, {}
    for t in tasks:
        keep = np.array([i for i, k in enumerate(cell_list) if k[0] != t])
        only = np.array([i for i, k in enumerate(cell_list) if k[0] == t])
        loto[str(t)], per_task[str(t)] = {}, {"n_cells": int(len(only))}
        st_keep = stats_for_index(arrays, all_arms, keep)
        st_only = stats_for_index(arrays, all_arms, only)
        for name, (fa, fb) in PAIRS.items():
            v, miss = bm.auc_with_support(rosters[fa], st_keep, rosters[fb], st_keep, B_L, B_H)
            loto[str(t)][name] = None if miss else v
            v2, miss2 = bm.auc_with_support(rosters[fa], st_only, rosters[fb], st_only, B_L, B_H)
            per_task[str(t)][name] = None if miss2 else v2
        per_task[str(t)]["ceiling"] = bool(all(st_only[a].s == 1.0 for a in all_arms))
        per_task[str(t)]["floor"] = bool(all(st_only[a].s == 0.0 for a in all_arms))
    out["leave_one_task_out"] = loto
    out["per_task_descriptive"] = per_task
    # --- C roster selector (family-agnostic) ---
    roster_sel, overflow = {}, []
    for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD):
        arms = rosters[fam]
        block = out["families"][fam]
        chosen: dict[str, list[str]] = {}
        if block["feasible_at_B_L"]:
            for a in block["active"]:
                chosen.setdefault(a, []).append("full_sample_active_basis")
            for side in ("B_L", "B_H"):
                for a in block["endpoint_bases"][side]:
                    chosen.setdefault(a, []).append(f"endpoint_basis_{side}")
        freq = verdicts["H1" if fam != FAMILY_S0 else "H2"].active_freq[fam]
        for a in arms:
            if freq[a] >= F_MIN:
                chosen.setdefault(a, []).append(f"bootstrap_active_freq>={F_MIN}")
        sel = [a for a in arms if a in chosen]  # roster order
        if len(sel) > M_MAX:
            overflow.append(f"{fam}: {len(sel)} > M_MAX={M_MAX}")
        roster_sel[fam] = {"arms": sel, "reasons": {a: chosen[a] for a in sel}, "active_freq": freq}
    out["c_roster_selection"] = roster_sel
    out["roster_overflow"] = overflow
    # --- verdict / gates ---
    h1 = out["hypotheses"]["H1"]
    out["decision_gates"]["A1_development_h1"] = {"pass": bool(h1["left_support_ok"] and h1["joint_miss"] <= 0.01 and h1["bootstrap_q05"] > 0.0),
                                                   "q05": h1["bootstrap_q05"]}
    if suite == "libero_spatial":
        out["verdict"] = "spatial_descriptive"
    elif overflow:
        out["verdict"] = "roster_overflow"
    elif not out["decision_gates"]["A1_development_h1"]["pass"]:
        out["verdict"] = "stop_before_C"
    else:
        out["verdict"] = "proceed_to_power"
    out["_verdict_objects"] = verdicts
    return out


def run(args) -> dict:
    if args.trials != FORMAL_TRIALS:
        raise SystemExit(f"formal outcome design is frozen at --trials {FORMAL_TRIALS}")
    cm_path = pathlib.Path(args.budget_cost_map)
    cm_bytes = cm_path.read_bytes()
    if hashlib.sha256(cm_bytes).hexdigest() != args.budget_cost_map_sha256:
        raise SystemExit("budget cost map SHA does not match the frozen value supplied")
    cost_map = json.loads(cm_bytes)
    if cost_map.get("protocol") != PROTOCOL or cost_map.get("outcome_blind") is not True:
        raise SystemExit("not a budget amendment cost map")
    if cost_map.get("estimator_version") != budget_mixture_digest():
        raise SystemExit("cost map estimator != budget_mixture_v1")
    if cost_map.get("replicates") != REPLICATES or cost_map.get("seed") != SEED:
        raise SystemExit(f"formal outcome design requires the frozen bootstrap (R={REPLICATES}, seed={SEED})")
    if not cost_map.get("a3_pass"):
        raise SystemExit(f"A-3' failed in the cost map: {cost_map.get('a3_problems')}; outcome design refused")
    src = load_sources(args, cost_map, args.trials)
    picks = shared_index_from_map(cost_map, src["grid"])
    out = design(cost_map, src["cells"], picks, suite=src["suite"], audit_input_digest=args.budget_cost_map_sha256)
    verdict_objs = out.pop("_verdict_objects")
    out.update({"protocol": PROTOCOL, "posthoc_design_amendment": True, "posthoc_exploratory": True,
                "development_only": True, "suite": src["suite"], "estimator_version": budget_mixture_digest(),
                "budget_cost_map_sha256": args.budget_cost_map_sha256, "budget_interval": cost_map["budget_interval"],
                "library_sha256": src["library_sha256"], "input_sha256": src["input_sha256"],
                "F_MIN": F_MIN, "M_MAX": M_MAX})
    if hashlib.sha256(cm_path.read_bytes()).hexdigest() != args.budget_cost_map_sha256:
        raise SystemExit("budget cost map changed during outcome design — refusing")
    out_path = pathlib.Path(args.out)
    for protected in (args.phase0_arm_matrix, args.phase0_journal, args.phase0_per_step):
        if out_path.resolve() == pathlib.Path(protected).resolve():
            raise SystemExit("refusing to overwrite a Phase 0 input")
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    os.replace(tmp, out_path)
    # --- c_roster.json (binds the design file) ---
    if args.out_roster:
        entries = []
        for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD):
            for a in out["c_roster_selection"][fam]["arms"]:
                loc = src["locations"][a]
                e = {"arm": a, "family": fam, "source": cost_map["sources"][a], "yaml_path": loc["yaml_path"],
                     "yaml_sha256": loc["yaml_sha256"], "artifact_path": loc.get("artifact_path"),
                     "artifact_sha256": loc.get("artifact_sha256"), "threshold_pair": loc.get("threshold_pair"),
                     "delta": cost_map["deltas"].get(a), "nominal": loc.get("nominal"),
                     "reasons": out["c_roster_selection"][fam]["reasons"][a],
                     "active_freq": out["c_roster_selection"][fam]["active_freq"][a]}
                entries.append(e)
        entries.append({"arm": ANCHOR_ARM, "family": "anchor", "source": {"kind": "phase0"}, "yaml_path": src["anchor"]["yaml_path"],
                        "yaml_sha256": src["anchor"]["yaml_sha256"], "artifact_path": None, "artifact_sha256": None,
                        "threshold_pair": None, "delta": None, "nominal": None, "reasons": ["fixed_anchor"], "active_freq": None})
        roster = {"schema": 1, "protocol": PROTOCOL + "_c_roster", "suite": src["suite"], "verdict": out["verdict"],
                  "budget_interval": cost_map["budget_interval"], "estimator_version": budget_mixture_digest(),
                  "outcome_design_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                  "budget_cost_map_sha256": args.budget_cost_map_sha256, "F_MIN": F_MIN, "M_MAX": M_MAX,
                  "library_sha256": src["library_sha256"], "arms": entries,
                  "active_bitset_rollup_sha256": {n: verdict_summary(v)["active_bitset_rollup_sha256"] for n, v in verdict_objs.items()}}
        pathlib.Path(args.out_roster).write_text(json.dumps(roster, indent=2, sort_keys=True))
    print(json.dumps({"verdict": out["verdict"], "H1": {k: out["hypotheses"]["H1"][k] for k in ("effect_plugin", "bootstrap_q05", "joint_miss", "reason")},
                      "roster": {f: b["arms"] for f, b in out["c_roster_selection"].items()}, "overflow": out["roster_overflow"]}, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-cost-map", required=True)
    ap.add_argument("--budget-cost-map-sha256", required=True)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--phase0-arm-matrix", required=True)
    ap.add_argument("--phase0-launch-manifest", required=True)
    ap.add_argument("--phase0-journal", required=True)
    ap.add_argument("--phase0-per-step", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--tgrid-package-manifest", default="")
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS, help=f"frozen at {FORMAL_TRIALS}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-roster", default="", help="c_roster.json (libero_10 only)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
