"""Finite R7 A1 snapshot: acceptance, cell comparisons, costs, exceptions, frontier.

Phase 1 (default) writes machine-readable tables, withholds acceptance verdicts,
and never writes ANALYSIS.md. Phase 2 explicitly requires --phase final. Missing
arms remain 'missing'; no rule is evaluated on an available-arm subset.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a1_common as C


RULE_TEXT = {
    "1": "Pooled 8-cell SR(SF)-SR(A) 95% lower bound > -0.010 and IR lower than A in at least 6/8 cells.",
    "2": "Pooled SR(SF)-SR(UF) lower bound > 0, OR SF has no significant loss versus A while UF has pooled lower bound < -0.010.",
    "3": "Pooled 4-cell SR(SW)-SR(A) lower bound > -0.015 and IR lower in 4/4 pi05 cells.",
    "4": "Pooled 4-cell SR(CT)-SR(CU) lower bound > 0 at |delta aggregate owner IR| <= 0.015 in every cell.",
    "5": "Each R7 point moves the frontier only if non-dominated by R6 single-run points; show CP NI bound, no certified NI claim.",
}
CONVENTIONS = {
    "bootstrap": "10,000 draws, seed 20260930; fixed tasks, paired init resampling; A/B three-replicate means per pair",
    "pooling": "equal-cell mean; independent cell resamples within each contrast, as R6 ops/c_validation.py",
    "IR": "ratio of total owner cost to nominal five-control blocks; A/B arithmetic mean of three ledger ratios",
    "rule2_no_significant_loss": "pooled SF-A two-sided 95% bootstrap upper endpoint >= 0; not equivalence or rule-1 NI",
    "rule4_cost_matching": "SELECTION section 8 pre-CT clarification: every cell must have |CT-CU aggregate owner IR| <= .015; pooled mean delta and equal-episode means are descriptive, not alternative acceptance gates",
    "frontier": "frozen R6 frontier_final eligible single-run points and shared pure references; strict dominance, ties retained; no convex interpolation",
    "NI": "Q2 conservative CP_lower(wins)-CP_upper(losses), alpha .05, margin .02; fractional means not passed to binary tests",
    "SW": "actual camera counters / owner_cost; wrist .055198 and completion .049890 use the R4 proportional-latency assumption; full-camera counterfactual and stock ledger also shown",
    "physical_controls": "nominal requested blocks only; no inferred actual-control denominator",
}


def pooled(table, cells, x, y):
    absent = [dict(cell=c, variant=v, status=table.get((c, v), {}).get("status", "missing"))
              for c in cells for v in (x, y) if table.get((c, v), {}).get("status") != "complete"]
    if absent:
        return dict(status="missing", expected_cells=len(cells), missing=absent)
    xs, ys = [table[c, x] for c in cells], [table[c, y] for c in cells]
    stats, _ = C.bootstrap_differences([(a["_outcomes"], b["_outcomes"]) for a, b in zip(xs, ys)])
    costs = {c: a["owner_IR"] - b["owner_IR"] for c, a, b in zip(cells, xs, ys)}
    result = dict(status="complete", cells=cells, candidate=x, reference=y, delta_SR=stats,
                  delta_owner_IR_by_cell=costs, delta_owner_IR_pooled=float(np.mean(list(costs.values()))))
    if all("owner_IR_equal_episode_mean" in r for r in xs + ys):
        means = {c: {x: a["owner_IR_equal_episode_mean"], y: b["owner_IR_equal_episode_mean"]}
                 for c, a, b in zip(cells, xs, ys)}
        result["equal_episode_IR_by_cell"] = means
        result["equal_episode_IR_pooled"] = {v: float(np.mean([r[v] for r in means.values()])) for v in (x, y)}
        result["delta_equal_episode_IR_pooled"] = result["equal_episode_IR_pooled"][x] - result["equal_episode_IR_pooled"][y]
    return result


def acceptance(pools, *, final=False, frontier_complete=False):
    """Pure rule evaluator; synthetic boundary tests run without evaluation data."""
    result = {k: dict(rule=text, verdict="deferred_phase1") for k, text in RULE_TEXT.items()}
    dependencies = {"1": ["SF1-A"], "2": ["SF1-UF1", "SF1-A", "UF1-A"],
                    "3": ["SW-A"], "4": ["CT30-CU30"]}
    for rule, deps in dependencies.items():
        missing = [d for d in deps if pools[d]["status"] != "complete"]
        result[rule]["contrasts"] = deps
        if missing:
            result[rule].update(verdict="missing", missing=missing)
    if not frontier_complete:
        result["5"]["verdict"] = "missing"
    if not final:
        return result
    if result["1"]["verdict"] != "missing":
        p = pools["SF1-A"]
        nlower = sum(v < 0 for v in p["delta_owner_IR_by_cell"].values())
        passed = p["delta_SR"]["lo"] > -.010 and nlower >= 6
        result["1"].update(verdict="supported" if passed else "not supported", lower_IR_cells=nlower)
    if result["2"]["verdict"] != "missing":
        first = pools["SF1-UF1"]["delta_SR"]["lo"] > 0
        no_loss = pools["SF1-A"]["delta_SR"]["hi"] >= 0
        uf_lower = pools["UF1-A"]["delta_SR"]["lo"] < -.010
        result["2"].update(verdict="supported" if first or (no_loss and uf_lower) else "not supported",
                           direct_SF_UF_branch=first, SF_no_significant_loss=no_loss,
                           UF_lower_below_minus_1pp=uf_lower,
                           fallback="Otherwise the lever works (or not) without the stage signal.")
    if result["3"]["verdict"] != "missing":
        p = pools["SW-A"]
        nlower = sum(v < 0 for v in p["delta_owner_IR_by_cell"].values())
        result["3"].update(verdict="supported" if p["delta_SR"]["lo"] > -.015 and nlower == 4 else "not supported",
                           lower_IR_cells=nlower)
    if result["4"]["verdict"] != "missing":
        p = pools["CT30-CU30"]
        sr_ok = p["delta_SR"]["lo"] > 0
        unmatched = [c for c, v in p["delta_owner_IR_by_cell"].items() if abs(v) > .015]
        each = not unmatched
        result["4"].update(verdict=("IR not matched" if unmatched else "supported" if sr_ok else "not supported"),
                           SR_lower_positive=sr_ok, all_cell_cost_match=each, unmatched_cells=unmatched,
                           cell_cost_verdict={c: "IR not matched in that cell" if c in unmatched else "IR matched"
                                              for c in p["delta_owner_IR_by_cell"]},
                           cost_scope=CONVENTIONS["rule4_cost_matching"])
    if frontier_complete:
        result["5"].update(verdict="reported pointwise", certified_NI_claim=False)
    return result


def historical_references():
    """Load current summaries/journals/timings, using only frozen R6 identities."""
    loaded, table = {}, {}

    def load(spec, cell, label):
        if spec not in loaded:
            run, arm = spec.split(":")
            loaded[spec] = C.load_arm(C.RUNS / run, arm, cell, label, raw=False, historical=True)
        return loaded[spec]

    for model, short in C.R6.CELLS:
        cell = C.R6.c_cell(model, short)
        for label, specs in zip(("A", "B"), C.R6.ab_arms(model, short)):
            table[cell, label] = C.average_records([load(s, cell, label) for s in specs], cell, label)
        table[cell, "L10"] = load(C.R6.l10_ref(model, short), cell, "L10")
        for i, spec in enumerate(C.R6.l5_refs(model, short), 1):
            table[cell, f"L5_inroot_{i}"] = load(spec, cell, "L5")
        tags = ("U30", "R30", "C30", "C45") if cell in C.SPARSE else ("U18", "R18", "C18")
        for tag in tags:
            arm_tag = "Cmax" if cell == "groot_l10_50" and tag == "C45" else tag
            table[cell, f"R6_{tag}"] = load(f"r06_c_validation:r6c_{cell}_{arm_tag}", cell, f"R6_{tag}")
    return table, list(loaded.values())


def frozen_frontier():
    data = C.read_json(C.FRONTIER / "frontier_data.json")
    outcomes = {arm: {tuple(map(int, k.split(":"))): v for k, v in o.items()}
                for arm, o in C.read_json(C.FRONTIER / "outcomes.json").items()}
    return data, outcomes


def add_l5_references(table, data, outcomes):
    by_id = {r["id"]: r for r in data["points"]}
    for cell in C.CELLS:
        model, suite, _ = cell.split("_")
        ref = next(r for r in data["references"] if r["model"] == model and r["suite"] == suite and r["L"] == 5)
        constituents = [dict(id=i, cell=cell, variant="L5", status="complete", SR=by_id[i]["SR"],
                             owner_IR=by_id[i]["owner_IR"], _outcomes=outcomes[i],
                             source="R6 frontier_final/outcomes.json and frontier_data.json") for i in ref["runs"]]
        table[cell, "L5"] = (C.average_records(constituents, cell, "L5") if len(constituents) > 1 else constituents[0])


def frontier_rows(data, cell):
    model, suite, _ = cell.split("_")
    return [p for p in data["points"] if p["eligible"] and
            (p["cell"] == cell or (p["pure"] and p["model"] == model and p["suite"] == suite))]


def lowest_crossing(rows, target, *, include_pure):
    choices = [r for r in rows if (include_pure or not r.get("pure", False)) and r["SR"] >= target - 1e-12]
    choices.sort(key=lambda r: (r["owner_IR"], -r["SR"], r["id"]))
    return {k: choices[0][k] for k in ("id", "SR", "owner_IR")} if choices else None


def frontier_placement(table, data, outcomes):
    placements, crossings, fronts = [], [], {}
    for cell in C.CELLS:
        before = frontier_rows(data, cell)
        old_front = C.pareto(before)
        new = [table[row["cell"], row["variant"]] for row in C.frozen_arms()
               if row["cell"] == cell and table[row["cell"], row["variant"]]["status"] == "complete"]
        combined = C.pareto(before + new)
        fronts[cell] = dict(before=[r["id"] for r in old_front], after_available=[r["id"] for r in combined])
        reference = table[cell, "L10"]
        if reference["status"] != "complete":
            crossings.append(dict(cell=cell, status="missing", reason="L10 reference missing"))
            continue
        expected_count = sum(row["cell"] == cell for row in C.frozen_arms())
        for p in new:
            dominators = [r for r in before if C.dominates(r, p)]
            lower_cost = [r for r in old_front if r["owner_IR"] <= p["owner_IR"]]
            near = max(lower_cost, key=lambda r: (r["SR"], -r["owner_IR"])) if lower_cost else None
            model, suite, _ = cell.split("_")
            l5 = next(r for r in data["references"] if r["model"] == model and r["suite"] == suite and r["L"] == 5)
            l5_exact = [dict(reference=ident, **C.exact_pair(p["_outcomes"], outcomes[ident], alpha=.05/len(l5["runs"])))
                        for ident in l5["runs"]]
            placements.append(dict(status="complete", cell=cell, id=p["id"], variant=p["variant"], SR=p["SR"], owner_IR=p["owner_IR"],
                nondominated_by_R6=not dominators, R6_dominators=[r["id"] for r in dominators],
                nondominated_after_available_R7=p["id"] in fronts[cell]["after_available"],
                R6_points_dominated=[r["id"] for r in old_front if C.dominates(p, r)],
                coincident_R6_points=[r["id"] for r in before if r["SR"] == p["SR"] and r["owner_IR"] == p["owner_IR"]],
                best_R6_at_no_greater_IR=({k: near[k] for k in ("id", "SR", "owner_IR")} if near else None),
                exact_vs_L10=C.exact_pair(p["_outcomes"], reference["_outcomes"]),
                exact_vs_R6_neighbor=(dict(reference=near["id"], **C.exact_pair(p["_outcomes"], outcomes[near["id"]])) if near else None),
                L5_reference_ids=l5["runs"], L5_reference_SR=l5["SR"],
                CP_lower_vs_L5_mean=float(np.mean([r["cp_lower95"] for r in l5_exact])),
                L5_constituent_CP=l5_exact, certified_NI_claim=False))
        crossings.append(dict(cell=cell, status="complete" if len(new) == expected_count else "missing",
                              available_R7=len(new), expected_R7=expected_count, reference_L10_SR=reference["SR"],
                              before_nonpure=lowest_crossing(before, reference["SR"], include_pure=False),
                              after_available_nonpure=lowest_crossing(before + new, reference["SR"], include_pure=False),
                              before_including_pure=lowest_crossing(before, reference["SR"], include_pure=True),
                              after_available_including_pure=lowest_crossing(before + new, reference["SR"], include_pure=True)))
    available = {p["id"] for p in placements}
    for row in C.frozen_arms():
        r = table[row["cell"], row["variant"]]
        if r["id"] not in available:
            placements.append(dict(cell=row["cell"], id=r["id"], variant=row["variant"], status="missing"))
    return dict(placements=placements, lowest_IR_reaching_L10=crossings, frontiers=fronts,
                definition=CONVENTIONS["frontier"], certification_claim=False,
                historical_wrist_pricing="R6 points retain frontier_final prices, including full-camera pricing of historical wrist arms; only R7 SW is repriced here")


def check_r6(data, outcomes, references):
    """Independent point/crossing reconstruction plus raw-reference snapshot checks."""
    mismatches = []
    by_id = {p["id"]: p for p in data["points"]}
    for cell in C.CELLS:
        rows = frontier_rows(data, cell)
        if set(r["id"] for r in C.pareto(rows)) != set(data["frontiers"][cell]):
            mismatches.append(dict(cell=cell, field="pareto"))
        gap = next(g for g in data["gaps"] if g["cell"] == cell and g["reference_L"] == 10)
        cross = lowest_crossing(rows, gap["reference_SR"], include_pure=True)
        if not np.isclose(cross["owner_IR"], gap["observed_crossing_IR"], atol=1e-12, rtol=0):
            mismatches.append(dict(cell=cell, field="lowest_L10_crossing"))
    for r in references:
        old = by_id.get(r["id"])
        if r["status"] != "complete":
            mismatches.append(dict(id=r["id"], field="reference_load", error=r.get("error")))
        elif old is None:
            mismatches.append(dict(id=r["id"], field="absent_in_R6_frontier"))
        elif (r["_outcomes"] != outcomes[r["id"]] or not np.isclose(r["owner_IR"], old["owner_IR"], atol=1e-12, rtol=0)):
            mismatches.append(dict(id=r["id"], field="outcomes_or_owner_IR"))
    return dict(status="ok" if not mismatches else "invalid", references_checked=len(references),
                cells_checked=len(C.CELLS), frozen_points=len(data["points"]), mismatches=mismatches,
                sources=[C.stamp(C.FRONTIER / n, True) for n in ("frontier_data.json", "outcomes.json")])


def write_csv(path, rows):
    rows = [C.public(r) for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: C.json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=C.RUNS / "r07_main")
    parser.add_argument("--out", type=Path, default=C.R7 / "analysis_r7")
    parser.add_argument("--phase", choices=("prepare", "final"), default="prepare")
    parser.add_argument("--inventory-from", type=Path,
                        help="reuse an earlier finite a1_snapshot inventory while debugging; Phase 2 should omit this")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("America/Chicago")).isoformat()
    previous = C.read_json(args.inventory_from) if args.inventory_from else None
    if previous and previous["run_root"] != str(args.run_root):
        raise ValueError("inventory belongs to a different run root")
    inventory = previous["inventory"] if previous else [{**row, **C.completion(args.run_root, row["arm"])} for row in C.frozen_arms()]
    inventory_time = previous.get("inventory_CDT", previous["snapshot_CDT"]) if previous else now
    # Inventory is taken once before potentially lengthy historical audit work.
    table, references = historical_references()
    print(f"Historical references loaded: {len(references)}", flush=True)
    r7 = []
    for row in inventory:
        if row["status"] == "complete":
            result = C.load_arm(args.run_root, row["arm"], row["cell"], row["variant"])
        else:
            result = dict(id=f"{args.run_root.name}/{row['arm']}", **row)
        table[row["cell"], row["variant"]] = result
        r7.append(result)
        print(f"{row['arm']}: {result['status']}", flush=True)
    data, outcomes = frozen_frontier()
    add_l5_references(table, data, outcomes)
    check = check_r6(data, outcomes, references)
    placement = frontier_placement(table, data, outcomes)
    pairs = [("SF1", "A", C.CELLS), ("UF1", "A", C.CELLS), ("SF1", "UF1", C.CELLS),
             ("SW", "A", C.PI05), ("CT30", "CU30", C.SPARSE)]
    pools = {f"{x}-{y}": pooled(table, cells, x, y) for x, y, cells in pairs}
    contrasts = []
    for row in C.frozen_arms():
        cell, variant = row["cell"], row["variant"]
        comparisons = ["A", "B", "L10", "L5"]
        if variant == "SF1":
            comparisons.append("UF1")
        if variant == "CT30":
            comparisons.append("CU30")
        if variant in ("CU30", "CT30"):
            comparisons += ["R6_U30", "R6_C30"]
        for ref in comparisons:
            comparison = dict(cell=cell, variant=variant, reference_label=ref,
                              **C.contrast(table[cell, variant], table[cell, ref]))
            if variant == "CT30" and ref == "CU30" and comparison["status"] == "complete":
                matched = abs(comparison["delta_owner_IR"]) <= .015
                comparison["rule4_IR_matched"] = matched
                comparison["rule4_cell_SR_verdict"] = ("IR not matched in that cell" if not matched else
                    "descriptive cell SR contrast; acceptance uses the pooled four-cell lower bound")
            contrasts.append(comparison)
    for cell in C.CELLS:
        for x, y in (("B", "A"), ("A", "L10"), ("B", "L10")):
            contrasts.append(dict(cell=cell, variant=x, reference_label=y, **C.contrast(table[cell, x], table[cell, y])))
    ready = all(r["status"] == "complete" for r in r7) and check["status"] == "ok"
    rules = acceptance(pools, final=args.phase == "final" and check["status"] == "ok", frontier_complete=ready)
    if check["status"] != "ok":
        for rule in rules.values():
            rule.update(verdict="invalid_reference_data")
    summary = dict(schema="r7.a1.v1", phase=args.phase, snapshot_CDT=now, inventory_CDT=inventory_time, run_root=str(args.run_root),
                   evaluation_complete=ready, expected_arms=28, complete_arms=sum(r["status"] == "complete" for r in r7),
                   missing_arms=[r["arm"] for r in r7 if r["status"] == "missing"],
                   invalid_arms=[dict(arm=r["arm"], error=r.get("error")) for r in r7 if r["status"] == "invalid"],
                   conventions=CONVENTIONS, inventory=inventory, acceptance=rules, pooled=pools,
                   r6_validation=check, chain_exception_audit=C.chain_audit(args.run_root),
                   scripts=[C.stamp(p, True) for p in sorted(C.HERE.glob("a1_*.py"))],
                   no_conclusions_in_phase1=True)
    C.write_json(args.out / "a1_snapshot.json", summary)
    C.write_json(args.out / "a1_arms.json", r7)
    C.write_json(args.out / "a1_references.json", references)
    C.write_json(args.out / "a1_contrasts.json", contrasts)
    C.write_json(args.out / "a1_frontier.json", placement)
    C.write_json(args.out / "a1_exception_audit.json", [dict(id=r["id"], status=r["status"],
                 audit=r.get("exception_audit"), error=r.get("error")) for r in references + r7])
    cell_rows, wide = [], []
    labels = ("A", "SF1", "UF1", "SW", "CU30", "CT30", "B", "L10")
    for cell in C.CELLS:
        w = dict(cell=cell)
        for label in labels:
            r = table.get((cell, label), dict(status="not applicable"))
            cell_rows.append(dict(cell=cell, variant=label, status=r["status"],
                                  **{k: r.get(k) for k in ("SR", "owner_IR", "owner_IR_equal_episode_mean", "n", "replicate_ids")}))
            w[label] = f"{r['SR']:.6f} @ {r['owner_IR']:.6f}" if r["status"] == "complete" else r["status"]
        wide.append(w)
    write_csv(args.out / "a1_per_cell.csv", cell_rows)
    write_csv(args.out / "a1_per_cell_wide.csv", wide)
    write_csv(args.out / "a1_paired_deltas.csv", [dict(cell=r["cell"], variant=r["variant"], reference=r["reference_label"],
        status=r["status"], delta_SR=r.get("delta_SR", {}).get("estimate"),
        delta_SR_lo=r.get("delta_SR", {}).get("lo"), delta_SR_hi=r.get("delta_SR", {}).get("hi"),
        delta_owner_IR=r.get("delta_owner_IR"), exact=r.get("exact")) for r in contrasts])
    write_csv(args.out / "a1_costs.csv", [dict(id=r["id"], status=r["status"], **{k: r.get(k) for k in
        ("N", "V", "M", "L", "owner_IR", "owner_IR_full_camera_counterfactual", "stock_ledger_IR",
         "owner_IR_from_camera_counts", "owner_IR_logged_minus_counted", "owner_IR_source",
         "owner_IR_equal_episode_mean", "owner_IR_components", "cost_assumption", "ledger_minus_client_decisions")}) for r in r7])
    print(C.json.dumps({k: summary[k] for k in ("snapshot_CDT", "phase", "complete_arms", "missing_arms", "invalid_arms")}))
    if summary["invalid_arms"] or check["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
