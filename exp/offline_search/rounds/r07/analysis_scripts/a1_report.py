"""Render A1 quantitative sections from a complete, validated final snapshot.

Never loads experiments or changes verdicts. Retains an existing A2 section.
Run with ANALYSIS_BRIEF's A1 CPU/environment prefix after analyze and validate.
"""
from __future__ import annotations

import csv
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

from exp.offline_search.rounds.r07.analysis_scripts import a1_common as C


A2_HEADING = "## 6–8 (A2: mechanism, generality, proposals)"
LABEL = {"A": "A", "B": "B", "SF1": "SF1", "UF1": "UF1", "SW": "SW",
         "CU30": "CU", "CT30": "CT", "L10": "pure L10"}


def cell_label(cell):
    model, suite, size = cell.split("_")
    return f"{'π0.5' if model == 'pi05' else 'GR00T'} {'LIBERO-10' if suite == 'l10' else 'Spatial'} / {size}"


def ci(row, scale=100, digits=2):
    # Remove numerical noise at exact decimal ties before display rounding.
    def signed(x):
        value = Decimal(str(round(x * scale, 10))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        return f"{value:+.{digits}f}"
    return f"{signed(row['estimate'])} [{signed(row['lo'])}, {signed(row['hi'])}]"


def point(row):
    return f"{float(row['SR'])*100:.2f}% @ {float(row['owner_IR']):.6f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def main():
    out = C.R7 / "analysis_r7"
    snap = C.read_json(out / "a1_snapshot.json")
    validation = C.read_json(out / "a1_validation.json")
    arms = C.read_json(out / "a1_arms.json")
    refs = C.read_json(out / "a1_references.json")
    contrasts = C.read_json(out / "a1_contrasts.json")
    frontier = C.read_json(out / "a1_frontier.json")
    cells = list(csv.DictReader((out / "a1_per_cell.csv").open()))
    if snap["phase"] != "final" or not snap["evaluation_complete"] or not validation["all_pass"]:
        raise ValueError("Report requires a complete final snapshot and passing validation")
    assert len(arms) == 28 and all(r["status"] == "complete" and r["n"] == 500 for r in arms)
    assert len(cells) == 64 and sum(r["status"] == "complete" for r in cells) == 52
    assert len(contrasts) == 164 and all(r["status"] == "complete" for r in contrasts)
    assert len(frontier["placements"]) == 28 and all(p["status"] == "complete" for p in frontier["placements"])
    assert all(x["status"] == "complete" for x in frontier["lowest_IR_reaching_L10"])
    by_cell = {(r["cell"], r["variant"]): r for r in cells}
    arms_by_cell = {(r["cell"], r["variant"]): r for r in arms}
    comparisons = {(r["cell"], r["variant"], r["reference_label"]): r for r in contrasts}
    pools, rules = snap["pooled"], snap["acceptance"]
    # Narrative below is written for this frozen R7 matrix. Refuse to attach it
    # to changed results, even if a replacement snapshot is otherwise complete.
    assert [rules[k]["verdict"] for k in "12345"] == [
        "not supported", "supported", "supported", "supported", "reported pointwise"]
    assert len(refs) == 84 and snap["r6_validation"]["status"] == "ok"
    assert sum(p["nondominated_by_R6"] for p in frontier["placements"]) == 20
    assert sum(p["nondominated_after_available_R7"] for p in frontier["placements"]) == 15
    assert all(not p["exact_vs_L10"]["nominal_NI"] and p["CP_lower_vs_L5_mean"] <= -.02
               for p in frontier["placements"])
    assert all(p["before_including_pure"] == p["after_available_including_pure"]
               for p in frontier["lowest_IR_reaching_L10"])
    lines = []

    def add(text):
        lines.append(text)

    add("# R7 analysis — stage-level allocation: quantitative results")
    add(f"A1 final snapshot: **{snap['snapshot_CDT']} (CDT)**. All 28 frozen evaluation arms are complete. "
        "Rules follow [SELECTION §5 and the pre-test §8 clarifications](SELECTION.md); "
        "[the analysis brief](ANALYSIS_BRIEF.md) assigns the mechanism and generality discussion to A2.")
    add(f"**Success preservation fails for extra blind execution, while the stage/state checks help.** "
        f"Allowing one extra five-control block only in stable stages lowers owner cost in all eight cells, "
        f"but changes success by **{ci(pools['SF1-A']['delta_SR'])} pp** versus the no-call cache, failing "
        "the preregistered −1 pp lower-bound threshold. Stage and state checks improve success over "
        f"extensions based on structural support alone by **{ci(pools['SF1-UF1']['delta_SR'])} pp**, "
        "so the signal rule passes. That comparison also spends more on vision; it is not a matched-cost test.")
    add(f"**Wrist-camera allocation and stage-tilted calls pass their respective rules.** On π0.5, using only "
        f"the wrist camera in easy stages changes success by **{ci(pools['SW-A']['delta_SR'])} pp**, "
        "with lower owner cost in all four cells under the stated wrist-price assumption. The lower bound "
        "clears its −1.5 pp threshold by only 0.10 pp. Tilting policy calls toward stage/state events "
        f"improves success over uniform calls by **{ci(pools['CT30-CU30']['delta_SR'])} pp**, "
        "and all four comparisons meet the per-cell cost-match limit.")
    add("**The cheapest observed match to pure-policy success does not improve.** Twenty new points are "
        "non-dominated by the frozen historical points; fifteen remain on the combined frontier. "
        "None of the new points clears the conservative nominal 2 pp noninferiority bound against pure L10 "
        "or pure L5. Mechanism and generality are reserved for A2 in §§6–8.")
    add("## 1. Data quality and cost reconciliation")
    add("Each arm contains the same 500 accepted task/init pairs: ten tasks × fifty inits. "
        "Completion requires a summary and either the plain `.DONE` marker or a `.manifest_*.DONE` marker. "
        "The stage-bounded and uniform-follow arms used the full Cartesian set without a manifest field; "
        "the pair sets are rechecked here, and the init-pool hash agreement is recorded in SELECTION §8 by the coordinator/A2.")
    audit_rows = []
    for variant in ("SF1", "UF1", "SW", "CU30", "CT30"):
        group = [r for r in arms if r["variant"] == variant]
        family = {"SF1": "Stage-bounded follow (SF1)", "UF1": "Structural-only follow (UF1)",
                  "SW": "Stage wrist (SW)", "CU30": "Uniform calls + stall (CU)",
                  "CT30": "Stage-tilted calls + stall (CT)"}[variant]
        audit_rows.append([family, len(group), sum(r["n"] for r in group),
                           *(f"{sum(r[k] for r in group):,}" for k in ("N", "V", "M")),
                           sum(r["ledger_minus_client_decisions"] != 0 for r in group),
                           sum(r["exception_audit"]["residual_exceptions"] for r in group)])
    add(table(["Family", "Arms", "Accepted episodes", "Decisions N", "Vision V", "Calls M",
               "Count-mismatch arms", "Accepted exceptions"], audit_rows))
    accepted = sum(r["exception_audit"]["accepted"] for r in arms + refs)
    covered = sum(r["exception_audit"]["covered"] for r in arms + refs)
    exc = sum(r["exception_audit"]["residual_exceptions"] for r in arms + refs)
    chain = snap["chain_exception_audit"]
    purged = sum(r["exception_audit"]["purged_records"] for r in arms)
    events = sum(e.get("n", 0) for e in chain["events"])
    exceptions = sum(r["exception_audit"]["exception_attempts_all"] for r in arms)
    assert exc == 0 and accepted == covered
    assert all(r["exception_audit"]["purged_all_failed"] and r["exception_audit"]["purged_all_match_exception"] for r in arms)
    assert events == purged
    add(f"**Audit:** summary successes equal accepted-journal successes in all 28 arms; accepted server streams "
        f"reproduce every arm's N/V/M and client request count exactly. Across R7 and the 84 in-root historical references, "
        f"all **{accepted:,}/{covered:,}** accepted episodes have joined termination reasons, with **{exc} surviving exceptions**. "
        f"R7 has {exceptions} exception attempts in timing logs, {purged} purged journal records and {events} purges recorded "
        "by the chain. No R7 exception purge was needed. The join uses "
        "`(task_uid, run_id, attempt)`; stale attempts do not become failures in the accepted set. "
        "[Per-arm exception evidence](analysis_r7/a1_exception_audit.json) and "
        "[chain evidence](analysis_r7/a1_snapshot.json) retain the audit details.")
    discarded = sum(r["decision_audit"]["discarded_prior_prefix"] for r in arms)
    duplicates = sum(r["decision_audit"]["exact_duplicate_decisions"] for r in arms)
    flags = [r for r in refs if r["ledger_minus_client_decisions"]]
    add(f"Accepted-stream selection discarded {discarded:,} stale decision-prefix rows and collapsed {duplicates:,} "
        "exact duplicate decisions. It requires contiguous steps, a client/server request-count match and consistent "
        "within-episode cumulative vision counters. "
        f"Historical references retain R6's registered ledger tolerances ({len(flags)} of the 84 loaded references); "
        "no such tolerance is used for R7. The frozen R6 frontier retains its original seven flagged arms and pricing.")
    add("**Owner cost.** For each non-wrist arm, IR is recomputed as "
        "`[c_v·V/N + (1−c_v)·M/N]·5/L`, with `c_v=.152` for π0.5 and `.148` for GR00T. "
        "All R7 requests have nominal `L=5`. Blind decisions cost zero; pure L10 costs `.5`, pure L5 costs `1`. "
        "The denominator is nominal five-control blocks, not measured applied controls. "
        "The summary's eager `ir_per_five_controls` is a separate cost convention and is never substituted for owner IR.")
    reconciliation = []
    for variant in ("SF1", "UF1", "SW", "CU30", "CT30"):
        group = [r for r in arms if r["variant"] == variant]
        delta = [r["owner_IR"]-r["stock_ledger_IR"] for r in group]
        reconciliation.append([LABEL[variant], f"{min(r['owner_IR'] for r in group):.6f}–{max(r['owner_IR'] for r in group):.6f}",
                               f"{min(r['stock_ledger_IR'] for r in group):.6f}–{max(r['stock_ledger_IR'] for r in group):.6f}",
                               f"{min(delta):+.6f} to {max(delta):+.6f}"])
    add(table(["Family", "Owner IR range", "Eager-summary IR range", "Owner − eager range"], reconciliation))
    add("**Stage wrist pricing.** Primary IR is the sum of accepted per-decision `owner_cost`, independently "
        "recomputed from actual camera/call counters: full look `.152`, wrist look `.055198`, missing-camera "
        "completion `.049890`, policy call `.848`. The wrist/completion prices use **R4's proportional-latency "
        "assumption**; these are cost estimates, not measured hardware speedups. The two-camera column charges "
        "`.152·V/N + .848·M/N` to the same decisions. Historical wrist points retain R6's frozen full-camera prices.")
    add(table(["π0.5 cell", "Wrist owner IR", "Two-camera owner IR", "Eager-summary IR", "Equal-episode owner IR", "Logged − counted IR"],
              [[cell_label(r["cell"]), f"{r['owner_IR']:.6f}", f"{r['owner_IR_full_camera_counterfactual']:.6f}",
                f"{r['stock_ledger_IR']:.6f}", f"{r['owner_IR_equal_episode_mean']:.6f}",
                f"{r['owner_IR_logged_minus_counted']:.2g}"] for r in arms if r["variant"] == "SW"]))
    add(f"**Validation passed:** {validation['unit_tests']} regression checks; all 44 profile arms "
        "(880 episodes, 37,999 decisions) reproduce the independent profile report's outcomes/counts/costs; "
        "four R6 uniform/calibrated-call arms pass raw decision reconciliation; 84 references match frozen R6 "
        "outcomes/costs; all eight R6 Pareto sets and pure-L10 crossing costs reproduce. "
        "[Validation](analysis_r7/a1_validation.json), [cost rows](analysis_r7/a1_costs.csv), "
        "[accepted arm data](analysis_r7/a1_arms.json).")

    add("## 2. Preregistered acceptance verdicts")
    add("All success-rate differences below are **percentage points (pp)**. Intervals are two-sided 95% "
        "task-stratified init bootstraps, 10,000 draws, seed **20260930**. Replicates of A/B are averaged within "
        "each task/init pair before resampling: 500 paired clusters per cell. Pooled comparisons give cells equal "
        "weight, with independent within-cell resampling as in R6. Rules use unrounded endpoints and strict `>` "
        "bounds; cost matching uses inclusive `≤`. These are the preregistered unadjusted comparisons, not "
        "simultaneous certification. Non-significance is not equivalence.")
    rule_rows = [
        ["1. Stage-bounded follow preserves SR while saving vision", "SF−A lower > −1.00 pp; IR lower in ≥6/8",
         f"SF−A {ci(pools['SF1-A']['delta_SR'])}; IR lower in {rules['1']['lower_IR_cells']}/8", rules["1"]["verdict"]],
        ["2. Stage/state signal adds value", "SF−UF lower >0; OR SF has no significant loss vs A and UF−A lower <−1.00 pp",
         f"SF−UF {ci(pools['SF1-UF1']['delta_SR'])}; SF−A upper {100*pools['SF1-A']['delta_SR']['hi']:+.2f}; UF−A lower {100*pools['UF1-A']['delta_SR']['lo']:+.2f}", rules["2"]["verdict"]],
        ["3. Stage wrist preserves SR while saving vision", "SW−A lower >−1.50 pp; IR lower in 4/4",
         f"SW−A {ci(pools['SW-A']['delta_SR'])}; IR lower in {rules['3']['lower_IR_cells']}/4", rules["3"]["verdict"]],
        ["4. Stage-tilted calls beat uniform calls", "CT−CU lower >0; absolute aggregate ΔIR ≤.015 in every cell",
         f"CT−CU {ci(pools['CT30-CU30']['delta_SR'])}; {4-len(rules['4']['unmatched_cells'])}/4 cells IR matched", rules["4"]["verdict"]],
        ["5. Frontier placement", "Non-dominated by frozen R6 single-run points; show CP bounds; no certified NI",
         f"{sum(p['nondominated_by_R6'] for p in frontier['placements'])}/28 non-dominated by R6; {sum(p['nondominated_after_available_R7'] for p in frontier['placements'])}/28 remain after all R7", "reported pointwise (§4)"]]
    add(table(["Rule", "Acceptance condition", "Computed evidence (95% CI)", "Verdict"], rule_rows))
    add("Rule 2's second branch interprets “no significant loss” as the pooled SF−A interval's upper endpoint "
        "being at least zero. It does not mean that rule 1 passed or that equality was established. "
        f"Direct SF−UF branch: **{rules['2']['direct_SF_UF_branch']}**; no-significant-SF-loss clause: "
        f"**{rules['2']['SF_no_significant_loss']}**; UF lower-bound clause: **{rules['2']['UF_lower_below_minus_1pp']}**.")
    add(table(["Pooled comparison", "Cells", "ΔSR pp [95% CI]", "Mean of per-cell aggregate ΔIR"],
              [[key.replace("30", ""), len(p["cells"]), ci(p["delta_SR"]), f"{p['delta_owner_IR_pooled']:+.6f}"]
               for key, p in pools.items()]))
    add("**Rule 4 cost match (fixed before any CT test episode).** A cell outside `.015` is labelled "
        "**IR not matched in that cell**; a favourable pooled mean cannot replace this condition. "
        "The cell SR intervals below remain descriptive; the acceptance SR test is the four-cell pooled bound.")
    rows = []
    for cell in C.SPARSE:
        cu, ct = arms_by_cell[cell, "CU30"], arms_by_cell[cell, "CT30"]
        d = comparisons[cell, "CT30", "CU30"]
        rows.append([cell_label(cell), f"{cu['owner_IR']:.6f}", f"{ct['owner_IR']:.6f}",
                     ci(d["delta_aggregate_owner_IR_CI"], 1, 6),
                     f"{cu['owner_IR_equal_episode_mean']:.6f} / {ct['owner_IR_equal_episode_mean']:.6f}",
                     ci(d["delta_SR"]), rules["4"]["cell_cost_verdict"][cell]])
    add(table(["Cell", "CU aggregate IR", "CT aggregate IR", "CT−CU aggregate ΔIR [95% CI]",
               "Equal-episode IR: CU / CT", "CT−CU SR pp [95% CI]", "Cost-match verdict"], rows))
    p = pools["CT30-CU30"]
    add(f"The four-cell mean aggregate ΔIR is **{p['delta_owner_IR_pooled']:+.6f}**. "
        f"Equal-cell averages of equal-episode IR are CU **{p['equal_episode_IR_pooled']['CU30']:.6f}** and "
        f"CT **{p['equal_episode_IR_pooled']['CT30']:.6f}**, difference **{p['delta_equal_episode_IR_pooled']:+.6f}**. "
        "These alternative summaries do not change the per-cell aggregate-IR gate.")

    add("## 3. Per-cell success, owner cost and paired differences")
    add("**Method key:** **A** is the no-call cache with ten-control commitment; **B** adds guard-triggered policy "
        "rescue. A and B are three-replicate means. **SF1** allows one extra five-control block when successor "
        "support, stable gripper stage and the state valve permit it; **UF1** allows that extension with structural "
        "support alone. **SW** uses the wrist camera in easy stages (π0.5 only). **CU** places calls uniformly with "
        "a calibrated stall trigger; **CT** tilts call placement by stage/state events, with the same stall trigger "
        "and nominal `.30` budget. **Pure L10** calls the policy every ten controls. Library sizes are 50 or 500 "
        "episodes; “—” means that variant was not evaluated in that cell.")
    variants = ("A", "SF1", "UF1", "SW", "CU30", "CT30", "B", "L10")
    add(table(["Model / suite / library", "A (3-rep)", "SF1", "UF1", "SW", "CU", "CT", "B (3-rep)", "Pure L10"],
              [[cell_label(cell), *[point(by_cell[cell, v]) if by_cell[cell, v]["status"] == "complete" else "—"
                                   for v in variants]] for cell in C.CELLS]))
    add("Entries are **SR% @ owner IR**, with IR pooled over decisions within each run. A/B IR is the arithmetic "
        "mean of their three count-derived run ratios, following R6. All R7 arms and pure L10 are single runs.")
    rows = []
    for r in arms:
        cell, v = r["cell"], r["variant"]
        rows.append([cell_label(cell), LABEL[v], *(ci(comparisons[cell, v, ref]["delta_SR"]) for ref in ("A", "B", "L10")),
                     f"{comparisons[cell, v, 'A']['delta_owner_IR']:+.6f}"])
    add(table(["Cell", "Method", "ΔSR vs A pp [95% CI]", "ΔSR vs B pp [95% CI]",
               "ΔSR vs pure L10 pp [95% CI]", "ΔIR vs A"], rows))
    add(table(["Cell", "B−A pp [95% CI]", "A−pure L10 pp [95% CI]", "B−pure L10 pp [95% CI]"],
              [[cell_label(cell), *(ci(comparisons[cell, x, y]["delta_SR"]) for x, y in (("B", "A"), ("A", "L10"), ("B", "L10")))]
               for cell in C.CELLS]))
    add("**Paired stage-signal contrasts.** W/L counts candidate-only/reference-only successes; p is exact "
        "two-sided McNemar. Bootstrap intervals and p values test different summaries and are reported separately.")
    add(table(["Cell", "Contrast", "ΔSR pp [95% CI]", "W/L", "Exact p", "Aggregate ΔIR"],
              [[cell_label(r["cell"]), f"{LABEL[r['variant']]}−{LABEL[r['reference_label']]}", ci(r["delta_SR"]),
                f"{r['exact']['wins']}/{r['exact']['losses']}", f"{r['exact']['mcnemar_two_sided_exact_p']:.5g}", f"{r['delta_owner_IR']:+.6f}"]
               for r in contrasts if (r["variant"], r["reference_label"]) in (("SF1", "UF1"), ("CT30", "CU30"))]))
    add("[All 164 paired contrasts](analysis_r7/a1_contrasts.json) include pure L5, R6 call controls, per-replicate "
        "binary tests, equal-episode cost intervals and aggregate-cost intervals where both arms have episode costs. "
        "McNemar is not applied to fractional A/B replicate means. Historical cost estimates use summary counts, "
        "so no paired cost interval is inferred for them; their paired **SR** intervals are available. "
        "[Per-cell CSV](analysis_r7/a1_per_cell.csv), [paired-difference CSV](analysis_r7/a1_paired_deltas.csv).")

    add("## 4. Placement on the R6 frontier")
    add("A point is dominated when another eligible point has no greater owner cost and no lower SR, with at "
        "least one strict inequality. The comparator is the frozen R6 single-run set, including older eligible "
        "rounds and shared pure references; the A/B means in §3 are not frontier points. No interpolation or "
        "replicate pooling is used. “R6: yes” satisfies rule 5's point-estimate definition; “combined: yes” "
        "also survives comparison with every R7 point in that cell.")
    add("The conservative Q2 lower bound is `CP_lower(W/n; α/2) − CP_upper(L/n; α/2)`, `α=.05`. "
        "A bound strictly above **−2 pp** clears the nominal 2 pp NI margin for that comparison. "
        "For the π0.5 pure-L5 three-run mean, alpha is split over three binary comparisons and their bounds "
        "are averaged. These bounds **do not certify noninferiority** or adjust for selecting points across "
        "the frontier. Pure L5 also contains DUAL references from a different harness/seed, retained as in R6.")
    placements = frontier["placements"]
    add(table(["Cell", "Method", "Non-dominated by R6?", "On combined frontier?", "CP lower vs pure L10 (pp)", "CP lower vs pure L5 (pp)"],
              [[cell_label(p["cell"]), LABEL[p["variant"]], "yes" if p["nondominated_by_R6"] else "no",
                "yes" if p["nondominated_after_available_R7"] else "no", f"{100*p['exact_vs_L10']['cp_lower95']:+.2f}",
                f"{100*p['CP_lower_vs_L5_mean']:+.2f}"] for p in placements]))
    counts_r6 = Counter(p["variant"] for p in placements if p["nondominated_by_R6"])
    counts_all = Counter(p["variant"] for p in placements if p["nondominated_after_available_R7"])
    add(table(["Method", "Non-dominated by R6", "Survives combined frontier"],
              [[LABEL[v], counts_r6[v], counts_all[v]] for v in ("SF1", "UF1", "SW", "CU30", "CT30")]))
    add("**Twenty points extend the historical trade-off; fifteen survive the full R7 comparison.** "
        "All four wrist-camera points have higher success and lower owner IR than their corresponding "
        "stage-bounded-follow points. On π0.5 Spatial / 50, stage-tilted calls also dominate the new "
        "uniform-call point. Three stage-bounded-follow points, five structural-only-follow points, "
        "four wrist-camera points and three stage-tilted-call points remain on the combined frontier. "
        "None of the 28 new points clears the nominal 2 pp CP margin against either pure reference; "
        "the strongest pure-L10 lower bound is −2.36 pp (stage-bounded follow, GR00T Spatial / 500).")
    add("[Frontier output](analysis_r7/a1_frontier.json) identifies every R6 dominator, any R6 frontier points "
        "dominated by each R7 point, coincident points, the best R6 point at no greater IR, paired CP/McNemar "
        "comparisons with that neighbour, and complete before/after frontier membership. All eight original "
        "frontiers reproduce [R6 frontier_final](../r06/frontier_final/frontier_data.json) exactly.")

    add("## 5. Lowest owner IR reaching pure-L10 success before and after R7")
    add("This is an **observed point-estimate crossing**, not equivalence or NI: select the lowest-cost "
        "evaluated point with SR at least the cell's pure-L10 SR. The table includes the pure policy as a "
        "fallback and reports the non-pure crossing separately. A point below the target is never "
        "interpolated upward. R6 prices and its registered historical ledger tolerances are retained.")
    crossings = frontier["lowest_IR_reaching_L10"]
    ids = []
    for r in crossings:
        for k in ("before_including_pure", "after_available_including_pure"):
            if r[k]["id"] not in ids:
                ids.append(r[k]["id"])
    names = {ident: f"P{i+1}" for i, ident in enumerate(ids)}
    def crossing(p):
        return "none" if p is None else f"{names[p['id']]}: {point(p)}"
    add(table(["Cell", "Pure L10 SR", "Before R7: SR @ IR", "After R7: SR @ IR", "IR change", "Non-pure crossing before → after"],
              [[cell_label(r["cell"]), f"{100*r['reference_L10_SR']:.2f}%", crossing(r["before_including_pure"]),
                crossing(r["after_available_including_pure"]),
                f"{r['after_available_including_pure']['owner_IR']-r['before_including_pure']['owner_IR']:+.6f}",
                f"{r['before_nonpure']['owner_IR']:.6f}" + " → " + f"{r['after_available_nonpure']['owner_IR']:.6f}"
                if r["before_nonpure"] and r["after_available_nonpure"] else
                f"{'none' if r['before_nonpure'] is None else r['before_nonpure']['owner_IR']} → {'none' if r['after_available_nonpure'] is None else r['after_available_nonpure']['owner_IR']}"]
               for r in crossings]))
    descriptions = {
        "r04_cost/r4f_p_l10_inf_k10_L10": "Pure policy, ten-control requests",
        "r05_b1/r5b_p_l10_500_hand": "R5 cache with hand-set guard thresholds",
        "r06_frontier/r6q2_pi05_spatial_50_risk_rho0p45": "Task-risk policy-call lottery, nominal IR .45",
        "r06_paper/r5q1_c10_p_sp_500_rep3": "Guard-triggered committed rescue, third replicate",
        "r06_frontier/r6q2_groot_l10_50_risk_rho0p45": "Task-risk policy-call lottery, nominal IR .45",
        "r06_c_validation/r6c_groot_l10_500_C18": "R6 calibrated calls with stall, nominal IR .18",
        "r06_c_validation/r6c_groot_spatial_50_C30": "R6 calibrated calls with stall, nominal IR .30",
        "r06_frontier/r6q2_groot_spatial_500_A15_confirmation": "No-call cache, fifteen-control commitment confirmation",
    }
    add(table(["Point label", "Plain explanation", "Source identity (`run/arm`)"],
              [[names[ident], descriptions[ident], f"`{ident}`"] for ident in ids]))
    add("**All eight crossing costs and selected point identities are unchanged.** π0.5 LIBERO-10 / 50 "
        "still has no non-pure point reaching pure-L10 success. The new GR00T Spatial / 50 stage-tilted-call "
        "point reaches the target at IR .307139, above the existing .297629 crossing. Both new follow "
        "variants on GR00T Spatial / 500 reach the target, also above the existing .051745 crossing. "
        "The other new points remain below their cells' pure-L10 success targets.")
    add("**Reproduction and provenance.** Run these CPU-only commands from the repository root. "
        "The analyzer takes a fresh finite inventory; no experiment is launched or polled.")
    add("```bash\n"
        "taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_analyze.py --phase final\n"
        "taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_validate.py\n"
        "taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_report.py\n``` ")
    add("The run root is `/home/weiland/trace_runs/os_closed_loop/r07_main/`. All numbers derive from "
        "arm summaries, accepted journals, client timing and server decision logs, or the frozen R6 frontier "
        "products. [Snapshot](analysis_r7/a1_snapshot.json) records source identities, hashes, conventions "
        "and rule evaluations; [references](analysis_r7/a1_references.json) records the 84 reloaded historical "
        "arms. No coordinator-ledger number enters a result.")
    target = C.R7 / "ANALYSIS.md"
    tail = A2_HEADING + "\n\n*Placeholder — coordinator to insert A2's mechanism, generality and proposal analysis.*\n"
    if target.exists():
        old = target.read_text()
        if A2_HEADING not in old:
            raise ValueError("Existing report has no A2 boundary; refusing to overwrite it")
        tail = A2_HEADING + old.split(A2_HEADING, 1)[1]
    body = "\n\n".join(lines)
    width = None
    for line in body.splitlines():
        if line.startswith("|"):
            current = len(line.split("|"))
            assert width is None or current == width, "Non-rectangular Markdown table"
            width = current
        else:
            width = None
    for link in re.findall(r"\]\(([^)]+)\)", body):
        assert (C.R7 / link.split("#")[0]).exists(), f"Missing report link: {link}"
    assert len(re.findall(r"^## [1-5]\. ", body, re.M)) == 5
    assert "<!--" not in body
    target.write_text(body + "\n\n" + tail)
    C.write_json(out / "a1_report_checks.json", dict(status="ok", complete_arms=28, cell_rows=64,
        paired_contrasts=164, frontier_points=28, crossing_cells=8, accepted_termination_coverage=covered,
        accepted_exceptions=exc, chain_purges=events, purged_records=purged,
        markdown_tables_rectangular=True, relative_links_exist=True, narrative_invariants_match=True,
        report=C.stamp(target, True), script=C.stamp(Path(__file__), True)))
    print(f"Wrote {target}; report completeness checks passed")


if __name__ == "__main__":
    main()
