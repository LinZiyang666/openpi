"""Stage 1 of the budget amendment: the cost-only interval (confirmation plan 3.3).

Reads ONLY cost rows (never ``success`` / ``status``) from three sources —
the archived Rev 1 discipline package, the Phase 0 exploratory run and the
finalised threshold-grid package — and freezes the common compute-budget
interval ``[B_L, B_H]`` with the protocol 3.3 mechanics: SV / S0 endpoints
by decreasing isotonic cost over delta, threshold endpoints = the cheapest
and dearest measured arms, shared task-stratified paired bootstrap
(PCG64(20260829), R = 10000), qL = sorted[9950] / qH = sorted[49], ceil / floor
to 0.1 ms, thirds for B_1 / B_2, and A-3' (width >= 4, >= 3 distinct point
costs per family, every family's min cost <= B_L and max cost >= B_H).

This module must never import ``budget_mixture``, ``budget_outcome_design``,
``h1_verdict`` or ``analyze_precheck`` (static source lock).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface import tgrid_package as tpkg
from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.analysis.analytic_cost import (
    assert_unit_costs_match,
    cost_model_digest,
    cost_model_payload,
)
from exp.dispatch_surface.analysis.cost_map import (
    MIN_WIDTH_MS,
    Q_HIGH,
    Q_LOW,
    REPLICATES,
    SEED,
    decreasing_isotonic,
    family_endpoints,
    interval_from_endpoints,
    ratio_of_sums,
    shared_bootstrap_index,
)
from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest
from exp.dispatch_surface.analysis.precheck_io import (
    load_accepted_cells_costonly,
    load_cost_cells_costonly,
)
from exp.dispatch_surface.cost_map_api import index_arrays
from exp.dispatch_surface.phase0_roster import (
    FAMILY_S0,
    FAMILY_SV,
    FAMILY_THRESHOLD,
    REV1_CANDIDATES,
    roster_spec,
)
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, NUM_TASKS, official_test_inits

PROTOCOL = "dispatch_surface_rev2_budget_amendment"


def _load_rev1(manifest_path: str, trials: int) -> dict:
    manifest, pkg, manifest_sha = pkgmod.load_manifest(manifest_path)
    pkgmod.verify_package(manifest_path)
    split = str(pkgmod.verify_member(manifest, pkg, "split_manifest"))
    officials = official_test_inits(split, trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = sorted(REV1_CANDIDATES)
    journal = str(pkgmod.verify_member(manifest, pkg, "journal"))
    per_step = str(pkgmod.verify_member(manifest, pkg, "per_step"))
    accepted = load_accepted_cells_costonly(journal, arms, grid)
    ledger = pkgmod.load_json_member(manifest, pkg, "ledger")
    executed = phase0_discipline.executed_arms_by_run(ledger.get("launches") or [])
    phase0_discipline.assert_rows_claimed(accepted, executed, what="Rev 1 package")
    cells, summary = load_cost_cells_costonly(per_step, arms, accepted, officials)
    verdict = pkgmod.load_json_member(manifest, pkg, "verdict")
    disc = verdict.get("discipline") or {}
    assert_unit_costs_match(disc["cost_inputs"]["unit_cost_ms"], what="Rev 1 verdict")
    yaml_sha = {a: pkgmod.member_sha(manifest, f"yaml.{a}") for a in arms}
    # Real deltas from the authenticated Rev 1 artifacts (never from names), as the Phase 0 cost map does.
    from openpi.cache.components.surface_judge import load_surface_artifact

    deltas = {a: None for a in arms}
    for arm, role in (("dsp_sv", "artifact.dsp_sv"), ("dsp_s0", "artifact.dsp_s0"), ("dsp_sv_minus", "artifact.dsp_sv_minus")):
        deltas[arm] = float(load_surface_artifact(str(pkgmod.verify_member(manifest, pkg, role))).delta)
    if deltas["dsp_sv"] != disc.get("delta_star") or deltas["dsp_s0"] != disc.get("delta_star"):
        raise SystemExit("Rev 1 primary artifacts do not carry the verdict delta_star")
    return {
        "suite": manifest["suite"], "grid": grid, "cells": cells,
        "families": {a: REV1_CANDIDATES[a][0] for a in arms},
        "quantiles": {a: REV1_CANDIDATES[a][1] for a in arms},
        "deltas": deltas,
        "sources": {a: {"kind": "rev1_package", "role": f"yaml.{a}", "yaml_sha256": yaml_sha[a],
                        "yaml_member": manifest["members"][f"yaml.{a}"]["member"],
                        "artifact_role": (f"artifact.{a}" if a in ("dsp_sv", "dsp_s0", "dsp_sv_minus") else None)}
                    for a in arms},
        "split_manifest_sha256": pkgmod.member_sha(manifest, "split_manifest"),
        "aprime_content_sha256": disc.get("aprime_content_sha256"),
        "policy_fingerprint": disc.get("policy_fingerprint"),
        "library_sha256": disc.get("library_sha256"),
        "manifest_sha256": manifest_sha,
        "input_sha256": {"rev1_package_manifest": manifest_sha,
                         "matrix": pkgmod.member_sha(manifest, "matrix"),
                         "ledger": pkgmod.member_sha(manifest, "ledger"),
                         "split_manifest": pkgmod.member_sha(manifest, "split_manifest"),
                         "journal": pkgmod.member_sha(manifest, "journal"),
                         "per_step": pkgmod.member_sha(manifest, "per_step")},
    }


def _load_phase0(matrix_path: str, ledger_path: str, split_path: str, journal: str, per_step: str,
                 trials: int) -> dict:
    ctx = phase0_discipline.validate(matrix_path, ledger_path, split_path, trials=trials)
    if not ctx["roster_complete"]:
        raise SystemExit("Phase 0 ledger did not execute the full roster")
    officials = official_test_inits(split_path, trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = [a for a in ctx["arms"] if ctx["families"][a] in (FAMILY_SV, FAMILY_S0)]
    accepted = load_accepted_cells_costonly(journal, arms, grid)
    executed = {k: set(v) for k, v in ctx["executed_arms_by_run"].items()}
    phase0_discipline.assert_rows_claimed(accepted, executed, what="Phase 0 run")
    cells, summary = load_cost_cells_costonly(per_step, arms, accepted, officials)
    matrix = json.loads(pathlib.Path(matrix_path).read_text())
    return {
        "suite": ctx["suite"], "grid": grid, "cells": cells,
        "families": {a: ctx["families"][a] for a in arms},
        "quantiles": {a: ctx["quantiles"][a] for a in arms},
        "deltas": {a: float(ctx["deltas"][a]) for a in arms},
        "sources": {a: {"kind": "phase0", "matrix_sha256": ctx["arm_matrix_sha256"], "yaml_path": matrix["arms"][a],
                        "yaml_sha256": matrix["arm_yaml_sha256"][a], "artifact_path": matrix["artifact_paths"][a],
                        "artifact_sha256": matrix["artifact_sha256"][a]} for a in arms},
        "split_manifest_sha256": ctx["split_manifest_sha256"], "aprime_content_sha256": ctx["aprime_content_sha256"],
        "policy_fingerprint": ctx["policy_fingerprint"], "library_sha256": ctx["library_sha256"],
        "input_sha256": {"matrix": ctx["arm_matrix_sha256"], "ledger": ctx["launch_manifest_sha256"],
                         "split_manifest": ctx["split_manifest_sha256"],
                         "journal": pkgmod.file_sha256(pathlib.Path(journal)),
                         "per_step": summary["per_step_sha256"],
                         "export_records": list(ctx["export_record_sha256"])},
    }


def _load_tgrid(manifest_path: str, trials: int, *, rev1_manifest_path: str | None = None) -> dict:
    """The finalised package is the only input: every YAML and the roster spec
    are opened through MANIFEST roles, never through the execution-time paths
    the matrix still records (G2R1-B8)."""
    manifest, pkg, manifest_sha = tpkg.load_manifest(manifest_path)
    tpkg.verify_package(manifest_path)
    matrix_p = tpkg.verify_member(manifest, pkg, "matrix")
    ledger_p = tpkg.verify_member(manifest, pkg, "ledger")
    split_p = tpkg.verify_member(manifest, pkg, "split_manifest")
    yaml_paths = {arm: str(tpkg.verify_member(manifest, pkg, f"yaml.{arm}")) for arm in tpkg.grid_arms()}
    ctx = phase0_discipline.validate_tgrid(str(matrix_p), str(ledger_p), str(split_p), trials=trials,
                                           yaml_paths=yaml_paths, rev1_manifest_path=rev1_manifest_path)
    if not ctx["roster_complete"]:
        raise SystemExit("threshold-grid ledger did not execute the full grid")
    officials = official_test_inits(str(split_p), trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = list(ctx["arms"])
    journal = str(tpkg.verify_member(manifest, pkg, "journal"))
    per_step = str(tpkg.verify_member(manifest, pkg, "per_step"))
    accepted = load_accepted_cells_costonly(journal, arms, grid)
    executed = {k: set(v) for k, v in ctx["executed_arms_by_run"].items()}
    phase0_discipline.assert_rows_claimed(accepted, executed, what="threshold-grid package")
    cells, summary = load_cost_cells_costonly(per_step, arms, accepted, officials)
    return {
        "suite": ctx["suite"], "grid": grid, "cells": cells,
        "families": {a: FAMILY_THRESHOLD for a in arms},
        "quantiles": {a: None for a in arms}, "deltas": {a: None for a in arms},
        "nominal": ctx["nominal"], "threshold_pairs": ctx["threshold_pairs"],
        "sources": {a: {"kind": "tgrid_package", "role": f"yaml.{a}", "yaml_sha256": tpkg.member_sha(manifest, f"yaml.{a}"),
                        "yaml_member": manifest["members"][f"yaml.{a}"]["member"],
                        "threshold_pair": ctx["threshold_pairs"][a], "nominal": ctx["nominal"][a]} for a in arms},
        "split_manifest_sha256": ctx["split_manifest_sha256"], "aprime_content_sha256": ctx["aprime_content_sha256"],
        "policy_fingerprint": ctx["policy_fingerprint"], "library_sha256": ctx["library_sha256"],
        "manifest_sha256": manifest_sha,
        "input_sha256": {"tgrid_package_manifest": manifest_sha, "matrix": tpkg.member_sha(manifest, "matrix"),
                         "ledger": tpkg.member_sha(manifest, "ledger"), "journal": tpkg.member_sha(manifest, "journal"),
                         "per_step": tpkg.member_sha(manifest, "per_step"),
                         "split_manifest": tpkg.member_sha(manifest, "split_manifest")},
    }


def build(rev1: dict, phase0: dict, tgrid: dict | None, *, seed: int = SEED, reps: int = REPLICATES) -> dict:
    sources = [rev1, phase0] + ([tgrid] if tgrid is not None else [])
    for key in ("suite", "grid", "split_manifest_sha256", "aprime_content_sha256", "policy_fingerprint", "library_sha256"):
        vals = {json.dumps(sorted(s[key]) if key == "grid" else s[key]) for s in sources}
        if len(vals) != 1:
            raise SystemExit(f"development sources disagree on {key}")
    suite = rev1["suite"]
    cells, families, quantiles, deltas, src = {}, {}, {}, {}, {}
    for s in sources:
        for a in s["cells"]:
            if a in cells:
                raise SystemExit(f"candidate {a} appears in two sources")
            cells[a] = s["cells"][a]
            families[a] = s["families"][a]
            quantiles[a] = s["quantiles"][a]
            deltas[a] = s["deltas"][a]
            src[a] = s["sources"][a]
    grid = sorted(rev1["grid"])
    for a in cells:
        if set(cells[a]) != set(grid):
            raise SystemExit(f"candidate {a} does not cover the full paired grid")
    point = {a: ratio_of_sums(cells[a], grid) for a in cells}
    dec = {a: int(sum(n for _c, n in cells[a].values())) for a in cells}

    fam_points: dict[str, dict] = {}
    for fam in (FAMILY_SV, FAMILY_S0):
        arms = sorted((a for a in cells if families[a] == fam), key=lambda a: deltas[a])
        ds = [float(deltas[a]) for a in arms]
        if any(b <= a for a, b in zip(ds, ds[1:])):
            raise SystemExit(f"{fam}: candidate deltas are not strictly increasing")
        iso = decreasing_isotonic(ds, [point[a] for a in arms], [dec[a] for a in arms])
        fam_points[fam] = {"arms": arms, "deltas": ds, "raw_cost": [point[a] for a in arms],
                           "decisions": [dec[a] for a in arms], "isotonic_cost": iso}
    t_arms = sorted((a for a in cells if families[a] == FAMILY_THRESHOLD), key=lambda a: (point[a], a))
    fam_points[FAMILY_THRESHOLD] = {"arms": t_arms, "raw_cost": [point[a] for a in t_arms],
                                    "decisions": [dec[a] for a in t_arms], "isotonic_cost": None}
    endpoints = {}
    for fam, fp in fam_points.items():
        if fam == FAMILY_THRESHOLD:
            lo, hi = t_arms[0], t_arms[-1]
        else:
            lo, hi = family_endpoints(fp["deltas"], fp["isotonic_cost"], fp["arms"])
        endpoints[fam] = {"low": lo, "high": hi}

    by_task: dict[int, list] = {}
    for t, i in grid:
        by_task.setdefault(t, []).append((t, i))
    picks = shared_bootstrap_index(by_task, seed, reps)
    index_sha = hashlib.sha256(json.dumps(picks, separators=(",", ":")).encode()).hexdigest()
    cell_list, idx = index_arrays(picks, grid)
    C = {a: np.array([cells[a][c][0] for c in cell_list], dtype=np.float64) for a in cells}
    N = {a: np.array([cells[a][c][1] for c in cell_list], dtype=np.float64) for a in cells}

    def rep_cost(arm):
        return C[arm][idx].sum(axis=1) / N[arm][idx].sum(axis=1)

    fam_min, fam_max = [], []
    for fam, ep in endpoints.items():
        a, b = rep_cost(ep["low"]), rep_cost(ep["high"])
        fam_min.append(np.minimum(a, b))
        fam_max.append(np.maximum(a, b))
    L = np.max(np.stack(fam_min), axis=0)
    H = np.min(np.stack(fam_max), axis=0)
    interval = interval_from_endpoints(L, H)

    a3 = []
    if interval["c_1"] is None or interval["c_H"] - interval["c_L"] < MIN_WIDTH_MS:
        a3.append(f"interval width {interval['c_H'] - interval['c_L']:.2f} < {MIN_WIDTH_MS}")
    for fam, fp in fam_points.items():
        raw = [point[a] for a in fp["arms"]]
        if len(set(raw)) < 3:
            a3.append(f"{fam}: fewer than 3 distinct point costs")
        if not (min(raw) <= interval["c_L"] and max(raw) >= interval["c_H"]):
            a3.append(f"{fam}: measured costs {min(raw):.3f}/{max(raw):.3f} do not enclose the interval")
    out = {
        "protocol": PROTOCOL, "posthoc_design_amendment": True, "posthoc_exploratory": True,
        "outcome_blind": True, "suite": suite, "seed": seed, "replicates": reps, "rng": "numpy PCG64",
        "quantile_methods": {"L": f"{Q_LOW} higher", "H": f"{Q_HIGH} lower"},
        "bootstrap_index_sha256": index_sha,
        "estimator_version": budget_mixture_digest(),
        "input_sha256": {"rev1": rev1["input_sha256"], "phase0": phase0["input_sha256"],
                         "tgrid": (tgrid["input_sha256"] if tgrid is not None else None)},
        "rev1_package_manifest_sha256": rev1["manifest_sha256"],
        "tgrid_package_manifest_sha256": (tgrid["manifest_sha256"] if tgrid is not None else None),
        "roster_spec": roster_spec(suite),
        "cost_model": cost_model_payload(), "cost_model_digest": cost_model_digest(),
        "candidates": sorted(cells), "families": families, "quantiles": quantiles, "deltas": deltas,
        "sources": src, "point_cost": point, "decisions": dec, "family_points": fam_points,
        "endpoints": endpoints,
        "budget_interval": {"B_L": interval["c_L"], "B_H": interval["c_H"], "B_1": interval["c_1"], "B_2": interval["c_2"],
                            "qL": interval["qL"], "qH": interval["qH"]},
        "interval_raw": interval,
        "a3_pass": not a3, "a3_problems": a3,
    }
    if tgrid is not None:
        out["tgrid_nominal"] = tgrid["nominal"]
        out["tgrid_threshold_pairs"] = tgrid["threshold_pairs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--phase0-arm-matrix", required=True)
    ap.add_argument("--phase0-launch-manifest", required=True)
    ap.add_argument("--phase0-journal", required=True)
    ap.add_argument("--phase0-per-step", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--tgrid-package-manifest", default="", help="finalised threshold-grid package (libero_10)")
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS, help=f"frozen at {FORMAL_TRIALS}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.trials != FORMAL_TRIALS:
        raise SystemExit(f"formal budget cost map is frozen at --trials {FORMAL_TRIALS}")
    rev1 = _load_rev1(args.rev1_package_manifest, args.trials)
    phase0 = _load_phase0(args.phase0_arm_matrix, args.phase0_launch_manifest, args.split_manifest,
                          args.phase0_journal, args.phase0_per_step, args.trials)
    tgrid = _load_tgrid(args.tgrid_package_manifest, args.trials, rev1_manifest_path=args.rev1_package_manifest) \
        if args.tgrid_package_manifest else None
    if len(rev1["grid"]) != NUM_TASKS * args.trials:
        raise SystemExit("grid size != tasks x trials")
    out = build(rev1, phase0, tgrid)
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("suite", "budget_interval", "endpoints", "a3_pass", "a3_problems")}, indent=2))
    print(f"budget_cost_map sha256 = {hashlib.sha256(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
