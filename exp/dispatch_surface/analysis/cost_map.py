"""Cost-only calibration of the Rev 2 operating points (protocol 3.3, plan 3.6).

OUTCOME-BLIND BY CONSTRUCTION. This module reads cost/tier rows only through
the cost-only loaders, never imports the outcome loaders, the hull module or
the analyzer, and never touches the ``success`` / ``status`` keys. A static
source-lock test enforces all of that.

Mechanical, non-circular order (G1R1-B3 / G1R2-B6):

1. candidate roster frozen before any rollout (Phase 0 roster + Rev 1
   primary points), every source authenticated through its discipline
   package / matrix; all arms share ONE paired task-stratified init-cluster
   bootstrap index (``PCG64(20260829)``, 10000 replicates);
2. SV / S0: decreasing isotonic fit of point-estimate cost on delta
   (weights = decisions, PAV, ties broken by ascending delta); the threshold
   family never passes through the isotonic step -- its order is
   pre-registered (fh70 < fh50 < fh30);
3. endpoints = lowest / highest isotonic cost per family;
4. per replicate: L_r = max over families of the min endpoint cost,
   H_r = min over families of the max endpoint cost;
   qL = quantile(L, .995, "higher"), qH = quantile(H, .005, "lower");
   c_L = ceil_0.1(qL), c_H = floor_0.1(qH), c_1 / c_2 at thirds;
5. middle: the unique remaining candidate, or (spatial SV) the candidate
   whose isotonic cost is closest to the interval midpoint, ties -> smaller
   delta; the interval is never revisited;
6. A-3 fail-closed: three strictly different raw costs per family, endpoints
   enclosing [c_L, c_H], width >= 4.0 ms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis.analytic_cost import (
    assert_unit_costs_match,
    cost_model_digest,
    cost_model_payload,
)
from exp.dispatch_surface.cost_map_api import index_arrays, shared_index as shared_bootstrap_index
from exp.dispatch_surface.analysis.phase0_discipline import (
    assert_rows_claimed,
    executed_arms_by_run,
    validate as validate_phase0,
)
from exp.dispatch_surface.analysis.precheck_io import (
    load_accepted_cells_costonly,
    load_cost_cells_costonly,
)
from exp.dispatch_surface.phase0_roster import (
    FAMILY_S0,
    FAMILY_SV,
    FAMILY_THRESHOLD,
    REV1_CANDIDATES,
    THRESHOLD_ORDER,
    roster_spec,
)
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, NUM_TASKS, official_test_inits

SEED = 20260829
REPLICATES = 10_000
MIN_WIDTH_MS = 4.0
Q_LOW, Q_HIGH = 0.995, 0.005


# ------------------------------------------------------------------
# pure helpers (unit-tested)
# ------------------------------------------------------------------

def ceil_0_1(x: float) -> float:
    return math.ceil(x * 10.0 - 1e-9) / 10.0


def floor_0_1(x: float) -> float:
    return math.floor(x * 10.0 + 1e-9) / 10.0


def round_0_1(x: float) -> float:
    return round(x * 10.0) / 10.0


def decreasing_isotonic(deltas, costs, weights) -> list[float]:
    """Pool-adjacent-violators fit of cost NON-INCREASING in delta.

    ``deltas`` must be strictly increasing (ties are resolved by the caller
    by ascending delta before entry). Weights are decision counts.
    """
    d = [float(x) for x in deltas]
    if any(b <= a for a, b in zip(d, d[1:])):
        raise ValueError("deltas must be strictly increasing")
    blocks = [[float(c), float(w), 1] for c, w in zip(costs, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 0.0:   # violation: cost rises with delta
            c0, w0, n0 = blocks[i]
            c1, w1, n1 = blocks[i + 1]
            merged = [(c0 * w0 + c1 * w1) / (w0 + w1), w0 + w1, n0 + n1]
            blocks[i:i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for c, _w, n in blocks:
        out.extend([c] * n)
    return out


def quantile_index(q: float, n: int, method: str) -> int:
    """Zero-based sorted index NumPy's ``method='higher'|'lower'`` selects for quantile q."""
    pos = q * (n - 1)
    if method == "higher":
        return int(math.ceil(pos - 1e-12))
    if method == "lower":
        return int(math.floor(pos + 1e-12))
    raise ValueError(method)


def interval_from_endpoints(L: np.ndarray, H: np.ndarray) -> dict:
    n = len(L)
    if n != len(H) or n == 0:
        raise SystemExit("L/H replicate arrays must be non-empty and equal length")
    iL, iH = quantile_index(Q_LOW, n, "higher"), quantile_index(Q_HIGH, n, "lower")
    sL, sH = np.sort(L), np.sort(H)
    qL, qH = float(sL[iL]), float(sH[iH])
    if qL != float(np.quantile(L, Q_LOW, method="higher")) or qH != float(np.quantile(H, Q_HIGH, method="lower")):
        raise SystemExit("order-statistic index disagrees with NumPy's quantile method")
    if n == REPLICATES and (iL, iH) != (9950, 49):
        raise SystemExit(f"frozen indices at R={REPLICATES} must be 9950/49, got {iL}/{iH}")
    c_L, c_H = ceil_0_1(qL), floor_0_1(qH)
    out = {"qL": qL, "qH": qH, "c_L": c_L, "c_H": c_H,
           "qL_zero_based_index": iL, "qH_zero_based_index": iH,
           "quantile_methods": {"L": f"{Q_LOW} higher", "H": f"{Q_HIGH} lower"}}
    if c_H > c_L:
        out["c_1"] = round_0_1(c_L + (c_H - c_L) / 3.0)
        out["c_2"] = round_0_1(c_L + 2.0 * (c_H - c_L) / 3.0)
    else:
        out["c_1"] = out["c_2"] = None
    return out


def family_endpoints(deltas: list[float], iso: list[float], arms: list[str]) -> tuple[str, str]:
    """Outer endpoints of a delta family (G2R1-B5).

    Lowest isotonic cost wins the low endpoint; ties go to the LARGEST delta
    (the outermost aggressive point). Highest cost wins the high endpoint;
    ties go to the SMALLEST delta (the outermost conservative point).
    """
    order = sorted(range(len(arms)), key=lambda i: deltas[i])
    lo = min(order, key=lambda i: (iso[i], -deltas[i]))
    hi = max(order, key=lambda i: (iso[i], -deltas[i]))
    if lo == hi:
        raise SystemExit("isotonic costs collapse to one endpoint")
    return arms[lo], arms[hi]


def pick_middle(candidates: list[tuple[float, float]], midpoint: float) -> float:
    """candidates = [(delta, isotonic_cost)]; nearest cost to midpoint, ties -> smaller delta."""
    if not candidates:
        raise ValueError("no middle candidates")
    return sorted(candidates, key=lambda dc: (abs(dc[1] - midpoint), dc[0]))[0][0]


def ratio_of_sums(cells: dict, keys) -> float:
    num = den = 0.0
    for k in keys:
        c, n = cells[k]
        num += c
        den += n
    if den <= 0:
        raise SystemExit("cost denominator is zero")
    return num / den


# ------------------------------------------------------------------
# sources
# ------------------------------------------------------------------

def _load_rev1_source(manifest_path: str, trials: int) -> dict:
    from openpi.cache.components.surface_judge import load_surface_artifact

    manifest, pkg, manifest_sha = pkgmod.load_manifest(manifest_path)
    pkgmod.verify_package(manifest_path)
    verdict = pkgmod.load_json_member(manifest, pkg, "verdict")
    disc = verdict["discipline"]
    assert_unit_costs_match(disc["cost_inputs"]["unit_cost_ms"], what="Rev 1 verdict")
    split_path = pkgmod.verify_member(manifest, pkg, "split_manifest")
    officials = official_test_inits(str(split_path), trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = sorted(REV1_CANDIDATES)
    journal = pkgmod.verify_member(manifest, pkg, "journal")
    per_step = pkgmod.verify_member(manifest, pkg, "per_step")
    ledger = pkgmod.load_json_member(manifest, pkg, "ledger")
    executed = executed_arms_by_run(ledger.get("launches") or [])
    accepted = load_accepted_cells_costonly(str(journal), arms, grid)
    assert_rows_claimed(accepted, executed, what="Rev 1 package")
    cells, summary = load_cost_cells_costonly(str(per_step), arms, accepted, officials)
    if summary["per_step_sha256"] != disc["cost_inputs"]["per_step_sha256"]:
        raise SystemExit("Rev 1 per_step member != verdict per_step digest")
    # Real deltas from the authenticated Rev 1 artifacts (never from names).
    deltas = {}
    for arm, role in (("dsp_sv", "artifact.dsp_sv"), ("dsp_s0", "artifact.dsp_s0"),
                      ("dsp_sv_minus", "artifact.dsp_sv_minus")):
        art = load_surface_artifact(str(pkgmod.verify_member(manifest, pkg, role)))
        deltas[arm] = float(art.delta)
    if deltas["dsp_sv"] != disc["delta_star"] or deltas["dsp_s0"] != disc["delta_star"]:
        raise SystemExit("Rev 1 primary artifacts do not carry the verdict delta_star")
    return {"suite": manifest["suite"], "manifest_sha256": manifest_sha, "grid": grid,
            "cells": cells, "families": {a: REV1_CANDIDATES[a][0] for a in arms},
            "quantiles": {a: REV1_CANDIDATES[a][1] for a in arms},
            "deltas": {a: deltas.get(a) for a in arms},
            "split_manifest_sha256": disc["split_manifest_sha256"],
            "aprime_content_sha256": disc["aprime_content_sha256"],
            "policy_fingerprint": disc["policy_fingerprint"], "library_sha256": disc["library_sha256"],
            "input_sha256": {"rev1_package_manifest": manifest_sha,
                             "matrix": pkgmod.member_sha(manifest, "matrix"),
                             "ledger": pkgmod.member_sha(manifest, "ledger"),
                             "split_manifest": pkgmod.member_sha(manifest, "split_manifest"),
                             "journal": pkgmod.member_sha(manifest, "journal"),
                             "per_step": pkgmod.member_sha(manifest, "per_step")}}


def _load_phase0_source(matrix_path: str, ledger_path: str, split_path: str,
                        journal: str, per_step: str, trials: int) -> dict:
    ctx = validate_phase0(matrix_path, ledger_path, split_path, trials=trials)
    if not ctx["roster_complete"]:
        raise SystemExit("Phase 0 ledger did not execute the full roster; cost-map needs every arm")
    officials = official_test_inits(split_path, trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = [a for a in ctx["arms"] if ctx["families"][a] in (FAMILY_SV, FAMILY_S0)]
    accepted = load_accepted_cells_costonly(journal, arms, grid)
    executed = {k: set(v) for k, v in ctx["executed_arms_by_run"].items()}
    assert_rows_claimed(accepted, executed, what="Phase 0 run")
    cells, summary = load_cost_cells_costonly(per_step, arms, accepted, officials)
    return {"suite": ctx["suite"], "grid": grid, "cells": cells, "families": {a: ctx["families"][a] for a in arms},
            "quantiles": {a: ctx["quantiles"][a] for a in arms}, "deltas": {a: float(ctx["deltas"][a]) for a in arms},
            "split_manifest_sha256": ctx["split_manifest_sha256"], "aprime_content_sha256": ctx["aprime_content_sha256"],
            "policy_fingerprint": ctx["policy_fingerprint"], "library_sha256": ctx["library_sha256"],
            "discipline": {k: v for k, v in ctx.items() if not k.startswith("_")},
            "input_sha256": {"matrix": ctx["arm_matrix_sha256"], "ledger": ctx["launch_manifest_sha256"],
                             "split_manifest": ctx["split_manifest_sha256"],
                             "journal": pkgmod.file_sha256(pathlib.Path(journal)),
                             "per_step": summary["per_step_sha256"],
                             "export_records": list(ctx["export_record_sha256"])}}


# ------------------------------------------------------------------
# the map
# ------------------------------------------------------------------

def build_cost_map(rev1: dict, phase0: dict, *, seed: int = SEED, reps: int = REPLICATES) -> dict:
    for key in ("suite", "grid", "split_manifest_sha256", "aprime_content_sha256",
                "policy_fingerprint", "library_sha256"):
        if rev1[key] != phase0[key]:
            raise SystemExit(f"Rev 1 and Phase 0 sources disagree on {key}")
    suite = rev1["suite"]
    spec = roster_spec(suite)
    cells = {**rev1["cells"], **phase0["cells"]}
    families = {**rev1["families"], **phase0["families"]}
    quantiles = {**rev1["quantiles"], **phase0["quantiles"]}
    deltas = {**rev1["deltas"], **phase0["deltas"]}
    expected = set(spec["rev1_candidates"]) | {a for a, s in spec["arms"].items() if s["family"] in (FAMILY_SV, FAMILY_S0)}
    if set(cells) != expected:
        raise SystemExit(f"candidate set {sorted(cells)} != frozen {sorted(expected)}")
    grid = sorted(rev1["grid"])
    for arm in cells:
        if set(cells[arm]) != set(grid):
            raise SystemExit(f"candidate {arm} does not cover the full paired grid")
    point = {a: ratio_of_sums(cells[a], grid) for a in cells}
    dec = {a: int(sum(n for _c, n in cells[a].values())) for a in cells}

    # step 2: isotonic per delta family; threshold family by pre-registered order
    fam_points: dict[str, dict] = {}
    for fam in (FAMILY_SV, FAMILY_S0):
        arms = sorted((a for a in cells if families[a] == fam), key=lambda a: deltas[a])
        ds = [float(deltas[a]) for a in arms]
        qs = [quantiles[a] for a in arms]
        if any(b <= a for a, b in zip(ds, ds[1:])):
            raise SystemExit(f"{fam}: candidate deltas are not strictly increasing: {ds}")
        if [q for _d, q in sorted(zip(ds, qs))] != sorted(qs):
            raise SystemExit(f"{fam}: delta order disagrees with quantile order")
        iso = decreasing_isotonic(ds, [point[a] for a in arms], [dec[a] for a in arms])
        fam_points[fam] = {"arms": arms, "deltas": ds, "quantiles": qs, "raw_cost": [point[a] for a in arms],
                           "decisions": [dec[a] for a in arms], "isotonic_cost": iso}
    t_arms = list(THRESHOLD_ORDER)
    if set(t_arms) != {a for a in cells if families[a] == FAMILY_THRESHOLD}:
        raise SystemExit("threshold candidates != pre-registered order")
    fam_points[FAMILY_THRESHOLD] = {"arms": t_arms, "raw_cost": [point[a] for a in t_arms],
                                    "decisions": [dec[a] for a in t_arms], "isotonic_cost": None}

    # step 3: endpoints
    endpoints = {}
    for fam, fp in fam_points.items():
        if fam == FAMILY_THRESHOLD:
            lo, hi = t_arms[0], t_arms[-1]
        else:
            lo, hi = family_endpoints(fp["deltas"], fp["isotonic_cost"], fp["arms"])
        endpoints[fam] = {"low": lo, "high": hi}

    # step 4: shared paired bootstrap on the endpoints only
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

    # step 5: middle
    middle = {}
    for fam, fp in fam_points.items():
        if fam == FAMILY_THRESHOLD:
            middle[fam] = t_arms[1]
            continue
        rest = [(d, c, a) for d, c, a in zip(fp["deltas"], fp["isotonic_cost"], fp["arms"])
                if a not in (endpoints[fam]["low"], endpoints[fam]["high"])]
        if not rest:
            raise SystemExit(f"{fam}: no middle candidate")
        if len(rest) == 1:
            middle[fam] = rest[0][2]
        else:
            mid = (interval["c_L"] + interval["c_H"]) / 2.0 if interval["c_1"] is not None else float("nan")
            chosen_d = pick_middle([(d, c) for d, c, _a in rest], mid)
            middle[fam] = next(a for d, _c, a in rest if d == chosen_d)

    # step 6: A-3
    selected = {fam: [endpoints[fam]["low"], middle[fam], endpoints[fam]["high"]] for fam in fam_points}
    a3_problems = []
    if interval["c_1"] is None or interval["c_H"] - interval["c_L"] < MIN_WIDTH_MS:
        a3_problems.append(f"interval width {interval['c_H'] - interval['c_L']:.2f} < {MIN_WIDTH_MS}")
    for fam, arms in selected.items():
        raw = [point[a] for a in arms]
        if len(set(raw)) != 3:
            a3_problems.append(f"{fam}: selected raw costs not strictly different {raw}")
        if not (min(raw) <= interval["c_L"] and max(raw) >= interval["c_H"]):
            a3_problems.append(f"{fam}: endpoints {min(raw):.3f}/{max(raw):.3f} do not enclose the interval")
    return {
        "protocol": "dispatch_surface_rev2_phase0",
        "posthoc_exploratory": True,
        "outcome_blind": True,
        "suite": suite,
        "seed": seed,
        "replicates": reps,
        "rng": "numpy PCG64",
        "quantile_methods": {"L": f"{Q_LOW} higher", "H": f"{Q_HIGH} lower"},
        "bootstrap_index_sha256": index_sha,
        "input_sha256": {"rev1": rev1["input_sha256"], "phase0": phase0["input_sha256"]},
        "deltas": {a: deltas[a] for a in cells},
        "roster_spec": spec,
        "rev1_package_manifest_sha256": rev1["manifest_sha256"],
        "phase0_discipline": phase0.get("discipline"),
        "cost_model": cost_model_payload(),
        "cost_model_digest": cost_model_digest(),
        "point_cost": point,
        "decisions": dec,
        "families": fam_points,
        "endpoints": endpoints,
        "interval": interval,
        "middle": middle,
        "selected": selected,
        "a3_pass": not a3_problems,
        "a3_problems": a3_problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--phase0-arm-matrix", required=True)
    ap.add_argument("--phase0-launch-manifest", required=True)
    ap.add_argument("--phase0-journal", required=True)
    ap.add_argument("--phase0-per-step", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rev1 = _load_rev1_source(args.rev1_package_manifest, args.trials)
    phase0 = _load_phase0_source(args.phase0_arm_matrix, args.phase0_launch_manifest,
                                 args.split_manifest, args.phase0_journal, args.phase0_per_step, args.trials)
    if len(rev1["grid"]) != NUM_TASKS * args.trials:
        raise SystemExit("grid size != tasks x trials")
    out = build_cost_map(rev1, phase0)
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("suite", "interval", "endpoints", "middle", "a3_pass", "a3_problems")}, indent=2))
    print(f"cost_map_frozen sha256 = {hashlib.sha256(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
