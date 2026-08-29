"""Stage 2 of Phase 0: outcome-aware DESIGN quantities (plan section 3.7).

Runs only after ``cost_map_frozen.json`` exists and its SHA is supplied on
the command line; before any outcome is read it recomputes the exact digest
of every input the cost map froze (Rev 1 package manifest / matrix / ledger /
split / journal / per-step and the Phase 0 matrix / ledger / split / journal /
per-step / export records) and refuses any drift (G2R1-B2). It then reloads
the cost cells and checks point costs and decision counts against the frozen
map. It never writes the cost map, never changes arms or the interval.

Frozen definitions (G1R3-B3):
  effect = full-sample plug-in AUC difference on all development cells with
           the frozen three points and [c_L, c_H]; the family hulls must cover
           the interval, else the gate fails;
  sd30   = sample SD of the paired bootstrap AUC differences (replicates
           lacking support are scored -1.0 and KEPT); joint miss rate > 1%
           fails the gate;
  power  = Phi(effect / (sd30 * sqrt(30/40)) - z_0.95).

Decision gates are reported separately (G2R1-B6): A-2 (spatial descriptive),
A-5 (H2 support), A-6 (H2 power only). Per-task descriptive blocks are
emitted for every task; they are never used as a selection gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.analysis.analytic_cost import cost_matches, cost_model_digest, cost_model_payload
from exp.dispatch_surface.analysis.cost_map import MIN_WIDTH_MS, Q_HIGH, Q_LOW, REPLICATES, SEED
from exp.dispatch_surface.analysis.frontier_hull import (
    SUPPORT_MISS_AUC,
    auc_with_support,
    covers,
    frontier_difference_extrema,
    upper_concave_hull,
)
from exp.dispatch_surface.analysis.precheck_io import load_accepted_episodes, load_analytic_cost
from exp.dispatch_surface.cost_map_api import index_arrays, shared_index_from_map
from exp.dispatch_surface.phase0_roster import (
    FAMILY_S0,
    FAMILY_SV,
    FAMILY_THRESHOLD,
    PROTOCOL_PHASE0,
    roster_spec,
)
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, official_test_inits

MAX_JOINT_MISS = 0.01
Z_095 = 1.6448536269514722
N_DEV, N_CONF = 30, 40
CEILING_SR, FLOOR_SR = 0.95, 0.05
PAIRS = {"H1": (FAMILY_SV, FAMILY_THRESHOLD), "H2": (FAMILY_SV, FAMILY_S0)}


COST_MAP_REQUIRED = (
    "protocol", "posthoc_exploratory", "outcome_blind", "suite", "seed", "replicates", "rng",
    "quantile_methods", "bootstrap_index_sha256", "input_sha256", "rev1_package_manifest_sha256",
    "cost_model", "cost_model_digest", "roster_spec", "point_cost", "decisions", "selected",
    "endpoints", "middle", "interval", "a3_pass",
)
_REV1_DIGEST_KEYS = frozenset({"rev1_package_manifest", "matrix", "ledger", "split_manifest", "journal", "per_step"})
_PHASE0_DIGEST_KEYS = frozenset({"matrix", "ledger", "split_manifest", "journal", "per_step", "export_records"})


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_cost_map_header(cost_map: dict) -> None:
    """Fail closed unless the cost map is the frozen protocol object (G2R2-B1)."""
    if not isinstance(cost_map, dict):
        raise SystemExit("cost map must be a JSON object")
    missing = [k for k in COST_MAP_REQUIRED if k not in cost_map]
    if missing:
        raise SystemExit(f"cost map lacks {missing}")
    if cost_map.get("protocol") != PROTOCOL_PHASE0:
        raise SystemExit("cost map protocol is not the frozen Phase 0 protocol")
    seed, reps = cost_map.get("seed"), cost_map.get("replicates")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != SEED:
        raise SystemExit(f"cost map seed {seed!r} is not the frozen {SEED}")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps != REPLICATES:
        raise SystemExit(f"cost map replicates {reps!r} is not the frozen {REPLICATES}")
    if cost_map.get("posthoc_exploratory") is not True or cost_map.get("outcome_blind") is not True:
        raise SystemExit("cost map is not marked posthoc_exploratory/outcome_blind")
    if cost_map.get("a3_pass") is not True:
        raise SystemExit("cost map failed A-3; outcome design is not defined")
    if cost_map.get("rng") != "numpy PCG64" or cost_map.get("quantile_methods") != {
        "L": f"{Q_LOW} higher", "H": f"{Q_HIGH} lower",
    }:
        raise SystemExit("cost map RNG/quantile method differs from the frozen protocol")
    if cost_map.get("cost_model_digest") != cost_model_digest() or cost_map.get("cost_model") != cost_model_payload():
        raise SystemExit("cost map cost model differs from the analytic cost authority")

    suite = cost_map.get("suite")
    try:
        expected_spec = roster_spec(suite)
    except (KeyError, TypeError):
        raise SystemExit(f"cost map suite {suite!r} has no frozen roster") from None
    if cost_map.get("roster_spec") != expected_spec:
        raise SystemExit("cost map roster spec differs from the frozen suite roster")

    inputs = cost_map.get("input_sha256")
    if not isinstance(inputs, dict) or set(inputs) != {"rev1", "phase0"}:
        raise SystemExit("cost map input_sha256 must contain exactly rev1 and phase0")
    for section, keys in (("rev1", _REV1_DIGEST_KEYS), ("phase0", _PHASE0_DIGEST_KEYS)):
        block = inputs.get(section)
        if not isinstance(block, dict) or set(block) != keys:
            raise SystemExit(f"cost map input_sha256.{section} has the wrong schema")
        for key, value in block.items():
            values = value if key == "export_records" else [value]
            if not isinstance(values, list) or not values or any(not _is_sha256(v) for v in values):
                raise SystemExit(f"cost map input_sha256.{section}.{key} is not a non-empty SHA-256 binding")
    if cost_map.get("rev1_package_manifest_sha256") != inputs["rev1"]["rev1_package_manifest"]:
        raise SystemExit("cost map Rev 1 manifest bindings disagree")

    candidates = {**expected_spec["rev1_candidates"], **expected_spec["arms"]}
    candidates.pop(expected_spec["anchor_arm"], None)
    expected_arms = set(candidates)
    if set(cost_map.get("point_cost") or {}) != expected_arms or set(cost_map.get("decisions") or {}) != expected_arms:
        raise SystemExit("cost map point-cost/decision roster differs from the frozen candidate set")
    for arm in expected_arms:
        cost, decisions = cost_map["point_cost"][arm], cost_map["decisions"][arm]
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost <= 0:
            raise SystemExit(f"cost map point cost for {arm} is not finite and positive")
        if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions <= 0:
            raise SystemExit(f"cost map decision count for {arm} is not a positive integer")
    selected = cost_map.get("selected")
    if not isinstance(selected, dict) or set(selected) != {FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD}:
        raise SystemExit("cost map selected families differ from the frozen three-family design")
    endpoints, middle = cost_map.get("endpoints"), cost_map.get("middle")
    if not isinstance(endpoints, dict) or set(endpoints) != set(selected) \
            or not isinstance(middle, dict) or set(middle) != set(selected):
        raise SystemExit("cost map endpoints/middle do not cover exactly the selected families")
    for family, arms in selected.items():
        if not isinstance(arms, list) or len(arms) != 3 or len(set(arms)) != 3:
            raise SystemExit(f"cost map selected {family} arms must be three distinct candidates")
        if any(arm not in candidates or candidates[arm]["family"] != family for arm in arms):
            raise SystemExit(f"cost map selected {family} contains an arm from another family")
        ep = endpoints[family]
        if not isinstance(ep, dict) or set(ep) != {"low", "high"} \
                or arms != [ep["low"], middle[family], ep["high"]]:
            raise SystemExit(f"cost map selected {family} does not equal low/middle/high")

    interval = cost_map.get("interval")
    if not isinstance(interval, dict) or not all(k in interval for k in ("c_L", "c_1", "c_2", "c_H")):
        raise SystemExit("cost map interval lacks its four frozen anchors")
    anchors = [interval[k] for k in ("c_L", "c_1", "c_2", "c_H")]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in anchors) \
            or not (anchors[0] < anchors[1] < anchors[2] < anchors[3]):
        raise SystemExit("cost map interval anchors must be finite and strictly increasing")
    if anchors[3] - anchors[0] < MIN_WIDTH_MS:
        raise SystemExit(f"cost map interval width is below the frozen {MIN_WIDTH_MS} ms A-3 minimum")
    for family, arms in selected.items():
        raw = [float(cost_map["point_cost"][arm]) for arm in arms]
        if len(set(raw)) != 3 or not (min(raw) <= anchors[0] and max(raw) >= anchors[3]):
            raise SystemExit(f"cost map selected {family} does not satisfy the mechanical A-3 cost checks")
    if not _is_sha256(cost_map.get("bootstrap_index_sha256")):
        raise SystemExit("cost map bootstrap index digest is not SHA-256")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power_n40(effect: float, sd30: float) -> float | None:
    if not (math.isfinite(effect) and math.isfinite(sd30)) or sd30 <= 0 or effect <= 0:
        return None
    return norm_cdf(effect / (sd30 * math.sqrt(N_DEV / N_CONF)) - Z_095)


def family_points(cells, sr, arms, keys):
    pts = []
    for a in arms:
        num = sum(cells[a][k][0] for k in keys)
        den = sum(cells[a][k][1] for k in keys)
        pts.append((num / den, float(np.mean([sr[a][k] for k in keys]))))
    return pts


class _Replicates:
    """Per-replicate (cost, SR) of every arm, vectorised over the shared index."""

    def __init__(self, cells, sr, arms, cell_list, idx):
        self.n = idx.shape[0]
        self.cost, self.sr = {}, {}
        for a in arms:
            C = np.array([cells[a][c][0] for c in cell_list], dtype=np.float64)
            N = np.array([cells[a][c][1] for c in cell_list], dtype=np.float64)
            S = np.array([sr[a][c] for c in cell_list], dtype=np.float64)
            self.cost[a] = C[idx].sum(axis=1) / N[idx].sum(axis=1)
            self.sr[a] = S[idx].mean(axis=1)

    def points(self, arms, r):
        return [(float(self.cost[a][r]), float(self.sr[a][r])) for a in arms]


def _hypothesis(fa, fb, sel, cells, sr, grid, reps: _Replicates, c_L, c_H) -> dict:
    plug_a, plug_b = family_points(cells, sr, sel[fa], grid), family_points(cells, sr, sel[fb], grid)
    ha, hb = upper_concave_hull(plug_a), upper_concave_hull(plug_b)
    plug_cov = covers(ha, c_L, c_H) and covers(hb, c_L, c_H)
    effect = auc_with_support(plug_a, plug_b, c_L, c_H)[0] if plug_cov else None
    boots = np.empty(reps.n)
    misses = 0
    for r in range(reps.n):
        v, miss = auc_with_support(reps.points(sel[fa], r), reps.points(sel[fb], r), c_L, c_H)
        boots[r] = v
        misses += int(miss)
    miss_rate = misses / reps.n
    sd30 = float(np.std(boots, ddof=1)) if reps.n > 1 else float("nan")
    support_ok = plug_cov and miss_rate <= MAX_JOINT_MISS
    return {
        "families": [fa, fb],
        "effect_plugin": effect,
        "plugin_hulls_cover": plug_cov,
        "sd30_including_support_miss": sd30,
        "bootstrap_mean": float(np.mean(boots)),
        "bootstrap_q05": float(np.quantile(boots, 0.05)),
        "bootstrap_q95": float(np.quantile(boots, 0.95)),
        "support_miss_rate": miss_rate,
        "support_miss_value": SUPPORT_MISS_AUC,
        "support_ok": support_ok,
        "power_n40": power_n40(effect, sd30) if (support_ok and effect is not None) else None,
        "verdict": "ok" if support_ok else "support_miss",
    }


def _per_task(sel, cells, sr, grid) -> dict:
    out = {}
    tasks = sorted({k[0] for k in grid})
    for t in tasks:
        keys = [k for k in grid if k[0] == t]
        block = {"n_cells": len(keys), "families": {}}
        for fam, arms in sel.items():
            pts = family_points(cells, sr, arms, keys)
            hull = upper_concave_hull(pts)
            block["families"][fam] = {"arms": arms, "points": pts, "hull": hull,
                                      "support": [hull[0][0], hull[-1][0]]}
        srs = [p[1] for fam in sel for p in block["families"][fam]["points"]]
        block["ceiling"] = min(srs) >= CEILING_SR
        block["floor"] = max(srs) <= FLOOR_SR
        for name, (fa, fb) in PAIRS.items():
            lo = max(block["families"][fa]["support"][0], block["families"][fb]["support"][0])
            hi = min(block["families"][fa]["support"][1], block["families"][fb]["support"][1])
            if hi > lo:
                v, miss = auc_with_support(block["families"][fa]["points"], block["families"][fb]["points"], lo, hi)
                block[f"{name}_descriptive_auc"] = None if miss else v
                block[f"{name}_descriptive_interval"] = [lo, hi]
            else:
                block[f"{name}_descriptive_auc"] = None
                block[f"{name}_descriptive_interval"] = None
            a_mid, b_mid = sel[fa][1], sel[fb][1]
            win = sum(1 for k in keys if sr[a_mid][k] > sr[b_mid][k])
            lose = sum(1 for k in keys if sr[a_mid][k] < sr[b_mid][k])
            block[f"{name}_middle_discordance"] = {"arms": [a_mid, b_mid], "win": win, "lose": lose}
        out[str(t)] = block
    return out


def design(cost_map: dict, cells: dict, sr: dict, picks: list, *, suite: str) -> dict:
    sel = cost_map["selected"]
    c_L, c_H = cost_map["interval"]["c_L"], cost_map["interval"]["c_H"]
    grid = sorted(next(iter(cells.values())).keys())
    cell_list, idx = index_arrays(picks, grid)
    arms = sorted({a for fam_arms in sel.values() for a in fam_arms})
    reps = _Replicates(cells, sr, arms, cell_list, idx)
    out = {"interval": [c_L, c_H], "families": {}, "hypotheses": {}, "decision_gates": {}}
    for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD):
        pts = family_points(cells, sr, sel[fam], grid)
        hull = upper_concave_hull(pts)
        out["families"][fam] = {"arms": sel[fam], "points": pts, "hull": hull,
                                "covers_interval": covers(hull, c_L, c_H)}
    for name, (fa, fb) in PAIRS.items():
        out["hypotheses"][name] = _hypothesis(fa, fb, sel, cells, sr, grid, reps, c_L, c_H)
    h2 = out["hypotheses"]["H2"]
    # A-2: spatial descriptive only -- is the SV hull dominated by threshold on the interval?
    h_sv = upper_concave_hull(out["families"][FAMILY_SV]["points"])
    h_t = upper_concave_hull(out["families"][FAMILY_THRESHOLD]["points"])
    a2 = {"applies": suite == "libero_spatial", "descriptive": True}
    if covers(h_sv, c_L, c_H) and covers(h_t, c_L, c_H):
        ext = frontier_difference_extrema(h_sv, h_t, c_L, c_H)     # exact, at breakpoints (G2R2-B2)
        a2.update({"sv_minus_threshold_sr_min": ext["min"], "sv_minus_threshold_sr_max": ext["max"],
                   "argmin_cost": ext["argmin_cost"], "argmax_cost": ext["argmax_cost"],
                   "checked_breakpoints": ext["breakpoints"],
                   "sv_dominated_on_interval": ext["a_dominated"]})
    else:
        a2.update({"reason": "a family hull does not cover the frozen interval"})
    out["decision_gates"]["A2"] = a2
    out["decision_gates"]["A5"] = {"pass": bool(h2["support_ok"]),
                                   "reason": ("H2 families cover the interval and joint support-miss <= 1%"
                                              if h2["support_ok"] else "H2 support gate failed (hull coverage or joint miss > 1%)")}
    a6_ok = bool(h2["support_ok"] and h2["effect_plugin"] is not None and h2["effect_plugin"] > 0
                 and math.isfinite(h2["sd30_including_support_miss"]) and h2["sd30_including_support_miss"] > 0
                 and h2["power_n40"] is not None and h2["power_n40"] >= 0.80)
    if not h2["support_ok"]:
        reason = "A-5 failed"
    elif h2["effect_plugin"] is None or h2["effect_plugin"] <= 0:
        reason = "H2 plug-in effect <= 0: no directional hypothesis"
    elif not (math.isfinite(h2["sd30_including_support_miss"]) and h2["sd30_including_support_miss"] > 0):
        reason = "sd30 not finite/positive"
    elif h2["power_n40"] is None or h2["power_n40"] < 0.80:
        reason = f"H2 power at N=40 is {h2['power_n40']} < 0.80"
    else:
        reason = "H2 power at N=40 >= 0.80"
    out["decision_gates"]["A6"] = {"pass": a6_ok, "power_n40": h2["power_n40"], "reason": reason}
    out["per_task_descriptive"] = _per_task(sel, cells, sr, grid)
    loto = {}
    for name, (fa, fb) in PAIRS.items():
        loto[name] = {}
        for t in sorted({k[0] for k in grid}):
            keys = [k for k in grid if k[0] != t]
            v, miss = auc_with_support(family_points(cells, sr, sel[fa], keys),
                                       family_points(cells, sr, sel[fb], keys), c_L, c_H)
            loto[name][str(t)] = None if miss else v
    out["leave_one_task_out"] = loto
    out["note"] = ("leave_one_task_out positive means the pooled AUC stays positive after dropping "
                   "that task; it is NOT a claim that every task is positive -- see per_task_descriptive")
    return out


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-map", required=True)
    ap.add_argument("--cost-map-sha256", required=True, help="the frozen SHA; must match the file")
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--phase0-arm-matrix", required=True)
    ap.add_argument("--phase0-launch-manifest", required=True)
    ap.add_argument("--phase0-journal", required=True)
    ap.add_argument("--phase0-per-step", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)


def run(args) -> dict:
    cm_path = pathlib.Path(args.cost_map)
    cm_bytes = cm_path.read_bytes()
    if hashlib.sha256(cm_bytes).hexdigest() != args.cost_map_sha256:
        raise SystemExit("cost map SHA does not match the frozen value supplied")
    cost_map = json.loads(cm_bytes)
    validate_cost_map_header(cost_map)
    frozen = cost_map["input_sha256"]
    # --- exact digest re-check of EVERY frozen input, before any outcome is read ---
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    if manifest_sha != frozen["rev1"]["rev1_package_manifest"] or manifest_sha != cost_map["rev1_package_manifest_sha256"]:
        raise SystemExit("--rev1-package-manifest is not the package the cost map froze")
    for role, key in (("matrix", "matrix"), ("ledger", "ledger"), ("split_manifest", "split_manifest"),
                      ("journal", "journal"), ("per_step", "per_step")):
        if pkgmod.member_sha(manifest, role) != frozen["rev1"][key]:
            raise SystemExit(f"Rev 1 package member {role} differs from the cost map's frozen digest")
    ctx = phase0_discipline.validate(args.phase0_arm_matrix, args.phase0_launch_manifest,
                                     args.split_manifest, trials=args.trials)
    if ctx["rev1_package_manifest_sha256"] != manifest_sha:
        raise SystemExit("Phase 0 matrix binds a different Rev 1 package than the cost map")
    p0 = frozen["phase0"]
    checks = {"matrix": ctx["arm_matrix_sha256"], "ledger": ctx["launch_manifest_sha256"],
              "split_manifest": ctx["split_manifest_sha256"],
              "journal": _sha(pathlib.Path(args.phase0_journal)),
              "per_step": _sha(pathlib.Path(args.phase0_per_step))}
    for key, got in checks.items():
        if got != p0[key]:
            raise SystemExit(f"Phase 0 {key} differs from the cost map's frozen digest")
    if list(ctx["export_record_sha256"]) != list(p0["export_records"]):
        raise SystemExit("Phase 0 export records differ from the cost map's frozen digests")
    checks["export_records"] = list(ctx["export_record_sha256"])
    if cost_map.get("suite") != ctx["suite"]:
        raise SystemExit("cost map suite != Phase 0 suite")
    # --- now outcomes may be read ---
    officials = official_test_inits(args.split_manifest, args.trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    needed = sorted({a for arms in cost_map["selected"].values() for a in arms})
    rev1_arms = [a for a in needed if a in cost_map["roster_spec"]["rev1_candidates"]]
    p0_arms = [a for a in needed if a not in cost_map["roster_spec"]["rev1_candidates"]]
    cells, sr = {}, {}
    rev1_ledger = pkgmod.load_json_member(manifest, pkg, "ledger")
    rev1_exec = phase0_discipline.executed_arms_by_run(rev1_ledger.get("launches") or [])
    if rev1_arms:
        j = str(pkgmod.verify_member(manifest, pkg, "journal"))
        ps = str(pkgmod.verify_member(manifest, pkg, "per_step"))
        acc = load_accepted_episodes(j, rev1_arms, grid)
        phase0_discipline.assert_rows_claimed(acc, rev1_exec, what="Rev 1 outcomes")
        c, _ = load_analytic_cost(ps, rev1_arms, acc, officials)
        cells.update(c)
        sr.update({a: {k: float(v["success"]) for k, v in acc[a].items()} for a in rev1_arms})
    if p0_arms:
        acc = load_accepted_episodes(args.phase0_journal, p0_arms, grid)
        phase0_discipline.assert_rows_claimed(acc, {k: set(v) for k, v in ctx["executed_arms_by_run"].items()},
                                              what="Phase 0 outcomes")
        c, _ = load_analytic_cost(args.phase0_per_step, p0_arms, acc, officials)
        cells.update(c)
        sr.update({a: {k: float(v["success"]) for k, v in acc[a].items()} for a in p0_arms})
    # --- consistency of reloaded cost cells with the frozen map ---
    keys = sorted(grid)
    for a in needed:
        num = sum(cells[a][k][0] for k in keys)
        den = sum(cells[a][k][1] for k in keys)
        if not cost_matches(num / den, cost_map["point_cost"][a]) or int(den) != cost_map["decisions"][a]:
            raise SystemExit(f"reloaded cost/decisions for {a} differ from the frozen cost map")
    picks = shared_index_from_map(cost_map, grid)
    out = design(cost_map, cells, sr, picks, suite=ctx["suite"])
    out.update({"protocol": "dispatch_surface_rev2_phase0", "posthoc_exploratory": True,
                "development_only": True, "cost_map_sha256": args.cost_map_sha256,
                "suite": ctx["suite"], "input_sha256": {"rev1": frozen["rev1"], "phase0": checks}})
    # The cost map must be byte-identical to the frozen value for the WHOLE run;
    # check again before anything is written, then write atomically.
    if hashlib.sha256(cm_path.read_bytes()).hexdigest() != args.cost_map_sha256:
        raise SystemExit("cost map changed during outcome design — refusing")
    out_path = pathlib.Path(args.out)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    os.replace(tmp_path, out_path)
    print(json.dumps(out["decision_gates"], indent=2))
    return out


if __name__ == "__main__":
    main()
