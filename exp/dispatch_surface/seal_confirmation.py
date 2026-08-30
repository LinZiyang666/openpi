"""Seal and unseal the fresh-init confirmation (confirmation plan 3.7-h, 3.8;
G2R1-B3 / B4 / B5 / B7).

``seal`` freezes everything the confirmation depends on into
``confirmation_seal.json`` and refuses unless every input is re-derived from
files, never from a self-reported field:

* the outcome design (verdict ``proceed_to_power``), the budget cost map and
  the C roster, which ``validate_c_roster`` rebuilds from the design;
* the formal power record (``validate_power_record``: 4 x 200 formal
  replicates, row / aggregate digests, per-N counts, CP bounds and the
  selection all recomputed) and its replay artifact (a digest-derived subset
  of replicates recomputed from the sources);
* the fresh-pool validation artifact (re-run: official state width, task
  manifest, k-ordered entries and seeds, materialised bytes, three-way
  exclusivity, asset rollup, cross-machine records);
* the P pilot record (re-run from its journal / ledger / plan: 100 Cartesian
  accepted anchor episodes, one launch, SR recomputed within tolerance);
* the task plan (exact Cartesian roster x 10 x N, bound to the C manifest);
* the Action Cache decision record and its seal branch.

``unseal`` is the only way to let an outcome-aware tool read C: it re-runs
``confirmation_discipline.certify`` on the seal, task plan, ledger, journal
and per-step rows and requires the supplied discipline artifact to equal
that result byte for byte, so a certification of another ledger can never
be reused.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib

from exp.dispatch_surface import action_cache_decision as acd
from exp.dispatch_surface import pilot as pilot_mod
from exp.dispatch_surface.analysis import confirmation_power_mc as pmc
from exp.dispatch_surface.analysis.analytic_cost import cost_model_digest
from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest
from exp.dispatch_surface.build_confirmation_task_plan import load_task_plan, verify_task_plan_against_pool
from exp.dispatch_surface.generate_fresh_inits import (
    POOL_QUOTA,
    assert_pool_complete,
    load_pool_manifest,
    validate_pool_validation,
)
from exp.dispatch_surface.phase0_roster import ANCHOR_ARM

PROTOCOL = "dispatch_surface_rev2_confirmation"
ANALYZER_VERSION = "confirmation_analyzer_v1"
SEAL_NAME = "confirmation_seal.json"
UNSEAL_NAME = "unseal_record.json"
PILOT_TOLERANCE_PT = pilot_mod.PILOT_TOLERANCE_PT
PHASE0_ANCHOR_SR = pilot_mod.PHASE0_ANCHOR_SR
SEAL_REQUIRED_KEYS = ("schema", "protocol", "suite", "N", "roster", "contract_arm", "library_sha256", "budget_interval",
                      "c_roster_sha256", "outcome_design_sha256", "budget_cost_map_sha256", "confirmation_task_plan_sha256",
                      "power_record_sha256", "power_replay_sha256", "pool", "pilot", "action_cache_record_sha256",
                      "action_cache_branch", "estimator_digest", "cost_model_digest", "analyzer_version",
                      "protocol_section13_sha256")
SEAL_KEYS = SEAL_REQUIRED_KEYS + ("sealed_at",)


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_section13_sha256(protocol_path: str) -> str:
    text = pathlib.Path(protocol_path).read_text(encoding="utf-8")
    i = text.find("## 13.")
    if i < 0:
        raise SystemExit("protocol has no section 13")
    return hashlib.sha256(text[i:].encode("utf-8")).hexdigest()


def roster_entry(roster: dict, arm: str) -> dict:
    for e in roster["arms"]:
        if e["arm"] == arm:
            return e
    raise SystemExit(f"{arm} not in c_roster")


def build_seal(args) -> dict:
    design = json.loads(pathlib.Path(args.outcome_design).read_text())
    if design.get("verdict") != "proceed_to_power":
        raise SystemExit(f"outcome design verdict {design.get('verdict')!r}: nothing may be sealed")
    if design.get("estimator_version") != budget_mixture_digest():
        raise SystemExit("outcome design estimator != budget_mixture_v1")
    roster = json.loads(pathlib.Path(args.c_roster).read_text())
    cost_map = json.loads(pathlib.Path(args.budget_cost_map).read_text())
    pmc.validate_c_roster(roster, design, args.outcome_design, cost_map, args.budget_cost_map)
    # --- power record + replay (G2R1-B2/B3) ---
    power = json.loads(pathlib.Path(args.power_record).read_text())
    pmc.validate_power_record(power, outcome_design_path=args.outcome_design, c_roster_path=args.c_roster,
                              budget_cost_map_path=args.budget_cost_map)
    if power["verdict"] != "n_selected" or not isinstance(power["selected_N"], int):
        raise SystemExit("power record did not mechanically select N")
    replay = json.loads(pathlib.Path(args.power_replay).read_text())
    pmc.validate_power_replay(replay, power, args.power_record)
    n_prefix = int(power["selected_N"])
    if n_prefix > POOL_QUOTA["C"]:
        raise SystemExit("selected N exceeds the C pool quota")
    # --- fresh pools (G2R1-B5) ---
    pool_validation = validate_pool_validation(args.pool_validation, p_manifest_path=args.pool_manifest_p,
                                               c_manifest_path=args.pool_manifest_c)
    pool_c = load_pool_manifest(args.pool_manifest_c)
    pool_p = load_pool_manifest(args.pool_manifest_p)
    assert_pool_complete(pool_c["pools"]["C"], "C")
    assert_pool_complete(pool_p["pools"]["P"], "P")
    if pool_validation["suite"] != design["suite"] or pool_c["suite"] != design["suite"]:
        raise SystemExit("fresh pools suite != outcome design suite")
    # --- P pilot (G2R1-B4) ---
    anchor = roster_entry(roster, ANCHOR_ARM)
    pilot = pilot_mod.validate_pilot(args.pilot_record, suite=design["suite"], pool_manifest_p_path=args.pool_manifest_p,
                                     anchor_yaml_sha256=anchor["yaml_sha256"])
    # --- task plan (G2R1-B6) ---
    plan, plan_sha = load_task_plan(args.task_plan)
    verify_task_plan_against_pool(plan, args.pool_manifest_c)
    if plan["pool_id"] != "C" or plan["suite"] != design["suite"]:
        raise SystemExit("task plan is not a C plan for this suite")
    if plan["N"] != n_prefix or plan["roster_arms"] != sorted(e["arm"] for e in roster["arms"]):
        raise SystemExit("task plan N/roster != power record / c_roster")
    # --- Action Cache branch (G2R1-B9) ---
    ac_record = acd.load_record(args.action_cache_record)
    ac_pkg = json.loads(pathlib.Path(args.action_cache_package).read_text()) if args.action_cache_package else None
    branch = acd.seal_branch(ac_record, ac_pkg)
    if not branch["ok"]:
        raise SystemExit(f"seal refused: {branch['reason']} {branch.get('required_digests')}")
    arms = [e["arm"] for e in roster["arms"]]
    fams = {e["arm"]: e["family"] for e in roster["arms"]}
    yaml_paths = {e["arm"]: e["yaml_path"] for e in roster["arms"]}
    yaml_sha = {}
    for arm, p in yaml_paths.items():
        got = _file_sha256(pathlib.Path(p))
        if got != roster_entry(roster, arm)["yaml_sha256"]:
            raise SystemExit(f"{arm}: yaml bytes != c_roster")
        yaml_sha[arm] = got
    artifact_paths = {e["arm"]: e["artifact_path"] for e in roster["arms"] if e.get("artifact_path")}
    artifact_sha = {}
    for arm, p in artifact_paths.items():
        got = _file_sha256(pathlib.Path(p))
        if got != roster_entry(roster, arm)["artifact_sha256"]:
            raise SystemExit(f"{arm}: artifact bytes != c_roster")
        artifact_sha[arm] = got
    sv_arms = [a for a in arms if fams[a] == "sv"]
    if not sv_arms:
        raise SystemExit("C roster has no SV arm to supply the launch contract")
    seal = {
        "schema": 1, "protocol": PROTOCOL, "suite": design["suite"], "N": n_prefix,
        "sealed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "roster": {"arms": arms, "families": fams, "yaml_paths": yaml_paths, "yaml_sha256": yaml_sha,
                   "artifact_paths": artifact_paths, "artifact_sha256": artifact_sha,
                   "threshold_pairs": {e["arm"]: e.get("threshold_pair") for e in roster["arms"] if e["family"] == "threshold"}},
        "contract_arm": sv_arms[0],
        "library_sha256": design["library_sha256"],
        "budget_interval": design["budget_interval"],
        "c_roster_sha256": _file_sha256(pathlib.Path(args.c_roster)),
        "outcome_design_sha256": _file_sha256(pathlib.Path(args.outcome_design)),
        "budget_cost_map_sha256": _file_sha256(pathlib.Path(args.budget_cost_map)),
        "confirmation_task_plan_sha256": plan_sha,
        "power_record_sha256": _file_sha256(pathlib.Path(args.power_record)),
        "power_replay_sha256": _file_sha256(pathlib.Path(args.power_replay)),
        "pool": {"pool_id": "C", "manifest_path": str(pathlib.Path(args.pool_manifest_c).resolve()),
                 "manifest_sha256": _file_sha256(pathlib.Path(args.pool_manifest_c)),
                 "p_manifest_sha256": _file_sha256(pathlib.Path(args.pool_manifest_p)),
                 "validation_sha256": _file_sha256(pathlib.Path(args.pool_validation)),
                 "exclusivity": pool_validation["exclusivity"], "state_dim": pool_validation["state_dim"]},
        "pilot": {"record_sha256": _file_sha256(pathlib.Path(args.pilot_record)), "sr": pilot["sr"],
                  "n_episodes": pilot["n_episodes"], "run_id": pilot["run_id"], "anchor_yaml_sha256": anchor["yaml_sha256"]},
        "action_cache_record_sha256": acd.record_sha256(ac_record),
        "action_cache_branch": branch,
        "estimator_digest": budget_mixture_digest(),
        "cost_model_digest": cost_model_digest(),
        "analyzer_version": ANALYZER_VERSION,
        "protocol_section13_sha256": protocol_section13_sha256(args.protocol),
    }
    return seal


def load_seal(path) -> tuple[dict, str]:
    p = pathlib.Path(path)
    seal = json.loads(p.read_text())
    if seal.get("protocol") != PROTOCOL or seal.get("schema") != 1 or set(seal) != set(SEAL_KEYS):
        raise SystemExit(f"{p}: not a confirmation seal")
    missing = [k for k in SEAL_REQUIRED_KEYS if k not in seal]
    if missing:
        raise SystemExit(f"{p}: seal lacks {missing}")
    if seal.get("estimator_digest") != budget_mixture_digest():
        raise SystemExit("seal estimator digest != budget_mixture_v1")
    if seal.get("cost_model_digest") != cost_model_digest():
        raise SystemExit("seal cost model digest != the cost authority")
    if not (seal.get("action_cache_branch") or {}).get("ok"):
        raise SystemExit("seal was written without a passing Action Cache branch")
    for arm, yp in seal["roster"]["yaml_paths"].items():
        if _file_sha256(pathlib.Path(yp)) != seal["roster"]["yaml_sha256"][arm]:
            raise SystemExit(f"{arm}: yaml bytes drifted since sealing")
    for arm, apath in seal["roster"]["artifact_paths"].items():
        if _file_sha256(pathlib.Path(apath)) != seal["roster"]["artifact_sha256"][arm]:
            raise SystemExit(f"{arm}: artifact bytes drifted since sealing")
    return seal, _file_sha256(p)


def write_unseal(seal_path: str, discipline_path: str, ledger_path: str, out_path: str, *,
                 task_plan_path: str, journal_path: str, per_step_path: str) -> dict:
    """Unseal only by re-certifying (G2R1-B7): the supplied discipline artifact
    must equal what ``certify`` produces right now for this seal / task plan /
    ledger / journal / per-step set; a ``passed`` flag is never trusted."""
    from exp.dispatch_surface.analysis import confirmation_discipline as cdisc

    seal, seal_sha = load_seal(seal_path)
    disc = json.loads(pathlib.Path(discipline_path).read_text())
    if not isinstance(disc, dict) or disc.get("protocol") != cdisc.PROTOCOL or disc.get("passed") is not True:
        raise SystemExit("confirmation discipline artifact is not a passing certification")
    expected = {"seal_sha256": seal_sha, "ledger_sha256": _file_sha256(pathlib.Path(ledger_path)),
                "task_plan_sha256": _file_sha256(pathlib.Path(task_plan_path)),
                "journal_sha256": _file_sha256(pathlib.Path(journal_path)),
                "per_step_sha256": _file_sha256(pathlib.Path(per_step_path)),
                "pool_manifest_sha256": seal["pool"]["manifest_sha256"], "suite": seal["suite"], "N": int(seal["N"]),
                "arms": list(seal["roster"]["arms"]), "roster_complete": True}
    for key, val in expected.items():
        if disc.get(key) != val:
            raise SystemExit(f"confirmation discipline {key} does not bind the current inputs; refusing to unseal")
    fresh = cdisc.certify(seal_path, task_plan_path, ledger_path, journal_path, per_step_path)
    if json.dumps(fresh, sort_keys=True) != json.dumps(disc, sort_keys=True):
        raise SystemExit("confirmation discipline artifact != a fresh certification of these inputs; refusing to unseal")
    rec = {"schema": 1, "protocol": PROTOCOL, "unsealed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "seal_sha256": seal_sha, "discipline_sha256": _file_sha256(pathlib.Path(discipline_path)),
           "ledger_sha256": expected["ledger_sha256"], "task_plan_sha256": expected["task_plan_sha256"],
           "journal_sha256": expected["journal_sha256"], "per_step_sha256": expected["per_step_sha256"]}
    pathlib.Path(out_path).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--outcome-design", required=True)
    s.add_argument("--c-roster", required=True)
    s.add_argument("--budget-cost-map", required=True)
    s.add_argument("--power-record", required=True)
    s.add_argument("--power-replay", required=True, help="confirmation_power_mc replay artifact")
    s.add_argument("--pool-manifest-c", required=True)
    s.add_argument("--pool-manifest-p", required=True)
    s.add_argument("--pool-validation", required=True, help="generate_fresh_inits validate artifact")
    s.add_argument("--pilot-record", required=True, help="pilot finalize artifact")
    s.add_argument("--task-plan", required=True)
    s.add_argument("--action-cache-record", required=True)
    s.add_argument("--action-cache-package", default="")
    s.add_argument("--protocol", required=True, help="logs/dispatch_surface_rev2_protocol_draft.md")
    s.add_argument("--out", required=True)
    u = sub.add_parser("unseal")
    u.add_argument("--seal", required=True)
    u.add_argument("--discipline", required=True)
    u.add_argument("--ledger", required=True)
    u.add_argument("--task-plan", required=True)
    u.add_argument("--journal", required=True)
    u.add_argument("--per-step", required=True)
    u.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "seal":
        seal = build_seal(args)
        p = pathlib.Path(args.out)
        p.write_text(json.dumps(seal, indent=2, sort_keys=True))
        print(f"sealed -> {p} sha256={_file_sha256(p)}")
    else:
        rec = write_unseal(args.seal, args.discipline, args.ledger, args.out, task_plan_path=args.task_plan,
                           journal_path=args.journal, per_step_path=args.per_step)
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
