"""P pilot: the one-shot anchor-only check on the fresh P pool
(confirmation plan 3.7-f, G2R1-B4).

The pilot runs ``always_full_inference`` on every P state (10 tasks x 10
states = 100 episodes) under ``run_precheck --layer pilot``; the ledger for a
pilot may hold exactly ONE launch. ``finalize`` certifies the run
outcome-blind first (P task plan bound to the P manifest, one registered
launch, 100 Cartesian accepted cells, anchor YAML by content, anchor all-MISS
per the cost authority, decisions == infers) and only then reads the
terminal status to recompute the success rate from the journal. The pilot
record it writes carries every input digest; ``validate_pilot`` re-derives
all of it from the referenced files, so the seal never trusts an ``sr``
field. ``|SR_P - SR_0a| > 10 pt`` is ``generator_validation_failed`` and is
never retried with another seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from exp.dispatch_surface.analysis.analytic_cost import cost_matches, cost_model_digest, unit_cost
from exp.dispatch_surface.analysis.confirmation_io import load_accepted_c, load_cost_cells_c
from exp.dispatch_surface.build_confirmation_task_plan import load_task_plan, verify_task_plan_against_pool
from exp.dispatch_surface.generate_fresh_inits import POOL_QUOTA, load_pool_manifest
from exp.dispatch_surface.phase0_roster import ANCHOR_ARM
from exp.dispatch_surface.run_precheck import LAYER_PILOT, NUM_TASKS, PROTOCOL_PILOT

PILOT_POOL = "P"
PILOT_TRIALS = POOL_QUOTA[PILOT_POOL]
PILOT_TOLERANCE_PT = 10.0
PHASE0_ANCHOR_SR = {"libero_10": 0.8466666666666667}
PILOT_FROZEN_LAUNCH_KEYS = ("task_plan_sha256", "pool_digest", "N", "cost_model_digest")
RECORD_KEYS = ("protocol", "suite", "arm", "pool_id", "attempt", "one_shot", "n_episodes", "successes", "sr",
               "reference_sr", "tolerance_pt", "passed", "run_id", "policy_fingerprint", "env_seed", "replan_steps",
               "verdict_counts", "inputs")
INPUT_ROLES = ("task_plan", "pool_manifest", "ledger", "journal", "per_step", "anchor_yaml")


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def validate_anchor_yaml(path: str) -> str:
    """The pilot arm must be the anchor configuration: always_search gate,
    threshold judge above 1.0 (never a hit), no warm tier, write_policy never."""
    from openpi.cache.config import load_cache_config

    cfg = load_cache_config(path)
    if cfg.routing is not None:
        raise SystemExit(f"anchor yaml has executor routing ({path})")
    cp1 = cfg.checkpoints.get("cp1")
    if cp1 is None or cp1.gate.type != "always_search":
        raise SystemExit(f"anchor yaml must probe every step (always_search) ({path})")
    if cp1.judge.type != "threshold" or not (float(cp1.judge.threshold) > 1.0) or cp1.judge.warm_tiers:
        raise SystemExit(f"anchor yaml must be a threshold judge > 1.0 with no warm tier ({path})")
    if cfg.write_policy.type != "never":
        raise SystemExit(f"anchor yaml write_policy must be 'never' ({path})")
    return _sha(path)


def completeness(task_plan_path: str, pool_manifest_path: str, ledger_path: str, journal_path: str,
                 per_step_path: str, anchor_yaml_path: str) -> dict:
    """Outcome-blind certification of a pilot run; reads no terminal status."""
    plan, plan_sha = load_task_plan(task_plan_path)
    if plan["pool_id"] != PILOT_POOL or plan["N"] != PILOT_TRIALS or plan["roster_arms"] != [ANCHOR_ARM]:
        raise SystemExit("pilot task plan must be the anchor on the whole P pool")
    verify_task_plan_against_pool(plan, pool_manifest_path)
    manifest = load_pool_manifest(pool_manifest_path)
    manifest_sha = _sha(pool_manifest_path)
    if manifest.get("suite") != plan["suite"]:
        raise SystemExit("P manifest suite != pilot task plan suite")
    anchor_sha = validate_anchor_yaml(anchor_yaml_path)
    ledger = json.loads(pathlib.Path(ledger_path).read_text())
    launches = ledger.get("launches") or []
    if ledger.get("schema_version") != 2 or len(launches) != 1:
        raise SystemExit(f"pilot ledger must hold exactly one launch (one-shot), found {len(launches)}")
    launch = launches[0]
    if launch.get("protocol") != PROTOCOL_PILOT or launch.get("layer") != LAYER_PILOT or launch.get("suite") != plan["suite"]:
        raise SystemExit("pilot launch protocol / layer / suite mismatch")
    if launch.get("executed_arms") != [ANCHOR_ARM] or launch.get("core_arms") != [ANCHOR_ARM] or launch.get("descriptive_arms") != []:
        raise SystemExit("pilot launch must execute exactly the anchor")
    if launch.get("task_plan_sha256") != plan_sha or launch.get("arm_matrix_sha256") != plan_sha:
        raise SystemExit("pilot launch is not bound to this P task plan")
    if launch.get("trials_per_task") != PILOT_TRIALS or launch.get("N") != PILOT_TRIALS:
        raise SystemExit("pilot launch trials != the P quota")
    if launch.get("frozen_yaml_sha256") != {ANCHOR_ARM: anchor_sha} or launch.get("executed_yaml_sha256") != {ANCHOR_ARM: anchor_sha}:
        raise SystemExit("pilot launch executed a different anchor yaml")
    if launch.get("cost_model_digest") != cost_model_digest():
        raise SystemExit("pilot launch cost model digest != the cost authority")
    pool = launch.get("pool") or {}
    if pool.get("pool_id") != PILOT_POOL or pool.get("manifest_sha256") != manifest_sha \
            or pool.get("prefix_n") != PILOT_TRIALS or pool.get("total_inits") != NUM_TASKS * PILOT_TRIALS \
            or launch.get("pool_digest") != pool.get("rollup_sha256"):
        raise SystemExit("pilot launch pool attestation != the P manifest")
    # Re-open the exact materialised directory used by the launch. Comparing
    # ledger fields only to one another would allow a copied rollup to attest
    # a different (or missing) P pool.
    from exp.dispatch_surface.run_precheck import validate_pool_files

    materialised = validate_pool_files(
        pool_manifest_path, manifest_sha, PILOT_POOL, pool.get("apool_dir", ""), PILOT_TRIALS
    )
    if materialised != pool:
        raise SystemExit("pilot launch pool attestation cannot be reproduced from its materialised files")
    run_id = launch.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SystemExit("pilot launch lacks a run id")
    for key in ("policy_fingerprint", "contract_binding", "env_seed", "replan_steps"):
        if launch.get(key) is None:
            raise SystemExit(f"pilot launch lacks {key}")
    binding = launch["contract_binding"]
    if not isinstance(binding, dict) or binding.get("policy_fingerprint") != launch["policy_fingerprint"] \
            or binding.get("h_exec") != launch["replan_steps"] or not isinstance(binding.get("servers"), dict):
        raise SystemExit("pilot launch contract binding disagrees with policy_fingerprint / replan_steps")
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(PILOT_TRIALS)}
    accepted = load_accepted_c(journal_path, [ANCHOR_ARM], grid)
    for key, rec in accepted[ANCHOR_ARM].items():
        if rec["run_id"] != run_id:
            raise SystemExit(f"pilot cell {key}: accepted under run {rec['run_id']!r}, not the registered launch")
        ent = plan["entries"].get(rec["task_uid"])
        if ent is None or (ent["task_id"], ent["prefix_idx"]) != key:
            raise SystemExit(f"pilot cell {key}: task_uid not in the P task plan")
    cells, summary = load_cost_cells_c(per_step_path, [ANCHOR_ARM], accepted)
    counts = summary["verdict_counts"][ANCHOR_ARM]
    num = sum(c for c, _n in cells[ANCHOR_ARM].values())
    den = sum(n for _c, n in cells[ANCHOR_ARM].values())
    if {k for k, v in counts.items() if v} != {"MISS"} or not cost_matches(num / den, unit_cost("MISS", None)):
        raise SystemExit("pilot anchor is not all-MISS at the authority cost (A-4 on P)")
    return {"plan": plan, "plan_sha256": plan_sha, "pool_manifest_sha256": manifest_sha, "anchor_yaml_sha256": anchor_sha,
            "launch": launch, "accepted": accepted, "verdict_counts": counts,
            "ledger_sha256": _sha(ledger_path), "journal_sha256": _sha(journal_path), "per_step_sha256": _sha(per_step_path)}


def _outcomes(journal_path: str, accepted: dict) -> dict[tuple[int, int], float]:
    """Terminal status of every accepted pilot episode (the only outcome read here)."""
    want = {(rec["task_uid"], rec["attempt"], rec["run_id"]): key for key, rec in accepted[ANCHOR_ARM].items()}
    out: dict[tuple[int, int], float] = {}
    for line in open(journal_path):
        row = json.loads(line)
        key = want.get((row.get("task_uid"), row.get("attempt"), row.get("run_id")))
        if key is None or row.get("accepted") is not True:
            continue
        if row.get("status") not in ("done", "failed") or not isinstance(row.get("success"), bool) \
                or ((row["status"] == "done") != row["success"]):
            raise SystemExit(f"pilot cell {key}: terminal status/success schema is invalid")
        if key in out:
            raise SystemExit(f"pilot cell {key}: two terminal rows")
        out[key] = float(row["success"])
    if set(out) != set(accepted[ANCHOR_ARM]):
        raise SystemExit("pilot outcomes do not cover the 100 accepted episodes")
    return out


def _record(ctx: dict, succ: dict, paths: dict) -> dict:
    suite = ctx["plan"]["suite"]
    ref = PHASE0_ANCHOR_SR.get(suite)
    if ref is None:
        raise SystemExit(f"no Phase 0 anchor reference SR for suite {suite!r}")
    n = len(succ)
    k = int(round(sum(succ.values())))
    sr = k / n
    launch = ctx["launch"]
    return {
        "protocol": PROTOCOL_PILOT, "suite": suite, "arm": ANCHOR_ARM, "pool_id": PILOT_POOL, "attempt": 1, "one_shot": True,
        "n_episodes": n, "successes": k, "sr": sr, "reference_sr": ref, "tolerance_pt": PILOT_TOLERANCE_PT,
        "passed": bool(abs(sr - ref) * 100.0 <= PILOT_TOLERANCE_PT),
        "run_id": launch["run_id"], "policy_fingerprint": launch["policy_fingerprint"], "env_seed": launch["env_seed"],
        "replan_steps": launch["replan_steps"], "verdict_counts": ctx["verdict_counts"],
        "inputs": {role: {"path": str(pathlib.Path(paths[role]).resolve()), "sha256": _sha(paths[role])} for role in INPUT_ROLES},
    }


def finalize(task_plan: str, pool_manifest: str, ledger: str, journal: str, per_step: str, anchor_yaml: str, out: str) -> dict:
    paths = {"task_plan": task_plan, "pool_manifest": pool_manifest, "ledger": ledger, "journal": journal,
             "per_step": per_step, "anchor_yaml": anchor_yaml}
    ctx = completeness(task_plan, pool_manifest, ledger, journal, per_step, anchor_yaml)
    succ = _outcomes(journal, ctx["accepted"])
    rec = _record(ctx, succ, paths)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    if not rec["passed"]:
        raise SystemExit(f"generator_validation_failed: pilot SR {rec['sr']:.4f} vs reference {rec['reference_sr']:.4f} "
                         f"exceeds {PILOT_TOLERANCE_PT} pt; the pilot is one-shot and is not retried")
    return rec


def validate_pilot(record_path: str, *, suite: str, pool_manifest_p_path: str, anchor_yaml_sha256: str) -> dict:
    """Re-derive the pilot record from its referenced files (never trust ``sr``)."""
    rec = json.loads(pathlib.Path(record_path).read_text())
    if not isinstance(rec, dict) or rec.get("protocol") != PROTOCOL_PILOT or tuple(sorted(rec)) != tuple(sorted(RECORD_KEYS)):
        raise SystemExit("not a pilot record (protocol / key set)")
    if rec["suite"] != suite or rec["arm"] != ANCHOR_ARM or rec["pool_id"] != PILOT_POOL:
        raise SystemExit("pilot record suite / arm / pool mismatch")
    if rec["attempt"] != 1 or rec["one_shot"] is not True or rec["passed"] is not True:
        raise SystemExit("pilot record is not a passing one-shot attempt")
    inputs = rec["inputs"]
    if set(inputs) != set(INPUT_ROLES):
        raise SystemExit("pilot record input roles are not exact")
    paths = {}
    for role in INPUT_ROLES:
        path = inputs[role]["path"]
        if not pathlib.Path(path).is_file() or _sha(path) != inputs[role]["sha256"]:
            raise SystemExit(f"pilot input {role} missing or drifted since the record was written")
        paths[role] = path
    if _sha(pool_manifest_p_path) != inputs["pool_manifest"]["sha256"]:
        raise SystemExit("pilot record binds a different P manifest than the one being sealed")
    if inputs["anchor_yaml"]["sha256"] != anchor_yaml_sha256:
        raise SystemExit("pilot anchor yaml != the C roster's anchor yaml")
    ctx = completeness(paths["task_plan"], paths["pool_manifest"], paths["ledger"], paths["journal"], paths["per_step"], paths["anchor_yaml"])
    succ = _outcomes(paths["journal"], ctx["accepted"])
    fresh = _record(ctx, succ, paths)
    if fresh != rec:
        diff = sorted(k for k in RECORD_KEYS if fresh.get(k) != rec.get(k))
        raise SystemExit(f"pilot record cannot be reproduced from its inputs (fields {diff})")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-plan", required=True)
    ap.add_argument("--pool-manifest", required=True, help="pool_manifest_P.json")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True)
    ap.add_argument("--anchor-yaml", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = finalize(args.task_plan, args.pool_manifest, args.ledger, args.journal, args.per_step, args.anchor_yaml, args.out)
    print(json.dumps({k: rec[k] for k in ("sr", "reference_sr", "passed", "n_episodes")}, indent=2))


if __name__ == "__main__":
    main()
