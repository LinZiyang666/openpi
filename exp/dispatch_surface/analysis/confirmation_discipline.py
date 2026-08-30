"""Outcome-blind completeness certification for fresh C, required before
unsealing (confirmation plan 3.6-b, G1R1-B2 / G1R3 fix 1).

Proves, without reading any outcome field, that the observed C grid is the
exact Cartesian product roster x 10 tasks x prefix 0..N-1 with exactly one
accepted episode per cell claimed by a registered launch; that every row's
identity resolves through ``task_uid`` to the frozen task plan and through
``run_id`` to a ledger entry bound to this seal and this task plan; that the
plan's fresh-state digests are the sealed C manifest's; and that every
accepted episode's per-step decisions agree with the client's ``infers``.
Its own SHA enters the unseal record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from exp.dispatch_surface import seal_confirmation
from exp.dispatch_surface.analysis.confirmation_io import load_accepted_c, load_cost_cells_c
from exp.dispatch_surface.build_confirmation_task_plan import load_task_plan, verify_task_plan_against_pool
from exp.dispatch_surface.run_precheck import CONFIRMATION_FROZEN_LAUNCH_KEYS, NUM_TASKS

PROTOCOL = "dispatch_surface_rev2_confirmation_discipline"


def _sha(p: str) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def certify(seal_path: str, task_plan_path: str, ledger_path: str, journal_path: str, per_step_path: str) -> dict:
    seal, seal_sha = seal_confirmation.load_seal(seal_path)
    plan, plan_sha = load_task_plan(task_plan_path)
    if plan_sha != seal["confirmation_task_plan_sha256"]:
        raise SystemExit("task plan bytes != the seal's task-plan digest")
    verify_task_plan_against_pool(plan, seal["pool"]["manifest_path"])
    if _sha(seal["pool"]["manifest_path"]) != seal["pool"]["manifest_sha256"]:
        raise SystemExit("C pool manifest drifted since sealing")
    n = int(seal["N"])
    arms = list(seal["roster"]["arms"])
    if plan["N"] != n or sorted(plan["roster_arms"]) != sorted(arms):
        raise SystemExit("task plan roster/N != seal")
    ledger = json.loads(pathlib.Path(ledger_path).read_text())
    launches = ledger.get("launches") or []
    if ledger.get("schema_version") != 2 or not launches:
        raise SystemExit("confirmation ledger is not a v2 ledger with launches")
    executed: dict[str, set[str]] = {}
    run_ids = []
    for idx, launch in enumerate(launches):
        if launch.get("seal_sha256") != seal_sha or launch.get("confirmation_task_plan_sha256") != plan_sha:
            raise SystemExit(f"launch {idx}: seal / task-plan digests differ from the current inputs")
        if launch.get("arm_matrix_sha256") != seal_sha or launch.get("N") != n:
            raise SystemExit(f"launch {idx}: not bound to this seal / N")
        for key in CONFIRMATION_FROZEN_LAUNCH_KEYS:
            if launch.get(key) != launches[0].get(key):
                raise SystemExit(f"launch {idx} drifts on frozen key {key}")
        if launch.get("frozen_yaml_sha256") != seal["roster"]["yaml_sha256"]:
            raise SystemExit(f"launch {idx}: frozen yaml digests != seal")
        if launch.get("estimator_version") != seal["estimator_digest"] or launch.get("cost_model_digest") != seal["cost_model_digest"]:
            raise SystemExit(f"launch {idx}: estimator / cost authority digests != seal")
        ex = launch.get("executed_arms") or []
        if not ex or not set(ex).issubset(arms):
            raise SystemExit(f"launch {idx}: executed arms outside the sealed roster")
        for arm in ex:
            if (launch.get("executed_yaml_sha256") or {}).get(arm) != seal["roster"]["yaml_sha256"][arm]:
                raise SystemExit(f"launch {idx}: executed a different yaml for {arm}")
        pool = launch.get("pool") or {}
        if pool.get("pool_id") != "C" or pool.get("manifest_sha256") != seal["pool"]["manifest_sha256"] \
                or pool.get("prefix_n") != n or pool.get("total_inits") != NUM_TASKS * n:
            raise SystemExit(f"launch {idx}: fresh pool attestation != seal")
        rid = str(launch.get("run_id"))
        if not rid or rid in executed:
            raise SystemExit(f"launch {idx}: missing or duplicated run id")
        executed[rid] = set(ex)
        run_ids.append(rid)
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(n)}
    accepted = load_accepted_c(journal_path, arms, grid)
    for arm in arms:
        for key, rec in accepted[arm].items():
            if rec["run_id"] not in executed or arm not in executed[rec["run_id"]]:
                raise SystemExit(f"arm {arm} cell {key}: accepted under an unregistered run {rec['run_id']!r}")
            ent = plan["entries"].get(rec["task_uid"])
            if ent is None or ent["arm"] != arm or (ent["task_id"], ent["prefix_idx"]) != key:
                raise SystemExit(f"arm {arm} cell {key}: task_uid not in the frozen task plan")
    # every (task, prefix) resolves to ONE fresh digest across arms
    digest_by_cell: dict[tuple[int, int], str] = {}
    for uid, ent in plan["entries"].items():
        key = (ent["task_id"], ent["prefix_idx"])
        d = digest_by_cell.setdefault(key, ent["fresh_state_sha256"])
        if d != ent["fresh_state_sha256"]:
            raise SystemExit(f"task plan maps cell {key} to two different fresh states")
    cells, summary = load_cost_cells_c(per_step_path, arms, accepted)
    executed_all = set().union(*executed.values())
    out = {
        "protocol": PROTOCOL, "passed": True, "suite": seal["suite"], "N": n, "arms": arms,
        "seal_sha256": seal_sha, "task_plan_sha256": plan_sha,
        "ledger_sha256": _sha(ledger_path), "journal_sha256": _sha(journal_path), "per_step_sha256": _sha(per_step_path),
        "pool_manifest_sha256": seal["pool"]["manifest_sha256"],
        "run_ids": run_ids, "executed_arms_by_run": {k: sorted(v) for k, v in executed.items()},
        "roster_complete": executed_all == set(arms),
        "cells_per_arm": {a: len(accepted[a]) for a in arms},
        "verdict_counts": summary["verdict_counts"], "excluded_rows": summary["excluded_rows"],
        "estimator_digest": seal["estimator_digest"], "cost_model_digest": seal["cost_model_digest"],
    }
    if not out["roster_complete"]:
        raise SystemExit("ledger did not execute the whole sealed roster")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--task-plan", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = certify(args.seal, args.task_plan, args.ledger, args.journal, args.per_step)
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"confirmation discipline passed -> {p} sha256={_sha(str(p))}")


if __name__ == "__main__":
    main()
