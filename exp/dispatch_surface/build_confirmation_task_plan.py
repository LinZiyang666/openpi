"""Confirmation task plan: the authoritative episode identity for a fresh
pool (confirmation plan 3.6-b, G1R3 fix 1, G2R1-B6).

Built BEFORE the seal from the frozen roster, N and the pool manifest, it
maps every ``task_uid = make_task_uid(arm, "eval", task_id, prefix_idx)`` to
``{arm, task_id, task_name, prefix_idx, pool_id, fresh_state_sha256}``. It
contains no seal digest (the seal binds the plan's SHA, never the other way
round), so there is no hash cycle. The producer rows keep their fixed
schema; the outcome-blind discipline joins them to this plan by ``task_uid``.

``validate_task_plan`` is the ONE structural validator (used at build time,
by the loader, by the seal, the runner and the discipline): the entry key
set must be exactly ``roster_arms x task_id(0..9) x prefix(0..N-1)``, every
entry self-consistent, one task name per task id, and no seal reference
anywhere in the document. ``verify_task_plan_against_pool`` then binds each
entry to the pool manifest's ORIGINAL ``k``-ordered entries (``prefix_idx ==
k``, ``status == ok``, same digest) and the manifest's task-id <-> name map.

The same plan format serves the C confirmation (``pool_id = "C"``) and the P
pilot (``pool_id = "P"``, anchor only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

from openpi.conductor.task import make_task_uid

from exp.dispatch_surface.generate_fresh_inits import POOL_QUOTA, load_pool_manifest
from exp.dispatch_surface.run_precheck import NUM_TASKS

PROTOCOL = "dispatch_surface_rev2_confirmation_task_plan"
FORBIDDEN_KEYS = ("seal_sha256", "seal")
ENTRY_KEYS = ("arm", "task_id", "task_name", "prefix_idx", "pool_id", "fresh_state_sha256")
PLAN_KEYS = ("schema", "protocol", "suite", "N", "pool_id", "pool_manifest_sha256", "roster_arms", "entries")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def assert_no_cycle(plan, _path: str = "plan") -> None:
    """Recursively refuse any seal reference (the seal binds the plan, never vice versa)."""
    if isinstance(plan, dict):
        for k, v in plan.items():
            if k in FORBIDDEN_KEYS:
                raise SystemExit(f"task plan must not reference the seal ({_path}.{k}); the seal binds the plan, not vice versa")
            assert_no_cycle(v, f"{_path}.{k}")
    elif isinstance(plan, list):
        for i, v in enumerate(plan):
            assert_no_cycle(v, f"{_path}[{i}]")


def _pool_tasks(manifest: dict, pool_id: str) -> list[tuple[str, dict]]:
    """The pool's tasks sorted by task id, after checking the 10 unique ids / names."""
    pool = (manifest.get("pools") or {}).get(pool_id)
    if pool is None:
        raise SystemExit(f"pool manifest lacks the {pool_id} pool")
    tasks = pool.get("tasks") or {}
    if len(tasks) != NUM_TASKS:
        raise SystemExit(f"{pool_id} pool holds {len(tasks)} tasks, expected {NUM_TASKS}")
    by_id = sorted(tasks.items(), key=lambda kv: int(kv[1]["task_id"]))
    if [int(info["task_id"]) for _n, info in by_id] != list(range(NUM_TASKS)):
        raise SystemExit(f"{pool_id} pool task ids are not exactly 0..{NUM_TASKS - 1}")
    if len({name for name, _i in by_id}) != NUM_TASKS:
        raise SystemExit(f"{pool_id} pool task names are not unique")
    return by_id


def validate_task_plan(plan: dict) -> None:
    """Exact Cartesian structure of a task plan (no pool file needed)."""
    if not isinstance(plan, dict) or tuple(sorted(plan)) != tuple(sorted(PLAN_KEYS)):
        raise SystemExit("task plan key set is not exact")
    if plan["protocol"] != PROTOCOL or plan["schema"] != 1:
        raise SystemExit("not a confirmation task plan")
    assert_no_cycle(plan)
    pool_id = plan["pool_id"]
    if pool_id not in POOL_QUOTA:
        raise SystemExit(f"task plan pool_id {pool_id!r} is not a fresh pool")
    n = plan["N"]
    if not _is_int(n) or not (1 <= n <= POOL_QUOTA[pool_id]):
        raise SystemExit(f"task plan N={n!r} outside 1..{POOL_QUOTA[pool_id]} for pool {pool_id}")
    arms = plan["roster_arms"]
    if not isinstance(arms, list) or not arms or any(not isinstance(a, str) or not a for a in arms) \
            or len(set(arms)) != len(arms) or arms != sorted(arms):
        raise SystemExit("task plan roster_arms must be a sorted, non-empty list of unique arm ids")
    if not isinstance(plan["pool_manifest_sha256"], str) or not _SHA_RE.match(plan["pool_manifest_sha256"]):
        raise SystemExit("task plan pool_manifest_sha256 malformed")
    entries = plan["entries"]
    if not isinstance(entries, dict):
        raise SystemExit("task plan entries must be an object keyed by task_uid")
    expected = {make_task_uid(arm, "eval", t, p) for arm in arms for t in range(NUM_TASKS) for p in range(n)}
    if set(entries) != expected:
        extra = sorted(set(entries) - expected)[:3]
        missing = sorted(expected - set(entries))[:3]
        raise SystemExit(f"task plan entries != roster x {NUM_TASKS} tasks x prefix 0..{n - 1} (extra {extra}, missing {missing})")
    names: dict[int, str] = {}
    digests: dict[tuple[int, int], str] = {}
    for uid, ent in entries.items():
        if not isinstance(ent, dict) or tuple(sorted(ent)) != tuple(sorted(ENTRY_KEYS)):
            raise SystemExit(f"task plan entry {uid} key set is not exact")
        if ent["arm"] not in arms or not _is_int(ent["task_id"]) or not _is_int(ent["prefix_idx"]):
            raise SystemExit(f"task plan entry {uid} has an arm outside the roster or non-integer indices")
        if uid != make_task_uid(ent["arm"], "eval", ent["task_id"], ent["prefix_idx"]):
            raise SystemExit(f"task plan entry {uid} is mislabelled")
        if ent["pool_id"] != pool_id or not (0 <= ent["prefix_idx"] < n) or not (0 <= ent["task_id"] < NUM_TASKS):
            raise SystemExit(f"task plan entry {uid} has an invalid pool/prefix/task")
        if not isinstance(ent["task_name"], str) or not ent["task_name"]:
            raise SystemExit(f"task plan entry {uid} lacks a task name")
        if names.setdefault(ent["task_id"], ent["task_name"]) != ent["task_name"]:
            raise SystemExit(f"task plan maps task {ent['task_id']} to two names")
        if not isinstance(ent["fresh_state_sha256"], str) or not _SHA_RE.match(ent["fresh_state_sha256"]):
            raise SystemExit(f"task plan entry {uid} digest malformed")
        key = (ent["task_id"], ent["prefix_idx"])
        if digests.setdefault(key, ent["fresh_state_sha256"]) != ent["fresh_state_sha256"]:
            raise SystemExit(f"task plan maps cell {key} to two different fresh states")
    if len(set(names.values())) != NUM_TASKS:
        raise SystemExit("task plan task names are not unique across the 10 tasks")


def build_task_plan(suite: str, roster_arms: list[str], n_prefix: int, pool_manifest_path: str,
                    pool_id: str = "C") -> dict:
    manifest = load_pool_manifest(pool_manifest_path)
    if manifest["suite"] != suite:
        raise SystemExit("pool manifest suite != requested suite")
    if pool_id not in POOL_QUOTA:
        raise SystemExit(f"unknown fresh pool {pool_id!r}")
    quota = POOL_QUOTA[pool_id]
    if not _is_int(n_prefix) or not (1 <= n_prefix <= quota):
        raise SystemExit(f"N={n_prefix!r} outside 1..{quota} for pool {pool_id}")
    arms = sorted(roster_arms)
    if not arms or len(set(arms)) != len(arms):
        raise SystemExit("roster arms must be non-empty and unique")
    by_id = _pool_tasks(manifest, pool_id)
    entries = {}
    for arm in arms:
        for name, info in by_id:
            raw = info["entries"]
            if len(raw) != quota or [e.get("k") for e in raw] != list(range(quota)):
                raise SystemExit(f"{name}: {pool_id} pool entries are not k = 0..{quota - 1} in order")
            for prefix_idx in range(n_prefix):
                e = raw[prefix_idx]
                if e.get("status") != "ok":
                    raise SystemExit(f"{name}: {pool_id} pool k={prefix_idx} is {e.get('status')!r}, not ok")
                uid = make_task_uid(arm, "eval", int(info["task_id"]), prefix_idx)
                entries[uid] = {"arm": arm, "task_id": int(info["task_id"]), "task_name": name,
                                "prefix_idx": prefix_idx, "pool_id": pool_id, "fresh_state_sha256": e["state_sha256"]}
    plan = {"schema": 1, "protocol": PROTOCOL, "suite": suite, "N": n_prefix, "pool_id": pool_id,
            "pool_manifest_sha256": _file_sha256(pathlib.Path(pool_manifest_path)),
            "roster_arms": arms, "entries": entries}
    validate_task_plan(plan)
    verify_task_plan_against_pool(plan, pool_manifest_path)
    return plan


def load_task_plan(path) -> tuple[dict, str]:
    p = pathlib.Path(path)
    plan = json.loads(p.read_text())
    if not isinstance(plan, dict) or plan.get("protocol") != PROTOCOL or plan.get("schema") != 1:
        raise SystemExit(f"{p}: not a confirmation task plan")
    validate_task_plan(plan)
    return plan, _file_sha256(p)


def verify_task_plan_against_pool(plan: dict, pool_manifest_path: str) -> None:
    """Bind every entry to the manifest's original k-ordered entries."""
    validate_task_plan(plan)
    manifest = load_pool_manifest(pool_manifest_path)
    if plan["pool_manifest_sha256"] != _file_sha256(pathlib.Path(pool_manifest_path)):
        raise SystemExit("task plan binds a different pool manifest")
    if manifest.get("suite") != plan["suite"]:
        raise SystemExit("task plan suite != pool manifest suite")
    pool_id = plan["pool_id"]
    by_id = _pool_tasks(manifest, pool_id)
    id_to_name = {int(info["task_id"]): name for name, info in by_id}
    tasks = manifest["pools"][pool_id]["tasks"]
    quota = POOL_QUOTA[pool_id]
    for uid, ent in plan["entries"].items():
        if id_to_name.get(ent["task_id"]) != ent["task_name"]:
            raise SystemExit(f"task plan entry {uid}: task id/name != pool manifest")
        raw = tasks[ent["task_name"]]["entries"]
        if len(raw) != quota or [e.get("k") for e in raw] != list(range(quota)):
            raise SystemExit(f"{ent['task_name']}: pool entries are not k = 0..{quota - 1} in order")
        e = raw[ent["prefix_idx"]]
        if e.get("k") != ent["prefix_idx"] or e.get("status") != "ok":
            raise SystemExit(f"task plan entry {uid}: pool k={ent['prefix_idx']} is not an ok state")
        if e.get("state_sha256") != ent["fresh_state_sha256"]:
            raise SystemExit(f"task plan entry {uid} digest != pool manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--pool-id", default="C", choices=sorted(POOL_QUOTA))
    ap.add_argument("--c-roster", default="", help="c_roster.json from budget_outcome_design (C plans)")
    ap.add_argument("--arms", default="", help="comma list of arms (P pilot: the anchor only)")
    ap.add_argument("--n", type=int, required=True, help="N per task (C: from the power record; P: 10)")
    ap.add_argument("--pool-manifest", required=True, help="pool_manifest_<pool>.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.c_roster:
        roster = json.loads(pathlib.Path(args.c_roster).read_text())
        arms = [e["arm"] for e in roster["arms"]]
    elif args.arms:
        arms = args.arms.split(",")
    else:
        raise SystemExit("give --c-roster or --arms")
    plan = build_task_plan(args.suite, arms, args.n, args.pool_manifest, pool_id=args.pool_id)
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(f"task plan: {len(plan['entries'])} entries -> {p} sha256={_file_sha256(p)}")


if __name__ == "__main__":
    main()
