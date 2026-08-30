"""Cost-only row loaders for the fresh-init confirmation (plan 3.6-b).

Fresh C episodes are identified by ``task_uid`` (arm, task, prefix); the
official 0..49 semantics of ``orig_init_state_idx`` never apply, so the
Phase 0 loaders (which cross-check the official index) cannot be reused.
These loaders join journal and per-step rows on the accepted
``(task_uid, attempt, run_id)`` triple, bill only accepted episodes, require
``orig_init_state_idx`` to be null and ``subset_init_state_idx`` to equal the
prefix index, and never read an outcome field.
"""

from __future__ import annotations

import json

from exp.dispatch_surface.analysis.analytic_cost import unit_cost
from exp.dispatch_surface.analysis.precheck_io import _is_json_int, parse_task_uid

VERDICTS = ("FULL_HIT", "WARM_START", "MISS")
WARM_START_T = 0.3


def load_accepted_c(journal_path: str, arms: list[str], grid: set[tuple[int, int]]) -> dict[str, dict]:
    """arm -> {(task, prefix): {task_uid, attempt, run_id}} from accepted eval rows only."""
    out: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in out:
            raise SystemExit(f"journal carries a row for arm {arm!r} outside the sealed roster")
        if not isinstance(row.get("accepted"), bool):
            raise SystemExit(f"arm {arm}: journal accepted must be a boolean")
        if row.get("accepted") is not True:
            continue
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm}: accepted journal row is not eval phase")
        uid_arm, task, prefix = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(f"journal row yaml_id={arm} disagrees with its task_uid {row['task_uid']}")
        key = (task, prefix)
        if key not in grid:
            raise SystemExit(f"arm {arm}: accepted episode {key} is outside the sealed C grid")
        if key in out[arm]:
            raise SystemExit(f"arm {arm}: two accepted records for cell {key}")
        if (not _is_json_int(row.get("attempt")) or row["attempt"] < 1
                or not isinstance(row.get("run_id"), str) or not row["run_id"]):
            raise SystemExit(f"arm {arm} cell {key}: accepted record lacks attempt/run_id")
        out[arm][key] = {"task_uid": row["task_uid"], "attempt": row["attempt"], "run_id": row["run_id"]}
    for a in arms:
        if set(out[a]) != grid:
            missing = sorted(grid - set(out[a]))[:3]
            raise SystemExit(f"arm {a}: {len(grid - set(out[a]))} cells have no accepted episode (e.g. {missing})")
    return out


def load_cost_cells_c(per_step_path: str, arms: list[str], accepted: dict[str, dict]) -> tuple[dict, dict]:
    """arm -> {(task, prefix): (cost_sum_ms, n_decisions)}; strict join and identity checks."""
    keyset = {}
    for a in arms:
        for key, rec in accepted[a].items():
            keyset[(a, rec["task_uid"], rec["attempt"], rec["run_id"])] = key
    cost_sum = {a: {} for a in arms}
    n_dec = {a: {} for a in arms}
    infers = {a: {} for a in arms}
    verdicts = {a: {v: 0 for v in VERDICTS} for a in arms}
    excluded = 0
    for line in open(per_step_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in accepted:
            raise SystemExit(f"per-step row for arm {arm!r} outside the sealed roster")
        jk = (arm, row.get("task_uid"), row.get("attempt"), row.get("run_id"))
        key = keyset.get(jk)
        if key is None:
            excluded += 1
            continue
        if row.get("_kind") == "client_timing":
            if key in infers[arm]:
                raise SystemExit(f"arm {arm} cell {key}: duplicate client_timing row")
            inf = row.get("infers")
            if not _is_json_int(inf) or inf < 0:
                raise SystemExit(f"arm {arm} cell {key}: infers must be a nonnegative integer")
            infers[arm][key] = inf
            continue
        if row.get("orig_init_state_idx") is not None:
            raise SystemExit(f"arm {arm} cell {key}: fresh C rows must carry orig_init_state_idx = null")
        if row.get("subset_init_state_idx") != key[1] or row.get("task_id") != key[0]:
            raise SystemExit(f"arm {arm} cell {key}: row identity disagrees with its task_uid")
        hit = row.get("hit_type")
        if hit not in VERDICTS:
            raise SystemExit(f"arm {arm} cell {key}: unknown verdict {hit!r}")
        st = row.get("start_t")
        if hit == "WARM_START":
            if st != WARM_START_T:
                raise SystemExit(f"arm {arm} cell {key}: WARM_START start_t must be {WARM_START_T}")
        elif st is not None:
            raise SystemExit(f"arm {arm} cell {key}: {hit} rows must carry start_t = null")
        cost_sum[arm][key] = cost_sum[arm].get(key, 0.0) + unit_cost(hit, st)
        n_dec[arm][key] = n_dec[arm].get(key, 0) + 1
        verdicts[arm][hit] += 1
    cells = {a: {} for a in arms}
    for a in arms:
        for key in accepted[a]:
            if key not in infers[a]:
                raise SystemExit(f"arm {a} cell {key}: accepted episode has no client_timing row")
            n = n_dec[a].get(key, 0)
            if n != infers[a][key]:
                raise SystemExit(f"arm {a} cell {key}: {n} decision rows != client infers {infers[a][key]}")
            if n <= 0:
                raise SystemExit(f"arm {a} cell {key}: accepted episode has no decisions")
            cells[a][key] = (cost_sum[a][key], n)
    return cells, {"verdict_counts": verdicts, "excluded_rows": excluded}
