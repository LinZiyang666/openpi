"""Strict, task-balanced MT50 aggregation; errors never count as failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from exp.metaworld import tasks as T
from exp.mw_finetune.journal import read_journal, write_json_atomic


def aggregate(rows, names=None, trials=10, require_complete=True):
    names = list(T.TASK_NAMES if names is None else names)
    if not names or len(set(names)) != len(names) or not set(names).issubset(T.TASK_NAMES) or not 1 <= trials <= 50:
        raise ValueError("invalid task/trial selection")
    expected = {(name, idx) for name in names for idx in range(trials)}
    complete = {}
    errors = []
    for row in rows:
        if type(row.get("idx")) is not int:
            raise ValueError("episode idx must be an integer")
        key = (row["task"], row["idx"])
        if key not in expected or row.get("seed") != 7:
            raise ValueError(f"unexpected episode identity {key}, seed={row.get('seed')}")
        if row.get("error"):
            errors.append(row)
            continue
        if type(row.get("success")) is not bool:
            raise ValueError("success must be a JSON boolean")
        if key in complete:
            raise ValueError(f"duplicate completed episode {key}")
        complete[key] = row
    missing = sorted(expected - complete.keys())
    if missing and require_complete:
        raise ValueError(f"incomplete: {len(complete)}/{len(expected)} episodes; {len(errors)} errors")
    per_task = {}
    for name in names:
        selected = [complete[(name, i)] for i in range(trials) if (name, i) in complete]
        n = len(selected)
        per_task[name] = {"n": n, "successes": sum(r["success"] for r in selected),
                          "sr": sum(r["success"] for r in selected) / n if n else None,
                          "difficulty": T.DIFFICULTY[name]}
    groups = {}
    for group in T.DIFFICULTY_GROUPS:
        members = [v for v in per_task.values() if v["difficulty"] == group]
        values = [v["sr"] for v in members if v["sr"] is not None]
        groups[group] = {"tasks": len(members), "episodes": sum(v["n"] for v in members),
                         "sr": sum(values) / len(values) if values else None}
    values = [v["sr"] for v in per_task.values() if v["sr"] is not None]
    group_values = [v["sr"] for v in groups.values() if v["sr"] is not None]
    formal = not missing and set(names) == set(T.TASK_NAMES) and trials == 10
    return {"complete": not missing, "completed_episodes": len(complete), "expected_episodes": len(expected),
            "missing": missing, "error_attempts": len(errors), "per_task": per_task, "per_group": groups,
            "overall_per_task_mean": sum(values) / len(values) if values else None,
            "equal_group_mean": sum(group_values) / len(group_values) if group_values else None,
            "formal_mt50": formal,
            "rate_scope": "MT50 500-episode protocol" if formal else
                ("complete selection; nonstandard MT50 protocol" if not missing else "observed completed episodes only; incomplete"),
            "units": "fraction; multiply by 100 for percent"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    manifest = json.loads((a.run / "manifest.json").read_text())
    rows = read_journal(a.run / "episodes.jsonl")
    result = aggregate(rows, manifest["tasks"], manifest["trials"], not a.allow_partial)
    if manifest.get("server_metadata", {}).get("test_policy"):
        result.update(formal_mt50=False, rate_scope="test policy capacity/smoke only; not a model success rate")
    result["provenance"] = manifest
    invocation_file = a.run / "invocations.jsonl"
    if invocation_file.exists():
        lines = invocation_file.read_text().splitlines()
        if lines:
            try:
                result["last_invocation"] = json.loads(lines[-1])
            except json.JSONDecodeError:
                pass  # An interrupted diagnostic append does not alter episode SR.
    write_json_atomic(a.run / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_task", "missing", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
