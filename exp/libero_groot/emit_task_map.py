"""Write the suite's ``task_id -> (name, language)`` map from LIBERO itself (plan §3.9, G2-B1).

Two different strings identify a LIBERO task and both appear in the cohort
evidence: ``task.name`` (the canonical, underscore-joined name that locates the
``.init`` pool file, and what the shadow manifest calls ``task_name``) and
``task.language`` (the human instruction the client sends on ``episode_start``
and the collector stores as the H5 ``task`` attr -- the string the model was
conditioned on). Neither is derived from the other here: both are read off the
benchmark object the client uses, on a machine where LIBERO is installed, and
the verifier binds the manifest, the client rows and the H5 files through
``task_id`` with this file as the authority.

Usage (LIBERO client environment):
  python -m exp.libero_groot.emit_task_map --suite libero_spatial --out <task_map.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib

PROTOCOL = "actioncache_baseline/task_map/v1"
TASKS_PER_SUITE = 10


def task_map_from_benchmark(suite: str) -> dict:
    """``{"protocol", "suite", "tasks": [{"task_id", "name", "language", "bddl_file"}]}`` via LIBERO."""
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite]()
    tasks = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        tasks.append({"task_id": int(task_id), "name": str(task.name), "language": str(task.language),
                      "bddl_file": str(task.bddl_file), "problem_folder": str(task.problem_folder)})
    return {"protocol": PROTOCOL, "suite": suite, "tasks": tasks}


def validate_task_map(doc: dict, suite: str) -> dict[int, dict]:
    """Return ``{task_id: {"name", "language"}}``; fail on any hole, duplicate or empty string."""
    if doc.get("protocol") != PROTOCOL or doc.get("suite") != suite:
        raise SystemExit(f"task map is not a {PROTOCOL} record for {suite!r}")
    tasks = doc.get("tasks") or []
    by_id: dict[int, dict] = {}
    for row in tasks:
        t = int(row["task_id"])
        name, language = str(row.get("name", "")), str(row.get("language", ""))
        if t in by_id or not name or not language:
            raise SystemExit(f"task map: bad or duplicate row for task {t}: {row}")
        by_id[t] = {"name": name, "language": language}
    if sorted(by_id) != list(range(TASKS_PER_SUITE)):
        raise SystemExit(f"task map must name tasks 0..{TASKS_PER_SUITE - 1}, got {sorted(by_id)}")
    names = [v["name"] for v in by_id.values()]
    languages = [v["language"] for v in by_id.values()]
    if len(set(names)) != len(names) or len(set(languages)) != len(languages):
        raise SystemExit("task map: names / languages are not unique across tasks")
    return by_id


def load_task_map(path: str | pathlib.Path, suite: str) -> dict[int, dict]:
    """Read and validate a task map written by this module."""
    return validate_task_map(json.loads(pathlib.Path(path).read_text(encoding="utf-8")), suite)


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=["libero_spatial", "libero_10"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = task_map_from_benchmark(args.suite)
    validate_task_map(doc, args.suite)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(doc['tasks'])} tasks)")


if __name__ == "__main__":
    main()
