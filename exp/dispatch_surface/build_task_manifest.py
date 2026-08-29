"""Build the git-tracked task manifest of a LIBERO suite (plan section 3.8-a).

For every task in LIBERO benchmark EXECUTION order the manifest records the
task id, name, BDDL path and SHA, and the goal atoms parsed with the ``bddl``
package's own parser (no regex over the source). The task names are checked
both ways against the frozen split manifest's ``assignment``. Runs inside the
LIBERO conda environment (needs ``libero`` + ``bddl``); the output JSON is
small and committed under ``exp/dispatch_surface/config/``.

Usage:
  LIBERO_CONFIG_PATH=... python -m exp.dispatch_surface.build_task_manifest \
      --suite libero_10 --split-manifest <split_manifest.json> --out <task_manifest_libero_10.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dist_version(dist: str, module) -> str:
    try:
        import importlib.metadata as im
        return im.version(dist)
    except Exception:  # noqa: BLE001 - fall back to the module attribute / path
        return getattr(module, "__version__", None) or f"unversioned:{pathlib.Path(module.__file__).parent}"


def _flatten_goal(goal) -> list[dict]:
    """Flatten bddl's packaged goal predicates into [{predicate, args}] in source order."""
    atoms: list[dict] = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], str) and node[0].lower() in ("and", "or", "not"):
                for child in node[1:]:
                    walk(child)
            elif node and isinstance(node[0], str) and all(isinstance(x, str) for x in node):
                atoms.append({"predicate": node[0], "args": list(node[1:])})
            else:
                for child in node:
                    walk(child)

    walk(goal)
    return atoms


def build(suite: str, split_manifest: pathlib.Path) -> dict:
    import bddl
    import libero
    from bddl.parsing import parse_problem
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[suite]()
    assignment = json.loads(split_manifest.read_text())["assignment"]
    tasks = []
    for task_id in range(bench.n_tasks):
        task = bench.get_task(task_id)
        bddl_path = pathlib.Path(benchmark.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl_path.is_file():
            raise SystemExit(f"task {task_id}: BDDL missing at {bddl_path}")
        text = bddl_path.read_text()
        problem_name, objects, _init, goal = parse_problem("libero", task.name, "robosuite", predefined_problem=text)
        atoms = _flatten_goal(goal)
        split_name = assignment[str(task_id)]["task_name"]
        if split_name != task.name:
            raise SystemExit(f"task {task_id}: benchmark name {task.name!r} != split manifest {split_name!r}")
        tasks.append({
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "problem_name": problem_name,
            "bddl_file": task.bddl_file,
            "bddl_path": str(bddl_path),
            "bddl_sha256": _sha(bddl_path),
            "objects": objects,
            "goal_atoms": atoms,
            "n_goal_atoms": len(atoms),
        })
    names_split = {v["task_name"] for v in assignment.values()}
    names_bench = {t["task_name"] for t in tasks}
    if names_split != names_bench:
        raise SystemExit("split manifest and benchmark disagree on the task set")
    if set(assignment) != {str(i) for i in range(bench.n_tasks)}:
        raise SystemExit(f"split manifest assignment keys {sorted(assignment)} are not exactly 0..{bench.n_tasks - 1}")
    return {
        "schema": 1,
        "suite": suite,
        "task_order": "LIBERO benchmark execution order (benchmark.get_task(i))",
        "libero_version": _dist_version("libero", libero),
        "bddl_version": _dist_version("bddl", bddl),
        "split_manifest_sha256": _sha(split_manifest),
        "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=["libero_10", "libero_spatial"])
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = build(args.suite, pathlib.Path(args.split_manifest))
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps([(t["task_id"], t["n_goal_atoms"], t["task_name"][:50]) for t in out["tasks"]], indent=0))


if __name__ == "__main__":
    main()
