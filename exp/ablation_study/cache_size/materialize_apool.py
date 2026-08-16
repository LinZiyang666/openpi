"""Freeze the official A-pool (pruned_init) into versioned ``.init`` artifacts.

The evaluation pool is otherwise an *implicit* fallback: when
``--init_states_dir`` is empty the episode runner calls
``task_suite.get_task_init_states(task_id)``, whose content depends on the
installed LIBERO version and cannot be hashed. Materializing it turns the pool
into a first-class artifact that can be digested, diffed against the difference
pool, and cited by an analysis run.

⚠ Requires the LIBERO client environment. The main uv venv has no ``libero``
module (``third_party/libero`` is an uninitialized submodule), so this is a
separate, explicitly-scoped step -- run it where the simulator lives, then carry
the digest record back.

Writes ``<out_dir>/<task_name>.init`` per task, byte-for-byte what
``get_task_init_states`` returned, then hands off to ``verify_apool.py`` for the
digest and the ``A ∩ B = ∅`` assertion.
"""

from __future__ import annotations

import argparse
import pathlib

import torch

SUITE_TASK_COUNT = 10


def materialize(suite: str, out_dir: pathlib.Path) -> dict[str, int]:
    """Dump each task's official init states verbatim. Returns {task_name: count}."""
    from libero.libero import benchmark  # noqa: PLC0415 -- client env only

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    n_tasks = task_suite.n_tasks
    if n_tasks != SUITE_TASK_COUNT:
        raise SystemExit(f"{suite}: expected {SUITE_TASK_COUNT} tasks, found {n_tasks}")

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        states = task_suite.get_task_init_states(task_id)
        dest = out_dir / f"{task.name}.init"
        # torch.save verbatim: no reshaping, no dtype coercion. Any transform
        # here would make the frozen copy subtly different from what the runner
        # actually loads, defeating the point of freezing it.
        torch.save(states, dest)
        written[task.name] = len(states)
        print(f"  {task.name}: {len(states)} inits -> {dest}")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, help="e.g. libero_spatial / libero_10")
    ap.add_argument("--out-dir", required=True,
                    help="e.g. exp/common/data/db_init/libero/<suite>_apool")
    ap.add_argument("--expect-per-task", type=int, default=50)
    args = ap.parse_args()

    written = materialize(args.suite, pathlib.Path(args.out_dir))

    bad = {k: v for k, v in written.items() if v != args.expect_per_task}
    if bad:
        raise SystemExit(
            f"unexpected init counts (expected {args.expect_per_task} each): {bad}"
        )
    total = sum(written.values())
    print(f"{len(written)} tasks, {total} inits total")
    print("Next: run verify_apool.py to record the digest and assert A ∩ B = ∅")


if __name__ == "__main__":
    main()
