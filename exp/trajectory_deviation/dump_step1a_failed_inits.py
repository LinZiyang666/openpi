"""Dump per-task Step-1a failed init-state subsets for Step 1b GT collection.

Consumes the aggregated ``cache_eval_results.json`` produced by
``exp/common/run_cache_experiments.py`` (plan §9.0 改动 2 + §19.B5) and, for every
``task_id`` that has at least one failed episode, writes:

    {out_dir}/{task.name}.init
        torch-saved tensor of shape ``(K, 92)`` —
        the subset of the benchmark's init states whose indices are the
        failed original ``init_state_idx`` values, in ascending order. This
        is the exact layout ``examples/libero/main.py::_load_init_states``
        reads via ``--init-states-dir`` (plan §9.1).

    {out_dir}/{task.name}.init_map.json
        List ``[orig_0, orig_1, ...]`` — index ``i`` is the original
        benchmark index of the init at position ``i`` in the ``.init``
        subset. ``main.py`` uses this to populate the ``orig_init_state_idx``
        HDF5 attr (plan §19.B6).

    {out_dir}/step1b_filter.json
        ``[{task_id, orig_init_state_idx, subset_init_state_idx}, ...]``
        — fed to ``main.py --episode-filter`` so Step 1b only runs the
        failed episodes even when ``--num-trials-per-task`` is set high
        enough to cover the whole subset (plan §19.B6).

Usage
-----
.. code-block:: bash

    python exp/trajectory_deviation/dump_step1a_failed_inits.py \\
        --step1a-results exp/trajectory_deviation/data/cache_eval_results.json \\
        --task-suite libero_spatial \\
        --out-dir data/step1b_inits/libero_spatial

The ``libero`` benchmark import only happens when writing ``.init`` files,
so the helper functions below can be unit-tested in the plain venv with
``--no-torch`` (an internal CI flag used by
``tests/scripts/test_dump_step1a_failed_inits.py``).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _group_failed_by_task(rows: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """Group failed ``init_state_idx`` values per ``task_id``.

    ``rows`` is the (already-deduped) content of ``cache_eval_results.json``
    written by ``exp/common/run_cache_experiments.py::_aggregate_episode_results``
    (plan §19.B5). Retry-recovered episodes must have been flipped to
    ``success=True`` by the aggregate dedup, so we can filter on ``success``
    alone here.

    We use ``orig_init_state_idx`` when present (Step 1a runs pass no filter
    so it equals ``init_state_idx``, but this preserves invariants if the
    aggregator is fed a filtered run). Duplicates within a task are de-duped
    because a single (task, init) must only appear once in the subset file.
    """
    failed_by_task: Dict[int, set] = defaultdict(set)
    for r in rows:
        # P2-2: name the predicate explicitly — the condition is "episode
        # failed", not "success is falsy", and conflating the two has bitten
        # readers scanning this loop.
        is_failed = not r.get("success", False)
        if not is_failed:
            continue
        task_id = int(r["task_id"])
        orig = int(r.get("orig_init_state_idx", r["init_state_idx"]))
        failed_by_task[task_id].add(orig)
    return {tid: sorted(indices) for tid, indices in sorted(failed_by_task.items())}


def _build_filter_entries(
    failed_by_task: Dict[int, List[int]],
) -> List[Dict[str, int]]:
    """Materialise the §19.B6 filter schema from a sorted failed map.

    Subset order matches ``.init`` file layout — ascending ``orig_init_state_idx``
    within each ``task_id``.
    """
    entries: List[Dict[str, int]] = []
    for task_id, sorted_bad in failed_by_task.items():
        for subset_pos, orig in enumerate(sorted_bad):
            entries.append(
                {
                    "task_id": int(task_id),
                    "orig_init_state_idx": int(orig),
                    "subset_init_state_idx": int(subset_pos),
                }
            )
    return entries


def _write_artifacts(
    out_dir: Path,
    failed_by_task: Dict[int, List[int]],
    task_names: Dict[int, str],
    init_subsets: Dict[int, Any] | None,
) -> Tuple[List[Path], Path]:
    """Write per-task ``.init`` / ``.init_map.json`` + ``step1b_filter.json``.

    ``task_names[task_id]`` is the LIBERO ``task.name`` (used as filename).
    ``init_subsets[task_id]`` is the already-sliced ``(K, 92)`` tensor; when
    ``None``, ``.init`` files are skipped (used by unit tests that do not
    have a real benchmark available).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written_init_files: List[Path] = []

    for task_id, sorted_bad in failed_by_task.items():
        name = task_names[task_id]
        (out_dir / f"{name}.init_map.json").write_text(json.dumps(sorted_bad))

        if init_subsets is not None:
            import torch  # Lazy: tests without torch skip this branch.
            init_path = out_dir / f"{name}.init"
            torch.save(init_subsets[task_id], init_path)
            written_init_files.append(init_path)

    filter_path = out_dir / "step1b_filter.json"
    filter_path.write_text(json.dumps(_build_filter_entries(failed_by_task), indent=2))
    return written_init_files, filter_path


def _run(
    step1a_results: Path,
    task_suite_name: str,
    out_dir: Path,
    *,
    skip_torch: bool = False,
) -> None:
    """End-to-end driver; also reachable from tests to avoid CLI plumbing."""
    rows = json.loads(Path(step1a_results).read_text())
    failed_by_task = _group_failed_by_task(rows)
    if not failed_by_task:
        print("No failed episodes found — nothing to dump.")
        return

    # Resolve task.name + init state slices from the live LIBERO benchmark.
    # Import is lazy so CI without the libero sim stack can still exercise
    # the helpers above via _run(... skip_torch=True).
    task_names: Dict[int, str] = {}
    init_subsets: Dict[int, Any] | None
    if skip_torch:
        # Pure-schema dump: name each task by its id; skip .init writing.
        init_subsets = None
        for tid in failed_by_task:
            task_names[tid] = f"task_{tid}"
    else:
        from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

        suite = get_benchmark_dict()[task_suite_name]()
        init_subsets = {}
        for task_id, sorted_bad in failed_by_task.items():
            task = suite.get_task(task_id)
            task_names[task_id] = task.name
            full_inits = suite.get_task_init_states(task_id)  # shape (50, 92)
            init_subsets[task_id] = full_inits[sorted_bad]

    written, filter_path = _write_artifacts(
        out_dir, failed_by_task, task_names, init_subsets
    )
    total = sum(len(v) for v in failed_by_task.values())
    print(
        f"Wrote {len(written)} .init file(s) covering {total} failed episode(s) "
        f"across {len(failed_by_task)} task(s) → {out_dir}"
    )
    print(f"Filter: {filter_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step1a-results", required=True, help="Aggregated cache_eval_results.json")
    ap.add_argument("--task-suite", required=True, help="LIBERO task suite name, e.g. libero_spatial")
    ap.add_argument("--out-dir", required=True, help="Output directory for .init + filter files")
    ap.add_argument(
        "--no-torch",
        action="store_true",
        help="Skip .init tensor writes (emits only .init_map.json + step1b_filter.json). "
        "Intended for CI smoke-tests without the libero_sim conda env.",
    )
    args = ap.parse_args()

    _run(
        step1a_results=Path(args.step1a_results),
        task_suite_name=args.task_suite,
        out_dir=Path(args.out_dir),
        skip_torch=args.no_torch,
    )


if __name__ == "__main__":
    main()
