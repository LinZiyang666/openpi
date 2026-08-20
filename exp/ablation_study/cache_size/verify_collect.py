"""P3 exit gate: is a suite's collection complete, well-formed, and accounted for?

Runs against ``<collect_dir>/<suite>/`` after a collection pass. Five gates, all
fail-loud, plus a ledger that becomes the provenance record for a multi-hour run:

1.  **Grid completeness** -- exactly ``task_{0..9}/episode_{0..49}.h5``. A missing
    episode is the dangerous case: every downstream statistic is per-task, so a
    short task silently shrinks its own denominator rather than announcing itself.
2.  **Schema** -- the datasets the builder's contract requires, with the shapes
    and dtypes it requires. This is the gate that separates this collection from
    the Phase 0 one, whose files looked fine and contained no embeddings at all.
3.  **Step-hole probe** -- ``attrs.num_steps`` must equal the number of ``step_*``
    groups, on *every* file. A live cache short-circuits inference on FULL_HIT and
    leaves the collector with holes; the collection server therefore runs with the
    cache off, and this is the check that would notice if it ever did not.
4.  **Ledger join** -- the per-episode results JSON must cover exactly the same
    ``(task_id, init_state_idx)`` grid as the h5 tree. The size grid is built from
    *successful* trajectories, so a results file that disagrees with the h5 tree
    would silently mis-select which trajectories enter the library.
5.  **sha256 accounting** -- one digest per file, so a later "is this still the
    same 500 episodes" question has an answer.

The per-task success histogram is reported too, because it decides whether the
top size tier is even attainable: a task with fewer than S6 successes caps that
task's contribution, which the grid rules have to handle explicitly rather than
discover late.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import h5py

NUM_TASKS = 10
EPISODES_PER_TASK = 50

# (dataset name, expected shape, expected dtype kind+size)
REQUIRED_STEP_DATASETS = {
    "vision_0": ((256, 2048), "float16"),
    "vision_1": ((256, 2048), "float16"),
    "vision_2": ((256, 2048), "float16"),
    "prompt_emb": ((200, 2048), "float16"),
    "robot_state": ((32,), "float32"),
    "clean_action": ((10, 32), "float32"),
}


def expected_grid() -> set[tuple[int, int]]:
    return {(t, e) for t in range(NUM_TASKS) for e in range(EPISODES_PER_TASK)}


def scan_paths(suite_dir: pathlib.Path) -> dict[tuple[int, int], pathlib.Path]:
    """``{(task_id, episode_idx): path}`` from the ``task_N/episode_M.h5`` layout."""
    found: dict[tuple[int, int], pathlib.Path] = {}
    for p in sorted(suite_dir.glob("task_*/episode_*.h5")):
        try:
            task_id = int(p.parent.name.split("_", 1)[1])
            ep_idx = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            raise SystemExit(f"unparseable collect path {p}")
        if (task_id, ep_idx) in found:
            raise SystemExit(f"duplicate episode {(task_id, ep_idx)}: {p}")
        found[(task_id, ep_idx)] = p
    return found


def assert_grid_complete(found: dict[tuple[int, int], pathlib.Path]) -> None:
    want = expected_grid()
    have = set(found)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        raise SystemExit(
            f"collection grid is not {NUM_TASKS}x{EPISODES_PER_TASK}: "
            f"{len(missing)} missing (e.g. {missing[:5]}), "
            f"{len(extra)} unexpected (e.g. {extra[:5]})"
        )


def check_one(path: pathlib.Path, *, deep: bool) -> dict:
    """Structure of one episode. ``deep`` checks every step, not just the ends."""
    with h5py.File(path, "r") as h:
        steps = sorted(k for k in h.keys() if k.startswith("step_"))
        declared = int(h.attrs["num_steps"])
        if declared != len(steps):
            raise SystemExit(
                f"{path}: attrs.num_steps={declared} but {len(steps)} step groups. "
                "A step hole means some inference was short-circuited -- the usual "
                "cause is a live cache on the collection server."
            )
        if not steps:
            raise SystemExit(f"{path}: no step groups at all")

        to_check = steps if deep else [steps[0], steps[-1]]
        for s in to_check:
            grp = h[s]
            for name, (shape, dtype) in REQUIRED_STEP_DATASETS.items():
                if name not in grp:
                    raise SystemExit(
                        f"{path}:{s} lacks {name!r}. The client-side --save_trajectory "
                        "dump has this exact shape of absence: it carries no embeddings "
                        "at all, which is why that pass could not build a library."
                    )
                ds = grp[name]
                if tuple(ds.shape) != shape:
                    raise SystemExit(f"{path}:{s}:{name} shape {ds.shape}, expected {shape}")
                if ds.dtype.name != dtype:
                    raise SystemExit(f"{path}:{s}:{name} dtype {ds.dtype}, expected {dtype}")
        return {
            "num_steps": declared,
            "success": bool(h.attrs["success"]),
            "task": str(h.attrs["task"]),
        }


def sha256_of(path: pathlib.Path, chunk: int = 8 << 20) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            d.update(block)
    return d.hexdigest()


def load_results(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    rows = json.loads(path.read_text())
    out: dict[tuple[int, int], bool] = {}
    for r in rows:
        key = (int(r["task_id"]), int(r["init_state_idx"]))
        if key in out:
            raise SystemExit(f"results JSON has duplicate entry for {key}")
        out[key] = bool(r["success"])
    return out


def assert_results_agree(
    results: dict[tuple[int, int], bool],
    h5_success: dict[tuple[int, int], bool],
) -> None:
    """Same grid, same verdicts.

    Two independent writers record the outcome of each episode: the client's
    results JSON and the collector's h5 attrs. They are the join basis for the
    size grid, so a disagreement is not a cosmetic mismatch -- it would put a
    failed trajectory into the library, or drop a successful one.
    """
    a, b = set(results), set(h5_success)
    if a != b:
        raise SystemExit(
            f"results JSON and h5 tree cover different grids: "
            f"{len(a - b)} only in JSON (e.g. {sorted(a - b)[:5]}), "
            f"{len(b - a)} only in h5 (e.g. {sorted(b - a)[:5]})"
        )
    disagree = sorted(k for k in a if results[k] != h5_success[k])
    if disagree:
        raise SystemExit(
            f"{len(disagree)} episode(s) disagree on success between the results JSON "
            f"and the h5 attrs (e.g. {disagree[:5]})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", required=True,
                    help="<collect_dir>/<suite>, i.e. the dir holding task_N/")
    ap.add_argument("--results-json", required=True,
                    help="the client's --episode-results-path output")
    ap.add_argument("--out", required=True, help="ledger JSON to write")
    ap.add_argument("--deep-per-task", type=int, default=1,
                    help="episodes per task to check at every step (rest: first+last "
                         "step only). The sampled fraction is recorded in the ledger.")
    ap.add_argument("--no-sha", action="store_true",
                    help="skip the digest pass (fast re-check; the ledger then says so)")
    args = ap.parse_args()

    suite_dir = pathlib.Path(args.suite_dir)
    found = scan_paths(suite_dir)
    assert_grid_complete(found)

    deep_keys = {(t, e) for t in range(NUM_TASKS) for e in range(args.deep_per_task)}
    per_file: dict[str, dict] = {}
    h5_success: dict[tuple[int, int], bool] = {}
    total_bytes = 0
    for key in sorted(found):
        path = found[key]
        info = check_one(path, deep=key in deep_keys)
        h5_success[key] = info["success"]
        size = path.stat().st_size
        total_bytes += size
        rel = f"task_{key[0]}/episode_{key[1]}.h5"
        per_file[rel] = {
            "num_steps": info["num_steps"],
            "success": info["success"],
            "bytes": size,
            "sha256": None if args.no_sha else sha256_of(path),
        }

    results = load_results(pathlib.Path(args.results_json))
    assert_results_agree(results, h5_success)

    by_task = {t: sum(h5_success[(t, e)] for e in range(EPISODES_PER_TASK))
               for t in range(NUM_TASKS)}
    ledger = {
        "suite_dir": str(suite_dir.resolve()),
        "episodes": len(found),
        "total_bytes": total_bytes,
        "successes": sum(h5_success.values()),
        "success_by_task": by_task,
        "deep_checked_per_task": args.deep_per_task,
        "sha256_recorded": not args.no_sha,
        "files": per_file,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2, sort_keys=True))

    print(json.dumps({
        "episodes": ledger["episodes"],
        "successes": ledger["successes"],
        "success_by_task": by_task,
        "min_task_successes": min(by_task.values()),
        "total_GB": round(total_bytes / 2**30, 2),
        "deep_checked": f"{args.deep_per_task}/{EPISODES_PER_TASK} per task",
        "sha256_recorded": ledger["sha256_recorded"],
    }, indent=2))
    # The top size tier is 45 successful trajectories per task; a task below that
    # caps its own contribution and the grid rules must treat it as topped out.
    if min(by_task.values()) < 45:
        print(f"NOTE: {sum(1 for v in by_task.values() if v < 45)} task(s) have fewer "
              "than 45 successes -- those are topped-out tasks for the S6 tier "
              "(plan 3.2 rules R1-R5).", file=sys.stderr)


if __name__ == "__main__":
    main()
