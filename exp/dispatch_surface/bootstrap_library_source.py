"""Re-collect a suite's D_lib library source so its provenance is recordable.

``split_init_pools.verify_dlib_content`` proves, per library episode, that the
collection pool's init state is byte-identical to the official state at the
recorded ``orig_init_state_idx``. That proof needs an init map carrying
``h5_path`` / ``attrs`` / ``entry_count`` / ``full_init_path`` / ``init_path``,
and H5 files whose attrs can be compared against it.

libero_spatial's library source has all of that. libero_10's does not: its init
map holds five fields (task_id / task_name / prompt / orig_init_state_idx /
subset_idx) and its H5 attrs carry no init identity at all, so relating a map
row to a specific H5 could only be guessed -- and a guessed identity is exactly
what the content census exists to reject (a swapped pair would silently corrupt
the D_lib/query/test partition and never be detectable afterwards).

This tool re-collects those episodes through the identity-stamping path, then
builds the full init map FROM THE COLLECTED FILES rather than from inference:

  plan         pick the official inits (from an existing map, or a fixed-seed
               draw), materialise the 5/task collection pool, and emit a
               cohort-plan-shaped file the LIBERO client already understands.
  rebuild-map  walk the collected H5 tree and write an init map in the schema
               verify_dlib_content requires, then run that census as a
               self-check before anything downstream trusts the result.

Usage:
  uv run python -m exp.dispatch_surface.bootstrap_library_source plan \
      --suite libero_10 \
      --source-init-map exp/common/data/db/libero_cache/libero_10_init_map.json \
      --apool-dir exp/common/data/db_init/libero/libero_10 \
      --out-root exp/dispatch_surface/data/libero_10/dlib_source

  uv run python -m exp.dispatch_surface.bootstrap_library_source rebuild-map \
      --plan exp/dispatch_surface/data/libero_10/dlib_source/dlib_plan.json \
      --h5-dir exp/dispatch_surface/data/libero_10/dlib_source/collected \
      --out exp/dispatch_surface/data/libero_10/dlib_init_map.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import torch

from exp.dispatch_surface.split_init_pools import (
    DLIB_PER_TASK,
    EXPECTED_TASKS,
    OFFICIAL_PER_TASK,
    _normalise_attr,
    census_dlib_inits,
)


def official_dlib_choice(source_map_path: str) -> dict[int, dict]:
    """task -> {task_name, indices} taken from an existing map's own record.

    Reusing the historical choice keeps the rebuilt library on the same official
    inits the previous one used, so the suite's D_lib occupancy does not move
    for reasons unrelated to this line.
    """
    rows = json.loads(pathlib.Path(source_map_path).read_text())
    grouped: dict[int, dict] = collections.defaultdict(
        lambda: {"task_name": None, "indices": []}
    )
    for row in rows:
        for key in ("task_id", "task_name", "orig_init_state_idx"):
            if row.get(key) is None:
                raise SystemExit(f"source init map row lacks {key!r}: {row}")
        tid = int(row["task_id"])
        entry = grouped[tid]
        name = str(row["task_name"])
        if entry["task_name"] is None:
            entry["task_name"] = name
        elif entry["task_name"] != name:
            raise SystemExit(f"task {tid}: inconsistent task_name in source map")
        entry["indices"].append(int(row["orig_init_state_idx"]))
    if sorted(grouped) != list(range(EXPECTED_TASKS)):
        raise SystemExit(f"source map task ids {sorted(grouped)} != 0..{EXPECTED_TASKS - 1}")
    for tid, entry in grouped.items():
        entry["indices"] = sorted(set(entry["indices"]))
        if len(entry["indices"]) != DLIB_PER_TASK:
            raise SystemExit(
                f"task {tid}: source map records {len(entry['indices'])} distinct "
                f"official inits, expected {DLIB_PER_TASK}"
            )
    return dict(grouped)


def cmd_plan(args) -> None:
    choice = official_dlib_choice(args.source_init_map)
    apool = pathlib.Path(args.apool_dir)
    out_root = pathlib.Path(args.out_root)
    pool_dir = out_root / "init_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    pool_digests: dict[str, dict] = {}
    for tid in sorted(choice):
        name = choice[tid]["task_name"]
        src = apool / f"{name}.init"
        if not src.is_file():
            raise SystemExit(f"official pool file missing: {src}")
        states = torch.load(src, weights_only=False)
        if len(states) != OFFICIAL_PER_TASK:
            raise SystemExit(
                f"{src}: {len(states)} inits, expected {OFFICIAL_PER_TASK}"
            )
        officials = choice[tid]["indices"]
        subset = states[officials]  # officials is already sorted
        torch.save(subset, pool_dir / f"{name}.init")
        pool_digests[name] = {"indices": officials, "count": len(officials)}
        for subset_idx, orig in enumerate(officials):
            episodes.append({
                "task_id": tid,
                "task_name": name,
                "subset_init_state_idx": subset_idx,
                "orig_init_state_idx": int(orig),
                "split": "dlib",
            })

    plan = {
        "suite": args.suite,
        "purpose": "D_lib library source re-collection with identity stamping",
        "source_init_map": str(args.source_init_map),
        "apool_dir": str(apool),
        "pool_dir": str(pool_dir),
        "episodes": episodes,
        "pool_digests": pool_digests,
    }
    out = out_root / "dlib_plan.json"
    out.write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(f"planned {len(episodes)} episodes -> {out}")
    print(f"materialised {DLIB_PER_TASK}/task pool -> {pool_dir}")
    print()
    print("server side:")
    print("  scripts/serve_policy.py --collect --collect_dir <out> --env LIBERO "
          "--non-concurrent policy:checkpoint --policy.config pi05_libero "
          "--policy.dir <ckpt>")
    print("client side:")
    print(f"  examples/libero/main.py --host 127.0.0.1 --port 8000 "
          f"--task-suite-name {args.suite} --num-trials-per-task {DLIB_PER_TASK} "
          f"--num-workers 1 --init-states-dir {pool_dir} --cohort-plan {out}")


def cmd_rebuild_map(args) -> None:
    import h5py

    plan = json.loads(pathlib.Path(args.plan).read_text())
    by_key = {
        (int(e["task_id"]), int(e["subset_init_state_idx"])): e
        for e in plan["episodes"]
    }
    h5_root = pathlib.Path(args.h5_dir).resolve()
    paths = sorted(h5_root.rglob("*.h5"))
    if len(paths) != len(by_key):
        raise SystemExit(
            f"{h5_root}: {len(paths)} H5 files but the plan has {len(by_key)} "
            "episodes — refusing to build a map over an incomplete collection"
        )

    rows = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with h5py.File(path, "r") as h5:
            attrs = {k: _normalise_attr(h5.attrs[k]) for k in h5.attrs}
            steps = sum(1 for k in h5 if k.startswith("step_"))
        need = ("task_id", "subset_init_state_idx", "orig_init_state_idx", "split")
        missing = [k for k in need if k not in attrs]
        if missing:
            raise SystemExit(
                f"{path}: H5 lacks stamped identity {missing} — this collection did "
                "not run through the cohort-plan path, so its provenance is not "
                "recorded and cannot be reconstructed"
            )
        key = (int(attrs["task_id"]), int(attrs["subset_init_state_idx"]))
        planned = by_key.get(key)
        if planned is None:
            raise SystemExit(f"{path}: episode {key} is not in the plan")
        if key in seen:
            raise SystemExit(f"{path}: duplicate episode for {key}")
        seen.add(key)
        if int(attrs["orig_init_state_idx"]) != int(planned["orig_init_state_idx"]):
            raise SystemExit(
                f"{path}: stamped official init {attrs['orig_init_state_idx']} != "
                f"planned {planned['orig_init_state_idx']}"
            )
        name = planned["task_name"]
        rows.append({
            "suite": plan["suite"],
            "trajectory_id": path.stem,
            "h5_path": str(path.relative_to(pathlib.Path.cwd())
                           if path.is_relative_to(pathlib.Path.cwd()) else path),
            "task_id": key[0],
            "task_name": name,
            "prompt": attrs.get("prompt") or attrs.get("task"),
            "episode_number": int(attrs.get("episode_id", key[1])),
            "subset_init_state_idx": key[1],
            "orig_init_state_idx": int(attrs["orig_init_state_idx"]),
            "init_path": str(pathlib.Path(plan["pool_dir"]) / f"{name}.init"),
            "full_init_path": str(pathlib.Path(plan["apool_dir"]) / f"{name}.init"),
            "entry_count": steps,
            "attrs": attrs,
        })
    if len(seen) != len(by_key):
        raise SystemExit(f"collection covers {len(seen)} of {len(by_key)} planned episodes")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote init map with {len(rows)} rows -> {out}")

    # Self-check: run the very census this map exists to satisfy, so a map that
    # cannot be verified never reaches the split step.
    per_task, digests = census_dlib_inits(
        out, h5_dir=h5_root, official_pool_dir=pathlib.Path(plan["apool_dir"]),
    )
    print(f"census OK: {len(per_task)} tasks x {DLIB_PER_TASK} official inits, "
          f"{len(digests['h5'])} H5 content digests")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--suite", required=True)
    p.add_argument("--source-init-map", required=True,
                   help="existing map supplying the official D_lib init choice")
    p.add_argument("--apool-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.set_defaults(func=cmd_plan)
    r = sub.add_parser("rebuild-map")
    r.add_argument("--plan", required=True)
    r.add_argument("--h5-dir", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_rebuild_map)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
