"""Merge per-run audit manifests of one teacher into a single library manifest.

The W13 RoboCasa365 corpus is collected by two drivers per teacher -- a pinned
PickPlace run and an unpinned run over the other tasks -- because a pinned
run refuses tasks its pin table does not cover. Each driver is audited on its
own (``verify_collection_artifacts.py``) and the builder consumes exactly one
manifest, so the audited sets are merged here: task sets must be disjoint,
``target`` must agree, ``plan_hashes`` are unioned, and ``pin_id`` survives
only when every input carries the same one (a mixed corpus is stamped with no
pin id; the builder would otherwise demand the pin on every episode).
"""

from __future__ import annotations

import argparse
import json
import pathlib


def merge(paths: list[pathlib.Path], only_tasks: set[str] | None = None) -> dict:
    docs = [json.loads(p.read_text()) for p in paths]
    targets = {d.get("target") for d in docs}
    if len(targets) != 1:
        raise SystemExit(f"manifests disagree on target: {targets}")
    tasks: dict[str, list] = {}
    hashes: list[str] = []
    for p, d in zip(paths, docs):
        for name, rows in d["tasks"].items():
            if only_tasks is not None and name not in only_tasks:
                continue
            if name in tasks:
                raise SystemExit(f"task {name} appears in more than one manifest ({p})")
            tasks[name] = rows
        for h in d.get("plan_hashes", []):
            if h not in hashes:
                hashes.append(h)
    pins = {d.get("pin_id") for d in docs}
    out = {
        "target": targets.pop(),
        "plan_hashes": hashes,
        "tasks": dict(sorted(tasks.items())),
        "merged_from": [str(p) for p in paths],
    }
    if len(pins) == 1 and None not in pins:
        out["pin_id"] = pins.pop()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument(
        "--only-tasks",
        default=None,
        help="comma-separated task names to keep; tasks outside the formal set are dropped",
    )
    args = ap.parse_args()
    only = set(args.only_tasks.split(",")) if args.only_tasks else None
    doc = merge(args.manifests, only)
    if only is not None:
        missing = sorted(only - set(doc["tasks"]))
        if missing:
            raise SystemExit(f"requested tasks absent from every manifest: {missing}")
    args.out.write_text(json.dumps(doc, indent=2))
    n = sum(len(v) for v in doc["tasks"].values())
    print(f"wrote {args.out}: {len(doc['tasks'])} tasks, {n} episodes, pin_id={doc.get('pin_id')}")


if __name__ == "__main__":
    main()
