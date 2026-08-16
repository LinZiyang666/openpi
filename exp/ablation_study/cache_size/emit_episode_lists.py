"""Turn a size grid into ``--episode-list`` files for the artifact builder.

One list per tier. Paths are written relative to ``<collect_dir>/<suite>``,
which is what ``--data-dir`` must point at: ``EpisodeDataCollector`` inserts an
``<experiment>`` level of its own (``out_dir = base_dir / experiment`` with
``experiment = task_suite_name``), so pointing ``--data-dir`` one level higher
would prepend an extra component to every relative path and reshape every
entry id.
"""

from __future__ import annotations

import argparse
import pathlib

import yaml


def lists_from_grid(grid: dict) -> dict[str, list[str]]:
    """Flatten ``episodes[tier][task] -> [rel_path]`` into one sorted list per tier."""
    out: dict[str, list[str]] = {}
    for tier, per_task in grid["episodes"].items():
        paths: list[str] = []
        for task in sorted(per_task, key=lambda t: int(t.split("_", 1)[1])):
            paths.extend(per_task[task])
        out[tier] = paths
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True, help="size_grid_<suite>.yaml")
    ap.add_argument("--out-dir", required=True, help="destination directory for the .txt lists")
    ap.add_argument("--suite", required=True)
    args = ap.parse_args()

    grid = yaml.safe_load(pathlib.Path(args.grid).read_text())
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tier, paths in lists_from_grid(grid).items():
        dest = out_dir / f"episodes_{args.suite}_{tier}.txt"
        dest.write_text("\n".join(paths) + "\n")
        print(f"{dest}  ({len(paths)} episodes)")


if __name__ == "__main__":
    main()
