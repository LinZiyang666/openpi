"""Copy selected frozen arm/config rows into a fresh local run root.

Fits and libraries stay explicit read-only external inputs. Do not rerun the
arm generator, copy result/state directories, or modify the source run.
"""
import argparse
import json
from pathlib import Path
import re

import yaml


def prepare(source, destination, names):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.name == destination.name:
        raise ValueError("source/destination must have distinct run tags")
    if not names or len(set(names)) != len(names):
        raise ValueError("supply a nonempty distinct arm list")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names):
        raise ValueError("invalid arm name")
    rows = {r["arm"]: r for r in json.loads((source / "arms.json").read_text())}
    missing = set(names) - rows.keys()
    if missing:
        raise ValueError("unknown arms: " + ", ".join(sorted(missing)))
    if destination.exists():
        raise FileExistsError("fresh destination required; reuse an existing prepared root directly")
    prepared, files = [], {}
    for name in names:
        # Freeze external <RUN> inputs to A, then move only each row's own config
        # paths to B. build_plan will relocate the external prefit as usual.
        row = json.loads(json.dumps(rows[name]).replace("<RUN>", str(source)))
        config, matrix = Path(row["yaml"]), Path(row["matrix"])
        new_config = destination / "config" / (name + ".yaml")
        new_matrix = destination / "config" / ("matrix_" + name + ".yaml")
        files[new_config] = config.read_text().replace("<RUN>", str(source))
        data = yaml.safe_load(matrix.read_text().replace("<RUN>", str(source)))
        data = json.loads(json.dumps(data).replace(str(config), str(new_config)))
        files[new_matrix] = yaml.safe_dump(data, sort_keys=False)
        row.update(yaml=str(new_config), matrix=str(new_matrix))
        prepared.append(row)
    destination.mkdir(parents=True)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (destination / "arms.json").write_text(json.dumps(prepared, indent=1) + "\n")
    (destination / "eval500.json").write_text(json.dumps([[t, i] for t in range(10) for i in range(50)]) + "\n")
    print(f"RUN_PREPARED root={destination} arms={len(prepared)} pairs=500")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    prepare(args.source, args.destination, args.arms)


if __name__ == "__main__":
    main()
