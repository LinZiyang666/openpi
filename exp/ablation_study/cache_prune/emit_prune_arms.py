"""Export the pruned libraries of the frozen grids and write one YAML per arm.

An arm is a pure-cache YAML (the checked-in retrieval template with only
``preload_path`` filled in) pointing at one library: P00 at the source library
itself, P01..P09 at the retained subset the frozen grid names. The retained
set is fixed by the frozen score matrix (``select_retained``), so exporting a
point is a deterministic projection of the source; once a library is on disk
its arm can be evaluated.

Points may be exported in parallel by running several copies over disjoint
``--points``; each point's directory is published atomically and reserved by
a sibling lock, so two copies cannot build the same one. The YAML/matrix pass
lists whatever libraries exist.

Usage:
  uv run python -m exp.ablation_study.cache_prune.emit_prune_arms \\
      --inputs <inputs.json> --artifacts-root <dir> --out <config dir> \\
      [--points libero_10:cs500_success:P01,...] [--yamls-only]
"""

from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path
import time

import torch
import yaml

from .common import (
    CENSUS,
    REGIMES,
    SUITES,
    check_seal,
    digest,
    read_json,
    require,
    retrieval_digest,
    template,
    validate_config,
)
from .prune_library import export_pruned, load_score_blocks, select_retained
from .select_prune_grid import grid_spec, match_targets


def arm_name(suite: str, regime: str, point: str) -> str:
    """Return the stable identity of an arm in the Cartesian product."""
    return f"cache_prune_{suite}_{regime}_{point}"


def validate_grid(source: dict, grid: dict) -> None:
    """Recheck census, candidate integrity and the exact preregistered rate match."""
    check_seal(source)
    check_seal(grid)
    require(
        grid["status"] == "frozen" and grid["complete"] is True,
        "source grid is not frozen",
    )
    require(
        grid["source_digest"] == source["digest"]
        and grid["source_file"] == source["file"],
        "grid/source mismatch",
    )
    require(
        (source["trajectories"], source["entries"])
        == CENSUS[(source["suite"], source["regime"])],
        "source differs from approved census",
    )
    require(len(source["task_counts"]) == 10, "source must cover ten tasks")
    require(
        grid["spec"] == grid_spec() and grid["spec_digest"] == digest(grid_spec()),
        "grid protocol changed",
    )
    require(
        grid["candidate_digest"] == digest(grid["candidates"])
        and grid["candidate_count"] == len(grid["candidates"]),
        "candidate table changed",
    )
    require(
        all(c["complete"] is True and c["seconds"] >= 0 for c in grid["candidates"]),
        "partial candidate table",
    )
    require(
        grid["retrieval_digest"] == retrieval_digest(template(source["suite"])),
        "grid used different retrieval settings",
    )
    expected = match_targets(
        grid["candidates"],
        source["entries"],
        grid_spec()["targets"],
        grid_spec()["max_gap"],
        grid_spec()["first_max"],
    )
    require(
        [p["point"] for p in grid["points"]] == [f"P{i:02d}" for i in range(10)],
        "grid is not P00..P09",
    )
    baseline = grid["points"][0]
    require(
        baseline
        == {
            "point": "P00",
            "threshold": None,
            "actual_rate": 0.0,
            "target_rate": 0.0,
            "removed_count": 0,
            "retained_count": source["entries"],
            "keep_digest": digest([r["id"] for r in source["rows"]]),
        },
        "invalid P00 grid bypass",
    )
    for point, candidate, target in zip(
        grid["points"][1:], expected, grid_spec()["targets"]
    ):
        require(
            {k: v for k, v in point.items() if k not in ("point", "target_rate")}
            == candidate
            and point["target_rate"] == target,
            "grid differs from exact mechanical selection",
        )


def library_path(source: dict, point: dict, artifacts_root: Path) -> Path | None:
    """Where a point's library lives, or None while it is not exported yet."""
    if point["point"] == "P00":
        return Path(source["file"]["path"])
    manifest = artifacts_root / arm_name(source["suite"], source["regime"], point["point"]) / "artifact.json"
    if not manifest.exists():
        return None
    return Path(read_json(manifest)["file"]["path"])


def export_points(source: dict, scores: dict, grid: dict, artifacts_root: Path, points: set[str]) -> None:
    """Materialise the named points of one source; existing ones are left alone."""
    blocks = None
    for point in grid["points"]:
        if point["point"] not in points or point["point"] == "P00":
            continue
        name = arm_name(source["suite"], source["regime"], point["point"])
        destination = artifacts_root / name
        if destination.exists():
            print(f"[{name}] present", flush=True)
            continue
        if blocks is None:
            blocks = load_score_blocks(scores, source["rows"])
        started = time.monotonic()
        selected = select_retained(source["rows"], blocks, point["threshold"])
        require(
            digest(selected.kept_ids) == point["keep_digest"],
            "regenerated retained set differs from grid",
        )
        export_pruned(source["file"]["path"], selected, destination, source_manifest=source)
        print(f"[{name}] exported in {time.monotonic() - started:.0f}s", flush=True)


def emit_yamls(sources: list[dict], grids: dict, artifacts_root: Path, out_dir: Path) -> dict[str, list[str]]:
    """One YAML per exported arm and one matrix per family; returns the missing arms."""
    (out_dir / "arms").mkdir(parents=True, exist_ok=True)
    missing: dict[str, list[str]] = {}
    for source in sources:
        suite, regime = source["suite"], source["regime"]
        rows = []
        for point in grids[suite, regime]["points"]:
            name = arm_name(suite, regime, point["point"])
            library = library_path(source, point, artifacts_root)
            if library is None or not library.exists():
                missing.setdefault(f"{suite}/{regime}", []).append(name)
                continue
            config = copy.deepcopy(template(suite))
            config["backend"]["in_memory"]["preload_path"] = str(library)
            validate_config(config, suite)
            path = out_dir / "arms" / f"{name}.yaml"
            text = yaml.safe_dump(config, sort_keys=False)
            require(
                not path.exists() or path.read_text() == text,
                f"{path} exists with different content",
            )
            path.write_text(text)
            rows.append({"arm": name, "yaml": str(path), "sidecar": None})
        matrix = out_dir / f"matrix_{suite}_{regime}.yaml"
        matrix.write_text(yaml.safe_dump({"arms": rows}, sort_keys=False))
        print(f"{suite}/{regime}: {len(rows)} arms -> {matrix}", flush=True)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="JSON: sources[{source, scores, grid}]")
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", default="",
                        help="suite:regime:Pxx comma list to export (default: every point)")
    parser.add_argument("--yamls-only", action="store_true",
                        help="skip exporting; write YAMLs for the libraries that exist")
    args = parser.parse_args()
    torch.set_num_threads(4)
    inputs = read_json(args.inputs)
    triples = [
        (read_json(item["source"]), read_json(item["scores"]), read_json(item["grid"]))
        for item in inputs["sources"]
    ]
    require(
        len(triples) == 4
        and {(s["suite"], s["regime"]) for s, _, _ in triples}
        == set(product(SUITES, REGIMES)),
        "need four source grids",
    )
    for source, scores, grid in triples:
        validate_grid(source, grid)
        check_seal(scores)
        require(grid["score_digest"] == scores["digest"], "wrong score manifest")
    wanted = {tuple(item.split(":")) for item in args.points.split(",") if item}
    if not args.yamls_only:
        for source, scores, grid in triples:
            points = {
                p for s, r, p in wanted if (s, r) == (source["suite"], source["regime"])
            } if wanted else {p["point"] for p in grid["points"]}
            export_points(source, scores, grid, args.artifacts_root, points)
    missing = emit_yamls(
        [s for s, _, _ in triples],
        {(g["suite"], g["regime"]): g for _, _, g in triples},
        args.artifacts_root,
        args.out,
    )
    for family, names in missing.items():
        print(f"{family}: not exported yet: {names}")


if __name__ == "__main__":
    main()
