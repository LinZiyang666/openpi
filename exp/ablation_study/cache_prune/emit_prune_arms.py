"""Materialize verified subsets and freeze the complete forty-arm experiment.

Every source has its own measured grid. This module validates all four before
publishing YAMLs, binds source/score/membership evidence, and emits matrices for
the existing cache-size conductor runner without changing that runner.
"""

from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path

import torch
import yaml

from .common import (
    CENSUS,
    REGIMES,
    SUITES,
    VERSION,
    check_identity,
    check_seal,
    digest,
    environment,
    file_identity,
    new_directory,
    read_json,
    require,
    retrieval_digest,
    seal,
    template,
    validate_config,
    write_json,
)
from .prepare_membership import validate_membership
from .prune_library import export_pruned, load_score_blocks, select_retained
from .select_prune_grid import grid_spec, match_targets
from .verify_prune import verify_pruned


def arm_name(suite: str, regime: str, point: str) -> str:
    """Return the stable identity of an arm in the frozen Cartesian product."""
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


def _validate_arm(source, grid, artifact, verification, point):
    for manifest in (artifact, verification):
        check_seal(manifest)
    require(
        artifact["source_digest"] == source["digest"]
        and artifact["parent_file"] == source["file"],
        "arm parent differs from P00",
    )
    selection = artifact["selection"]
    require(
        selection["keep_digest"]
        == point["keep_digest"]
        == digest(selection["kept_ids"]),
        "arm retained set differs from grid",
    )
    require(
        selection["threshold"] == point["threshold"]
        and artifact["entries"] == point["retained_count"],
        "arm threshold/census mismatch",
    )
    require(
        selection["source_rows_digest"] == source["rows_digest"],
        "arm source coordinates mismatch",
    )
    require(
        verification["passed"] is True
        and verification.get("score_reference") == "frozen_float32_v1"
        and verification["source_digest"] == source["digest"]
        and verification["artifact_digest"] == artifact["digest"]
        and verification["score_digest"] == grid["score_digest"]
        and verification["queries"] == source["entries"],
        "missing complete artifact verification",
    )
    require(
        verification["retrieval_digest"] == grid["retrieval_digest"],
        "verification used another retrieval configuration",
    )
    if point["point"] == "P00":
        require(
            artifact["file"] == source["file"],
            "P00 must serve the source's exact bytes",
        )


def emit_arms(
    source_manifests: list[dict],
    artifact_manifests: dict[str, dict],
    templates: dict[str, dict],
    grid: list[dict],
    out_dir: str | Path,
    *,
    memberships: dict[str, dict],
    verifications: dict[str, dict],
) -> dict:
    """Atomically publish all 40 configs and their provenance-bound freeze manifest."""
    pairs = set(product(SUITES, REGIMES))
    sources = {(s["suite"], s["regime"]): s for s in source_manifests}
    grids = {(g["suite"], g["regime"]): g for g in grid}
    require(
        len(source_manifests) == len(grid) == 4 and set(sources) == set(grids) == pairs,
        "expected four distinct source grids",
    )
    require(
        set(memberships) == set(templates) == set(SUITES),
        "missing suite membership/template",
    )
    expected_arms = {arm_name(s, r, f"P{i:02d}") for s, r in pairs for i in range(10)}
    require(
        set(artifact_manifests) == set(verifications) == expected_arms,
        "artifact matrix is not exactly 40 arms",
    )
    for suite in SUITES:
        validate_config(templates[suite], suite)
        validate_membership(memberships[suite])
        require(
            memberships[suite]["source_digests"]
            == {r: sources[suite, r]["digest"] for r in REGIMES},
            "membership bound to other sources",
        )
    for pair in pairs:
        validate_grid(sources[pair], grids[pair])
        check_identity(sources[pair]["file"])
    arms, matrices = [], {suite: [] for suite in SUITES}
    destination = Path(out_dir).resolve()
    with new_directory(destination) as temporary:
        (temporary / "arms").mkdir()
        for suite, regime in sorted(pairs):
            source, frozen_grid = sources[suite, regime], grids[suite, regime]
            for point in frozen_grid["points"]:
                name = arm_name(suite, regime, point["point"])
                artifact, verification = artifact_manifests[name], verifications[name]
                _validate_arm(source, frozen_grid, artifact, verification, point)
                check_identity(artifact["file"])
                config = copy.deepcopy(templates[suite])
                config["backend"]["in_memory"]["preload_path"] = artifact["file"][
                    "path"
                ]
                validate_config(config, suite)
                relative = Path("arms") / f"{name}.yaml"
                (temporary / relative).write_text(
                    yaml.safe_dump(config, sort_keys=False)
                )
                identity = file_identity(temporary / relative)
                identity["path"] = str(destination / relative)
                matrices[suite].append(
                    {"arm": name, "yaml": identity["path"], "sidecar": None}
                )
                arms.append(
                    {
                        "arm": name,
                        "suite": suite,
                        "regime": regime,
                        "point": point["point"],
                        "yaml": identity,
                        "artifact": artifact,
                        "verification": verification,
                        "grid_digest": frozen_grid["digest"],
                        "source_digest": source["digest"],
                        "retrieval_digest": retrieval_digest(config),
                        "actual_rate": point["actual_rate"],
                        "target_rate": point["target_rate"],
                    }
                )
        matrix_files = {}
        for suite, rows in matrices.items():
            name = f"matrix_{suite}.yaml"
            (temporary / name).write_text(
                yaml.safe_dump({"arms": rows}, sort_keys=False)
            )
            identity = file_identity(temporary / name)
            identity["path"] = str(destination / name)
            matrix_files[suite] = identity
        freeze = seal(
            {
                "version": VERSION,
                "kind": "freeze",
                "complete": True,
                "arms": arms,
                "sources": source_manifests,
                "grids": grid,
                "memberships": memberships,
                "matrices": matrix_files,
                "environment": environment(),
                "numerical_expectation": {
                    "role": "record_only",
                    "hard_gate": False,
                    "reference": "g2_r1_20260911/four_source_P05",
                    "observed_envelope": 0.00010631978511810303,
                    "per_source_max_score_error": {
                        "libero_spatial/rit50": 0.0,
                        "libero_spatial/cs500_success": 0.0,
                        "libero_10/rit50": 0.00010631978511810303,
                        "libero_10/cs500_success": 0.0,
                    },
                },
                "protocol": {
                    "num_steps_wait": 10,
                    "replan_steps": 5,
                    "trials": 50,
                    "tasks": 10,
                    "max_steps": {"libero_spatial": 220, "libero_10": 520},
                },
            }
        )
        write_json(temporary / "freeze.json", freeze)
    return freeze


def validate_freeze(freeze: dict, *, rehash: bool = True) -> None:
    """Check complete identities, immutable YAMLs, matrices and all source bindings."""
    check_seal(freeze)
    require(
        freeze["kind"] == "freeze" and freeze["complete"] is True,
        "incomplete experiment freeze",
    )
    expected = {
        arm_name(s, r, f"P{i:02d}") for s, r, i in product(SUITES, REGIMES, range(10))
    }
    require(
        len(freeze["arms"]) == 40 and {a["arm"] for a in freeze["arms"]} == expected,
        "not exactly 40 arms",
    )
    sources = {(s["suite"], s["regime"]): s for s in freeze["sources"]}
    grids = {(g["suite"], g["regime"]): g for g in freeze["grids"]}
    require(
        len(freeze["sources"]) == len(freeze["grids"]) == 4
        and set(sources) == set(grids) == set(product(SUITES, REGIMES)),
        "four-source coverage mismatch",
    )
    require(
        freeze["protocol"]
        == {
            "num_steps_wait": 10,
            "replan_steps": 5,
            "trials": 50,
            "tasks": 10,
            "max_steps": {"libero_spatial": 220, "libero_10": 520},
        },
        "evaluation protocol changed",
    )
    for pair, source in sources.items():
        validate_grid(source, grids[pair])
        if rehash:
            check_identity(source["file"])
    for suite in SUITES:
        membership = freeze["memberships"][suite]
        validate_membership(membership, rehash=rehash)
        require(
            membership["source_digests"]
            == {r: sources[suite, r]["digest"] for r in REGIMES},
            "membership/source mismatch",
        )
    for arm in freeze["arms"]:
        pair = arm["suite"], arm["regime"]
        require(arm["arm"] == arm_name(*pair, arm["point"]), "arm tuple/name mismatch")
        source, grid = sources[pair], grids[pair]
        point = next(p for p in grid["points"] if p["point"] == arm["point"])
        _validate_arm(source, grid, arm["artifact"], arm["verification"], point)
        require(
            arm["source_digest"] == source["digest"]
            and arm["grid_digest"] == grid["digest"],
            "arm provenance mismatch",
        )
        require(
            arm["actual_rate"] == point["actual_rate"]
            and arm["target_rate"] == point["target_rate"],
            "arm rate mismatch",
        )
        check_identity(arm["yaml"])
        config = yaml.safe_load(Path(arm["yaml"]["path"]).read_text())
        validate_config(config, arm["suite"])
        require(
            config["backend"]["in_memory"]["preload_path"]
            == arm["artifact"]["file"]["path"],
            "YAML serves another library",
        )
        require(
            arm["retrieval_digest"] == retrieval_digest(config),
            "arm retrieval fingerprint changed",
        )
        if rehash:
            check_identity(arm["artifact"]["file"])
    for suite in SUITES:
        check_identity(freeze["matrices"][suite])
        matrix = yaml.safe_load(Path(freeze["matrices"][suite]["path"]).read_text())
        wanted = [
            {"arm": a["arm"], "yaml": a["yaml"]["path"], "sidecar": None}
            for a in freeze["arms"]
            if a["suite"] == suite
        ]
        require(matrix == {"arms": wanted}, "runner matrix differs from frozen arm set")


def main() -> None:
    """Materialize all four pre-frozen grids, verify artifacts, then emit 40 YAMLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON: sources[{source, scores, grid}], memberships{suite:path}",
    )
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
        "need four source grids before any export",
    )
    for source, scores, grid in triples:
        validate_grid(source, grid)
        check_seal(scores)
        require(grid["score_digest"] == scores["digest"], "wrong score manifest")
    memberships = {
        suite: read_json(path) for suite, path in inputs["memberships"].items()
    }
    require(set(memberships) == set(SUITES), "need both membership freezes")
    for membership in memberships.values():
        validate_membership(membership)
    artifacts, verifications = {}, {}
    for source, scores, grid in triples:
        blocks = load_score_blocks(scores, source["rows"])
        for point in grid["points"]:
            name = arm_name(source["suite"], source["regime"], point["point"])
            destination = args.artifacts_root / name
            if destination.exists():
                artifact = read_json(destination / "artifact.json")
            else:
                selected = select_retained(source["rows"], blocks, point["threshold"])
                require(
                    digest(selected.kept_ids) == point["keep_digest"],
                    "regenerated retained set differs from grid",
                )
                artifact = export_pruned(
                    source["file"]["path"],
                    selected,
                    destination,
                    source_manifest=source,
                )
            report_path = destination / "verification.json"
            if report_path.exists():
                verification = read_json(report_path)
                check_identity(artifact["file"])
            else:
                verification = verify_pruned(
                    source, artifact, template(source["suite"]), score_manifest=scores
                )
                write_json(report_path, verification)
            _validate_arm(source, grid, artifact, verification, point)
            artifacts[name], verifications[name] = artifact, verification
        del blocks
    freeze = emit_arms(
        [s for s, _, _ in triples],
        artifacts,
        {s: template(s) for s in SUITES},
        [g for _, _, g in triples],
        args.out,
        memberships=memberships,
        verifications=verifications,
    )
    validate_freeze(freeze)


if __name__ == "__main__":
    main()
