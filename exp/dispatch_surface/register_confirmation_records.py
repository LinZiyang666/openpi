"""Write data_authority records for the confirmation-plan artifacts
(plan section 2: ``tgrid_dev``, ``fresh_pools``, ``budget_amendment``).

Records are derived from the bytes on disk (member list, sizes, SHA rollup),
so each record can only be written once its artifact exists; the registry's
``validate`` then re-derives the integrity from the files. Nothing here
mutates an existing record.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib

from exp.dispatch_surface import tgrid_package as tpkg

RECORDS_DIR = pathlib.Path("exp/data_authority/records")


def _members(root: pathlib.Path, files: list[pathlib.Path]) -> list[list]:
    out = []
    for f in sorted(files):
        out.append([str(f.relative_to(root)), f.stat().st_size, tpkg.file_sha256(f)])
    return out


def _integrity(root: pathlib.Path, files: list[pathlib.Path]) -> dict:
    members = _members(root, files)
    rollup = hashlib.sha256("\n".join(f"{m[0]} {m[2]}" for m in members).encode()).hexdigest()
    return {"file_count": len(members), "size_bytes": int(sum(m[1] for m in members)), "sha256": rollup, "members": members}


def _base(dataset_id: str, kind: str, title: str, suite: str, path: pathlib.Path, repo_root: pathlib.Path,
          produced_by: str, consumers: list[str], content: dict, caveats: list[str]) -> dict:
    return {
        "schema_version": 1, "dataset_id": dataset_id, "kind": kind, "title": title,
        "experiment": "exp/dispatch_surface", "suite": suite, "status": "authoritative",
        "authority": {"node": "local", "path": str(path.relative_to(repo_root)), "access": "local"},
        "content": content,
        "provenance": {"produced_by": produced_by, "measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                       "measured_by": "Execution Authority (Rev 2 confirmation plan)", "source_machines": ["weilandserver", "timan107"]},
        "consumers": consumers, "caveats": caveats,
    }


def tgrid_record(package_dir: pathlib.Path, repo_root: pathlib.Path) -> dict:
    manifest = tpkg.verify_package(package_dir / tpkg.MANIFEST_NAME)
    files = [package_dir / v["member"] for v in manifest["members"].values()] + [package_dir / tpkg.MANIFEST_NAME]
    rec = _base(f"dispatch_surface/{manifest['suite']}/tgrid_dev", "cache_artifact",
                f"dispatch_surface / {manifest['suite']} / dense threshold-grid development package (29 arms x 300 cells, posthoc_exploratory)",
                manifest["suite"], package_dir, repo_root, "exp/dispatch_surface/finalize_tgrid_package.py",
                ["exp/dispatch_surface/analysis/budget_cost_map.py", "exp/dispatch_surface/analysis/budget_outcome_design.py"],
                {"manifest": tpkg.MANIFEST_NAME, "roles": sorted(manifest["members"]), "run_ids": manifest["run_ids"],
                 "resolution": "by role through MANIFEST.json only", "posthoc_exploratory": True},
                ["Development data on the official A' inits; never a confirmation result."])
    rec["integrity"] = _integrity(package_dir, files)
    return rec


def fresh_pools_record(pools_dir: pathlib.Path, manifests: list[pathlib.Path], suite: str, repo_root: pathlib.Path,
                       validation_path: pathlib.Path) -> dict:
    """Exclusivity / cross-machine claims come from the re-run validation
    artifact only (G2R1-B5); the record refuses to exist without it."""
    from exp.dispatch_surface.generate_fresh_inits import validate_pool_validation

    by_pool = {}
    for m in manifests:
        doc = json.loads(pathlib.Path(m).read_text())
        for pool in doc.get("pools") or {}:
            by_pool[pool] = pathlib.Path(m)
    if set(by_pool) != {"P", "C"}:
        raise SystemExit("fresh pool record needs exactly the P and C manifests")
    art = validate_pool_validation(validation_path, p_manifest_path=by_pool["P"], c_manifest_path=by_pool["C"])
    files = sorted(pools_dir.rglob("*.init")) + [pathlib.Path(m) for m in manifests] + [pathlib.Path(validation_path)]
    rec = _base(f"dispatch_surface/{suite}/fresh_pools", "init_pool",
                f"dispatch_surface / {suite} / fresh initial-state pools P (10/task) and C (60/task) from BDDL placement re-sampling",
                suite, pools_dir, repo_root, "exp/dispatch_surface/generate_fresh_inits.py",
                ["exp/dispatch_surface/run_precheck.py (confirmation layer)", "exp/dispatch_surface/build_confirmation_task_plan.py",
                 "exp/dispatch_surface/seal_confirmation.py"],
                {"manifests": [str(pathlib.Path(m).relative_to(repo_root)) for m in manifests],
                 "validation_artifact": str(pathlib.Path(validation_path).relative_to(repo_root)),
                 "validation_sha256": tpkg.file_sha256(pathlib.Path(validation_path)),
                 "exclusivity_recomputed": art["exclusivity"], "state_dim": art["state_dim"],
                 "cross_machine": {pool: art["pools"][pool]["cross_machine"] for pool in ("P", "C")},
                 "identity": "task_uid -> task plan"},
                ["Not official LIBERO inits; orig_init_state_idx is null for every episode."])
    rec["integrity"] = _integrity(pools_dir if all(str(f).startswith(str(pools_dir)) for f in files) else repo_root, files)
    return rec


def amendment_record(artifact_dir: pathlib.Path, suite: str, repo_root: pathlib.Path) -> dict:
    files = sorted(p for p in artifact_dir.iterdir() if p.is_file() and p.suffix in (".json", ".sha256"))
    rec = _base(f"dispatch_surface/{suite}/budget_amendment", "cache_artifact",
                f"dispatch_surface / {suite} / budget-mixture amendment artifacts (cost map, outcome design, C roster, power record)",
                suite, artifact_dir, repo_root, "exp/dispatch_surface/analysis/budget_cost_map.py + budget_outcome_design.py + confirmation_power_mc.py",
                ["exp/dispatch_surface/seal_confirmation.py", "exp/dispatch_surface/analysis/confirmation_analyzer.py"],
                {"files": [f.name for f in files], "estimator": "budget_mixture_v1", "posthoc_design_amendment": True},
                ["development-only; Phase 0 exact-cost artifacts are kept unchanged alongside."])
    rec["integrity"] = _integrity(artifact_dir, files)
    return rec


def write(rec: dict, records_dir: pathlib.Path = RECORDS_DIR) -> pathlib.Path:
    from exp.data_authority.registry import validate_record

    problems = validate_record(rec) if callable(validate_record) else []
    if problems:
        raise SystemExit(f"record invalid: {problems}")
    name = rec["dataset_id"].replace("/", "__") + ".json"
    path = records_dir / name
    if path.exists():
        raise SystemExit(f"{path} exists; records are immutable")
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tgrid")
    a.add_argument("--package-dir", required=True)
    b = sub.add_parser("fresh_pools")
    b.add_argument("--pools-dir", required=True)
    b.add_argument("--manifest", action="append", required=True)
    b.add_argument("--suite", required=True)
    b.add_argument("--validation", required=True, help="generate_fresh_inits validate artifact")
    c = sub.add_parser("amendment")
    c.add_argument("--artifact-dir", required=True)
    c.add_argument("--suite", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.repo_root).resolve()
    if args.cmd == "tgrid":
        rec = tgrid_record(pathlib.Path(args.package_dir).resolve(), root)
    elif args.cmd == "fresh_pools":
        rec = fresh_pools_record(pathlib.Path(args.pools_dir).resolve(), [pathlib.Path(m).resolve() for m in args.manifest], args.suite, root,
                                 pathlib.Path(args.validation).resolve())
    else:
        rec = amendment_record(pathlib.Path(args.artifact_dir).resolve(), args.suite, root)
    print(write(rec))


if __name__ == "__main__":
    main()
