"""Archive one suite's Rev 1 primary discipline package (plan section 3.8-c).

Copies the frozen bytes of the arm matrix, the six arm YAMLs, the three
surface artifacts, the two fit records, the D0 record, the rebuild record,
the split manifest, the launch ledger, the verdict, the journal and the
per-step log into ``<out>/`` UNCHANGED, and writes a MANIFEST that maps each
logical role to its package-relative member and SHA. The MANIFEST is what
every Phase 0 consumer resolves through (``rev1_package``); the frozen files
keep their historical absolute paths inside and are never rewritten.

Inputs are given as local paths (pull them with ``tether pull`` first); the
script only validates and copies. It also emits the data_authority record
JSON for the package so the ledger can be updated in the same change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil

from exp.dispatch_surface import rev1_package as pkgmod

ROLE_ARGS = {
    "matrix": "--matrix",
    "fit.sv": "--fit-sv",
    "fit.s0": "--fit-s0",
    "artifact.dsp_sv": "--artifact-sv",
    "artifact.dsp_s0": "--artifact-s0",
    "artifact.dsp_sv_minus": "--artifact-sv-minus",
    "d0": "--d0",
    "rebuild": "--rebuild",
    "split_manifest": "--split-manifest",
    "ledger": "--ledger",
    "verdict": "--verdict",
    "journal": "--journal",
    "per_step": "--per-step",
}
MEMBER_NAME = {
    "matrix": "arm_matrix_primary.json",
    "fit.sv": "fit_record.json",
    "fit.s0": "fit_record_s_only.json",
    "artifact.dsp_sv": "surface_sv_primary.npz",
    "artifact.dsp_s0": "surface_s_only_primary.npz",
    "artifact.dsp_sv_minus": "surface_sv_minus.npz",
    "d0": "d0_record.json",
    "rebuild": "rebuild_record.json",
    "split_manifest": "split_manifest.json",
    "ledger": "per_step.jsonl.launch.json",
    "verdict": "verdict.json",
    "journal": "journal.jsonl",
    "per_step": "per_step.jsonl",
}


def build_package(suite: str, sources: dict[str, pathlib.Path], out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"package directory {out} must be empty")
    members = {}
    matrix = json.loads(sources["matrix"].read_text())
    for role, src in sources.items():
        if not src.is_file():
            raise SystemExit(f"{role}: source missing {src}")
        member = MEMBER_NAME[role]
        shutil.copyfile(src, out / member)
        entry = {"member": member, "sha256": pkgmod.file_sha256(out / member),
                 "size_bytes": (out / member).stat().st_size}
        declared = pkgmod.MATRIX_DECLARED.get(role)
        if declared is not None:
            field, key = declared
            entry["declared_in_matrix"] = {"field": field, "key": key,
                                           "sha256": (matrix.get(field) or {}).get(key)}
        members[role] = entry
    yaml_dir = sources["matrix"].parent
    for arm in pkgmod.YAML_ARMS:
        src = yaml_dir / f"{arm}.yaml"
        if not src.is_file():
            raise SystemExit(f"yaml for {arm} missing next to the matrix: {src}")
        member = f"{arm}.yaml"
        shutil.copyfile(src, out / member)
        members[f"yaml.{arm}"] = {
            "member": member, "sha256": pkgmod.file_sha256(out / member),
            "size_bytes": (out / member).stat().st_size,
            "declared_in_matrix": {"field": "arm_yaml_sha256", "key": arm,
                                   "sha256": (matrix.get("arm_yaml_sha256") or {}).get(arm)},
        }
    manifest = {
        "schema": pkgmod.PACKAGE_SCHEMA,
        "package": "dispatch_surface_rev1_primary_discipline",
        "suite": suite,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "members": members,
        "note": "Frozen Rev 1 bytes; absolute paths inside the members are historical and "
                "must be resolved by role through this manifest only.",
    }
    (out / pkgmod.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    pkgmod.verify_package(out / pkgmod.MANIFEST_NAME)
    return manifest


def data_authority_record(suite: str, out: pathlib.Path, manifest: dict, repo_root: pathlib.Path) -> dict:
    members = sorted(
        [[v["member"], v["size_bytes"], v["sha256"]] for v in manifest["members"].values()]
        + [[pkgmod.MANIFEST_NAME, (out / pkgmod.MANIFEST_NAME).stat().st_size,
            pkgmod.file_sha256(out / pkgmod.MANIFEST_NAME)]]
    )
    import hashlib
    rollup = hashlib.sha256("\n".join(f"{m[0]} {m[2]}" for m in members).encode()).hexdigest()
    return {
        "schema_version": 1,
        "dataset_id": f"dispatch_surface/{suite}/rev1_discipline",
        "kind": "cache_artifact",
        "title": f"dispatch_surface / {suite} / Rev 1 primary discipline package (frozen matrix, "
                 "yamls, artifacts, fit records, D0, rebuild, split manifest, ledger, verdict, journal, per_step)",
        "experiment": "exp/dispatch_surface",
        "suite": suite,
        "status": "authoritative",
        "authority": {"node": "local", "path": str(out.relative_to(repo_root)), "access": "local"},
        "integrity": {"file_count": len(members), "size_bytes": int(sum(m[1] for m in members)),
                      "sha256": rollup, "members": members},
        "content": {"manifest": pkgmod.MANIFEST_NAME, "roles": sorted(manifest["members"]),
                    "resolution": "by role through MANIFEST.json only"},
        "provenance": {
            "produced_by": "exp/dispatch_surface/archive_rev1_discipline.py",
            "measured_at": manifest["created_at"],
            "measured_by": "Execution Authority (Rev 2 Phase 0)",
            "source_machines": ["weilandserver:/tmp/dsp_shared", "timan107:/tmp/dsp_precheck"],
        },
        "consumers": ["exp/dispatch_surface/export_exploratory_surface.py",
                      "exp/dispatch_surface/emit_precheck_yamls.py (exploratory layer)",
                      "exp/dispatch_surface/run_precheck.py (exploratory layer)",
                      "exp/dispatch_surface/analysis/cost_map.py"],
        "caveats": ["Member files keep historical absolute paths (/tmp/dsp_shared, /data/openpi_dispatch); "
                    "they are frozen bytes and must never be rewritten."],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=["libero_10", "libero_spatial"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-out", default="", help="where to write the data_authority record JSON")
    for role, flag in ROLE_ARGS.items():
        ap.add_argument(flag, required=True, dest=role.replace(".", "_"))
    args = ap.parse_args()
    sources = {role: pathlib.Path(getattr(args, role.replace(".", "_"))) for role in ROLE_ARGS}
    out = pathlib.Path(args.out)
    manifest = build_package(args.suite, sources, out)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    record = data_authority_record(args.suite, out.resolve(), manifest, repo_root)
    if args.record_out:
        pathlib.Path(args.record_out).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"suite": args.suite, "members": len(manifest["members"]),
                      "manifest_sha256": pkgmod.file_sha256(out / pkgmod.MANIFEST_NAME)}, indent=2))


if __name__ == "__main__":
    main()
