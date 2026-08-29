"""Rev 1 discipline package: role -> member mapping for the archived artifacts.

The Rev 1 arm matrix and fit records name their inputs by ABSOLUTE paths on
the machines that ran them (``/tmp/dsp_shared/...``). Those bytes are frozen
(their SHA is the verdict's authority) and may not be rewritten, so a
migrated copy cannot be resolved by path. The package MANIFEST fixes that:
every logical role maps to a package-relative member with its SHA, and every
Phase 0 consumer resolves members by role only (G1R2-B3). A consumer that
still opens a historical absolute path is a bug the migration test catches.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

MANIFEST_NAME = "MANIFEST.json"
PACKAGE_SCHEMA = 1

SURFACE_ROLES = ("artifact.dsp_sv", "artifact.dsp_s0", "artifact.dsp_sv_minus")
FIT_ROLES = ("fit.sv", "fit.s0")
YAML_ARMS = ("dsp_s0", "dsp_sv", "dsp_sv_minus", "dsp_t_fh30_ws20",
             "dsp_t_fh50_ws20", "dsp_t_fh70_ws10")
REQUIRED_ROLES = (
    ("matrix",) + FIT_ROLES + SURFACE_ROLES
    + tuple(f"yaml.{arm}" for arm in YAML_ARMS)
    + ("d0", "rebuild", "split_manifest", "ledger", "verdict", "journal", "per_step")
)
#: Roles whose SHA the Rev 1 matrix DECLARES under an absolute path; the
#: package validator ties the member SHA to that declaration.
MATRIX_DECLARED = {
    "artifact.dsp_sv": ("artifact_sha256", "dsp_sv"),
    "artifact.dsp_s0": ("artifact_sha256", "dsp_s0"),
    "artifact.dsp_sv_minus": ("artifact_sha256", "dsp_sv_minus"),
    "fit.sv": ("fit_record_sha256", "sv"),
    "fit.s0": ("fit_record_sha256", "s0"),
    **{f"yaml.{arm}": ("arm_yaml_sha256", arm) for arm in YAML_ARMS},
}


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | pathlib.Path) -> tuple[dict, pathlib.Path, str]:
    """Return (manifest, package_dir, manifest_sha256); refuse a malformed one."""
    mp = pathlib.Path(path)
    if not mp.is_file():
        raise SystemExit(f"rev1 package manifest missing: {mp}")
    manifest = json.loads(mp.read_text())
    if manifest.get("schema") != PACKAGE_SCHEMA or not isinstance(manifest.get("members"), dict):
        raise SystemExit(f"{mp}: not a rev1 discipline package manifest")
    missing = [r for r in REQUIRED_ROLES if r not in manifest["members"]]
    if missing:
        raise SystemExit(f"{mp}: package lacks roles {missing}")
    return manifest, mp.parent, file_sha256(mp)


def member_path(manifest: dict, package_dir: pathlib.Path, role: str) -> pathlib.Path:
    entry = manifest["members"].get(role)
    if entry is None:
        raise SystemExit(f"rev1 package has no role {role!r}")
    member = entry.get("member")
    if not isinstance(member, str) or member.startswith("/") or ".." in member.split("/"):
        raise SystemExit(f"rev1 package role {role!r} has a non package-relative member {member!r}")
    return package_dir / member


def member_sha(manifest: dict, role: str) -> str:
    return str(manifest["members"][role]["sha256"])


def verify_member(manifest: dict, package_dir: pathlib.Path, role: str) -> pathlib.Path:
    """Resolve a role and refuse it unless the bytes match the manifest."""
    path = member_path(manifest, package_dir, role)
    if not path.is_file():
        raise SystemExit(f"rev1 package member for {role!r} missing: {path}")
    got = file_sha256(path)
    if got != member_sha(manifest, role):
        raise SystemExit(f"rev1 package member for {role!r} content-drifted ({got[:12]}...)")
    return path


def load_json_member(manifest: dict, package_dir: pathlib.Path, role: str) -> dict:
    return json.loads(verify_member(manifest, package_dir, role).read_text())


def verify_package(manifest_path: str | pathlib.Path) -> dict:
    """Full package check: every member's SHA, the matrix's declared SHAs,
    the verdict's discipline SHAs and the cost authority. Returns the manifest."""
    from exp.dispatch_surface.analysis.analytic_cost import assert_unit_costs_match

    manifest, pkg, _ = load_manifest(manifest_path)
    for role in REQUIRED_ROLES:
        verify_member(manifest, pkg, role)
    matrix = load_json_member(manifest, pkg, "matrix")
    for role, (field, key) in MATRIX_DECLARED.items():
        declared = (matrix.get(field) or {}).get(key)
        if declared != member_sha(manifest, role):
            raise SystemExit(
                f"matrix declares {field}[{key}]={str(declared)[:12]}... but package "
                f"member {role} is {member_sha(manifest, role)[:12]}..."
            )
    verdict = load_json_member(manifest, pkg, "verdict")
    disc = verdict.get("discipline") or {}
    if disc.get("arm_matrix_sha256") != member_sha(manifest, "matrix"):
        raise SystemExit("verdict discipline.arm_matrix_sha256 != package matrix member")
    for arm in ("dsp_sv", "dsp_s0", "dsp_sv_minus"):
        if (disc.get("artifact_sha256") or {}).get(arm) != member_sha(manifest, f"artifact.{arm}"):
            raise SystemExit(f"verdict discipline.artifact_sha256[{arm}] != package member")
    for name in ("sv", "s0"):
        if (disc.get("fit_record_sha256") or {}).get(name) != member_sha(manifest, f"fit.{name}"):
            raise SystemExit(f"verdict discipline.fit_record_sha256[{name}] != package member")
    if disc.get("split_manifest_sha256") != member_sha(manifest, "split_manifest"):
        raise SystemExit("verdict discipline.split_manifest_sha256 != package member")
    cost_inputs = disc.get("cost_inputs") or {}
    if cost_inputs.get("per_step_sha256") != member_sha(manifest, "per_step"):
        raise SystemExit("verdict discipline.cost_inputs.per_step_sha256 != package per_step member")
    assert_unit_costs_match(cost_inputs.get("unit_cost_ms"), what="archived verdict")
    if manifest.get("suite") != disc.get("suite") or manifest.get("suite") != matrix.get("suite", manifest.get("suite")):
        raise SystemExit("package suite disagrees with the verdict")
    return manifest
