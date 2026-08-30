"""Immutable finalization package for the dense threshold-grid development
rollout (confirmation plan 3.2-f, G1R1-B4).

The grid rollout executes under ``/tmp`` like every precheck; nothing
downstream may read those scattered files. ``finalize_tgrid_package`` copies
the matrix, the complete ledger, every executed YAML, the split manifest,
journal, per-step rows and launch metadata into one directory with a
MANIFEST (role -> member + SHA). ``budget_cost_map`` resolves members by role
only, and ``verify_package`` refuses drift, incomplete grids or foreign
rows. Mirrors ``rev1_package`` deliberately without touching it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from exp.dispatch_surface.phase0_roster import LAYER_TGRID, PROTOCOL_TGRID, tgrid_cells, tgrid_arm_id

MANIFEST_NAME = "MANIFEST.json"
PACKAGE_SCHEMA = 1
PACKAGE_KIND = "tgrid_dev"
FIXED_ROLES = ("matrix", "ledger", "journal", "per_step", "split_manifest", "launch_meta", "roster_spec")


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def grid_arms() -> list[str]:
    return [tgrid_arm_id(fh, ws) for fh, ws in tgrid_cells()]


def required_roles() -> tuple[str, ...]:
    return FIXED_ROLES + tuple(f"yaml.{arm}" for arm in grid_arms())


def load_manifest(path) -> tuple[dict, pathlib.Path, str]:
    mp = pathlib.Path(path)
    if not mp.is_file():
        raise SystemExit(f"tgrid package manifest missing: {mp}")
    manifest = json.loads(mp.read_text())
    if manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("package") != PACKAGE_KIND \
            or not isinstance(manifest.get("members"), dict):
        raise SystemExit(f"{mp}: not a tgrid development package manifest")
    missing = [r for r in required_roles() if r not in manifest["members"]]
    if missing:
        raise SystemExit(f"{mp}: package lacks roles {missing[:5]}...")
    return manifest, mp.parent, file_sha256(mp)


def member_path(manifest: dict, package_dir: pathlib.Path, role: str) -> pathlib.Path:
    entry = manifest["members"].get(role)
    if entry is None:
        raise SystemExit(f"tgrid package has no role {role!r}")
    member = entry.get("member")
    if not isinstance(member, str) or member.startswith("/") or ".." in member.split("/"):
        raise SystemExit(f"tgrid package role {role!r} has a non package-relative member {member!r}")
    return package_dir / member


def member_sha(manifest: dict, role: str) -> str:
    return str(manifest["members"][role]["sha256"])


def verify_member(manifest: dict, package_dir: pathlib.Path, role: str) -> pathlib.Path:
    path = member_path(manifest, package_dir, role)
    if not path.is_file():
        raise SystemExit(f"tgrid package member for {role!r} missing: {path}")
    got = file_sha256(path)
    if got != member_sha(manifest, role):
        raise SystemExit(f"tgrid package member for {role!r} content-drifted ({got[:12]}...)")
    return path


def load_json_member(manifest: dict, package_dir: pathlib.Path, role: str) -> dict:
    return json.loads(verify_member(manifest, package_dir, role).read_text())


def assert_grid_complete(journal_path: pathlib.Path, arms: list[str], grid: set[tuple[int, int]],
                         matrix_sha: str, ledger: dict) -> dict[str, dict]:
    """Exactly one accepted eval record per (arm, cell); every accepted run id
    is a ledger launch that executed that arm and was bound to this matrix.
    Returns arm -> {cell: run_id}. Reads no outcome field."""
    from exp.dispatch_surface.analysis.precheck_io import parse_task_uid

    launches = ledger.get("launches") or []
    executed: dict[str, set[str]] = {}
    for launch in launches:
        if launch.get("arm_matrix_sha256") != matrix_sha:
            raise SystemExit("ledger contains a launch bound to a different matrix")
        executed[str(launch["run_id"])] = set(launch.get("executed_arms") or [])
    seen: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in seen:
            raise SystemExit(f"journal carries a row for arm {arm!r} outside the grid roster")
        if row.get("accepted") is not True:
            continue
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit("journal yaml_id disagrees with task_uid")
        key = (task, subset)
        if key not in grid:
            raise SystemExit(f"arm {arm}: accepted cell {key} is off-grid")
        if key in seen[arm]:
            raise SystemExit(f"arm {arm}: duplicate accepted record for cell {key}")
        run_id = row.get("run_id")
        if run_id not in executed or arm not in executed[run_id]:
            raise SystemExit(f"arm {arm} cell {key}: accepted under an unregistered run {run_id!r}")
        seen[arm][key] = run_id
    for a in arms:
        if set(seen[a]) != grid:
            raise SystemExit(f"arm {a}: {len(grid - set(seen[a]))} cells missing — grid incomplete")
    return seen


def verify_package(manifest_path) -> dict:
    """Every member SHA, matrix/ledger/YAML cross-bindings and grid completeness."""
    from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, official_test_inits, validate_tgrid_arms

    manifest, pkg, _ = load_manifest(manifest_path)
    for role in required_roles():
        verify_member(manifest, pkg, role)
    matrix = load_json_member(manifest, pkg, "matrix")
    if matrix.get("layer") != LAYER_TGRID or matrix.get("protocol") != PROTOCOL_TGRID:
        raise SystemExit("package matrix is not a threshold-grid development matrix")
    arms = grid_arms()
    if sorted(matrix.get("arms") or {}) != sorted(arms):
        raise SystemExit("package matrix roster != frozen threshold grid")
    for arm in arms:
        if (matrix.get("arm_yaml_sha256") or {}).get(arm) != member_sha(manifest, f"yaml.{arm}"):
            raise SystemExit(f"matrix arm_yaml_sha256[{arm}] != package yaml member")
    # the package's own YAML members must be valid grid arms for the matrix pairs
    validate_tgrid_arms({arm: str(member_path(manifest, pkg, f"yaml.{arm}")) for arm in arms}, matrix)
    ledger = load_json_member(manifest, pkg, "ledger")
    matrix_sha = member_sha(manifest, "matrix")
    for launch in ledger.get("launches") or []:
        if launch.get("arm_matrix_sha256") != matrix_sha:
            raise SystemExit("ledger launch bound to a different matrix than the package member")
    if manifest.get("suite") != matrix.get("suite"):
        raise SystemExit("package suite != matrix suite")
    officials = official_test_inits(str(verify_member(manifest, pkg, "split_manifest")), FORMAL_TRIALS)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    assert_grid_complete(verify_member(manifest, pkg, "journal"), arms, grid, matrix_sha, ledger)
    return manifest
