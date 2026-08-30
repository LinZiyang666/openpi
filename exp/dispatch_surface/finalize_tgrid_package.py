"""Finalise the threshold-grid development rollout into an immutable package
(confirmation plan 3.2-f, G1R1-B4).

Run only after the full 29 x 300 grid is complete. The discipline validator
must accept the matrix / ledger / split manifest, the journal must hold
exactly one accepted record per (arm, cell) claimed by a registered launch,
and every executed YAML must match the matrix digest. The resulting
MANIFEST is the only thing ``budget_cost_map`` will read.

Usage:
  python -m exp.dispatch_surface.finalize_tgrid_package \
      --arm-matrix /tmp/dsp_shared/config/precheck_libero_10_exploratory_tgrid/arm_matrix_exploratory_tgrid.json \
      --journal /tmp/dsp_precheck/libero_10_exploratory_tgrid/journal.jsonl \
      --per-step /tmp/dsp_precheck/libero_10_exploratory_tgrid/per_step.jsonl \
      --launch-manifest /tmp/dsp_precheck/libero_10_exploratory_tgrid/per_step.jsonl.launch.json \
      --split-manifest exp/dispatch_surface/data/libero_10/init_pools/split_manifest.json \
      --out-dir exp/dispatch_surface/data/tgrid_dev/libero_10
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import shutil

from exp.dispatch_surface import tgrid_package as tpkg
from exp.dispatch_surface.analysis import phase0_discipline
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, official_test_inits


def finalize(arm_matrix: str, journal: str, per_step: str, launch_manifest: str,
             split_manifest: str, out_dir: str, *, trials: int = FORMAL_TRIALS,
             rev1_manifest_path: str | None = None) -> pathlib.Path:
    ctx = phase0_discipline.validate_tgrid(arm_matrix, launch_manifest, split_manifest, trials=trials,
                                           rev1_manifest_path=rev1_manifest_path)
    if not ctx["roster_complete"]:
        raise SystemExit("threshold-grid ledger did not execute the full grid; refusing to finalise")
    matrix = json.loads(pathlib.Path(arm_matrix).read_text())
    ledger = json.loads(pathlib.Path(launch_manifest).read_text())
    officials = official_test_inits(split_manifest, trials)
    grid = {(t, i) for t in officials for i in range(len(officials[t]))}
    arms = tpkg.grid_arms()
    claimed = tpkg.assert_grid_complete(pathlib.Path(journal), arms, grid, ctx["arm_matrix_sha256"], ledger)
    out = pathlib.Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a finalised package is immutable")
    (out / "yaml").mkdir(parents=True, exist_ok=True)
    members: dict[str, dict] = {}

    def add(role: str, src: pathlib.Path, member: str):
        dst = out / member
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        members[role] = {"member": member, "sha256": tpkg.file_sha256(dst)}

    add("matrix", pathlib.Path(arm_matrix), "arm_matrix_exploratory_tgrid.json")
    add("ledger", pathlib.Path(launch_manifest), "per_step.jsonl.launch.json")
    add("journal", pathlib.Path(journal), "journal.jsonl")
    add("per_step", pathlib.Path(per_step), "per_step.jsonl")
    add("split_manifest", pathlib.Path(split_manifest), "split_manifest.json")
    add("roster_spec", pathlib.Path(matrix["tgrid_roster_spec_path"]), "tgrid_roster_spec.json")
    for arm, path in matrix["arms"].items():
        add(f"yaml.{arm}", pathlib.Path(path), f"yaml/{arm}.yaml")
    launch_meta = {
        "run_ids": ctx["launch_run_ids"],
        "batches": len(ledger.get("launches") or []),
        "executed_arms_by_run": ctx["executed_arms_by_run"],
        "contract_binding": (ledger["launches"][0].get("contract_binding") or {}),
        "policy_fingerprint": ctx["policy_fingerprint"],
        "aprime_content_sha256": ctx["aprime_content_sha256"],
        "cells_per_arm": {arm: len(cells) for arm, cells in claimed.items()},
        "cost_model_digest": ctx["cost_model_digest"],
        "estimator_version": ctx["estimator_version"],
    }
    lm = out / "launch_meta.json"
    lm.write_text(json.dumps(launch_meta, indent=2, sort_keys=True))
    members["launch_meta"] = {"member": "launch_meta.json", "sha256": tpkg.file_sha256(lm)}
    manifest = {
        "schema": tpkg.PACKAGE_SCHEMA,
        "package": tpkg.PACKAGE_KIND,
        "suite": ctx["suite"],
        "layer": matrix["layer"],
        "protocol": matrix["protocol"],
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "arms": arms,
        "cells_per_arm": trials * len(officials),
        "run_ids": ctx["launch_run_ids"],
        "members": members,
    }
    mp = out / tpkg.MANIFEST_NAME
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tpkg.verify_package(mp)
    return mp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True)
    ap.add_argument("--launch-manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rev1-package-manifest", default="", help="local copy of the Rev 1 package the matrix bound")
    args = ap.parse_args()
    mp = finalize(args.arm_matrix, args.journal, args.per_step, args.launch_manifest,
                  args.split_manifest, args.out_dir, rev1_manifest_path=args.rev1_package_manifest or None)
    print(f"tgrid package finalised: {mp} sha256={tpkg.file_sha256(mp)}")


if __name__ == "__main__":
    main()
