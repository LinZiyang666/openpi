"""Emit the conductor arm matrix for the cache-size ablation.

Consumed by ``run_size_eval.py`` -- **not** by the family's executor-substitution
driver, which rejects every arm here (it requires a ``routing`` section on any
non-baseline arm). All arms are pure-cache, so none needs a sidecar: the
``sidecar: null`` column stays null throughout, which is itself worth asserting.
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

from exp.ablation_study.cache_size.emit_size_yamls import (
    SENSITIVITY_TIERS,
    TIERS,
    arm_name,
)


def build_matrix(suite: str, yaml_dir: str, *, with_sensitivity: bool = True) -> dict:
    arms = []
    for tier in TIERS:
        name = arm_name(suite, tier)
        arms.append({"arm": name, "yaml": f"{yaml_dir.rstrip('/')}/{name}.yaml", "sidecar": None})
        if with_sensitivity and tier in SENSITIVITY_TIERS:
            sname = arm_name(suite, tier, recal=True)
            arms.append({"arm": sname, "yaml": f"{yaml_dir.rstrip('/')}/{sname}.yaml",
                         "sidecar": None})
    return {
        "suite": suite,
        "derived_from": f"exp/ablation_study/config/common/{suite}_baseline.yaml",
        "arms": arms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--yaml-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    matrix = build_matrix(args.suite, args.yaml_dir, with_sensitivity=not args.no_sensitivity)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(matrix, sort_keys=False))
    print(f"{out}  ({len(matrix['arms'])} arms)")


if __name__ == "__main__":
    main()
