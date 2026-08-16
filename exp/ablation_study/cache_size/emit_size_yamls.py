"""Derive the 16 evaluation arms from the production baseline config.

12 main arms (6 tiers x 2 suites) plus 4 sensitivity arms (S1 and S6 endpoints
x 2 suites). Each main arm differs from the baseline in exactly two places --
the verdict becomes ``always_hit`` and the backend points at that tier's pkl --
so the retrieval and gate chain stays byte-identical across the size axis.

Sensitivity arms differ from their corresponding main arm in exactly one place:
the per-field normalizer parameters. They deliberately **reuse the main arm's
pkl** (normalization is a scoring transform, not library content), which makes
the comparison a strict single-variable control.
"""

from __future__ import annotations

import argparse
import copy
import pathlib

import yaml

TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")
SENSITIVITY_TIERS = ("S1", "S6")


def make_main_arm(baseline: dict, preload_path: str) -> dict:
    """Baseline -> pure-cache arm: always_hit verdict, tier-specific library."""
    cfg = copy.deepcopy(baseline)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["judge"] = {"type": "always_hit"}
    cp1["gate"] = {"type": "always_search"}
    cfg["backend"]["in_memory"]["preload_path"] = preload_path
    return cfg


def make_sensitivity_arm(main_arm: dict, recal_fields: dict) -> dict:
    """Main arm -> recalibrated twin: only ``score_normalization.*.params`` change.

    ``recal_fields`` carries ``{field: {"method": ..., "params": ...}}`` as
    produced by ``run_recal.to_arm_fields``. Only ``params`` is spliced; the
    ``method`` is cross-checked against the arm being modified, because pairing
    one normalizer's parameters with another's method would be silently wrong
    while still loading cleanly.
    """
    cfg = copy.deepcopy(main_arm)
    fields = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    for field, spec in recal_fields.items():
        if field not in fields:
            raise KeyError(
                f"recalibrated field {field!r} absent from the baseline normalizer block; "
                f"known fields: {sorted(fields)}"
            )
        if not isinstance(spec, dict) or "params" not in spec:
            raise ValueError(
                f"{field}: expected {{'method': ..., 'params': ...}} from to_arm_fields, "
                f"got {spec!r} -- splicing this whole object into 'params' would nest wrongly"
            )
        method = spec.get("method")
        if method is not None and method != fields[field].get("method"):
            raise ValueError(
                f"{field}: recalibrated method {method!r} differs from the arm's "
                f"{fields[field].get('method')!r}; sensitivity arms are params-only"
            )
        fields[field]["params"] = copy.deepcopy(spec["params"])
    return cfg


def arm_name(suite: str, tier: str, *, recal: bool = False) -> str:
    return f"cache_size_{suite}_{tier}{'_recal' if recal else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="config/common/<suite>_baseline.yaml")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--artifact-dir", required=True,
                    help="directory holding cache_size_<suite>_<tier>.pkl")
    ap.add_argument("--recal-dir", default=None,
                    help="directory holding recal_norm_<suite>_<tier>.yaml (S1/S6)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    baseline = yaml.safe_load(pathlib.Path(args.baseline).read_text())
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for tier in TIERS:
        pkl = f"{args.artifact_dir.rstrip('/')}/cache_size_{args.suite}_{tier}.pkl"
        main_cfg = make_main_arm(baseline, pkl)
        name = arm_name(args.suite, tier)
        (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(main_cfg, sort_keys=False))
        written.append(name)

        if args.recal_dir and tier in SENSITIVITY_TIERS:
            recal_file = pathlib.Path(args.recal_dir) / f"recal_norm_{args.suite}_{tier}.yaml"
            if not recal_file.exists():
                raise FileNotFoundError(f"sensitivity arm needs {recal_file}")
            recal = yaml.safe_load(recal_file.read_text())
            sens_cfg = make_sensitivity_arm(main_cfg, recal["fields"])
            sname = arm_name(args.suite, tier, recal=True)
            (out_dir / f"{sname}.yaml").write_text(yaml.safe_dump(sens_cfg, sort_keys=False))
            written.append(sname)

    for n in written:
        print(n)
    print(f"{len(written)} arms -> {out_dir}")


if __name__ == "__main__":
    main()
