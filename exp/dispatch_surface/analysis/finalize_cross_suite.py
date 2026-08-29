"""Cross-suite finalizer: the confirmatory claim needs BOTH suites, all four Gates.

Section 7.4 makes the Rev 1 claim conditional on libero_10 AND libero_spatial
each clearing Gate 1 and Gate 2. Nothing in the per-suite analyzer can enforce
that -- it only ever sees one suite -- so a single passing suite could be
written up as a general win. This tool is the mechanical block: it takes the two
formal per-suite verdicts and refuses anything that is not four-for-four.

Everything here fails closed. A missing suite, a duplicated suite, a verdict
produced under a different protocol or Gate schema, or an incomplete provenance
block is a refusal, not a downgrade -- a downgrade would still emit a result
that reads as an adjudication.

Usage:
  uv run python -m exp.dispatch_surface.analysis.finalize_cross_suite \
      --verdict exp/.../verdict_libero_10.json \
      --verdict exp/.../verdict_libero_spatial.json \
      --out exp/dispatch_surface/analysis/cross_suite_verdict.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

REQUIRED_SUITES = frozenset({"libero_10", "libero_spatial"})
REQUIRED_GATES = ("gate1", "gate2")
# Fields every per-suite verdict must carry for its provenance to be auditable.
REQUIRED_PROVENANCE = ("discipline", "point_estimates", "suite")
# Rev 1 Gate 2 encodes its cost side as a required saving; a non-negative
# cost_gate means the verdict came from the retired rule.
RETIRED_GATE2_MARKERS = ("gate2_compute_slack", "dc_upper_slack")
PROTOCOL = "dispatch_surface_rev1"
GATE_SCHEMA = {
    "gate1": {
        "d_sr_lower_quantile": 0.05,
        "d_c_upper_quantile": 0.95,
        "cost_gate": -0.05,
    },
    "gate2": {
        "d_sr_lower_quantile": 0.05,
        "d_c_upper_quantile": 0.95,
        "cost_gate": -0.05,
    },
}


def _load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def check_one(v: dict, where: str) -> str:
    """Validate a single per-suite verdict; return its suite."""
    for field in REQUIRED_PROVENANCE:
        if field not in v:
            raise SystemExit(f"{where}: missing required field {field!r}")
    suite = v["suite"]
    if suite not in REQUIRED_SUITES:
        raise SystemExit(f"{where}: suite {suite!r} not in {sorted(REQUIRED_SUITES)}")
    for marker in RETIRED_GATE2_MARKERS:
        if marker in v:
            raise SystemExit(f"{where}: carries retired Gate 2 field {marker!r}")
    if (v.get("protocol") != PROTOCOL or v.get("analysis_layer") != "primary"
            or v.get("confirmatory") is not True):
        raise SystemExit(f"{where}: verdict is not a primary confirmatory {PROTOCOL} result")
    if v.get("gate_schema") != GATE_SCHEMA:
        raise SystemExit(f"{where}: gate schema differs from the frozen Rev 1 schema")
    discipline = v["discipline"]
    if (not isinstance(discipline, dict) or discipline.get("suite") != suite
            or discipline.get("protocol") != PROTOCOL
            or discipline.get("layer") != "primary"):
        raise SystemExit(f"{where}: discipline protocol/layer/suite provenance is inconsistent")
    for field in ("arm_matrix_sha256", "split_manifest_sha256", "artifact_sha256",
                  "fit_record_sha256", "launch_run_ids"):
        if not discipline.get(field):
            raise SystemExit(f"{where}: discipline lacks nonempty provenance field {field}")

    g1 = v.get("gate1")
    if not isinstance(g1, dict) or not isinstance(g1.get("pass"), bool):
        raise SystemExit(f"{where}: gate1 has no boolean pass field")
    if (g1.get("d_sr_lower_quantile") != GATE_SCHEMA["gate1"]["d_sr_lower_quantile"]
            or g1.get("d_c_upper_quantile") != GATE_SCHEMA["gate1"]["d_c_upper_quantile"]
            or g1.get("cost_gate") != GATE_SCHEMA["gate1"]["cost_gate"]):
        raise SystemExit(f"{where}: gate1 block does not encode the frozen schema")
    g2 = v.get("gate2")
    if not g1["pass"]:
        if g2 is not None:
            raise SystemExit(f"{where}: gate2 was evaluated after Gate 1 failed")
        return suite
    if not isinstance(g2, dict) or not isinstance(g2.get("pass"), bool):
        raise SystemExit(f"{where}: Gate 1 passed but gate2 has no boolean pass field")
    if (g2.get("dsr_lower_quantile") != GATE_SCHEMA["gate2"]["d_sr_lower_quantile"]
            or g2.get("dc_upper_quantile") != GATE_SCHEMA["gate2"]["d_c_upper_quantile"]
            or g2.get("cost_gate") != GATE_SCHEMA["gate2"]["cost_gate"]):
        raise SystemExit(f"{where}: gate2 block does not encode the frozen schema")
    return suite


def finalize(verdicts: dict[str, dict]) -> dict:
    """Four-for-four or nothing."""
    if set(verdicts) != set(REQUIRED_SUITES):
        raise SystemExit(
            f"cross-suite claim needs exactly {sorted(REQUIRED_SUITES)}, got "
            f"{sorted(verdicts)}"
        )
    gates = {
        suite: {
            "gate1": v["gate1"]["pass"],
            "gate2": v.get("gate2", {}).get("pass") if v["gate1"]["pass"] else None,
        }
        for suite, v in verdicts.items()
    }
    all_pass = all(gs["gate1"] is True and gs["gate2"] is True for gs in gates.values())
    failed = sorted(
        f"{suite}.{g}" for suite, gs in gates.items() for g, ok in gs.items() if ok is False
    )
    not_evaluated = sorted(
        f"{suite}.{g}" for suite, gs in gates.items() for g, ok in gs.items() if ok is None
    )
    return {
        "suites": sorted(verdicts),
        "gates": gates,
        "n_gates": sum(ok is not None for gs in gates.values() for ok in gs.values()),
        "failed_gates": failed,
        "not_evaluated_fixed_sequence": not_evaluated,
        "cross_suite_confirmed": all_pass,
        "verdict": "cross_suite_confirmed" if all_pass else "suite_specific_only",
        "note": (
            "Both suites cleared Gate 1 and Gate 2."
            if all_pass else
            "Not all four Gates passed; only suite-specific evidence may be "
            f"reported. Failed: {failed}; not evaluated by fixed sequence: {not_evaluated}."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", action="append", required=True,
                    help="per-suite verdict json; pass exactly twice")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.verdict) != len(REQUIRED_SUITES):
        raise SystemExit(
            f"expected exactly {len(REQUIRED_SUITES)} --verdict paths, got {len(args.verdict)}"
        )
    verdicts: dict[str, dict] = {}
    for path in args.verdict:
        v = _load(path)
        suite = check_one(v, path)
        if suite in verdicts:
            raise SystemExit(f"suite {suite!r} supplied twice; the two must be distinct")
        verdicts[suite] = v

    result = finalize(verdicts)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["cross_suite_confirmed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
