"""Re-run the production normalizer calibration at the S1 and S6 endpoints.

The sensitivity arms exist because the production ``mu``/``sigma`` were fit on a
50-init library (roughly tier S3), so both ends of the size axis sit off that
anchor -- S6 an order of magnitude above it, S1 an order of magnitude below.
The small-library end is the more dangerous one: the vision fields' sigma is
~7e-3, and ``ZScoreNormalizer``'s ``0.5*(tanh(z)+1)`` saturates a few sigma out,
so with distant nearest neighbours both vision terms flatten toward 0 for every
candidate and the ranking is effectively handed to ``robot_state``. That is the
retrieval rule itself changing with size -- precisely the confound this
experiment must exclude.

Calibration is the **production script, run unmodified** on the tier's own pkl.
``calibrate_score_normalizers`` already does leave-one-episode-out internally
(every entry serves as query, scored against entries outside its own
trajectory), so it needs no external query pool. Substituting one -- e.g.
rollouts held outside the library -- would move two variables at once (library
size *and* query distribution) and the arm would attribute nothing.

The calibrator consumes a *directory* of artifacts and emits JSON keyed by pkl
stem, so this driver stages one symlink per tier and reshapes the result into
the ``fields`` mapping that ``emit_size_yamls`` splices into an arm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import yaml

CALIBRATOR = "exp/common/calibrate_score_normalizers.py"
SENSITIVITY_TIERS = ("S1", "S6")


def to_arm_fields(calibration: dict, pkl_stem: str, *, baseline_methods: dict | None = None) -> dict:
    """Reshape the calibrator's JSON into ``{field: {method, params}}``.

    The production shape is::

        {<pkl stem>: {"builder_type": ..., "vector_dims": ...,
                      "fields": {<field>: {"sim_type": ..., "shortlist": [...],
                                           "selected": {"method": ..., "params": ...}}}}}

    Note the ``fields`` wrapper -- reading ``selected`` off the stem level walks
    ``builder_type`` / ``vector_dims`` / ``fields`` instead and finds nothing.

    ``baseline_methods`` enforces the frozen sensitivity-arm contract: those arms
    may differ from their main twin in ``params`` **only**. If the calibrator
    picks a different normalizer *method* for some field, splicing just the
    params would silently pair one method's parameters with another's -- so that
    is a hard failure, not something to paper over.
    """
    if pkl_stem not in calibration:
        raise KeyError(
            f"calibration output has no entry for {pkl_stem!r}; got {sorted(calibration)}"
        )
    entry = calibration[pkl_stem]
    if "fields" not in entry:
        raise KeyError(
            f"{pkl_stem}: calibration entry has no 'fields' block; got {sorted(entry)}. "
            "Expected the production calibrate_score_normalizers shape."
        )

    out: dict[str, dict] = {}
    method_changes: list[str] = []
    for field, spec in entry["fields"].items():
        selected = spec.get("selected") if isinstance(spec, dict) else None
        if not selected:
            continue
        method, params = selected["method"], selected["params"]
        if baseline_methods is not None and field in baseline_methods:
            if method != baseline_methods[field]:
                method_changes.append(
                    f"{field}: baseline={baseline_methods[field]!r} recalibrated={method!r}"
                )
        out[field] = {"method": method, "params": params}

    if method_changes:
        raise ValueError(
            "recalibration selected a different normalizer method for "
            + "; ".join(method_changes)
            + ". The sensitivity arm contract allows params-only deltas, so splicing "
            "these params into the baseline method would be incoherent."
        )
    if not out:
        raise ValueError(f"{pkl_stem}: calibrator selected no fields")
    return out


def baseline_methods_from(baseline_path: str | pathlib.Path) -> dict[str, str]:
    """{field: normalizer method} from a baseline arm yaml."""
    cfg = yaml.safe_load(pathlib.Path(baseline_path).read_text())
    fields = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    return {f: spec["method"] for f, spec in fields.items() if "method" in spec}


def calibrate_tier(
    *,
    pkl: pathlib.Path,
    out_yaml: pathlib.Path,
    extra: list[str],
    baseline_methods: dict | None = None,
    python: str = sys.executable,
) -> dict:
    """Run the production calibrator against exactly one artifact."""
    with tempfile.TemporaryDirectory() as staging:
        link = pathlib.Path(staging) / pkl.name
        link.symlink_to(pkl.resolve())
        raw_out = pathlib.Path(staging) / "calibration.json"
        cmd = [
            python, CALIBRATOR,
            "--artifact-dir", staging,
            "--output", str(raw_out),
            *extra,
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        calibration = json.loads(raw_out.read_text())

    # Check the params-only contract here, at calibration time, rather than
    # letting the mismatch surface later during yaml emission.
    fields = to_arm_fields(calibration, pkl.stem, baseline_methods=baseline_methods)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump({"source_pkl": str(pkl), "fields": fields},
                                       sort_keys=False))
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tiers", default=",".join(SENSITIVITY_TIERS))
    ap.add_argument("--calibrator-arg", action="append", default=[],
                    help="extra flag forwarded verbatim to the production calibrator")
    ap.add_argument("--baseline", default=None,
                    help="baseline arm yaml; enables the params-only method check")
    args = ap.parse_args()

    methods = baseline_methods_from(args.baseline) if args.baseline else None

    for tier in args.tiers.split(","):
        pkl = pathlib.Path(args.artifact_dir) / f"cache_size_{args.suite}_{tier}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(pkl)
        out = pathlib.Path(args.out_dir) / f"recal_norm_{args.suite}_{tier}.yaml"
        fields = calibrate_tier(pkl=pkl, out_yaml=out, extra=args.calibrator_arg,
                                baseline_methods=methods)
        print(f"  {tier}: {out}  ({len(fields)} fields)")


if __name__ == "__main__":
    main()
