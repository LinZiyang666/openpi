"""Emit the RIT-Pareto arm yamls and matrices from an export record.

Every arm is the GTP server template of its suite with exactly three things
replaced -- the same three the GTP emitter touches, so a difference between
this line and the GST frontier can only come from one of them:

*   ``backend.in_memory.preload_path`` -- the calibrated library (server path);
*   ``checkpoints.cp1.judge`` -- ``dispatch_surface`` with one IR-addressed
    artifact (three-way verdict: FULL_HIT / WARM_START at t=0.3 / MISS);
*   ``checkpoints.cp1.gate`` -- ``always_search`` (no-gate layer) or the N4
    hysteresis gate at the export record's ``gate_theta`` with the GTP
    constants j=3 / probe_interval=3 / L=6 (H-gate layer).

One matrix per layer (``arm_matrix_ng.yaml`` / ``arm_matrix_hg.yaml``) in the
``run_gtp`` schema ``{arms: [{arm, yaml, suite}]}``. Artifact and yaml paths
are absolute: the runner host validates them at launch, so the directory must
be replicated at the same path on the client.

Usage:
  uv run python -m exp.rit_pareto.emit_arms \
      --export-record <out-dir>/export_record.json --suite libero_spatial \
      --library-pkl <server path> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib

import yaml

from exp.dispatch_surface.emit_precheck_yamls import LAYER_PRIMARY, LAYER_SECONDARY, gate_section
from exp.gate_threshold_pareto import libraries as libs
from openpi.cache.config import load_cache_config

SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}
LAYER_TAG = {LAYER_PRIMARY: "ng", LAYER_SECONDARY: "hg"}
LAYERS = (LAYER_PRIMARY, LAYER_SECONDARY)


def _sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_template(suite: str) -> dict:
    path = libs.REPO_ROOT / libs.TEMPLATE[suite]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_arm(template: dict, *, artifact_path: str, preload_path: str, theta: float, layer: str) -> dict:
    """Template + the three replaced sections. ``theta`` is ignored on the primary layer."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    doc = copy.deepcopy(template)
    cp1 = doc["checkpoints"]["cp1"]
    cp1["judge"] = {"type": "dispatch_surface", "surface_artifact_path": artifact_path}
    cp1["gate"] = gate_section(layer, float(theta))
    doc["backend"]["in_memory"]["preload_path"] = preload_path
    doc["write_policy"] = {"type": "never"}
    return doc


def arm_id(suite: str, layer: str, name: str) -> str:
    return f"rit_{SUITE_TAG[suite]}_{LAYER_TAG[layer]}_{name}"


def emit(export_record_path: pathlib.Path, *, suite: str, preload_path: str, out_dir: pathlib.Path) -> dict:
    record = json.loads(export_record_path.read_text())
    if record.get("suite") != suite:
        raise SystemExit(f"export record is for {record.get('suite')!r}, emitting {suite!r}")
    if record.get("library_pkl") != preload_path:
        raise SystemExit(
            f"export record was fitted on {record.get('library_pkl')!r}; the arms would load "
            f"{preload_path!r} -- the contract in the artifacts would not match that file"
        )
    theta = float(record["gate_theta"])
    template = load_template(suite)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, dict] = {}
    emit_record = {
        "protocol": record["protocol"],
        "suite": suite,
        "export_record_path": str(export_record_path.resolve()),
        "export_record_sha256": _sha(export_record_path),
        "gate_theta": theta,
        "gate": gate_section(LAYER_SECONDARY, theta),
        "preload_path": preload_path,
        "template": str(libs.TEMPLATE[suite]),
        "layers": {},
    }
    for layer in LAYERS:
        tag = LAYER_TAG[layer]
        layer_dir = out_dir / tag
        layer_dir.mkdir(exist_ok=True)
        rows = []
        arms = {}
        for name, art in sorted(record["artifacts"].items(), key=lambda kv: kv[1]["target_ir"]):
            art_path = pathlib.Path(art["path"])
            if not art_path.is_file() or _sha(art_path) != art["output_sha256"]:
                raise SystemExit(f"artifact {name} missing or changed since export: {art_path}")
            arm = arm_id(suite, layer, name)
            doc = build_arm(template, artifact_path=str(art_path), preload_path=preload_path,
                            theta=theta, layer=layer)
            path = layer_dir / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            loaded = load_cache_config(str(path))  # strict schema + artifact existence
            cp1 = loaded.checkpoints["cp1"]
            if cp1.judge.type != "dispatch_surface" or cp1.judge.surface_artifact_path != str(art_path):
                raise SystemExit(f"{path}: judge did not round-trip")
            rows.append({"arm": arm, "yaml": str(path.resolve()), "suite": suite})
            arms[arm] = {"artifact": str(art_path), "artifact_sha256": art["output_sha256"],
                         "target_ir": art["target_ir"], "predicted_ir": art["predicted_ir"],
                         "yaml_sha256": _sha(path)}
        matrix = {"protocol": record["protocol"], "suite": suite, "layer": layer, "arms": rows}
        matrix_path = out_dir / f"arm_matrix_{tag}.yaml"
        matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
        matrices[layer] = matrix
        emit_record["layers"][layer] = {"matrix": str(matrix_path.resolve()), "arms": arms}
    (out_dir / "emit_record.json").write_text(json.dumps(emit_record, indent=2, sort_keys=True) + "\n")
    return emit_record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-record", required=True)
    ap.add_argument("--suite", required=True, choices=tuple(SUITE_TAG))
    ap.add_argument("--library-pkl", required=True, help="server-side library path (preload_path)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rec = emit(pathlib.Path(args.export_record), suite=args.suite, preload_path=args.library_pkl,
               out_dir=pathlib.Path(args.out_dir))
    for layer, info in rec["layers"].items():
        print(f"{layer}: {len(info['arms'])} arms -> {info['matrix']}")


if __name__ == "__main__":
    main()
