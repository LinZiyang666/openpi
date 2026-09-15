"""Validate the acyclic parity -> table -> calibration artifact lineage on CPU."""

from __future__ import annotations

import json
import math
import pathlib

from exp.online_rit.common import SCHEDULE, sha256_file

GATE_IDENTITY_KEYS = (
    "suite", "library_sha256", "scales_sha256", "checkpoint_identity_sha256",
    "h_exec", "schedule_id", "template_sha256", "corpus_manifest_sha256", "label_code_sha256",
)
PARITY_REPLAYS = ("full", "warm875", "warm750", "warm500")


def validate_parity(gate: dict, identity: dict) -> None:
    """Require the preregistered four-tier, 200-row gate and the exact input identity."""
    if gate.get("status") != "PASS":
        raise SystemExit(f"parity gate is not PASS: {gate.get('reasons')}")
    mismatches = [k for k in GATE_IDENTITY_KEYS if identity.get(k) is None or gate.get("identity", {}).get(k) != identity[k]]
    if mismatches:
        raise SystemExit(f"parity gate was computed for a different input: {mismatches}")
    counts = gate.get("per_task_rows", {})
    sample = gate.get("sample", [])
    if (gate.get("n_rows", 0) < 200 or set(counts) != {str(i) for i in range(10)}
            or any(v < 20 for v in counts.values()) or sum(counts.values()) != gate["n_rows"]
            or len(sample) != gate["n_rows"]
            or len({(r["trajectory_id"], r["decision_id"]) for r in sample}) != len(sample)):
        raise SystemExit("parity gate lacks the full unique 200-row / 20-per-task sample")
    floor = gate.get("floor_median")
    if floor is None or not math.isfinite(floor) or floor < 0 or gate.get("ratio") != 0.1:
        raise SystemExit("invalid parity floor or preregistered ratio")
    for name in PARITY_REPLAYS:
        stats = gate.get("replays", {}).get(name, {})
        value = stats.get("max" if floor == 0 else "p90")
        if stats.get("n") != gate["n_rows"] or value is None or not math.isfinite(value) or value < 0 or value > 0.1 * floor:
            raise SystemExit(f"parity {name} does not satisfy the numerical gate")


def table_record(table: str | pathlib.Path) -> dict:
    """Load the builder's record and bind it to the actual, complete table bytes."""
    path = pathlib.Path(str(table) + ".record.json")
    if not path.is_file():
        raise SystemExit(f"table record missing: {path}")
    record = json.loads(path.read_text())
    if record.get("out_sha256") != sha256_file(table) or record.get("smoke") is not False:
        raise SystemExit("table record is stale or describes a smoke table")
    identity = record.get("identity", {})
    if any(identity.get(k) is None for k in GATE_IDENTITY_KEYS) or identity["schedule_id"] != SCHEDULE.schedule_id:
        raise SystemExit("table record has missing or incompatible input identity")
    if not record.get("parity_gate_sha256"):
        raise SystemExit("table record has no parity provenance")
    return record
