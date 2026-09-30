"""Actual per-request SW owner costs; use this table with ops.kpi --cost-table."""
import argparse
import json
from collections import Counter
from pathlib import Path


def ledger(rows):
    modes, total, completions = Counter(), 0., 0
    for row in rows:
        if not row.get("ok", True):
            raise ValueError("failed requests require an explicit incomplete-work audit")
        mode = row["camera_mode"]
        vision, miss = bool(row["vision"]), not bool(row["hit"])
        nc = row["camera_completion_calls"]
        nl = row["camera_stage1_calls"]
        if nl != int(vision) or (miss and not vision) or nc != int(mode == "wrist_only" and miss):
            raise ValueError("camera dispatch/completion count violates the policy contract")
        expected = (0 if not vision else (.055198 if mode == "wrist_only" else .152)) + .049890 * nc + .848 * miss
        if abs(row["owner_cost"] - expected) > 1e-12:
            raise ValueError("logged owner cost differs from actual dispatches")
        modes[mode] += 1; total += expected; completions += nc
    return dict(decisions=len(rows), camera_modes=dict(modes), completion_calls=completions,
                total_cost=total, ir_per_request=total/len(rows) if rows else None,
                ir_per_five_controls=total/len(rows) if rows else None,
                controls_note="five nominal controls/request; terminal requests may execute fewer",
                basis="owner .152/.848; wrist .055198 and completion .049890 are R4 proportional-latency assumptions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = []
    for path in sorted(a.log_dir.glob("**/decisions_*.jsonl")):
        rows.extend(row for line in path.read_text().splitlines() if (row := json.loads(line)).get("ev") == "dec")
    result = ledger(rows)
    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
