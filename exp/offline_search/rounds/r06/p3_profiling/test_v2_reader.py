"""Synthetic actuator/outcome fixture over real-store replay, not an SR result."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from .read_v2 import server_rows, audit_group, control_join, main as read_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    arm = "pi05_l10_50_p1"
    source = a.matrix / "runs" / arm / "server_cpu"
    target = a.out / "runs" / arm
    (target / "client").mkdir(parents=True)
    (target / "server_cpu").symlink_to(source, target_is_directory=True)
    groups, configs, ends = server_rows(source)
    outcomes = []
    for key, records in groups.items():
        root = a.out / "telemetry" / str(len(outcomes))
        root.mkdir(parents=True)
        end = ends[key]
        ident = dict(task_uid=key[0], attempt=key[1])
        outcomes.append(dict(**ident, accepted=True, status="done", error=None, success=end["success"]))
        rows = [dict(ident, ev="attempt_start")]
        control = 0
        for i, r in sorted(records["p3_decision"].items()):
            with np.load(source / r["input_archive"]) as data:
                wire = data["wire_chunk"].tolist()
            rows.append(dict(ident, ev="decision", decision_step=i, control=control,
                             wire_chunk=wire, marker=dict(p3_anchor=r["vision"])))
            n = 2 if i == max(records["dec"]) else 5
            for offset in range(n):
                rows.append(dict(ident, ev="control", decision_step=i, control=control, chunk_offset=offset,
                    action_issued=wire[offset], after={"toy_position": [control+1]},
                    done=bool(end["success"]) and i == max(records["dec"]) and offset == n-1))
                control += 1
        rows.append(dict(ident, ev="rollout_end", controls=control, success=end["success"],
                         final_chunk_completed=2, synthetic_fixture=True))
        (root / "controls.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    (target / "client/journal.jsonl").write_text("".join(json.dumps(x)+"\n" for x in outcomes))
    original_args = sys.argv
    try:
        sys.argv = ["read_v2", "--run-root", str(a.out), "--arms", arm, "--client-root", str(a.out/"telemetry"),
                    "--out", str(a.out/"tables")]
        read_main()
    finally:
        sys.argv = original_args
    key, record = next(iter(groups.items()))
    first = record["p3_decision"][0]
    config = configs[first["tag"], first["conn"]]
    failures = []
    for corruption in ("missing_decision", "propensity", "action", "tail", "outcome", "nominal_p"):
        rr, ee = copy.deepcopy(record), copy.deepcopy(ends[key])
        if corruption == "missing_decision":
            del rr["p3_decision"][1]
        elif corruption == "propensity":
            rr["p3_anchor"][0]["assignment"]["actual_propensity"] = .3
        elif corruption == "nominal_p":
            rr["p3_anchor"][0]["assignment"]["propensity"] = .3
        elif corruption == "action":
            rr["p3_anchor"][0]["executed_chunk"][0][0] += 1
        elif corruption == "tail":
            rr["dec"][1]["src"] = "cache_blind"
        else:
            ee["n_decisions"] -= 1
        try:
            audit_group(source, rr, config, ee)
        except ValueError:
            failures.append(corruption)
        else:
            raise AssertionError("corrupted log accepted: " + corruption)
    client_file = a.out / "telemetry/0/controls.jsonl"
    client_rows = [json.loads(x) for x in client_file.read_text().splitlines()]
    client_rows.append(next(r for r in client_rows if r["ev"] == "decision"))
    bad_client = a.out / "duplicate_client.jsonl"
    bad_client.write_text("".join(json.dumps(x)+"\n" for x in client_rows))
    try:
        control_join(bad_client, key, record, outcomes[0])
    except ValueError:
        failures.append("duplicate_client_decision")
    else:
        raise AssertionError("duplicate client decision accepted")
    duplicate = dict(outcomes[0], attempt=2)
    with (target / "client/journal.jsonl").open("a") as f:
        f.write(json.dumps(duplicate)+"\n")
    try:
        sys.argv = ["read_v2", "--run-root", str(a.out), "--arms", arm, "--client-root", str(a.out/"telemetry"),
                    "--out", str(a.out/"rejected_tables")]
        try:
            read_main()
        except ValueError as exc:
            assert "multiple accepted attempts" in str(exc), str(exc)
            failures.append("multiple_accepted_attempts")
        else:
            raise AssertionError("multiple accepted attempts accepted")
    finally:
        sys.argv = original_args
    report = dict(PASS=True, fixture="synthetic controls, no simulator/SR evidence", rejected=failures,
                  tables=json.loads((a.out/"tables/audit.json").read_text()))
    (Path(__file__).parent / "results/reader_v2.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
