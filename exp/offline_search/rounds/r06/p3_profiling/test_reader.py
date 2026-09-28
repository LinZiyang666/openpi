"""Strict-log negative cases and final CSV command exercise on CPU fixtures."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from .read_logs import load, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    arm = "pi05_l10_50_p30"
    source = a.matrix / "runs" / arm
    original_path = next((source / "server_cpu").glob("decisions_*.jsonl"))
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    anchors = [i for i, r in enumerate(original) if r["ev"] == "p3_anchor"]
    passed = []
    for case in ("missing_anchor", "wrong_propensity", "wrong_action", "missing_tail", "wrong_outcome", "duplicate_conflict"):
        target = a.out / case / "runs" / arm
        shutil.copytree(source / "client", target / "client")
        (target / "server_cpu").mkdir()
        rows = copy.deepcopy(original)
        pos = anchors[0]
        if case == "missing_anchor":
            rows.pop(pos)
        elif case == "wrong_propensity":
            rows[pos]["assignment"]["propensity"] = .123
        elif case == "wrong_action":
            rows[pos]["executed_chunk"][0][0] += .1
        elif case == "missing_tail":
            for i, row in enumerate(rows):
                if row["ev"] == "dec" and row["step"] == 1:
                    rows.pop(i)
                    break
        elif case == "wrong_outcome":
            p = target / "client/journal.jsonl"
            js = [json.loads(line) for line in p.read_text().splitlines()]
            js[0]["success"] = not js[0]["success"]
            p.write_text("".join(json.dumps(j) + "\n" for j in js))
        else:
            duplicate = copy.deepcopy(rows[pos])
            duplicate["policy_chunk"][0][0] += .1
            rows.append(duplicate)
        (target / "server_cpu" / original_path.name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            load(a.out / case, arm)
        except ValueError as err:
            passed.append(dict(case=case, rejected=str(err)))
        else:
            raise AssertionError(f"forgery accepted: {case}")
    tables = load(a.matrix, arm)
    for name, rows in zip(("anchors", "neighbours", "action_steps", "episodes"), tables):
        write_csv(a.out / f"{name}.csv", rows)
    result = dict(PASS=True, negative_cases=passed, rows=[len(r) for r in tables])
    (Path(__file__).parent / "results/reader.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
