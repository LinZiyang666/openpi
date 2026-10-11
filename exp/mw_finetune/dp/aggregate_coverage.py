"""Supplement A's aggregate with nominal ML45 held-out / annotation coverage."""
import argparse
import json
from pathlib import Path

from exp.metaworld import tasks as T

ML45_HELD_OUT = {"bin-picking-v3", "box-close-v3", "hand-insert-v3", "door-lock-v3", "door-unlock-v3"}
ADDITIONAL_UNANNOTATED = {"dial-turn-v3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.run / "summary.json").read_text())
    groups = {}
    for label, names in {
        "nominal_ml45_held_out_5": ML45_HELD_OUT,
        "additional_unannotated_dial": ADDITIONAL_UNANNOTATED,
        "source_annotated_44": set(T.TASK_NAMES) - ML45_HELD_OUT - ADDITIONAL_UNANNOTATED,
    }.items():
        selected = {name: summary["per_task"].get(name, {"n": 0, "sr": None}) for name in sorted(names)}
        rates = [row["sr"] for row in selected.values() if row["sr"] is not None]
        groups[label] = dict(tasks=sorted(names), completed_episodes=sum(row["n"] for row in selected.values()),
            per_task_mean=sum(rates)/len(rates) if rates else None, missing_tasks=[name for name,row in selected.items() if not row["n"]])
    report = dict(groups=groups, overall_per_task_mean=summary["overall_per_task_mean"],
        limitation="Source44 mapping inferred from published task descriptions, not proof of checkpoint training coverage",
        source="https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld/env_dict.py")
    (a.run / "dp_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
