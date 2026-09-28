"""Emit eight v2 client smoke specs (four cohorts in each of two cells)."""
import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def specs():
    rows = json.loads((HERE/"arms_v2.json").read_text())
    selected = []
    for cell in ("pi05_l10_50", "groot_l10_500"):
        for cohort in ("A", "P10", "factorial", "window"):
            row = copy.deepcopy(next(x for x in rows if x["name"] == "r6p3v2_{}_{}_r0".format(cell, cohort)))
            name = row["name"] + "_client_smoke"
            row["plugin_args"] = [x.replace(row["name"], name) for x in row["plugin_args"]]
            row["name"] = name
            row["manifest"] = "<RUN>/manifests/smoke.json"
            # Exercise every resampling code path; window triggers are forced
            # for smoke only, while source/duration/hold semantics stay intact.
            row["kwargs"]["resample_p"] = 1.
            if cohort == "window":
                row["kwargs"]["p"] = 1.
            selected.append(row)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path)
    a = ap.parse_args()
    rows = specs()
    if a.run_root is None:
        (HERE/"arms_smoke_v2.json").write_text(json.dumps(rows, indent=2)+"\n")
        print("wrote 8 placeholder specs (4 cohorts x 2 cells)")
        return
    run = a.run_root.resolve()
    (run/"manifests").mkdir(parents=True, exist_ok=True)
    (run/"manifests/smoke.json").write_text(json.dumps(dict(selected=[dict(task=t, init=i) for t in range(2) for i in range(2)]), indent=2)+"\n")
    (run/"arms_in.json").write_text(json.dumps(rows, indent=2).replace("<RUN>", str(run))+"\n")
    schedule = [dict(arm=r["name"], smoke_manifest=str(run/"manifests/smoke.json"),
                     pilot_manifest=str(run/"manifests/smoke.json"),
                     client_env=dict(P3_ENV_SEED="603", P3_SNAPSHOT_EVERY="1", P3_SNAPSHOT_P="1")) for r in rows]
    (run/"schedule_v2.json").write_text(json.dumps(schedule, indent=2)+"\n")
    (run/"arm_names.txt").write_text("\n".join(r["name"] for r in rows)+"\n")
    print(json.dumps(dict(arms=len(rows),episodes_per_arm=4,episodes=32,run_root=str(run))))


if __name__ == "__main__":
    main()
