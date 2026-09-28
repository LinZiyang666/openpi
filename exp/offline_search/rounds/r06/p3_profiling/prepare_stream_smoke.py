"""Two fresh stream smoke arms: P10, one per model, four task/init pairs each."""
import argparse
import json
from pathlib import Path

from .prepare_smoke import specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    a = ap.parse_args()
    run = a.run_root.resolve()
    rows = [r for r in specs() if "_P10_" in r["name"]]
    for row in rows:
        old = row["name"]
        row["name"] = old.replace("_client_smoke", "_stream_smoke")
        row["plugin_args"] = [x.replace(old, row["name"]) for x in row["plugin_args"]]
    (run/"manifests").mkdir(parents=True, exist_ok=True)
    if (run/"arms_in.json").exists():
        raise ValueError("stream smoke requires a fresh run root")
    (run/"manifests/smoke.json").write_text(json.dumps(dict(selected=[dict(task=t, init=i) for t in range(2) for i in range(2)]), indent=2)+"\n")
    (run/"arms_in.json").write_text(json.dumps(rows, indent=2).replace("<RUN>", str(run))+"\n")
    schedule = [dict(arm=r["name"], smoke_manifest=str(run/"manifests/smoke.json"),
                     client_env=dict(P3_ENV_SEED="603", P3_SNAPSHOT_EVERY="1", P3_SNAPSHOT_P="1")) for r in rows]
    (run/"schedule_v2.json").write_text(json.dumps(schedule, indent=2)+"\n")
    (run/"arm_names.txt").write_text("\n".join(r["name"] for r in rows)+"\n")
    print(json.dumps(dict(arms=len(rows), episodes=8, run_root=str(run))))


if __name__ == "__main__":
    main()
