"""Resolve a per-arm client schedule without launching anything."""
import argparse
import json
import math
import os
from pathlib import Path
import re


def resolve(run, arm, env=None):
    env = os.environ if env is None else env
    run = Path(run).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run.name) or not re.fullmatch(r"[A-Za-z0-9_]+", arm):
        raise ValueError("run basename and arm must be shell-safe identifiers")
    rows = json.loads((run / "arms.json").read_text())
    row = next(x for x in rows if x["arm"] == arm)
    schedule_path = Path(env.get("P3_SCHEDULE", str(run / "schedule_v2.json")))
    schedule = json.loads(schedule_path.read_text().replace("<RUN>", str(run))) if schedule_path.exists() else []
    matches = [x for x in schedule if x["arm"] == arm]
    if len(matches) > 1:
        raise ValueError("duplicate schedule arm")
    entry = matches[0] if matches else {}
    settings = dict(P3_SNAPSHOT_EVERY="1", P3_SNAPSHOT_P="1")
    settings.update(entry.get("client_env", {}))
    settings.update(row.get("client_env", {}))
    for key in ("P3_ENV_SEED", "P3_SNAPSHOT_EVERY", "P3_SNAPSHOT_P"):
        if key in row:
            settings[key] = row[key]
        if key in env:
            settings[key] = env[key]
    if "P3_ENV_SEED" not in settings:
        raise ValueError("P3_ENV_SEED missing from schedule, arm or environment")
    seed, every, probability = int(settings["P3_ENV_SEED"]), int(settings["P3_SNAPSHOT_EVERY"]), float(settings["P3_SNAPSHOT_P"])
    if not 0 <= seed < 2**32 or every < 1 or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid client seed/snapshot settings")
    phase = env.get("P3_PHASE", "pilot")
    if phase not in ("pilot", "continuation", "smoke"):
        raise ValueError("P3_PHASE must be pilot, continuation or smoke")
    manifest = env.get("OSCL_MANIFEST") or row.get("manifest") or entry.get(phase + "_manifest", "")
    remote = "/scratch/zixuans8/openpi_trace/os_cl/p3_client/{}/{}/p3_telemetry".format(run.name, arm)
    chosen = dict(P3_ENV_SEED=str(seed), P3_SNAPSHOT_EVERY=str(every), P3_SNAPSHOT_P=str(probability), P3_SNAPSHOT_DIR=remote)
    return dict(arm=arm, phase=phase, manifest=manifest, client_env=chosen,
                telemetry_remote=remote, telemetry_local=str(run / "runs" / arm / "client_telemetry"),
                env_prefix=" ".join(k+"="+v for k, v in chosen.items()) + " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--field", choices=("manifest", "env_prefix", "telemetry_remote", "telemetry_local"))
    a = ap.parse_args()
    result = resolve(a.run_root, a.arm)
    print(result[a.field] if a.field else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
