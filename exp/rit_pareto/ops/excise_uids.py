#!/usr/bin/env python
"""excise_uids.py <suite> <rule> <uids.json>   (runs on timan108, runner must be stopped)

Remove every journal / per_step row belonging to the listed task_uids so the next
``run_group_k3.sh <suite> <rule>`` launch re-runs exactly those episodes (episode-level
resume). Backs up both files first (``.pre_excise_<HHMMSS>``) and refuses if the runner
is alive. Prints before/after counts.
"""
import json
import os
import subprocess
import sys
import time

suite, rule, uids_path = sys.argv[1], sys.argv[2], sys.argv[3]
out = f"/tmp/dsp_precheck/rit_pareto/{suite}_k3_{rule}"
if subprocess.run(["pgrep", "-f", "[r]un_gtp"], capture_output=True).returncode == 0:
    sys.exit("REFUSE: run_gtp runner is alive; stop it first")
uids = set(json.load(open(uids_path))["uids"])
tag = time.strftime("%H%M%S")
for name in ("journal.jsonl", "per_step.jsonl"):
    path = f"{out}/{name}"
    backup = f"{path}.pre_excise_{tag}"
    os.rename(path, backup)
    kept = dropped = 0
    with open(backup) as src, open(path, "w") as dst:
        for line in src:
            if json.loads(line).get("task_uid") in uids:
                dropped += 1
            else:
                dst.write(line)
                kept += 1
    print(f"{name}: kept {kept} dropped {dropped} (backup {backup})")
print(f"excised {len(uids)} uids; relaunch run_group_k3.sh {suite} {rule} to re-run them")
