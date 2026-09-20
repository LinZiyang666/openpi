#!/usr/bin/env bash
# Pull one arm's server-side evidence (rows_*.jsonl + arrays_*/ + manifest_*.json) to the local tree
# and verify every arrays file against the sha256 its rows carry.
# usage: pull_server_rows.sh <user@host:remote_out_dir/<arm_id>> <local exp/step_diag/data/server/<arm_id>>
set -eu
SRC=${1:?user@host:dir}; DST=${2:?local dir}
mkdir -p "$DST"
rsync -a --itemize-changes "$SRC/" "$DST/"
python3 - "$DST" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); bad = 0; n = 0
for rows in sorted(root.glob("rows_*.jsonl")):
    for line in rows.read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if not r or r.get("status") != "finalize" or not r.get("arrays"):
            continue
        n += 1
        p = root / r["arrays"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r["arrays_sha256"]:
            bad += 1; print("SHA MISMATCH", rows.name, r["task_uid"], r["arrays"])
print(f"PULL_VERIFY {root}: {n} arrays files checked, {bad} bad")
sys.exit(1 if bad else 0)
PY
