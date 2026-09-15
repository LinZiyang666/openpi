#!/usr/bin/env bash
# One-line health probe: server tmux alive, journal growth, latest state snapshot.
# Usage: health.sh <tmux_name> <journal_dir> <state_root>
set -uo pipefail
NAME=$1; JDIR=$2; STATE=$3
alive=$(tmux has-session -t "$NAME" 2>/dev/null && echo yes || echo no)
done_n=$(grep -c '"status": "done"' "$JDIR/journal.jsonl" 2>/dev/null || echo 0)
fail_n=$(grep -c '"status": "failed"' "$JDIR/journal.jsonl" 2>/dev/null || echo 0)
latest=$(ls -t "$STATE"/*/state_latest.json 2>/dev/null | head -1)
n_upd=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_updates'])" "$latest" 2>/dev/null || echo -)
echo "$(date +%H:%M:%S) server=$alive done=$done_n failed=$fail_n n_updates=$n_upd latest=$latest"
