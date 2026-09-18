#!/bin/bash
# weilandserver / h100: run the bounded-retry queue (exp/dp_nfe/x0_queue.py) in a tmux session, sourcing the DP env.
# usage: x0_queue.sh train|eval|status <arms e.g. core,explore,image> [extra x0_queue args...]
# Artifacts: runs at $X0_DATA/runs/<cell>, results at $X0_DATA/results_trailing/<cell>/<split>_<sampler>_<k>, ledger at
# $X0_DATA/queue_<mode>.json, log at /tmp/x0/queue_<mode>.log. Idempotent: an existing tmux session of the same mode is reused.
set -u
MODE=${1:?train|eval|status}; ARMS=${2:?arms}; shift 2
source "$(dirname "$0")/dp_x0_env.sh"
CELLS=$X0_DATA/cells
TASKS=$X0_CODE/exp/dp_nfe/config/x0_multimodal/tasks.yaml
# Training and evaluation persist independently; status reads a combined snapshot.
if [ "$MODE" = status ]; then
  python - "$X0_DATA" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
jobs = {}
for mode in ("train", "eval"):
    path = root / f"queue_{mode}.json"
    if path.is_file():
        jobs.update(json.loads(path.read_text()))
(root / "queue_status.json").write_text(json.dumps(jobs))
PY
  [ $? -eq 0 ] || exit 2
fi
CMD="python -m exp.dp_nfe.x0_queue $MODE --tasks $TASKS --cells $CELLS --runs $X0_DATA/runs --results $X0_DATA/results_trailing --state $X0_DATA/queue_$MODE.json --arms $ARMS --dp-root $DP_ROOT $*"
if [ "$MODE" = status ]; then $CMD; exit $?; fi
NAME=x0q_$MODE
tmux has-session -t "=$NAME" 2>/dev/null && { echo "$NAME already running"; exit 0; }
tmux new -s "$NAME" -d "source $(dirname "$0")/dp_x0_env.sh; $CMD 2>&1 | tee -a /tmp/x0/queue_$MODE.log; echo QUEUE_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/x0/queue_$MODE.log"
echo "started tmux $NAME: $CMD"
