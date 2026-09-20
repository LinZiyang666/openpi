#!/bin/bash
# weilandserver / h100: run the bounded-retry queue (exp/dp_nfe/x0_queue.py) in a tmux session, sourcing the DP env.
# usage: x0_queue.sh train|eval|status <arms e.g. core,explore,image> [extra x0_queue args, e.g. --task-names pusht,blockpush --parallel 3]
# LANE=<name> (env, optional) runs an independent lane: tmux x0q_<mode>_<lane>, ledger $X0_DATA/queue_<mode>_<lane>.json,
# log /tmp/x0/queue_<mode>_<lane>.log -- several train lanes with disjoint --task-names/--only share one GPU.
# Artifacts: runs at $X0_DATA/runs/<cell>, results at $X0_DATA/results_trailing/<cell>/<split>_<sampler>_<k>.
# status merges every queue_*.json ledger into queue_status.json (a snapshot; artifacts are re-verified when lanes run).
set -u
MODE=${1:?train|eval|status}; ARMS=${2:?arms}; shift 2
source "$(dirname "$0")/dp_x0_env.sh"
CELLS=$X0_DATA/cells
TASKS=$X0_CODE/exp/dp_nfe/config/x0_multimodal/tasks.yaml
LANE=${LANE:-}
SUF=${LANE:+_$LANE}
if [ "$MODE" = status ]; then
  python - "$X0_DATA" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); jobs = {}
for path in sorted(root.glob("queue_*.json")):
    if path.name == "queue_status.json":
        continue
    jobs.update(json.loads(path.read_text()))
(root / "queue_status.json").write_text(json.dumps(jobs))
PY
  [ $? -eq 0 ] || exit 2
  python -m exp.dp_nfe.x0_queue status --tasks $TASKS --cells $CELLS --runs $X0_DATA/runs --results $X0_DATA/results_trailing --state $X0_DATA/queue_status.json --arms $ARMS --dp-root $DP_ROOT "$@"
  exit $?
fi
CMD="python -m exp.dp_nfe.x0_queue $MODE --tasks $TASKS --cells $CELLS --runs $X0_DATA/runs --results $X0_DATA/results_trailing --state $X0_DATA/queue_$MODE$SUF.json --arms $ARMS --dp-root $DP_ROOT $*"
NAME=x0q_$MODE$SUF
# tmux must not see the conda env's libtinfo (LD_LIBRARY_PATH from dp_x0_env.sh breaks the system tmux)
TMUX="env -u LD_LIBRARY_PATH tmux"
$TMUX has-session -t "=$NAME" 2>/dev/null && { echo "$NAME already running"; exit 0; }
$TMUX new -s "$NAME" -d "source $(dirname "$0")/dp_x0_env.sh; $CMD 2>&1 | tee -a /tmp/x0/queue_$MODE$SUF.log; echo QUEUE_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/x0/queue_$MODE$SUF.log" || { echo "tmux launch failed"; exit 1; }
sleep 1; $TMUX has-session -t "=$NAME" 2>/dev/null && echo "started tmux $NAME: $CMD" || { echo "tmux session $NAME died immediately (see /tmp/x0/queue_$MODE$SUF.log)"; exit 1; }
