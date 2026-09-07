#!/bin/bash
# G-A1 (plan W3): step-sensitivity screen, client side on timan107.
# Same recipe as the W8 floor arm (5 PickPlace tasks x 50 trials, layout/style
# (1,1), base_seed 1000000, pinned objects, replan 5, 14 workers) so the k=4
# point is the existing ws2t arm; this launches ONE k at a time against a
# ksweep server on weilandserver:23160. Usage: ga1_launch_client.sh <k>
set -u
K=${1:?k}
source /tmp/w8_common.sh
PREFIX=ga1k${K}
tmux kill-session -t ga1 2>/dev/null
tmux new-session -d -s ga1 "cd $REPO && PYTHONPATH=$PP $PY -m exp.robocasa365.run_ws_search \
  --teacher groot_tp --server ziyanglin.com:23160 --cid teacher --run-prefix $PREFIX \
  --tasks $TASKS --episodes 50 \
  --layout 1 --style 1 --base-seed 1000000 --replan-steps 5 \
  --pinned-objects $PIN --env-config $ENVCFG \
  --workers 14 --gpu-ids 0,1,2,3,4,5,6,7 \
  --role all 2>&1 | tee /tmp/ga1_${PREFIX}.log; echo GA1_DONE_k${K} >> /tmp/ga1_${PREFIX}.log"
echo "ga1 spawned: $PREFIX (250 ep, 14 workers -> ziyanglin.com:23160)"
