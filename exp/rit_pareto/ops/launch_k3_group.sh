#!/bin/bash
# launch_k3_group.sh <N> <suite> <rit|gst>  — start K3 group N on timan108 in tmux k3_gN (refuses if a runner is alive)
set -euo pipefail
N=$1; suite=$2; rule=$3
LOG=/tmp/rit_k3_${suite}_${rule}.log
tether exec --timeout 60s timan108 -- bash -lc "export PATH=/usr/local/bin:/usr/bin:/bin
if pgrep -f '[r]un_gtp' >/dev/null; then echo 'REFUSE: a run-gtp runner is still alive'; pgrep -af '[r]un_gtp' | cut -c1-160; exit 3; fi
if tmux has-session -t k3_g$N 2>/dev/null; then echo 'REFUSE: tmux k3_g$N exists'; exit 3; fi
if [ -f $LOG ]; then mv $LOG $LOG.\$(date +%H%M%S).old; fi
tmux new -s k3_g$N -d \"/tmp/dsp_precheck/rit_pareto/run_group_k3.sh $suite $rule 2>&1 | tee $LOG; echo RIT_GROUP_WRAP_DONE | tee -a $LOG\"
sleep 5; tmux ls | grep k3_g$N; echo LAUNCHED k3_g$N $suite $rule"
