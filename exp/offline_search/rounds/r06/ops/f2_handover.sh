#!/bin/bash
# After the orphaned F2 arm's remote driver ends: stop its local server on 23166, release the claim, start lane G1 on 23166.
A=r6q2_groot_l10_50_risk_rho0p45; R=/home/weiland/trace_runs/os_closed_loop/r06_frontier
while timeout 60 tether exec timan107 -- bash -c "tmux -L oscl has-session -t oscl_$A 2>/dev/null && echo RUN" 2>/dev/null | grep -q RUN; do sleep 60; done
echo "REMOTE_DRIVER_ENDED $(date -Is)"
pid=$(ss -ltnp 2>/dev/null | grep ":23166 " | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$pid" ] && ps -o args= -p $pid | grep -q "os-tag ${A}_23166" && kill $pid && echo "killed server $pid"
sleep 5; tmux kill-session -t oscl23166 2>/dev/null
rm "$R/state/fclaim_$A/by"; rmdir "$R/state/fclaim_$A" && echo "claim released"
tmux new -s flane_G1 -d "bash /home/weiland/.claude/jobs/a607dd74/tmp/frontier_lane.sh G1 23166 18-21,62-65 10500" && echo "G1 started $(date -Is)"
