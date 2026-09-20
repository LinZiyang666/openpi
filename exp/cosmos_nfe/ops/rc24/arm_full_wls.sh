#!/bin/bash
# weilandserver: once the pi0.5 pair-A ladder has stopped (ladder_prc.log "LADDER STOPPED"/"FINISHED"), start 4 RoboCasa-2024
# Cosmos replicas on the public ports 23180..23183 (ziyanglin.com:23100-23199 is the only reachable range). Idempotent.
export HOME=/home/weiland; export PATH=/home/weiland/.local/bin:/usr/local/bin:/usr/bin:/bin
OPS=/home/weiland/cosmos_exp/exp/cosmos_nfe/ops/rc24
mkdir -p /tmp/cosmos
tmux has-session -t cosmos_arm_rc24 2>/dev/null || tmux new -s cosmos_arm_rc24 -d \
  "while ! grep -qE '^LADDER (STOPPED|FINISHED)' /tmp/nfe/ladder_prc.log 2>/dev/null; do sleep 20; done; sleep 30; echo \"RC24 SERVERS UP \$(date +%H:%M:%S)\" >> /tmp/cosmos/rc24_full.log; bash $OPS/launch_replicas_rc24.sh 4 23180 >> /tmp/cosmos/rc24_full.log 2>&1"
sleep 1; tmux ls | grep -E "cosmos_arm|cosmos_srv"
