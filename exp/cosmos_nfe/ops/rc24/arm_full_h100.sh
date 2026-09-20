#!/bin/bash
# h100: once the pi0.5 pair-B ladder has stopped (nfestopk_prc2 wrote "LADDER STOPPED"/"LADDER FINISHED" to ladder_prc2.log),
# start 10 RoboCasa-2024 Cosmos replicas on 23250..23259. Idempotent (has-session).
export HOME=/home/exouser; export PATH=/home/exouser/.local/bin:/usr/local/bin:/usr/bin:/bin
OPS=/data/cosmos/openpi_exp/exp/cosmos_nfe/ops/rc24
mkdir -p /tmp/cosmos
tmux has-session -t cosmos_arm_rc24 2>/dev/null || tmux new -s cosmos_arm_rc24 -d \
  "while ! grep -qE '^LADDER (STOPPED|FINISHED)' /tmp/nfe/ladder_prc2.log 2>/dev/null; do sleep 20; done; sleep 30; echo \"RC24 SERVERS UP \$(date +%H:%M:%S)\" >> /tmp/cosmos/rc24_full.log; bash $OPS/launch_replicas_rc24.sh 10 23250 >> /tmp/cosmos/rc24_full.log 2>&1"
sleep 1; tmux ls | grep -E "cosmos_arm|cosmos_srv"
