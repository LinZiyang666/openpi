#!/bin/bash
# timan107/108: once this box's pi0.5 lane has stopped (lane log "RCLANE STOPPED"/"FINISHED") and the assigned servers listen,
# run the RoboCasa-2024 client ladder k=1..5 on this box's trial ranges. Idempotent.
# usage: arm_full_client.sh <lane log> <server host> <base port> <n replicas> <workers> <gpu ids csv> <ranges csv> <url host for ws>
set -u
LANE=${1:?lane log}; SH=${2:?server host}; BP=${3:?base port}; NR=${4:?n replicas}; W=${5:?workers}; GPUS=${6:?gpus}; RANGES=${7:?ranges}; WSHOST=${8:-$SH}
export HOME=/home/zixuans8
OPS=/scratch/zixuans8/cosmos/openpi_exp/exp/cosmos_nfe/ops/rc24
URLS=$(for i in $(seq 0 $((NR-1))); do printf 'ws://%s:%d,' "$WSHOST" $((BP+i)); done | sed 's/,$//')
mkdir -p /tmp/cosmos
tmux has-session -t cosmos_arm_rc24 2>/dev/null || tmux new -s cosmos_arm_rc24 -d \
  "while ! grep -qE '^RCLANE (STOPPED|FINISHED)' $LANE 2>/dev/null; do sleep 20; done; \
   while :; do up=0; for i in \$(seq 0 $((NR-1))); do timeout 3 bash -c \"</dev/tcp/$SH/\$(($BP+i))\" 2>/dev/null && up=\$((up+1)); done; [ \$up -ge $NR ] && break; sleep 30; done; \
   echo \"RC24 CLIENT START servers=\$up \$(date +%H:%M:%S)\" >> /tmp/cosmos/rc24_full.log; \
   RC24_RANGES=$RANGES NFE_HOME=/home/zixuans8 bash $OPS/ladder_rc24_client.sh 1,2,3,4,5 $URLS $W $GPUS /scratch/zixuans8/cosmos/results_rc24 >> /tmp/cosmos/rc24_full.log 2>&1"
sleep 1; tmux ls | grep -E "cosmos_arm"
