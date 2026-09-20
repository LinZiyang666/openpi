#!/bin/bash
# Stop RoboCasa-2024 Cosmos server replicas on BASE..BASE+N-1 by tmux session name; release locks.
set -u
N=${1:-7}; BASE=${2:-23250}
for i in $(seq 0 $((N-1))); do P=$((BASE+i)); tmux kill-session -t "cosmos_srv$P" 2>/dev/null && echo "killed cosmos_srv$P"; rmdir "/tmp/cosmos/lock_srv$P" 2>/dev/null; done
sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader
