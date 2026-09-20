#!/bin/bash
# Stop the Cosmos LIBERO client/ladder sessions on this box (weilandserver or timan107). Idempotent.
for s in $(tmux ls 2>/dev/null | grep -oE "^cosmos_(ladder|libero_[^:]*)"); do tmux kill-session -t "$s" 2>/dev/null && echo "killed $s"; done
sleep 3
n=$(pgrep -f "[r]un_libero_shard" | wc -l); echo "run_libero_shard procs left: $n"
for p in $(pgrep -f "[r]un_libero_shard"); do kill "$p" 2>/dev/null; done
echo "COSMOS STOPPED $(date +%H:%M:%S)" >> /tmp/cosmos/ladder.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" "/"; echo
