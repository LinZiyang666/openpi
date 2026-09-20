#!/bin/bash
# h100: stop all Cosmos replicas (ports 23240-23249). Idempotent.
HERE=$(cd "$(dirname "$0")" && pwd)
for p in $(seq 23240 23249); do bash "$HERE/stop_h100.sh" $p > /dev/null 2>&1; done
tmux ls 2>/dev/null | grep -c cosmos_srv; ss -tln | grep -cE ":2324[0-9] "; nvidia-smi --query-gpu=memory.used --format=csv,noheader
