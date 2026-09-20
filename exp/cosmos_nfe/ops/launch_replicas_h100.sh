#!/bin/bash
# h100: start N Cosmos server replicas on ports 23240..23240+N-1 (one model copy each, ~6.5 GB). Idempotent.
set -u
N=${1:-10}; BASE=${2:-23240}
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 0 $((N-1))); do bash "$HERE/serve_h100.sh" $((BASE+i)); sleep 15; done
tmux ls | grep -c cosmos_srv
