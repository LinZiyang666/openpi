#!/bin/bash
# Start N RoboCasa-2024 Cosmos server replicas on BASE..BASE+N-1 (default 23250). Idempotent.
set -u
N=${1:-7}; BASE=${2:-23250}
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 0 $((N-1))); do bash "$HERE/serve_rc24.sh" $((BASE+i)); sleep 20; done
tmux ls | grep -c cosmos_srv
