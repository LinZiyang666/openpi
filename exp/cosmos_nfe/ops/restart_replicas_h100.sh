#!/bin/bash
# h100: stop all replicas on BASE..BASE+N-1 and start them again (after a code push). Idempotent.
set -u
N=${1:-10}; BASE=${2:-23240}
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 0 $((N-1))); do bash "$HERE/stop_h100.sh" $((BASE+i)) > /dev/null 2>&1; done
sleep 2; bash "$HERE/launch_replicas_h100.sh" "$N" "$BASE"
