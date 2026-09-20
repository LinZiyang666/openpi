#!/bin/bash
# h100: stop the Cosmos server on <port> by PID and release its lock.
set -u
PORT=${1:-23240}
for pid in $(ss -tlnpH "sport = :$PORT" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do kill "$pid" && echo "TERM $pid"; done
sleep 3; tmux kill-session -t "cosmos_srv$PORT" 2>/dev/null; rmdir "/tmp/cosmos/lock_srv$PORT" 2>/dev/null; ss -tln | grep -c ":$PORT " || true
