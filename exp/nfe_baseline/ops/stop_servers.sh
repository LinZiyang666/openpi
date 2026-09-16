#!/bin/bash
# Stop the nfesrv<port> servers on <base_port>..<base_port+n-1> by PID (no pkill patterns).
#
# usage: stop_servers.sh <base_port> <n>
set -u
BASE=${1:?base port}
N=${2:?n}
export PATH=/usr/local/bin:/usr/bin:/bin
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  PIDS=$(ss -tlnpH "sport = :$P" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for pid in $PIDS; do kill "$pid" 2>/dev/null && echo "port $P: sent TERM to $pid"; done
done
for _ in $(seq 1 30); do
  busy=0
  for i in $(seq 0 $((N-1))); do ss -tlnH "sport = :$((BASE+i))" | grep -q . && busy=$((busy+1)); done
  [ "$busy" -eq 0 ] && break
  sleep 2
done
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  PIDS=$(ss -tlnpH "sport = :$P" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for pid in $PIDS; do kill -9 "$pid" 2>/dev/null && echo "port $P: KILLed $pid"; done
  tmux kill-session -t "nfesrv$P" 2>/dev/null
  rmdir "/tmp/nfe/lock_nfesrv$P" 2>/dev/null
done
echo "ports still listening: $(for i in $(seq 0 $((N-1))); do ss -tlnH "sport = :$((BASE+i))" | grep -q . && printf '%s ' $((BASE+i)); done)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
