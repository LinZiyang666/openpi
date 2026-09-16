#!/bin/bash
# One-line server health for the L3 cron (weilandserver): nfesrv sessions, listeners, GPU, error lines.
#
# usage: probe_server.sh <base_port> <n>
export PATH=/usr/local/bin:/usr/bin:/bin
BASE=${1:?base port}; N=${2:?n}
live=$(tmux ls 2>/dev/null | grep -c "^nfesrv")
up=0; for i in $(seq 0 $((N-1))); do ss -tlnH "sport = :$((BASE+i))" | grep -q . && up=$((up+1)); done
err=$(cat /tmp/nfe/nfesrv*.log 2>/dev/null | grep -c -E "Traceback|CUDA out of memory|Error")
k=$(grep -h -m1 -o -E "KSWEEP num_steps=[0-9]+|denoising steps: [0-9]+" /tmp/nfe/nfesrv*.log 2>/dev/null | sort -u | tr '\n' ' ')
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')
echo "NFE SERVER tmux=$live listening=$up/$N err_lines=$err steps=[$k] gpu=$gpu"
