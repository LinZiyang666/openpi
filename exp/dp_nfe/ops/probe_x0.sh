#!/bin/bash
# L3 probe for the x0 experiment: four PROBE lines (wls, h100, queues, notes), nothing else.
ts=$(TZ=America/Chicago date +%H:%M)
w=$(tether exec --timeout 25s weilandserver -- bash -lc 'g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d " "); t=$(tmux ls 2>/dev/null | grep -oE "^x0[a-zA-Z0-9_]*" | tr "\n" ","); echo "gpu=$g tmux=${t:-none} load=$(cut -d" " -f1 /proc/loadavg) disk=$(df -h /data | awk "NR==2{print \$4}")"' 2>/dev/null | awk '!seen[$0]++' | tail -1)
h=$(tether exec --timeout 25s h100 -- bash -lc 'export HOME=/home/exouser; g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d " "); t=$(tmux ls 2>/dev/null | grep -oE "^x0[a-zA-Z0-9_]*" | tr "\n" ","); echo "gpu=$g tmux=${t:-none} load=$(cut -d" " -f1 /proc/loadavg) disk=$(df -h /data | awk "NR==2{print \$4}") root=$(df -h / | awk "NR==2{print \$4}")"' 2>/dev/null | awk '!seen[$0]++' | tail -1)
q=$(for hh in weilandserver h100; do tether exec --timeout 25s $hh -- bash -lc 'export HOME=/home/exouser; for f in /tmp/x0/queue_*.log /tmp/x0/pilot.log /tmp/x0/pilot2.log /tmp/x0/p0_can_mh.log; do [ -f $f ] && echo "$(basename $f .log)=$(grep -aoE "QUEUE (DONE|INCOMPLETE)[^|]*|QUEUE_EXIT=[0-9]+|P0_DONE [a-z_]+|PILOT_DONE [a-z_]+|attempt [0-9]+|status=[a-z]+ [a-z/0-9_]+" $f 2>/dev/null | tail -1 | cut -c1-70)"; done | tr "\n" " "' 2>/dev/null | awk '!seen[$0]++' | tail -1 | sed "s/^/$hh:/"; done | tr '\n' ' ')
echo "PROBE $ts wls $w"
echo "PROBE $ts h100 $h"
echo "PROBE $ts jobs $q"
echo "PROBE $ts note $(cat /home/weiland/.claude/jobs/9267c51a/tmp/probe_note.txt 2>/dev/null | head -1)"
