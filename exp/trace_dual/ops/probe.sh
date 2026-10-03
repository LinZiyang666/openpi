#!/bin/bash
# probe.sh -- one PROBE line: current group, journal progress on timan107, local servers, trace volume.
B=/home/weiland/trace_runs/dual_20260923
G=$(cat $B/state/current 2>/dev/null || echo none)
ts=$(date +%H:%M)
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
srv=$(tmux ls 2>/dev/null | grep -c '^trsrv')
lis=$(ss -ltnH | grep -cE ':231[0-9][0-9] ')
tr=$(du -sh $B/runs/$G/trace 2>/dev/null | cut -f1); nh5=$(find $B/runs/$G/trace -name '*.h5' 2>/dev/null | wc -l)
homefree=$(df -h /home | awk 'NR==2{print $4}')
ramavail=$(free -g | awk '/^Mem/{print $7"G"}')
rssmax=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -o rss= -p $p; done | sort -n | tail -1 | awk '{printf "%.1fG", $1/1048576}')
remote=$(timeout 60 tether exec timan107 -- bash -c "f=/scratch/zixuans8/trace_runs/dual_20260923/$G/journal.jsonl; if [ -f \$f ]; then python3 -c \"
import json,sys
u={};ok=0
for l in open('\$f'):
    try: r=json.loads(l)
    except Exception: continue
    u[r.get('task_uid')]=r
s=sum(1 for r in u.values() if r.get('success'))
print(len(u), s)
\"; else echo 0 0; fi; pgrep -fc '[w]orker_entry'; uptime | sed 's/.*average: //'" 2>/dev/null | tr '\n' ' ')
echo "PROBE $ts group=$G journal_uids,succ/workers/load=[$remote] gpu_mib,util=$gpu srv_tmux=$srv listen=$lis trace=${tr:-0}/${nh5}h5 home_free=$homefree ram_avail=$ramavail srv_rss_max=$rssmax"
