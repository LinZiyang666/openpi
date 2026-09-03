#!/bin/bash
# rit_health.sh <suite> <layer> <total_ep>   one-line health summary for the running group
export PATH=/usr/local/bin:/usr/bin:/bin
suite=$1; layer=$2; TOTAL=$3
OUT=/tmp/dsp_precheck/rit_pareto/${suite}_k3_${layer}
J=$OUT/journal.jsonl; LOG=/tmp/rit_k3_${suite}_${layer}.log
TS=$(date "+%H:%M:%S")
if [ -f "$J" ]; then
  done=$(grep -cE '"status": ?"(done|failed)"' "$J" 2>/dev/null); [ -z "$done" ] && done=0
  ok=$(grep -cE '"status": ?"done"' "$J" 2>/dev/null); [ -z "$ok" ] && ok=0
else done=0; ok=0; fi
runner=$(pgrep -f "[r]un_gtp" | wc -l)
workers=$(pgrep -f "[w]orker_entry" | wc -l)
srv=$(timeout 4 bash -c "echo > /dev/tcp/ziyanglin.com/23150" 2>/dev/null && echo UP || echo DOWN)
err=$(grep -ciE "FatalError|refused|out of memory|CUDA error|Killed|1011 |worker .* (died|exited|crash)" "$LOG" 2>/dev/null); [ -z "$err" ] && err=0
mem=$(free -g | awk '/Mem:/{print $7}')
cexc=$(grep -c "Caught exception" "$LOG" 2>/dev/null); [ -z "$cexc" ] && cexc=0
restarts=$(grep -c "died; restart" "$LOG" 2>/dev/null); [ -z "$restarts" ] && restarts=0
big=$(for p in $(pgrep -f "[w]orker_entry"); do awk "/VmRSS/{if (\$2/1048576>=6) print 1}" /proc/$p/status 2>/dev/null; done | wc -l)
conns=$(PATH=$PATH:/usr/sbin:/sbin ss -tnp state established "( dport = :23150 )" 2>/dev/null | grep -oE "pid=[0-9]+" | sort -u | wc -l)
zomb=$((workers - conns)); [ "$zomb" -lt 0 ] && zomb=0
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
echo "[$TS] $suite/k3-$layer progress=$done/$TOTAL (${pct}%) ok=$ok runner=$runner workers=$workers server=$srv err=$err cexc=$cexc restarts=$restarts bigW=$big zomb=$zomb freeGB=$mem"
[ "$done" -ge "$TOTAL" ] && echo "RIT GROUP DONE"
grep -q "RIT_RUN_EXIT" "$LOG" 2>/dev/null && [ "$done" -lt "$TOTAL" ] && echo "ALERT runner exited at $done"
[ "$srv" = "DOWN" ] && echo "ALERT server DOWN"
[ "$mem" -lt 30 ] && echo "ALERT low memory freeGB=$mem bigW=$big"
[ "$big" -ge 4 ] && echo "ALERT ballooned workers bigW=$big (kill by pid: RSS>=6GB worker_entry owned by me)"
[ "$zomb" -ge 2 ] && echo "ALERT zombie workers zomb=$zomb (workers without WS to :23150; kill by pid)"
[ "$runner" -eq 0 ] && [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ] && echo "ALERT runner dead at $done"
exit 0
