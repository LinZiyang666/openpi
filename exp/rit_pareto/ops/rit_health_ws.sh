#!/bin/bash
# rit_health_ws.sh <suite> <layer> <total_ep>  — weilandserver-hosted fleet
export PATH=/usr/local/bin:/usr/bin:/bin
suite=$1; layer=$2; TOTAL=$3
OUT=/tmp/dsp_shared/rit_pareto/runs/${suite}_${layer}
J=$OUT/journal.jsonl; LOG=/tmp/rit_ws_${suite}_${layer}.log
TS=$(date "+%H:%M:%S")
if [ -f "$J" ]; then
  done=$(grep -cE '"status": ?"(done|failed)"' "$J" 2>/dev/null); [ -z "$done" ] && done=0
  ok=$(grep -cE '"status": ?"done"' "$J" 2>/dev/null); [ -z "$ok" ] && ok=0
else done=0; ok=0; fi
runner=$(pgrep -f "[r]un_gtp" | wc -l)
workers=$(pgrep -f "[w]orker_entry" | wc -l)
srv=$(timeout 4 bash -c "echo > /dev/tcp/127.0.0.1/23150" 2>/dev/null && echo UP || echo DOWN)
err=$(grep -ciE "FatalError|refused|out of memory|CUDA error|Killed|1011 |worker .* (died|exited|crash)" "$LOG" 2>/dev/null); [ -z "$err" ] && err=0
mem=$(free -g | awk '/Mem:/{print $7}')
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)
pct=$(awk "BEGIN{printf \"%.1f\", $done*100.0/$TOTAL}")
echo "[$TS] ws $suite/$layer progress=$done/$TOTAL (${pct}%) ok=$ok runner=$runner workers=$workers server=$srv err=$err freeGB=$mem gpu=$gpu"
[ "$done" -ge "$TOTAL" ] && echo "RIT GROUP DONE"
grep -q "RIT_RUN_EXIT" "$LOG" 2>/dev/null && [ "$done" -lt "$TOTAL" ] && echo "ALERT runner exited at $done"
[ "$srv" = "DOWN" ] && echo "ALERT server DOWN"
[ "$runner" -eq 0 ] && [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ] && echo "ALERT runner dead at $done"
exit 0
