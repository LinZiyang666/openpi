#!/bin/bash
# guard_wait.sh [SECONDS]: block until an incident or SECONDS pass, then print a compact status.
C=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/chain.log
W=/home/weiland/projects/openpi_ext/scratch/r17_ops/server_watch.log
A=/home/weiland/projects/openpi_ext/scratch/r17_ops/server_watch.alerts.log; touch $A
T=${1:-540}; start=$(wc -l < $C); astart=$(wc -l < $A); end=$((SECONDS+T)); why=timeout
while [ $SECONDS -lt $end ]; do
  new=$(tail -n +$((start+1)) $C | grep -E "STOPPED|FAILED|receipt=FAIL|receipt=PARTIAL" | grep -v "hold file present")
  [ -n "$new" ] && { why="chain: $new"; break; }
  pgrep -f "[s]erver_watch.py" >/dev/null || { why="server_watch not running"; break; }
  newa=$(tail -n +$((astart+1)) $A | grep -A3 "ALERT" | grep -v "^  batches\|^  servers\|^  workers")
  [ -n "$newa" ] && { why="server_watch alert: $newa"; break; }
  pgrep -f "[j]upyter_watch.sh" >/dev/null || { why="jupyter_watch not running"; break; }
  sleep 15
done
echo "$(date +%T) wake: $why"
tail -n +$((start+1)) $C | tail -6
tail -1 $W | python3 -c "
import json,sys
d=json.loads(sys.stdin.read()); print('watch', d['time'], ' '.join(f\"{k}:gpu{v['gpu_mean']:.0f}\" for k,v in d['servers'].items()))
for k,v in d['batches'].items(): print(f\"  {k:22s} active {v['active']:4} desired {v['desired']:4} rate {v['rate']}\")"
