#!/bin/bash
# Emit a STALL line when a live remote driver's journal has not grown for > 12 min (network resets can leave drivers idle
# while their tmux stays up; the chains then wait forever). Seen-file per (arm, journal line count) keeps it lossless.
SEEN=/home/weiland/.claude/jobs/a607dd74/tmp/stall_watch.seen; touch $SEEN
while true; do
  out=$(timeout 120 tether exec timan107 -- bash -c 'now=$(date +%s); for s in $(tmux -L oscl ls -F "#{session_name}" 2>/dev/null); do a=${s#oscl_}; j=$(ls -d /scratch/zixuans8/openpi_trace/os_cl/runs/*/$a 2>/dev/null | head -1)/journal.jsonl; [ -f "$j" ] || continue; age=$(( now - $(stat -c %Y $j) )); echo "$a $(wc -l < $j) $age"; done' 2>/dev/null)
  while read -r a n age; do
    [ -z "$a" ] && continue
    if [ "${age:-0}" -gt 720 ]; then key="$a $n"; grep -qxF "$key" $SEEN || { echo "STALL $a journal=$n idle=${age}s $(date +%H:%M)"; echo "$key" >> $SEEN; }; fi
  done <<< "$out"
  sleep 300
done
