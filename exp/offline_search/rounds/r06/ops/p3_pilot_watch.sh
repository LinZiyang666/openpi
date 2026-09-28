#!/bin/bash
# Pilot events only: lane problems, first finished arm, and each completed cell (27 arms). Seen-file makes re-arms lossless.
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
SEEN=/home/weiland/.claude/jobs/a607dd74/tmp/p3_pilot_watch.seen; touch $SEEN
emit() { grep -qxF "$1" $SEEN || { echo "$1"; echo "$1" >> $SEEN; }; }
while true; do
  for f in $RUN/lane_*.log; do
    [ -e "$f" ] || continue
    grep -E "ARM_GAVE_UP|REMOTE_CLEANUP_FAILED|CLEANUP_SKIPPED|LANE_END|QUEUE_EMPTY|STOP file" "$f" | while read -r l; do emit "PILOT $(basename $f .log): $l"; done
  done
  n=$(ls $RUN/state/ 2>/dev/null | grep -c '\.DONE$')
  [ "$n" -ge 1 ] && emit "PILOT first arm done ($(ls -t $RUN/state/ | grep '\.DONE$' | tail -1 | sed 's/.manifest.*//'))"
  for c in pi05_l10_50 groot_sp_50 pi05_sp_50 groot_l10_50 pi05_l10_500 groot_l10_500 pi05_sp_500 groot_sp_500; do
    k=$(ls $RUN/state/ 2>/dev/null | grep "^r6p3v2_${c}_" | grep -c '\.DONE$')
    [ "$k" -ge 27 ] && emit "PILOT cell $c complete (27/27), total $n/216"
  done
  sleep 60
done
