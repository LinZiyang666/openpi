#!/bin/bash
# One-line client health for the L3 cron (timan107): sessions, exits, last progress per shard.
#
# usage: probe_clients.sh <suite> <k> [tag]
SUITE=${1:?suite}; K=${2:?k}; TAG=${3:-run}
PFX="nfecli_${TAG}_${SUITE}_k${K}_"
live=$(tmux ls 2>/dev/null | grep -c "^$PFX")
exits=$(grep -l "NFECLI_EXIT=" /tmp/nfe/${PFX}*.log 2>/dev/null | wc -l)
bad=$(grep -l "NFECLI_EXIT=[1-9]" /tmp/nfe/${PFX}*.log 2>/dev/null | wc -l)
tb=$(cat /tmp/nfe/${PFX}*.log 2>/dev/null | grep -c "Traceback")
dl=$(cat /tmp/nfe/${PFX}*.log 2>/dev/null | grep -c "in __del__")
prog=""
for f in /tmp/nfe/${PFX}*.log; do
  [ -f "$f" ] || continue
  s=${f##*_s}; s=${s%.log}
  p=$(tr '\r' '\n' < "$f" | grep -o "[0-9]*/[0-9]* \[" | tail -1 | tr -d ' [')
  prog="$prog s$s=${p:-?}"
done
mem=$(free -g | awk '/Mem:/{print $7}')
echo "NFE CLIENTS $SUITE k=$K live=$live exited=$exits bad_exit=$bad tb=$tb del=$dl freeGB=$mem$prog"
