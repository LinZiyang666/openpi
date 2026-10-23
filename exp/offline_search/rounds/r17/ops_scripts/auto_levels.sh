#!/bin/bash
# auto_levels.sh <ref job> <levels name>: when the ref ends rc=0 (and its knob-off is complete), build the levels (CPU) and exit.
C=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/chain.log
until grep -qE "\] $1 ended rc=" $C; do sleep 30; done
grep -E "\] $1 ended rc=" $C | tail -1
grep -qE "\] $1 ended rc=0" $C || { echo "$1 did not end rc=0"; exit 1; }
cd /home/weiland/projects/openpi && nice -n 10 bash exp/offline_search/rounds/r17/filter_failed_knob/commands/build_levels_$2.sh > /home/weiland/projects/openpi_ext/scratch/r17_ops/ff/build_levels_$2.log 2>&1
echo "build rc=$?"; grep -v Warning /home/weiland/projects/openpi_ext/scratch/r17_ops/ff/build_levels_$2.log | tail -4 | cut -c1-250
