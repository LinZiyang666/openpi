#!/bin/bash
# Pull closed runs' H100 servers/ dirs back to /nvme (bwlimit), verify by checksum, then report. Deletion on H100 is a separate step.
D=/nvme/weiland/openpi/h100_pullback_1024
for r in "$@"; do
  echo "$(date +%T) pull $r"; mkdir -p $D/$r
  rsync -a --bwlimit=15000 --timeout=120 rsync://149.165.153.233:23117/eval/$r/servers/ $D/$r/servers/ || { echo "$(date +%T) PULL_FAIL $r"; continue; }
  diff=$(rsync -anc --out-format='%n' rsync://149.165.153.233:23117/eval/$r/servers/ $D/$r/servers/ | grep -v '/$' | wc -l)
  echo "$(date +%T) verified $r checksum_diff_files=$diff size=$(du -sh $D/$r | cut -f1)"
done
echo "$(date +%T) done"
