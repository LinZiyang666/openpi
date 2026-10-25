#!/bin/bash
# zst_per_step.sh LIST: compress each finished run's driver/per_step.jsonl (paths in LIST) in place to per_step.jsonl.zst,
# compare the decompressed stream with the original byte by byte, and only then delete the original.
# Readers go through exp/offline_search/zst_logs (845d1514). Only for closed runs: RECEIPT PASS, whole continuation chain done.
# Two files at a time, low priority (the owner runs CPU-heavy jobs here). Log: path, original bytes, compressed bytes.
LIST=$1; LOG=/nvme/weiland/openpi/ZST_PER_STEP_1024.tsv
one(){
  f=$1
  [ -f "$f" ] && [ ! -e "$f.zst" ] || { echo "$(date +%T) [zst] skip $f"; return; }
  [ "$(stat -c %h "$f")" = 1 ] || { echo "$(date +%T) [zst] skip (hardlinked) $f"; return; }
  nice -n 10 ionice -c3 zstd -3 -T4 -q "$f" -o "$f.zst.tmp" && zstd -dcq "$f.zst.tmp" | cmp -s - "$f" || { echo "$(date +%T) [zst] FAIL $f"; rm -f "$f.zst.tmp"; return; }
  mv "$f.zst.tmp" "$f.zst" && printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(stat -c %s "$f.zst")" >> $LOG && rm "$f"
}
export -f one; export LOG
xargs -a "$LIST" -P 2 -I{} bash -c 'one "$@"' _ {}
echo "$(date +%T) [zst] done; /nvme $(df -h /nvme | tail -1 | awk '{print $4" free, "$5" used"}')"
