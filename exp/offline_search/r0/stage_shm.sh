#!/bin/bash
# stage_shm.sh -- pin the hot part of the offline-search store in RAM (tmpfs /dev/shm) so every run
# memmaps from memory instead of re-reading the SSD. Idempotent: files already present with the same size are
# skipped. Token/image subdirs (tok/) stay on the SSD and are symlinked; profile_cache/ is symlinked too so the
# profile tools keep writing their caches to persistent storage. Lost on reboot -> just rerun this script.
# usage: bash stage_shm.sh [SRC] [DST]     then use --root $DST for harness/profile commands.
set -euo pipefail
SRC=${1:-/home/weiland/trace_runs/offline_search_store}
DST=${2:-/dev/shm/offline_search_store}
mkdir -p "$DST"
copy_tree() {  # $1 = subdir (queries | library | floor)
  local sub=$1
  (cd "$SRC" && find "$sub" -path '*/tok' -prune -o -path '*/_run' -prune -o -name '*.partial' -prune -o -type f -print) |
  while read -r f; do
    local s="$SRC/$f" d="$DST/$f"
    if [ -e "$d" ] && [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ]; then continue; fi
    mkdir -p "$(dirname "$d")"; cp "$s" "$d.tmp" && mv "$d.tmp" "$d"
  done
  (cd "$SRC" && find "$sub" -type d -name tok -not -path '*.partial*') |
  while read -r t; do
    mkdir -p "$(dirname "$DST/$t")"
    [ -L "$DST/$t" ] || ln -s "$SRC/$t" "$DST/$t"
  done
}
for sub in queries library floor; do copy_tree "$sub"; done
mkdir -p "$SRC/profile_cache"
[ -L "$DST/profile_cache" ] || ln -s "$SRC/profile_cache" "$DST/profile_cache"
# verify: every non-tok file present with identical size
bad=$( (cd "$SRC" && find queries library floor -path '*/tok' -prune -o -path '*/_run' -prune -o -name '*.partial' -prune -o -type f -print) |
  while read -r f; do [ "$(stat -c %s "$SRC/$f")" = "$(stat -c %s "$DST/$f" 2>/dev/null || echo x)" ] || echo "$f"; done | wc -l)
echo "$(date '+%F %T %Z') staged $SRC -> $DST  mismatches=$bad  shm used: $(du -sh "$DST" | cut -f1)"
[ "$bad" = 0 ]
