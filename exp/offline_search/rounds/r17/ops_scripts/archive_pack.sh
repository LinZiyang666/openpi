#!/bin/bash
# archive_pack.sh NAME BASE LIST: pack the paths in LIST (relative to BASE) into one sequential
# /archive/openpi/nvme_cold_1024/NAME.tar.zst (/archive is btrfs on an SMR HC620: one large stream, one writer),
# compare the pack against the originals byte by byte (tar -d), and only then delete the originals.
# The pack's member listing goes beside it (NAME.list); /nvme/weiland/openpi/ARCHIVED_1024.tsv maps path -> pack.
set -uo pipefail
NAME=$1; BASE=$2; LIST=$3
A=/archive/openpi/nvme_cold_1024; P=$A/$NAME.tar.zst; INDEX=/nvme/weiland/openpi/ARCHIVED_1024.tsv
mkdir -p $A
log(){ echo "$(date +%T) [archive] $*"; }
[ -e $P ] && { log "$NAME: $P exists, refusing"; exit 2; }
n=$(wc -l < $LIST); size=$(cd $BASE && du -sc --block-size=1G $(cat $LIST) | tail -1 | cut -f1)
log "$NAME: packing $n paths (${size}G) from $BASE"
tar -C $BASE -cf - -T $LIST | zstd -T8 -3 -q -o $P || { log "$NAME: PACK_FAIL"; rm -f $P; exit 1; }
sync -f $P
log "$NAME: packed $(du -h $P | cut -f1); comparing with the originals"
diff=$(zstd -dc $P | tar -C $BASE -df - 2>&1); rc=$?
if [ $rc -ne 0 ] || [ -n "$diff" ]; then log "$NAME: COMPARE_FAIL rc=$rc"; echo "$diff" | head -20; exit 1; fi
zstd -dc $P | tar -tf - > $A/$NAME.list
sha=$(sha256sum $P | cut -c1-64)
log "$NAME: identical ($(wc -l < $A/$NAME.list) members, sha256 $sha); deleting originals"
while read -r rel; do rm -rf -- "$BASE/$rel" && printf '%s\t%s\t%s\n' "$BASE/$rel" "$P" "$sha" >> $INDEX; done < $LIST
log "$NAME: done; /nvme now $(df -h /nvme | tail -1 | awk '{print $4" free, "$5" used"}')"
