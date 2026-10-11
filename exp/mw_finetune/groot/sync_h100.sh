#!/usr/bin/env bash
# Run locally: only B's code and already-converted data are transferred.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
remote_repo=${R17_H100_REPO:-/home/exouser/openpi}
dataset=${1:-/data/mw_finetune/datasets/metaworld_mt50_groot}
[[ -f $dataset/conversion/verification.json ]] || { echo 'Verify the converted dataset first' >&2; exit 1; }
code_tar=$(mktemp /tmp/r17_groot_code.XXXXXX.tar.gz)
parts_dir=$(mktemp -d /tmp/r17_groot_data.XXXXXX)
tar --exclude=__pycache__ -czf "$code_tar" -C "$repo" exp/mw_finetune/groot
# Broker ceilings can be below the advertised 2 GiB limit. Use small pieces.
tar -cf - -C "$(dirname "$dataset")" "$(basename "$dataset")" | split -b 512M - "$parts_dir/data.part-"
tether push --timeout 90s --force "$code_tar" h100:/tmp/r17_groot_code.tar.gz
tether exec --cwd /tmp --safe h100 mkdir -p "$remote_repo" /data/mw_finetune/datasets
tether exec --cwd /tmp --safe h100 tar -xzf /tmp/r17_groot_code.tar.gz -C "$remote_repo"
remote_prefix=/tmp/$(basename "$parts_dir")
for part in "$parts_dir"/data.part-*; do
  tether push --timeout 5m "$part" "h100:$remote_prefix.$(basename "$part")"
done
tether exec --cwd /tmp --safe h100 bash -c 'cat "$1".data.part-* | tar -xf - -C /data/mw_finetune/datasets' sync "$remote_prefix"
tether exec --cwd /tmp --safe h100 bash "$remote_repo/exp/mw_finetune/groot/setup.sh" h100
# Temporary tarballs are deliberately retained: shared rules require approval
# before deleting files. Their paths are printed for coordinator cleanup.
printf 'Retained transfer files: %s %s (local), %s.data.part-* (H100)\n' "$code_tar" "$parts_dir" "$remote_prefix"
