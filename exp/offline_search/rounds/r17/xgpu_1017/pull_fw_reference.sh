#!/usr/bin/env bash
# Coordinator supplies a new port after starting the existing narrow read-only daemon.
set -euo pipefail
port=${1:?Usage: bash pull_fw_reference.sh NEW_APPROVED_RSYNC_PORT}
if [[ ! $port =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1024 || 10#$port > 65535 || 10#$port == 23177 )); then
  echo 'Use an approved port in 1024..65535, excluding campaign port 23177.' >&2
  exit 2
fi
port=$((10#$port))
cat /proc/pressure/io
# One public rsync stream; no SSH, credentials, deletion, daemon or GPU launch.
# The server-side include-list limits these runs to request artifacts + receipts.
tether exec --timeout 10m h100 -- rsync -a --bwlimit=16384 --stats \
  '--include=/4090_fastwam_concurrent_fw4090_reduced_20261018T140530Z/***' \
  '--include=/4090_fastwam_concurrent_fw4090_reduced_20261018T141216Z/***' \
  '--exclude=*' \
  "rsync://ziyanglin.com:${port}/xc_local_results/" \
  /home/exouser/xgpu_1017/reference_results/
