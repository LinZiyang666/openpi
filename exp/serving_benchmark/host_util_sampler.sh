#!/bin/bash
# Sample CPU + GPU utilisation every interval seconds, append one CSV row per tick.
# Usage:
#   host_util_sampler.sh <output.csv> [interval_s=5]
#
# Columns: ts,gpu_idx,gpu_util,gpu_mem_used_mb,cpu_us,cpu_sy

set -euo pipefail
OUT="${1:?output csv path required}"
INTERVAL="${2:-5}"

if [ ! -f "$OUT" ]; then
  echo "ts,gpu_idx,gpu_util,gpu_mem_used_mb,cpu_us,cpu_sy" > "$OUT"
fi

while true; do
  TS=$(date +%Y-%m-%dT%H:%M:%S)
  CPU=$(top -bn1 | awk '/Cpu\(s\)/ {printf "%.1f,%.1f", $2, $4; exit}')
  # nvidia-smi: index,util,memory.used (no units)
  while IFS=, read -r idx util mem; do
    printf "%s,%s,%s,%s,%s\n" "$TS" "$(echo $idx | tr -d ' ')" "$(echo $util | tr -d ' ')" "$(echo $mem | tr -d ' ')" "$CPU" >> "$OUT"
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  sleep "$INTERVAL"
done
