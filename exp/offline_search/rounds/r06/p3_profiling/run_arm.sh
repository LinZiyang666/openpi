#!/usr/bin/env bash
# Coordinator/client host only: reuse the stock script, changing one entry line.
set -euo pipefail
export P3_SNAPSHOT_DIR=${P3_SNAPSHOT_DIR:-${5:?client output directory}/snapshots}
export P3_SNAPSHOT_EVERY=${P3_SNAPSHOT_EVERY:-1}
stock=${P3_STOCK_RUN_ARM:-/scratch/zixuans8/openpi_trace/os_cl/run_arm.stock.sh}
test -f "$stock"
test "$(grep -c '^if .* then ENTRY=' "$stock")" -eq 1
sed '/^if .* then ENTRY=/c\ENTRY=(-m exp.offline_search.rounds.r06.p3_profiling.run_gtp)' "$stock" | bash -s -- "$@"
