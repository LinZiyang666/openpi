#!/usr/bin/env bash
# Coordinator only. Use a run-specific launcher; no shared script is replaced.
set -euo pipefail
export P3_SNAPSHOT_DIR=${P3_SNAPSHOT_DIR:-${5:?client output directory}/snapshots}
export P3_SNAPSHOT_EVERY=${P3_SNAPSHOT_EVERY:-1}
export P3_SNAPSHOT_P=${P3_SNAPSHOT_P:-1}
stock=${P3_STOCK_RUN_ARM:-/scratch/zixuans8/openpi_trace/os_cl/run_arm.stock.sh}
test -f "$stock"
test "$(grep -c '^if .* then ENTRY=' "$stock")" -eq 1
sed '/^if .* then ENTRY=/c\ENTRY=(-m exp.offline_search.rounds.r06.p3_profiling.run_gtp_v2)' "$stock" | bash -s -- "$@"
