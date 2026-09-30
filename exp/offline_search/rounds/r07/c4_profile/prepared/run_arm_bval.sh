#!/usr/bin/env bash
# Coordinator-only; ordinary plugin arms with audited B initial states.
set -euo pipefail
suite=${1:?suite}; arm=${2:?arm}
case "$arm" in r7_*_profile) ;; *) echo 'R7 PROFILE arm required' >&2; exit 2;; esac
case "$suite" in libero_10|l10) short=l10; full=libero_10;; libero_spatial|spatial|sp) short=spatial; full=libero_spatial;; *) exit 2;; esac
island=/scratch/zixuans8/openpi_trace/os_cl
record="$island/r7_profile_bval/bval_pool_${short}.yaml"
pool="/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero/$full"
test -f "$record" && test -d "$pool"
for f in "$pool"/*.pruned_init; do
  if [ -e "$f" ]; then echo "B pool shadowed by $f" >&2; exit 2; fi
done
: "${OSCL_MANIFEST:?exact R7 B-val manifest required}"
# Last flags override the stock launcher's A defaults, retaining digest checks.
exec bash "$island/run_arm_v2.sh" "$@" --apool-record "$record" --apool-dir "$pool"
