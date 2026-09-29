#!/usr/bin/env bash
# Coordinator-only remote wrapper. Never installed or executed by Q1.
# Same positional arguments as P3 run_arm_v2.sh. No shared launcher is edited.
set -euo pipefail
suite=${1:?suite};arm=${2:?arm}
case "$arm" in r6c_cal_*) ;; *) echo 'B-val launcher only accepts r6c_cal_* arms' >&2;exit 2;; esac
case "$suite" in libero_10|l10) short=l10; full=libero_10;; libero_spatial|spatial|sp) short=spatial; full=libero_spatial;; *) exit 2;; esac
island=/scratch/zixuans8/openpi_trace/os_cl
record="$island/q1_c_bval/bval_pool_${short}.yaml"
pool="/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero/$full"
test -f "$record" && test -d "$pool"
for f in "$pool"/*.pruned_init; do
  if [ -e "$f" ]; then echo "B-val pool shadowed by $f" >&2;exit 2;fi
done
if [ -z "${OSCL_MANIFEST:-}" ]; then echo 'exact B-val manifest required' >&2;exit 2;fi
exec bash "$island/run_arm_v2.sh" "$@" --apool-record "$record" --apool-dir "$pool"
