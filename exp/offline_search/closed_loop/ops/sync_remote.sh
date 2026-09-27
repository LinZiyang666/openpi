#!/bin/bash
# sync_remote.sh <run-root> [<arm> ...]  -- push the timan107-side scripts and the arms' yamls / matrices into the
# island tree /scratch/zixuans8/openpi_trace/os_cl/. tether push only reaches /home /tmp /srv on timan107, so files
# are staged in /tmp/oscl_stage and copied into /scratch by a tether exec. Idempotent; prints sha256 of both sides.
set -u
RUN=${1:?run-root}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
ISL=/scratch/zixuans8/openpi_trace/os_cl
STAGE=/tmp/oscl_stage
rx() { timeout 300 tether exec timan107 -- bash -c "$1"; }
rx "mkdir -p $STAGE $ISL/cfg $ISL/runs" || exit 1
push() {  # $1 local file, $2 remote destination (absolute, under $ISL)
  local b; b=$(basename "$1")
  timeout 300 tether push --force "$1" "timan107:$STAGE/$b" >/dev/null || { echo "push failed: $1"; return 1; }
  rx "cp $STAGE/$b $2 && sha256sum $2 | cut -c1-16" | sed "s|^|remote $b |"
  echo "local  $b $(sha256sum "$1" | cut -c1-16)"
}
push "$HERE/remote/run_arm.sh" "$ISL/run_arm.sh" || exit 1
push "$HERE/remote/run_gtp_subset.py" "$ISL/run_gtp_subset.py" || exit 1
push "$HERE/remote/count.py" "$ISL/count.py" || exit 1
for arm in "$@"; do
  push "$RUN/config/$arm.yaml" "$ISL/cfg/$arm.yaml" || exit 1
  push "$RUN/config/matrix_$arm.yaml" "$ISL/cfg/matrix_$arm.yaml" || exit 1
done
echo SYNC_OK
