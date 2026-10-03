#!/bin/bash
# collect.sh <arm> ...  -- pull timan107 journal/per_step/driver.log of each group into runs/<arm>/client/ (sha-checked)
set -u
B=/home/weiland/trace_runs/dual_20260923; O=/scratch/zixuans8/trace_runs/dual_20260923
for arm in "$@"; do
  mkdir -p $B/runs/$arm/client
  timeout 300 tether exec timan107 -- bash -c "cd $O && tar -cf /tmp/trc_$arm.tar $arm && sha256sum /tmp/trc_$arm.tar" | tee /tmp/trc_remote_sha_$arm
  rm -f /tmp/trc_local_$arm.tar
  timeout 300 tether pull timan107:/tmp/trc_$arm.tar /tmp/trc_local_$arm.tar | tail -1
  r=$(cut -c1-64 /tmp/trc_remote_sha_$arm); l=$(sha256sum /tmp/trc_local_$arm.tar | cut -c1-64)
  [ "$r" = "$l" ] || { echo "SHA MISMATCH $arm"; exit 1; }
  tar -xf /tmp/trc_local_$arm.tar -C $B/runs/$arm/client --strip-components=1 && rm /tmp/trc_local_$arm.tar
  echo "$arm collected: $(ls $B/runs/$arm/client | tr '\n' ' ')"
done
