#!/bin/bash
# collect_pod.sh POD MODEL SEED PUBLIC_PORT DRIVER_PORT [extra library_campaign args]: one MT50 library seed on a jupyter pod (collect only).
POD=$1; MODEL=$2; SEED=$3; PORT=$4; DRIVER=$5; shift 5
F=/data/r17_frozen/openpi_5f026a53
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
export R17_H100_NODE=$POD R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=$DRIVER
cd $F
TAG=${MODEL%_mw}; RUN=${COLLECT_RUN:-w3_${TAG:0:2}_s${SEED}}
echo "$(date +%H:%M:%S) [collect-$POD] start $RUN" >> $M/chain.log
bash $M/collect_seed.run.sh $F $MODEL h100 $SEED /data/r17_library/seed$SEED $RUN --collect-only \
  --worker-counts timan107=40,timan1=36 --port $PORT --driver-port $DRIVER "$@"
echo "$(date +%H:%M:%S) [collect-$POD] $RUN ended rc=$?" >> $M/chain.log
