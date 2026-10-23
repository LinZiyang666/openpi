#!/bin/bash
# DP MT50 library seeds 9 then 10 on sean lane 2 (driver 23198, public 23280), collect only; finish later on a CPU host.
F=/data/r17_frozen/openpi_5f026a53
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
export R17_H100_NODE=h100-sean R17_H100_HOST=149.165.169.137 R17_DRIVER_PORT=23198 R17_PUBLIC_PORT=23280
cd $F
for s in 9 10; do
  [ -f $M/hold_collect_dp_sean2 ] && { echo "$(date +%H:%M:%S) [collect-dp-sean2] hold, stop before seed $s" >> $M/chain.log; break; }
  echo "$(date +%H:%M:%S) [collect-dp-sean2] start seed $s" >> $M/chain.log
  bash $M/collect_seed.run.sh $F dp_mw h100 $s /data/r17_library/seed$s w3_dp_s$s --collect-only \
    --checkpoint /data/mw_finetune/dp/mt50_policy_100000 --replicas 2 --memory-budget-gib 12 --max-batch 8 --batch-wait-ms 5 \
    --worker-counts timan107=40,timan1=36 --port 23280 --driver-port 23198
  echo "$(date +%H:%M:%S) [collect-dp-sean2] seed $s ended rc=$?" >> $M/chain.log
done
echo "$(date +%H:%M:%S) [collect-dp-sean2] queue done" >> $M/chain.log
