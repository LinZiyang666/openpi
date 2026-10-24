#!/bin/bash
# pod1 queue after its GR00T seed 10: redo GR00T seed 9 (the -02 run w3_gr_s9 failed on a shard transfer, rsync code 35, at 2,289/2,500),
# then set up Fast-WAM and collect Fast-WAM seed 10 (after the -02 Fast-WAM smoke passed).
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
O=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/ops_scripts
log(){ echo "$(date +%H:%M:%S) [pod1-q] $*" >> $M/chain.log; }
until grep -q "\[collect-jupyter-ziyang10\] w3_gr_s10 ended" $M/chain.log; do sleep 30; done
log "GR00T seed 9 redo w3_gr_s9b"
COLLECT_RUN=w3_gr_s9b bash $O/collect_pod.sh jupyter-ziyang10 groot_mw 9 23470 23193 --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-60000 --replicas 1 --memory-budget-gib 14 >> $CLAUDE_JOB_DIR/tmp/collect_gr_s9b.log 2>&1
until grep -q "\[collect-fw-pods\] -02 smoke ok" $M/chain.log; do grep -q "\[collect-fw-pods\] -02 smoke FAILED" $M/chain.log && { log "FW smoke failed; stop"; exit 1; }; sleep 30; done
timeout 1500 tether exec --timeout 1500s jupyter-ziyang10 -- bash -c 'cd /home/ziyang10/zl_r17 && HOME=/home/ziyang10 bash setup.sh fastwam_nfs rsync://none/x 2>&1 | tail -2' >> $M/chain.log 2>&1
F=/data/r17_frozen/openpi_cc7a2c10
export R17_H100_NODE=jupyter-ziyang10 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23193
cd $F; log "w3_fw_s10"
bash $M/collect_seed.run.sh $F fastwam_mw h100 10 /data/r17_library/seed10 w3_fw_s10 --collect-only \
  --worker-counts timan107=40,timan1=36 --port 23470 --driver-port 23193 \
  --checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 --replicas 1 --memory-budget-gib 15.5 \
  --max-batch 8 --batch-wait-ms 5 --fastwam-text-table /data/mw_finetune/fastwam_text_tables/table_h100.pt
log "w3_fw_s10 ended rc=$?"
