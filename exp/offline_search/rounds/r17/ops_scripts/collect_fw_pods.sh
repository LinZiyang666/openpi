#!/bin/bash
# Fast-WAM MT50 library seeds on the jupyter pods (text table, R1 B8 15.5 GiB), after each pod finishes its GR00T seed.
# -02: wait w3_gr_s9 -> smoke -> w3_fw_s9.   pod1: wait w3_gr_s10 -> setup fastwam_nfs -> w3_fw_s10.
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
F=/data/r17_frozen/openpi_cc7a2c10
O=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/ops_scripts
FW=(--checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 --replicas 1 --memory-budget-gib 15.5
    --max-batch 8 --batch-wait-ms 5 --fastwam-text-table /data/mw_finetune/fastwam_text_tables/table_h100.pt)
log(){ echo "$(date +%H:%M:%S) [collect-fw-pods] $*" >> $M/chain.log; }
pod2(){
  until grep -q "\[collect-jupyter-ziyang10-02\] w3_gr_s9 ended" $M/chain.log; do sleep 30; done
  export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
  cd $F
  log "-02 smoke f9sm"
  bash $M/collect_seed.run.sh $F fastwam_mw h100 9 /data/r17_library/seed9 f9sm --collect-only --smoke \
    --worker-counts timan1=4 --port 23474 --driver-port 23192 "${FW[@]}" || { log "-02 smoke FAILED"; return 1; }
  log "-02 smoke ok; w3_fw_s9"
  bash $M/collect_seed.run.sh $F fastwam_mw h100 9 /data/r17_library/seed9 w3_fw_s9 --collect-only \
    --worker-counts timan107=40,timan1=36 --port 23474 --driver-port 23192 "${FW[@]}"
  log "-02 w3_fw_s9 ended rc=$?"
}
pod1(){
  until grep -q "\[collect-jupyter-ziyang10\] w3_gr_s10 ended" $M/chain.log && grep -q "\[collect-fw-pods\] -02 smoke ok" $M/chain.log; do
    grep -q "\[collect-fw-pods\] -02 smoke FAILED" $M/chain.log && return 1; sleep 30; done
  timeout 1500 tether exec --timeout 1500s jupyter-ziyang10 -- bash -c 'cd /home/ziyang10/zl_r17 && HOME=/home/ziyang10 bash setup.sh fastwam_nfs rsync://none/x 2>&1 | tail -2' >> $M/chain.log 2>&1
  export R17_H100_NODE=jupyter-ziyang10 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23193
  cd $F
  log "pod1 w3_fw_s10"
  bash $M/collect_seed.run.sh $F fastwam_mw h100 10 /data/r17_library/seed10 w3_fw_s10 --collect-only \
    --worker-counts timan107=40,timan1=36 --port 23470 --driver-port 23193 "${FW[@]}"
  log "pod1 w3_fw_s10 ended rc=$?"
}
pod2 & pod1 & wait
