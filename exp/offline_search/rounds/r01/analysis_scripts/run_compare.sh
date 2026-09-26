#!/bin/bash
# paired compares (episode-bootstrap CIs); outputs under scratch/compare/<A>__vs__<B>/
cd /home/weiland/projects/openpi
R1=exp/offline_search/results/r01
R0=exp/offline_search/results/r00
OUT=/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01/compare
mkdir -p $OUT
PY="taskset -c 0-37,44-81 .venv/bin/python -m exp.offline_search.profile.compare"
run() {  # A B
  a=$1; b=$2
  echo "=== $a vs $b ==="
  $PY $a $b --root /dev/shm/offline_search_store --procs 38 --reps 2000 --out $OUT/$(basename $a)__vs__$(basename $b) 2>&1 | grep -v "^\[compare\]" | head -14
}
run $R1/M1_big_a0p25_kmean5_stWin_b0 $R0/B0_current
run $R1/M1_big_a0p25_kmean5_stWin_b0 $R1/Rtail_passthrough
run $R1/M1_big_a0p25_kmean5_stWin_b0 $R1/M1_big_a0p25_med3_stWin_b0
run $R1/M1_big_a0p25_kmean5_stWin_b0 $R1/M1_big_a0p25_top1_stWin_b0
run $R1/M8_b0big_k5_mean $R0/B0_current
run $R1/M8_b0big_k5_mean $R1/M2sw_big_m3_k8_mean
run $R1/M8_b0big_k5_mean $R1/M8_b0big_top1
run $R1/M2sw_big_m3_k8_mean $R0/B0_current
run $R1/M2sw_big_m3_k5_mean $R1/M2sw_current_m3_k5_mean
run $R1/M7_cascade_pca32_both_med3 $R1/M7_cascade_none_none_med3
run $R1/M7_cascade_raw_both_med3 $R1/M7_cascade_pca32_both_med3
run $R1/M7_cascade_pca32_both_med3 $R1/M1_big_a0p25_med3_stWin_b0
run $R1/M3hmm_big_a60b15g10_em1_cont_lev $R1/M2sw_big_m1_k5_mean
run $R1/M4_b0cons_k5_kernel $R0/B0_current
run $R1/M5_b0sl_cont_linf_k3 $R1/M4_b0cons_k3_med
run $R1/M6_ccf_b0_eq $R0/B0_current
run $R1/M9c_vzsum_big_pca128_tm_both_st1_med3 $R1/M8_b0big_k5_mean
run $R1/M1_big_a0p25_kmean5_stWin_b0 $R1/M8_b0big_k5_mean
run $R1/M7_cascade_raw_b0fused_med3 $R1/M8_b0big_k5_mean
run $R1/M8_b0big_k5_mean $R1/M7_cascade_raw_both_med3
run $R1/M2sw_big_m3_k8_mean $R1/M7_cascade_none_none_med3
echo ALL_COMPARES_DONE
