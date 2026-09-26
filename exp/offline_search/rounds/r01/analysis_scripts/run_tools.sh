#!/bin/bash
# breakdown / timeline / explain / runprof for the top methods; outputs under scratch/<tool>/
cd /home/weiland/projects/openpi
R1=exp/offline_search/results/r01
R0=exp/offline_search/results/r00
S=/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01
ROOT=/dev/shm/offline_search_store
P="taskset -c 0-37,44-81 .venv/bin/python -m exp.offline_search.profile"
TOP1="M1_big_a0p25_kmean5_stWin_b0,M8_b0big_k5_mean,M2sw_big_m3_k8_mean,M7_cascade_pca32_both_med3,M7_cascade_none_none_med3,M7_cascade_raw_both_med3,M6_ccf_b0_eq,M3hmm_big_a60b15g10_em1_cont_lev,M9c_vzsum_big_pca128_tm_both_st1_med3,Rtail_passthrough,M4_b0cons_k5_kernel,M5_b0sl_cont_linf_k3,M1_big_a0p25_med3_stWin_b0,M2sw_big_m3_k5_mean"
echo "=== breakdown r01 ==="
$P.breakdown $R1 --method $TOP1 --root $ROOT --procs 38 --out $S/breakdown 2>&1 | tail -3
echo "=== breakdown r00 B0 ==="
$P.breakdown $R0 --method B0_current,B3_oracle --root $ROOT --procs 38 --out $S/breakdown 2>&1 | tail -3
echo "=== timeline ==="
$P.timeline $R1 --method M1_big_a0p25_kmean5_stWin_b0,M8_b0big_k5_mean,M2sw_big_m3_k8_mean,M7_cascade_pca32_both_med3,M3hmm_big_a60b15g10_em1_cont_lev,M1_big_a0p25_med3_stWin_b0 --root $ROOT --procs 38 --no-csv --out $S/timeline 2>&1 | tail -3
$P.timeline $R0 --method B0_current,B3_oracle --root $ROOT --procs 38 --no-csv --out $S/timeline 2>&1 | tail -3
echo "=== explain ==="
for c in pi05_spatial_inf groot_l10_inf; do
  $P.explain $R1/M1_big_a0p25_kmean5_stWin_b0 --cell $c --worst 25 --root $ROOT --procs 38 --out $S/explain/M1k5__$c 2>&1 | tail -2
done
for c in pi05_spatial_cache groot_l10_cache; do
  $P.explain $R1/M8_b0big_k5_mean --cell $c --worst 25 --root $ROOT --procs 38 --out $S/explain/M8k5__$c 2>&1 | tail -2
done
echo "=== runprof ==="
$P.runprof summary $R1 --method M1_big_a0p25_kmean5_stWin_b0,M8_b0big_k5_mean,M2sw_big_m3_k8_mean,M7_cascade_pca32_both_med3,M7_cascade_raw_both_med3,M7_cascade_raw_b0fused_med3,M9c_vzsum_big_pca128_tm_both_st1_med3,M3hmm_big_a60b15g10_em1_cont_lev --root $ROOT --procs 38 --out $S/runprof 2>&1 | tail -3
$P.runprof summary $R0 --method B0_current --root $ROOT --procs 38 --out $S/runprof 2>&1 | tail -3
echo ALL_TOOLS_DONE
