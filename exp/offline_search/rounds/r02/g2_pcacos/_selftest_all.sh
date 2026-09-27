#!/usr/bin/env bash
# Closed-loop CPU selftests (exp/offline_search/closed_loop/selftest.py) of the CL controls and the G2 methods.
# Usage: bash _selftest_all.sh [procs] [episodes]
set -u
cd /home/weiland/projects/openpi
P=${1:-3}
EPS=${2:-20}
OUT=/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/selftest
CFG=/home/weiland/trace_runs/os_closed_loop/smoke_20260926/config
mkdir -p $OUT/logs
F3=exp/offline_search/rounds/r01/f3_b0plus/method.py
G2=exp/offline_search/rounds/r02/g2_pcacos
cat > $OUT/jobs.txt <<J
cl0_b0big_top1	$F3:B0BigLibCons	{"k":1,"synth":"top1"}	pi05_spatial_cache	oscl_smk_pi05_sp_probe
cl0_b0big_top1	$F3:B0BigLibCons	{"k":1,"synth":"top1"}	groot_spatial_cache	oscl_smk_groot_sp_probe
cl1_b0cons_k5mean	$F3:B0TopkConsensus	{"k":5,"synth":"mean"}	pi05_spatial_cache	oscl_smk_pi05_sp_probe
cl1_b0cons_k5mean	$F3:B0TopkConsensus	{"k":5,"synth":"mean"}	groot_spatial_cache	oscl_smk_groot_sp_probe
m8x_top1	$G2/m8fast.py:B0BigLibConsFast	{"k":1,"synth":"top1"}	pi05_spatial_cache	oscl_smk_pi05_sp_probe
m8x_top1	$G2/m8fast.py:B0BigLibConsFast	{"k":1,"synth":"top1"}	groot_spatial_cache	oscl_smk_groot_sp_probe
m8x_top1	$G2/m8fast.py:B0BigLibConsFast	{"k":1,"synth":"top1"}	pi05_l10_cache	oscl_smk_pi05_l10_probe
v4_big	$G2/method.py:PcaCosV4	{}	pi05_spatial_cache	oscl_smk_pi05_sp_probe
v4_big	$G2/method.py:PcaCosV4	{}	groot_spatial_cache	oscl_smk_groot_sp_probe
v4_big	$G2/method.py:PcaCosV4	{}	groot_l10_cache	oscl_smk_groot_l10_probe
v4_cur	$G2/method.py:PcaCosV4	{"lib":"current"}	pi05_spatial_cache	oscl_smk_pi05_sp_probe
v4_cur	$G2/method.py:PcaCosV4	{"lib":"current"}	groot_spatial_cache	oscl_smk_groot_sp_probe
J
run_one() {
  IFS=$'\t' read -r tag meth kw cell yml <<< "$1"
  d=$OUT/${tag}_$cell
  extra=()
  case $tag in v4_*) extra=(--fit-artifact $OUT/fits/${tag}_$cell.pkl);; esac
  mkdir -p $OUT/fits
  taskset -c 9-17,53-61 .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell $cell --yaml $CFG/$yml.yaml \
      --method "$meth" --kwargs "$kw" --episodes $EPS --out $d "${extra[@]}" > $OUT/logs/${tag}_$cell.log 2>&1
  rc=$?
  # fit-artifact methods: a second run LOADS the pickle written by the first
  if [ ${#extra[@]} -gt 0 ] && [ $rc -eq 0 ]; then
    taskset -c 9-17,53-61 .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell $cell --yaml $CFG/$yml.yaml \
        --method "$meth" --kwargs "$kw" --episodes $EPS --out ${d}_pkl "${extra[@]}" > $OUT/logs/${tag}_${cell}_pkl.log 2>&1
    rc2=$?
    echo "$tag $cell rc=$rc pkl_rc=$rc2"
  else
    echo "$tag $cell rc=$rc"
  fi
}
export -f run_one
export OUT CFG EPS
tr '\n' '\0' < $OUT/jobs.txt | xargs -0 -P $P -I{} bash -c 'run_one "$@"' _ {}
