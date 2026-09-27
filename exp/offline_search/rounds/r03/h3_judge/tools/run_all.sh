#!/usr/bin/env bash
# H3 development chain (CPU range 26-29,70-73 = 8 logical): prefit artifacts (pickle sizes / fit time), the batch.json
# variants + the bare base on all 8 cells through harness.run (outputs under the derived dir, no scoreboard), then the
# plugin selftest (pure-cache mode) + verify_logs for the full variant.
#   nohup taskset -c 26-29,70-73 bash exp/offline_search/rounds/r03/h3_judge/tools/run_all.sh > <log> 2>&1 &
set -u
cd /home/weiland/projects/openpi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
D=/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge
S=/home/weiland/.claude/jobs/a607dd74/tmp/r03_h3
ROOT=/dev/shm/offline_search_store
J=exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge
AWM=exp/offline_search/rounds/r02/g1_awm/awm.py:AWM
mkdir -p $D/fits $D/runs $S
CUR='"base": "'$AWM'", "base_kwargs": {"lib": "current", "kref": 5}'
BIG='"base": "'$AWM'", "base_kwargs": {}'
FULL='"guards": true, "events": "all", "burst": 2, "ret_margin": 0.1'
GUARD='"guards": true, "events": "none"'
echo "[$(date +%T)] prefits"
prefit() {  # tag cell kwargs
  local t0=$(date +%s.%N)
  $PY -m exp.offline_search.closed_loop.plugin --os-method $J --os-kwargs "{$3}" --os-cell $2 \
      --os-log-dir $D/fits --os-fit-artifact $D/fits/$1.pkl > $D/fits/prefit_$1.log 2>&1
  local rc=$?
  echo "prefit $1 rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc) s size=$(stat -c %s $D/fits/$1.pkl 2>/dev/null)" | tee -a $D/fits/PREFITS.txt
}
( prefit mxj_p_sp_cur pi05_spatial_cache "$CUR, $FULL" ) &
( prefit mxj_p_l10_cur pi05_l10_cache "$CUR, $FULL" ) &
( prefit mxj_g_sp_cur groot_spatial_cache "$CUR, $FULL" ) &
( prefit mxj_g_l10_cur groot_l10_cache "$CUR, $FULL" ) &
wait
( prefit mxj_p_sp_big pi05_spatial_cache "$BIG, $FULL" ) &
( prefit mxj_p_l10_big pi05_l10_cache "$BIG, $FULL" ) &
( prefit mxj_g_sp_big groot_spatial_cache "$BIG, $FULL" ) &
( prefit mxj_g_l10_big groot_l10_cache "$BIG, $FULL" ) &
wait
echo "[$(date +%T)] harness runs"
runv() {  # method kwargs
  $PY -m exp.offline_search.harness.run --method "$1" --kwargs "$2" --cells all --root $ROOT --out $D/runs \
      --workers 8 --round r03 --family h3_judge --scoreboard none > $S/run_$(echo "$2" | md5sum | cut -c1-8).log 2>&1
  echo "[$(date +%T)] run rc=$? kwargs=$2"
}
runv $J "{$CUR, \"guards\": false, \"events\": \"none\"}"
runv $J "{$CUR, $GUARD}"
runv $J "{$CUR, $FULL}"
runv $J "{$CUR, \"guards\": false, \"events\": \"all\", \"burst\": 2, \"ret_margin\": 0.1}"
runv $AWM '{"lib": "current", "kref": 5}'
runv $J "{$BIG, $GUARD}"
runv $J "{$BIG, $FULL}"
echo "[$(date +%T)] selftest (pure-cache) + verify_logs"
for cell in pi05_spatial_cache groot_l10_cache; do
  out=$D/cl/selftest_$cell
  mkdir -p $D/cl
  $PY -m exp.offline_search.closed_loop.selftest --cell $cell --yaml exp/trace_dual/config/tr_$(echo $cell | sed 's/pi05_spatial/pi05_sp/;s/groot_spatial/groot_sp/;s/_cache//')_cache.yaml \
      --method $J --kwargs "{$CUR, $FULL}" --episodes 12 --out $out > $D/cl/selftest_$cell.log 2>&1
  echo "[$(date +%T)] selftest $cell rc=$?"
  $PY -m exp.offline_search.closed_loop.verify_logs --log-dir $out --work $D/cl/verify_work_$cell --out $D/cl/verify_$cell.json > $D/cl/verify_$cell.log 2>&1
  echo "[$(date +%T)] verify_logs $cell rc=$?"
done
echo "[$(date +%T)] CHAIN_DONE"
touch $D/CHAIN_DONE
