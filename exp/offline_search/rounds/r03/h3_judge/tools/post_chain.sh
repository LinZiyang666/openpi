#!/usr/bin/env bash
# After tools/run_all.sh: replay the shadow triggers on the closed-loop logs (final defaults), build tables (a)/(c),
# emit arms_mx.json.   nohup taskset -c 26-29,70-73 bash exp/offline_search/rounds/r03/h3_judge/tools/post_chain.sh > <log> 2>&1 &
set -u
cd /home/weiland/projects/openpi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
D=/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge
T=exp/offline_search/rounds/r03/h3_judge/tools
while [ ! -e $D/CHAIN_DONE ]; do sleep 20; done
echo "[$(date +%T)] chain done; replay"
.venv/bin/python $T/replay_cl.py                      # every complete cl2 / cl3 arm (GR00T included when complete)
echo "[$(date +%T)] tables"
.venv/bin/python $T/tables_ac.py > $D/tables/tables_ac.stdout 2>&1
echo "[$(date +%T)] arms"
.venv/bin/python $T/make_arms.py
echo "[$(date +%T)] POST_DONE"
touch $D/POST_DONE
