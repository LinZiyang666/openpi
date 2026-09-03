#!/bin/bash
# export_k3_remote.sh <suite> — on weilandserver: K3 export (rit + gst) from the tau1 K3 table -> /tmp/dsp_shared/rit_pareto/<suite>/k3
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
SUITE=$1
B=/tmp/dsp_shared/rit_pareto/$SUITE
PKL=/home/weiland/openpi/exp/common/data/cache_artifacts/$SUITE/cp1_spatial_pool_16.pkl
PY=/home/weiland/openpi/.venv/bin/python
cd $R
export PYTHONPATH=$R/src:$R
rm -rf $B/k3
$PY -m exp.rit_pareto.export_k3 --suite $SUITE --table $B/table_tau1_k3.jsonl --library-pkl $PKL --library-pkl-local $PKL --out-dir $B/k3 2>&1 | grep -v -iE "warning|pynvml" || { echo K3_EXPORT_FAILED; exit 1; }
cd /tmp/dsp_shared/rit_pareto && tar -czf /tmp/dsp_shared/rit_pareto/${SUITE}_k3_arms.tgz $SUITE/k3 && sha256sum /tmp/dsp_shared/rit_pareto/${SUITE}_k3_arms.tgz
echo K3_EXPORT_DONE
