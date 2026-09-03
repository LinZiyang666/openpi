#!/bin/bash
# finish_export.sh <suite>  — export (tau1) -> emit -> package, assuming table_tau1 passed the guard
set -uo pipefail
SUITE=$1
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
B=/tmp/dsp_shared/rit_pareto/$SUITE
PKL=/home/weiland/openpi/exp/common/data/cache_artifacts/$SUITE/cp1_spatial_pool_16.pkl
CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PY=/home/weiland/openpi/.venv/bin/python
cd $R
export PYTHONPATH=$R/src:$R
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
step export tau1
rm -rf $B/export_tau1
$PY -m exp.rit_pareto.export_rit fit --suite $SUITE --table $B/table_tau1.jsonl --weights-npz $B/table_tau1.jsonl.weights.npz --cache-yaml $B/calibration_retrieval.yaml --library-pkl $PKL --checkpoint-dir $CKPT --ref-mode tau1 --out-dir $B/export_tau1 2>&1 | grep -v -iE "warning|pynvml" || { echo FINISH_FAILED export; exit 1; }
step emit
rm -rf $B/arms
$PY -m exp.rit_pareto.emit_arms --export-record $B/export_tau1/export_record.json --suite $SUITE --library-pkl $PKL --out-dir $B/arms 2>&1 | grep -v -iE "warning|pynvml" || { echo FINISH_FAILED emit; exit 1; }
step package
cd /tmp/dsp_shared/rit_pareto && tar -czf /tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz $SUITE/export_tau1 $SUITE/arms && sha256sum /tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz
echo FINISH_DONE
