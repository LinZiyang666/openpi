#!/usr/bin/env bash
# Coordinator only. No transfer/sync, no automatic port selection.
set -euo pipefail
K1_QUEUE=${1:?absolute queue.txt}
K1_WORKERS=${2:-64}
K1_SNAPSHOT=${R16_K1REF_SNAPSHOT:?set immutable snapshot directory}
export R16_K1REF_OUTPUT=${R16_K1REF_OUTPUT:-/home/weiland/projects/openpi/exp/offline_search/rounds/r16/k1ref}
cd "$K1_SNAPSHOT"
K1_CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.snapshot verify --destination "$K1_SNAPSHOT"
while IFS= read -r K1_LINE || [[ -n "$K1_LINE" ]]; do
  [[ -z "$K1_LINE" ]] && continue
  [[ "$K1_LINE" == '@local /'* && "$K1_LINE" != *'|'* ]] || { echo 'Invalid k1ref queue' >&2; exit 2; }
  "${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.runner run --group "${K1_LINE#@local }" \
    --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --workers "$K1_WORKERS" --port 23159
done < "$K1_QUEUE"
