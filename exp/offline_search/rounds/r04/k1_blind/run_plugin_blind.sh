#!/usr/bin/env bash
set -euo pipefail
out=exp/offline_search/rounds/r04/k1_blind/results
py=(taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python)
for model in pi05 groot; do
  for scale in 50 500; do
    lib=current; kr=5
    if [[ $scale == 500 ]]; then lib=big; kr=8; fi
    for serving in phase_particles anchor_tail; do
      tag="plugin_${model}_${scale}_${serving}"
      "${py[@]}" -m exp.offline_search.closed_loop.selftest --blind \
        --cell "${model}_spatial_cache" --yaml "exp/trace_dual/config/tr_${model}_sp_cache.yaml" \
        --method exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM \
        --kwargs "{\"lib\":\"$lib\",\"kref\":$kr,\"serving\":\"$serving\",\"budget\":2,\"gates\":\"budget_only\"}" \
        --judge periodic:5 --episodes 2 --root /home/weiland/trace_runs/offline_search_store \
        --out "$out/$tag" > "$out/$tag.log" 2>&1
    done
    if [[ $model == pi05 ]]; then
      tag="plugin_mixed_${model}_${scale}"
      "${py[@]}" -m exp.offline_search.closed_loop.selftest --blind \
        --cell "${model}_spatial_cache" --yaml "exp/trace_dual/config/tr_${model}_sp_cache.yaml" \
        --method exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge \
        --kwargs "{\"base_kwargs\":{\"lib\":\"$lib\",\"kref\":$kr,\"budget\":2,\"gates\":\"budget_only\"},\"ncal\":64}" \
        --judge guard_only --episodes 2 --root /home/weiland/trace_runs/offline_search_store \
        --out "$out/$tag" > "$out/$tag.log" 2>&1
    fi
  done
done
