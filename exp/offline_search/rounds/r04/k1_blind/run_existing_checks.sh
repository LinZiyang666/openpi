#!/usr/bin/env bash
set -euo pipefail
out=exp/offline_search/rounds/r04/k1_blind/results
py=(taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python)
for model in pi05 groot; do
  for scale in 50 500; do
    lib=current; kr=5
    if [[ $scale == 500 ]]; then lib=big; kr=8; fi
    "${py[@]}" exp/offline_search/rounds/r02/g3_recovery/tools/contract_check.py \
      --base exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM \
      --base-kwargs "{\"lib\":\"$lib\",\"kref\":$kr}" --cell "${model}_l10_cache" \
      --episodes 2 --root /home/weiland/trace_runs/offline_search_store --out "$out/g3_${model}_${scale}" \
      > "$out/g3_${model}_${scale}.log" 2>&1
  done
  "${py[@]}" -m exp.offline_search.closed_loop.selftest \
    --cell "${model}_spatial_cache" --yaml "exp/trace_dual/config/tr_${model}_sp_cache.yaml" \
    --method exp.offline_search.closed_loop.probe:ProbeB0 --episodes 2 \
    --root /home/weiland/trace_runs/offline_search_store --out "$out/selftest_old_${model}" \
    > "$out/selftest_old_${model}.log" 2>&1
  "${py[@]}" -m exp.offline_search.closed_loop.selftest \
    --cell "${model}_spatial_cache" --yaml "exp/trace_dual/config/tr_${model}_sp_cache.yaml" \
    --method exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM \
    --kwargs '{"lib":"current","kref":5,"budget":0}' --episodes 2 \
    --root /home/weiland/trace_runs/offline_search_store --out "$out/selftest_b0_${model}" \
    > "$out/selftest_b0_${model}.log" 2>&1
  "${py[@]}" -m exp.offline_search.closed_loop.selftest \
    --cell "${model}_spatial_cache" --yaml "exp/trace_dual/config/tr_${model}_sp_cache.yaml" \
    --method exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM \
    --kwargs '{"lib":"current","kref":5,"budget":0}' --judge periodic:1 --replay-cell "${model}_spatial_inf" --episodes 2 \
    --root /home/weiland/trace_runs/offline_search_store --out "$out/selftest_miss_${model}" \
    > "$out/selftest_miss_${model}.log" 2>&1
done
