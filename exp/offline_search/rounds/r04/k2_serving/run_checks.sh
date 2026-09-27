#!/bin/bash
set -eu
cd /home/weiland/projects/openpi
OUT=exp/offline_search/rounds/r04/k2_serving/results
run() {
    local tag=$1 model=$2 suite=$3 method=$4
    shift 4
    local letter=p
    [ "$model" = groot ] && letter=g
    local cell_suite=spatial
    [ "$suite" = l10 ] && cell_suite=l10
    taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell "${model}_${cell_suite}_cache" --yaml "/home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_${letter}_${suite}_cl0.yaml" --method "$method" --episodes 2 --out "$OUT/$tag" "$@" > "$OUT/$tag.log" 2>&1
}
PROBE=exp.offline_search.closed_loop.probe
K1=exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge
run final_pure pi05 sp "$PROBE:ProbeB0" & p1=$!
run final_mixed pi05 sp "$PROBE:ProbeForce" --judge threshold:0.985 --judge-cap 4 --judge-burst 2 & p2=$!
run final_quantile pi05 sp "$PROBE:ProbeForce" --judge quantile:0.7:20:0.985 & p3=$!
run final_inf pi05 sp "$PROBE:ProbeB0" --judge periodic:1 --replay-cell pi05_spatial_inf & p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
run final_blind_p50 pi05 sp "$PROBE:ProbeBlind" --blind --judge guard_only & p1=$!
run final_blind_g500 groot sp "$PROBE:ProbeBlind" --kwargs '{"library":"bpool_all"}' --blind --judge guard_only & p2=$!
run final_periodic_p500 pi05 sp "$PROBE:ProbeBlind" --kwargs '{"library":"bpool_cs"}' --blind --judge periodic:5 & p3=$!
run final_periodic_step0 pi05 sp "$PROBE:ProbeBlind" --blind --judge periodic:1 --judge-step0 hit & p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
run final_budget0 pi05 sp "$PROBE:ProbeBlind" --kwargs '{"budget":0}' --blind --judge guard_only & p1=$!
run final_no_blind_method pi05 sp "$PROBE:ProbeB0" --blind --judge periodic:5 & p2=$!
wait "$p1"; wait "$p2"
run final_l10_p50 pi05 l10 "$K1" --kwargs '{"base_kwargs":{"lib":"current","kref":5,"gates":"all","budget":2}}' --blind --judge periodic:5 & p1=$!
run final_l10_p500 pi05 l10 "$K1" --kwargs '{"base_kwargs":{"lib":"big","kref":8,"gates":"all","budget":2}}' --blind --judge periodic:5 & p2=$!
run final_l10_g50 groot l10 "$K1" --kwargs '{"base_kwargs":{"lib":"current","kref":5,"gates":"all","budget":2}}' --blind --judge periodic:5 & p3=$!
run final_l10_g500 groot l10 "$K1" --kwargs '{"base_kwargs":{"lib":"big","kref":8,"gates":"all","budget":2}}' --blind --judge periodic:5 & p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
run final_spatial_p50 pi05 sp "$K1" --kwargs '{"base_kwargs":{"lib":"current","kref":5,"gates":"all","budget":2}}' --blind --judge periodic:5 & p1=$!
run final_spatial_p500 pi05 sp "$K1" --kwargs '{"base_kwargs":{"lib":"big","kref":8,"gates":"all","budget":2}}' --blind --judge periodic:5 & p2=$!
run final_spatial_g50 groot sp "$K1" --kwargs '{"base_kwargs":{"lib":"current","kref":5,"gates":"all","budget":2}}' --blind --judge periodic:5 & p3=$!
run final_spatial_g500 groot sp "$K1" --kwargs '{"base_kwargs":{"lib":"big","kref":8,"gates":"all","budget":2}}' --blind --judge periodic:5 & p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
run final_blind_cap_burst pi05 sp "$PROBE:ProbeBlind" --blind --judge guard_only --judge-cap 2 --judge-burst 2 & p1=$!
run final_hist pi05 sp "$PROBE:ProbeHist" & p2=$!
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k2_serving/check_log_only.py > "$OUT/final_log_only.log" 2>&1 & p3=$!
wait "$p1"; wait "$p2"; wait "$p3"
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.verify_logs --log-dir "$OUT/final_hist" --tag selftest --work "$OUT/final_hist/verify" --out "$OUT/final_hist/verify.json" > "$OUT/final_hist/verify.log" 2>&1
