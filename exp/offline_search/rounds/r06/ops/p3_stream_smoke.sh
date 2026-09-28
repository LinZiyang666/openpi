#!/bin/bash
# Stream smoke (codex STREAMING.md recipe): receiver 23170, policy 23166, 2 P10 arms x 4 episodes, budget-gated.
set -uo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_stream_smoke
export P3_STREAM_PORT=23170
POLICY_PORT=23166
P3PY=(taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
echo "STREAM_SMOKE_START $(date -Is)"
bash "$D/deploy_client.sh" "$RUN/client_bundle_stream" > "$RUN/client_deploy_stream.log" 2>&1 || { echo "DEPLOY_FAILED"; exit 1; }
echo "DEPLOY_OK $(tail -1 $RUN/client_deploy_stream.log)"
tmux has-session -t p3rx_r06_p3_stream_smoke 2>/dev/null || tmux new-session -d -s p3rx_r06_p3_stream_smoke "cd /home/weiland/projects/openpi && taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.stream_receiver --run-root '$RUN' --port '$P3_STREAM_PORT' --ready-file '$RUN/state/p3_stream.ready.json' >> '$RUN/receiver.log' 2>&1"
for i in $(seq 1 30); do
  "${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" >/dev/null 2>&1 && break
  sleep 1
done
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$P3_STREAM_PORT" || { echo "RECEIVER_UNHEALTHY"; exit 1; }
echo "RECEIVER_OK"
ok=0; while [ $ok -lt 3 ]; do if admit 10500 0; then ok=$((ok+1)); else ok=0; fi; sleep 10; done
echo "GPU_ADMIT $GATE_MSG $(date -Is)"
mapfile -t ARMS < "$RUN/arm_names.txt"
PORTS="$POLICY_PORT" WPS=2 SERVER_CPUS=6-9,50-53 STAGE1_ONLY=0 P3_PHASE=smoke \
  bash "$D/chain_p3.sh" "$RUN" "${ARMS[@]}" 2>&1 | tee -a "$RUN/chain_console.log"
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 \
  --run-root "$RUN" --arms "${ARMS[@]}" --client-root "$RUN/runs" \
  --require-stage-counts --require-snapshots --out "$RUN/tables_stream_smoke" > "$RUN/read_v2.log" 2>&1
echo "READ_V2_RC=$? $(tail -1 $RUN/read_v2.log | cut -c1-200)"
echo "STREAM_SMOKE_END $(date -Is)"
