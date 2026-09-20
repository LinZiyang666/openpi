#!/usr/bin/env bash
# Local controller for the X-WAM RoboCasa-2024 step ladder (runs on the workstation in tmux, drives the fleet via tether).
# For each k in <ks>: (re)start the policy servers with --action_denoise_steps k (h100 x2 on the same GPU, plus weilandserver
# joining the h100 broker when its checkpoint is present), wait until they report "connected to broker", launch one
# robocasa client per task (24) on timan108 against the h100 broker (results_k<k>/eval_results_<rank>.json; a task whose
# result exists is skipped, so a k can be resumed), relaunch clients that exit without a result (<=2 retries), merge, pull.
# usage: ladder_xwam_local.sh <ks csv> [evals per task=50] [video steps=50]
set -u
KS=${1:?ks csv}; NE=${2:-50}; VK=${3:-50}; SPLIT=${4:-2}   # SPLIT clients per task (world_size = 24*SPLIT)
H100=149.165.153.233; FP=23270; BP=23271
LOCAL=/home/weiland/projects/openpi/exp/xwam_nfe/data
LOG=/home/weiland/.claude/jobs/9267c51a/tmp/xwam_ladder.log
mkdir -p "$LOCAL"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
tx() { tether exec --timeout "${TX_TO:-60s}" "$1" -- bash -lc "$2" 2>/dev/null | awk '!seen[$0]++'; }

for K in ${KS//,/ }; do
  OUT=/scratch/zixuans8/xwam/results_k$K
  say "XWAM k=$K UP"
  # --- servers: restart with this k (idempotent scripts; kill first so the new k is picked up)
  # the broker keeps READY entries of servers that died -> a client whose first request lands on a stale identity hangs
  # forever; restart the broker together with the servers (servers announce READY on connect)
  tx h100 "tmux kill-session -t xwam_srv0 2>/dev/null; tmux kill-session -t xwam_srv1 2>/dev/null; tmux kill-session -t xwam_broker 2>/dev/null; sleep 5; bash /tmp/xwam/serve_xwam.sh $K $VK 0 $FP $BP 0; sleep 25; bash /tmp/xwam/serve_xwam.sh $K $VK 0 $FP $BP 1" | tail -2 | tee -a "$LOG"
  if tx weilandserver "test -f /data/xwam/wan22_5b/config.json && test -f /data/xwam/checkpoints/robocasa_sft/config.yaml && ls /data/xwam/checkpoints/robocasa_sft/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt" | grep -q mp_rank; then
    tx weilandserver "tmux kill-session -t xwam_srv0 2>/dev/null; sleep 5; BROKER_ADDR=$H100 bash /tmp/xwam/serve_xwam.sh $K $VK 0 $FP $BP 0" | tail -1 | tee -a "$LOG"
    WLS=1
  else
    say "wls checkpoint not ready; h100 servers only"; WLS=0
  fi
  # --- wait for the servers (model load + connect; first inference compiles later)
  for i in $(seq 1 40); do
    n=$(tx h100 "grep -c 'connected to broker' /tmp/xwam/srv0.log /tmp/xwam/srv1.log 2>/dev/null | awk -F: '{s+=\$2} END {print s}'")
    w=0; [ "$WLS" = 1 ] && w=$(tx weilandserver "grep -c 'connected to broker' /tmp/xwam/srv0.log 2>/dev/null")
    [ "${n:-0}" -ge 2 ] && { [ "$WLS" = 0 ] || [ "${w:-0}" -ge 1 ] || [ "$i" -ge 30 ]; } && break
    sleep 15
  done
  say "servers ready h100=${n:-0} wls=${w:-0}"
  # --- clients on timan108: SPLIT clients per task (rank r -> task r % 24, seeds rank*NE+i, so ranks r and r+24 are
  # disjoint halves of one task), at most W concurrent (GPU = rank % 3); resume on existing result; <=3 launches each
  NR=$((24 * SPLIT)); NEW=$((NE / SPLIT)); W=${XWAM_POOL:-24}
  declare -A tries=()
  while :; do
    # a client hangs at interpreter exit after writing its result (zmq linger): reap sessions whose result exists
    tx timan108 "for s in \$(tmux ls 2>/dev/null | grep -oE '^xwam_k${K}_r[0-9]+'); do r=\${s##*_r}; [ -f $OUT/eval_results_\$r.json ] && tmux kill-session -t \$s; done; true" >/dev/null
    st=$(tx timan108 "echo \"done=\$(ls $OUT/eval_results_*.json 2>/dev/null | wc -l) running=\$(tmux ls 2>/dev/null | grep -c '^xwam_k${K}_r')\"")
    done_n=$(echo "$st" | grep -oE "done=[0-9]+" | cut -d= -f2); run_n=$(echo "$st" | grep -oE "running=[0-9]+" | cut -d= -f2)
    [ "${done_n:-0}" -ge "$NR" ] && break
    launched=0
    if [ "${run_n:-0}" -lt "$W" ]; then
      have=$(tx timan108 "ls $OUT/eval_results_*.json 2>/dev/null | sed -E 's/.*eval_results_([0-9]+).json/\1/' | tr '\n' ' '")
      alive=$(tx timan108 "tmux ls 2>/dev/null | grep -oE '^xwam_k${K}_r[0-9]+' | sed -E 's/.*_r//' | tr '\n' ' '")
      for r in $(seq 0 $((NR-1))); do
        [ "${run_n:-0}" -ge "$W" ] && break
        echo " $have " | grep -q " $r " && continue
        echo " $alive " | grep -q " $r " && continue
        [ "${tries[$r]:-0}" -ge 3 ] && continue
        tries[$r]=$(( ${tries[$r]:-0} + 1 ))
        tx timan108 "bash /tmp/xwam/run_client_xwam.sh $r $NEW $H100 $FP $((r % 3)) k$K $OUT" | tail -1 >> "$LOG"
        run_n=$((run_n+1)); launched=$((launched+1)); sleep 2
      done
    fi
    say "k=$K $st launched=$launched"
    if [ "$launched" -eq 0 ] && [ "${run_n:-0}" -eq 0 ]; then say "XWAM k=$K FAIL ($st, tries exhausted)"; break; fi
    sleep 120
  done
  unset tries
  if [ "${done_n:-0}" -lt "$NR" ]; then continue; fi
  # --- merge + pull
  tx timan108 "cd /scratch/zixuans8/xwam/X-WAM && .venv/bin/python evaluation/merge_results.py $OUT 2>&1 | tail -3; cd $OUT && tar czf /tmp/xwam/results_k$K.tgz eval_results_*.json eval_summary.json 2>/dev/null; ls -la /tmp/xwam/results_k$K.tgz | cut -c1-80" | tail -4 | tee -a "$LOG"
  mkdir -p "$LOCAL/results_k$K" && tether pull timan108:/tmp/xwam/results_k$K.tgz "$LOCAL/results_k$K.tgz" --force >/dev/null 2>&1 && tar xzf "$LOCAL/results_k$K.tgz" -C "$LOCAL/results_k$K"
  # server-side latency: mean of "Inferred in" over this k's serving window (server logs are per start)
  tx h100 "grep -h 'Inferred in' /tmp/xwam/srv0.log /tmp/xwam/srv1.log | tail -n +2 | sed -E 's/.*Inferred in ([0-9.]+)s/\1/' | awk '{n++; s+=\$1} END {if(n) printf \"h100 queries=%d mean_s=%.3f\\n\", n, s/n}'" | tee -a "$LOG"
  python3 - "$LOCAL/results_k$K/eval_summary.json" "$K" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
tasks = {k: v for k, v in d.items() if isinstance(v, dict) and "success_rate" in v and not k.startswith("__")}
macro = sum(v["success_rate"] for v in tasks.values()) / max(len(tasks), 1)
n = sum(v.get("num_rollouts", 0) for v in tasks.values()); s = sum(v.get("num_success_rollouts", 0) for v in tasks.values())
print(f"XWAM k={sys.argv[2]} DONE macro={macro:.4f} pooled={s}/{n} tasks={len(tasks)}")
PY
done
tx h100 "tmux kill-session -t xwam_srv0 2>/dev/null; tmux kill-session -t xwam_srv1 2>/dev/null; echo servers stopped" | tee -a "$LOG"
[ "${WLS:-0}" = 1 ] && tx weilandserver "tmux kill-session -t xwam_srv0 2>/dev/null; echo wls server stopped" | tee -a "$LOG"
say "XWAM LADDER FINISHED"
