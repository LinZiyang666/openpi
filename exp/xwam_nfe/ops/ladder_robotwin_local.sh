#!/usr/bin/env bash
# Local controller for the X-WAM RoboTwin 2.0 (demo_randomized) step ladder, clients on THREE hosts (weilandserver 4090,
# h100 next to the policy server, timan107 8x GTX 1080 with CuRobo on the torch LBFGS path and no CUDA graphs). For each k: restart the h100 broker + NSRV policy servers (--action_denoise_steps k,
# robotwin_sft), wait for "connected to broker", then run one client per task (50 tasks) spread over the hosts, each host at
# most its pool (files rtw_pool_wls / rtw_pool_h100 in the job tmp dir, re-read every loop), resuming on an existing
# <out>/<task>/_result.txt on either host, relaunching a client that exits without a result (<=3 launches per task), then
# pull every host's _result.txt set, aggregate locally, print "XWAM-RT k=<k> DONE macro=...".
# Why two hosts: timan107 is Pascal (CuRobo's kernels reject it), timan108's CUDA is dead; the 4090 caps at 7 clients (VRAM).
# usage: [RESUME_NO_RESTART=1] ladder_robotwin_local.sh <ks csv> [evals per task=20] [video steps=50]
set -u
KS=${1:?ks csv}; NE=${2:-20}; VK=${3:-50}
H100=149.165.153.233; FP=23270; BP=23271
LOCAL=/home/weiland/projects/openpi/exp/xwam_nfe/data
TMPD=/home/weiland/.claude/jobs/9267c51a/tmp
LOG=$TMPD/robotwin_ladder.log
HOSTS="weilandserver h100 timan107"
declare -A OUTD=( [weilandserver]=/data/robotwin [h100]=/data/xwam/robotwin_data [timan107]=/srv/local/zixuans8/robotwin )
declare -A POOLF=( [weilandserver]=$TMPD/rtw_pool_wls [h100]=$TMPD/rtw_pool_h100 [timan107]=$TMPD/rtw_pool_t107 )
declare -A GPUARG=( [weilandserver]=0 [h100]=0 [timan107]=auto )   # timan107: 8x GTX 1080, the launcher picks the emptiest card
TASKS="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch"
NT=$(echo $TASKS | wc -w)
mkdir -p "$LOCAL"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
tx() { tether exec --timeout "${TX_TO:-60s}" "$1" -- bash -lc "$2" 2>/dev/null | awk '!seen[$0]++'; }
pool() { cat "${POOLF[$1]}" 2>/dev/null || echo 0; }

for K in ${KS//,/ }; do
  say "XWAM-RT k=$K UP"
  NSRV=$(cat $TMPD/rtw_nsrv 2>/dev/null || echo 1)
  if [ "${RESUME_NO_RESTART:-0}" = 1 ] && [ -z "${_resumed:-}" ]; then _resumed=1; say "resuming k=$K on the running servers"; else
  # broker restarted with the servers (stale READY identities otherwise hang the first client that lands on them)
  tx h100 "tmux kill-session -t xwam_srv0 2>/dev/null; tmux kill-session -t xwam_srv1 2>/dev/null; tmux kill-session -t xwam_broker 2>/dev/null; sleep 5; XWAM_CKPT=robotwin_sft bash /tmp/xwam/serve_xwam.sh $K $VK 0 $FP $BP 0; if [ $NSRV -ge 2 ]; then sleep 25; XWAM_CKPT=robotwin_sft bash /tmp/xwam/serve_xwam.sh $K $VK 0 $FP $BP 1; fi" | tail -2 | tee -a "$LOG"
  for i in $(seq 1 40); do
    n=$(tx h100 "grep -c 'connected to broker' /tmp/xwam/srv0.log /tmp/xwam/srv1.log 2>/dev/null | awk -F: '{s+=\$2} END {print s}'")
    [ "${n:-0}" -ge "$NSRV" ] && break
    sleep 15
  done
  say "servers ready h100=${n:-0} (NSRV=$NSRV)"
  # h100 client pool follows its server count: 1 server (28.5 GB) leaves room for 7 clients (~6.5 GB each), 2 servers for 3
  [ "$NSRV" -ge 2 ] && echo 1 > ${POOLF[h100]} || echo 4 > ${POOLF[h100]}
  fi
  declare -A tries=()
  while :; do
    have=""; alive=""; st=""; hostfail=0; declare -A run=()
    for H in $HOSTS; do
      OUT=${OUTD[$H]}/results_k$K
      # a client hangs at interpreter exit after writing its result (zmq linger): reap sessions whose result exists
      # ... and a client whose log has not moved for 25 min is stuck (seen: 45 min at 'Policy Name' with no episode), or one
      # spinning in 'error occurs !' (a sticky CUDA error after an OOM makes every seed fail instantly), or one whose log shows
      # an OIDN error (the denoiser silently hands un-denoised frames to the policy when VRAM is short -> tainted episodes):
      # kill it so the task restarts from scratch (<=3 launches per task)
      tx $H "mkdir -p $OUT; for s in \$(tmux ls 2>/dev/null | grep -oE '^rtw_k${K}_[a-z0-9_]+'); do t=\${s#rtw_k${K}_}; [ -f $OUT/\$t/_result.txt ] && tmux kill-session -t =\$s; [ -n \"\$(find /tmp/robotwin -name \$s.log -mmin +25 2>/dev/null)\" ] && { echo STALL \$s; tmux kill-session -t =\$s; }; [ \"\$(tr '\\r' '\\n' < /tmp/robotwin/\$s.log 2>/dev/null | grep -c 'error occurs')\" -gt 300 ] && { echo ERRLOOP \$s; tmux kill-session -t =\$s; }; grep -q 'OIDN Error' /tmp/robotwin/\$s.log 2>/dev/null && { echo RENDER_OOM \$s; tmux kill-session -t =\$s; }; done; true" | grep -E "STALL|ERRLOOP|RENDER_OOM" | tee -a "$LOG"
      hq=$(tx $H "echo HOSTOK; ls $OUT/*/_result.txt 2>/dev/null | awk -F/ '{print \$(NF-1)}' | tr '\n' ' '")
      # a host whose query failed (tether timeout under load) must not look empty: that once relaunched 9 finished tasks
      echo "$hq" | grep -q HOSTOK || { say "host $H query failed; retry"; hostfail=1; }
      have="$have $(echo "$hq" | grep -v HOSTOK)"
      al=$(tx $H "tmux ls 2>/dev/null | grep -oE '^rtw_k${K}_[a-z0-9_]+' | sed 's/^rtw_k${K}_//' | tr '\n' ' '")
      alive="$alive $al"; run[$H]=$(echo $al | wc -w)
      st="$st $H:done=$(echo $have | tr ' ' '\n' | grep -c .)/run=${run[$H]}/pool=$(pool $H)"
    done
    [ "$hostfail" = 1 ] && { sleep 60; continue; }
    done_n=$(echo $have | tr ' ' '\n' | grep -c .)
    [ "$done_n" -ge "$NT" ] && break
    launched=0
    for t in $TASKS; do
      echo " $have " | grep -q " $t " && continue
      echo " $alive " | grep -q " $t " && continue
      [ "${tries[$t]:-0}" -ge 3 ] && continue
      # the host with the most free slots takes the task
      best=""; bestfree=0
      for H in $HOSTS; do f=$(( $(pool $H) - ${run[$H]} )); [ "$f" -gt "$bestfree" ] && { best=$H; bestfree=$f; }; done
      [ -z "$best" ] && break
      tries[$t]=$(( ${tries[$t]:-0} + 1 ))
      tx $best "bash /tmp/robotwin/run_client_robotwin.sh $t $NE $H100 $FP ${GPUARG[$best]} k$K ${OUTD[$best]}/results_k$K" | tail -1 >> "$LOG"
      run[$best]=$(( ${run[$best]} + 1 )); launched=$((launched+1)); sleep 3
    done
    say "k=$K done=$done_n$st launched=$launched"
    if [ "$launched" -eq 0 ] && [ "$(( ${run[weilandserver]} + ${run[h100]} + ${run[timan107]} ))" -eq 0 ]; then say "XWAM-RT k=$K FAIL (done=$done_n, tries exhausted)"; break; fi
    sleep 180
  done
  unset tries run
  if [ "${done_n:-0}" -lt "$NT" ]; then continue; fi
  # --- pull every host's _result.txt set, aggregate locally
  D=$LOCAL/robotwin_k$K; mkdir -p $D/merged
  for H in $HOSTS; do
    OUT=${OUTD[$H]}/results_k$K
    tx $H "cd $OUT && tar czf /tmp/robotwin/results_k$K.tgz */_result.txt 2>/dev/null; ls -la /tmp/robotwin/results_k$K.tgz | cut -c1-80" | tail -1 >> "$LOG"
    mkdir -p $D/$H && tether pull $H:/tmp/robotwin/results_k$K.tgz $D/$H.tgz --force >/dev/null 2>&1 && tar xzf $D/$H.tgz -C $D/$H && cp -r $D/$H/. $D/merged/
  done
  python3 /home/weiland/projects/openpi/exp/xwam_nfe/ops/agg_robotwin.py $D/merged $K $NE $VK | tee -a "$LOG" && cp $D/merged/summary.json $D/summary.json
  tx h100 "grep -h 'Inferred in' /tmp/xwam/srv0.log /tmp/xwam/srv1.log 2>/dev/null | tail -n +2 | sed -E 's/.*Inferred in ([0-9.]+)s/\1/' | awk '{n++; s+=\$1} END {if(n) printf \"h100 queries=%d mean_s=%.3f\\n\", n, s/n}'" | tee -a "$LOG"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"XWAM-RT k={d['k']} DONE macro={d['macro_success_rate']:.4f} tasks={d['n_tasks']} n_per_task={d['n_per_task']}\")" "$D/summary.json" | tee -a "$LOG"
done
tx h100 "tmux kill-session -t xwam_srv0 2>/dev/null; tmux kill-session -t xwam_srv1 2>/dev/null; tmux kill-session -t xwam_broker 2>/dev/null; echo servers stopped" | tee -a "$LOG"
say "XWAM-RT LADDER FINISHED"
