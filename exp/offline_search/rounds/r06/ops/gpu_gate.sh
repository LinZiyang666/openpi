# Sourced by R6 lane scripts. Our GPU budget (MiB) lives in tmp/gpu_budget_mb (owner allocation; edit live).
# ours_mb = GPU memory of our closed-loop servers (processes whose cmdline carries --os-tag).
# pilot_reserve_mb = 10500 per pilot lane whose server is not up (between arms), so other lines do not starve it.
GATE_T=/home/weiland/.claude/jobs/a607dd74/tmp
budget_mb() { cat $GATE_T/gpu_budget_mb 2>/dev/null || echo 23500; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ours_mb() {
  local s=0 pid m
  while IFS=', ' read -r pid m; do
    [ -r /proc/$pid/cmdline ] && tr '\0' ' ' < /proc/$pid/cmdline | grep -q -- '--os-tag' && s=$((s + m))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  echo $s
}
pilot_reserve_mb() {
  local lanes up=0 pp pid
  lanes=$(tmux ls 2>/dev/null | grep -c "^p3pilot_L")
  for pp in 23164 23165; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$pp " | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "$pid" ] && tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q 'os-tag r6p3v2_[a-z0-9_]*_r[0-9]_' && up=$((up + 1))
  done
  [ "$lanes" -gt "$up" ] && echo $(( (lanes - up) * 10500 )) || echo 0
}
# admit NEED [RESERVE]: free >= need+reserve and ours+need+reserve <= budget
admit() {
  local need=$1 res=${2:-0} f o b
  f=$(free_mb); o=$(ours_mb); b=$(budget_mb)
  GATE_MSG="free=$f ours=$o need=$need reserve=$res budget=$b"
  [ "${f:-0}" -ge $((need + res)) ] && [ $((o + need + res)) -le "$b" ]
}
