#!/bin/bash
# devices_free.sh — are the RIT-Pareto devices free to start a group? Prints one verdict line:
#   DEVICES FREE   : timan107 GPUs idle (no foreign compute processes, <1.5 GB used per card), no runner of
#                    ours alive, RAM >= 120 GB free, load < 8; weilandserver GPU either runs our srv0 (4
#                    replicas, :23150 listening) or has >= 34 GB free for it.
#   DEVICES BUSY   : anything else (details on the preceding lines). Exit 0 = FREE, 1 = BUSY.
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
busy=0
c=$(tether exec --timeout 60s timan107 -- bash -lc 'export PATH=/usr/local/bin:/usr/bin:/bin; me=$(id -un); echo "gpu_used_max_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"; echo "gpu_procs_total=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; echo "gpu_procs_mine=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do [ "$(stat -c %U /proc/$p 2>/dev/null)" = "$me" ] && echo x; done | wc -l)"; echo "runner=$(pgrep -f "[r]un_gtp" | wc -l) workers=$(pgrep -f "[w]orker_entry" | wc -l)"; echo "free_gb=$(free -g | awk "/Mem:/{print \$7}") load=$(cut -d" " -f1 /proc/loadavg)"' 2>/dev/null)
echo "timan107: $(echo "$c" | tr '\n' ' ')"
[ -z "$c" ] && { echo "timan107: NO REPLY"; busy=1; }
g=$(echo "$c" | sed -n 's/gpu_used_max_mib=//p'); pt=$(echo "$c" | sed -n 's/gpu_procs_total=//p'); pm=$(echo "$c" | sed -n 's/gpu_procs_mine=//p')
r=$(echo "$c" | sed -n 's/runner=\([0-9]*\).*/\1/p'); f=$(echo "$c" | sed -n 's/.*free_gb=\([0-9]*\).*/\1/p'); l=$(echo "$c" | sed -n 's/.*load=\([0-9.]*\).*/\1/p')
[ "${g:-99999}" -gt 1500 ] && { echo "  busy: GPU memory in use (${g} MiB on the busiest card)"; busy=1; }
[ "${pt:-0}" -gt "${pm:-0}" ] && { echo "  busy: foreign GPU processes ($((pt-pm)))"; busy=1; }
[ "${r:-1}" -gt 0 ] && { echo "  busy: our runner still alive"; busy=1; }
[ "${f:-0}" -lt 120 ] && { echo "  busy: only ${f} GB RAM free"; busy=1; }
awk -v l="${l:-99}" 'BEGIN{exit !(l>=8)}' && { echo "  busy: load ${l}"; busy=1; }
# The server counts as ours only when its argv is exactly what start_eval_server.sh launches (same
# checkpoint / replicas / port); a different serve_policy on :23150 belongs to another session -> BUSY.
SIG="serve_policy.py --replicas 4 --replica-spawn-batch 2 --port 23150 policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
w=$(tether exec --timeout 60s weilandserver -- bash -lc 'export PATH=/usr/local/bin:/usr/bin:/bin; sig="'"$SIG"'"; L=$(pgrep -af "[s]erve_policy.py" | grep -vE "tmux|bash -l?c"); n=$(echo "$L" | grep -c .); m=$(echo "$L" | grep -cF "$sig"); echo "srv_any=$n srv0=$m port=$(ss -ltn | grep -c ":23150 ") gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) gpu_total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"' 2>/dev/null)
echo "weilandserver: $w"
[ -z "$w" ] && { echo "weilandserver: NO REPLY"; busy=1; }
a=$(echo "$w" | sed -n 's/.*srv_any=\([0-9]*\).*/\1/p'); s=$(echo "$w" | sed -n 's/.*srv0=\([0-9]*\).*/\1/p'); p=$(echo "$w" | sed -n 's/.*port=\([0-9]*\).*/\1/p')
[ "${a:-0}" -gt "${s:-0}" ] && { echo "  busy: a serve_policy with a different config is running (another session's server) -> do not touch"; busy=1; }; u=$(echo "$w" | sed -n 's/.*gpu_used_mib=\([0-9]*\).*/\1/p'); t=$(echo "$w" | sed -n 's/.*gpu_total_mib=\([0-9]*\).*/\1/p')
if [ "${s:-0}" -ge 1 ] && [ "${p:-0}" -ge 1 ]; then echo "  server: our srv0 is up"; else
  [ $((${t:-0}-${u:-0})) -lt 34000 ] && { echo "  busy: srv0 down and only $((${t:-0}-${u:-0})) MiB GPU free (need 34000 to start it)"; busy=1; } || echo "  server: srv0 down, GPU has room -> start_eval_server.sh before launching"; fi
[ $busy -eq 0 ] && echo "DEVICES FREE" || echo "DEVICES BUSY"
exit $busy
