#!/bin/bash
# R16 workers on the 4090 machine for the 4090's own GR00T servers (owner 10-18: same-machine workers first).
# Copies the arguments of one live timan107 worker of this group (ports, suite, seed, init pools),
# rewrites its paths to this island, and adds workers while this machine has room.
# Usage: attach_local.sh <group run dir>   (runs until the group's workers stop)
G=$1; I=/data/mw_finetune/qualification/r16_local_island; PY=/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python
REMOTE=/scratch/zixuans8/openpi_r16_conductor_v1; mkdir -p $I/logs; cd $I
log(){ echo "$(date +%H:%M:%S) $*" >> $I/logs/attach.log; }
args=""
for _ in $(seq 1 120); do
  args=$(timeout 30 tether exec --safe --timeout 20s timan107 -- bash -lc 'ps -eo args | grep "[w]orker-id r16_groot_" | head -1' 2>/dev/null | grep -o -- "--server-key.*")
  [ -n "$args" ] && break; sleep 30
done
[ -z "$args" ] && { log "no timan107 worker of $G seen; giving up"; exit 0; }
args=${args//$REMOTE/$I}
log "attach $G with: $args"
n=0
while [ -d "$G" ]; do
  live=$(pgrep -fc "[w]orker-id r16_groot_local_")
  [ -f $I/STOP ] && break
  free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free_ram=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$free_vram" -gt 6144 ] && [ "$free_ram" -gt 40 ]; then
    id=$(printf "r16_groot_local_%03d" $n); n=$((n+1))
    port=$((23150 + n % 4)); a=$(echo "$args" | sed -E "s/--server-key ([^:]+):[0-9]+/--server-key \1:$port/")
    setsid bash -c "env PYTHONPATH=$I/loopback:$I/packages/openpi-client/src:$I/src:$I HOME=/home/weiland MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 LIBERO_CONFIG_PATH=$I/os_cl/libero $PY -m examples.libero.worker_entry --worker-id $id $a >> $I/logs/$id.log 2>&1" > /dev/null 2>&1 < /dev/null &
    log "spawn $id (live $live, free VRAM ${free_vram} MiB, RAM ${free_ram} GiB)"
  fi
  # A finished group shuts its workers down; stop when none of ours remain after we had some.
  [ $n -gt 0 ] && [ "$(pgrep -fc "[w]orker-id r16_groot_local_")" -eq 0 ] && { log "group finished"; break; }
  sleep 10
done
