#!/usr/bin/env bash
# Saturated native supply, one model at a time, with R clients of one private MPS server.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: supply_sweep.sh --model {pi05,groot,dp,fastwam} --out-dir DIR [options]
  --host local|h100       Select formal_eval.host_python/server_env (default local)
  --replicas '1 2 4'      Replica counts, space or comma separated (default '1 2 4')
  --sizes '1,2,4,8,16'    Full batch sizes (default 1,2,4,8,16)
  --phase-seconds 12      Seconds per hit/call slot
  --gap-seconds 3         Gap between slots (increase if a batch overruns a slot)
  --hit-shares 0.82,0.5,0 Request shares for the harmonic mixed-throughput estimate
  --gpu 0                Physical GPU index or UUID (default 0)
  --cpus 0-7             Optional taskset CPU list, shared by all replicas
  --descriptor FILE      Formal assets/load.json override
  --checkpoint DIR       Checkpoint path override; descriptor remains unchanged
  --vendor DIR           GR00T vendor checkout override (host default otherwise)
  --observation FILE     Override the real assembly MT50 NPZ
  --steps K              MISS K (default descriptor/adapter K)
  --warmup N             Untimed hit/call batches per B (default 1)
  --dp-uncached-text      Disable the existing batched DP CPU text memo
  --python PATH          Override the model's formal Python island
  --ready-timeout 1800   Load/warmup timeout per replica, seconds
Requires a free GPU and nvidia-cuda-mps-control. OUT must be new. Launches locally
on the chosen host; it does not SSH, change compute mode, or start a campaign.
EOF
}

die() { echo "$*" >&2; exit 1; }
model= out= host=local replicas='1 2 4' sizes=1,2,4,8,16 phase_s=12 gap_s=3
hit_shares=0.82,0.5,0 gpu=0 cpus= vendor= python_override= ready_timeout=1800
proc_args=()
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dp-uncached-text) proc_args+=("$1"); shift; continue ;;
    esac
    (($# >= 2)) || die "Missing value for $1"
    case "$1" in
        --model) model=$2 ;; --out-dir) out=$2 ;; --host) host=$2 ;;
        --replicas) replicas=${2//,/ } ;; --sizes) sizes=$2 ;;
        --phase-seconds) phase_s=$2 ;; --gap-seconds) gap_s=$2 ;;
        --hit-shares) hit_shares=$2 ;; --gpu) gpu=$2 ;; --cpus) cpus=$2 ;;
        --vendor|--vendor-root) vendor=$2 ;;
        --python) python_override=$2 ;; --ready-timeout) ready_timeout=$2 ;;
        --descriptor|--checkpoint|--observation|--steps|--warmup) proc_args+=("$1" "$2") ;;
        *) die "Unknown option: $1" ;;
    esac
    shift 2
done
[[ $model =~ ^(pi05|groot|dp|fastwam)$ && -n $out ]] || { usage >&2; exit 1; }
[[ $host == local || $host == h100 ]] || die "--host must be local or h100"
[[ $ready_timeout =~ ^[1-9][0-9]*$ ]] || die "--ready-timeout must be a positive integer"
read -r -a counts <<< "$replicas"
((${#counts[@]})) || die "Empty replica list"
declare -A seen=()
for count in "${counts[@]}"; do
    [[ $count =~ ^[1-9][0-9]*$ && -z ${seen[$count]:-} ]] || die "Replica counts must be distinct positive integers"
    seen[$count]=1
done
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
if [[ $model == groot && -z $vendor ]]; then
    if [[ $host == h100 ]]; then vendor=/home/exouser/gr00t_n15
    else vendor=/home/weiland/projects/openpi_ext/third_party/gr00t_n15; fi
fi
[[ -z $vendor ]] || proc_args+=(--vendor "$vendor")

# Reuse exactly the formal host Python and environment; this only imports helpers.
launch_text=$(PYTHONPATH="$root:$root/src:$root/packages/openpi-client/src" python3 - "$model" "$host" "$root" "$vendor" <<'PY'
import sys
from exp.r17_integration.ops.formal_eval import host_python, server_env
model, host, root, vendor = sys.argv[1:]
campaign = dict(model=model, server_node=host, domain="metaworld", vendor=vendor, stage_batching=True)
python = host_python(campaign)
if model == "groot":
    campaign["vendor"] = vendor
env = server_env(campaign, root)
if model == "groot" and vendor:
    env["PYTHONPATH"] += ":" + vendor
print(python)
for key, value in env.items():
    print(key + "=" + value)
PY
)
mapfile -t launch <<< "$launch_text"
model_python=${python_override:-${launch[0]}}
launch_env=("${launch[@]:1}")
task_prefix=()
[[ -z $cpus ]] || task_prefix=(taskset -c "$cpus")
# Validate CLI numbers without importing a model or touching CUDA.
PYTHONPATH="$root" python3 - "$sizes" "$phase_s" "$gap_s" "$hit_shares" <<'PY'
import math, sys
from exp.r17_integration.profile.supply_proc import sizes_arg, positive_seconds
from exp.r17_integration.profile.supply_report import shares_arg
sizes_arg(sys.argv[1]); positive_seconds(sys.argv[2]); shares_arg(sys.argv[4])
gap = float(sys.argv[3])
if not math.isfinite(gap) or gap < 0:
    raise ValueError("gap seconds must be finite and nonnegative")
PY
[[ ! -e $out ]] || die "Output already exists: $out (use a fresh directory)"
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
mps_dir=$(mktemp -d /tmp/r17mps.XXXXXXXX)
export CUDA_MPS_PIPE_DIRECTORY="$mps_dir/p" CUDA_MPS_LOG_DIRECTORY="$mps_dir/l"
[[ ${#CUDA_MPS_PIPE_DIRECTORY} -le 85 ]] || die "MPS pipe path exceeds 85 bytes"
mkdir "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
unset CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING
pids=() smi_pid= mps_started=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ -n $smi_pid ]]; then kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true; fi
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    if ((mps_started)); then printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true; fi
    cp -a "$CUDA_MPS_LOG_DIRECTORY" "$out/mps-logs" 2>/dev/null || true
    rm -rf "$mps_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"  # UUID avoids MPS physical-index remapping.
nvidia-cuda-mps-control -d
mps_started=1
nvidia-smi -i "$gpu_uuid" --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits -lms 500 > "$out/smi.csv" 2> "$out/smi.log" &
smi_pid=$!

for count in "${counts[@]}"; do
    directory="$out/R$count"
    mkdir "$directory"
    pids=()
    # Serialize load/warmup to avoid measuring simultaneous initialization peaks.
    for ((i=1; i<=count; i++)); do
        env "${launch_env[@]}" "${task_prefix[@]}" "$model_python" -u -m exp.r17_integration.profile.supply_proc \
            --model "$model" --sizes "$sizes" --phase-seconds "$phase_s" "${proc_args[@]}" \
            --out "$directory/replica$i.jsonl" --ready "$directory/replica$i.ready" \
            --start "$directory/START.json" > "$directory/replica$i.log" 2>&1 &
        pids+=("$!")
        limit=$((SECONDS + ready_timeout))
        until [[ -f $directory/replica$i.ready ]]; do
            for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null || die "R$count replica died; see $directory/*.log"; done
            ((SECONDS < limit)) || die "R$count replica$i load/warmup timed out; see its log"
            sleep 1
        done
    done
    kill -0 "$smi_pid" 2>/dev/null || die "nvidia-smi sampler died; see $out/smi.log"
    python3 - "$directory/START.json" "$gap_s" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(dict(t0=time.time() + 3, gap_s=float(sys.argv[2]))))
temporary.replace(path)
PY
    for pid in "${pids[@]}"; do wait "$pid" || die "R$count replica failed; see $directory/*.log"; done
    pids=()
    echo "R$count finished"
done
kill "$smi_pid" 2>/dev/null || true
wait "$smi_pid" 2>/dev/null || true
smi_pid=
PYTHONPATH="$root" python3 -m exp.r17_integration.profile.supply_report "$out" --hit-shares "$hit_shares" | tee "$out/report.txt"
