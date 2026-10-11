#!/usr/bin/env bash
# Generated concrete Q slot. CPU launcher; GPU admission happens inside run_slot.
set -uo pipefail
cd /home/weiland/projects/openpi || exit 1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/weiland/projects/openpi:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src
run_dir="/data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_A_$(date +%Y%m%d_%H%M%S)_$$"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --run && $((i+1)) -lt ${#args[@]} ]]; then run_dir="${args[i+1]}"; fi
  if [[ "${args[i]}" == --run=* ]]; then run_dir="${args[i]#--run=}"; fi
done
/usr/bin/timeout --signal=TERM --kill-after=5s 885s /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.qualification.run_slot --slot /home/weiland/projects/openpi/exp/r17_integration/qualification/specs/slot_G2_fastwam_libero_A.json --run "$run_dir" "$@"
rc=$?
if [[ "$rc" == 124 || "$rc" == 137 || "$rc" == 143 ]]; then
  /home/weiland/projects/openpi/.venv/bin/python -m exp.r17_integration.qualification.timeout_receipt "$run_dir" 'slot_G2_fastwam_libero_A'
  exit 2
fi
exit "$rc"
