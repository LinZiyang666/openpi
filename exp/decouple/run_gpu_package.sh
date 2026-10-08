#!/usr/bin/env bash
# Preparation: bash exp/decouple/run_gpu_package.sh --dry-run
# Owner, only after reserving the 48 GB card: bash exp/decouple/run_gpu_package.sh
set -euo pipefail
cd /home/weiland/projects/openpi_decouple
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/weiland/projects/openpi_decouple:/home/weiland/projects/openpi_decouple/src:/home/weiland/projects/openpi_decouple/packages/openpi-client/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
if [[ " ${*} " == *" --dry-run "* ]]; then
    export CUDA_VISIBLE_DEVICES=
    package_cpus=10-17,54-61
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    package_cpus=2-43,46-87
fi
# Benchmark arms (current and split) share the same broad CPU set; R16 resident servers run unpinned (0-87).
exec taskset -c "$package_cpus" /home/weiland/projects/openpi/.venv/bin/python -B -m exp.decouple.gpu_package "$@"
