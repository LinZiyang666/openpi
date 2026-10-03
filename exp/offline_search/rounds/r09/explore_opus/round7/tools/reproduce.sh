#!/usr/bin/env bash
# Reproduce opus round-7 offline outputs (no remote, no GPU). CPUs 2-9,46-53 only. Needs round-5 outputs (gatesim, thresholds).
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round7.tools
"${P[@]}" -m $M.predict7          # PREDICTION.md basis
"${P[@]}" -m $M.prepare_arms      # artifacts + emitted arms in r09_opus_r7
"${P[@]}" -m $M.selftests         # CPU plugin selftests (fresh selftest dirs required)
