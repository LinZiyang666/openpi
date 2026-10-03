#!/usr/bin/env bash
# Reproduce opus round-4 offline outputs (no remote, no GPU). CPUs 2-9,46-53 only.
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round4.tools
"${P[@]}" -m $M.predict          # simulation basis of PREDICTION.md (fable corrector runs, inits 20-29)
"${P[@]}" -m $M.prepare_arms     # stack artifacts + emitted arms in r09_opus_r4
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/explore_opus/round4/tools/tests
