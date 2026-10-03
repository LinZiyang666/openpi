#!/usr/bin/env bash
# Reproduce opus round-3 outputs (no remote, no GPU). CPUs 2-9,46-53 only.
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round3.tools
mkdir -p exp/offline_search/rounds/r09/explore_opus/round3/out
"${P[@]}" -m $M.dissect        # round-2 screen per episode (inits 20-29)
"${P[@]}" -m $M.evidence       # crosstab, proximity, futility, early prediction, guard calls
"${P[@]}" -m $M.action_map     # maps from inits 0-19 debug data
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/explore_opus/round3/tools/tests \
                                    exp/offline_search/rounds/r09/explore_opus/round2/tools/tests
"${P[@]}" -m $M.prepare_arms   # artifacts + emitted arms in r09_opus_r3
