#!/usr/bin/env bash
# Reproduce opus round-6 offline outputs (no remote, no GPU). CPUs 2-9,46-53 only.
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round6.tools
"${P[@]}" -m $M.ledger6 --workers 12      # ledgers (round-5 set + round-5 screen + takeover runs; inits 0-29 only)
"${P[@]}" -m $M.exhaust                   # Q1 tables + fitted (end_rows, dwell) on inits 0-19 -> out/exhaust.json
"${P[@]}" -m $M.predict6                  # PREDICTION.md basis (replay on the stacks' screen runs)
"${P[@]}" -m $M.prepare_arms              # artifacts + emitted arms in r09_opus_r6 (asserts exhaust.json == constants)
"${P[@]}" -m $M.selftests                 # CPU plugin selftests (fresh selftest dirs required)
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/explore_opus/round6/tests
