#!/usr/bin/env bash
# Reproduce opus round-5 offline outputs (no remote, no GPU). CPUs 2-9,46-53 only.
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round5.tools
"${P[@]}" -m $M.ledger5 --workers 12        # per-decision ledgers with guard extras (inits 0-29 only)
"${P[@]}" -m $M.waste                       # Q1: wasted-call anatomy, recovery after call vs cache, survival
"${P[@]}" -u -m $M.gatesim --draws 300      # Q2: prefix-exact gate simulation on inits 0-19 (pair bootstrap)
"${P[@]}" -m $M.fit_thresholds              # L0 / C by the pre-stated rules (inits 0-19)
"${P[@]}" -m $M.anatomy5                    # Q3: residual failure anatomy on inits 20-29
"${P[@]}" -m $M.predict                     # PREDICTION.md basis
"${P[@]}" -m $M.prepare_arms                # artifacts + emitted arms in r09_opus_r5 (asserts thresholds.json == constants)
"${P[@]}" -m $M.selftests                   # CPU plugin selftests (fresh selftest dirs required)
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/explore_opus/round5/tests
