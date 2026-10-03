#!/usr/bin/env bash
# Rebuild the R9Recipe equivalence artifacts / arms and re-prove equivalence (no remote, no GPU). CPUs 2-9,46-53 only.
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.recipe.tools
"${P[@]}" -m $M.build_eq          # 8 R9Recipe artifacts + emitted arms in r09_recipe_eq (run root must be fresh)
"${P[@]}" -m $M.equivalence       # CPU plugin selftests: recipe vs reference, 0 differing decisions expected (fresh dirs)
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/recipe/tests
