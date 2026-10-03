#!/usr/bin/env bash
# Reproduce every opus round-2 offline output (discovery inits 0-29 only; nothing remote, no GPU).
# Order matters: catalog -> ledgers -> analyses -> artifacts/arms.  Writes only to
#   exp/offline_search/rounds/r09/explore_opus/round2/{out,artifacts,arms_in.json}
#   /home/weiland/trace_runs/offline_search_store/derived/r09_opus/
#   /home/weiland/trace_runs/os_closed_loop/r09_opus_escalation/   (emitted arms, manifest, local plan)
set -euo pipefail
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round2.tools
"${P[@]}" -m $M.catalog --workers 14            # 1108 arms, discovery outcomes (init >= 30 dropped at parse)
"${P[@]}" -m $M.episodes --workers 14           # standard-mode per-decision ledgers of the replicate families
"${P[@]}" -m $M.r8ledger --workers 12 --arms "$(ls /home/weiland/trace_runs/os_closed_loop/r08_main/runs | grep -E 'l10_(50|500)_(A|CU|CT|IP|O5a|SHIFT)$|l10_P10$|spatial_50_(A|CU|IP)$|spatial_P10$' | paste -sd,)"
"${P[@]}" -m $M.serverledger --families cache,only_no_progress,guards_B --models groot   # GR00T rows from server logs
"${P[@]}" -m $M.forensics                       # R8 forensic labels, inits 0-29 rows only
"${P[@]}" -m $M.replicates                      # O1: replicate-pooled outcome decomposition
"${P[@]}" -m $M.anatomy                         # O2: failure anatomy, fragility, recovery, onset
"${P[@]}" -m $M.triggers                        # trigger specificity + escalation simulator (cache)
"${P[@]}" -m $M.triggers --deadline 70
for fam in guards_B only_no_progress uniform_calls corrector; do
  "${P[@]}" -m $M.triggers --family $fam --cells pi05:l10:50,pi05:l10:500,groot:l10:50,groot:l10:500
done
"${P[@]}" -m $M.preregister                     # frozen predictions for the screen
"${P[@]}" -m pytest -q -p no:cacheprovider exp/offline_search/rounds/r09/explore_opus/round2/tools/tests
"${P[@]}" -m $M.prepare_arms                    # artifacts + emitted arms in r09_opus_escalation
