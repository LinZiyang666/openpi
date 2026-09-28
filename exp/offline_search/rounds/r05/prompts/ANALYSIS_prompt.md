<task>
You are the R5 analysis agent of the offline_search exploration line (action cache for VLA policies π0.5 / GR00T N1.5 on
LIBERO) in /home/weiland/projects/openpi (branch Ziyang). Your full brief is
`exp/offline_search/rounds/r05/ANALYSIS_BRIEF.md` — read it first and follow it exactly (reading list, data table, cost
bases, tools, deliverable sections 1–11, rules). Deliver `exp/offline_search/rounds/r05/ANALYSIS.md` (English).
Three arms are still running when you start: `r05_x/r5x_g_sp_50_tail1u` and `r05_x/r5x_p_l10_500_k7tail_rep{a,b}`.
Use an arm only once `<run>/state/<arm>.DONE` exists and `<run>/runs/<arm>/summary.json` is present; write everything
else, leave clearly marked `TODO-PENDING` slots for those three arms, and stop. The coordinator will resume you with
a short instruction to fill them.
</task>

<context>
The coordinator's ledger (`logs/offline_search_exploration.log.md` §10, entries from 2026-09-27 23:1x to 2026-09-28
04:3x CDT) already lists a quick paired number for most R5 arms — recompute every number yourself; do not copy. R4's
analysis (`rounds/r04/ANALYSIS.md`) and its scripts in `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/`
(`an.py`, `blind_stats.py`, `blind_runs.py`, `frontier.py`) are available to copy and extend. A small paired helper is
`/home/weiland/.claude/jobs/a607dd74/tmp/pair.py <run:arm-ref> <run:arm> ...` (exact McNemar on the intersection of
(task, init) pairs).
</context>

<hard_constraints>
- CPU only: prefix every python command with `taskset -c 26-29,70-73`, at most 8 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python: `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only `exp/offline_search/rounds/r05/ANALYSIS.md` and scratch files under `/tmp/r5_analysis/`.
  `/home/weiland/trace_runs` is read-only for you.
- No git. No `rm -rf`. Never `pkill -f`. Never touch servers, chains, tmux sessions, relays, ports 23150-23169 or
  timan107 — closed-loop experiments are running and the coordinator runs all of them. Do not read `tests/review_tests/`.
</hard_constraints>

<grounding_rules>
Report only numbers you computed or read from files, with the command or path. Owner IR must be recomputed from each
per-arm `cost_ledger` v, m (π0.5 .152·v + .848·m; GR00T .148·v + .852·m; wrist-only .0552·v + m·(.848 + .0499), a
labelled ASSUMPTION). Paired tests on the same (task, init) pairs against the measured noise floor. Label anything
unverified or assumed.
</grounding_rules>

<structured_output_contract>
Write ANALYSIS.md with the sections of the brief. Then print: a one-paragraph summary, the five most important
quantitative findings as bullets, and the list of TODO-PENDING slots still open.
</structured_output_contract>
