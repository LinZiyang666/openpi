<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 (R4) of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. You are one of four coding agents working in parallel. Read first:
exp/offline_search/rounds/r04/CODING_BRIEF.md (binding: file ownership, live-system rule, the method<->plugin contract,
log fields, cost ledger, rules), then exp/offline_search/rounds/r04/SELECTION.md, the ideation reports in
exp/offline_search/rounds/r04/ideation_{A,B,C}/ (REPORT.md; C also REPORT_2.md), exp/offline_search/closed_loop/README.md,
exp/offline_search/harness/README.md, and the code you build on (named in your task).
Store: /dev/shm/offline_search_store (hot) or /home/weiland/trace_runs/offline_search_store (cold copy, same content).
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root (GR00T servers use their own venv, see
closed_loop/ops/start_server.sh).
</context>

<hard_constraints>
- Your CPU range: <CPUS>. Prefix every python/numpy command with `taskset -c <CPUS>` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: <GPU>
- Edit only the files you own (CODING_BRIEF.md "File ownership"). Develop in a copy, keep existing behaviour
  byte-identical when your new options are absent, install each shared file atomically (temp file + mv), and re-run the
  existing selftests after installing.
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No timan107, no LIBERO workers, no
  closed-loop chains. Never touch ports, processes or tmux sessions you did not start. Do not read tests/review_tests/.
  Do not modify src/, harness/, profile/ or rounds/r01..r03/.
</hard_constraints>

<completeness_contract>
Resolve your task fully before stopping: implementation, tests, and the hand-back file. Do not stop at the first
working version; check edge cases (episode reset, step 0, after a MISS, both models where required, both library sizes
50 and 500 where required).
</completeness_contract>

<verification_loop>
Before finishing, verify every deliverable against the task and the brief with real runs (selftests, parity checks,
smoke runs as allowed). If a check fails, fix and re-run; report the final numbers, not the first attempt.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label anything unverified.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/<YOURDIR>/HANDBACK.md (files changed/added with install times; exact switches/arm-spec fields; tests
and their numbers; what the coordinator must do next; caveats), then print a one-paragraph summary as your final
message.
</structured_output_contract>
