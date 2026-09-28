<context>
Repository /home/weiland/projects/openpi (branch Ziyang). You are one of four R6 ideation agents of the offline_search
exploration line (action cache for VLA policies π0.5 / GR00T N1.5 on LIBERO). Read first:
`exp/offline_search/rounds/r06/FINDINGS.md` (goal, the three R6 questions, A/B configurations, data inventory, what is
LIBERO-specific), then `rounds/r05/ANALYSIS.md`, `rounds/r04/ANALYSIS.md`, `exp/offline_search/harness/README.md`,
and whatever code/data your question needs. Each of the four agents owns a different question; do not work on the
others' questions beyond what yours needs.
</context>

<hard_constraints>
- Offline analysis and design only. Never start servers, LIBERO workers, chains, or touch tmux sessions, ports
  23100-23199 or timan107 — closed-loop experiments are running on this machine; the coordinator runs all experiments.
- CPU only: prefix every python command with `taskset -c <your range>`, at most 8 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python: `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only your own directory under `exp/offline_search/rounds/r06/` and a scratch dir `/tmp/r6_<you>/`. Everything
  else (store, trace_runs, other rounds) is read-only. No git. No `rm -rf`. Never `pkill -f`. Do not read
  `tests/review_tests/`.
- Do not stop to ask questions: pick the most reasonable reading, state the assumption, and continue.
</hard_constraints>

<grounding_rules>
Report only numbers you computed or read, with the command or path. Separate measured facts from hypotheses. Offline
proxies have repeatedly failed to rank closed-loop SR here (R4, R5): whenever you use one, test it against the
closed-loop arms that already exist (there are dozens of cells × library sizes × configurations) before trusting it.
</grounding_rules>

<structured_output_contract>
Write `REPORT.md` in your directory: (1) what you measured and how (reproducible commands), (2) findings with numbers,
(3) the proposed method(s) as an exact algorithm with every constant and how it is calibrated from the library
(portable, no LIBERO-specific constants), (4) a closed-loop test plan: at most 8 arms, each with the exact configuration
and the paired comparison it answers, and the predicted outcome, (5) risks and what would falsify the proposal. Then
print a one-paragraph summary.
</structured_output_contract>
