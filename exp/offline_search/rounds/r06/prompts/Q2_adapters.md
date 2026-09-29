<task>
Continuation of your Q2 thread. Your frontier-completion plan (`ideation_Q2/frontier/completion_plan.md`, 43 arms, 33 of
them "needs new deployment adapter") is the plan we will run after the pilot (all 8 cells expected ≈ 06:00 CDT). Build
the adapters now so the plan can go to closed loop immediately:

1. **B + extra anchor dose d**: exactly deployed B (π0.5 `rounds/r05/q1_commit/judge.py:CommitJudge`, GR00T
   `rounds/r06/p1_groot_commit/judge.py:GrootCommitJudge`, same kwargs as the `r5q1_c10_p_*` / `r6p1_c10_g_*` specs),
   plus, at every vision anchor where B would serve the cache, an independent keyed coin (seeded per episode/decision,
   no dependence on outcomes) that calls the policy with probability d and executes its chunk with the same 10-control
   commit as B's rescue. No shadow policy, no profiling overhead: this is a deployment method whose IR is what the ledger
   counts. d = 0 must be bit-identical to B on replay.
2. **Risk-allocated target-IR lottery (ρ)**: A (`BlindAWM` A spec) plus, at episode start, a keyed lottery between the
   two adjacent fixed doses in {0, 1/8, 1/4, 1/2, 1} whose expected dose equals the task's p_t from your frozen budget
   algorithm (library-only risk, fallback as in PREREG; c1 and lengths from the library); then at each anchor an
   independent coin with that episode's dose, 10-control commit. Log the chosen dose / propensity per episode. ρ = 0 must
   be bit-identical to A. If the plan also has a uniform-allocation arm at the same ρ, support `allocation=uniform`.
3. For both: the method↔plugin contract of the existing commit judges (read `rounds/r05/q1_commit/`, `rounds/r06/
   p1_groot_commit/HANDBACK.md`, `closed_loop/README.md`), subclass/wrap in your directory, no shared-file edits; CPU
   replay tests on recorded queries (d = 0 / ρ = 0 identity, realized call rates vs nominal on replay, commit length,
   seeds reproducible, GR00T gripper sign handled by the existing judge), and emit_arms specs for ALL 43 arms of the plan
   (`<RUN>` placeholders; every arm must carry `--os-root /home/weiland/trace_runs/offline_search_store` — the old default
   `/dev/shm/offline_search_store` no longer exists), plus the prefit commands and a 2-arm smoke recipe
   (one π0.5, one GR00T, 4 episodes each). Put everything under `ideation_Q2/frontier/adapters/` and a HANDBACK section in
   `ideation_Q2/frontier/ADAPTERS.md`.
Do not run any server or episode. Final message: short summary.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 18-21,62-65`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only under `exp/offline_search/rounds/r06/ideation_Q2/`. Prefits: write to `/tmp/q2_adapter_fits/` if a fit is
  needed and list the commands; the coordinator copies them. Never start servers, workers or chains; never touch tmux
  sessions, ports 23100-23199, timan107 or other hosts (no tether). No git. No `rm -rf`. Never `pkill -f`. Do not edit
  `src/`, `closed_loop/*`, `ops/*`, earlier rounds' files, the P3 directory or any run root. Do not read
  `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
