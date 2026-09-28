<task>
You are coding agent P3 of R6 in the offline_search exploration line (action cache for VLA policies π0.5 / GR00T N1.5 on
LIBERO) in /home/weiland/projects/openpi (branch Ziyang). Read `exp/offline_search/rounds/r06/FINDINGS.md` first: the
owner's three R6 questions are (Q1) how good is the library — globally, per task, per state; (Q2) how many policy calls
(MISS) to spend; (Q3) where to place them. The owner judges that the existing closed-loop logs are not enough and asks
for **profiling tools that collect new data**, which the coordinator will then run as data-collection experiments.
Build these instruments (all opt-in, method/plugin side, default behaviour byte-identical):

1. **Shadow-policy probe**: while running A (or B), at every vision anchor also run the full policy on the same
   observation and log its chunk next to the served cache chunk, without executing it (the served action must stay
   bit-identical to the unprobed run given the same history). Log per anchor: cache chunk, policy chunk, their distance
   in normalized action units (whole chunk and per step), retrieval features (nearest distances d(1..16), kernel
   weights, neighbour action dispersion, matched library episode / progress / phase, state), and timing. Cost is
   profiling-only and must be reported separately from the ledger IR. This gives per-state labels "how wrong is the
   cache here" at scale for Q1/Q3.
2. **Randomized MISS injection**: at each vision anchor, with a seeded, per-(episode, step) deterministic probability p
   (configurable; optionally stratified by a logged signal), force a MISS whose policy chunk is executed with the same
   10-step commit as B; log the propensity. This extends the existing K5 randomization (`rounds/r04/k5_rand/`, look at
   its overlay and estimator and reuse them) from guard points to every anchor, so the causal value of a MISS can be
   estimated as a function of state features (for Q2/Q3).
3. **Branch rollouts (feasibility first)**: assess whether the LIBERO client used by the chains
   (`exp/offline_search/closed_loop/ops/remote/run_gtp_subset.py`, `run_arm.sh`, and the client/runner they call) can save
   the simulator state at a chosen decision and run two continuations (cache vs policy at that decision, A afterwards)
   to get paired per-state counterfactual outcomes. If feasible without editing `src/`, implement it as an opt-in client
   mode in your own directory plus the smallest wrapper; otherwise write the design and the exact blockers.
4. **Analysis readers**: small scripts that turn these logs into tidy per-anchor tables (parquet or csv) joined with
   episode outcomes, so the ideation agents can use them directly.

Requirements: follow the method↔plugin contract (`rounds/r04/CODING_BRIEF.md`, `rounds/r05/CODING_BRIEF.md`,
`exp/offline_search/closed_loop/README.md`); develop in `exp/offline_search/rounds/r06/p3_profiling/` (you own it). If a
shared plugin file must change, keep behaviour byte-identical without the new flags, re-run the existing regression
matrices (see `rounds/r05/q5_gpu/HANDBACK.md` regression table and `rounds/r06/p1_groot_commit/HANDBACK.md`), and
install in ONE atomic step (temp file in the same directory + os.replace) — closed-loop chains start new servers that
import the plugin every 15–25 minutes. Provide arm specs (emit_arms format, `<RUN>` placeholders) for a proposed
data-collection campaign: shadow-probe runs of A at 50 and 500 libraries for both models × both suites, and randomized
injection runs (propose p and the number of episodes needed for useful precision, with a power calculation), plus the
smoke recipe. Write `HANDBACK.md` (files, install time and sha256, switches, tests with numbers, specs, campaign cost
estimate in GPU-hours, caveats). Final message: short summary.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 2-5,46-49`, at most 8 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1. GPU: `CUDA_VISIBLE_DEVICES=''` unless a real-model parity check needs it
  (then at most 10 GB, one process, and release it). Python `.venv/bin/python` from the repo root. CPUs 38-43 and
  82-87 belong to another project: never use them.
- Never start servers, LIBERO workers or chains; never touch tmux sessions, ports 23100-23199, timan107 or other hosts
  (no tether) — the coordinator runs every experiment. No git. No `rm -rf`. Never `pkill -f`. Do not edit `src/`.
  Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it in HANDBACK.md, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
