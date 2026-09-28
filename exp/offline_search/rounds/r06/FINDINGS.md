# R6 findings brief (common input for the R6 ideation agents)

## Owner's goal and the three R6 questions (2026-09-28, verbatim intent)
Use as little inference (IR) as possible while reaching as high a success rate (SR) as possible — ideally equal to
pure inference. When the library is strong, rely on the library more; when it is weaker, insert MISS (policy
inference) into the trajectory to keep SR. R6 answers three questions:
1. **Library quality**: how do we judge how good the library is (globally, per task, per state)? Owner's idea: as in
   the earlier offline solving, we have a library and many recorded trajectories — use leave-one-out (leave one
   episode / trajectory out) and search the library along held-out trajectories to measure how well it covers them.
2. **How many MISS**: once we decide to use MISS, what budget / rate, as a function of library quality and the SR
   target?
3. **Where to MISS**: at which decisions / states does a MISS buy the most SR, so that SR is kept while IR is pushed
   down?
**Generality constraint**: the method must transfer to another benchmark and another robot. Nothing may be hard-coded
for LIBERO (thresholds, task structure, gripper conventions, horizon). Anything calibrated must be calibrated from the
library itself (plus, at most, a small number of recorded trajectories), with a documented recipe.

## The two current configurations (paper headline, see `rounds/r05/ANALYSIS.md`, `rounds/r06/p1_groot_commit/HANDBACK.md`)
- **A (commit-cache)**: every other decision is a vision anchor: stage-1 visual key → PCA-64 per camera + 8-d state →
  per-task closed-form metric (action-similar pairs within-class whitening) → top-16 kernel synthesis → execute 10
  control steps (anchor + one blind decision). No policy calls. IR ≈ c1/2 ≈ 7.5 %.
- **B = A + committed rescue**: MISS when a reactive guard fires (stuck with vision confirmation, terminal row with
  closed gripper, overtime-and-lag, no progress); the policy chunk is also executed for 10 steps. IR 12–20 %.
- Cost basis: IR = c1·v + (1−c1)·m (π0.5 c1 = .152, GR00T .148; v = vision share, m = MISS share; blind = 0).

| cell | A SR @ IR | B SR @ IR | pure inference L=5 (L=10) |
|---|---|---|---|
| π0.5 LIBERO-10, 500-ep lib | .828 @ .077 | .866 @ .161 | .851 (.904) |
| π0.5 LIBERO-10, 50-ep lib | .706 @ .076 | .830 @ .181 | same |
| π0.5 Spatial, 500 | .982 @ .078 | .982 @ .121 | .991 (.986) |
| π0.5 Spatial, 50 | .838 @ .078 | .910 @ .141 | same |
| GR00T LIBERO-10, 500 | .830 @ .074 | .828 @ .186 (periodic B) | .870 (.866) |
| GR00T LIBERO-10, 50 | .608 @ .074 | .718 @ .185 (periodic B) | same |
| GR00T Spatial, 500 | .964 @ .076 | .952 @ .199 (periodic B) | .940 (.938) |
| GR00T Spatial, 50 | .868 @ .075 | .920 @ .198 (periodic B) | same |
(GR00T guard-triggered B is running now in `r06_paper`; the table's GR00T B is the periodic "MISS every 4th anchor".)
Pattern: with the 500-episode library B ≈ A except π0.5 LIBERO-10 (+3.8 pp); with the 50-episode library B beats A by
+5 to +12 pp. Reactive guards still fire at 500 (m ≈ .05–.14) and waste calls there. R3's confidence-quantile
controllers (fixed 30–50 % MISS) reached high SR but at IR .44–.60. Offline error has repeatedly failed to rank
closed-loop SR (R4, R5 B1) — every claim must end in closed-loop paired tests.

## Data you can use (all read-only)
- **Store** `/home/weiland/trace_runs/offline_search_store/` (layout and Method API: `exp/offline_search/harness/README.md`):
  libraries per model×suite `library/<model>_<suite>/{current (50 ep, deployed), bpool_cs / bpool_all (500 ep),
  demo100, demo200, demo300 (nested demo subsets), grow250 (50 + 250 policy episodes)}`; recorded trajectories
  `queries/<model>_<suite>_{inf,cache}/` (pure-inference and pure-cache runs: keys, state, executed actions, success).
  These are exactly the "trajectories to search the library along". The harness already runs methods offline on
  recorded queries.
- **Leave-one-out machinery already built**: AWM's library pseudo-queries (every library row queried against the
  OTHER episodes of its task; `rounds/r02/g1_awm/awm.py` confidence section); R5 ideation B's LOTO / LOEO offline solver
  (`rounds/r05/ideation_B/` — `library_solver.py`, `calibration.py`, `cost_solver.py`, their logs and results);
  R3 V7 drift-calibrated confidence (`rounds/r02/g3_recovery/wrappers.py`, library-LOEO calibrated).
- **Closed-loop results** `/home/weiland/trace_runs/os_closed_loop/<run>/`: per arm `runs/<arm>/summary.json`
  (SR, `cost_ledger` v, m), `runs/<arm>/client/journal.jsonl` (per episode `task_uid = <arm>:eval:<task>:<init>`,
  success), `runs/<arm>/server_<port>/decisions_*.jsonl` (per decision: vision, src, judge / MISS reason, top1, rows,
  scores, weights, extras incl. distances, confidence, stuck counters). Key run roots: `r02_g50`, `r02_g500` (5-step
  pure cache), `r03_mx` (guard-only, confidence-quantile, periodic), `r04_frontier` (periodic 6/8/12, noprog), `r04_k5`
  (**randomized CALL vs CACHE at guard points — causal data**; estimator `rounds/r04/k5_rand/estimate.py`, analysis
  `rounds/r05/q3_callvalue/`), `r04_k7`, `r04_blind`, `r04_gblind`, `r04_cost` (pure inference L=5 / L=10), `r05_ptail`,
  `r05_x` (A), `r05_q1` (π0.5 B), `r05_q2` (GR00T periodic B, GR00T policy L10), `r05_growth`, `r05_demo_curve`
  (library-size curves: SR of A at 50/100/200/300/500 and grow250), `r05_b1`, `r06_paper` (running: GR00T guard B,
  replicates of A and B).
- Analyses: `rounds/r04/ANALYSIS.md`, `rounds/r05/ANALYSIS.md` (+ their `analysis_scripts/`), ledger
  `logs/offline_search_exploration.log.md` §9–§10. Paired-test helper: `/home/weiland/.claude/jobs/a607dd74/tmp/pair.py`.

## What is currently LIBERO-specific (must be generalized or justified)
Candidates restricted to the task named by the language instruction; 5 control steps per decision and 10-step commit;
gripper dimension index and sign convention (opposite between π0.5 and GR00T); 8-d state; two cameras; step-count
based overtime; thresholds from library percentiles (these are portable if the recipe is).
