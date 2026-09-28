<task>
Continuation of your R6 ideation thread. Since your report, the owner asked for more data; the superset profiler (P3 v2,
`exp/offline_search/rounds/r06/p3_profiling/`, read `HANDBACK.md` sections "One pilot and continuation campaign" and
`SCHEMA_V2.md`) is now collecting the **pilot**: run root `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot`, 216
arms = 8 cells (π0.5 / GR00T × libero_10 / spatial × 50 / 500-episode library) × 9 cohorts (A; fixed anchor dose
1/8, 1/4, 1/2; P10 = policy at every anchor; B; factorial = pre-guard randomization at distance-quantile bins with
policy duration 5/10 and hold 1–3; window = one delayed call; dose_mix = episode-level random dose) × 3 seed blocks,
20 episodes each (10 tasks × inits 0–1), 4,320 episodes. Every cohort logs the full policy (shadow) at every decision,
K4 resamples at 1/16 anchors, retrieval / guard / calibration features, client control telemetry and simulator
snapshots. Cells finish in queue order (`pilot_queue.txt`): π0.5 l10-50 first (≈ 18:30–19:00 CDT today), then GR00T
sp-50, π0.5 sp-50, GR00T l10-50, then the 500-episode cells; all ≈ 15 h. A validated 8-arm smoke table set (32
episodes, π0.5 l10-50 and GR00T l10-500 × A / P10 / factorial / window) is at
`/home/weiland/trace_runs/os_closed_loop/r06_p3_v2_client_smoke/tables_v2_smoke8b/` (anchors, decisions, controls,
neighbours, action_steps, episodes, attempts CSVs + audit.json); it is for plumbing only — draw no conclusions from it.
Pilot tables are produced with `read_v2 --run-root <RUN> --arms <arms> --client-root <RUN>/runs
--require-stage-counts --require-snapshots --out <dir>` (the coordinator runs it; you may run it read-only on finished
arms into your own directory).

**Your job now (before the pilot data arrive):** pre-register and build the analysis that answers YOUR question from
the pilot, so it runs the moment a cell is complete:
1. Primary estimands, contrasts and the decision rule written down *before* seeing pilot outcomes (what result leads
   to which method choice), using the design's actual propensities / blocks / repeated-init clustering; state what the
   pilot can and cannot resolve at its size (the HANDBACK power section gives n_eff).
2. A script in your ideation directory that takes a list of finished arms (or a run root + cell) and produces the
   tables / estimates / plots for your question; tested end-to-end on the smoke tables.
3. What you would need from the full campaign (arms, episodes) if the pilot is inconclusive.
4. Keep the owner's constraints: minimize IR at SR ≈ pure inference; the method must transfer to another benchmark /
   robot — anything calibrated must be calibrated from the library (plus a few recorded trajectories) with a recipe.
Write `PREREG.md` in your ideation directory (estimands, rules, commands, smoke test output) and end with a short
summary. Do not wait for data; do not poll.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 22-25,66-69`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Read-only on everything outside your ideation directory. Never start servers, workers or chains; never touch tmux
  sessions, ports 23100-23199, timan107 or other hosts (no tether). No git. No `rm -rf`. Never `pkill -f`. Do not edit
  `src/`, the P3 directory or any run root. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>

<your_question>
Q3 where to MISS (ideation dir `rounds/r06/ideation_Q3/`). The pilot randomizes calls before the guards (factorial: distance-quantile bins with p .125/.25/.5, policy duration 5/10, hold 1–3, cooldown 1), a delayed single call (window: delay 0/1/2 anchors), and fixed-dose coins at every anchor, with logged propensities; every decision also has the shadow policy chunk and cache/policy disagreement. Pre-register the estimator of the causal value of a call as a function of state features / timing (IPW or doubly robust with the logged propensities, init-clustered; reuse `rounds/r04/k5_rand/estimate.py` where it fits), the candidate placement signals (distance, disagreement, K4 spread, calibrated LOO CDFs, guard signals, progress), and the rule that turns it into a placement policy that beats B at equal or lower IR. Read also `rounds/r06/ideation_G/REPORT.md` (guard calibration, unified conformal α).
</your_question>
