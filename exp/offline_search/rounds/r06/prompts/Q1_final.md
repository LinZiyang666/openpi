<task>
Continuation of your R6 thread. **All 8 pilot cells are complete** (216 arms × 20 episodes = 4,320 episodes, run root
`/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot`). The coordinator produced strict `read_v2` tables for every cell
in `r06_p3_pilot/tables/<cell>/` (`--require-stage-counts --require-snapshots` passed on all 8) and already ran your
preregistered per-cell script on each cell into your ideation directory (`pilot_<cell>/`). Note on data: one arm
(`groot_l10_50_window_r2`) was repaired after an attempt-key collision (see `p3_profiling/ATTEMPT_REUSE.md`); its
repair copies live in `r06_p3_pilot/repair_archive/`, not under `runs/`.

Now produce the **preregistered all-cell result** for your question: run your analysis over all 8 cells jointly exactly
as frozen in your PREREG (primary estimands, simultaneous intervals, survival / selection rules), report which
preregistered decisions are reached and which remain inconclusive, and what the full continuation (+31,680 episodes)
would buy at the measured variances / ICCs (replace the planning assumption .20 / ρ = .3 by the measured values).

Then, separately and clearly labelled **post-hoc / exploratory**, integrate the closed-loop evidence that arrived after
your preregistration (500-episode arms, paired on the same (task, init) set; all numbers are in
`exp/offline_search/rounds/r06/PAPER_AB.md`, `ABLATIONS.md` and the ledger `logs/offline_search_exploration.log.md`
§10 entries of 2026-09-28/29):
- A / B three-replicate table (`PAPER_AB.md`).
- Trigger leave-one-out of B on the four 50-episode cells: the no-progress guard carries essentially all of B's gain on
  both models' LIBERO-10-50 and ≈ 5 of 7 pp on π0.5 Spatial-50, but on GR00T Spatial-50 removing it **raises** SR to
  .906 (> B .875 and A .867, p ≤ .01) at lower IR; the other three guards are individually removable everywhere.
- Frontier-completion arms (`/home/weiland/trace_runs/os_closed_loop/r06_frontier/runs/*/summary.json`, specs in
  `ideation_Q2/frontier/adapters/`): B + extra random anchor dose d and risk-allocated target-IR lotteries ρ, e.g.
  π0.5 l10-50 B+.25 .842@.26, B+.5 .880@.34 (vs pure L10 .904@.5, p=.21), B+.75 .874@.42, ρ.35 .862@.34; GR00T l10-50
  B+.5 .808@.36, B+.75 .850@.43 (vs L10 .866, p=.43), ρ.35 .832@.34 (p=.11); π0.5 l10-500 B+.125 .842, B+.25 .882,
  ρ.24 .884@.24 (vs L10 p=.33); π0.5 sp50 B+.5 .962@.32 (vs L10 .986, p=.03); GR00T sp50 B+.125 .878, B+.25 .902@.23.
- Visual-key PCA ablation (direct token PCA vs 4×4 pooled PCA) in `ABLATIONS.md` §2.
Read the arms' own summaries / journals for exact values; recompute owner IR from `cost_ledger` (π0.5 .152·v + .848·m,
GR00T .148·v + .852·m).

Deliverable: `FINAL.md` in your ideation directory: (1) preregistered result, (2) exploratory integration, (3) a
concrete recommendation for the R6 method on YOUR question — what the library-derived / calibrated signal is, how it is
calibrated from the library (plus at most a few recorded trajectories) so it transfers to another benchmark / robot,
what closed-loop arms (≤ 8 configurations, paired against A, B and pure inference) would validate it, and what result
would falsify it. Final message: short summary (≤ 12 lines).
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 14-17,58-61`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Read-only outside your ideation directory. Never start servers, workers or chains; never touch tmux sessions, ports
  23100-23199, timan107 or other hosts (no tether). No git. No `rm -rf`. Never `pkill -f`. Do not edit `src/`, the P3
  directory or any run root. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified; keep preregistered and
post-hoc results visibly separate.
</grounding_rules>

<your_question>
Q1 library quality. Your per-state candidates (C, D, R, Qrisk) plus secondary dispersion / V7 / neff; apply the preregistered 8-cell survival rule and give the portable recipe for the surviving score (or say none survives). Also state how the score should feed Q2 (budget) and Q3 (placement), given the new evidence that extra random calls help sparse libraries but not dense ones.
</your_question>
