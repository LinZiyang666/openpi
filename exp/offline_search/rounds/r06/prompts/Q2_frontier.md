<task>
Continuation of your Q2 thread (how much policy inference to buy). The owner's goal restated: use as little IR as possible
while reaching pure-inference SR; the owner wants the **IR–SR Pareto frontier** for every cell, and wants to know, per
cell, the minimum IR at which SR reaches pure inference. In the 50-episode-library cells B (guard-triggered rescue) is
still below pure inference (e.g. π0.5 LIBERO-10-50: B .817 @ owner IR .181 vs pure inference L=10 .904 @ .5 and L=5
.851 @ 1.0; GR00T LIBERO-10-50: B .719 @ .221 vs pure inference .866). Final three-replicate A/B table:
`exp/offline_search/rounds/r06/PAPER_AB.md`.

Do two things now (CPU only, from existing data), and prepare the third:
1. **Frontier table from all existing closed-loop runs.** Collect every 500-episode (or otherwise complete) closed-loop
   arm in `/home/weiland/trace_runs/os_closed_loop/*/runs/*/summary.json` (+ `client/journal.jsonl`) for the 8 cells
   (π0.5 / GR00T × LIBERO-10 / Spatial × 50- / 500-episode library): A, B, periodic MISS, guard-only, confidence-quantile
   (R3), 5-step pure cache, policy tails, library-growth variants, pure inference L=5 / L=10, etc. Recompute **owner IR**
   from each summary's `cost_ledger` v and m (π0.5 .152·v + .848·m, GR00T .148·v + .852·m; blind decisions cost 0; pure
   inference L=10 = .5, L=5 = 1.0), and state per arm which library it used (50 vs 500 episodes, and variants like
   demo100/200/300, grow250 which are separate libraries — keep them in separate columns or exclude them from the 50/500
   frontier, say which). Use `logs/offline_search_exploration.log.md` §10 and the round ANALYSIS files to map arm names to
   methods; skip smoke / subset runs and anything not comparable (different episode set), and list what you skipped and why.
   Output: `ideation_Q2/frontier/frontier_points.csv` (cell, arm, run, method family, library, n, SR, SR 95% interval,
   owner IR, v, m, notes) and `frontier.md` with, per cell, the Pareto-efficient points (no other point has ≥ SR at ≤ IR),
   the pure-inference references, and the current gap: the lowest-IR point whose SR is non-inferior to pure inference
   (paired exact McNemar on (task, init) where the episode sets match; state the margin you used — the Q2 prereg margin
   2 pp — and report both L=5 and L=10 references).
2. **A draft figure** of the per-cell frontiers (SR vs owner IR, pure-inference references as horizontal lines, Pareto
   staircase), written OUTSIDE the repo to `/home/weiland/projects/openpi_ext/artifacts/frontier_r6/` (png + pdf). The
   plotting script must be named `plot_frontier.py` (plot scripts are never committed; the repo ignores `plot_*.py`).
3. **A frontier-completion plan to run after the pilot**: per cell, a small set of 500-episode arms (target ≤ 6 per cell,
   ≤ 48 total) placed where the frontier is missing between B and pure inference — e.g. B plus a fixed anchor call rate
   d, your risk-allocated target-IR (ρ) policy at ρ values chosen to bracket the pure-inference crossing, and any existing
   method family that is already close. Use the pilot's per-cell dose curves (`r06_p3_pilot/tables/<cell>` for finished
   cells; your `pilot_<cell>` outputs) only to choose where to place points, not as final evidence. For each proposed arm
   say whether it already exists as code (which class / kwargs, e.g. existing periodic-B, P3 Profile with a fixed dose
   and no shadow, your lottery policy) or needs new code, and write the emit_arms specs for the ones that exist
   (`<RUN>` placeholders; copy other fields from the A / B specs). Mark the plan as provisional: it will be finalized
   when all 8 pilot cells are in (≈ 06:00 CDT).
Final message: short summary with the per-cell gap to pure inference.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 18-21,62-65`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Read-only outside your ideation directory and the artifact directory above. Never start servers, workers or chains;
  never touch tmux sessions, ports 23100-23199, timan107 or other hosts (no tether). No git. No `rm -rf`. Never `pkill -f`.
  Do not edit `src/`, the P3 directory or any run root. Do not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
