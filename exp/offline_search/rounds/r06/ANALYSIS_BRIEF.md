# R6 analysis brief

Write `rounds/r06/ANALYSIS.md`, the R6 round report. Follow the style and rigor of `rounds/r05/ANALYSIS.md`:
- a short result summary at the top;
- provenance and conventions;
- numbered sections with tables;
- a synthesis;
- ranked proposals for the owner.

Report only what was computed from the arms' own `summary.json` and client journals. The ledger
(`logs/offline_search_exploration.log.md` §10, 2026-09-28/29) is context, not a source of numbers.

## What R6 set out to do (owner, 2026-09-28)

1. **Paper finalization.** Configuration A is Commit-Cache. Configuration B is A plus guard-triggered committed rescue.
   This covers three replicates each, GR00T B, the metric ablation and the trigger leave-one-out.
2. **The owner's three questions.**
   - Q1: How can library quality be judged (leave-one-out)?
   - Q2: How many MISSes should be spent?
   - Q3: Where should MISSes go so that SR is kept at minimal IR?

   The answers must be general across benchmarks and robots.
3. **A superset profiler** that collects the needed data in as few runs as possible.
4. **The IR–SR Pareto frontier** per cell. The ideal is minimal IR at pure-inference SR.

## Inputs (all under `exp/offline_search/rounds/r06/` unless noted)

- **Headline and ablations:**
  - `PAPER_AB.md` (regenerate with `ops/paper_ab.py`);
  - `ABLATIONS.md` (`ops/ablations.py`);
  - the Bmech arms in run root `r06_method`.
- **Pilot and preregistration:** `ideation_Q{1,2,3}/PREREG.md` and `FINAL.md`. Each FINAL.md now carries a
  coordinator erratum at the top.
- **Method selection:** `SELECTION.md`, including §7 and §7b, which record amendments made before any C outcome.
- **Configuration C implementation:** `ideation_Q1/method_c/HANDBACK.md` (code, calibration, tests) and
  `ideation_Q3/stall/HANDBACK.md` (stall component).
- **C calibration** on non-test B-val initial states:
  - run root `r06_c_cal`;
  - tables in `r06_c_cal/tables/<cell>`;
  - calibrations in `r06_c_cal/cal/<cell>/calibrated/calibration.json`.
- **C smoke:** run root `r06_c_smoke`.
- **C validation:** run root `r06_c_validation`, 28 arms. Evaluate them with `ops/c_validation.py`, which implements
  SELECTION §4 exactly.
- **C validation verdicts (already computed):** `C_VALIDATION.md`.
- **Frontier:**
  - final version `frontier_final/` (all arms, 2026-09-29 13:2x; `ledger_tolerance_flags.json` lists 7 repaired arms whose ledger differs by ≤ 2 %);
  - the interim version is `frontier_repaired/`;
  - completion arms live in run root `r06_frontier` (41 of the 43 planned; 2 skipped as redundant).
- **Figure:** `~/projects/openpi_ext/artifacts/frontier_r6/frontier_r6_final.png` / `.pdf`.
- **Data-quality repair:**
  - `closed_loop/ops/remote/purge_exc.py`;
  - `closed_loop/ops/chain.sh` (the purge step);
  - each affected arm's `client/purged_exceptions.jsonl`.

Run roots are under `/home/weiland/trace_runs/os_closed_loop/`.

## Required sections

1. **Data-quality repair.** Report:
   - the bug: `examples/libero/main.py:390-395` swallows generic exceptions and returns `success=False`;
   - how it was detected: the client_timing row carries `termination_reason == "exception"`;
   - its scope: 33 arms and 316 episodes, all in R6, with none in R2–R5 or in the pilot;
   - the repair and its verification;
   - which earlier conclusions changed, with before and after values;
   - the ledger/client decision-count tolerance used for 7 repaired arms (≤ 2 %, listed in
     `ledger_tolerance_flags.json`).
2. **A / B headline.** Report the three-replicate table and the replicate spread.
3. **Ablations.** Cover the metric, the direct-token PCA (with the withdrawn claim), the trigger leave-one-out and
   Bmech (the mechanism).
4. **Q1 / Q2 / Q3 results.**
   - Preregistered outcomes first. For each question, state the decision reached, what stayed inconclusive, and why:
     one initial state per task in the calibration split.
   - What the full continuation would buy.
   - Make clear which statements are post-hoc.
5. **Configuration C.** Cover:
   - the design (SELECTION §1);
   - the amendments (§7 and §7b) and their timing;
   - the calibration (feasibility table, and the Ehat coefficients a and b per cell);
   - the smoke checks;
   - every §4 acceptance rule with its verdict, exactly as preregistered: placement R30 vs U30; the stall component
     C30 vs R30, B and B-without-no-progress; reach C45 / Cmax vs L10 / L5 with the 2 pp CP bound; the dense cells
     C18 vs B; and cost calibration, realized IR within ρ ± .02.

   State plainly which parts of C help, where, and by how much, and which parts do not.
6. **Frontier.** Per cell, give:
   - the Pareto points and the pure-inference references;
   - the lowest IR whose point estimate reaches pure L10 SR;
   - the lowest IR with nominal 2 pp NI, and with simultaneous NI.

   Show where configuration C sits on each frontier. The figure is
   `~/projects/openpi_ext/artifacts/frontier_r6/frontier_r6_final.png`; reference it, do not embed plotting code.
7. **Synthesis and proposals.** Answer the owner's three questions in plain terms, using the whole R6 evidence.
   Give a ranked list of next steps: what to keep, what to drop, and what to test on a second benchmark or robot.
   Include the pending owner decisions: the full continuation, the unsupervised whitening control, and the l10 +0.04
   display convention.

## Conventions

- **Owner IR:**
  - π0.5: .152·v + .848·m;
  - GR00T: .148·v + .852·m;
  - blind and tail slots cost 0;
  - pure inference: L=10 costs .5 and L=5 costs 1.0.
- **Pairing:** every arm is paired on the same 500 (task, init) pairs.
  - Use the exact McNemar test for single-run pairs.
  - For intervals, use a task-stratified init bootstrap with 10,000 draws and seed 20260929.
  - B / A replicates are averaged within each pair.
  - Non-significance is not equivalence.
  - NI uses Q2's conservative Clopper–Pearson paired bound with margin .02.
- **Times** are in CDT.

## Constraints

- Prefix every Python command with `taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src` and run `.venv/bin/python`.
- Never touch CPUs 38-43 or 82-87.
- Write only `rounds/r06/ANALYSIS.md`, new files under `rounds/r06/analysis_scripts/`, and new output directories under
  `rounds/r06/` whose names you choose.
- Do not start servers, workers or chains. Do not touch tmux sessions, ports or timan107.
- No git. No `rm -rf`. No codex.
- Do not read `tests/review_tests/`.
- Plot scripts never go in the repo.
