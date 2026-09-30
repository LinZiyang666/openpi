# R7 analysis brief

Two analysis agents work from this brief with **disjoint jobs** (they must not duplicate each other):
- **A1 (codex gpt-6-astra): the quantitative verdicts.** Compute every preregistered acceptance rule, all paired
  statistics, the frontier placement and the cost accounting; write `rounds/r07/analysis_scripts/` (new scripts) and
  machine-readable outputs under `rounds/r07/analysis_r7/`, and draft `rounds/r07/ANALYSIS.md` §§1–5 (tables).
- **A2 (opus agent): mechanism, data quality and generality.** Explain *why* the numbers came out as they did from the
  decision logs: where extensions / wrist looks / calls happened (by stage), which episodes flipped versus A and at what
  stage, valve and LOOK reasons, cost reconciliation and exception-episode audit, and a generality/portability
  assessment against the E5 checklist; then ranked proposals for R8. Return the text (the coordinator saves it as
  `ANALYSIS.md` §§6–8); A2 may write helper scripts under `rounds/r07/analysis_scripts/a2_*`.

Report only what is computed from arm `summary.json`, client journals and server decision logs. The ledger
(`logs/offline_search_exploration.log.md` §10 R7) is context, not a source of numbers.

## What R7 set out to do
Owner (2026-09-29/30): study **stage-level allocation** — split trajectories into stages, spend more on hard stages and
apply atomic levers (fewer calls, look less, look half) on easy ones; the method must be simple, effective and general
(no LIBERO-special tuning, no model swaps). Terminology: atomic levers / signals / allocation policy.
Read `rounds/r07/SELECTION.md` (methods, preregistered acceptance §5, freeze record §8), `rounds/r07/IDEATION_BRIEF.md`,
the five `ideation/E*/PROPOSAL.md` and `PROFILE_ANALYSIS.md`, and the four coder `HANDBACK.md`s.

## Inputs
- **Evaluation (test pairs, 500 per arm):** run root `/home/weiland/trace_runs/os_closed_loop/r07_main/`, 28 arms:
  SF1 ×8, UF1 ×8, SW ×4 (π0.5), CU(.30) ×4 and CT(.30) ×4 (sparse cells). Arm names `r7_<cell>_{SF1,UF1,CU30,CT30}`,
  `r7_sw_pi05_<cell>`.
- **Profile (non-test B-val, 20 per arm, 44 arms):** `/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/`;
  report `rounds/r07/profile_results/closed_loop/profile_report.json`; offline tools `profile_results/offline/`.
- **References on the same 500 pairs:** A and B three replicates (`rounds/r06/analysis_scripts/common.py:ab_arms`),
  pure L10/L5 (`common.py:l10_ref/l5_refs`), R6 C-family arms (`r06_c_validation`), the R6 frontier
  (`rounds/r06/frontier_final/`, `ops/frontier_refresh.py`), `rounds/r06/ANALYSIS.md`.
- Tools: `rounds/r07/c4_profile/{profile_report,stage_value,follow_audit,failure_clock}.py`,
  `rounds/r06/analysis_scripts/common.py` (loaders, bootstrap, McNemar, NI bound), `rounds/r06/ops/c_validation.py`.

## Conventions
- Owner IR per five-control decision: π0.5 `.152·v + .848·m`, GR00T `.148·v + .852·m`; blind 0. **SW arms: owner cost
  from the per-decision `owner_cost` / camera counters** (wrist look .055198, completion .049890, R4 proportional-latency
  assumption, labelled), not the ledger's two-camera pricing. Report both for SW.
- Pairing on (task, init). Exact McNemar for single runs; task-stratified init bootstrap 10,000 draws seed 20260930;
  A/B as three-replicate means per pair; NI via Q2's conservative CP bound, margin .02. Non-significance ≠ equivalence.
- Exception episodes: `chain.sh` purges client exceptions automatically (`EXC_PURGED` events in `runs/chain.log`);
  audit that no `termination_reason == "exception"` episode survives in any accepted journal.
- Times in CDT.

## Required content (A1 tables; A2 text)
1. Data quality and cost reconciliation (both; A2 leads the explanation).
2. Every SELECTION §5 acceptance rule with its verdict exactly as preregistered (A1): (1) SF vs A pooled 8-cell SR lower
   bound > −1.0 pp and IR lower in ≥ 6/8; (2) stage signal: SF vs UF; (3) SW vs A on π0.5; (4) CT vs CU at
   |ΔIR| ≤ .015; (5) frontier placement per cell.
3. Per-cell table: SR @ owner IR for A (3-rep), SF1, UF1, SW, CU, CT, B (3-rep), pure L10; paired deltas with intervals.
4. Frontier: where R7 points sit on each cell's R6 frontier; any new non-dominated point; lowest IR reaching pure L10
   (point estimate) before/after R7.
5. Stage-level mechanism (A2): extension grant share and valve fire rate by stage; where UF's extra extensions happened
   that SF refused and what happened to those episodes; SW camera mode by stage; CT call placement by stage versus CU.
6. Synthesis for the owner (plain language, no codenames) and ranked R8 proposals, including what stays shelved
   (few-step denoising, library growth, task-level knob, SF+SW, SA).

## Constraints
- Python prefix: `taskset -c <range> env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python`; A1 CPUs 22-25,66-69; A2 CPUs 26-29,70-73. Never 38–43,
  82–87. No servers, workers, chains, tmux, ports, timan107, git, `rm -rf`, `pkill -f`; do not read
  `tests/review_tests/`. Plot scripts never go in the repo.
