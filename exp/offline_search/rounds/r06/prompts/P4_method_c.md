<task>
Continuation of your R6 thread. The coordinator selected the R6 method: read `exp/offline_search/rounds/r06/SELECTION.md`
completely first (configuration C = A + library-calibrated budgeted rescue; your R score is its placement signal). You
build the **C controller and everything it needs except the stall component** (a Q3 agent builds that in parallel
against the interface in SELECTION.md §5; code against the interface and use a stub `StallTracker` whose status is
always `inactive` until `ideation_Q3/stall/stall.py` exists).

Deliver under `exp/offline_search/rounds/r06/ideation_Q1/method_c/`:
1. **Controller** (`methods.py`): one class, e.g. `CalibratedRescue`, built on the existing A method / adapter framework
   (read `ideation_Q2/frontier/adapters/methods.py` (`CacheDose`, `RiskLottery`) and `budget.py`,
   `closed_loop/README.md`, `rounds/r05/q1_commit/` and `rounds/r06/p1_groot_commit/HANDBACK.md` for the
   method↔plugin contract, the 10-control commit, hold/handback, and the GR00T gripper convention). Kwargs:
   `rho`, `placement in {'uniform','R'}`, `stall_model_path=None`, `calibration_path`, `random_seed`,
   `randomization_key`. At each free vision anchor: if a stall tracker is configured, feed it the observation first;
   `slow_confirmed` (not in cooldown) → call; `slow_ambiguous` → serve cache and force the next decision to be a vision
   LOOK after min(R, L) controls (at most one extra LOOK per W·L executed controls; find how B's no-progress blind-LOOK
   veto forces a LOOK and reuse that mechanism); otherwise an independent keyed Bernoulli with p(s) = d(ρ) (uniform)
   or min(1, λ·Ehat(s)) (R). Calls commit min(H, 2R) controls, hold 1, fresh retrieval anchored on the executed tail,
   one free-anchor cooldown, exactly B's rescue semantics. Log per decision: R(s), Ehat, p, coin, stall status, extra
   LOOK flag. No shadow/profiling at deployment.
2. **R bank artifact** (`precompute_r.py`): r_j per bank row for all 8 cells with the deployed retriever/kernel and
   whole-source-episode exclusion (your FINAL §4 steps 1–3), fingerprinted to the bank + fit. Write to
   `/tmp/q1_method_c_fits/<cell>/` and list the commands; the coordinator copies them.
3. **Calibration fit** (`fit_calibration.py`): from calibration recordings (read_v2 tables of P3 v2 A-cohort runs with
   the primary shadow chunk at every anchor) fit Ehat = a + b·R (your weighted recipe) and solve λ(ρ) so that the
   modeled per-episode owner IR on those recordings equals ρ, **including** modeled stall calls and extra LOOKs when a
   `StallModel` is given (replay the tracker over the recorded keys); uniform mode solves d(ρ) the same way. Report
   infeasible targets. Dry-run it on pilot A tables ONLY to test the code, and label that output `DRYRUN_TEST_INITS`
   (pilot inits are official test inits; such artifacts must never be used by a validation arm).
4. **Calibration recording specs**: 10 episodes per cell = one per task, on LIBERO **B-val** initial states (non-test;
   `exp/ablation_study/config/common/split_<suite>.yaml` and `exp/common/data/db_init/libero/<suite>/`; choose the
   lowest-numbered val init per task, outcome-blind). Find out how the closed-loop driver chooses init states (the
   remote runner lives under `/scratch/zixuans8/openpi_trace/os_cl/` on timan107; its local sources are in
   `exp/offline_search/closed_loop/` and `rounds/r06/p3_profiling/`), whether it accepts a non-default init-state set,
   and if not, write the minimal change as a patch file plus exact deploy commands for the coordinator. Check whether
   the 500-episode banks contain episodes from those B-val inits and state how that is handled (e.g. exclusion of that
   source episode during recording) — do not guess.
5. **Validation specs** (`emit_arms_c32.json`): the 32 arms of SELECTION.md §4 (U.30/R.30/C.30/C.45 on the four
   50-episode cells, U.18/R.18/C.18 on the four 500-episode cells; Bmech specs come from the Q3 agent) with `<RUN>`
   placeholders, `--os-root /home/weiland/trace_runs/offline_search_store`, `<CAL>`/`<STALL>` artifact placeholders, and
   a 2-arm smoke recipe (one π0.5, one GR00T, 4 episodes each).
6. **Tests** (CPU replay on recorded queries): ρ at the A floor / p ≡ 0 is bit-identical to A; uniform mode reproduces
   `RiskLottery(allocation='uniform')` realized rates; R mode realized mean p ≈ target on the calibration replay; seeds
   reproducible; commit length 10; GR00T gripper sign unchanged vs the existing judge; stub tracker path exercised.
   Write `HANDBACK.md` (what exists, commands, artifact paths, what the coordinator must run).
Final message: short summary (≤ 12 lines).
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 14-17,58-61`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only under `exp/offline_search/rounds/r06/ideation_Q1/method_c/` and `/tmp/q1_method_c_fits/`. Never start
  servers, workers or chains; never touch tmux sessions, ports 23100-23199, timan107 or other hosts except read-only
  inspection (`tether exec timan107 -- bash -lc '<read-only command>'` is allowed; no tether push). No git. No `rm -rf`.
  Never `pkill -f`. Do not edit `src/`, `closed_loop/*`, other rounds' or agents' files, or any run root. Do not read
  `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
