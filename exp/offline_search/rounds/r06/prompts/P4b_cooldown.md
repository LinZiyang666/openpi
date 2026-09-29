<task>
Continuation of your P4 thread. Coordinator decision on your blocker (recorded in `SELECTION.md` §7, made before any C
outcome): the one-free-anchor cooldown applies **only after stall-triggered calls** (`slow_confirmed` → call → the next
free anchor cannot be a stall call or any call); R-placed and uniform lottery calls have **no** cooldown, exactly like
`RiskLottery`/`CacheDose`. Controller version string `R6-C-v2` (randomization key default `R6-C-v2`).

Do:
1. Implement it in `methods.py` (e.g. `cooldown_scope in {'stall','all'}`, default `'stall'`; keep `'all'` only for tests),
   in `fit_calibration.py`/`budget.py` (the exact cooldown integration now only follows stall calls), and in the specs:
   regenerate `emit_arms_c32.json`, `emit_arms_smoke2.json` and the recording/validation prefit specs with the v2 kwargs.
2. Rerun the dry-run feasibility on the pilot DRYRUN_TEST_INITS for ρ ∈ {.18,.30,.45} (no-stall and stall) and report the
   new floor/ceiling table; rerun the replay/packaging tests that touch cooldown/budget. Keep all other semantics.
3. Make the calibration artifact location a parameter so the coordinator can use a durable `<CAL>` =
   `/home/weiland/trace_runs/os_closed_loop/r06_c_cal/cal` (you may not write there: list the exact copy + prefit
   commands for that path; prefit outputs go to `/tmp/q1_method_c_fits/final_*`).
4. Check `exp/offline_search/rounds/r06/p3_profiling/chain_p3.sh`'s collect step: the tether single-file ceiling is
   447,074,607 bytes, and ten P3 episodes can exceed it. Say whether the recording should run in stream mode
   (`chain_p3.stream.frozen.sh` + `stream_receiver`, as the pilot finished) or file mode with
   `r06_p3_pilot/ops/collect_client_pilot.py`, and give the exact patched chain for `<RUN>/chain_calibration.sh` for
   the recommended mode (a new file in your directory; do not edit the P3 directory).
Update HANDBACK.md (short section at the top). Final message: short summary with the new feasibility table.
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 14-17,58-61`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only under `exp/offline_search/rounds/r06/ideation_Q1/method_c/` and `/tmp/q1_method_c_fits/`. Never start
  servers, workers or chains; never touch tmux sessions, ports 23100-23199; read-only `tether exec` only. No git. No
  `rm -rf`. Never `pkill -f`. Do not edit `src/`, `closed_loop/*`, other agents' files or any run root. Do not read
  `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>
