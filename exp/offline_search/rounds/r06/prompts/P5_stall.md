<task>
Continuation of your R6 thread. The coordinator selected the R6 method: read `exp/offline_search/rounds/r06/SELECTION.md`
completely first (configuration C; your FINAL §3 stall/progress-reliability proposal is its stall component; the call
budget of your step 5 is replaced by C's shared ρ budget, solved by the controller, so you only provide the status).
A Q1 agent builds the C controller in parallel against the interface in SELECTION.md §5.

Deliver under `exp/offline_search/rounds/r06/ideation_Q3/stall/`:
1. **`stall.py`** implementing exactly the interface in SELECTION.md §5 (`StallModel.fit/save/load`,
   `StallTracker.observe/status`) and your FINAL §3 steps 1–4: library windows at L = min(H, 2R), W = max(2,
   ceil(.05·T_med)), per-template monotone alignment DP in the deployed A metric (the same fitted representation and
   metric as A for that cell), K = min(5, E−1) best templates, span adjustment, LOEO calibration on successful library
   episodes with per-episode nearest-context references, e90 / a10, the ≥ 4-episode availability rule, and the
   `slow_confirmed` / `slow_ambiguous` / `ok` / `inactive` states. Deterministic tie-breaks as you specified. Library
   and manifest fields only; no benchmark/robot constants.
2. **Artifacts**: fit `StallModel` for all 8 cells into `/tmp/q3_stall_fits/<cell>/` (fingerprinted to bank + fit);
   list commands; the coordinator copies them.
3. **Bmech judge**: B with only the no-progress MISS bit disabled, keeping its blind-LOOK veto and diagnostic histories
   (`p2_ablations/judge.py` disables both — read it and the B judges; subclass, no shared-file edits), plus emit specs
   `emit_arms_bmech.json` for the four 50-episode cells (`<RUN>` placeholders, `--os-root
   /home/weiland/trace_runs/offline_search_store`, other fields copied from the deployed B specs).
4. **Tests / measurements** (CPU, recorded data only): LOEO tables reproducible; tracker status rates on the pilot's A
   and B trajectories per cell (how often `slow_confirmed` / `slow_ambiguous` fire, and at B's no-progress-flagged
   anchors — report the overlap on both L10-50 cells and GR00T Spatial-50); per-anchor CPU time of `observe()`
   (median / p99, per cell). Pilot trajectories are official test inits: use them only as descriptive/engineering
   checks, never to tune any constant. Bmech replay identity: with the flag off it must equal B except for the masked
   MISS decisions.
Write `HANDBACK.md`. Final message: short summary (≤ 12 lines).
</task>

<hard_constraints>
- CPU: prefix every python command with `taskset -c 22-25,66-69`, at most 4 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Write only under `exp/offline_search/rounds/r06/ideation_Q3/stall/` and `/tmp/q3_stall_fits/`. Never start servers,
  workers or chains; never touch tmux sessions, ports 23100-23199, timan107 or other hosts (no tether). No git. No
  `rm -rf`. Never `pkill -f`. Do not edit `src/`, `closed_loop/*`, other rounds' or agents' files, or any run root. Do
  not read `tests/review_tests/`.
- Do not stop to ask questions: choose the most reasonable reading, state it, continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed with commands and paths; label anything unverified.
</grounding_rules>
