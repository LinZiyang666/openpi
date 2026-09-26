# R1 coding brief (common to the 4 coding agents)

Read first: `logs/offline_search_exploration.log.md` (protocol; §3.1 valid dims, §5, §6, §8),
`exp/offline_search/harness/README.md` (Method API, metrics, runner), `exp/offline_search/rounds/r01/SELECTION.md`
(the approved method list), `exp/offline_search/rounds/r01/FINDINGS.md` (R0 numbers), and the ideation originals
`rounds/r01/NOTES_ideation_A.md`, `NOTES_ideation_B.md` (exact algorithms + the numbers the ideation agents measured —
your smoke results must be consistent with them; diag scripts in `rounds/r01/diag_A`, `diag_B` show working code).

## Where / how
- Code: `exp/offline_search/rounds/r01/<family_dir>/` (create it; add `__init__.py`). One or more modules with
  `Method` subclasses (from `exp.offline_search.harness.api`). Each variant is run as a separate method NAME, so make
  `name` encode the variant (e.g. `M1_big_a025_med3_stWin`), deterministic from kwargs.
- Deliver `<family_dir>/batch.json`: a list of `{"method": "exp/offline_search/rounds/r01/<family_dir>/<module>.py:<Class>",
  "kwargs": {...}, "family": "<family_dir>", "cells": "all", "subsample": null|"tok", "allow_gpu_fit": false}` for
  every variant to run in the full round. The coordinator runs all families together with the parallel batch scheduler
  (`python -m exp.offline_search.harness.batch`, budget 88 workers). Do NOT run full-scale yourself.
- Store root: `/dev/shm/offline_search_store` (hot arrays in RAM; tok/ symlinked to SSD). Libraries per model×suite:
  `current`, `bpool_all` (GR00T: 500 episodes; π0.5: ≈ current), `bpool_cs` (π0.5 only: 500 episodes, Aug batch).
  "Big library" = `bpool_cs` for π0.5, `bpool_all` for GR00T.
- Harness extensions landing concurrently (check `harness/README.md`; code against them): `q.prev_hit` (bool | None:
  was the previous decision served from the library; None at step 0; in the trace inf arms are always False, cache arms
  always True), `Result.library` may name any library present in the store for the cell, method attribute
  `uses_nonlibrary_action` (reference rows only), and the batch scheduler. If `prev_hit` is not in the API yet when
  you smoke, derive it identically as a temporary fallback (prev row's executed chunk equals a library chunk on [:, :7])
  and leave a TODO.
- Smoke every method you deliver: `python -m exp.offline_search.harness.smoke --method <path:Class> --kwargs '<json>'
  --cell <cell> --episodes 5 --root /dev/shm/offline_search_store` on at least one π0.5 and one GR00T cell, inf and
  cache. Report the smoke numbers.

## Rules for methods
- ⛔ Valid dims: actions only [:, :7] (dim 6 gripper), executed steps [:5]; π0.5 rs only [:8]. Use `harness/dims.py`.
  Never compute action distances / statistics over dims 7..31; never whiten/PCA over padded rs dims.
- Only online-legal inputs (QueryView). Fitted parameters from the library only.
- Regime awareness (critical): after a HIT (`prev_hit=True`) the previous chunk's tail is a library chunk — continuity
  to it is trivially ~0 along the same library trajectory, so any continuity-based confidence must not be used as-is in
  that regime (it would be spuriously high). Report per-regime behaviour via extras.
- Expose per-decision internals via `Result.extras` (small scalars: e.g. cont, rs_dist, disp, regime flag, chosen
  library row's episode/step), so profile tools and the analysis agent can dissect behaviour.
- Returned synthesized actions: full (H,32) chunk; only [:5,:7] is scored; build it from library rows only (except
  explicitly flagged reference rows).
- Speed: vectorize (per-task matrices prebuilt in `fit`, GEMM per query); workers are single-threaded; fit runs in each
  job's main process (single-threaded unless you use explicit threads) — keep fit ≲ 1–2 min per cell; cache heavy fitted
  artefacts (e.g. PCA bases) under `ctx.scratch` or a family cache dir inside your family dir's `_cache/` (gitignored-size
  files only, no large arrays in the repo tree: put big arrays under
  `/home/weiland/trace_runs/offline_search_store/derived/r01/<family_dir>/`).
- Memory: forked workers share read-only pages; don't copy big library arrays per query; float32.
- CPU only for everything (CUDA_VISIBLE_DEVICES=""). Use as many cores as useful for one-off precomputation scripts
  (≤ 32 processes, the machine is shared with other R1 agents).
- No edits outside your family dir and your derived dir; never modify harness/ or profile/ (report harness bugs to the
  coordinator); no git; no `rm -rf`; never `pkill -f`; do not read tests/review_tests/.

## Hand-back
File list, the batch.json content summary (how many variants, names), smoke numbers per method (err, AURC, regime
split if relevant), measured fit time and ms/query from smoke, and any caveats.

## ⛔ CPU budget (owner rule, mandatory)
The machine is shared by several agents running at once. You get a fixed logical-CPU range in your prompt
(physical cores + their hyperthread siblings; topology: CPUs 0–43 are physical cores, 44–87 their siblings).
- Prefix EVERY command that runs Python/numpy work with `taskset -c <your range>`.
- Never run more processes/workers than the number of logical CPUs in your range; OMP/MKL/OPENBLAS threads = 1 per process
  (a single one-off BLAS-heavy process may use threads = your range size instead).
- If you started something that exceeds the budget, stop it by PID (never `pkill -f`) and relaunch within budget.
- Full-scale runs belong to the coordinator (it reserves CPUs with batch `--cpus`).
