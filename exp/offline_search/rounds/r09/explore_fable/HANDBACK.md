# R9 explore_fable — hand-back (what ran, where things are, how to reproduce)

Explorer fable (Claude), 2026-10-01 20:0x → 2026-10-02 (CDT). Directory `exp/offline_search/rounds/r09/explore_fable/`.
Nothing outside this directory was edited; no git state was changed; no process, tmux session or port of anyone
else was touched. The other researcher's directory was not read before PROPOSALS.md was written.

## Deliverables

| File | Content |
|---|---|
| `REPORT.md` | Owner section in plain Chinese first, then the technical body |
| `DATA_ANALYSIS.md` | Evidence, commands, numbers for every analysis (sections 0–6) |
| `PROPOSALS.md` | Ranked proposals with per-cell estimates, failure modes and exact confirmation arms |
| `tools/` | Reusable offline tools (see below) + `tools/tests/test_tools.py` (14 unit tests, all pass) |
| `out/` | JSON/parquet outputs of every analysis (`paired/`, `shadow/`, `task_alloc/`, `synth/`, `student/`, `confirm_grown.json`, `digest_cells.json`) |

## Tools (all `python -m exp.offline_search.rounds.r09.explore_fable.tools.<name>`; CPU unless noted)

| Tool | What it does | Main outputs |
|---|---|---|
| `common.py` | Paths, prices, cell list, task-stratified bootstrap, JSON writer | — |
| `extract.py` | Per-arm episodes / decisions parquet + decision-aligned arrays (served, shadow, keys, state, rows, weights) from the osdebug.v1 capture via `debug.reader` | `derived/r09_fable/{episodes,decisions,arrays}/<arm>.*` |
| `extract_norm.py` | Adds `state_norm` arrays per arm | `derived/r09_fable/arrays_norm/<arm>.npz` |
| `paired.py` | Paired outcome matrix per cell; dose-assignment simulator (mixture SR/IR, bootstrap, paired delta); per-task Lagrangian hull; K-fold cross-validated hull | `out/paired/<cell>.json` |
| `shadow_gap.py` | Served-vs-shadow gap per decision/episode/task; noise floor from independent draws; AUROC of early features for failure | `out/shadow/{decisions,episodes,summary}_<arm>.*` |
| `task_alloc.py` | Per-task calibration signals, correlation with the P10−A gap, fold stability, calibration-size stability; signal-driven CV frontiers; `--rules`: pre-declared top-k rules with paired bootstrap vs IR-matched uniform mix | `out/task_alloc/{task_alloc,rules}.json` |
| `synth_eval.py` | Re-synthesis of the served chunk from the logged 16 neighbours with alternative rules, scored against the shadow (uses the frozen fit artifact) | `out/synth/<arm>.json` |
| `student.py` (GPU) | DAgger student MLP on (keys, state, task) → shadow chunk; validation on held-out inits; `--curve`: learning curves + kernel kNN on the labelled rows | `out/student/*.json` |
| `grow_library.py` | Extends a frozen BlindAWM fit with shadow-labelled rows as a registered library `grown` (no new serving code), with an offline emulation check | `<run>/fits/r9f_grown*.pkl` + `.json` |
| `confirm_grown.py` | Paired analysis (McNemar, bootstrap) of the confirmation run's journals | `out/confirm_grown.json` |
| `pertask_counts.py`, `print_hulls.py` | Printing helpers used while exploring | stdout |

Standard prefix: `taskset -c 22-29 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python`
(GPU tools: `CUDA_VISIBLE_DEVICES=0`, local RTX 4090, < 2 GB used). Full reproduction order: `extract` → `extract_norm` →
`paired` → `shadow_gap` → `task_alloc` (+ `--rules`) → `synth_eval` → `student` (+ `--curve`) → `grow_library` → closed loop → `confirm_grown`.
Total CPU time ≈ 20 min with 12 workers for extraction, < 10 min for the rest; GPU ≈ 5 min.

## Data written

- `/home/weiland/trace_runs/offline_search_store/derived/r09_fable/` — derived tables, ≈ 7 GB (disposable, regenerable).
- `/home/weiland/trace_runs/os_closed_loop/r09_fable_grown/` — confirmation run root: `arms_in.json`, `arms.json`, `config/`,
  `fits/` (6 grown artifacts; 3 used), `manifest_inits20_29.json`, `h100_sync/`, `runs/<arm>/` (journals, summaries, server logs),
  `state/`, `chain_console.log`.

## Closed loop that ran

One chain, run root `r09_fable_grown`, h100 ports 23240–23243 (4 single-replica π0.5 servers), WORKER_HOST=timan107, WPS=12
(48 workers), MAX_ATTEMPTS=2, POLL_SECONDS=30, OSCL_MANIFEST = 100 pairs (tasks 0–9 × inits 20–29; discovery, held out from the
grown rows which use inits 0–19). Arms in order: `r9f_ctrlA_p_l10_50` (pure cache A, frozen R5 fit), `r9f_grownA_p_l10_50`
(A + 6,763 shadow-labelled rows from the R8 A arm), `r9f_grownall_p_l10_50` (A + 73,945 rows from all 14 R8 arms of the cell),
`r9f_P10_p_l10` (pure policy, full model). Sync used the reserved forwarded port 23195 (`sync --concurrent`; the other
researcher's chain on timan108 was active, no asset conflict). tmux session `r9f_chain` (mine); the chain stops its own servers and
workers at the end (verified in `runs/chain.log`: SERVERS/driver stopped per arm). Results: see `DATA_ANALYSIS.md` §5.3 and
`out/confirm_grown.json`.

Commands used (from the repository root):

```
R=/home/weiland/trace_runs/os_closed_loop/r09_fable_grown
$P -m exp.offline_search.rounds.r09.explore_fable.tools.grow_library --cell pi05:l10:50 --arms A   --out $R/fits/r9f_grownA_p_l10_50.pkl
$P -m exp.offline_search.rounds.r09.explore_fable.tools.grow_library --cell pi05:l10:50 --arms all --out $R/fits/r9f_grownall_p_l10_50.pkl
$P -m exp.offline_search.closed_loop.ops.emit_arms --run-root $R --spec $R/arms_in.json
WORKER_HOST=timan107 $P -m exp.offline_search.closed_loop.ops.h100.control plan $R <arms>
WORKER_HOST=timan107 SYNC_PORT=23195 $P -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent $R <arms>
tmux new -d -s r9f_chain "WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 \
   OSCL_MANIFEST=$R/manifest_inits20_29.json taskset -c 22-25,66-69 env ... .venv/bin/python -m exp.offline_search.closed_loop.ops.h100.control chain $R <arms>"
$P -m exp.offline_search.rounds.r09.explore_fable.tools.confirm_grown
```

## Not done / left for the coordinator

- Closed-loop confirmation of the per-task allocation (PROPOSALS §P2) was **not** run: its evidence is already a paired mixture of
  real closed-loop outcomes; the remaining question (holdout inits 30–49, controller implementation) is the coordinator's.
- GR00T L10-50 and π0.5 Spatial-50 grown artifacts were built and checked offline but not run closed-loop (time budget).
- The deployable grown library must be regrown from B-pool (non-test) rollouts; the artifacts here use test-task inits 0–19.
- No arm on inits 30–49 was run or read.

## Confirmation run outcome (final)

Chain `r09_fable_grown` finished 21:06 CDT (`CHAIN_DONE`, exit 0): A .740 @ IR .0765; grown (6,763 A-arm rows) .700 @ .0763
(−4 pp [−15, +6] vs A); grown (73,945 rows) .730 @ .0764 (−1 pp [−11, +9]); P10 .930 @ .504 (+19 pp [+10, +27]).
Per-task and post-hoc per-task-library mixture: `DATA_ANALYSIS.md` §5.3, `out/confirm_grown.json`. Cleanup verified after
CHAIN_DONE: tmux session `r9f_chain` exited, h100 GPU memory 0 MiB, no `worker_entry` processes of the timan107 island.
