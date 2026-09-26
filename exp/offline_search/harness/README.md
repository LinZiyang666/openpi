# Offline retrieval harness (frozen after R0)

Evaluation harness of the offline retrieval-method exploration (`logs/offline_search_exploration.log.md`).
Coding agents treat `harness/` as **read-only**; report harness bugs to the coordinator.

## ⚠ Read first: only some dimensions are real

| tensor | stored shape | valid part | padding |
|---|---|---|---|
| action chunk (`a_inf`, `a_hit`, `a_exec`, library `action`) | `(H, 32)`, H = 10 π0.5 / 16 GR00T | **steps 0..4** (replan 5 executes only these) × **dims 0..6** (dim 6 = gripper, bimodal at ±1) | π0.5 dims 7..31 ≈ 0.001 constant; **GR00T dims 7..31 are noise with std 0.7–1.0, larger than the real dims (0.13–0.39)** |
| robot_state key `rs` | π0.5 32-d, GR00T 8-d | dims 0..7 | π0.5 dims 8..31 exactly 0 |

Rules:
- Never compute an action distance / similarity / statistic on all 32 dims. On GR00T that measures noise.
  Use `dims.valid_action(a)` (`a[..., :5, :7]`) or `dims.valid_action_chunk(a)` (`a[..., :7]`).
- Never whiten / PCA / z-score / fit a covariance over padded dims: the zero-variance π0.5 rs dims make it singular.
  Use `dims.valid_state(rs, model)` (`rs[..., :8]`).
- A synthesized action only needs dims 0..6 (steps 0..4) to be meaningful; the metrics ignore everything else.
- B0 uses the raw `rs` exactly as the online system did (the zero padding does not change L2), so it reproduces the
  recorded search.

Constants: `harness/dims.py` (`ACT_VALID`, `EXEC_STEPS`, `GRIPPER_DIM`, `HORIZON`, `RS_VALID`, `KEY_DIM`).

## Quick start

```bash
cd /home/weiland/projects/openpi
PY=.venv/bin/python
# smoke-test a method (few episodes of one cell, in-process, API contract checks, metrics vs B0)
$PY -m exp.offline_search.harness.smoke --method exp/offline_search/rounds/r01/f1_whiten/method.py:Whiten \
    --kwargs '{"k": 64}' --cell pi05_spatial_inf --episodes 5 --root /home/weiland/trace_runs/offline_search_store_smoke
# a whole round: many methods x cells concurrently on the machine (or inside a reserved CPU set)
$PY -m exp.offline_search.harness.batch --spec rounds/r01/batch.json --root /dev/shm/offline_search_store \
    --out exp/offline_search/results/r01 [--cpus 26-35,70-79] [--budget N]
# one method sequentially over the cells (workers default = #CPUs in the affinity mask)
$PY -m exp.offline_search.harness.run --method exp/offline_search/rounds/r01/f1_whiten/method.py:Whiten \
    --kwargs '{"k": 64}' --cells all --root /home/weiland/trace_runs/offline_search_store \
    --out exp/offline_search/results/r01
# all reference methods B0..B4 (sequential; for speed put them in a batch spec)
$PY -m exp.offline_search.harness.baselines --root <store> --out exp/offline_search/results/r00
# acceptance gates
$PY -m exp.offline_search.harness.gates --root <store>
# synthetic fixture (same layout, tiny) for offline development
$PY -m exp.offline_search.harness.tests_fixture --out /tmp/os_fixture
```

Store roots: full `/home/weiland/trace_runs/offline_search_store`, smoke `/home/weiland/trace_runs/offline_search_store_smoke`
(3 episodes per cell), RAM copy of the full store `/dev/shm/offline_search_store`. Cells:
`{pi05,groot}_{spatial,l10}_{inf,cache}`. Use `taskset -c <cpus>` / `--cpus` when the machine is shared: every
worker / budget default follows the process affinity mask.

## Method API (`harness/api.py`)

```python
from exp.offline_search.harness import api, dims

class MyMethod(api.Method):
    name = "my_method"      # unique per variant; set self.name in __init__ when kwargs change behavior
    tier = "T1"             # "T0" zero-training | "T1" closed-form stats (seconds) | "T2" trained
    def __init__(self, k=64): self.k = k; self.name = f"my_method_k{k}"
    def fit(self, lib, ctx): ...             # once per cell, main process, before the fork
    def reset(self, episode): ...            # start of every episode
    def query(self, q) -> api.Result: ...    # one decision; episodes are fed in time order
    def bytes_per_entry(self) -> float: ...  # library-side bytes per entry of the representation used
```

`api.Result(topk, scores, confidence, action=None, library="current", extras=None)`

| field | contract |
|---|---|
| `topk` | 1-D **integer** array of library rows (into `library`), best first, length ≥ 1 (first 10 are saved) |
| `scores` | same length as `topk`, no NaN, method's own scale |
| `confidence` | finite float, higher = more willing to accept; drives risk–coverage |
| `action` | optional synthesized `(H, 32)` float action built **only from library actions** (no policy calls); only `[:5, :7]` is scored and must be finite. When given, `err`/`grip_mis` use it instead of `topk[0]`'s action (`phase_err` still uses `topk[0]`). A method declaring `uses_nonlibrary_action = True` may build it from other online data (reference rows only, e.g. the unexecuted tail of the previous chunk); the flag is recorded in the npz / json / scoreboard |
| `library` | any library stored for this model × suite (`"current"`, `"bpool_all"`, `"bpool_cs"`, … = `ctx.library_names`, no registration needed) or a name registered with `ctx.register_library` in `fit()` |
| `extras` | optional `{key: float or ndarray}` per-decision diagnostics, ≤ 4096 bytes per decision (counted as float32); saved as `x_<key>` in the npz |

**fit(lib, ctx)** — `lib` is a `LibraryView` of `library/<m>_<s>/current`. Arrays load lazily as read-only memmaps via
attributes: `key_v0, key_v1` f32[L,32768], `rs` f32[L,Drs], `action` f32[L,H,32], `task_id`, `episode`, `step`,
`ep_len`, `progress`, `success`, `prev`, `next` (−1 = none), plus `ids`, `episodes`, `meta` (manifest: task map),
`L`, `H`, `tasks()`, `rows_of_task(t)` (ascending = library order), `has(name)`, tokens `tok(name)` / `tok_rows` /
`tok_index` / `tok_of(name, row)` (library tok subset). A missing array raises `store.StoreError`; test with
`lib.has(name)`. `ctx` gives `model, suite, arm, cell, lib_key, root, seed, scratch` (a per-cell scratch dir),
`horizon, rs_valid, action_sigma` (σ_d of the err metric), `prof`, `library_names` (stored libraries of this
model × suite), `open_library("bpool_all" | "bpool_cs" | ...)`, `current_params()` (online weights/μ/σ),
`lda_weights()`, `register_library(name, action, progress=None, task_id=None)` (method-built libraries only; stored
names are refused).
Raise `api.SkipCell("reason")` from `fit()` when the method does not apply to a cell.

**reset(episode)** — `EpisodeView`: `uid, task, task_id, init, index, seed, rng`. **No `num_steps`, no `success`.**
Use `episode.rng` (seeded by hash(run seed, uid)) for any randomness.

**query(q)** — `QueryView`, online information only:

| attribute | content |
|---|---|
| `key_v0`, `key_v1` | pooled vision keys f32[32768] (current row) |
| `rs` | robot_state key as the online search saw it (π0.5 32-d, dims 8.. = 0) |
| `raw_state` | f32[8] |
| `task_id`, `step`, `episode`, `model` | step = index of this decision in the episode |
| `hist_key_v0`, `hist_key_v1`, `hist_rs`, `hist_raw_state` | the earlier rows of this episode, `[step, ...]` |
| `hist_a_exec`, `prev_a_exec` | executed chunks of the earlier decisions (only `[:5, :7]` of each was executed) |
| `prev_hit` | was the previous decision served from the library (`True`, HIT) or by the policy (`False`, MISS); `None` at step 0. Derived from the data (that row's executed chunk equals its `a_hit` / `a_inf` bitwise on `[:, :7]`), not from the arm name. Full store: 100 % `True` in cache arms, 100 % `False` in inf arms |
| `hist_hit` | int8 `[step]`: the same flag for every earlier decision (1 HIT, 0 MISS, −1 undecidable) |
| `has_tok`, `tok_v0`, `tok_v1` (f16[256,2048]), `img0`, `img1` (uint8) | tok-subsample rows only (inits 0/10/20/30/40, whole episodes), else `TokensUnavailable`; use `--subsample tok` |

Touching `a_inf`, `a_hit`, `success`, `num_steps`, `rec_*`, `progress`, `row`, ... raises `api.ForbiddenAccess`
(the backing object does not even hold those arrays). All arrays are read-only memmap slices: copy before writing.

Rules:
- **No state across `reset()`** and deterministic given `episode.seed`/`episode.rng`: smoke re-runs the episodes in
  reversed order after a fresh fit and requires identical outputs. Workers see episodes in arbitrary chunks.
- Heavy work belongs in `fit()` (runs once, before the fork; its arrays are shared copy-on-write by the workers).
  Precompute per-task candidate blocks, normalize once, keep float32 contiguous arrays. Never copy library matrices
  per query.
- `bytes_per_entry()`: bytes per library entry of the retrieval representation (keys, codes, projections, extra
  per-entry fields). The action payload shared by all methods is excluded. B0 = 2·32768·4 + Drs·4.
- `uses_gt = True` is reserved for reference baselines (B3); it is recorded in the scoreboard.
- `uses_nonlibrary_action = True` (default False) declares that `Result.action` is not built from library rows;
  recorded in the npz (`uses_nonlibrary_action`), `<cell>.json` and the scoreboard column of the same name.
- Profiling: wrap stages in `with self.prof.section("key_build"): ...` (`self.prof` is injected; a no-op unless
  profiling is on).
- Method-built libraries: build in `fit()` from stored library actions (e.g. from `ctx.open_library("bpool_all")`),
  register with `ctx.register_library(name, action, progress=..., task_id=...)`, return `Result(library=name)`.

### Layout of later-round methods

```
exp/offline_search/rounds/rNN/fK_<family>/method.py     # one or more Method classes
exp/offline_search/rounds/rNN/fK_<family>/*.py          # helpers (import as siblings: the file's dir is on sys.path)
exp/offline_search/rounds/rNN/fK_<family>/smoke_*.txt   # smoke outputs handed back
```
Refer to a method as `exp/offline_search/rounds/rNN/fK_<family>/method.py:ClassName` (file path form; no
`__init__.py` needed) or as a dotted module if you add `__init__.py` files. The family label defaults to the
`fK_<family>` directory name (override with a `family` class attribute or `--family`). Variants: one class + `--kwargs`,
with `self.name` encoding the kwargs so every variant gets its own output directory.

## Metrics (`harness/metrics.py`) — GT = `a_inf` (full_inference) for BOTH arms

Per decision, on the executed valid block `[:5, :7]`:
- `err = sqrt(mean_{t<5,d<7}(((â − a*)/σ_d)²))`, σ_d = std (ddof 0) of `library/<m>_<s>/current` `action[:, :5, d]`.
- `grip_mis = mean_{t<5} [(â[t,6] ≥ 0) ≠ (a*[t,6] ≥ 0)]`. Dim 6 verified bimodal at ±1 in both models (library
  `|g| < 0.5` in 0.00–0.01 % of values; gate G0 re-checks it).
- `phase_err = |step/(num_steps−1) − progress[top1]|` (metric side may use num_steps; NaN if the library has no progress).
- `oracle_err` = min err over current-library candidates of the same task (`oracle_row` = argmin); `regret = err − oracle_err`.
- `bin = min(2, 3·step // num_steps)`: early / mid / late thirds.
- Floor (`floor/<m>_<s>/`, optional): step0_pairs (inf vs cache sample at step 0 → early) + resample (every fresh
  teacher sample vs the recorded one, by third). Per bin median / p90 (bins with < 20 samples fall back to the pooled
  floor). `indist = err ≤ floor median of the bin`, `err_over_floor = err / that median`, `bad = err > floor p90`.
  No floor → these columns are NaN. Note: the GR00T teacher floor is tiny (median ≈ 0.02–0.06 vs π0.5 ≈ 0.13–0.15),
  so `indist` is ≈ 0 for every retrieval method on GR00T; compare `err_over_floor` there.

Per cell: `err_mean/median/p90`, per-bin err and indist, `grip_mis`, `phase_err_mean/median`, `oracle_err_mean`,
`regret_mean/median`, `indist`, `err_over_floor_median/mean`; risk–coverage: sort by confidence descending (stable,
ties by row order), `risk(c)` = mean err of the accepted top-c fraction, `aurc` = mean of risk over c = 1/N..1,
`aurc_opt` (sorted by err itself), `eaurc = aurc − aurc_opt`, `risk_c30/50/70/90`, `bad_c30/50/70/90/100` and
`bad_aurc`; `flip_rate` = top-1 ≠ recorded online top-1 (decisions on `current` only), `frac_current`, `synth_frac`.
`summary.json` per method adds inf − cache deltas per model × suite.

## Runner (`harness/run.py`)

```
python -m exp.offline_search.harness.run --method <module_or_file>:<Class> [--kwargs JSON] --cells all|<list>
    --root <store> --out <results/rNN> [--workers <#CPUs in mask>] [--subsample tok] [--seed 0] [--profile]
    [--round rNN] [--family NAME] [--scoreboard PATH|default|none] [--episodes-limit K] [--no-timing] [--allow-gpu-fit]
    [--job] [--timing-only] [--pin-core K]
```
Per cell: fresh instance → `fit` in the main process (timed) → fork `--workers` processes (default and cap = number
of CPUs in the process affinity mask, 88 on this machine) over contiguous, decision-balanced episode chunks (4 per
worker) → gather in episode order → metrics → 300-query single-thread timing pass in the main process (a warm-up
pass, then a timed pass with the profiler on; fixed seeded episode set per cell).
The oracle (method-independent) is computed once per cell and cached in `<out>/_cache/oracle_<cell>_<fp>.npz`
(fingerprint = store paths + file sizes/mtimes + σ; per-cell flock so concurrent jobs compute it once; runs covering
< 50 % of a cell compute their rows directly). Same arithmetic, so outputs do not change.
Batch modes (used by `batch.py`): `--job` = no DONE/ERROR, run_meta, summary or scoreboard writes, the per-cell
outcome goes to `_jobs/<cell>.json`; `--timing-only` = re-fit + timing pass only, merged into the existing
`<cell>.json` (atomic replace; `timing.timing_source = "timing_only"`, `timing_fit_s`, `pinned_core`), outcome in
`_jobs/<cell>.timing.json`; `--pin-core K` pins the process to logical CPU K before numpy loads.
Workers: `CUDA_VISIBLE_DEVICES=""`, OMP/MKL/OPENBLAS threads = 1 (set before numpy loads). `--allow-gpu-fit` keeps
CUDA visible in the main process for T2 training; the fitted model must run on CPU in `query`.
Determinism: same inputs → bit-identical npz arrays (except `t_query_us`), independent of the worker count (gate G4).

## Batch runner (`harness/batch.py`) — how full rounds are run

```
python -m exp.offline_search.harness.batch --spec <batch.json> --root <store> --out <results/rNN>
    [--budget <#CPUs in mask>] [--cpus 26-35,70-79] [--max-job-workers budget//2] [--round rNN]
    [--scoreboard default|none|PATH] [--timing-concurrency 40] [--no-timing-phase] [--no-numa] [--tag NAME]
```
`batch.json` is a list of `{"method": "<module_or_file>:<Class>", "kwargs": {}, "family": "", "cells": "all" | [..],
"subsample": null | "tok", "allow_gpu_fit": false, "seed": 0, "profile": false}` (only `method` required; variants =
several entries with different kwargs and distinct `name`s).
1. **Accuracy phase.** One subprocess per (method, cell): `run.py --job --no-timing --workers W`. W ∝ the cell's
   decision count (`round(max_job_workers · n / n_max)`, min 4, max `max_job_workers` = budget/2 by default: 44 for
   the l10 cache cells, ~12–16 for spatial). Biggest jobs start first; first-fit packing keeps Σ W ≤ budget, per
   NUMA node (each job is pinned to one node's CPUs, so its fitted arrays are node-local and its forked workers
   inherit the mask; `--no-numa` turns this off). `allow_gpu_fit` jobs: at most one at a time. Fits, workers and
   metrics of different jobs overlap. A failed job never stops the others; no retries.
2. **Timing phase** (skip with `--no-timing-phase`). For every successful job: `run.py --timing-only`, ≤
   `--timing-concurrency` at once, each pinned to its own physical core (one logical CPU per core, inside the mask).
   Timing / profile fields are merged into `<cell>.json`. The accuracy-job `fit_s` stays the scoreboard's `fit_s`.
   ⚠ Concurrent timing processes share memory bandwidth: at 40 at once the bandwidth-bound kNN baselines measured
   2–7× slower than a lone process (B0 3.8 → 7.9 ms/query, B4_v0 0.9 → 6.3 ms; B2/B3/B4_rs unaffected).
   `ms_per_query` is comparable only between runs with the same `timing.timing_concurrency` (recorded per cell);
   use `--timing-concurrency 1` for absolute single-process numbers.
3. **Finalize.** Per method: `summary.json`, `run_meta.json` (mode "batch", per-cell W and job walls), `DONE` / `ERROR`
   (same format as run.py; ERROR lists failed cells with phase and traceback). Scoreboard: one row per successful
   (method, cell), appended once, after the timing phase (so rows carry `ms_per_query`; NaN with
   `--no-timing-phase`). Markers and rows appear only at the end of the batch.
Batch files: `<out>/_batch/<tag>.log`, `<tag>.progress.jsonl` (batch/phase/job start + end: wall, W, node, core, rc),
`<tag>.DONE` / `<tag>.ERROR` (JSON: walls per phase, failed jobs), per-job logs `<out>/_batch/<tag>/{acc,timing}/`.
`--cpus` pins the batch and all its jobs to a CPU list (NUMA nodes and timing cores are chosen inside it); the budget
defaults to the size of the mask. Outputs are bit-identical to sequential `run.py` runs (verified on all 56
baseline npz files).

### Output files (`<out>/<method.name>/`)

`<cell>.npz` — per decision, in episode order:

| key | dtype | meaning |
|---|---|---|
| `row` | int64 | row in `queries/<cell>/` arrays |
| `ep`, `step`, `task_id` | int32, int16, int32 | episode index (episodes.json), step, task |
| `top1`, `topk`, `topk_scores` | int32, int32[N,10] (−1 pad), f32[N,10] (NaN pad) | the method's rows / scores |
| `confidence` | f64 | |
| `err`, `grip_mis`, `phase_err`, `oracle_err`, `oracle_row`, `regret` | f64 / f32 / int32 | metrics above |
| `bin` | int8 | 0 early / 1 mid / 2 late |
| `used_synth` | bool | the method returned an action |
| `synth_seg` | f32[N,5,7] | only when any action was returned (NaN rows otherwise) |
| `lib_code`, `lib_names`, `library` | int8, str[], 0-d str | library per decision; `library` = the single name or "mixed" |
| `flip` | int8 | 1 top1 ≠ recorded top-1, 0 same, −1 not on `current` |
| `t_query_us` | f32 | wall time of `query()` in the worker (µs, perf_counter_ns) |
| `uses_nonlibrary_action` | 0-d bool | the method's declaration |
| `x_<key>` | f32 | extras: same-shape values stacked `[N, *shape]` (NaN where missing); ragged values flattened and NaN-padded to the max length, plus `x_<key>__len` int32 (−1 missing) |

`<cell>.json` — `metrics`, `rc_curve` (20 points), `floor` (source, per-bin median/p90, counts), `sigma`, `timing`
(`ms_per_query` mean / p50 / p95 / cold, `timing_source` in_run | timing_only, `fit_s`, `worker_wall_s`, workers,
chunks, `phases` = fit / workers / gather / metrics / timing / save seconds of the job's main process),
`data_checks.exec_hit_counts` (HIT / MISS / both / neither over all rows and over the rows `prev_hit` reports),
`uses_nonlibrary_action`, `fit` (`fit_s`,
`rss_mb_after_fit`, `peak_rss_mb_after_fit`, `bytes_per_entry`, `L`, library sizes, `cands_per_task` min/mean/max),
`profile` (`timing_pass` per section n/total/mean/p50/p95/per-query µs; `workers` the same over all decisions with
`--profile`; `fit` sections; `query_total_us`), `extras_keys`, kwargs, tier, family.

`progress.jsonl` — one JSON line per event: `run_start`, `cell_start`, `fit_done`, `worker` (every ~10 s per worker:
episodes done/total, decisions done, elapsed, decisions/s, RSS MB), `chunk_done` (per chunk: episodes, decisions,
seconds, decisions/s, RSS), `cell_done`, `skip`, `error` (with traceback), `run_done`.
`run_meta.json`, `summary.json`, `scratch/<cell>/` (ctx.scratch), and finally `DONE` or `ERROR` (failed cells +
tracebacks). Monitor waits on `DONE`/`ERROR`. JSON files may contain `NaN` literals (Python's `json` reads them).

Scoreboard: one row per cell appended (flock) to `--scoreboard` (default `<out>/../scoreboard.csv`, i.e.
`exp/offline_search/results/scoreboard.csv`) with the fixed columns of `metrics.SCOREBOARD_COLUMNS` (round, run_ts,
method, family, tier, uses_gt, uses_nonlibrary_action, cell, …, bytes_per_entry, ms_per_query, fit_s, peak_rss_mb,
L, out). Re-runs append; take the latest `run_ts` per (method, cell). When the column set grows, the first append
rewrites the file under the lock with the new header (old rows keep their values, new columns empty).

## Reference methods (`harness/baselines.py`)

`B0Current` (online formula, reproduces `rec_top1`), `B1LDA` (LDA weights, fusion_ablation manifest; all four
model × suite present), `B2Random` (uniform within task), `B3Oracle` (argmin err, uses GT), `B4V0` / `B4V1` / `B4RS`
(single field; confidence = raw similarity of top-1). B0/B1 extras: `cos_v0, cos_v1, dist_rs, n_v0, n_v1, n_rs,
margin` of the top-1. B0 arithmetic: float32 prenormalized dot / L2, `0.5·(tanh((x−μ)/σ)+1)`, fields summed in the
order v0, v1, rs as the torch backend; ties → lowest library row.

## Gates (`harness/gates.py`)

G0 gripper bimodality + task map consistency + executed chunk == a_hit on every cache-arm row / == a_inf on every
inf-arm row · G1 B0 reproduces `rec_top1` (≥ 99.9 %, every disagreement with fused gap
< 1e-4, |score − rec_score| < 1e-4) · G2 B3 ≤ B0 on every decision · G3 B2 mean err ≥ 1.2 × B0 in every cell · G4
B0 and B2 bit-identical across worker counts · G5 `library/*/current` equals the pkl export
(`trace_dual/audit/libs`) within 1e-6. Output: `exp/offline_search/results/gates/<root basename>/gates.json`.

## Smoke (`harness/smoke.py`)

Checks, each PASS/FAIL/WARN: attrs; fit + query on `--episodes` evenly spaced episodes with every Result validated
(`ForbiddenAccess` / `TokensUnavailable` / `ContractError` reported with the offending field, cell, uid, step, row);
`bytes_per_entry`; extras; leak (fresh fit, reversed episode order, identical outputs); static scan of the class source
for forbidden names (WARN); `prev_hit` consistent with the arm on the selected episodes; WARN when
`uses_nonlibrary_action` is declared. Prints the metrics and the timing-pass profile, plus B0 on the same episodes. Exit 0 iff no FAIL.
