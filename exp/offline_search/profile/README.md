# offline_search profile tools

Diagnostics so that no round's run is a black box: why a method picks what it picks, where it wins or
loses, whether the library or the search is the bottleneck, and where time and memory go. CPU only
(the package pins `OMP/MKL/OPENBLAS_NUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=""` before numpy loads).

The tools use the whole machine by default (owner, 2026-09-26). `--procs` defaults to `os.cpu_count()`
and can be lowered; worker processes are forked, single-threaded and never see CUDA. The parallel units
are:
- `coverage`: cell x library x task x 1024-query block;
- `breakdown`, `timeline`, `runprof summary`: method x cell;
- `compare`: cells, then dAURC bootstrap replicates in chunks;
- `explain`: chunks of the selected decisions, for batches of 16 or more.

Outputs are identical to a serial run. A test checks this, and on the full store the new coverage cache and
breakdown equal the old serial ones exactly. `repr` still runs one job per (model x suite, representation).

All metric definitions come from the frozen harness: `harness/dims.py` (only `[:5, :7]` of an action chunk,
`rs[:8]`), `harness/metrics.py` (err, sigma_d from library current, floor thresholds per step bin, phase,
AURC with a stable sort). `common.py` delegates to them and only falls back to identical local copies when the
harness cannot be imported. On the fixture and on the smoke store, the tools reproduce the runner's
`<cell>.json` metrics exactly and the harness oracle bit for bit (see Tests).

The tools read the stores and run dirs and never write to them. The only writes go to
`<store root>/profile_cache/` (the coverage cache plus `reports/<tool>/<tag>/`), or to `--out DIR`.

| tool | answers | main outputs |
|---|---|---|
| `coverage` | What could ANY search reach with this library? Where is coverage missing? | per cell / task / step-third: oracle err, number of "good" candidates (err <= floor median), any-good rate, rank structure, where the recorded B0 pick ranks. Cache `profile_cache/coverage_<lib>.npz` |
| `explain` | Why did the method pick this row, and which B0 field misled it? | per decision: top-k (library episode, step, progress, score, err), the oracle and its rank, gap, extras, coverage context, raw B0 per-field sims (top-1 vs oracle vs recorded pick) |
| `breakdown` | Where does the method win or lose? | per cell, by task, step-third, outcome, gripper transition, confidence decile, candidate count, #good candidates, plus a pooled inf-vs-cache table |
| `compare` | Is A really better than B, and where? | paired d = err_A - err_B, win/tie/loss, episode-bootstrap 95% CI of the mean d, of dAURC and of d indist, top-1 flip rate, per-slice d, the most divergent decisions (as ready-made `explain` commands) |
| `timeline` | Does the method track a library trajectory or jump around? Does the oracle? | per-episode timeline (move flags T/A/=/B/S), per cell switch / track / advance / stay rates, run length, err after each move type, the same rates for the oracle |
| `repr` | Is this representation able to tell good candidates apart? | saturation, spread, task separation, PCA effective dimension, hubness, Spearman(sim, -err), AUROC for good vs bad candidates, the representation's own 1-NN err, the oracle's rank under it |
| `runprof` | Where do time and memory go? How does latency grow with L? | per-cell fit / wall / throughput / ms per query / t_query percentiles / RSS; per-section us (Profiler); worker balance and stragglers; `scaling` (latency vs library size); `live` (one-screen status of an in-flight run) |

Store root: `--root` defaults to `/dev/shm/offline_search_store`, the hot copy pinned in RAM (its `tok/`
and `profile_cache/` are symlinks to the SSD store). When that directory is absent the default falls back to
`/home/weiland/trace_runs/offline_search_store`. Smoke store: `/home/weiland/trace_runs/offline_search_store_smoke`.
When one call reports several methods, `--out DIR` gets one subdirectory per method. Run arguments accept either a method dir
(`results/rNN/<method>`, which holds `<cell>.npz`) or a round dir plus `--method NAME[,NAME]`.
`--cells` takes `all`, cell names, `model_suite` prefixes (`pi05_l10`) or an arm (`inf`).

## coverage (library-side ceiling)

```
python -m exp.offline_search.profile.coverage [--root <store>] [--lib auto|current,bpool_all] [--cells all] [--procs NCPU]
       [--topn 50] [--floor-mode third|overall] [--thr X] [--force] [--brief]
```
For every query of every cell it computes the err to every task-filtered candidate of the library. Candidates
are screened with a GEMM, then the top 58 are recomputed exactly with the harness arithmetic. The per-query
cache (`<cell>__<field>` keys) holds these fields:
`top_rows/top_errs [N,50]`, `n_cand`, `n_good` (err <= thr), `n_good2x`, `oracle_err` (== harness oracle_err),
`second_err`, `med_err`, `thr`, `rec_err`, `rec_rank` (the recorded B0 pick, library current only). The cache is
fingerprinted (file size and mtime of the queries, the library and the floor) and reused unless something
changed or `--force` is given.

```
| cell | n | cand | thr | orc_mean | orc_p50 | orc_p90 | any_good | frac_good | ngood_p50 | any_2x | e@5 | e@10 | med | B0rec | B0regret | B0rank_p50 | B0indist |
| pi05_l10_cache | 258 | 268 | 0.097 | 0.305 | 0.265 | 0.541 | 0.035 | 0.000 | 0 | 0.244 | 0.438 | 0.525 | 1.201 | 0.714 | 0.409 | 20 | 0.012 |
```
Files: `reports/coverage/<lib>/coverage_<lib>.json`, `coverage_<lib>_cells.csv`, `coverage_<lib>_slices.csv`
(task, third and task x third).

## explain (decision inspector)

```
python -m exp.offline_search.profile.explain RUN [--method M] --cell pi05_l10_inf
       (--episode UID|IDX ... | --row R ... | --worst N | --best N | --random N) [--sort err|regret] [--k 10]
```
The harness `oracle_row` always refers to library current. For mixed-library runs, every decision is shown
against its own library (`rc.lib_of(pos)`). The B0 block gives the raw cosine on the pooled v0 and v1 keys,
the rs L2 over the valid dims, the normalized scores `0.5(tanh(z)+1)` with the trace_dual mu and sigma, and
the fused score. It then states which weighted field pushed the top-1 above the oracle, or that B0 itself
would prefer the oracle. Files: `reports/explain/<method>__<cell>/explain.{json,txt}`.

## breakdown (slices)

```
python -m exp.offline_search.profile.breakdown RUN [--method M] [--cells all] [--slices task,third,outcome,grip,conf_dec,cand,good] [--tie-aware]
```
Per slice: n, share, err mean and p50, indist (harness floor median of the decision's step bin), regret,
orc_hit (top-1 == oracle), grip, phase, conf, AURC. The `all` row carries risk@30/50/70/90.
Success and num_steps are metric-side labels only. The run's `oracle_err` is cross-checked against
`coverage_current`, and a note is printed if they disagree. Files: `reports/breakdown/<method>/breakdown.{json,csv,md}`.

## compare (paired A vs B)

```
python -m exp.offline_search.profile.compare RUN_A RUN_B [--method-a A] [--method-b B] [--cells all] [--reps 2000] [--seed 0] [--tie 1e-6]
```
```
| cell | n | err_A | err_B | d mean [95% CI] | win/tie/loss | AURC A/B | dAURC [95% CI] | d indist [95% CI] | top1 flip |
| pi05_l10_inf | 147 | 0.460 | 0.232 | +0.2276 [+0.1952,+0.2929] | 0.00/0.14/0.86 | 0.376/0.138 | +0.2384 [+0.2196,+0.2578] | -0.054 [-0.078,-0.040] | 0.864 |
```
The bootstrap resamples whole episodes (fixed seed). A flip means the top-1 `(library, row)` differs.
Files: `reports/compare/<A>__vs__<B>/compare.{json,md}`, `compare_cells.csv`, `compare_slices.csv`.

## timeline (trajectory behaviour)

```
python -m exp.offline_search.profile.timeline RUN [--method M] [--cells all] [--episode UID|IDX ...] [--worst-episodes N] [--random-episodes N] [--no-csv]
```
Move flags relative to the previous decision of the same episode: `T` means the library `next` of the
previous choice, `A` a later step of the same library episode, `=` the same row, `B` an earlier step, and `S`
another library episode (a library change counts as `S`). Files: `timeline.json`, `timeline_summary.csv` and
`timeline.csv` (every decision).

## repr (representation diagnostics)

```
python -m exp.offline_search.profile.repr [--ms all] [--reprs key_v0,key_v1,rs,tokmean_v0,tokmean_v1] [--lib current]
       [--encode pkg.mod:fn | file.py:fn [--metric cos|l2|dot] [--name X]] [--n-queries 2000] [--k 10] [--lib-max 8000]
```
User encoders: `fn(view) -> ndarray[len(view), D]`, where `view.get(field)` returns the rows of the library or
of a query cell, and also exposes `view.kind`, `view.task_id`, `view.step`, `view.lib` (for library-only
statistics) and `view.idx`. Encoders must not read GT fields on the query side. An object with `.encode`
(plus optional `.lib_rows` and `.query_rows`) also works. Examples are in `encoders.py`
(`task_centered_v0/v1`, `concat_centered`). `rs` drops the pi0.5 padding dims before any statistic. On GR00T
the floor is tight (about 0.02 to 0.06), so `AUROC good` is often NaN; `AUROC top10%` (good = the query's best
10% of candidates) is always defined. Files: `reports/repr/<lib>__<reprs>/repr.{json,md}`,
`repr_library.csv`, `repr_query.csv`.

## runprof (time and memory)

```
python -m exp.offline_search.profile.runprof [summary] RUN [--method M] [--cells all]
python -m exp.offline_search.profile.runprof scaling --method <module_or_file>:<Class> [--kwargs JSON] --cell C [--scales 0.5,1,2,4] [--n-queries 300]
python -m exp.offline_search.profile.runprof live RUN [--method M] [--stale 60]
```
The summary reads `<cell>.json` (timing, fit, profile.timing_pass, profile.workers), the `t_query_us` field of
`<cell>.npz`, and the `chunk_done` / `worker` / `error` events in `progress.jsonl`.

`scaling` rescales library current: whole episodes are subsampled per task, or every row is duplicated with the
episode, prev and next fields remapped. The library is loaded into memory before an untimed warm-up fit, so
`fit_s` excludes store IO. Queries are then timed single-threaded on the runner's fixed timing set. Example
(B0, smoke store, pi05 spatial): L 427/1018/2036/4072 gives 0.68/2.04/5.87/11.5 ms per query, i.e.
ms ~ L^1.28, and 98% of the time goes to `sim_v0`/`sim_v1`.

`live` is one-shot and never polls. Files: `reports/runprof/<method>/runprof.{json,md}`, `runprof_cells.csv`,
`runprof_sections.csv`, and `scaling__<method>__<cell>/scaling.{json,csv}`.

## Wall time on the full store (`/dev/shm`, 88 procs, 2026-09-26)

| tool | serial (before) | parallel (now) | outputs |
|---|---|---|---|
| `coverage --force` (current + bpool_all, 8 cells) | 11.2 s | 6.4 s (454 jobs, 220 CPU-s of work in a 3.9 s pool) | cache arrays bit-equal (176/176) |
| `breakdown` over r00 (7 methods x 8 cells) | 12.0 s | 2.5 s (1.0 s in-process) | identical |
| `timeline` over r00 with the 1.3M-row CSV | - | 7.3 s (33.6 s when the CSV was assembled in the parent) | CSV byte-identical |
| `compare` (8 cells, 2000 reps) / `runprof summary` / `explain --worst 200` | - | 2.7 s / 0.5 s / 0.7 s | identical to `--procs 1` |

## Tests

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider exp/offline_search/profile/tests --basetemp <scratch>
```
`fixture.py` builds a small synthetic store that follows the real layout, including GR00T noise padding,
pi0.5 zero rs padding and both `current` and `bpool_all`. It then produces run dirs with the real
`harness/run.py`: B0, B2 and a test method that mixes libraries. The tests check coverage against brute force
and against the harness oracle bit for bit, breakdown against the runner json, explain B0 fields against the
B0 extras, bootstrap identities, the timeline move codes, a perfect-encoder check for `repr`, and
ScaledLibrary pointer remapping.
