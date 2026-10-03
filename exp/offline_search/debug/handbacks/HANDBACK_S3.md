# S3 handback — R8 reader, validation, capacity, deferred augmentation

Implemented the S3 assignment against `osdebug.v1`. All execution was local and CPU-only, pinned to
22–25,66–69 with CUDA hidden and numerical-library threads limited to one. No remote processes, simulators,
servers, git state changes, or files outside S3 ownership were changed. Historical trace data were read only.

## Files

- `debug/reader.py`, `debug/validate.py`, `debug/capacity.py`, `debug/fixtures.py`
- `debug/aug/{__init__.py,models.py,retrieval.py,job.py,validate.py}`
- `debug/tests/{test_reader.py,test_validate.py,test_aug.py,test_aug_models.py}`
- This handback. All paths above are relative to `exp/offline_search/`.

## Delivered behavior

`reader.open_arm` implements every shared reader method. Journal outcomes determine accepted attempts; retries
remain accessible with `accepted_only=False`. Decisions outer-join server/client records by `decision_id`, reject
duplicate or conflicting identities, and expose missing joins. Requested array order and duplicate requests are
preserved. Controls concatenate ragged contact offsets and preserve variable actuator-substep counts by padding
the in-memory view with NaN. Parquet caches under `debug/derived` include source fingerprints and round-trip
nested diagnostics. Array files always use `allow_pickle=False`.

The read-only P3 adapter consumes raw server records, input archives, CSV tables, journals, and client telemetry.
It exposes numeric robot/object/predicate/substep arrays plus original legacy records. It marks legacy joins and
geom naming as unverified, does not invent wire images, and never creates derived files in legacy trees.
`pair` joins accepted episodes by `(task_id, init)`. `catalog` accepts an explicit catalog reference, an explicit
manifest/arm-spec cell, or the model/suite/library layout; missing catalogs return `None`.

Validation reports separate capture and augmentation PASS/INCOMPLETE with exact issue codes, paths and details.
The production default is exactly 500 accepted pairs; smoke overrides must be explicit. Checks include identity,
contiguity, control/decision assignment, issued-action equality to wire chunks, bidirectional joins, image shapes,
block bounds, pickle exclusion, physics dtypes, contact offsets/catalog bounds, sampling, snapshots, statuses,
complete receipts and actual file bytes/SHA, augmentation coverage, duplicate IDs, provenance and policy shapes.
Complete receipts and individual file receipts reconcile by content rather than unrelated receipt metadata.
Truncated ZIP files become INCOMPLETE rather than an uncaught archive exception.

Capacity reports source bytes excluding derived caches and unpublished `.part` files, compressed/uncompressed
bytes per field, per-field and per-episode p50/p95/max/mean, shared/unaccepted overhead, writer/receiver/sender
stats, and remaining-episode scenarios. Shared decision blocks are apportioned by episode membership.
`forecast_campaign` pools measured model/suite cells against emitted arm specs; unmeasured cells stay unavailable.
The budget ledger recomputes N/V/M and owner/control IR from live dispatch counts, excluding settling and deferred
work. It reports declared, measured and R4-assumed prices separately and flags logged-cost disagreement. Measured
prices: pi05 .152/.848, GR00T .148/.852, pi05 wrist .0646/completion .0502; R4 partial-camera prices .055198/.049890.
GR00T partial-camera and third-only live prices are unavailable rather than guessed.

Augmentation loads a model once for an ordered arm list. Pi05 reuses policy transforms, `Pi05StageBatcher`,
production stage IO and CP1 key building. GR00T uses the serving policy/data configuration, staged runner,
key builder and production shape buckets; its stages 1/2 are serial and compatible stage-3 inputs are batched.
Both use explicit private per-row torch generators. Draw seeds are SHA-derived from
`campaign|task_id|init|decision_seq|draw`; three sampled extra draws reuse stage-2 conditioning.

Frozen SHA-checked A fits compute full retrieval using factual prefix actions/history. Pi05 camera shadows use
frozen R7 wrist fits and a third-camera metric fitted with the same library-only A recipe and the third-camera
PCA basis. Pure-policy arms require both current and bpool_cs/pi05 or bpool_all/GR00T fits. Extra-library arrays
use library suffixes, e.g. `cache_chunk_bpool_cs`; the first configured library also has canonical fields.
Fit pickles are trusted, SHA-checked coordinator inputs, never captured data files.

Publication is atomic and append-only under `aug/`, protected by a per-arm nonblocking flock. Completed IDs are
skipped; incompatible checkpoint/code/fit-config provenance and duplicate or incomplete published parts fail
explicitly. A resumed partial episode may re-encode its prefix to rebuild factual retrieval history, but completed
policy outputs are not recomputed. Augmentation and agreement/benchmark reads disable derived-cache writes.

## CPU commands and verification

From `/home/weiland/projects/openpi`, use this prefix for each CPU Python command:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python
```

Full S3 suite executed:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest \
  exp/offline_search/debug/tests/test_reader.py exp/offline_search/debug/tests/test_validate.py \
  exp/offline_search/debug/tests/test_aug.py exp/offline_search/debug/tests/test_aug_models.py \
  --basetemp=/tmp/r8_S3/pytest_final --override-ini cache_dir=/tmp/r8_S3/pytest_cache -q
```

Result: **37 passed in 33.33s**, one pre-existing torch `pynvml` deprecation warning. Final reader/capacity
follow-up (manifest-only/nullable suite and explicit catalog cell): **14 passed in 8.99s**. Python-3.8 AST syntax
check passed for all nine implementation files. Tests cover both model fakes, partial/full resume, private NumPy
and torch RNG isolation, actual CPU stage/key/bucketing components, frozen A/wrist/third metric fitting, camera
separation, SHA rejection, corrupt capture files, identities, receipts, ragged controls and budget/storage math.

Real read-only P3 smoke on `r6p3v2_groot_sp_50_A_r0`: **20 accepted episodes, 459 decisions**, all legacy joins
explicitly unverified. First episode has 90 controls and qpos `(90,48)`; qvel/eef/object/predicate/actuator data
and normalized/wire/policy archives were also exercised. Logged legacy environment seed is retained.

The shared fixture is `/tmp/r8_S3/cli_fixture/runs/synthetic`; it has six accepted episodes, 54 decisions and
282 controls. Validator CLI returned capture PASS and augmentation PASS with no missing items. Capacity and
budget CLIs exited zero and wrote `/tmp/r8_S3/cli_{capacity,budget}.json`. Fixture images are 16x16, so their
storage or throughput is not a real-campaign estimate.

S4 provenance CLI and S6 physical selfcheck CLI both exited zero on that fixture, using `--procs 1`.
Outputs: `/tmp/r8_S3/decision_provenance` and `/tmp/r8_S3/physical_selfcheck`.

Operational CPU invocations (append to the prefix; set actual run/arm names):

```bash
-m exp.offline_search.debug.validate --run-root "$R8_RUN" --arms "$R8_ARM" --capture-only --expected-pairs 20
-m exp.offline_search.debug.validate --run-root "$R8_RUN" --arms "$R8_ARM" --expected-pairs 20
-m exp.offline_search.debug.capacity --run-root "$R8_RUN" --arms "$R8_ARM" --budget --out /tmp/r8_S3/budget.json
-m exp.offline_search.debug.capacity --run-root "$R8_RUN" --arms "$R8_ARM" --remaining-arms 70 --out /tmp/r8_S3/capacity.json
-m exp.offline_search.debug.capacity --run-root "$R8_RUN" --arms "$R8_PI_ARM" "$R8_GROOT_ARM" \
  --forecast-specs "$R8_RUN/arms.json" --out /tmp/r8_S3/campaign_forecast.json
```

For the full campaign omit `--expected-pairs`, keeping the 500-pair default. Forecast from representative smoke
arms in every model/suite cell; the two-arm example above remains unavailable if other cells lack measurements.

## Frozen fit configuration

`--fit-config` is a JSON mapping from actual arm name to these specifications (paths and SHA values must be
replaced with verified frozen artifacts):

```json
{
  "r8_pi05_l10_50_A_smoke": {
    "store_root": "/home/weiland/trace_runs/offline_search_store",
    "cell": "pi05_l10_cache",
    "full": [{"name": "current", "path": "<frozen-A.pkl>", "sha256": "<content-SHA256>"}],
    "wrist": [{"name": "current", "path": "<frozen-R7-wrist.pkl>", "sha256": "<content-SHA256>"}]
  }
}
```

For 500-library arms use the matching bpool fit/library name. For P10 include both libraries in `full` and both
pi05 libraries in `wrist`. GR00T needs only `full`. Optional `third` uses the same spec structure; otherwise the
same-library third metric is computed once in memory. Existing PCA files are read only; current-library PCA
falls back to deterministic computation in memory. `store_root` and `cell` are required for automatic third fit.
The config content SHA and each fit SHA enter augmentation provenance. The loader unwraps supported wrapper
attributes `base`, `awm`, or `method` to a fitted A retrieval object.

## Coordinator GPU validation commands — not executed by S3

Run locally on the coordinator's allocated GPU; no server or remote worker is needed. Set `R8_RUN` to the smoke
run, `R8_PI_ARM`/`R8_GROOT_ARM` to captured smoke arms, `R8_FITS_JSON` to the config above, and
`R8_PI_SHA`/`R8_GROOT_SHA` to the authoritative checkpoint content hashes. Checkpoint hashes are required inputs,
not directory-name hashes. Example emitted arm names: `r8_pi05_l10_50_A_smoke` and `r8_groot_spatial_50_A_smoke`.

Pi05 throughput at batches 1/8/32:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m exp.offline_search.debug.aug.job --run-root "$R8_RUN" --arms "$R8_PI_ARM" \
  --model pi05 --checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --checkpoint-sha "$R8_PI_SHA" --device cuda:0 --benchmark --batch-sizes 1 8 32 --limit 96
```

Pi05 all shadows (supply additional same-model arms after the first in priority order; no limit for completeness):

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m exp.offline_search.debug.aug.job --run-root "$R8_RUN" --arms "$R8_PI_ARM" \
  --model pi05 --checkpoint /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --checkpoint-sha "$R8_PI_SHA" --device cuda:0 --batch-size 8 --fit-config "$R8_FITS_JSON"
```

GR00T Spatial throughput and all shadows (for L10 substitute the actual L10 checkpoint and its SHA):

```bash
R8_GROOT_REPO=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
R8_REPO=/home/weiland/projects/openpi
R8_GROOT_PY=/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python
R8_GROOT_PATH="$R8_GROOT_REPO:$R8_GROOT_REPO/examples/Libero:$R8_REPO:$R8_REPO/src:$R8_REPO/packages/openpi-client/src"
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R8_GROOT_PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$R8_GROOT_PY" -m exp.offline_search.debug.aug.job \
  --run-root "$R8_RUN" --arms "$R8_GROOT_ARM" --model groot --checkpoint /data/ckpt/n15_libero_spatial \
  --checkpoint-sha "$R8_GROOT_SHA" --device cuda:0 --benchmark --batch-sizes 1 8 32 --limit 96
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R8_GROOT_PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$R8_GROOT_PY" -m exp.offline_search.debug.aug.job \
  --run-root "$R8_RUN" --arms "$R8_GROOT_ARM" --model groot --checkpoint /data/ckpt/n15_libero_spatial \
  --checkpoint-sha "$R8_GROOT_SHA" --device cuda:0 --batch-size 8 --fit-config "$R8_FITS_JSON"
```

Agreement after augmentation, CPU only:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.debug.aug.validate --run-root "$R8_RUN" --arms "$R8_PI_ARM" "$R8_GROOT_ARM"
```

Inspect `aug/benchmark.json`, `aug/job_stats.json`, and `aug/agreement.json`. Agreement compares normalized live
policy chunks on actual MISS decisions with shadows using declared valid action channels, alongside independent
policy–policy draw MSE. It reports finite MISS/noise sample coverage, MSE ratio and excess MSE; absent live chunks
or noise draws yield unavailable. The two populations can differ, so this is a descriptive check, not a numerical
parity certificate. Assess sampling noise and batch numerical differences before admission. Repeat augmentation
with identical checkpoint/code/fit config to verify zero newly written IDs; batch size can change on resume.
Limit simultaneous Python processes to three. No remote validation command is required for S3.

## Remaining validation and integration notes

- Real checkpoint loading, GPU memory/throughput at 1/8/32, and live MISS agreement remain coordinator checks.
  CPU stage-component tests do not claim real-model GPU parity.
- Real storage capacity and receiver throughput must be measured on smoke captures in all four model/suite
  cells. The fixture's compressed images cannot establish the campaign's 1.6 TB admission limit.
- P3 wire images are absent; contact naming is deliberately uncertified. P3 data cannot satisfy R8 completeness
  or deferred wire-image augmentation, and snapshots are not restore-certified.
- Supply catalog paths/cells and frozen fit specs when available; absent library catalogs remain `None`. No
  outside-ownership edits were needed. No live capture or policy behavior was modified by S3.

## Fix round 1

Read `rounds/r08/REVIEW_1.md` completely and implemented S3's M5/m9 fixes plus the coordinator's shared layout
and init semantics. The statements below supersede the earlier cell-name catalog fallback and whole-config/code
resume hashing descriptions. Coordinator edits outside S3 ownership were preserved.

Added `debug/catalog.py` and `debug/tests/test_reader_catalog.py`; the other modified files are the existing S3
reader, validator, capacity, fixtures, augmentation job/retrieval, and S3 regression tests. No capture-path file was
edited. All execution used CPUs 22–25,66–69, CUDA hidden, numerical-library threads one, at most three Python
processes. No GPU, server, remote process, port, git state or protected trace data was changed.

The catalog builder writes `<RUN>/catalog/<model>_<suite>_<lib>/rows.parquet` and `rows.json`. It reads the store's
row arrays without fitting a model, hashes all twelve core library arrays, and includes exact frozen R7
`mode/event_near/rows_to_event/stage_run` labels when available. It supports standalone StageTable envelopes and
StageTables embedded in SF1/CT/wrapper fits. Stage content fingerprints and row identities must match the requested
library before publication. Missing frozen stages are explicit in metadata, with no fabricated labels.
Catalog and metadata publication use temporary files, fsync and replace. Suite aliases `libero_10` and
`libero_spatial` resolve to the store's `l10` and `spatial` keys. The reader never falls back to a serving cell;
even explicit references must match the model/suite/library key. Current and big catalogs cannot be interchanged.

Coordinator catalog command (writes into the selected run; S3 ran only the scratch-root variant):

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.debug.catalog --run-root "$R8_RUN" \
  --cells pi05_l10_current pi05_l10_bpool_cs pi05_spatial_current pi05_spatial_bpool_cs \
          groot_l10_current groot_l10_bpool_all groot_spatial_current groot_spatial_bpool_all
```

Defaults are the read-only offline-search store and `r07_main/fits`. Override with `--store-root`, `--r7-fits`, or
repeated `--stage-fit CELL=PATH` for trusted frozen artifacts. S3's real-library build succeeded at
`/tmp/r8_S3/catalog_real/catalog/pi05_l10_current`: **2,640 rows**, frozen R7 labels available from
`r07_main/fits/stages_pi05_l10_50.pkl`, with verified fingerprint and source-array/Parquet hashes. It was not copied
into the read-only smoke run.

Reader, validation and capacity now consume all `decisions*.jsonl`, `meta*.json` and `writer_stats*.json`, including
legacy single-file names. Metadata is matched to the record's process; an existing reader discovers a later
process's metadata, and unrelated legacy metadata cannot conceal a missing new-process file. Source fingerprints
include every process log. A final JSONL line without newline is skipped and reported with path, line and byte
count in `read_issues`; interior malformed lines remain errors. Cached reader results retain these diagnostics.
Skipped tail bytes remain accounted for in shared storage overhead. Completeness still depends on accepted
records, so a harmless extra tail diagnostic alone does not invalidate otherwise complete accepted episodes.

Subset `init` and `task_id` are derived from `task_uid`; `orig_init_state_idx` remains separate. Legacy single-file
records preserve their original logged init field as the original index, and existing explicit original indices
are retained. Validation checks both identities against the accepted client episode. Private augmentation seeds
cast all numeric components with `int()`, eliminating pandas float-format differences.

NPZ reads inflate only requested fields plus required identity/prompt/image-mask dependencies. Asking for state,
prompts or settling flags does not inflate images, keys or chunks. Image reads reject unavailable requested rows;
validation independently reports false/malformed `img_*_available` masks and top-level server `status="error"`.
Fixtures and wire-action validation now require float64 `served_wire`, with a regression value that demonstrably
loses precision in float32 and must match the issued float64 controls exactly.

Resume code SHA covers only augmentation-producing `job.py/models.py/retrieval.py`, excluding shared serving,
capture and reporting code. The existing `fit_config_sha` metadata field now identifies only the fit specs used
by that kind: policy shadows/draws have no fit dependency; full look uses full fits; camera look uses wrist/third
fits and full-fit/store/recipe inputs only when third fitting actually needs them. Other arms, unused fit types,
notes and whole-campaign config changes do not block resume. Changed consumed specs/checkpoints/augmentation code
still do. The old `run_arm(..., fit_config_sha=...)` argument is accepted but no longer drives the resume key.
GR00T's default fit loading excludes unused camera fits.

Regression suite command:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest \
  exp/offline_search/debug/tests/test_reader.py exp/offline_search/debug/tests/test_reader_catalog.py \
  exp/offline_search/debug/tests/test_validate.py exp/offline_search/debug/tests/test_aug.py \
  exp/offline_search/debug/tests/test_aug_models.py --basetemp=/tmp/r8_S3/pytest_fix1_release \
  --override-ini cache_dir=/tmp/r8_S3/pytest_cache -q
```

Result: **53 passed in 47.95s**, one pre-existing torch `pynvml` deprecation warning. Python-3.8 AST syntax checks
passed for all ten implementation files. Regression
coverage includes every fix above, exact current/big separation, matching and mismatched frozen stage sources,
selective member reads, truncated-tail diagnostics/cache persistence, process metadata/stats, init identity,
float seeds, used-fit resume dependencies, image masks and float64 wire precision.

Real read-only audit of the plain `r8_pi05_l10_50_A_smoke` directory (not the capfail directories):

- Reader: **20 accepted episodes, 1,380 decisions, all 1,380 joins verified**. Subset init matches every UID;
  original init matches every client episode. No skipped tails. The live run currently has no library catalog.
- Validation with `expected_pairs=20, require_aug=False`: **capture PASS**, 7,067 controls, zero missing items.
- Full validation with `expected_pairs=20`: **INCOMPLETE** only for deferred augmentation: 1,380 policy shadows,
  55 sampled three-draw records, 1,380 full shadow looks and 1,380 camera shadows absent.
- File sizes/mtime inventories before and after the reader/validator audit were identical. Full report:
  `/tmp/r8_S3/fix1_smoke_audit.json`.
- Capacity: source 199,476,319 bytes; per-episode mean 9.971 MB, p50 9.093 MB, p95 15.105 MB, max 15.389 MB
  (decimal MB; this one model/suite does not establish a full-campaign forecast).
- Live budget: **N/V/M = 1380/693/0**, declared owner IR **0.0763304348**, no reconciliation warnings.
  Reports: `/tmp/r8_S3/fix1_smoke_capacity.json` and `/tmp/r8_S3/fix1_smoke_budget.json`.

Repeat capture validation using the earlier CPU prefix plus:

```bash
-m exp.offline_search.debug.validate --run-root /home/weiland/trace_runs/os_closed_loop/r08_smoke \
  --arms r8_pi05_l10_50_A_smoke --expected-pairs 20 --capture-only --out /tmp/r8_S3/smoke_capture.json
```

GPU throughput and live MISS agreement remain coordinator checks using the commands above. No new GPU/remote
validation is required for these CPU reader/catalog fixes; populate catalogs in the writable campaign run and
run deferred augmentation before treating the smoke arm as augmentation-complete.

## Performance round

Implemented 2026-09-30 on CPUs **32–33,76–77** only. No GPU, server, remote process, git operation or live-run
write was performed. Edited `aug/models.py`, `aug/job.py`, `tests/test_aug_models.py`; added
`aug/cpu_fake.py`, `aug/profile_cpu.py`, `aug/compare.py`, `tests/test_aug_performance.py`. Earlier reader,
validation, capacity, catalog and capture fixes are retained.

### Execution changes and invariants

GR00T now batches eager stage 1 and the production `run_stage2_llm` across compatible decisions. Stage 1 is a
local transcription of the pinned production embedding/vision/scatter operations: transforms and
`prepare_input` remain per observation, exact-shaped Eagle inputs are concatenated without tokenizer padding,
and image-token counts are checked independently per row. The runner still verifies upstream sources.
Compiled vision buffers are rejected. Stage-1 and stage-2 grouping reuse `GrootStageBatcher.bucket_key` through
conditioning proxies; stage 1 additionally checks Eagle tensor shapes/dtypes/devices and action-input fields.
Stage 3 keeps the existing production shape/embodiment/schedule buckets. `--stage-mode serial` selects the
production B=1 stage-1/stage-2 reference; stage 3 still batches compatible inputs as before.

Encoder outputs serve all requested kinds. Sampled draws select existing stage-2 conditioning instead of
re-encoding their observations. Pi05 uses the production KV stack/split helpers with independently cloned
shards. GR00T caches host conditioning identities across draws and uses the production `_noise_dtype` cache,
replacing the old extra head prologue per row per draw. CP1 keys use exactly the production slicers and pooling,
but transfer three stacked fields to CPU instead of three fields per row. Seeds and private row generators are
unchanged; outputs remain keyed by decision_id, and no completed policy chunk is redrawn.

A bounded decoder/transform thread prepares the next batch; a separate single ordered worker performs CPU
retrieval and atomic publication while the model thread advances. There is at most one future input batch and
one output batch. Factual retrieval history is constructed in episode order on the main thread, using the same
captured actions/camera modes as before. Only CPU keys reach retrieval. Worker exceptions propagate, executors
join, the arm lock releases, and any finished parts remain resumable. `--no-prefetch` gives a sequential path.
Batching currently stops at episode boundaries; small episodes and isolated sampled draws have smaller batches.

### Recorded code upgrades and scratch outputs

Resume still requires matching model, checkpoint SHA, used-fit SHA and dtype. An old code SHA fails by default.
With **`--allow-code-upgrade`**, every retained old-code part is validated, hashed and listed in an atomic,
append-only `aug/provenance/code_upgrade_<fingerprint>.json` note. The note names the explicit flag, old/new
code SHAs, kept part paths, part content SHAs, row counts and each kind's provenance. New parts reference the
note and carry the new SHA. Old parts and their metadata are never rewritten. The same upgrade produces one
idempotent note. This flag is required again for subsequent resumes while old-code parts remain; checkpoint or
used-fit changes cannot be authorized by it.

`--output-root DIR` directs augmentation to `DIR/<arm>` while reading the original capture with derived caches
disabled. This permits comparisons and benchmarks without changing `/home/weiland/trace_runs`. Default
publication remains the arm's `debug/aug`. `--benchmark-all --output-root DIR` runs all requested kinds at each
`--batch-sizes` value, including decode, transforms, retrieval and atomic part I/O. It warms one policy decision
outside the timed runs, writes `DIR/<arm>/benchmark_bN/` plus `benchmark_all.json`, and refuses to reuse any
benchmark directory that already contains parts. The older `--benchmark` is still policy-only and excludes I/O.
Camera fits are materialized before benchmark-all warmup/timing so the first batch-size trial does not pay an
unfair one-time fitting cost.

### Profile and measured CPU breakdown

`--profile` records CPU wall intervals for decode, transforms, factual-history I/O, key building, CPU retrieval,
publication per kind, and model stages. CUDA events measure stage-1/stage-2/key-building/stage-3 stream elapsed
time, with one synchronization at job end. `stage3.policy_shadow` and `stage3.policy_draws` separate action work;
`kind.*` includes draw assembly and transfers and nests stage-3 timing. Shared encoding is counted once.
Pipeline wall intervals overlap and must not be summed as if exclusive; GPU event elapsed includes stream
contention/idle gaps and is not a kernel-only profiler. Initialization/fit loading are outside timed throughput;
camera-fit setup inside a job has its own timer.

CPU profiles (private output roots, warmup excluded):

- `/tmp/r8_S3/performance_profile_v2/profile.json`: both models on 96 synthetic decisions. GR00T fake executes
  the real production stage-1/stage-2 implementations with toy modules; pi05 uses the deterministic CPU fake.
- `/tmp/r8_S3/performance_smoke_profile_v2/profile.json`: 96 actual observations from read-only
  `r8_groot_l10_P10_smoke`, using the GR00T CPU fake and both fake retrieval libraries. One decision was sampled
  for three extra draws. Real NPZ decode and atomic output I/O are included; model/transform/retrieval compute
  remains fake. Profiles were measured while other CPU work was active and are descriptive, not GPU estimates.

Real-smoke fake, B32 **without prefetch**, exclusive kind runs, milliseconds per run:

| Kind / emitted rows | Decode | Transform | S1 | Keys | S2 | S3 | Retrieval | Publish | History I/O | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy_shadow / 96 | 1265.53 | <.01 | 249.71 | 37.50 | 9.32 | 1.81 | 0 | 100.74 | 0 | 1.8771 |
| policy_draws / 1, three draws | 23.41 | <.01 | 1.67 | .45 | .34 | .88 | 0 | 22.34 | 0 | .0658 |
| shadow_look / 96 | 1162.32 | <.01 | 247.42 | 35.40 | 0 | 0 | 1.12 | 86.40 | 111.59 | 1.6846 |
| camera_shadow | — | — | — | — | — | — | — | — | — | unsupported on GR00T |

Pi05 fake on 96 synthetic decisions, B32 without prefetch:

| Kind / emitted rows | Decode ms | Fake encoder ms | Fake action ms | Retrieval ms | Publish ms | History ms | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy_shadow / 96 | 33.39 | 9.82 | 8.33 | 0 | 156.55 | 0 | .2599 |
| policy_draws / 4, three draws each | 9.24 | .82 | 1.65 | 0 | 54.69 | 0 | .1137 |
| shadow_look / 96 | 38.80 | 11.30 | 0 | 1.71 | 197.98 | 18.58 | .3280 |
| camera_shadow / 96, wrist + third | 41.15 | 12.47 | 0 | 3.06 | 201.01 | 20.82 | .3375 |

Pi05's fake does not implement separate transform/LLM stages, so those numbers are unavailable, rather than
invented. The real adapter measures them separately with `--profile`. Differences between totals and listed
intervals include seeds, input SHA, conditioning/noise assembly, array transfers and Python bookkeeping.

All kinds on the 96 real GR00T observations: serial B1/no-prefetch **8.3194 s**; batched B32/no-prefetch
**1.8738 s**; batched B32/prefetch **1.6695 s** (4.98× relative to B1, 1.12× relative to B32 sequential, on this
CPU fake). Stage-1 and stage-2 dispatches dropped **96 → 4**, processing the same 96 rows with no sampled-row
re-encoding. Decode wall work was 2.4003 s at B1, 1.0893 s at B32 sequential; history reads 1.0524 → .1145 s;
publication across all kinds 4.3408 → .2114 s. Real-sample decoding is material even when model work is fake.
Synthetic all-kind totals: GR00T 3.8972 → .6699 s sequential B32; pi05 5.9071 → .8732 s. These comparisons include
fewer fsync publications at larger batches, not just model batching; short synthetic episodes limit overlap.

Production cost reasoning: policy_shadow needs one S1, one LLM S2, and one denoising S3 per observation; sampled
policy_draws add three S3s per sampled row, with zero additional encoder/LLM work in an all-kind job.
shadow_look needs CP1 S1 keys plus CPU PCA/metric/top-k/kernel/action mixing, with zero LLM/action work.
pi05 camera_shadow reuses those same full CP1 keys and performs wrist/third CPU retrieval; no camera encoder
passes are added. Full/current and large-library retrieval both remain required on P10. The real transform chain
is eval-mode center crop/resize/normalization/tokenization; it is prepared per row on one CPU thread. Real key
slicing/pooling and device transfers remain separately measured.

The supplied serving B1 times (S1 ~45 ms, S23 ~256 ms) do not separate LLM from action cost and cannot be summed
into augmentation B32 latency. Serial vision alone imposes ~45 ms/row before its shared keys; serial LLM is an
additional batch-resistant term in the previous adapter. Both are now batched, while sparse extra action draws,
retrieval and decode/I/O remain. At measured old augmentation 6.2 rows/s, 1.75M decisions would take ~78.4 GPU
hours if all had GR00T's rate; the campaign contains both models. **No real GPU speedup or numeric tolerance pass
is claimed here.** Measure B8/B16/B32 memory and throughput with the coordinator commands below.

Reproduce the CPU profile using a fresh output directory:

```bash
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.debug.aug.profile_cpu --out /tmp/r8_S3/profile_repeat --rows 96 \
  --run-root /home/weiland/trace_runs/os_closed_loop/r08_smoke \
  --arm r8_groot_l10_P10_smoke --model groot
```

### Coordinator GPU validation commands (not run by S3)

From `/home/weiland/projects/openpi`, initialize paths and reconstruct the exact two frozen fit specs from the
old smoke provenance. This reads old output and writes only scratch:

```bash
R8_PERF_RUN=/home/weiland/trace_runs/os_closed_loop/r08_smoke
R8_PERF_ARM=r8_groot_l10_P10_smoke
R8_PERF_OLD="$R8_PERF_RUN/runs/$R8_PERF_ARM/debug/aug"
R8_PERF_G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
R8_PERF_PY=/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python
R8_PERF_PATH="$R8_PERF_G:$R8_PERF_G/examples/Libero:/home/weiland/projects/openpi:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src"
R8_PERF_SHA=4133afb655297ce9cdcfe24d1c1af09de847e4f3b0d72b3a773c87624e1c90a1
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python - <<'PY'
import json
from pathlib import Path
arm = 'r8_groot_l10_P10_smoke'
old = Path('/home/weiland/trace_runs/os_closed_loop/r08_smoke/runs') / arm / 'debug/aug'
p = json.loads((old / 'job_stats.json').read_text())['provenance']
spec = {arm: {'full': [dict(p['fits']['full/' + name], name=name) for name in p['libraries']]}}
Path('/tmp/r8_S3').mkdir(parents=True, exist_ok=True)
Path('/tmp/r8_S3/performance_groot_fits.json').write_text(json.dumps(spec, indent=2) + '\n')
PY
```

First compare a B8 candidate against the **actual historical outputs**, using the same 96-observation prefix,
checkpoint hash and private seeds. Scratch directories must be fresh:

```bash
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R8_PERF_PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$R8_PERF_PY" -m exp.offline_search.debug.aug.job \
  --run-root "$R8_PERF_RUN" --arms "$R8_PERF_ARM" --model groot --checkpoint /data/ckpt/n15_libero_10 \
  --checkpoint-sha "$R8_PERF_SHA" --batch-size 8 --limit 96 --profile \
  --fit-config /tmp/r8_S3/performance_groot_fits.json --output-root /tmp/r8_S3/gpu_candidate_b8
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
  -m exp.offline_search.debug.aug.compare --reference-aug "$R8_PERF_OLD" \
  --candidate-aug "/tmp/r8_S3/gpu_candidate_b8/$R8_PERF_ARM" --limit 96 --atol .03 --rtol .02 \
  --out /tmp/r8_S3/gpu_old_vs_new_b8.json
```

Compare all kinds at B1/B8/B16/B32 and measure end-to-end throughput on 256 decisions:

```bash
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R8_PERF_PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$R8_PERF_PY" -m exp.offline_search.debug.aug.job \
  --run-root "$R8_PERF_RUN" --arms "$R8_PERF_ARM" --model groot --checkpoint /data/ckpt/n15_libero_10 \
  --checkpoint-sha "$R8_PERF_SHA" --benchmark-all --batch-sizes 1 8 16 32 --limit 256 \
  --fit-config /tmp/r8_S3/performance_groot_fits.json --output-root /tmp/r8_S3/gpu_throughput
for R8_PERF_B in 8 16 32; do
  taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python \
    -m exp.offline_search.debug.aug.compare \
    --reference-aug "/tmp/r8_S3/gpu_throughput/$R8_PERF_ARM/benchmark_b1" \
    --candidate-aug "/tmp/r8_S3/gpu_throughput/$R8_PERF_ARM/benchmark_b$R8_PERF_B" \
    --atol .03 --rtol .02 --out "/tmp/r8_S3/gpu_b1_vs_b$R8_PERF_B.json"
done
```

The comparison requires exact checkpoint/used-fit/dtype provenance, decision coverage, private seeds, input SHAs
and library strings; joins by ID independently of part or row order. Every float output (including cache chunks,
keys, weights and scores) must pass the explicit tolerances. Neighbour row changes are counted separately, since
near ties can switch under batch numerics; inspect those counts alongside action deltas. `.03/.02` are explicit
screening thresholds, not a measured tolerance bound; reports include max/mean absolute errors. Reports failing
coverage or comparisons exit nonzero. The historical comparison intentionally compares different code SHAs.

For a stage-1/stage-2 serial reference throughput comparison, repeat the benchmark-all command with
`--stage-mode serial --no-prefetch --batch-sizes 32 --output-root /tmp/r8_S3/gpu_serial_reference`.
This serial reference has the new reuse/noise-dtype optimizations; the actual old `job_stats.json` is the historical
throughput baseline (1091 rows / 149.974 s = 7.2746 rows/s at B16, a different contention/time window from the
coordinator's later 6.2 rows/s measurement). Repeating fresh output roots under comparable live-server load is
necessary for a defensible GPU speedup claim. B32 model-stage batching has not been tested for available VRAM.

For pi05, use the same benchmark-all/comparison commands with `.venv/bin/python`, `PYTHONPATH=.:src`,
`--model pi05`, checkpoint `/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`, the
authoritative pi05 SHA, its smoke arm and existing full/wrist/third fit-config. Include `camera_shadow` (default
for pi05); the pi05 changes reuse conditioning, stack key transfers and overlap CPU work rather than introducing
a new encoder/LLM batching implementation.

Production continuation after coordinator numeric validation: invoke the normal augmentation command on the
writable campaign run with `--allow-code-upgrade --batch-size <validated-size> --profile`. Keep the existing arm
priority list and frozen fit config. This is the explicit authorization to retain old parts under a recorded note;
omit `--output-root` for production continuation.

### Tests and remaining validation

Release suite: **64 passed in 56.08 s**, one pre-existing torch/pynvml warning. This includes all 53 prior S3 tests,
the new pi05 KV subset test and ten performance tests. Regression coverage: production GR00T stage-1/stage-2
serial/batched bitwise equivalence; prompt/embodiment shape buckets; per-row image-token guards; private Torch
RNG; all-kind bitwise output parity across serial B1 and prefetched B8 for both models; encoder/LLM reuse;
retrieval/model overlap; worker-failure resume/lock release; code-upgrade byte/hash preservation and strict
checkpoint/used-fit checks; ID-based numeric/seed comparison; all-kind benchmark I/O and fresh-output guard.
Python-3.8 AST syntax checks passed for all eight augmentation implementation modules.
After the final conditioning-identity cache change, the model/performance subset passed again: **13 passed in
22.38 s**. The full reader/validate audit on the real GR00T smoke arm returned **20 accepted episodes, 1091
decisions, all 1091 joins verified; capture PASS and full augmentation PASS**, no missing items or truncated-tail
diagnostics. Its file-size/mtime inventory was identical before and after the audit. Report:
`/tmp/r8_S3/performance_smoke_audit.json`. This validates historical capture/aug completeness, not new GPU results.
The final benchmark camera-fit warmup adjustment passed its CLI regression separately: **1 passed in 3.39 s**;
the final eight-module Python-3.8 syntax check also passed.

```bash
taskset -c 32-33,76-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest \
  exp/offline_search/debug/tests/test_reader.py exp/offline_search/debug/tests/test_reader_catalog.py \
  exp/offline_search/debug/tests/test_validate.py exp/offline_search/debug/tests/test_aug.py \
  exp/offline_search/debug/tests/test_aug_models.py exp/offline_search/debug/tests/test_aug_performance.py \
  --basetemp=/tmp/r8_S3/pytest_performance_repeat --override-ini cache_dir=/tmp/r8_S3/pytest_cache -q
```

Real checkpoint loading, batch numeric agreement, VRAM admission and GPU throughput remain coordinator checks.
CPU profiles establish dispatch reduction, reuse and pipeline correctness; they cannot establish those GPU gates.
