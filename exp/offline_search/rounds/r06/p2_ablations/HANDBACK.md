# P2 paper ablations: identity metric and B trigger leave-one-out

Delivered 8 metric arms and 16 trigger arms with 24 fresh prefits (1,106,853,913 bytes total). The metric check has
zero ordered top-16 mismatches on 23,522 recorded queries. Same-history B audits have zero mismatches over 188,844
decisions / 95,151 vision anchors. New all-off equals original B-off exactly on 108,237 recorded decisions; it
equals stable-tie A exactly, while deployed A differs at one pre-existing GR00T l10/50 tied anchor and its tail.

Repository: `/home/weiland/projects/openpi`. Implementation and evidence files are confined to
`exp/offline_search/rounds/r06/p2_ablations/`; generated fits, replay logs and emitted validation configs are under
`/home/weiland/trace_runs/os_closed_loop/r06_abl/`. No shared implementation was edited. No git commands,
servers, LIBERO workers, experiment chains, tmux sessions or remote hosts were used. All Python commands used
CPUs `26-29,70-73`, the repo `.venv/bin/python`, OMP/OpenBLAS/MKL threads 1 and an empty CUDA_VISIBLE_DEVICES.
The plugin itself records its ordinary read-only HEAD metadata. No `tests/review_tests/` access.

## Files and switches

- `arms_metric.json`: 8 arms, models pi05/GR00T × l10/spatial × 50/500. Method
  `exp.offline_search.rounds.r06.p2_ablations.metric:IdentityBlindAWM`.
- `arms_trigger_loo.json`: 16 arms, models pi05/GR00T × l10/spatial × four disabled guards, all 50-library.
  Methods `...p2_ablations.judge:TriggerCommitJudge` (pi05) and `...:TriggerGrootCommitJudge` (GR00T).
  Added kwarg `disabled_guard`: `stuck`, `terminal`, `overtime`, or `no_progress`.
  Test/control values `none` and `all` are also supported; they are not extra paper arms.
- `make_arms.py`, `results/provenance.json`: exact source rows and arm derivation.
- `prefit.py`, `run_jobs.py`: fresh prefit and bounded CPU subprocess runners.
- `metric_check.py`, `replay_audit.py`, `allvision_check.py`, `nesting_check.py`, `logic_check.py`, `validate.py`,
  `summarize.py`: executable checks and aggregation.
- `results/summary.json`, `results/TABLES.md`: final replay numbers and checks.
- `results/validation.json`, `results/fit_sha256.txt`: every final prefit's size, SHA256, library rows, class and metadata.
- `results/initial_float32/`: preserved initial metric-check failures and original prefit command record.
  Superseded float32 artifacts are outside the final fit directory, at `r06_abl/initial_float32/`.

The two arm specs are emit_arms input lists. Only name, method, fit-artifact pathname, and the trigger switch differ
from their authoritative A/B source rows. All other fields (including their absence), kwargs, plugin-arg ordering,
full_model, cost_ledger, client settings, kref and libraries are copied exactly. The pi05 spatial/500 A source's
absent `cost_ledger` is intentionally preserved. GR00T B explicitly retains `--os-policy-tail-blocks 1`, resize 256,
and replan_steps 5; pi05 B retains its implicit one-block setting. Artifact paths are `<RUN>/fits/<arm>.pkl`.
`validate.py` restores the changed fields and asserts whole-row equality with the original spec.

## Metric ablation: exact identity, not a large-lambda approximation

`codes=0` means full Cholesky metric; `codes>0` selects generalized-eigen directions after metric whitening.
Neither means identity. Finite `lam` retains the action-similarity covariance in
`(S_w + lam tr(S_w)/d I)^-1`; `lam=inf` is not a usable exact identity setting. No large lambda is used.

`identity_metric` returns the stock float64 mean, `std + 1e-6`, and `eye(136)` for BOTH normal and early fits.
AWM.fit has no injectable metric hook, so the subclass binds its unchanged function bytecode to a private globals
dictionary containing the replacement fitter. It never assigns to the earlier module or monkeypatches a shared
function. The local fitter captures the features/statistics for the explicit Euclidean evaluator; no evaluation
query is used in fitting. Stock PCA, action scales, pseudo-query calibration, blind statistics and synthesis run
unchanged. Every delivered prefit was freshly generated using the plugin prefit entry point.

Representation and branches:

- Same PCA-64 per camera, plus the same eight valid state features (136 total), with the same current/big PCA bases.
- Per-task main normalization uses all that task's fit-library rows. Early normalization uses step <= 2 rows of that
  task. Step 0 uses the early metric against ALL candidate rows; subsequent pure-cache anchors use the main metric.
- Euclidean `norm(Z_candidates - (x_query - mean)/std)` in float64. Identity is exact; standard deviation's existing
  `+1e-6` is retained. The query-side PCA projection remains stock float32.
- Same top-16 selection routine, kernel, kref 5/8, full-chunk synthesis, anchor_tail, budget 1, budget_only gates.
  No policy calls. Post-MISS continuity code is preserved but unreachable under these A specs.

Numerical issue found and resolved: simply returning I from the metric fitter while keeping the stock float32
quadratic/early-affine distance arithmetic produced reordered near-tied neighbors against a direct float64
Euclidean reference. The initial check failed in five cells; the per-query evidence is preserved in
`results/initial_float32/`. The final evaluator uses direct differences in float64, while retaining the same early
fit/step-0 decision branch. This is an explicit numerical implementation difference from stock AWM, needed to make
the requested Euclidean ranking exact on the tested queries. It retains two additional 136-d float64 candidate-code
arrays (2,176 bytes/entry beyond stock); this is a behavior ablation, not a memory/latency claim. Stock distance work
is still performed by the inherited `_dist` before replacement; no search-latency benefit is claimed.

The independent test re-reads the library PCA projections and raw valid state, recomputes each task's main/early
mean and std, constructs direct float64 Euclidean distances, and compares ORDERED top-16 row IDs. It also checks
PCA/baseline array equality and exact kernel synthesis. Query selection is all 500 episode starts plus 1,000 evenly
spaced rows in EACH of the recorded cache and inf cells (duplicates removed), for each of the eight fits. The inf
stream is used as recorded observations with A's no-MISS regime; no recorded policy history enters retrieval.

## Trigger semantics

The mask runs AFTER the complete original CommitJudge/GrootCommitJudge query (including P1's GR00T terminal-sign
fix). It clears the selected os_flags bit and re-derives lowest-bit os_reason, os_force_miss, and `_s['flag'][-1]`.
Everything that produced the flags is still computed. Non-verdict diagnostics and fitted thresholds are untouched.

| disabled_guard | bit removed | retained computation |
|---|---:|---|
| stuck | 1 | vision-confirmed stuck counter, V7 stuck feature, overtime's `stuck_n >= 1` prerequisite |
| terminal | 2 | terminal-row diagnostic, executed gripper and model-specific closed sign |
| overtime | 4 | overtime, lag and stuck diagnostics |
| no_progress | 8 | progress history and span; only no-progress MISS and its blind LOOK veto are disabled |

Thus `stuck` does NOT set the counter to zero or change its threshold. An observation with flags 1|4 becomes 4 and
still MISSes. Disabling a lower-priority bit can change the logged reason while retaining MISS; only a singleton
firing set can lose its MISS verdict. This is what "equals B except where the disabled guard was the only firing
reason" means for the decision: payload/confidence are always unchanged on equal histories, diagnostic flag/reason
fields necessarily record the remaining firing set.

Disabling no-progress also bypasses K1's `LookReason(8, 'noprog_span')` blind veto; progress computation is retained.
This is the complete leave-one-out of that guard, and makes `all` use the same blind gate as A. Lifecycle,
anchor-tail budget, stuck/terminal/overtime logic, and the committed policy tail remain inherited. The explicit
blind-veto tests cover a positive span even where normal committed B replays do not encounter a separate veto.
Only deployed B settings are accepted: guards enabled before masking, events none, burst 0, monitor off,
policy_tail_gate lifecycle, vision-confirmed stuck, noprog_span, no memo reset, anchor_tail/budget1/budget_only.

## Replay proof and its limits

`replay_audit.py` uses P1's real installed-plugin replay driver and deployed B artifact as the reference. Each
freshly fitted LOO artifact plus none/all controls receives the SAME online QueryView at each reference B vision
anchor. Separate per-episode state is maintained; cache blind and policy-tail calls are checked too. B supplies
served action/HIT/vision histories, so the proof isolates the removed trigger instead of conflating it with later
history divergence. Every recorded cache and inf episode is replayed (500 per stream × four 50-library cells).
Assertions cover exact top-k, scores, action, confidence, every non-verdict extra, unchanged stuck/progress state,
correct flag memo, independently recomputed B predicates, and removed MISS iff only the disabled guard fired.
The table reports direct removals on that fixed B history. It is NOT an estimate of removal totals in independent
LOO rollouts: changed cache/policy histories can affect later selection and guards.

`allvision_check.py` additionally queries 30 evenly spaced recorded episodes from each cache/inf stream per cell,
using recorded execution histories. This exercises nontrivial all-vision stuck/terminal/overtime/no-progress logic
and the GR00T terminal correction. The regular B schedule mostly uses this branch at step 0 only.

`nesting_check.py` runs each new all-off path through the installed plugin on all 500 cache-stream episodes for
each 50-library cell. It compares original B with guards=False, deployed A, and P1's test-only stable-tie A control.
The test loads the fresh `stuck` artifact, sets its runtime disabled mask to 15, and adds comparison hooks; its
startup metadata therefore still names that artifact. This avoids fitting four redundant all-off artifacts.
The known pre-existing AWM argpartition vs G3 stable-top-k tie discrepancy is audited explicitly; exact deployed-A
nesting must not be claimed where the table reports tied anchors/tails. No deployed tie rule was changed.

The 24 separate plugin smoke replays use EACH actual deliverable artifact and spec on three evenly spaced cache
stream episodes (including first/last). Their own histories evolve according to that variant. The final aggregator
checks cache rows 5..9, each nonfinal MISS's single policy tail, no metric MISS, stage-1/vision count agreement,
and at most one blind decision in a row.

All replay observations are fixed recordings. No simulated action-dependent state evolution, success-rate result,
GPU latency, hardware control or actual policy inference was measured. Policy MISSes in the CPU replay receive the
recorded `a_inf` chunk. No closed-loop experiment was launched; that remains the coordinator's work.

## Commands and reproducibility

Run from the repository root. Every Python invocation, including children of the runners, uses this prefix:

```bash
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
D=exp/offline_search/rounds/r06/p2_ablations
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.make_arms
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.prefit
```

`prefit.py` defaults to `/home/weiland/trace_runs/os_closed_loop/r06_abl`, refuses existing artifacts unless
`--resume`, and permits at most seven children plus its coordinator. For a full fresh reproduction use
`--run /absolute/fresh/run/root`; the supplied specs' `<RUN>` must resolve to that root. Existing final prefits
should be loaded rather than refitted. Hashes identify delivered bytes; refitting changes recorded fit timing.
Initial full prefit commands are in `results/initial_float32/prefit_commands.json`; final metric-refit commands are
in `results/prefit_commands.json`. The initial metric artifacts were moved aside, then the final eight were fitted
with `prefit --parallel 2 --resume` (log `results/prefit_metric_final.log`).

The executed test matrices are fully specified by the `args` arrays in `results/audit_jobs.json` and
`results/verification_jobs.json`:

```bash
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.run_jobs "$D/results/audit_jobs.json" --parallel 4 --logs /home/weiland/trace_runs/os_closed_loop/r06_abl/test_logs/audit
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.run_jobs "$D/results/verification_jobs.json" --parallel 2 --logs /home/weiland/trace_runs/os_closed_loop/r06_abl/test_logs/verification
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.logic_check
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.validate
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.summarize
```

The two matrix runners were concurrent: 4+2 children and two coordinators = eight Python processes maximum.
Other Python commands ran only after enough slots were free. On rerun choose fresh replay `--out` directories
in the job files and a fresh validation output directory; replay/validation refuse existing directories. Result
JSONs and text logs persist in the new directory and `r06_abl/test_logs/`; raw replay decisions and normalized
served chunks persist in `r06_abl/replay/`. Tests are read-only toward the source fits and store.

A minimal CPU smoke recipe (fresh output directory; no server or worker):

```bash
"${P[@]}" -m exp.offline_search.rounds.r06.p1_groot_commit.replay \
  --method exp.offline_search.rounds.r06.p2_ablations.judge:TriggerGrootCommitJudge \
  --kwargs '{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off","disabled_guard":"stuck"}' \
  --cell groot_l10_cache --replay-cell groot_l10_cache --episodes 3 \
  --fit-artifact /home/weiland/trace_runs/os_closed_loop/r06_abl/fits/r6p2_stuck_g_l10_50.pkl \
  --judge guard_only --policy-tail --blocks 1 --out /absolute/fresh/smoke/output
```

For coordinator emission, resolve `<RUN>` in BOTH JSON specs before invoking
`exp.offline_search.closed_loop.ops.emit_arms --run-root <RUN> --spec <resolved-spec>` (same Python prefix).
`validate.py` executed this exact emitter/config/parser path into `r06_abl/arm_validation/` without launching
anything. The two source arm lists remain unresolved as requested.

## Observed numbers and artifact hashes

See [results/TABLES.md](results/TABLES.md) for all eight metric checks, every trigger variant's removed MISS count
in both replay streams, all-off nesting exceptions, smoke counts, and all 24 prefit byte sizes/SHA256 hashes.
The machine-readable complete report is [results/summary.json](results/summary.json).

## Continuation: visual-key ablation of A (direct token PCA)

Completed 2026-09-28: all eight prefits and spec checks passed. The artifacts total 2,463,441,619 bytes.
Replay verification passed 5,468 direct-feature comparisons, 160 independent metric refits and 5,336 anchor-tail
checks; exact grid4 control parity passed another 5,468 comparisons. Token checks passed on 64 query rows and 64
library rows, both cameras. The two four-episode plugin smokes passed 202 decisions with zero MISSes.
Individual fit times ranged from 339.3 to 6,662.9 seconds; peak combined fit-worker RSS was 6.81 GiB.
Measured two-camera CPU projection medians ranged from 9.53 to 14.03 ms at three BLAS threads under shared load.
No live closed-loop experiment or GPU benchmark was run.

Files: `token_pca.py:TokenPCAAWM`, `arms_pca.json` (8 arms), `pca_prepare.py`, `pca_tokens_check.py`, `pca_ladder_check.py`,
`pca_check.py`, `pca_grid4_check.py`, `pca_checks.sh`, `pca_validate.py`, `pca_smoke.py`, `pca_report.py`, and `results/pca/`.
Final fits use `/home/weiland/trace_runs/os_closed_loop/r06_abl/fits/r6p2_direct_*.pkl`; PCA intermediates use
`r06_abl/pca/grid16/<model>_<suite>/<current|bpool_cs|bpool_all>/{v0,v1}/`.
Every source A field is preserved exactly except name, method, added `pooling_grid: 16`, and fit-artifact pathname
`<RUN>/fits/<arm>.pkl`. `pca_validate.py` checks this and executes the emitter/config/parser path. There is no GPU
flag, additional policy call, or shared plugin modification. Existing metric/trigger artifacts are untouched.

### Definition and resolution ladder

`pooling_grid` is the output grid SIDE: 1, 2, 4, 8, or 16. Each camera supplies f16 `[256,2048]` tokens, widened to
f32. Grid 16 is a row-major flatten to 524,288 values, with no pooling. Grids 1/2/8 average spatial blocks in f32,
then flatten. Grid 4 explicitly delegates to the original A fit/query path so it is an exact deployed-A control,
including the original PCA cache and model-specific pooled-key arithmetic. Only grid-16 specs are delivered.
Other grid settings need new prefits; grid-specific PCA directories prevent cache reuse across resolutions.
`pca_ladder_check.py` passed 20 recorded-token pooling cases (both models, both cameras, all five grids), with zero
observed error against a spatial-block reference, plus four invalid-grid refusals. Only grid16 has new fitted
artifacts; complete fits at grids1/2/8 have not been run.

For direct/other new grids, PCA is fit separately per camera, on the SAME candidate library A uses. It retains the
existing centered randomized-SVD recipe: sketch rank 128+32, seed 0, three QR-normalized power iterations, leading
64 PCs. Unlike stock's pooled-key reader, the input reader uses 128-row f16 token memmap blocks and drops its own
file-mapping resident pages after each detached f32 conversion. It never materializes N×524,288 in RAM. The PCA
mean is accumulated in f64 and stored in f32; sketch/projections/basis remain f32 as in A. The smaller streaming
block can change floating-point summation order. Explained variance is the actual projected centered energy over
the full library, divided by its full centered squared norm; individual ratios are saved in each camera's meta.json.

The original `AWM.fit` function bytecode is bound to a PRIVATE globals dictionary with only the PCA cache/loader
redirected. The original `fit_metric` is unchanged. Therefore the 64+64 visual coordinates plus 8 raw state values,
per-task z-scoring, action-similar neighbors, learned M, separate step<=2 early fit, step-0 affine branch,
confidence calibration, top-16 selection, kref 5/8, synthesis and anchor_tail budget 1 all follow the original code.
State values, their task/early normalization statistics, action chunks/scales and blind statistics are checked.
The new learned metric intentionally changes cross-terms involving state too; an unchanged raw/z-scored state
block does not imply identical final metric-weighted state coordinates.

At query time a narrow facade replaces only visual keys with flattened/pooled `q.tok_v0/q.tok_v1`; it forwards
state, episode, action and HIT histories untouched. `AWM.query`, `_dist`, synthesis and K1 blind continuation remain
the stock implementations. Tokens are read only on vision anchors. The optional, unused `still` diagnostic uses a
retained previous token key on truly adjacent vision calls; it is omitted when previous tokens are unavailable.
The plugin has no dense full-token history. This does not change retrieval, confidence, actions or vision schedule;
A's blind mixin already omits that diagnostic across blind gaps. The extra per-connection retained token keys occupy
4 MiB for two direct cameras. No token-projected feature is learned from evaluation queries.
The unchanged plugin/key builder still constructs its normal pooled keys for the surrounding cache interface;
the direct method ignores those keys. Eliminating that unused pooling work would require separate shared-path
work. The ablated retrieval representation itself performs no pooling at grid16.

### Token definition and online equality

Evidence: `results/pca/token_equality.json` and `pca_tokens_check.py`. Eight samples from each of the eight
model/suite/{cache,inf} query cells were joined to their original trace HDF5 `step_NNNN/vision_{0,1}` datasets.
The check invokes the ACTUAL trace serializers, key-builder `collect/_slice`, and
`closed_loop/plugin.py:PluginSession.tokens` on CPU stage-1 containers reconstructed from those recorded tokens.
All 64 rows × two cameras are bit-identical through HDF5 → query store → serializer → plugin, with output f16
`[256,2048]`, row-major spatial order. Sampled stored tokens also round-trip through bf16 exactly.
An additional 64 rows (eight per fitted library) are bit-identical through original build HDF5 → library token
array → serializer → plugin. All eight libraries' source-file sets are disjoint from both evaluation query streams.
This verifies extraction/layout/dtype parity on recorded inputs; no new vision-model forward or live GPU transfer
was run. Do not describe it as fresh-policy numerical parity.

- pi05: `models_pytorch/pi0_pytorch.py:embed_prefix` uses PaliGemma image features (SigLIP plus its multimodal
  projection), BEFORE the language transformer. The installed Transformers `PaliGemmaModel.get_image_features`
  takes the vision tower's `last_hidden_state` and applies `multi_modal_projector`.
  `components/key_builder.py:_slice_cp1_fields` selects prefix
  `[0:256]` for base_0_rgb and `[256:512]` for left_wrist_0_rgb. The third padded camera and prompt are excluded.
  `cache/trace/pi05.py:slice_prefix_tokens` uses that same slicer and casts to f16.
- GR00T: `cache/groot/staged.py` ends stage 1 after Eagle `extract_feature` is scattered into language input
  embeddings, BEFORE the language-model forward. `cache/groot/key_builder.py:slice_groot_cp1_fields` finds the two
  image-token mask runs; LIBERO camera order is video.image then video.wrist_image. Fixed pi05 offsets would be
  incorrect. `cache/trace/groot.py:slice_prefix_tokens` uses the identical mask slicer and f16 serialization.
  The installed Eagle source is `/home/weiland/projects/openpi_ext/third_party/gr00t_n15/gr00t/model/backbone/`.
  Its `eagle2_hg_model/config.json` has vision `select_layer=-1`, `use_pixel_shuffle=false`, one MLP connector,
  SigLIP hidden width 1152, 27 layers, 224-pixel input and 14-pixel patches. `extract_feature` therefore uses the
  vision tower's `last_hidden_state` (256 tokens), applies `mlp1` to width 2048, and performs no spatial reduction.
  `EagleBackbone` loads this local configuration. Both `/data/ckpt/n15_libero_{10,spatial}/config.json` specify
  backbone `select_layer=12`; that controls the later LANGUAGE output and is not the stage-1 image-key layer.
- `plugin.py:tokens` calls the model key builder's `_slice`, reshapes to tokens×channels, transfers to CPU and casts
  to f16. Online `tok_v0/tok_v1` map to vision_0/vision_1. `--os-tokens on` is already the source-spec default.
- `r0/build_library_store.py:process_episode` requires f16 `[256,2048]` HDF5 vision fields and copies them to library
  tok/v*.npy. `r0/extract_queries.py` copies those same trace vision fields into the query tok subsample. All fitted
  library manifests must have tok_complete=true and tok/rows.npy exactly arange(L); the fitter verifies both.

A baseline precision nuance was measured: pi05's online 4×4 key pools bf16 tokens in bf16, then casts to f32.
Repooling those tokens in f32 differed from recorded keys by up to 0.5; the actual bf16 builder matched exactly on
all sampled pi05 rows. GR00T's slicer casts to f32 BEFORE pooling; f32 repooling matched within 2.98e-8, while bf16
pooling differed. This is why grid4 uses deployed A directly, rather than silently claiming f32 repooling equals A.
Direct grid16 does no averaging or bf16 rounding; f16 serialization preserved all sampled stage-1 values exactly.
Hashes of the inspected token-definition sources/configs are in `results/pca/token_source_sha256.txt`.

### Tests, descriptive metrics, and resource measurements

Observed tables, fit times, memory, explained variance, projection latency, artifact sizes and SHA256 are in
[results/pca/TABLES.md](results/pca/TABLES.md); complete data are in `results/pca/summary.json`.

For every arm, `pca_check.py` uses each available token-subsample episode start plus 300 evenly spaced token rows
per cache/inf query stream. Query observations come only from the separate recorded evaluation trace store;
all PCA and metric fitting uses library rows. Held out here means separate trace files, not unseen tasks or a claim
of disjoint initial states. The evaluator sets prev_hit=True after step 0 because this is pure A,
so policy-continuity is never introduced by an inf-stream recording. It compares the new method with unmodified
BlindAWM on the new visual-key facade, and compares a grid4 control with the original deployed A artifact.
Comparisons include topk, scores, actions, confidence, library and every extra except the unused `still` diagnostic.
The separate `pca_grid4_check.py` checks ALL payload fields, including `still`, for exact deployed-A control parity.
It independently refits every task's main/early learned metric using stock `fit_metric`, and checks the untouched
state block and exact inherited anchor tails. It reports mean top-16 set overlap with A and mean per-query
sigma-scaled RMS action error against recorded `a_inf[:5,:7]` (MAE also saved). These are descriptive action
agreement statistics on fixed held-out observations, not predicted success rates. Closed-loop decides.

CPU projection timing measures both cameras, warmed up, 30 repetitions, three BLAS threads, using already-CPU
recorded inputs. Full query median and projection p95 are also saved. Values were measured under concurrent fits
and shared machine load; they exclude stage 1 and live device transfer. Projection matrices are f32 64×524,288:
128 MiB per camera, 256 MiB total plus 4 MiB of means (vs A's 8 MiB per camera). No GPU benchmark was attempted:
the hard constraint requires CUDA_VISIBLE_DEVICES='', and the installed K9 GPU adapter consumes pooled keys,
not these full tokens. A token GPU adapter would require separate authorized work; none was changed here.

Prefits use four three-thread workers plus a one-thread coordinator (13 threads); one three-thread test process
can overlap them (16 total). Maximum Python process count is six during that overlap. All use CPUs 18-25,62-69.
The supervisor samples the SUM of live fit-worker RSS every two seconds and aborts its own children above 60 GiB;
per-process high-water RSS is also recorded with each camera. The report includes the observed peaks. Token
memmaps are read-only; no global page-cache flushing, shared files, servers, workers, ports or tmux were touched.

### Commands and smoke recipe

All commands run from `/home/weiland/projects/openpi`. The NEW CPU allocation below supersedes the earlier
metric/trigger section for this continuation:

```bash
P=(taskset -c 18-25,62-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
D=exp/offline_search/rounds/r06/p2_ablations
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_tokens_check
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_ladder_check
taskset -c 18-25,62-69 env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p2_ablations.pca_grid4_check
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_prepare --fit
bash "$D/pca_checks.sh"
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_report
```

`pca_prepare.py` emits arms_pca.json and records every exact prefit argv in
`results/pca/prefit_commands.json`; its children carry their own correct taskset/env prefix. `--resume` skips
completed fit artifacts, while completed camera PCA caches are reused only when their library fingerprint,
grid, row alignment and recipe agree. No fallback directory was required. Existing fits are not overwritten.
For a fresh artifact, copy its recorded prefit argv and select fresh artifact/log paths; use a fresh `pca_cache`
kwarg too if recomputing PCA rather than reusing these camera caches. Hashes identify the delivered bytes;
refitting records a new fit time.
For another ladder point, change only pooling_grid and artifact name, then run the same plugin prefit command.

`pca_checks.sh` waits for each artifact, runs all eight checks, executes `pca_validate`, and runs TWO actual
installed-plugin smoke replays, one pi05 spatial/50 and one GR00T spatial/50, with FOUR complete token-subsample
episodes each. It checks every returned blind head equals source rows 5..9, no MISS, at most one blind decision,
and vision count == stage-1 call count. The following commands reproduce those smokes with fresh output paths:

```bash
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_smoke --model pi05 --out /absolute/fresh/pi05_pca_smoke
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_smoke --model groot --out /absolute/fresh/groot_pca_smoke
```

The smoke uses the installed plugin and recorded stage-1 tokens through FakeKB; there is no server, simulator or
new model inference. Do not use arbitrary recorded episodes without checking their token coverage. The coordinator
can resolve `<RUN>` in arms_pca.json and emit/start actual experiments using the existing A procedure. No such
experiment was started here. Tests/validation outputs require fresh paths on rerun.

For a separate **coordinator-run closed-loop smoke**, select `r6p2_direct_p_sp_50` and
`r6p2_direct_g_sp_50`, add `_smoke` to their arm names, retain their delivered fit paths and all runtime kwargs,
and emit them into a fresh run root. Set `OSCL_TASKS=0` and `OSCL_EPISODES=0,1,2,3` in the existing coordinator
launcher: one task × four initial states gives four episodes per arm, eight total. Keep the original five-control
client replan interval; anchor plus its single blind tail implements the ten-control commit. The coordinator must
allocate its own resources using its existing procedure. Check no MISS/policy inference, alternating anchor/tail
decisions, and no missing-token/shape errors. This live smoke recipe is unexecuted; the completed CPU replays above
do not substitute for action-dependent simulator validation.

The first check invocation stopped at import because optional threadpoolctl was unavailable; no check executed in
that attempt. The dependency was removed, and timings use the explicit environment thread limit. Its log is
preserved as `results/pca/initial_missing_threadpoolctl.log`.

`results/pca/files.sha256` records the final continuation source files and updated hand-back. The earlier
`results/files.sha256` is a historical snapshot from the metric/trigger work and predates this appended section.
