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
