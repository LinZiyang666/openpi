# HANDBACK P5 — GR00T N1.5-MetaWorld embodiment integration

Date: 2026-10-09. Checkout: `/home/weiland/projects/openpi`, branch `Ziyang`.

P5 implements the opt-in model hooks through the published R17 framework and the existing GR00T staged/cache path. Fix round 1 corrects the worker observation blocker, deploys per-bundle five-layer recipes and server/worker debug, separates teacher-library K from MISS K, and narrows capability declarations. Current CPU evidence is **214 unique passing tests, 0 failed, 0 skipped: 61 P5 + 153 legacy**. Original evidence was 43 P5 + 153 legacy tests. No GPU, real checkpoint forward, renderer, or real MetaWorld episode was run. G2/G3/G4 remain unqualified where the table below says pending.

## Owned files and actual change

All P5 implementation files are new; no existing model, benchmark, shared framework, frozen R16/harness, or `exp/mw_finetune/` file was edited by P5. Other packages' shared edits in this checkout are not P5 changes. New files are untracked in the shared checkout; ordinary `git diff` therefore does not display their contents. No commit or push was made.

| Owned path, relative to `exp/r17_integration/` | Actual change |
|---|---|
| `models/groot/__init__.py` | Lazy public exports; import installs no provider and loads no Torch/JAX/vendor. |
| `models/groot/manifest.py` | Saved DatasetMetadata validation, immutable content hashes, fixed-K endpoint specs, worker model/sampler metadata, honest capability statuses; full-action library identities exclude live MISS K while teacher provenance and native resume remain strict. |
| `models/groot/loader.py` | Explicit loading through B's unchanged MW `load_policy`; strict slot31/state4/action4/H16 metadata and eval checks. |
| `models/groot/key_builder.py` | One-camera subclass of stock spatial16 CP1 builder; first4 state mask checks; optional explicitly dimensioned MW CP2 projection. |
| `models/groot/provider.py` | P1 ModelAdapter/provider hooks, private runner/key memo per connection, shared model lock, request-local noise, native snapshots/resume, state-only normalization, trace/CP2/warm-reset hooks. |
| `models/groot/wire_policy.py` | Accept canonical and P2 worker observation keys, reject conflicting aliases; unchanged B transforms around stock policy/interceptor, exactly one inverse, no LIBERO gripper inversion, live-K guard even on HIT. |
| `models/groot/serve.py` | Thin common connection/recipe factory, R17 subclass of stock WS server and pre-publication validation; default-off `--debug-directory` uses P6 core-bound capture writers and seals them on shutdown. Separate opt-in stock advanced-hook assembly shim. |
| `models/groot/batcher.py` | Thin stock vectorized S3 wrapper; MW geometry/schedule/bucket checks and private writable reply storage. S1/S2 retain explicit rejection. |
| `models/groot/trace.py` | Capture already computed decision tensors; native snapshots and original widths; stock trace runtime configured for only `video.image`. |
| `models/groot/qualify.py` | Future bounded fixed-observation qualification CLI and partial receipt. Only `--help` executed today. |
| `models/groot/build_cpu_sample.py`, `samples/cpu_stub_capture.{json,npz}`, `samples/cpu_stub_bundle.yaml` | Reproducible clearly marked CPU diagnostic fixture, ModelManifest, native snapshots, normalized/wire arrays, and opt-in YAML. |
| `models/groot/legacy_sources.json`, `.gitignore` | Initial SHA256 snapshot of 18 unchanged reused GR/LIBERO/B source files; keep declared JSON/YAML artifacts reviewable despite repository ignore rules. |
| `tests/models/groot/{__init__,conftest,test_integration,test_shared_factory,test_library_recipe,test_loopback,test_p6_bridge,test_fix_round1,test_fix_round1_bundles}.py` | Independent unsplit noise/Euler fixture, complete formal request/ack contexts, real P2 worker messages, real GR entry/localhost transport/P2 worker entry/P6 capture, sealed K4 library reuse and hot YAML regressions. |
| `tests/models/groot/{fix1_before.xml,fix1_final.xml,fix1_legacy.xml,verification_fix1.json,.gitignore}` | Fix round 1 failing-before and passing-after evidence, exact commands/content hashes, scratch exclusions. Original `cpu_final.xml`, `legacy_final.xml`, `p6_yaml.xml`, and `verification_receipt.json` remain historical evidence. Intermediate failure receipts are not final passing evidence. |
| `tests/models/groot/record_fix1_receipt.py` | Reproducible receipt/AST/sample/source-hash/Bash-syntax verification; consumes already-run XML, executes no GPU/model commands. |
| `REQUESTS/P5_to_{P1,P2,P6}.md` | Shared interface requests and current handoff/status. |
| `handbacks/HANDBACK_P5.md` | This handback. |

The source implementation reuses unchanged `GrootStagedRunner`, `GrootCacheInterceptor`, `GrootStageBatcher`, GR trace/warm-reset builders, stock CacheOrchestrator/backend/component factories, P1 connection policy/five-layer math and R17 WebsocketPolicyServer subclass, P2 sealed protocol/endpoint types, and P6 writer/library tools. P1's R17 config/WS extension modules preserve the sealed stock source files. There is no new Eagle network or separate model-owned server protocol.

## Interface version and binding

Consumed interface: **`r17.contracts.v1`**, directly from P1. Provider ID: `r17_groot_mw_v1`; model ID: `groot_n15_metaworld_sawyer`; CP1 builder ID: `r17_groot_mw_cp1_spatial16_v1`; YAML key builder type: `adapter_native`.

Exports: `register_provider()`, `load_adapter(spec, ...)`, `make_connection_factory(...)`, `make_stage_batcher(...)`, `make_trace_adapter(...)`. `make_provider(backend, manifest)` returns a provider whose loader creates a fresh wrapper/guarded runner over the resident backend/model. The shared lock protects backend transforms and model stages; mutable connection recipe/cursor/memos are private. The generic factory uses NamedFieldsKeyBuilder over already computed retrieval fields; the opt-in stock shim uses the native GR key builder.

The manifest uses `ModelSpec`, `ActionStateLayout`, `FeatureSpec`, `SamplerSpec`, `Capabilities`, RequestContext/RequestRNG, NativeSnapshot, and DecisionCapture from P1. Sample JSON contains the P1 `r17.model_manifest.v1` envelope. YAML binding is `r17.adapter_binding.v1`; per-bundle recipes use `r17.recipe.v1`; decisions/snapshots use `r17.decision_capture.v1` and `r17.native_snapshot.v1`. Extended commit requires `r17.execution.v1` acknowledgment negotiation. P2's `FORMAL_SPEC.identity` is the protocol digest; the human label `mt50_seed7_corner2_r17_v1` is not used as compatibility identity.

Weights/config/metadata/code/tokenizer/transform identity is content-based, including all safetensor shards. A same-size, same-mtime weight change is detected by a passing CPU test. Caller-supplied tokenizer/code asset lists must be complete: this loader does not discover or attest an omitted dependency. ModelSpec marks only deployable generic full/K/CP1/concurrent/bundle/episode/debug paths `supported`, with CPU-only qualification reasons. Library qualification, trace, S3 batching, native resume, CP2 keys and warm-reset hooks are `implemented_unverified`: direct hook tests do not establish generic deployment. `GrootManifest.to_dict()` also preserves the conservative unqualified hook envelope. Nothing is GPU-qualified.

Each endpoint has exactly one immutable K in `{4,2,1}`. P2 EndpointSpec uses `model_identity=checkpoint_identity`, exact norm/layout/protocol identities, and `allowed_k=(liveK,)`. The original handback's YAML-switchable recipe/library claim was unsupported by the original serve entry; Fix round 1 now wires it and tests it over localhost. A new bundle generation can select a sealed store/fit, corrector, guard, gap or target IR, or plain serving. Commit must be 5/10/15 for this formal H16/replan5 layout. A different MISS K requires another endpoint/process; no request changes the shared head K. A K4 full-action teacher library is valid at K2/K1 when immutable checkpoint/norm/layout/features/code/protocol match. Its teacher sampler identity stays distinct from the endpoint sampler, while native snapshots still require exact K/grid identity.

## Default-off certificate

`test_default_off_import_and_legacy_source_bytes_unchanged` passed. A fresh Python process imports only `exp.r17_integration.models.groot`, checks Torch/JAX/vendor are absent, and confirms the provider is not registered. The same test compares all 18 files in `legacy_sources.json` byte-for-byte with the initial SHA256 snapshot. Production runner construction preserves **both** stock vendor source guards; `verify_upstream=False` is not used. A source-drift fixture is rejected. Only the explicit diagnostic fixture pins its own stub source hashes inside its test/sample process.

The 153 targeted legacy GR/LIBERO tests passed with default hooks off. This is a source-byte certificate plus exact CPU action/parity regressions for the selected paths; it is not a claim that every legacy system behavior or GPU numerical path was exercised. P1 owns the default-off proof for its separately edited shared files.

## CPU commands and actual results

Run from `/home/weiland/projects/openpi`. The following environment was used for every pytest command; GPUs were hidden and JAX forced to CPU:

```bash
P5_CPU=(env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python)

"${P5_CPU[@]}" -m pytest exp/r17_integration/tests/models/groot \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/groot/scratch_fix1_final_guard \
  --junitxml=exp/r17_integration/tests/models/groot/fix1_final.xml --tb=short -q

"${P5_CPU[@]}" -m pytest \
  tests/cache/groot/test_groot_staged.py \
  tests/cache/groot/test_groot_stage3.py \
  tests/cache/groot/test_groot_key_builder.py \
  tests/cache/groot/test_libero_key_builder.py \
  tests/cache/groot/test_groot_interceptor.py \
  tests/cache/groot/test_groot_batcher.py \
  tests/cache/groot/test_groot_load_guard.py \
  tests/cache/groot/test_trace_groot.py \
  tests/libero_groot/test_dynamic_bundle_guards.py \
  tests/libero_groot/test_policy_adapter.py \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/groot/scratch_fix1_legacy \
  --junitxml=exp/r17_integration/tests/models/groot/fix1_legacy.xml --tb=short -q

"${P5_CPU[@]}" -m exp.r17_integration.models.groot.build_cpu_sample
"${P5_CPU[@]}" -m exp.r17_integration.models.groot.serve --help
"${P5_CPU[@]}" -m exp.r17_integration.models.groot.qualify --help
"${P5_CPU[@]}" -m exp.r17_integration.tests.models.groot.record_fix1_receipt
```

| Evidence | Exit | Pass | Skip | Result |
|---|---:|---:|---:|---|
| `fix1_final.xml`, updated full P5 suite | 0 | 61 | 0 | 7.05s pytest output; includes original tests, deployment regressions and default-off proof; no failures/errors. |
| `fix1_legacy.xml` | 0 | 153 | 0 | 3.51s pytest output; 0 failures/errors. |
| `fix1_before.xml` | 1 | 0 | 0 | Six regressions failed on the reviewed P5 code, before owned fixes; intentionally retained negative evidence. |
| Sample generation | 0 | n/a | n/a | Stub NPZ/JSON/YAML regenerated successfully; actual NFE4. |
| Both CLIs `--help` | 0 | n/a | n/a | Parser/import boundary only; no model load. |
| Fix round 1 standalone AST/NPZ/hash receipt check | 0 | n/a | n/a | 21 owned Python modules parsed, diagnostic shapes/dtypes/finiteness, all 18 legacy source hashes, three Bash blocks syntax-checked; current dependency hashes recorded. |

The Torch import emits a pre-existing `pynvml` deprecation warning. The legacy suite also logs `trace writer: exp/ep1 failed at stop: writer stopped with episode open` from a legacy trace test; pytest still reports 153 passing tests. This log is not evidence of a closed trace. P5's own stock H5 and P6 bridge tests explicitly close/drain/read their captures and pass.

Machine-readable commands, metrics, sample/source/interface SHA256s, and limits are in `tests/models/groot/verification_fix1.json`; `verification_receipt.json` is the original historical receipt. Shared package files remain subject to concurrent work; the new receipt records their content at evidence generation. The documented Bash blocks are syntax-checked without executing GPU commands. No real safetensor checkpoint was loaded: the content-tamper test hashes a tiny diagnostic byte fixture, not model weights. No lint run is claimed.

## Model/benchmark hook and qualification table

| Hook / framework feature | Implementation and CPU evidence | Status / remaining owner |
|---|---|---|
| MW wire/normalization | B's native `to_groot_observation`/`from_groot_actions`; apply once and inverse once; nonidentity fixture proves active4 transform and unchanged native gripper. State-only extraction uses original state transform components and matches S1 normalized state. | CPU pass; real vendor transform/GPU parity pending coordinator/P5. |
| CP1/native stage cuts | Stock guarded runner; actual image-token mask selects one contiguous 256-token run; stock float32 spatial pooling; state64 mask must select first4; slot31 and hidden width validated. | CPU pass; actual checkpoint token geometry pending. |
| Full, K reduction, noise, native resume | K4/2/1 staged versus independent unsplit exact bytes; capture on/off exact and no extra RNG/NFE; per-request semantic noise; every recoverable native suffix exact; bad identity/index/shape/dtype rejected. | CPU pass; BF16/GPU numerical receipt pending. Low-K quality remains U. |
| CP1 HIT/MISS | Real CacheOrchestrator/InMemoryBackend; HIT stage calls `(1,0,0)`, MISS `(1,1,K)`; inverse once. Frozen live K is checked even when the library HIT skips LLM/head. | CPU pass in generic and opt-in stock paths. |
| Concurrent server/YAML switching | P1 R17 subclass of stock WS: two clients, episode/disconnect cleanup, immutable generation identity, atomic bad-bundle rejection. Actual GR serve switches two distinct sealed stores/recipe settings and plain serving independently at K4/2/1. | CPU pass; real GPU WS smoke pending P1/coordinator. Concurrency does not certify stage batching. |
| Commit horizon/execution | H16 commit10/15, negotiated acknowledgment, actual execution cursor and stage-free tail; nonaligned commit7/12 rejected. Actual P2 runner and worker entry terminate after seven fake-environment controls with final acknowledgment. | CPU pass; real renderer/benchmark receipt pending. |
| Five-layer library math/deployment | P1 layout recipe and P6 sealed fit: one view/state4/task50, LOEO, active-channel padding invariance, motion corrector, guard, periodic forced MISS. Actual GR serve publishes YAML corrector/guard/target-IR/store choices; stock-preloaded K4 teacher actions work with native K4/2/1 MISS grids. Bad store/fit/teacher identities fail before publication. | CPU pass for full-action CP1 serving and deployment; complete library/replay/augmentation G3 qualification remains pending. |
| Debug/model capture/P6 library | Actual GR main `--debug-directory` + localhost WS + P2 worker main `--debug-directory` + P6 writers/closed readers at K4/2/1. Native model tensors, original normalized H16×32, wire/state4, native-shaped float64 physical data and final executed counts/cursors are verified. NPZ → H5 → store → stock artifact → YAML preload also passes. | CPU pass with model/environment fixtures; real paired debug/GPU transport/cost pending P6/coordinator. Debug rows are excluded from pricing by shared framework metadata. |
| Stock model trace | Stock trace writer/runtime with only `video.image`, raw state4, native snapshots, K4/2/1 H5 reads. | CPU pass for stock hook; generic advanced trace configuration currently rejects. Certified factory registration is P1 pending. |
| Stage batching | Stock vectorized S3 MISS and warm buckets, explicit per-request noise/trace points; B2 equals CPU singles; wrong shape/schedule/semantics rejected; cloned replies do not alias and are writable. | CPU pass; GPU B2/cost and shared BatchingCore submission integration pending P1/coordinator. S1/S2 unsupported. |
| CP2/online RIT | `make_cp2_key_builder`/`cp2_fields`: explicit token/state feature sizes, first4 state, slot31, MW projection metadata; no LIBERO 640-token default. | CP2 key hook CPU pass; full CP2/online-RIT execution unsupported/unqualified. P1/P5 certified combinations pending. |
| Native warm reset | `make_warm_reset_parts` reuses stock engine/provider with live schedule guard and active state4; native schedule/resume and engine hook tests pass. | Hook CPU pass; complete generic warm-reset configuration rejects. P1/P5 advanced executor and real MW smoke pending. |
| Conductor/benchmark | P2 sealed formal digest, exact native4 layout and fixed-K EndpointSpec; incompatible episode/bundle rejected in CPU tests. | Endpoint compatibility CPU pass; P2 owns queue/runner/renderer, coordinator owns mechanical reach episode. No full campaign receipt. |
| Split/CPU blueprint | Explicit unsupported exception; no fake decoupled model or tensors sent to worker. | Unsupported; P1/P5 future qualification. |
| Dual-camera ablation | One real `video.image`; fake wrist fields refused. | Not applicable. |

## Native geometry, normalization and storage samples

Production declaration: `metaworld_sawyer`, embodiment slot31, RGB480×480 corner2 input; normalized state4 backed by state64/mask-first4; H16×32 original normalized actions with active indices `[0,1,2,3]`, motion `[0,1,2]`, gripper3; wire H16×4 before common commit-tail truncation. Wire gripper stays native MetaWorld. No output clipping or LIBERO toggle is added here. The benchmark owns applied-control clipping and acknowledgment.

CP1 production field widths: `vision_0=16×2048=32768`, `robot_state=4`. Mask-derived native image tokens are 256×2048, reshaped 16×16 and pooled to spatial4×4 in float32. There are no `vision_1`/`vision_2` fields. Metadata validation reads the real vendor saved DatasetMetadata shape/statistics schema, not the conversion input's start/end schema. The real checkpoint's 256-token run and hidden2048 are asserted at runtime, not presumed qualified.

| K | Native flow input points | Euler delta | Recoverable saved inputs |
|---:|---|---|---|
| 4 | `[0, 0.25, 0.5, 0.75]` | 0.25 | loop1/2/3 (`t=0.25/0.5/0.75`) |
| 2 | `[0, 0.5]` | 0.5 | loop1 (`t=0.5`) |
| 1 | `[0]` | 1 | none; no valid partial resume point |

The native schedule is the stock GR ascending noise-to-clean flow loop; K2/K1 are separate full native grids, not prefixes of K4. Snapshot identity binds checkpoint/norm/layout/sampler/grid/index/time. Portable snapshots preserve original native dtype in `noise_state={native_dtype, storage_dtype:"float32"}`; resume permits that declared transport only when values reconstruct exactly in the live dtype. Real BF16 transport is unqualified today.

`samples/cpu_stub_capture.json` and `.npz` explicitly set `diagnostic_fixture=true`, `real_checkpoint=false`, `gpu_qualified=false`. Their hidden8/vision128 tokens are not production features. Shapes: vision128, state4, normalized `[16,32]`, wire `[16,4]`, initial noise `[16,32]`, snapshots `[16,32]`; all stored floating arrays are float32. The fixture uses state `2*x-1` and wire `3*x+0.25` to exercise nonidentity normalization. The sample YAML has commit10, K4 and only true fields; its diagnostic identities must not be used with a real checkpoint.

P6 stock artifact metadata keeps `{model_spec, r17_binding}` under `artifact.model`, which survives the unchanged stock loader. P5/P1 validate immutable retrieval model identity (all ModelSpec fields except endpoint default sampler and capability declarations), teacher sampler provenance and exact geometry/protocol. This resolves top-level `r17_manifest` loss without coupling full teacher actions to live MISS K. Both explicit and stock preload roundtrips pass; native warm snapshots do not use this compatibility relaxation.

## GPU-ready commands — NOT EXECUTED

Coordinator runs only after training releases the device, under its exclusive GPU scheduling/lock procedure. No automatic retry and no mutation of a live endpoint's K. Each invocation below loads one immutable final checkpoint at one K and ends within a **14-minute timeout plus at most 10 seconds for its own child cleanup**. Run K4, K2 and K1 in separately scheduled slots if all three cannot fit the first slot. Do not use a still-changing training directory.

Coordinator supplies `P5_FINAL_CHECKPOINT` (immutable final B export), `P5_FIXED_OBS` (no-pickle NPZ: images `[N,480,480,3]`, states `[N,4]`, string prompts `[N]`, at least two rows), and `P5_TOKENIZER_FILES` (Bash array containing every actual local tokenizer asset). Tokenizer/model assets must already be available in the declared vendor island. Choose K once per invocation. These variables deliberately require the final immutable export; no current training checkpoint is selected in this handback.

```bash
cd /home/weiland/projects/openpi
P5_GROOT_PY=/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python
P5_VENDOR=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
P5_K=4
# Supply P5_FINAL_CHECKPOINT, P5_FIXED_OBS and P5_TOKENIZER_FILES before proceeding.
: "${P5_FINAL_CHECKPOINT:?immutable final export required}"
: "${P5_FIXED_OBS:?fixed native MW observations required}"
: "${P5_TOKENIZER_FILES[0]:?complete local tokenizer asset array required}"
mapfile -t P5_CODE_FILES < <(rg --files "$P5_VENDOR/gr00t" \
  exp/mw_finetune/groot exp/r17_integration/models/groot src/openpi/cache/groot -g '*.py')
P5_ASSET_ARGS=()
for P5_ASSET in "${P5_TOKENIZER_FILES[@]}"; do
  P5_ASSET_ARGS+=(--tokenizer-file "$P5_ASSET")
done
for P5_ASSET in "${P5_CODE_FILES[@]}"; do
  P5_ASSET_ARGS+=(--code-file "$P5_ASSET")
done
timeout --signal=TERM --kill-after=10s 14m \
  env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$P5_VENDOR:$PWD:src:packages/openpi-client/src" \
  "$P5_GROOT_PY" -m exp.r17_integration.models.groot.qualify \
    --checkpoint "$P5_FINAL_CHECKPOINT" --observations "$P5_FIXED_OBS" \
    --k "$P5_K" --device cuda:0 --deadline-seconds 840 \
    --receipt "exp/r17_integration/models/groot/samples/gpu_fixed_k${P5_K}.json" \
    "${P5_ASSET_ARGS[@]}"
```

This probe records content identities, actual one-camera/key shapes, native NFE/grid, staged versus B unsplit same-seed outputs (`rtol=atol=1e-4`), exact debug-on/off bytes, and every available native suffix. It also records synchronized preparation, S1/pooling/transfer, LLM, native-K sampler and wire timing without additional forwards for timing. Ckey is preparation + S1/key transfer; these cold/preceding-probe, capture-enabled isolated timings are qualification observations, not steady-state/server cost claims. A partial JSON exists before loading; exception/timeout never counts as passed. External timeout can leave only that initial partial receipt.

Opt-in real server command for the coordinator's **separate ≤15-minute P1/P2 mechanical slot**, using the same assets/K:

```bash
: "${P5_GPU_DEBUG_ROOT:?fresh coordinator-owned model capture directory required}"
timeout --signal=TERM --kill-after=10s 14m \
  env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$P5_VENDOR:$PWD:src:packages/openpi-client/src" \
  "$P5_GROOT_PY" -m exp.r17_integration.models.groot.serve \
    --checkpoint "$P5_FINAL_CHECKPOINT" --port "$P5_RESERVED_PORT" \
    --denoising-steps "$P5_K" --device cuda:0 --commit-horizon 10 \
    --debug-directory "$P5_GPU_DEBUG_ROOT" \
    "${P5_ASSET_ARGS[@]}"
```

The server command alone is not a qualification receipt. P1/P2/P6 must attach their bounded two-client/two-YAML forced HIT/MISS/tail and short reach-v3 runner hook, framework context/actual execution acknowledgments, closed debug capture, and owned cleanup. Use a coordinator-reserved free port. Do not infer real WS, renderer, stage-batch, quality or full-feature gates from a successful fixed-observation probe. No such live commands were executed today.

## Known limits and pending owners

| Pending item | Owner | Boundary / next evidence |
|---|---|---|
| Immutable final checkpoint and complete actual tokenizer/code assets | Coordinator + B | Required before GPU loading; training and checkpoint quality are outside P5. |
| Actual hidden2048/run256/slot31/norm, BF16 transport, K4/2/1 upstream parity and Ckey/LLM/head costs | Coordinator runs P5 CLI; P5 diagnoses failures | G2 pending. CPU stubs do not certify real vendor arithmetic or feature cost. |
| Real WS two-bundle HIT/MISS/tail, real S3 B2 and replica behavior | P1 + coordinator, consuming P5 hooks | CPU behavior passes; shared model lock is concurrency, not batch scheduling. Need explicit BatchingCore submission and GPU receipt. |
| Certified advanced trace/CP2/online-RIT/warm-reset/routing factory registration and combination checks | P1 + P5 | Generic executor fails closed today; stock local shim/hook tests do not certify every advanced combination. See P5_to_P1. |
| Real formal protocol runner/renderer reach-v3 episode with actual execution/terminal acknowledgments | P2 + coordinator | Real P2 runner/worker entry and final acknowledgments pass with a CPU environment fixture. Real simulator/renderer not instantiated. |
| Real capture/library fit/replay/search/augmentation/transport/analysis qualification | P6 + coordinator, consuming P5 trace/geometry | CPU writer/store/YAML roundtrip passes. Full G3 and paired GPU debug receipts remain pending. |
| Whole MetaWorld physical goal/restore semantics | P2 | P5 supplies native4 actions; no additional physical-semantic claims. |
| S1/S2 batch and split blueprint | P1/P5 future work | Explicit unsupported; do not silently advertise or execute. |
| Final 500-episode baseline, library/dev/test split sealing, low-K/commit quality and paired cost comparison | Coordinator | G4 unqualified; no SR/latency conclusions from CPU data or historical training smoke. |

Decisions: consume published contracts rather than fork them; keep fixed K per process; preserve the genuine one-view/first4/native32 geometry; use original normalization and no new gripper convention; retain vendor guards; transport native snapshots with explicit dtype; let shared framework own conductor/serving/layer math; request shared advanced registration rather than edit another package. The P1 lifecycle/import and P6 stock-preload metadata seams found during integration are resolved in the passing final CPU tests.

## Fix round 1

Both `review/REVIEW_A.md` and `review/REVIEW_B.md` were read fully. P5 agrees with all assigned findings; the original tests missed deployment seams. No reviewer finding is disputed. Regressions mirror the reviewer wire/library/metadata repros and extend them through actual entrypoints. The retained `tests/models/groot/fix1_before.xml` records **six failing tests on the reviewed P5 code**: two worker-key failures, two cross-K library failures, missing worker metadata, and overstated trace capability. The expanded updated suite passes all 61 tests. Its additional deployment fixtures use the real framework and localhost WebSocket transport, with CPU model/environment substitutes.

| Finding ID | Owned change and shared integration | Regression evidence | Status |
|---|---|---|---|
| A #1, blocker | `wire_policy.observation_field` accepts `image/state` and `observation/image/observation/state`; provider preparation and state-only extraction use it; B's validation/transforms remain unchanged. Conflicting aliases fail closed. | `test_review_a1_real_p2_runner_worker_wire_reaches_groot`: actual P2 episode task, runner/kernel and worker message, original init7, rotated frame, one MISS then free tail and seven controls. `test_review_a1_both_wire_key_sets_have_exact_actions`: byte-identical canonical/worker actions. Both failed with KeyError before fixing. The three actual worker-entry debug cases also cover this path. | Fixed; CPU pass. |
| A #4, major: five-layer deployment/YAML | GR `serve` explicitly passes `common.recipe.make_recipe_factory(provider)` to shared connection factory. P1 owns `r17.recipe.v1` parsing, fit/store seals, per-bundle recipe identity and pre-publication fitting. P5 consumes the published APIs and removes temporary local wrappers. | `test_review_a4_production_hot_yaml_recipe_library_and_plain_independent[4/2/1]`: actual serve assembly and localhost load/select controls; switches sealed 24-row and 32-row stores, corrector/guard/target-IR choices and plain serving. Verifies real CP1 HIT versus native-K MISS, recipe identity, and rejection before publication of corrupt store/fit/teacher seals. The regression explicitly requires production serve to pass its recipe factory, absent in the reviewed entry. Existing LOEO/padding/corrector/periodic tests also rerun. | Fixed; deployment tested on CPU. Complete G3/library quality remains unqualified. |
| A #4, major: library/endpoint K coupling | `manifest` and provider validate immutable retrieval model identity separately from teacher sampler. P1 publishes distinct teacher identity and native MISS sampler binding; P6 retains complete stock artifact model spec plus compact binding. Native resume remains exact-K/grid. | `test_review_a4_k4_teacher_full_actions_compatible_with_reduced_miss[2/1]` failed before and passes now. `test_review_a4_stock_teacher_artifact_and_recipe_miss_grids_independent[4/2/1]`: real stock preload of sealed K4 teacher artifact, library HIT then periodic MISS with NFE equal to endpoint K. `test_review_a4_native_snapshot_still_rejects_cross_k` preserves strict native resume. Production hot-switch cases also preload K4 teacher at each endpoint K. | Fixed; CPU pass. |
| A #5, major: deployable debug | GR main adds `--debug-directory` and `--host`; metadata exposes actual `r17_model_spec` and fixed-native `r17_samplers`. Serve uses P6 core-aware writer factory through P1's capture seam and seals writers on own shutdown. P2 worker consumes the same metadata and P6 physical writer. | `test_review_a5_metadata_is_real_p2_worker_model_binding` failed before and passes now. `test_review_a5_real_serve_cli_and_worker_debug_over_localhost[4/2/1]`: real GR CLI assembly, P1 transport, P2 worker CLI/runner and P6 model/physical writers/closed readers. Verifies one native MISS, stage-free tail, native snapshots, normalized H16×32, one-view/state4, seven physical controls, task UID/init7, executed counts `[5,2]`, cursors `[5,7]` and terminal acknowledgment. Model loader, renderer/environment and conductor WorkerLoop transport are CPU substitutes; live conductor journaling is not claimed. | Fixed; CPU deploy-path pass. Real GPU/renderer debug and pricing receipts pending. |
| A #12, major: capability/handback claims | Direct trace/S3/resume/warm-reset/CP2-key hooks and broad library qualification marked `implemented_unverified`, with explicit deployment limits. Basic generic serving capabilities retain CPU-only reasons. Historical YAML claim explicitly retracted and replaced with current tested scope. | `test_review_a12_direct_hooks_not_advertised_as_deployed` failed on `trace=supported` before; now asserts unverified status and `Capabilities.require` rejection. All direct hook tests still pass. | Fixed; truthful capability declarations. No GPU-qualified capability claimed. |
| A #9, shared commit defect affecting GR entry | Consume P1's strict formal replan-aligned commit validator; P5 does not allow partial formal commits. | `test_review_a9_gr_entry_rejects_non_replan_aligned_commit[7/12]` rejects both reviewed invalid values; original valid commit10/15/tail tests still pass. | Fixed by P1; owned regression pass. |

Default-off revalidation: fresh import registers no provider and loads no Torch/JAX/vendor; all 18 reused legacy files match the initial byte hashes; both production upstream source guards remain enabled. The complete P5 suite and the complete selected 153-test legacy suite were rerun after consuming the P1/P2/P6 fixes. `verification_fix1.json` records exact commands, XML counts, source/fixture hashes, limitations and CPU device; diagnostic sample was regenerated with actual NFE4. Intermediate migration failures are retained as development evidence, not counted as passing results. The old receipt remains historical and is not relabeled as testing today's sources.

Resolved cross-package requests are documented in `REQUESTS/P5_to_P1.md`, `P5_to_P2.md`, and `P5_to_P6.md`: shared sealed-source config/WS migration, per-bundle recipes, teacher/MISS identity separation, capture sink factory, actual worker sampler/metadata and stock artifact provenance. No unresolved cross-package blocker remains for the CPU paths tested here. Remaining owners are unchanged: coordinator/B for final immutable checkpoint and GPU slot; P1/P5 for generic advanced trace/CP2/warm/batch dispatch; P2/coordinator for real renderer/conductor episodes; P6/coordinator for real-library fit/replay/augmentation/debug transport and pricing. GPU commands above remain **not executed** and require one model at a time in a coordinator-controlled ≤15-minute slot.
