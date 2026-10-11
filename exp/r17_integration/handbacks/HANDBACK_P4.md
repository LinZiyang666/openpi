# HANDBACK P4 — DP-T LIBERO / MetaWorld

2026-10-09, shared checkout `/home/weiland/projects/openpi`, branch Ziyang.
**CPU implementation delivered; GPU / renderer / final MT50 quality unqualified.**

**The 2026-10-10 dynamic batching section at the end is the latest release.
It adds explicit DP-LIBERO batching; default B1 and K1 admission remain enforced.
Initial/fix round1/2/B3/GPU fix1/2 sample pins are historical for that release.**
Earlier rounds used zero GPU. GPU fix2 used one explicitly owner-authorized
bounded diagnostic; no training/process/session changes, git commit/push, frozen R16/harness
edits, or edits under `exp/mw_finetune/`. All implementation changes are new
P4-owned files under R17. This does not declare G2/G3/G4 fully qualified.

## Owned files and actual diff

| Added files under `exp/r17_integration/` | Change |
|---|---|
| `models/dp/{__init__,config,loader,manifest}.py` | Lazy opt-in exports; strict E loader/475 tensors; immutable path + content pins; JSON manifests binding DP/text/tokenizer/code/dependency/norm content. |
| `models/dp/{staged,key_builder,normalization}.py` | Exact native two-branch ReLU64 features, stage2 condition assembly, scheduler-space payload, two inverse transforms, one/two real-camera geometry, canonical P2 MW wire. |
| `models/dp/{sampler,request_noise}.py` | Native DDPM/DDIM descriptors, rebuilt K grids, sequential private initial/posterior stream, exact suffix RNG state, native add_noise reset. |
| `models/dp/{batcher,provider,wire_policy,serve}.py` | P1 provider/factory, stock three-queue BatchingCore, per-request RNG/capture, stage shape/K/profile/resume buckets, stock concurrent WS entry, bound YAML and metadata. |
| `models/dp/{trace,probe}.py`, `models/dp/README.md` | Observe-only capture + explicit matched-noise variants; bounded real checkpoint probe; portable JSON/numeric NPZ fixture. |
| `tests/models/dp/{conftest,test_staged,test_sampler,test_dp_batcher,test_dp_serving,test_interop}.py` | 56 CPU tests over E vision + reduced Transformer, real schedulers, actual localhost WS, stock cache/core, P1 layers, P2 fake episode, P6 closed writer/ack/H5. |
| `models/dp/samples/`, `models/dp/cpu_*.xml`, `tests/models/dp/scratch_*` | Audited inert metadata fixture, real native captures/manifests/receipts, pytest evidence and retained scratch. Earlier development XML/receipts are historical, not acceptance totals. |
| `REQUESTS/P4_to_{P1,P2,P3,P6}.md`, this handback | Interface requests/resolution, MW wire alignment, native capture handoff, unique test module names for aggregate collection. |

15 implementation modules / 6 test modules. Shared `src/openpi` changes visible
in this checkout belong to P1; `logs/session_handoff.md` and E's ongoing files
were already owned by other agents. P4 did not edit them. No replacement server,
cache orchestrator, recipe math, conductor scheduler, or physics implementation.

## Interface version / framework path

Directly consumes **`r17.contracts.v1`** (`common/contracts.py`, provider,
request_rng, serving). No temporary contract shim remains; `config.LoadSpec`
only specifies loader options. Exports `load_adapter`, `register_provider`,
`make_connection_factory`, `make_stage_batcher`, `make_trace_adapter`.

`P1.make_connection_factory → stock components / CacheOrchestrator → DP staged
adapter → to_wire` is the execution path. An optional `CoordinatedAdapter`
routes those same calls through the stock `BatchingCore`; no alternate cache
executor. `WebsocketPolicyServer` receives the P1 bundle validator and factory.
Registration is explicit; shared resident weights/language/norm are read only,
while wrapper, keys, orchestrator history, tail cursor and episode state are
connection owned. DDPM/DDIM providers have distinct identities.

MetaWorld accepts E's `image/state` and P2's `observation/image` /
`observation/state`. Bundles bind **P2 FORMAL_SPEC.identity**. WS metadata
includes `r17_model_spec`. LIBERO currently binds the explicit `libero_r17_v1`
protocol label; richer suite content identities await P2/coordinator.

## Default-off proof

- Fresh-process import of `exp.r17_integration.models.dp` imports neither torch
  nor JAX nor the provider registry. No registration/load happens at import.
- CPU tests compare staged actions **byte exactly** against E's unchanged
  full sampler for LIBERO and MW at K100/2/1. Real 475-tensor probes additionally
  compare full `E.Policy.infer` at default DDPM100, including preprocessing.
- Capture on/off has identical actions and final RNG state; debug adds zero
  model forwards/RNG draws. HIT stage calls are `(1,0,0)`, MISS `(1,1,K)`, tail
  `(0,0,0)`. Closed P6 writer and execution ack preserve actual count/cursor.
- **47 existing legacy serving tests pass** with optional hooks unregistered:
  identity, per-step recorder, router bundle and replica proxy, including six
  localhost fake-backend socket tests. E inference dependencies are read only;
  source hashes are in real receipts and `samples/handback_summary.json`.

This proves CPU numerical/default-path behavior tested here, not GPU bit parity
or unpublished author-wrapper equivalence.

## CPU commands and actual receipts

All commands used `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`,
`PYTHONDONTWRITEBYTECODE=1`, OMP/OpenBLAS/MKL threads=1, and
`PYTHONPATH=.:src:packages/openpi-client/src`. Working directory is the repo root.

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m pytest \
  exp/r17_integration/tests/models/dp -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_delivered \
  --junitxml=exp/r17_integration/models/dp/cpu_delivered.xml -q
```

**Exit 0: 56 passed, 0 skipped, 0 failures/errors, 29.44s.**

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m pytest \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py tests/serving/test_replica_proxy.py \
  --run-manual -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_legacy_handback \
  --junitxml=exp/r17_integration/models/dp/cpu_legacy_handback.xml -q
```

**Exit 0: 47 passed, 0 skipped/failures/errors, 8.84s.**
The `--run-manual` targets here use fake localhost backends, not GPUs/renderers.

```bash
# Ran once per listed domain/checkpoint, sequential real probes; CPU only.
timeout -k 5s 300s env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/release_libero_cpu
# Same command: checkpoint metaworld / domain metaworld /
# output .../samples/release_metaworld_cpu
```

Real LIBERO: **exit 0, 11 probe checks passed / 0 skipped**, 36.70s.
Real MW: **exit 0, 12 probe checks passed / 0 skipped**, 35.76s.
These are probe checks, separate from **103 distinct pytest tests**, and are not
SR runs. Both strictly loaded 475 tensors and tested K100/2/1, full-wrapper
parity, ReLU64 geometry, native resume, capture parity, explicit DDIM, real B2.
MW additionally proves its zero-wrist feature changes with prompt.

Toy B1/B2 and reversed B2 have exact RNG state and maximum action difference
`1.788139343e-7` DDPM / `1.192092896e-7` DDIM; tests use rtol=2e-5, atol=3e-6.
Real B2 tolerances are rtol=1e-4, atol=1e-5; exact measured maxima are in receipts
(LIBERO `5.960464478e-7`, MW `3.427267075e-7`). B1 and debug comparisons require byte equality.
Warnings are Torch nested-tensor and legacy Swig deprecations, not skips.

Final artifact-load receipt: `samples/loader_bundle_receipt.json`, **exit 0,
8 checks passed / 0 skipped, 16.66s**. Both pinned LoadSpecs strictly loaded the
real 475-tensor checkpoints; each resident provider then validated its generated
K100/2/1 YAML bindings. These eight checks are separate from pytest and the 23
sampler probe checks above.

## Model-specific hook table

| Hook / capability | Implementation and tested scope |
|---|---|
| Strict load / provenance | E `Policy/DiffusionPolicy` unchanged; strict475 actual CPU; rejects latest/symlink, wrong digest, pooled text, missing complete formal export. Not an author-source reproduction claim. |
| Stage1 / CP1 keys | E image resize96/crop84/internal normalization, both FiLM encoders; exact post-ReLU64. LIBERO two keys/state8, MW one key/state4; zero branch remains in conditioning and cost. |
| Stage2 | Exact E feature-order concatenation, no LLM; MISS continues existing features and never re-encodes vision. |
| Stage3 / step reduction | Real diffusers DDPM100/K2/K1 and explicitly separate DDIM/eta. K builds an entire native grid; actual forward count recorded. |
| RNG / resume / reset | Sequential per-request initial+posterior stream, P1 semantic seed or explicit E seed. Snapshot validates model/norm/layout/grid/native point/conditioning/dtype/shape, restores RNG, runs exact suffix. Reset calls native add_noise. |
| Normalized storage / wire | Scheduler internal-affine action stored. Internal inverse then external quantile inverse, once each. CPU full/cache HIT parity; no gripper inversion, output clipping, padding or fictitious camera. |
| Concurrent / bundle / lifecycle | P1 factory + real two-client localhost WS K100/2/1 hot YAML; wrong norm/unknown bundle rejection, same-ID idempotence, episode/disconnect cleanup. Weights shared; mutable state isolated. GPU concurrency pending. |
| Batching | Stock StageBatcher protocol, real CPU queues, native profile/K/shape/capture/index buckets, private posterior noise, output/snapshot ownership, bad-shape bucket isolation; real CPU B2 math receipt. GPU stream/throughput pending. |
| Commit / tail | H16 permits configured 10/15. Actual acknowledgment advances cursor; absent/stale ack cannot silently consume tail. Short success P2 fake episode applies5+2 actual controls. |
| Five layers / library | P1 sole layout recipe + per-connection fork; synthetic LIBERO/MW libraries fit LOEO corrector/metric/guard/periodic and execute through factory. No private DP recipe math. Real teacher library / full replay qualification pending P6. |
| Debug / capture | P1 post-MISS sample_fields and native_snapshots; P6 DecisionSink NPZ/H5, integer t and uint8 posterior RNG, normalized/wire, actual ack, detached native conditioning/language/branch features. Ordinary capture observes only. |
| Explicit trace variants | Matched initial private stream, native sampler per variant, summed extra NFE tested; ordinary debug does not call it. Generic stock trace executor remains rejected pending certification. |
| CP2 / RIT / routing | Unsupported by generic executor, rejected at bundle validation. First native update captured with actual discrete semantics; no flow factor substitution. |
| Split CPU blueprint | JSON-only ModelSpec/config, no model/vendor/GPU tensors. `split_qualified=False`; no split runtime claim. |
| Flow-shoot / two-view MW ablation | `not_applicable`: discrete diffusion / one real MW camera. Native reset and explicit DDIM are the meaningful separate APIs. |

## Native/grid/norm/storage samples

`models/dp/samples/release_{libero,metaworld}_cpu/{receipt.json,capture/capture.json,
capture/capture.npz}` contain real computed data, not fabricated fixtures.
Capture arrays include `key_vision_0:float32[64]` (plus LIBERO vision1), raw
state, normalized action `[16,7/4]`, wire action, native_grid int64 `[50,0]`,
snapshot_x `[1,16,7/4]`, loop index `[1]`, integer timestep `[0]`, private RNG
bytes, first input/update, native conditioning/language and policy wrist feature.
MW's latter is model conditioning, **not** a real wrist retrieval key.

Default native grids: DDPM100 `99..0`; K2 `[50,0]`; K1 `[0]`.
`point_kind=native_integer_timestep`; any float batch warm point is opaque index
encoding only. DDIM config includes eta/set_alpha_to_one and its own identity.
Norm identity hashes external statistics and actual checkpoint affine tensors;
model identity binds weights, projected CLIP/tokenizer, feature order, code,
dependencies, preprocessing and absent-branch policy. Complete hashes, actual
versions and content pins are in receipt/manifest, not mtime-based identity.
Model code/contract/RNG are bound into model identity; framework registry,
serving, stock config/core source hashes are recorded separately as runtime
provenance. P1's ongoing registry/lifecycle changes do not change native model
math identity; the runtime still requires its own G0 release receipt/seal.

Generated `samples/{libero,metaworld}_load_cpu.json` and corresponding default
YAML show the exact pinned loader and P1 binding. Regenerate pins if the source
or final export changes; stale content must fail closed.
Six generated K100/2/1 YAML files were loaded/validated successfully against
actual pinned resident models; `samples/loader_bundle_receipt.json` records
the final eight checks. Counts, artifact digests and current model/runtime
source hashes are sealed in `samples/handback_summary.json`. Earlier intermediate receipts/XML are retained
for audit and are not the authoritative acceptance receipts above.

## GPU-ready commands — not executed

Coordinator must supply one freed GPU and immutable checkpoint, **one model at
a time**, after the training lanes finish. The commands below have an external
840s cap and internal830s cap (<15min); they load one checkpoint, no workers,
and record partial/failure instead of inventing a passing gate. They qualify
model math, capture and B2 only; benchmark/EGL/live debug smoke remains a separate
P2/P6 mechanical gate. No GPU authorization is inferred from these commands.

```bash
# Future LIBERO model qualification; DP_GPU is set by the coordinator.
timeout -k 5s 840s env CUDA_VISIBLE_DEVICES="$DP_GPU" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --device cuda --deadline-seconds 830 \
  --output exp/r17_integration/models/dp/samples/gpu_libero_qualification

# Separate later slot. Substitute E's final immutable MT50 export; current
# ML45 path below is diagnostic only, including the receipt's quality label.
timeout -k 5s 840s env CUDA_VISIBLE_DEVICES="$DP_GPU" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/metaworld --domain metaworld \
  --device cuda --deadline-seconds 830 \
  --output exp/r17_integration/models/dp/samples/gpu_metaworld_diagnostic
```

For a future live mechanical slot, use the same resident entry with a generated
GPU LoadSpec and an ops-owned reserved port:

```bash
timeout -k 5s 840s env CUDA_VISIBLE_DEVICES="$DP_GPU" PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.serve \
  --load-spec "$DP_GPU_LOAD_SPEC" --host 127.0.0.1 --port "$DP_OWNED_PORT" \
  --batch-size 2 --manifest-out "$DP_RUN/model_manifest.json" --yaml-out "$DP_RUN/full.yaml"
```

P2/coordinator supplies episode conductor and approved renderer lane; P6 injects
capture sink/recipe_factory for library+physical+closed writer evidence. Startup
budget0–3min, probes3–7min, native resume/B2 7–10min, one mechanical episode and
two bundle generations10–13min, flush/receipt13–15min. No default100→K2 switch
may be labeled full. The pure CLI exposes plain/cache execution; richer fitted
recipe/debug composition uses the tested factory API. No replica copies or
formal500 run in this slot. Stop only the service/worker receipts created there.

## Known limits / pending owners

| Pending item | Owner / condition |
|---|---|
| Final scratch MT50 immutable export + complete receipt/content pins | E + coordinator. Current ML4585k is a declared one-camera transfer diagnostic. No formal SR, low-step quality or source-camera reproduction claim. |
| GPU full/staged/default100/B2/streams/concurrency/per-stage pricing | P4 + coordinator's serial slots. CPU pass is not GPU numerical/latency/trajectory qualification. |
| Real MW/LIBERO mechanical episode + physical/EGL/client payload + complete join | P2 + P6 + coordinator; P4 provides canonical wire/metadata/factory/capture. No renderer or real episode ran here. |
| Real teacher library, warmup factors, fit/replay/analysis/replicas/operations coverage | P6 + P1; P4 native samples and hooks are ready. Synthetic five-layer tests do not establish policy quality. |
| Native WARM_START/reset payload through stock YAML executor; CP2/RIT/routing/stock twins | P1 executor + P4 model certification. Direct native adapter and batch resume/reset pass CPU; generic stock flow combinations remain explicitly rejected. Exact suffix snapshots bind conditioning; different-observation transfer uses explicit native reset rather than silently retargeting a suffix. |
| Full LIBERO suite protocol/checkpoint/taskdata content binding | P2/coordinator; current label is explicit, CPU two-view geometry is tested, four-suite real SR is unknown. |
| Split/compiled/replica GPU memory or throughput | P1/P6 + P4 GPU gate. CPU blueprint/proxy tests alone are not qualification. |

Requests P1→P4 and P6→P4 were checked at startup and repeatedly. P1 capture,
explicit-seed and shared contracts were consumed; P6 retained native capture
and execution ack. No unresolved interface request blocks this CPU handback.

## Fix round 1

2026-10-09. Read REVIEW_A and REVIEW_B fully, including the DP oracle and
real-frame repros. No disputed DP finding. **This section supersedes the old
leading-grid receipts, K1 quality implication, capability claims, CLI limits
and sample pins above.** Historical artifacts are retained for audit. No GPU,
renderer, service deployment, training/session change, commit/push, R16/harness
or E-source edit occurred.

### Finding → change → regression → status

| Finding | Owned change | Regression / evidence | Status |
|---|---|---|---|
| B3 major | Default DDPM100 unchanged. Reduced auto profiles use epsilon DDIM, eta0, trailing spacing, first timestep99. Explicit reduced DDPM and leading DDIM rejected. Every K1..100 has exactly K points; update uses actual next-grid alpha even for nondivisor K. Sampler v2 binds spacing/grid/update rule. K1 remains available and explicitly `quality_status=lossy`. | `test_sampler.py`: every K, divisor equality with diffusers, nondivisor K3/7/13/99 updates, reviewer MMSE oracle, old-grid identity rejection. Default100 exact E tests for both domains and real strict475 probes.48 reviewer frames / K1,2,4,10,100 proxy below. | Fixed math and labelling. Low-K benchmark quality/descent gate remains unqualified. |
| B5 minor | Reject LIBERO224 or arbitrary dimensions; only paired96/raw256 uint8 RGB of matching size. Wire sizes bind FeatureSpec preprocessing digest. Optional sealed P2 `LiberoBenchmarkSpec` documents bind exact taskmap/protocol and chosen wire size; formal LIBERO loads require them. | `test_libero_rejects_resampled_wire_and_binds_expected_size`; `test_LIBERO_wire_size_is_bound_to_P2_benchmark_identity`, with96/256 mismatch rejection. | Fixed owned adapter; formal taskmap supplied by P2/coordinator. |
| B7 minor | Native warm_resume/reset and library/recipe set `implemented_unverified`; unsupported generic advanced executors stay unsupported. k_step reason explicitly says reduced quality unqualified / K1 lossy. | `test_capabilities_do_not_claim_generic_warm_library_recipe_qualification` for both domains verifies `Capabilities.require` rejects unqualified claims. | Fixed metadata. CPU recipe hooks are exercised without claiming full G3 coverage. |
| B8 minor | Serve CLI exposes `--recipe-factory`, `--capture-sink`, and `--debug-directory`. All use P1 factory and P6 writer. Default factory consumes per-bundle `r17.recipe.v1`; teacher sampler independent of MISS K; disabled recipes stay plain. Debug writer closes on service return. | Four actual-entrypoint composition tests: K100/K2 × callback/closed directory. Shared five layers, MISS NFE, native numeric capture, actual3-step terminal ack, writer flush and pricing exclusion. Real sealed K100 teacher → K2 recipe integration test. | Fixed CPU deployment composition. CLI tests replace only model loading/socket start, not framework/recipe/writer. |
| B9 minor | Conditioning digest computed once per request before snapshot loop; resume validates once. Ordinary capture still clones/synchronizes numeric data. | K100 test asserts99 snapshots / one conditioning digest, no digest in uncaptured sampling, identical action/RNG, exact suffix. CLI tests assert capture_enabled and pricing_eligible=False from P1 ledger. | Fixed owned repeated digest. Captured costs excluded from pricing; no GPU latency claim. |
| B11 minor | New read-only `fixtures.libero_observation` loads actual independent png/wrist_png from prepared96 SQLite, raw state and task prompt, with manifest/index/PNG hashes. Probe and quality comparison use this helper. | Real-pair adapter regression changes only wrist and proves base key unchanged/wrist key changes; exact E native condition. Real475 LIBERO probe on recorded pair:13 pass. Quality uses same48 real paired reviewer frames. | Owned probe fixed. External `qualification/model_runtime.py` still fabricates wrist at final check; request P4_to_P6 asks Q/coordinator to consume helper/preserve real pair. |
| B1/B2 cross-package; A7/A9 | Consume P1 strict context/actual acknowledgment checks, terminal ack and original-index alias; consume P2 sealed LIBERO spec when supplied. Test fixtures now declare complete identity and ack protocol rather than hiding missing data. | Old LIBERO client without ack/context rejected; valid native tail exact. Actual localhost WS hot switch uses terminal ack. P2 fake MW episode still executes5+2 and acknowledges actual controls. | Shared fixes consumed. Formal LIBERO environment/conductor/worker and official taskmap remain P2/coordinator qualification. |
| A2 cross-package | P4 config/server imports migrated to local `common.config` / `common.websocket_policy_server`. No dependence on edits in the four frozen stock files; stock registry and batch core remain reused. | Full79 tests and legacy/default-off48; `git diff --numstat` for the four sealed sources was empty. P1 reports its R16 verify separately. | P4 migration complete; no claim P4 ran R16 analysis. |
| A3/A6 cross-package | Consume P1 bundle-specific recipe identity/fit/teacher sampler and cost/capture eligibility. Export `r17_sampler_specs` for K1/2/4/10/100 plus default, so workers need not guess sampler grids. | Disabled K2 recipe remains None; K100 teacher works at K2; CLI K100 recipe and terminal3-step capture; shared capture pricing exclusion. | CPU composition passes; real paired GPU pricing and worker physical binding remain P2/P6/coordinator. |

Two extra scheduler defects were found by the new every-K tests: diffusers'
floating `arange` can produce K+1 points at K29, and DDIM's `t-T//K` update uses
the wrong preceding alpha for nondivisor K. P4's local native scheduler hook
fixes both without modifying diffusers or E. For divisor K its DDIM arithmetic
and private eta-noise state are byte equal with diffusers.

Owned changes are in `models/dp/{sampler,provider,serve,staged,manifest,config,
loader,wire_policy,probe}.py`, new `fixtures.py` and `quality.py`, README,
7 test modules plus isolated negative-control runner, current samples/receipts,
P4 request files and this handback.17 implementation modules total. Shared
interfaces remain **r17.contracts.v1**; sampler schema is **r17.dp.sampler.v2**.

### Corrected action proxy on the reviewer's frames

`models/dp/samples/fix_round1_quality.json` records all48 indices, real paired
camera PNG hashes, full grids/config/eta and code/weight identities. Indices are
exactly `np.random.default_rng(0).integers(len(ds),size=48)`. Noise is the
reviewer's single seed0 sequential B48 generator, with the same initial draw
for all profiles. Adapter preprocessing/condition equals E byte exactly;
DDPM100 output equals E's untouched full sampler byte exactly.

MSE units below match the reviewer: **external quantile-normalized actions**
(internal affine reversed), averaged over48 frames × first5 actions ×7 channels.
Additional all16 and physical-wire MSE are in the receipt.

| K | Profile / native grid | MSE vs DDPM100, first5 | MSE vs dataset, first5 | Internal clip fraction |
|---|---|---:|---:|---:|
| 1 | DDIM trailing `[99]`, **lossy** | 1.767129898 | 1.948036551 | 0.983444929 |
| 2 | DDIM trailing `[99,49]` | 0.020039799 | 0.228040382 | 0.056547619 |
| 4 | DDIM trailing `[99,74,49,24]` | 0.015954781 | 0.235429242 | 0.079241075 |
| 10 | DDIM trailing `[99,89,79,69,59,49,39,29,19,9]` | 0.018993597 | 0.239900112 | 0.100446425 |
| 100 | Default DDPM `[99..0]` | 0 | 0.238635078 | 0.106026784 |

K1 has a valid starting grid and clearly poor proxy quality; it remains labelled
lossy as requested. These are **training frames**, not held-out or closed-loop
SR evidence. K2/4/10 remain unqualified for formal descent until a separate
held-out/benchmark quality gate. No proxy cutoff was invented to declare them
qualified. There are no reviewer MW frames; MW native math/default parity uses
its diagnostic real checkpoint probe, not an invented MW quality result.

### CPU commands and current receipts

All commands ran from repo root with this CPU-only environment:

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.:src:packages/openpi-client/src
```

```bash
/data/mw_finetune/dp/env/bin/python -m pytest \
  exp/r17_integration/tests/models/dp -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_fix_round1_delivered \
  --junitxml=exp/r17_integration/models/dp/cpu_fix_round1_delivered.xml -q --tb=short
```

**Exit0:79 passed,0 skipped/failures/errors,39.854s.** Includes recorded paired
views, K100 E byte parity, real native schedulers, B2, suffix RNG, CLI hooks,
sealed teacher libraries, P6 writer, P2 fake episode and localhost hot YAML.

```bash
/data/mw_finetune/dp/env/bin/python -m pytest \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py tests/serving/test_replica_proxy.py \
  exp/r17_integration/tests/common/test_default_off.py --run-manual \
  -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_fix_round1_legacy \
  --junitxml=exp/r17_integration/models/dp/cpu_fix_round1_legacy.xml -q --tb=short
```

**Exit0:48 passed,0 skipped/failures/errors,14.349s.**47 existing serving tests
plus the shared HEAD/default-off config/error/metadata/ack/action-byte probe.
The manual tests here are fake localhost backends.127 distinct acceptance
pytest tests; repeated development runs are not added to this total.

```bash
# Sequential domains; no worker/renderer. Each below actually ran on CPU.
timeout -k 5s 300s /data/mw_finetune/dp/env/bin/python \
  -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/fix_round1_libero_cpu
# Same command: metaworld checkpoint / domain metaworld /
# output .../samples/fix_round1_metaworld_cpu.
```

LIBERO **exit0:13 checks passed/0 skipped,37.511s**; MW **exit0:13 checks
passed/0 skipped,35.401s**. These26 probe checks are separate from pytest and
not SR. Both actual checkpoint loads strictly verify475 tensors. LIBERO uses
its true paired frame; only K100 compares E's sampler/wrapper, while corrected
low-K compares an independent native loop. Capture/no-capture and resume remain
byte exact. Real B2 tolerance and measured maxima are in receipts; RNG exact.

```bash
# Overrides threads only for the reviewer's real B48 offline proxy; still CPU.
timeout -k 5s 600s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.quality \
  --output exp/r17_integration/models/dp/samples/fix_round1_quality.json

# Actual pinned resident reload + all10 generated YAML bindings.
timeout -k 5s 90s /data/mw_finetune/dp/env/bin/python \
  exp/r17_integration/models/dp/samples/fix_round1_validate_samples.py
```

Quality **exit0,48 frames / five K profiles,100.100s**. Pinned serving artifacts
**exit0:12 checks passed/0 skipped,16.251s**: two actual pinned model loads plus
10 K1/2/4/10/100 resident YAML validations.

Negative control (not acceptance):

```bash
/data/mw_finetune/dp/env/bin/python \
  exp/r17_integration/tests/models/dp/run_fix_round1_negative.py
```

**Intentional exit1:6 regression failures/0 skips,3.732s**, captured in
`cpu_fix_round1_negative.xml`. In an isolated process it recreates reviewed
leading grid, per-step digest, overclaimed capabilities, arbitrary wire size,
fake wrist and absent CLI flags without editing source. Each targeted regression
fails on that fault; the same regressions pass in the79-test delivered suite.
This is a fault recreation, not a claim that an archived checkout was executed.

`serve --help`, AST parsing25 owned Python files, safe numeric NPZ loading and
`git diff --check` also passed. Torch nested-tensor/legacy Swig warnings are
warnings, not skips. Initial failing development XML remains historical evidence.

### Current samples, provenance and pending work

Authoritative release seal: `models/dp/samples/fix_round1_summary.json`. It
contains current model/framework source hashes, all pass counts, quality
metrics and artifact hashes. No model-source drift was found between the real
quality/CPU probe receipts and the seal. Runtime changes still require P1's
release seal; model hash is not a substitute for framework certification.

Use `samples/fix_round1_{libero,metaworld}_cpu/receipt.json`, their numeric
`capture/{capture.json,capture.npz}`, pinned `load_cpu.json`, and `k1/k2/k4/k10/
k100.yaml` files. `fix_round1_loader_bundle_receipt.json` verifies the pins and
all10 bundles against real resident models. Earlier `release_*`, old bare
`*_load_cpu.json`, leading-grid YAML and XML are **superseded**; old source/grid
identities fail closed against this release.

Current deployment arguments (future service only, **not executed**):

```bash
python -m exp.r17_integration.models.dp.serve \
  --load-spec PINNED_LOAD_SPEC.json --cache-config BOUND_RECIPE.yaml \
  --host 127.0.0.1 --port "$DP_OWNED_PORT" --debug-directory "$DP_RUN/debug"
# Optional explicit hooks: --recipe-factory module:function or
# --capture-sink module:function (mutually exclusive with --debug-directory).
```

The previously listed ≤15-minute, single-model GPU probe commands remain
**not executed**. They now use corrected reduced DDIM and true paired LIBERO
fixture. Coordinator must reseal GPU LoadSpecs/config/specs and provide a freed
GPU; GPU, renderer, physical joins, formal500, policy quality and throughput
remain unqualified. Current MW weights remain ML45 diagnostic, pending E's
immutable final MT50 export. Formal LIBERO requires P2's official taskmap and
sealed selected wire protocol, then a real renderer/worker gate; this island
has no installed official LIBERO package and does not infer a taskmap from
training-frame prompts.

P4_to_P1/P2/P6 requests record all cross-package dependencies. P1 local
config/server, recipe, strict context/ack and ledger interfaces were consumed;
P2 benchmark specs and actual MW runner were consumed. **Pending B11 external
qualification fixture**: Q/coordinator/P6 must replace the synthetic wrist in
`qualification/model_runtime.py` using P4's real helper or sealed paired NPZ.
P4 did not edit that unowned file. Native warm/reset/trace/CP2/RIT executor
certification, real teacher dataset and low-K held-out gate remain with their
previous owners. No unresolved P4 code defect is concealed as a passing GPU,
renderer or formal-quality capability.

## Fix round 2

Read REVIEW_A2 and REVIEW_B2 completely; reviewed the real DP low-K repro.
Completed on 2026-10-09 in the shared Ziyang checkout. **Zero GPU use**,
no commit/push, no training/process/session operations, no E/R16/harness edits.
All P4 edits stay in the owned model/tests directories, P4 requests and this
handback. P1/P2/Q made their own shared/benchmark/qualification changes.

### Findings → changes → regressions → status

| Finding | Change | Passing regression/evidence | Status |
|---|---|---|---|
| B2 N6 / B3 follow-up: K1 unusable and previously admitted everywhere | New model admission callbacks reject K1 by default. Require strict `allow_lossy_sampler: true` **and** nonempty `sampler_study_label`; only plain step reduction at commit5. Enabled CP1/cache MISS, recipe, library/teacher, warm/trace/routing/shadow or extended commit cannot combine with the study. K1 teacher rejected with K2 MISS too; inspect small sealed READY.json teacher descriptor to reject other-eta K1 identities. Native DDIM arithmetic stays available for explicitly invoked offline studies. | `test_dp_fix_round2.py`: shared binding/config/factory tests in both domains, malformed flags/labels, forbidden roles, other-eta teacher, labelled one-NFE capture parity, default DDPM100. `test_dp_serving.py`: real local WS refuses unlabelled K1 and flagged cache MISS before registry publication; valid bundles remain usable. Actual pinned sample check18/18. | **Fixed, CPU verified.** K2/K4/K10 remain admitted. K1 has no production/cache/teacher/descent qualification. |
| B2 N6 descent admission | Metadata publishes `r17_sampler_admission`: descent steps2/4/10/100, minimum cache MISS/teacher/recipe K2, K1 plain-study role. P2 propagates study flags/label/quality marker and rejects marked episodes from dev descent. | `test_N6_bound_K1_study_markers_rejected_by_P2_dev_descent` consumes a real DP binding and P2 strategy; K1 study rejected, K2 accepted. Shared wire and capture diagnostics retain study label and `lossy_study`. | **Fixed through P1/P2, CPU verified.** |
| B11 partial / B2 N1(d): qualification fabricated wrist | Owned fixture writer/reader seal three real recorded96 base+wrist views into numeric/string NPZ, preserving raw state/prompt and independent PNG hashes. Missing wrist fails; no flipping, resizing or synthetic fallback. Q's current runtime reader preserves the actual wrist key. | `test_B11_qualification_fixture_uses_recorded_pair_verbatim` checks both owned and actual Q runtime readers against recorded PNGs, then E conditioning equivalence. `test_B11_missing_recorded_wrist_fails_instead_of_synthesizing` exercises both readers. Fabricated-wrist fault makes the regression fail. | **Owned fixture and Q consumer CPU verified.** Full Q asset/raw256/GPU slot qualification remains Q/coordinator scope. |
| A2 N1/N2 consumption in P4-owned recipe fixtures | Replace the former target_ir=.25 shortcut fixture with explicit periodic_gap3. Enabled recipes carry the complete canonical ordered table and table identity; synthetic teacher rows use correct global IDs, including reach-v3 index43. Supplied P2 LIBERO specs register explicitly at adapter construction. | Both-domain five-layer integration, full/K2 CLI capture hooks, shared recipe fitting, and actual P2 MW runner tests pass in the full suite. P1's optional request task-table field is consumed without stripping identity evidence. | **Fixed for P4 consumers.** IR calibration itself belongs to P1; no new GPU cost claim. |

### Owned changes and interfaces

New `models/dp/admission.py` supplies model-specific callbacks; `provider.py`
registers them through P1's additive optional `AdapterProvider.binding_validator`
and `config_validator` hooks plus teacher artifact validation. P1 calls them
at binding construction/validation and config validation before hot publication
or connection construction. P4's factory also validates its default config.
There is **no remaining interface shim or shared monkeypatch**.

`serve.py` exposes study binding options through `plain_config`; `probe.py`
generates a labelled plain K1/commit5 YAML and ordinary K2/4/10/100 bundles.
`wire_policy.py` publishes admission policy; `staged.py` reports K1's restriction
and registers explicitly supplied P2 specs. `fixtures.py` adds the sealed pair
writer/reader; README documents admission and canonical recipe requirements.
Owned test changes are in conftest, interop, serving, fix-round1 fixtures, new
`test_dp_fix_round2.py` (**35 regressions in the final suite**), and the isolated
`run_fix_round2_negative.py`. Current samples and checks are under `samples/`.
Requests `P4_to_P1`, `P4_to_P2`, `P4_to_P6` record ownership handoffs/resolutions.

Interface remains **r17.contracts.v1** / adapter_binding.v1 / sampler.v2 with
additive default-off callback/study fields. No flag or label appears in normal
bindings. RequestContext's optional canonical table identity is excluded from
noise identity by P1. Default remains E's stochastic DDPM100; reduced profiles
remain deterministic trailing DDIM unless an explicit eta baseline is chosen.
Admission changes do not alter grid construction, native updates or RNG draws.

| Model/benchmark hook | Framework consumer | Current receipt / limit |
|---|---|---|
| Sampler binding/config admission | P1 binding, config, shared connection and hot-YAML validator | Ordinary K1 and cache/teacher/recipe K1 rejected; labelled plain K1 works; K2+ unchanged |
| Study label / lossy role metadata | P1 wire/capture diagnostics and P2 episode/descent dispatch | Label retained in real inference/capture; marked dev descent rejected |
| Real paired qualification fixture | Owned probe/quality plus Q runtime reader | Actual independent camera arrays byte exact; no missing-wrist fallback |
| Complete canonical task table | P1 recipe and P2 benchmark/request envelope | Both-domain synthetic recipes and real P2 fake-env runner pass; official LIBERO renderer still pending |
| Full/staged/RNG/capture/batcher hooks | Existing shared path, unchanged native sampler | Default DDPM100 E byte parity, low-K math, capture on/off, batch/resume included in full suite; GPU unqualified |

### Default-off and actual CPU receipts

Authoritative pytest receipts (all CPU, no skipped tests):

| Receipt | Result | Time |
|---|---|---|
| `models/dp/cpu_fix_round2_final_current.xml` | **114 passed, 0 skipped/failures/errors** |59.717s|
| `models/dp/cpu_fix_round2_legacy_current.xml` | **49 passed, 0 skipped/failures/errors** |20.772s|
| `models/dp/cpu_fix_round2_negative.xml` | **3 expected regression failures, 0 skips/errors, intentional exit1** |2.136s|

The negative process recreates the old missing admission callbacks (two-domain
failures) and fabricated wrist (one failure), without changing source. Each
same regression passes in the final suite. This is an isolated fault recreation,
not a claim that an archived checkout was run. The earlier development XMLs
retain real failures while shared interfaces/fixture IDs were being updated;
`cpu_fix_round2_delivered.xml` is **interim**, not acceptance evidence.

Native sampler/staged/batcher subset52 passed and focused round2 suite34 passed
before the final added descent witness; those overlap the final114 and are not
additional distinct passes. Distinct final accepted pytest count is **163**.
Tests include fresh default-off import/registration proofs, staged default
DDPM100 byte equality with E in both domains, RNG and debug on/off byte parity,
stock serving identity/recorder/router and localhost replica proxy behavior.
The four sealed stock config/types/storage/server sources still compare to
HEAD (`git diff --exit-code`, exit0). No default flags or legacy source edits.
AST parsing28 owned module/test Python files, `serve --help` and
`git diff --check` also passed.

Actual commands, run from repository root (all targets/output paths owned by
P4 except read-only legacy/shared test targets):

```bash
P4_CPU=(env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH=.:src:packages/openpi-client/src /data/mw_finetune/dp/env/bin/python)
"${P4_CPU[@]}" -m pytest exp/r17_integration/tests/models/dp \
  -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_fix_round2_final_current \
  --junitxml=exp/r17_integration/models/dp/cpu_fix_round2_final_current.xml -q --tb=short
"${P4_CPU[@]}" -m pytest tests/serving/test_policy_identity.py \
  tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  tests/serving/test_replica_proxy.py exp/r17_integration/tests/common/test_default_off.py \
  --run-manual -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_fix_round2_legacy_current \
  --junitxml=exp/r17_integration/models/dp/cpu_fix_round2_legacy_current.xml -q --tb=short
"${P4_CPU[@]}" exp/r17_integration/tests/models/dp/run_fix_round2_negative.py
# Negative command intentionally exits1: all3 faults detected.
"${P4_CPU[@]}" -m exp.r17_integration.models.dp.probe --domain libero \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/fix_round2_release_libero_cpu
"${P4_CPU[@]}" -m exp.r17_integration.models.dp.probe --domain metaworld \
  --checkpoint /data/mw_finetune/dp/checkpoints/metaworld --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/fix_round2_release_metaworld_cpu
"${P4_CPU[@]}" exp/r17_integration/models/dp/samples/fix_round2_validate_samples.py
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 600 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.quality \
  --output exp/r17_integration/models/dp/samples/fix_round2_release_quality.json
```

Probe commands actually had an outer `timeout300`; both loaded **475 tensors**
strictly and passed **13 checks, 0 skips**: LIBERO35.781s, MW37.900s. Pinned
load plus ten bound YAML and forbidden K1 cases passed **18 checks, 0 skips**,
16.068s (`fix_round2_loader_bundle_receipt.json`). These are CPU mechanics and
diagnostic model receipts, not benchmark SR or GPU qualification.

### Samples and study data

Use `samples/fix_round2_release_{libero,metaworld}_cpu/receipt.json`, their
`load_cpu.json`, native numeric captures and K1/2/4/10/100 YAML files. Current
model-source seal covers28 dependencies; older source pins fail against this
release. K1's actual YAML has:

```yaml
adapter_binding:
  # Exact provider/model/norm/layout/feature/sampler/protocol pins also present.
  num_steps: 1
  commit_horizon: 5
  allow_lossy_sampler: true
  sampler_study_label: DP_K1_plain_step_reduction_only
checkpoints: {}
recipe:
  enabled: false
```

`samples/fix_round2_qualification/wire_observations.npz` and sibling JSON seal
three real recorded96 training pairs. Its NPZ SHA is
`sha256:f838079a39027334eae18b8d09e05ce460a380d67af4200fc12e6c59ffa8438e`.
`synthetic=false`, real_paired_views=true; source/frame/shard/row and two PNG
hashes are retained. No original256, real-reset or held-out SR claim.

`fix_round2_release_quality.json` reruns the reviewer's48 real paired frames,
B48 seed0 noise and untouched E DDPM100 reference with current source seal.
First5-step MSE in quantile-normalized action units remains:

| K | Native schedule | MSE vs DDPM100 | Internal clip fraction |
|---|---|---:|---:|
|1|DDIM `[99]`|1.767129898|98.3445%|
|2|DDIM `[99,49]`|0.020039799|5.6548%|
|4|DDIM `[99,74,49,24]`|0.015954781|7.9241%|
|10|DDIM trailing|0.018993597|10.0446%|
|100|DDPM full leading grid|0|10.6027%|

Baseline and reduced profiles differ in both step count and sampler family:
stochastic DDPM100 versus deterministic DDIM. This proxy supports the K1 gate;
it does not measure SR, certify K2+ held-out quality or authorize dev descent.
The earlier `fix_round2_quality.json` was run before the final P1 optional
request field and provider shim removal; retain as historical, use the release
receipt and `fix_round2_summary.json` for current source/artifact hashes.

Release quality command exited0 in103.881s, with all48 real pairs and baseline
E byte parity. `fix_round2_summary.json` verifies **zero model-source drift**
across both real probes and quality. P1 recipe.py changed after the probes;
their exact older framework hashes remain intact. The full114 and legacy49
suites were rerun after that change with **zero framework-source drift during
the rerun**, captured in `fix_round2_runtime_under_test.json`. The summary
records that one recipe-source delta,28 model-source dependencies, ten current
runtime framework hashes, consumed canonical/descent/Q reader sources,
26 artifact hashes and the actual XML/probe/bundle counts above.

### GPU-ready commands, NOT EXECUTED

Coordinator runs only after releasing and allocating a GPU, one model at a
time. Each native mechanics command has an840s internal deadline and900s hard
timeout (≤15min). Keep LIBERO/MW separate; do not load both models on a GPU.
Commands use the existing diagnostic weights, not final MT50 or formal SR.

```bash
# Later allocated GPU only; P4_QUAL_GPU must be supplied by coordinator.
env CUDA_VISIBLE_DEVICES="$P4_QUAL_GPU" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 900 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --domain libero --checkpoint /data/mw_finetune/dp/checkpoints/libero \
  --device cuda:0 --deadline-seconds 840 \
  --output exp/r17_integration/models/dp/samples/gpu_fix_round2_libero
# Separate later slot, after the first command has exited:
env CUDA_VISIBLE_DEVICES="$P4_QUAL_GPU" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 900 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --domain metaworld --checkpoint /data/mw_finetune/dp/checkpoints/metaworld \
  --device cuda:0 --deadline-seconds 840 \
  --output exp/r17_integration/models/dp/samples/gpu_fix_round2_metaworld
```

Full concurrent/YAML/real renderer qualification remains the coordinator's
following slot using freshly GPU-sealed LoadSpecs and P2's protocol/worker.
Native probe success alone does not certify those framework/renderer gates.
K1 SR may be measured only in its explicitly labelled plain study, separate
from cache MISS/teacher/default recipe/dev-descent arms.

### Known limits and pending owners

- **P1/P2 request interfaces resolved:** admission callbacks, study diagnostics,
  request table field, canonical recipe tables and descent study isolation were
  consumed and tested. No owned round2 defect remains pending.
- **Q/coordinator:** full qualification asset pass-count/shape/raw256 migration,
  actual GPU receipts and real LIBERO renderer/SR. Current Q runtime pair
  preservation is CPU tested by P4; P4 did not edit qualification files.
- **E/coordinator:** immutable final MT50 export and formal weight qualification;
  current MW checkpoint remains ML45 diagnostic.
- **P1/P6/model owners:** advanced native warm/reset/CP2/RIT/twins/shadow/routing
  executor certificates, real teacher libraries, physical joins, replica/batch
  throughput and production GPU costs remain as previously unqualified.
- **P2/coordinator:** official LIBERO task-map/renderer gate and any K1 plain SR
  study. Independent recorded training cameras are CPU fixtures, not episodes.

## GPU fix 1

2026-10-09. Implemented `REQUESTS/Q_to_P4.md` in owned code only. No GPU,
qualification-file edits, running-process actions, commits or pushes. Reviewer
B3's `protocols.py`, staged protocol registration and eleven protocol regression
tests remain unchanged. Historical `review_b3_release_*` artifacts are retained;
this probe edit changes the source-bound identity, so they require resealing.

### Owned files, diff and interfaces

| Owned file / hook | Change and verification |
|---|---|
| `models/dp/probe.py`: `--load-spec PATH` | Load the sealed JSON through the existing `load_adapter`; override only `device`. Preserve domain, checkpoint, text encoder, sampler, diagnostic/formal mode, P2 protocol documents and all three pins. Optional checkpoint/domain arguments must agree. Receipt reports the loaded adapter's actual diagnostic flag. |
| `models/dp/probe.py`: `--observations PATH` | Read no-pickle NPZ row0 verbatim; require aligned nonempty rows, Unicode prompt and recorded LIBERO wrist. Support canonical MW keys and E's aliases, including native reference parity. Without the options, retain checkpoint/domain CLI and existing standalone fixtures. |
| `tests/models/dp/test_dp_gpu_fix1.py` | Nine regressions: actual strict475 official wire256 paired-input native probe; formal descriptor/receipt boundary; original CLI; two conflicting argument cases; four invalid NPZ cases. |
| `tests/models/dp/run_gpu_fix1_negative.py` | Isolated missing-option mutation reproduces the old API defect and causes the descriptor regression to fail; no implementation source is modified. |
| `models/dp/README.md`, `cpu_gpu_fix1_*.xml`, `samples/gpu_fix1_*` | API documentation, real test evidence, current pinned native examples, source/runtime/artifact audit and explicit owned reseal of Q's LIBERO descriptor. |
| `REQUESTS/P4_to_{P1,P6}.md`, this handback | API-ready notice and Q/coordinator asset/runtime reseal handoff. Current Q wrist-preserving reader checked read-only; no reader change requested. |

Interface stays **`r17.contracts.v1`** / existing LoadSpec and
`r17.dp.qualification.v1`; receipt provenance fields are additive. There is no
new server, scheduler, protocol parser or qualification executor. Existing
framework paths, default DDPM100, trailing DDIM and K1 admission remain intact.

| Finding / request | Change → regression | Status |
|---|---|---|
| Q sealed LIBERO protocol differs from standalone probe | Descriptor passed unchanged except CPU device; actual official four-suite40-task wire256 spec + Q paired NPZ run all13 checks and emit exactly the resealed ModelSpec. Fallback96 helper is patched to fail if selected in this test. | **Fixed; CPU pass.** |
| Q paired input API missing | Both row0 cameras and state compared byte exactly with Q's NPZ. Missing wrist, empty/misaligned rows and pickle-dependent prompts reject; no fabricated wrist path. | **Fixed; CPU pass.** |
| Q formal MW descriptor/mode must survive | Boundary regression supplies diagnostic=False, all weight/norm/model pins and sealed text encoder; asserts only device changes and receipt diagnostic=False. | **Fixed API; CPU boundary pass. Actual remote MT50 qualification pending Q.** |
| Existing CLI / reviewer B3 preservation | Original CLI boundary regression and two real fallback probes pass; all11 B3 protocol tests included in full suite. Conflicting optional arguments reject. | **Fixed; CPU pass.** |

### Default-off proof and CPU commands actually run

All inference/tests ran with CUDA hidden and CPU selected. Define the command
prefix used for these runs from the repository root:

```bash
p4_cpu() {
  env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=.:src:packages/openpi-client/src \
    /data/mw_finetune/dp/env/bin/python "$@"
}
p4_cpu -m pytest exp/r17_integration/tests/models/dp -p no:cacheprovider \
  -o junit_family=legacy --basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix1_full \
  --junitxml=exp/r17_integration/models/dp/cpu_gpu_fix1_full.xml -q --tb=short
p4_cpu -m pytest tests/serving/test_policy_identity.py \
  tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  tests/serving/test_replica_proxy.py exp/r17_integration/tests/common/test_default_off.py \
  --run-manual -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix1_legacy \
  --junitxml=exp/r17_integration/models/dp/cpu_gpu_fix1_legacy.xml -q --tb=short
p4_cpu exp/r17_integration/tests/models/dp/run_gpu_fix1_negative.py
```

| Evidence | Pass / skip / failure | Actual elapsed / exit |
|---|---|---|
| `cpu_gpu_fix1_full.xml` | **134 / 0 / 0** | 84.951s / 0 |
| `cpu_gpu_fix1_legacy.xml` | **49 / 0 / 0** | 15.641s / 0 |
| Targeted nine new regressions, `cpu_gpu_fix1_targeted.xml` | 9 / 0 / 0, included in134 | 33.469s / 0 |
| Missing-flag mutation, `cpu_gpu_fix1_negative.xml` | 0 / 0 / **1 expected failure**, 8 deselected | 0.169s / intentional1 |

Accepted distinct pytest total: **183 passed, zero skips/errors/failures**.
The mutation removes argparse registration of the two new options in isolation;
the formal-descriptor regression fails with unrecognized `--load-spec`. This is
a scoped reproduction, not a historical-checkout test. Default-off suite tests
shared HEAD parity of config/errors/metadata/acks/action bytes. Full DP includes
E DDPM100 parity, private RNG/capture on/off, default-off import/registration,
localhost bundle switching and framework hooks. The four frozen stock
config/types/storage/server files compare byte-for-byte to HEAD (exit0).

### Current samples and real native receipts

```bash
p4_cpu -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --load-spec exp/r17_integration/models/dp/samples/gpu_fix1_q_libero_seal/load.json \
  --observations /data/mw_finetune/qualification/assets/dp_libero_fix2_b3a3_v1/wire_observations.npz \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix1_q_libero_cpu
p4_cpu -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix1_release_libero_cpu
p4_cpu -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/metaworld --domain metaworld \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix1_release_metaworld_cpu
p4_cpu -m exp.r17_integration.models.dp.probe \
  --load-spec exp/r17_integration/models/dp/samples/gpu_fix1_release_metaworld_cpu/load_cpu.json \
  --observations exp/r17_integration/models/dp/samples/gpu_fix1_mw_canonical_fixture/wire_observations.npz \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix1_mw_canonical_cpu
p4_cpu exp/r17_integration/models/dp/samples/fix_round2_validate_samples.py gpu_fix1_release
```

Each actual native command also had an outer300s timeout. All four strictly
load475 tensors and pass **13 checks / 0 skips**: Q wire25628.214s;
standalone LIBERO29.969s; standalone MW27.964s; canonical-key MW28.916s.
Pinned loaders, ten bound YAML and negative K1/cache/teacher cases pass
**18 checks / 0 skips**,12.357s. These counts are separate from pytest totals.

Q's actual recorded NPZ contains three paired256 RGB frames and state8/prompt.
Its SHA256 is `sha256:4c480ecbb04193a38da09d85c24431547537cdf1150243d6067c019c8170db4a`.
The native probe intentionally uses row0, with the existing B2 prompt variant;
it does not certify all three frames or a closed-loop episode. Canonical MW's
owned NPZ is explicitly **synthetic**, used only for alias/reference mechanics;
both MW native examples use the old diagnostic ML45 checkpoint. No MW real
renderer, formal MT50 load, SR or action-quality claim is made by these runs.

Owned `gpu_fix1_q_libero_seal/load.json` preserves Q's original descriptor
including CUDA device; explicit resealing changes only expected_model_identity
to `sha256:777919b5255dcb6bfa6f071fedfefd48d200659ebe0e26956c81fbabb45b8d69`.
The actual CPU probe overrides only device and emits exactly the owned sealed
ModelSpec. Q's official protocol remains
`589fd0293f29d91fe3f011121b5855d550b9d06288ee67ce12aeae50e0d3afd0`.
The seal receipt records old/new pins and unchanged source descriptor hash.
Probe code never strips or relaxes a stale pin.

`gpu_fix1_runtime_under_test.json` and `gpu_fix1_summary.json` record29 model
dependencies, ten framework source digests, current test/probe counts and
artifact hashes. All four native manifests and test-time snapshot match the
final sources with **zero model/framework drift**. No reviewer B3 source was
replaced. Native captures retain integer DDPM/DDIM grids, snapshots and RNG;
K1 YAML is still an explicitly labelled plain lossy study at commit5. Normal
deployment remains DDPM100; K2/K4/K10 remain available. Prior quality MSEs are
historical evidence, not rerun or re-certified in this API fix.

### GPU-ready command, NOT EXECUTED

Q/coordinator must first refresh the asset descriptor, sealed ModelSpec and
runtime/payload hashes against this source. Allocate a free GPU and run one
model at a time; each command is bounded to≤15min. Supply current paths in
`P4_SEALED_LOAD`, `P4_PAIRED_OBS`, `P4_GPU_OUTPUT` and allocated index in
`P4_QUAL_GPU`. LIBERO must use its official wire256 paired asset. For formal MW,
Q must use its immutable MT50 descriptor and real sealed observation NPZ on
H100; the local diagnostic MW example cannot substitute for that gate.

```bash
# Future coordinator slot only. Not executed by P4.
env CUDA_VISIBLE_DEVICES="$P4_QUAL_GPU" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 900 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --load-spec "$P4_SEALED_LOAD" --observations "$P4_PAIRED_OBS" \
  --device cuda:0 --deadline-seconds 780 --output "$P4_GPU_OUTPUT"
```

### Known limits and pending owners

- **Q/coordinator:** reseal assets/runtime/payload, retry blocked DP GPU slots,
  supply the remote immutable MT50 checkpoint and collect actual GPU/renderer
  receipts. P4's owned wire256 CPU example is ready; Q assets were not edited.
- **P1/P6:** reseal handoff is recorded in P4_to_P1/P6 for forwarding to Q;
  there is no shared interface change or shim needed.
- **P2/Q/coordinator:** real episode/SR, official renderer/context and full
  qualification bundle gates remain independent of the bounded native probe.
- **P4:** requested API/regressions/default-off checks complete on CPU. Formal
  diagnostic=False is proven at the descriptor/receipt boundary only locally;
  `/data/mw_finetune/dp/mt50_policy_100000` is unavailable on this host. No
  permission, training stop or GPU command was attempted.

## GPU fix 2

2026-10-09. Chose **(a): production DP B1 only**. Q's native GPU failure is
real evidence of composition-dependent arithmetic, not a reason to loosen a
production tolerance. At `probe.py:172`, five of112 elements exceeded the old
rtol1e-4/atol1e-5 gate, max_abs6.56545e-5, max_rel5.3227e-4. Source: Q run
`/data/mw_finetune/qualification/runs/slot_G2_dp_libero_A_20261009_194429_754466/`.
The owner explicitly authorized one≤5min local4090 diagnostic; P4 ran exactly
one22.026s measurement. No other GPU invocation, training/process/session
action, qualification edit, E reconstruction edit, commit or push.

### Owned files, actual changes and interface

| Owned hook / file | Change |
|---|---|
| `models/dp/staged.py` | All production stage1/sample batches above1 reject before forwards/RNG sampling. `batch_stage1/2/3` capabilities now **unsupported**, reason “GPU batch composition changes outputs; production B1 only”. Explicit `for_batch_diagnostic(nonempty_label)` shallow clones permit offline measurement while keeping unsupported metadata. Diagnostic result captures carry the label; ordinary B1 captures retain their original fields. |
| `models/dp/batcher.py` | Stage2 and stage3 MISS/warm envelopes also reject production B>1. B1 native noise/envelope/suffix behavior is unchanged. |
| `models/dp/provider.py` | Diagnostic clones cannot register/serve production. Factory/coordinated adapter reject cores above1 or without a verifiable limit, rechecking before every submit. Stage guards cover hot changes between submit and dispatch. Uses stock core's `_max_batch_size` as a local read-only shim; no shared API edit. |
| `models/dp/serve.py`, `wire_policy.py` | CLI rejects every batch-size except1 before loading weights. Concurrent P1 connections continue with B1 forwards. Wire metadata includes `r17_batch_policy`, production max1 and reason. |
| `models/dp/probe.py` | Replace passing B2 equivalence claim with `real_production_B1_only`: unsupported capabilities, B2 rejection and exact queued/direct B1 actions/RNG. B2 numbers live under labelled `diagnostics`, with finite/RNG checks and old-tolerance mismatch count, without a numerical parity gate. Still13 passing checks/domain. |
| `models/dp/quality.py` | Existing B48 CPU proxy uses a labelled offline diagnostic clone and records the label/unsupported production status. Proxy math unchanged; quality sweep was not rerun in this fix. |
| `tests/models/dp/test_dp_gpu_fix2.py`, `run_gpu_fix2_negative.py` | Nineteen new regressions plus isolated missing-guard mutation. Existing B2 math/no-alias test explicitly labelled diagnostic; production async/core serving tests now use B1. Paired256 native integration asserts the new gate/diagnostic distinction. |
| `models/dp/README.md`, `samples/gpu_fix2_*`, `cpu_gpu_fix2_*.xml` | Current docs, strict475 CPU samples, owned Q reseal/runtime snapshot, original GPU measurement/tensors and audited summary. Historical source-sealed samples remain unchanged. |
| `REQUESTS/P4_to_{Q,P1,P6}.md`, this handback | Q native gate/reseal notice and B1-only experiment/cost handoff. |

Interface remains **`r17.contracts.v1`**, existing LoadSpec/provider/stock core
contracts and `r17.dp.qualification.v1`. Metadata/probe diagnostics are additive;
batch support status changes intentionally. There is no production switch to
enable B>1; diagnostic opt-in is not a LoadSpec/serve option and cannot register
an endpoint. B3 protocol parser/profile checks and K1 binding restrictions stay
intact. Default sampler remains DDPM100, commit5, with trailing DDIM K2/K4/K10.

| Finding → change | Regression / evidence | Status |
|---|---|---|
| GPU B1/B2 output divergence → production B>1 disabled | Ten cases across both domains and stage1/stage2/direct sample/MISS/warm reject before a forward or RNG state change. Unsupported capability/wire metadata checked. | **Fixed, CPU pass.** |
| CLI/factory/coordinator could still admit B>1 → entry and hot-update gates | CLI0/2/4 rejects before load; factory/direct coordinated adapter reject B2; hot1→4 rejects before submit. Existing real B1 core async resume/fault buckets and two-connection debug parity pass. | **Fixed, CPU pass.** |
| B2 parity gate blocks valid B1 qualification → label diagnostic and gate B1 | Repro tensor with five measured-size errors fails old assert_close and produces labelled diagnostic now. Actual sealed wire256 native CPU probe verifies exact B1 queue/direct parity and all13 checks. | **Fixed; Q notified. New-release GPU retry pending.** |
| Diagnostic opt-in might leak into production → registration/factory prohibition | Empty/blank/missing labels reject; explicit labelled clone runs actual CPU B2 math but cannot register/serve and does not change the original adapter/capabilities. | **Fixed, CPU pass.** |

### One GPU measurement actually executed

```bash
env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 300 \
  /data/mw_finetune/dp/env/bin/python \
  exp/r17_integration/models/dp/samples/gpu_fix2_measure.py \
  --job /data/mw_finetune/qualification/runs/slot_G2_dp_libero_A_20261009_194429_754466/job.json \
  --output exp/r17_integration/models/dp/samples/gpu_fix2_measurement \
  --deadline-seconds 270
```

Exit0,22.025683s, NVIDIA GeForce RTX4090. This ran **before** implementation
edits against GPU fix1's Q-sealed model pin777919…, verified exact sealed
ModelSpec, and retained weight/norm/protocol/model pins unchanged. Job and asset
paths came from the failed run. `gpu_fix2_measurement/receipt.json` records
actual manifests/hash/device and eight finite comparisons with exact per-request
RNG; `normalized_chunks.npz` saves the outputs. This is a composition diagnostic
of the previous release, not a GPU qualification receipt of GPU fix2.

The three real paired256 NPZ rows are used in order0/1/2/0, with seeds18/37/56/75;
the fourth row repeats real row0 with its own seed. Values below compare full
H16 **scheduler-space internal-affine normalized** chunks, not physical/applied
actions. End-to-end comparison includes independent B1 versus batched stage1;
sampler-only comparison fixes B1 conditioning and changes only denoising batch.

| K | B | Sampler-only max_abs | End-to-end max_abs | End-to-end RMSE | End-to-end old-tolerance mismatches |
|---:|---:|---:|---:|---:|---:|
|100|2|5.96046e-8|1.07795e-4|3.22523e-5|64 /224|
|100|4|1.19209e-7|1.29364e-4|3.47808e-5|119 /448|
|2|2|4.85778e-6|3.63909e-4|4.07091e-5|69 /224|
|2|4|4.30048e-5|1.21027e-3|1.48463e-4|134 /448|

Batch stage1 conditioning max_abs differs by5.24461e-4 at B2 and4.40776e-4 at
B4. All RNG states match. These are four bounded observations/seeds, no SR,
throughput or universal error bound. The original probe used its row0 prompt
variant; this matrix uses Q's actual recorded row prompts, so its maxima differ
from the original five-element failure. No production GPU tolerance is declared.

### Cache experiment implications

Every production arm, including plain, cache MISS, teacher/library generation,
periodic and five-layer recipe, must use B1. Otherwise changed HIT/MISS patterns
can change co-batched requests, condition keys and actions despite paired noise;
that mixes queue composition with the intended cache intervention. Small key
changes can also alter retrieval near a decision boundary. B1 avoids that batch
composition variable. Existing concurrent connection isolation, hot YAML and
commit/tail framework paths stay available; new-release concurrent GPU behavior
still needs Q's qualification. Throughput batching is unsupported. Labelled
batch diagnostics and the B48 CPU action proxy cannot be treated as production
cost/throughput evidence or as an SR certificate.

### CPU/default-off commands and real counts

All following runs hide CUDA and explicitly select CPU. Prefix from repo root:

```bash
p4_cpu() {
  env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=.:src:packages/openpi-client/src \
    /data/mw_finetune/dp/env/bin/python "$@"
}
p4_cpu -m pytest exp/r17_integration/tests/models/dp -p no:cacheprovider \
  -o junit_family=legacy --basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix2_full \
  --junitxml=exp/r17_integration/models/dp/cpu_gpu_fix2_full.xml -q --tb=short
p4_cpu -m pytest tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py tests/serving/test_replica_proxy.py \
  exp/r17_integration/tests/common/test_default_off.py --run-manual -p no:cacheprovider \
  -o junit_family=legacy --basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix2_legacy \
  --junitxml=exp/r17_integration/models/dp/cpu_gpu_fix2_legacy.xml -q --tb=short
p4_cpu exp/r17_integration/tests/models/dp/run_gpu_fix2_negative.py
```

| Receipt | Actual pass / skip / failure | Elapsed / exit |
|---|---|---|
| `cpu_gpu_fix2_full.xml` | **153 /0 /0** | 89.882s /0 |
| `cpu_gpu_fix2_legacy.xml` | **49 /0 /0** | 14.905s /0 |
| New regressions + existing batcher targeted, `cpu_gpu_fix2_targeted.xml` | 23 /0 /0, all included in153 | 8.614s /0 |
| Scoped guard removal, `cpu_gpu_fix2_negative.xml` | 0 /0 /**1 expected failure**,18 deselected | 0.623s /intentional1 |

Distinct accepted pytest total **202 passed, zero skips/errors/failures**.
The mutation disables only `_check_batch` in the test process, reproducing the
old missing guard; stage1 LIBERO regression fails when a forbidden forward is
called. No model source is changed for that run. The tensor magnitude repro
independently proves the old numerical assert fails. Default-off proof includes
stock HEAD config/error/metadata/ack/action-byte parity, real legacy socket
checks, DP default DDPM100 E byte parity, capture/RNG parity and import isolation.
The four frozen stock sources remain byte-identical to HEAD; `git diff --check`
passes. No B>1 production support is inferred from CPU diagnostic math tests.

### Current pinned CPU samples

```bash
p4_cpu exp/r17_integration/models/dp/samples/gpu_fix2_reseal.py
p4_cpu -m exp.r17_integration.models.dp.probe \
  --load-spec exp/r17_integration/models/dp/samples/gpu_fix2_q_libero_seal/load.json \
  --observations /data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix2_q_libero_cpu
p4_cpu -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero --domain libero \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix2_release_libero_cpu
p4_cpu -m exp.r17_integration.models.dp.probe \
  --checkpoint /data/mw_finetune/dp/checkpoints/metaworld --domain metaworld \
  --device cpu --deadline-seconds 290 \
  --output exp/r17_integration/models/dp/samples/gpu_fix2_release_metaworld_cpu
p4_cpu exp/r17_integration/models/dp/samples/fix_round2_validate_samples.py gpu_fix2_release
```

Native probes also had outer300s timeouts. All three strictly load475 tensors
and pass **13 /0 skips**, CPU: Q paired25633.151s, standalone paired96
LIBERO30.080s, diagnostic MW29.614s. Probe B2 data is labelled separately and
does not count as a passing numerical parity check; CPU diagnostic deltas are
nonzero. Current pinned loaders, ten K1/2/4/10/100 YAML and forbidden K1 cases
pass **18 /0 skips**,12.526s. K1 remains plain labelled lossy study only.

Owned Q copy preserves every original field except explicitly resealed
expected_model_identity, now
`sha256:67331d0af13c19fc3e993e5a89f81a1c89469e68d66049f0ec26122de203a022`.
Probe itself overrides only device; emitted full ModelSpec equals the owned
seal. Original Q descriptor/paired asset files remain unchanged. NPZ SHA stays
`sha256:4c480ecbb04193a38da09d85c24431547537cdf1150243d6067c019c8170db4a`;
official wire256 protocol stays589fd029… with40 tasks. The bounded native
probe uses real row0 and a B2 prompt variant; it does not certify three episodes.
Local MW remains ML45 diagnostic; no actual formal MT50 checkpoint load here.

`gpu_fix2_runtime_under_test.json` / `gpu_fix2_summary.json` seal29 model
dependencies, ten framework hashes, XML/probe/measurement counts and artifacts.
Current native manifests/test-time runtime have zero final source drift.
The GPU diagnostic's exact old source manifest and its intentional source delta
are retained separately. Its pin is historical; current code will reject it
until Q reseals. B3 `protocols.py` hash remains04b16028… unchanged.

### GPU qualification command, NOT EXECUTED for the new release

The one authorized diagnostic is exhausted. Q/coordinator owns a later allocated
slot after asset ModelSpec/model/runtime/payload resealing. Same bounded native
command as GPU fix1, using freshly sealed descriptor/input/output paths;≤15min:

```bash
# Later Q/coordinator slot only; not run by P4.
env CUDA_VISIBLE_DEVICES="$P4_QUAL_GPU" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src timeout 900 \
  /data/mw_finetune/dp/env/bin/python -m exp.r17_integration.models.dp.probe \
  --load-spec "$P4_SEALED_LOAD" --observations "$P4_PAIRED_OBS" \
  --device cuda:0 --deadline-seconds 780 --output "$P4_GPU_OUTPUT"
```

### Known limits and pending owners

- **P4:** owned fix, capability corrections, regression/legacy checks and source
  handback complete. Production batches above1 intentionally unsupported;
  measured batch differences are not accepted via a relaxed tolerance.
- **Q/coordinator:** revised criterion in `REQUESTS/P4_to_Q.md`; reseal and
  retry native/concurrent/framework gates with B1. Current13-check CPU pass
  cannot replace a new-release GPU receipt. Qualification files untouched.
  Final scan consumed Q_to_P4's appended “GPU fix3” proposal, citing the same
  failed run. Owner-authorized option(a) replaces the proposed batch-tolerance
  profile: no B>1 production profile is admitted. Native/reference/debug/suffix
  exact gates stay unchanged, including failure on a one-bit native mismatch.
- **P1/P6:** maintain B1 in experiment deployment/teacher/cost paths; notices in
  P4_to_P1/P6. No shared code change is required for this package fix.
- **P2/Q/coordinator:** real renderer/episodes/SR and formal MT50 remain pending;
  the real paired LIBERO training-frame diagnostic is not closed-loop quality.
# 2026-10-10 — Formal DP-LIBERO dynamic batching

Owner's latest ruling supersedes GPU fix2's B1 restriction **only for explicit
DP-LIBERO matched-batch studies**. Defaults stay B1. Contract remains
`r17.contracts.v1`; sampler interface remains `r17.dp.sampler.v2`.

## Owned files / resulting behavior

| Owned files | Change / hook |
|---|---|
| models/dp/staged.py | Opt-in vector scheduler; one private generator per request; DDPM100/trailing eta0 DDIM10/5; default row loop unchanged |
| models/dp/provider.py, batcher.py | Existing S07 stage submissions and buckets; semantic connection/request membership; profile/capture/resume isolation; default core max1 guard retained |
| models/dp/batch_runtime.py | max1..32/window0..50 config; one worker/stage; default-off creation/restoration; bounded exact-B1 frozen CPU language memo; per-stage/decision JSONL ledger |
| models/dp/batch_graph.py | Consume E's read-only DenoiserGraph; startup signature prewarm; prohibit live capture |
| models/dp/formal_batch_server.py | Same sealed provider/YAML and P1/shared WS execution through S07; concurrent resident has no inference lock; active sampler/commit/graph attestation |
| models/dp/serve.py, wire_policy.py | Explicit batch/max/window/ledger arguments; preserve recipe/debug hooks and cleanup; runtime capability metadata/K5 listing; optional partial commit flag |
| models/dp/formal_launch.py, formal_python.sh, formal_eval.sh | Local-only shim preserves P6 sealing/worker deployment/conductor/ownership; B1 delegates unchanged; redirects only resident module |
| tests/models/dp/test_dp_dynamic_batching.py | 24 stream/order/queue/commit/config/default-off/memo/prewarm regressions |
| tests/models/dp/test_dp_formal_launch.py, test_dp_node_ram.py | 6 launcher/default delegation/native-dispatch checks and2 Cosmos RAM-floor checks |
| models/dp/BATCHING.md, samples/dynamic_batch_*.py, samples/cap32_completion.py, samples/node_ram_preflight.py | Pinned CPU seal, real paired NPZ, bounded owned GPU/WS measurement, actual results/report, read-only RAM admission |

New code is owned. No common/P3/P6/qualification/E/frozen R16/harness file was
edited; no commit/push, H100, campaign stop or other agent/process operation.
Initial inspection consumed P3_to_P6, working-tree serving/batcher code and
incoming P1/P2/P3/P6/Q requests. P3 runtime is Fast-WAM-specific; DP reuses the
same model-agnostic stock S07 core directly. Incoming checks continued during work.

## Default-off evidence / CPU commands

`create_runtime(adapter)` returnsNone without threads/files/registry or language
mutation; explicit runtime restores its predecessor state on close. ModelSpec
retains the default capabilities/provider identity; runtime metadata separately
advertises admitted batching. Existing native/reference two-inverse, request
RNG, suffix resume, capture on/off and B1 guards continue to pass. GPU B1 rows
also matched their eager physical references exactly.

Run in `/home/weiland/projects/openpi`, actual E Python island:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m pytest \
  exp/r17_integration/tests/models/dp -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_dynamic_batch_full \
  --junitxml=exp/r17_integration/models/dp/cpu_dynamic_batch_full.xml -q --tb=short
# exit0:185 passed,0 skipped,104.521s (JUnit receipt)

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python -m pytest \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py tests/serving/test_replica_proxy.py \
  exp/r17_integration/tests/common/test_default_off.py --run-manual \
  -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_dynamic_batch_legacy \
  --junitxml=exp/r17_integration/models/dp/cpu_dynamic_batch_legacy.xml -q --tb=short
# exit0:49 passed,0 skipped,15.612s (JUnit receipt)
```

Final launcher/RAM helpers also passed5 targeted tests in0.05s after the read-only
preflight reused P6's cgroup-aware capacity function. `bash -n` passed both
wrappers. Exact launch arguments were parsed with the common launcher replaced
by a read-only parser; no deployment occurred. An initial explicit K100+C5 CLI
example was rejected by P6's combined-override gate; the final native-default
example omits both flags and passes parsing.

After P3's incoming12:07 update, three additional regressions check the normal
launcher at DP max8/32, resident dispatch preserving cuda_graph=True, and the
measured one-worker/stage stream environment. The full185-test suite and49
legacy/default-off checks were rerun after that publication. BATCHING.md's exact
command now uses the normal ops/formal_eval.sh; the owned shim remains fallback.
P4 made no edits to P3's shared files.

Final targeted recheck of test_dp_formal_launch.py and test_dp_node_ram.py:
8passed,0skipped,0.11s; receipt cpu_dynamic_batch_launch_final.xml. The final
normal command's baseline, K5/K10 and horizon5/10/15/16 arguments also parse
without main/deployment. Combined step+horizon remains the documented CLI gate.

Final read-only process/socket inspection found a separately owned new
hs3_dp_b16w5_4090_k100_c5_control resident on23181 and listeners23180/23189.
It was not touched. The exact handback command uses currently free
23160/23161/23179, a fresh run name and the cooperative GPU lock. Candidate
23380 was rejected by the shared reserved-port parser; the final command uses
its permitted range and is revalidated without deployment. All own
measurement PIDs remain closed; the earlier0MiB observation was at their exit.

CPU seal (`samples/dynamic_batch_seal.py`, hidden CUDA) strictly loads475 keys,
missing/unexpected0, final LIBERO100000 weights. It writes official50/4 suites/
40 tasks/2000 episodes, three sampler YAML and the actual paired256 NPZ.
Model identity: `sha256:ed3ae8c67f9686d81227ccd5de627391a7b3ef37f7f302ae8a3f8214b7a16229`.
Weights: `sha256:e460ae1e705b8f9c071f481ac33a8da7ceb56faa8aa9f83752ca3a6736acd269`.
NPZ: `sha256:4c480ecbb04193a38da09d85c24431547537cdf1150243d6067c019c8170db4a`.
Native/framework hashes match the final CPU seal; report also records extra
graph/launcher/measurement sources. All output is under owned models/dp/.

## GPU measurement / real receipts

Latest user authorized20 GPU minutes on local UUID
`GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9`, under`/tmp/r17_gpu.lock`.
Aggregate recorded GPU windows **1199.589115s**, including failures, stopped
capacity sweeps and owned context cleanup. No GPU work remains running;
final read-only nvidia-smi reported0MiB. Only recorded own PIDs were signalled.

Actual shared WebSocket formal residents were tested with1/2 processes, max
1/4/8/16/24/32, real paired views, separate requests/seeds, NFE assertions,
execution acknowledgments and semantic stage ledgers. The final RAM-feasible
matrix uses12 clients and50ms think time. **35/36 configurations measured**;
K5/r2/B32 remains unmeasured. Its final CPU-preloaded attempt proved CPU-only
475-key loading/`cuda_initialized=False`, then hit the4.8s GPU cutoff before
server readiness. There is no passing timing receipt for that cell.

Results/table and exact official50 command are in [BATCHING.md](../models/dp/BATCHING.md).
Raw CSV/JSON/summary live in models/dp/samples/dynamic_batch_results.* and
dynamic_batch_summary.json. Throughput/power use the fixed sampling window;
latency includes window-started calls draining after it. First30 points use4s;
K5/r2's five measured points use1s and have37–50 latency samples each.
Actions/s=5 returned committed rows/call; full predicted horizon is16.

15 first-resident startup graph checks counted in summary pass exact wire bytes
and RNG; every arm also has its own ready metadata. Native B32 qualification
checks352 private request RNG receipts exactly against B1 and compares all16
wire rows. End-to-end serving's largest committed action difference is
0.000345707. No SR/episode/simulator success claim is made from these frames.

The initial B4 GPU run failed because E's lazy global graph capture overlapped
another stage's CPU readback. Fixed before corrected sweeps: capture B1..max
before starting threads, then prohibit all live captures. New CPU prewarm
regression and35 passing concurrent timing points witness the fix. Early failed
and deliberately interrupted runs are retained, clearly distinct from measured
configuration receipts. Earlier64-client data is capacity evidence; the final
launch is sized against the actual node reservation.

## Recommendation / resource admission / pending owners

Recommend **one replica/B8/window5ms/12 timan108 workers**:K10014.5 calls/s,
72.5 action rows/s,708/1248ms p50/p90,155.3W; observed Stage3 max8/mean6.69.
This actually fills B8 under the RAM-feasible measured load. Median latency is
about2.0x lower than1409ms. Provisional capacity speedup1.70x uses inferred
old12/1.409=8.52 calls/s; old aggregate rate and comparable timing distribution
were not measured. Two/B4 achieved18 calls/s/575ms but reached/averagedB3 and
drew214.5W. Keep matched settings across the fresh control and all arms.

At the last successful timan108 read11:33 CDT, total/available251.63/138.85GiB;
Cosmos24 workers RSS90.98GiB with144GiB reservation leaves53.02GiB pending.
With25.16GiB headroom and48GiB for DP12×4GiB, about12.67GiB remains. DP16 does
not fit that snapshot. Latest read-only preflight fails closed: Tether timan108
OFFLINE/exit64; no agent or campaign was restarted. Re-run live admission before
launch. The exact command includes that guard and normal P6 admission; timan1
and H100 are unused. New full baseline/quality campaign was not launched.

| Pending item | Owner / status |
|---|---|
| K5/r2/B32 concurrent timing | Q/coordinator later slot;20min allowance exhausted; native B32 math/RNG passed |
| Matched-batch SR/official50 control and arms | Coordinator; prepared exact launch command, not executed |
| timan108 live connection/RAM admission | Owner/P6; current Tether offline; read-only preflight supplied |
| Shared DP batching launcher/dispatch | Resolved by P3 publication; normal command and native CPU regressions supplied |
| 6GiB Cosmos legacy reservation | P6; REQUESTS/P4_to_P6.md; owned read-only admission enforces floor meanwhile |
| Shared DP paired-control enforcement | Coordinator/P4/P6; DP per-decision membership ledger is supplied, common enforcement is Fast-WAM only |
| Combined step/horizon CLI admission | P6; model hooks pass all3×4 combinations, shared CLI currently separates studies |
| Cache/recipe/capture GPU performance and quality | P1/P6/Q/coordinator; formal batch entry currently pure inference; generic serve keeps hooks |

For a later <=15min smoke (NOT EXECUTED), use BATCHING.md's guarded command,
fresh run name, `--smoke --smoke-episodes4 --seconds780 --episode-seconds600
--retries0` replacing official50/suites. No more GPU use is authorized by this
completed20min measurement budget. Older default-off/B1 handbacks above remain
historical; current opt-in runtime is described by this section.

## Cache study framework integration — 2026-10-10

### Owned files / changes / interface

Implements all model hooks requested in REQUESTS/P6_to_P4.md. CPU-only work;
no GPU call, simulator episode, running process signal, campaign runtime edit,
shared framework/qualification edit, commit or push. P6's authorized CPU
preparation created a new sealed campaign and prepared worker files; it did
not dispatch workers or alter the existing horizon/step fleets.

| Owned file | Change |
|---|---|
| models/dp/admission.py | Read sealed READY/VIEW descriptor binding; exact view digest and K1 rejection |
| models/dp/library.py (new) | Lazy frozen-provider compatibility hook, content/dependency/stat verification, explicit teacher/resident receipt |
| models/dp/library_release.py (new) | Audited teacher producer/asset pins; no broad code-digest bypass |
| models/dp/provider.py | Export and attach configure_library_provider on normal registration |
| models/dp/formal_batch_server.py | Explicit matched cache-study admission and startup capability metadata; reuse resident factory/queues/prewarm |
| tests/models/dp/test_dp_cache_study.py (new) | 16 VIEW, mutation, identity, default-off, K1, topology and eight-group native/WebSocket regressions |
| tests/models/dp/test_dp_fix_round2.py | Existing other-eta K1 fixture now uses a valid sealed descriptor |
| models/dp/samples/cache_study_cpu.py (new) | Strict real-weight/native-reference CPU replay and all16 real-fit receipts |
| models/dp/samples/cache_study_* | CPU preparation/native producer/ledger/log/summary receipts |
| models/dp/cpu_cache_study_{full,legacy,framework}.xml | Final suites; 201/49/33 passed, no skips/failures |
| models/dp/CACHE_STUDY.md; README.md | Current matched configuration, evidence and future GPU smoke |
| REQUESTS/P4_to_P6.md; this handback | Coordinator/P6 receipt, unresolved runtime gates |

Consumes r17.contracts.v1 and existing frozen AdapterProvider API.
Compatibility receipt: **r17.teacher_compatibility.v1 / PASS** with exact
retrieval identities and explicit content evidence. Model-owned configuration
hook returns dataclasses.replace(provider, library_validator=...), preserving
ID, ModelSpec and all original callbacks; no registry replacement. P6's CPU
sealer and normal resident provider both use the same hook.

### Request checklist / hooks

| P6_to_P4 item | Change / test / status |
|---|---|
| Explicit formal cache-study admission | formal_batch_server.py:14 admits only local R2/B8/5ms/MPS, native DDPM100, no CP; plain startup defaults preserved. Topology/default regression at test_dp_cache_study.py:139. **Done on CPU** |
| Exact teacher compatibility | library.py:57 and library_release.py:1; teacher/resident retrieval IDs, known native producer/weights/norm/cameras/preprocess/layout/output, explicit B→A protocol bridge. Frozen provider and eight negative identity/geometry cases at test_dp_cache_study.py:61/76; changed real producer file test at :103. **Done on CPU** |
| B8 HIT/tail and gate-only native DDPM100 | test_dp_cache_study.py:195 real WS/S07/E small model,8 clients,8 YAML groups,256 replies, one resident generation, reload forbidden. HIT[1,0,0], tail[0,0,0], MISS[1,1,100]. **Done on CPU; GPU combination pending** |
| READY→sealed VIEW binding | admission.py:2 uses sealed_store_descriptor with view digest. Both actual B500 VIEW paths tested at test_dp_cache_study.py:24; parent-digest substitution rejected. Other-eta K1 sealed VIEW rejected at :118. **Done** |
| Functional provider configure seam | provider.py:10/28, library.py:98; frozen copy retains identity and callbacks, normal registration attaches hook. Default-off/lazy test at :92; P6 real CPU sealer passes. **Done** |

All group behavior remains in P6/P1's recipe/fitted loader, R6/Q3 controller,
resident YAML manager and native stage scheduler. DP adds no local cache math,
corrector, lottery or stall heuristic. CP1-equivalent recipe HIT computes its
policy-owned64D visual fields; cached tails compute no stage. The gate is the
calibrated successful-B LOEO R6-C-v2/Q3 implementation, ordinary MISS disabled.

### Default-off / native producer evidence

The explicit campaign cache_study gate does not change ordinary formal startup
rejection or plain full.yaml behavior. Merely registering a DP provider reads
no production library or producer assets; its ModelSpec identity is preserved.
Full DP201 tests retain prior DDPM100/native grid, independent RNG, capture
off/on, normalization, resume, batch ordering, horizon and plain wire checks.
Stock49 checks cover default-off import/registration plus byte-level legacy
config/error/metadata/ack/action paths and sockets.

Compatibility is intentionally limited to the audited teacher checkpoint
`sha256:b3eaa7cc24777f398042e08b6ee0b6b9aeefbf67bcbb3a9edc4558133fe2b3c8`.
19 current producer source pins,10 immutable model/CLIP/normalization asset
pins and6 dependency versions are checked. E's native feature/action producer
bytes are unchanged since teacher capture. Its old staged adapter digest is
recorded separately; current staged math is pinned and verified against E.
The accepted pure-inference control hs5dp_ctl_142812 has exactly these current
producer and asset pins. Nonproducer admission/serving additions change the
current complete ModelSpec/code identity; the explicit hook proves the
producer relationship rather than silently ignoring that identity.

Current sealed resident:
`sha256:bd1dd4c1b2e1bfbb4c660ed8ec0a50362f461dc8a631c16edfa9d3097879d0c5`.
Weights remain sha256:e460ae1e705b8f9c071f481ac33a8da7ceb56faa8aa9f83752ca3a6736acd269;
normalization sha256:e77ada20c0ebce241c70a33741a6a9e40deb1a17fe256ebcd391dfb8a1e51923.
Actual475 tensors strictly loaded, missing/unexpected0, CPU only. A real paired
256×256 source frame from each suite passes exact native conditioning,
DDPM100 wire actions at seed18, and native affine→external-quantile decode.
All16 real fitted recipes load, including actual calibrated stall artifacts.
Selected B500 libraries have12,028 /31,546 rows. Fresh Q3 anchors are inactive/
warming_up as expected; these real frames do not prove a stall/SR outcome.
Stored GPU-B1 vs CPU visual-key max absolute differences0.000889719 /
0.000669837 are recorded diagnostics; state keys exact. No CPU/GPU bitwise
claim or success-rate/tolerance qualification follows from this replay.

### CPU commands / actual results

All commands executed with:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/models/dp -p no:cacheprovider -o junit_family=legacy \
  --basetemp=exp/r17_integration/tests/models/dp/scratch_cache_study_full \
  --junitxml=exp/r17_integration/models/dp/cpu_cache_study_full.xml -q --tb=short
```

Same environment/options, legacy targets:
`tests/serving/test_router_bundle_rebind.py tests/serving/test_per_step_recorder.py
tests/serving/test_replica_proxy.py tests/serving/test_policy_identity.py
exp/r17_integration/tests/common/test_default_off.py --run-manual`, scratch
`scratch_cache_study_legacy`, XML `cpu_cache_study_legacy.xml`.
Framework target: `exp/r17_integration/tests/ops/test_cache_study.py`, scratch
`scratch_cache_study_framework`, XML `cpu_cache_study_framework.xml`.

| Final receipt | Exit | Pass / skip / fail | Seconds |
|---|---:|---:|---:|
| cpu_cache_study_full.xml | 0 | 201 /0 /0 | 144.012 |
| cpu_cache_study_legacy.xml | 0 | 49 /0 /0 | 16.711 |
| cpu_cache_study_framework.xml | 0 | 33 /0 /0 | 89.988 |
| samples/cache_study_real_cpu.json | 0 | PASS;2 actual paired frames,16 fits,475 tensors | CPU only |
| P6 smoke --prepare-only | 0 | PASS;16 YAMLs,32 planned episodes;dispatch=false | CPU only |

The eight-group WS regression's actual ledger contains160 HITs,48 library
tails,32 native stall MISSes and16 policy tails. All3200 action-head NFE are
in cg/crg/chg/chrg (8 MISSes each). Stage1:192 requests/maxB8; stage2:32/maxB8;
stage3:32/maxB7. The synthetic sealed fit fixture isolates mechanics, using
real paired96 views and E's small native CPU model; a20ms CPU batch window is
labelled separately from the unchanged5ms production treatment. Every response
checks per-decision stage/NFE/source/corrector diagnostics; final partial
terminal acknowledgments and one resident PID/generation are asserted. Raw
ledger/summary are samples/cache_study_batching.jsonl and _summary.json.

Actual producer replay (same hidden-CUDA/thread environment):

```bash
.venv/bin/python -m exp.r17_integration.models.dp.samples.cache_study_cpu \
  --campaign /data/mw_finetune/eval/mps_dp_libero_1010_164432_2c1f7e32 \
  --output exp/r17_integration/models/dp/samples/cache_study_real_cpu.json
```

Unmodified P6 CPU smoke, same hidden-CUDA/thread environment:

```bash
bash exp/r17_integration/ops/cache_study/smoke.sh dp --prepare-only
```

Campaign /data/mw_finetune/eval/mps_dp_libero_1010_164432_2c1f7e32 records
PREPARED/PREFLIGHT PASS/cpu_only,16 real group YAMLs and closed source seals;
dispatch_started=false. timan107 imports, all official state pools, transport
and private config probes passed. **memory_admissible=false**: current fleets
occupy the node, after reservations/headroom available0MiB for the requested
32×6144MiB. CPU preparation PASS is not permission to launch. The full receipt
is samples/cache_study_prepare_summary.json; no other campaign was modified.

### GPU-ready command / limits / pending owners

Exact coordinator command after GPUs and timan107 RAM are released, NOT RUN:

```bash
cd /home/weiland/projects/openpi
flock -n /tmp/r17_gpu.lock bash exp/r17_integration/ops/cache_study/smoke.sh dp --run
```

Uses local4090 R2/B8/5ms/MPS, nativeDDPM100, group-commit, timan107×32,6144MiB/
worker, four episodes per worker lifetime,32 smoke episodes across16 barriers.
Expected15–25min **estimate**, GPU duration unmeasured; script ceiling120min.
The small two-active-episodes/barrier smoke proves mechanics, not full-study
batch fill or SR. Full study uses all50 A states/task ×10 tasks/suite ×16
suite/group combinations. Existing GPU campaigns/runtimes were untouched.

| Pending item | Owner / status |
|---|---|
| Live timan107 RAM admission after horizon/step fleets finish | Coordinator/P6; current admission refuses launch; preserve matched32 workers |
| Real GPU graph/cache/R2/MPS smoke, all16 YAML acknowledgments and per-replica stable generation | Coordinator/Q/P6; command supplied, not executed under CPU-only task |
| Actual calibrated stall trajectories, NFE audit and final terminal acks | Coordinator/P6; injected/native CPU mechanics passed, actual GPU/simulator receipt pending |
| Complete official A500 per suite/group SR and matched pricing/composition | Coordinator; mechanical CPU evidence cannot establish quality or batch neutrality |
| Update P6-owned DP model-admission status in STUDY/AUDIT metadata | P6; notified via P4_to_P6; real CPU sealer currently passes all16 groups |

P4 model hooks are CPU-complete. Full operational admission still requires
resource admission and the coordinator's actual GPU smoke; no GPU PASS is
claimed. models/dp/CACHE_STUDY.md is the current cache-study reference and
supersedes historical pure-inference-only statements for this explicit mode.

