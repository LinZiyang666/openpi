# P3 — Fast-WAM R17 handback (2026-10-09)

**Delivered model hooks and CPU integration certificates for LIBERO and MetaWorld. G2/G3 GPU and physical qualification remain pending. Zero GPU use; no real checkpoint tensor loading, training/process/tmux operations, commit or push.** Shared serving/cache/recipe/conductor mechanisms are consumed, not reimplemented. All P3 writes are in the owned Fast-WAM directories, P3 request files and this handback.

The **Fix round 1** section at the end supersedes the original capability,
stage1 cost, LIBERO commit/protocol and batching descriptions below. Earlier
49-test receipts remain historical evidence, not the current test count.

## Owned files and actual diff

New code in `exp/r17_integration/models/fastwam/`:

| File | Change |
|---|---|
| `loader.py` | Separate published LIBERO and C MT50 delta loaders; config/head/header validation; content-sealed base/Wan/delta/norm/tokenizer/code identities; strict parameter audits; bounded restoration of C's loader/seed effects; same loader for offline augmentation. |
| `staged.py` | Native VAE/pre-DiT → video-prefill → action-flow cuts; request-local initial noise; exact native suffix resume/reset; stage identity and dtype checks; shared VAE stage lock; CPU-only blueprint. |
| `key_builder.py` | Native pre-DiT spatial pool, one fused real-view field; stock collect/build/clear interface; actual geometry checking; no synthetic cameras. |
| `batcher.py` | Existing StageBatcher protocol, vector B>1 forwards/stack/split, per-request snapshots, identity/K/shape/dtype buckets, native index encoding; submit-only wrapper for BatchingCore. |
| `wire_policy.py` | Published LIBERO normalization/toggle and C MW normalization/inverse; canonical benchmark aliases; exact per-camera resize/concat order; missing/conflicting views rejected. |
| `trace.py` | Observe-only copies of executed arrays; actual sampled grid, tokens/state/keys/CP2/snapshots/RNG; matched-seed offline variants with explicit extra NFE. |
| `manifest.py` | v1 model/norm/layout/feature/sampler/capability manifests and streaming SHA identities. |
| `provider.py`, `serve.py`, `__init__.py` | Explicit provider registration, fresh resident wrappers, shared factory and existing WS, optional existing BatchingCore, P1/P6 recipe/capture callback injection. Package import registers nothing. |
| `qualify.py` | Bounded future native full/staged/K/resume/debug/B2 qualification CLI, CPU-tested with stubs; no physical/SR claim. |
| `{libero,metaworld}_descriptor.template.json` | Diagnostic descriptors; MW immutable export placeholder; protocol allowlist explicit; no weak identity fallback. |

New tests in `tests/models/fastwam/`: `conftest.py`, `test_native.py`, `test_batcher.py`, `test_loader_wire.py`, `test_serving.py`; small native-source/config fixtures, synthetic captures/manifests/snapshots, true native grids, small published config/norm/header probe, unmeasured cost ledger, XML/JSON receipts and test-owned scratch. No shared source edits by P3. The tracked shared diff visible in this checkout belongs to P1; unrelated pre-existing untracked training/results and `logs/session_handoff.md` are untouched by P3.

Requests read: `P1_to_P3.md`, `P2_to_P3.md`, `P6_to_P3.md`. Responses/requests: `P3_to_P1.md`, `P3_to_P2.md`, `P3_to_P6.md`. No local substitute server/cache/controller was introduced while awaiting P1; published contracts are used directly.

## Interface version and entry points

Consumes **`r17.contracts.v1`**, `r17.adapter_binding.v1`, `r17.model_manifest.v1`, `r17.native_snapshot.v1`, `r17.decision_capture.v1`; P6 bridge `r17.p6.v1`. Imports are CPU-safe and do not import torch/JAX/vendor or register a provider. Explicit loading imports the Fast-WAM Python island `/data/mw_finetune/envs/fastwam/bin/python`.

Exports `load_adapter(LoadSpec, device=...)`, `load_offline_adapter`, `register_provider(adapter, provider_id=..., coordinator=...)`, `make_connection_factory(provider, config, storage, **kwargs)`, `make_stage_batcher`, `make_trace_adapter`. Provider clone shares only resident model/wire configuration and a VAE stage lock; key builder, wrapper, orchestrator/facade/history and recipe fork are per connection.

Native raw Wan VAE `encode` clears mutable temporal caches. The shared stage-1 lock serializes those cache mutations. Prefill KV and output shards are cloned per request. **Fix round 1 disables production B>1**; the vector CPU fixture tests below do not authorize production batching.

`LoadSpec.benchmark_protocols` must contain the coordinator's sealed benchmark identity (MW: P2 `FORMAL_SPEC.identity`). The supplied diagnostic templates use fixed-observation protocol tags and cannot be mistaken for formal MW500. No loading of mutable `policy_latest`; no loading without content-sealed expected digests. Checkpoint filename immutability is supplemented by full content verification at each explicit load.

## Default-off proof

No production import or legacy entry point changed by P3. Fresh subprocess import asserts no torch/JAX/LeRobot/FastWAM import and no provider registration. The frozen R16/harness samples, C training/serving sources and original `scripts/serve_policy.py` are byte-identical to the pre-work hashes in `receipts/legacy_before.json`.

The P1 `test_default_off_matches_head_config_errors_metadata_ack_and_action_bytes` was actually run: original Git HEAD vs current default-off path has equal config/error/metadata/ack/action bytes. It passed within the 91-check legacy run. This is a CPU behavioral certificate; it does not claim real released-weight GPU parity. Real GPU and real EGL were not used.

## CPU commands and real results

Common prefix (all commands executed from `/home/weiland/projects/openpi`):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest
```

1. P3 final targets/arguments:

```text
exp/r17_integration/tests/models/fastwam
-p no:cacheprovider
--basetemp=exp/r17_integration/tests/models/fastwam/receipts/final_cpu_v4_tmp
--junitxml=exp/r17_integration/tests/models/fastwam/receipts/cpu.xml
--tb=short -q
```

**Exit 0; 49 passed, 0 skipped/failures/errors; 3.47 s.** Native loop/scheduler definitions are unchanged extracted local vendor source with source SHA checks. VAE/pre-DiT/prefill/head are small CPU stubs; the certificate covers cuts, math, control and ownership, not released model numerics. C's actual strict delta loader ran on tiny parameter views and rejected missing/extra/wrong-shape tensors. The real published local config, normalization files and 1,651-tensor **header** passed inspection without reading weight tensors.

2. Legacy/default-off targets/arguments:

```text
exp/r17_integration/tests/common/test_default_off.py
tests/serving/test_policy_identity.py
tests/serving/test_per_step_recorder.py
tests/serving/test_router_bundle_rebind.py
tests/serving/test_replica_proxy.py
tests/exp/metaworld/test_env.py
tests/exp/metaworld/test_runner.py
exp/offline_search/rounds/r16/test_conductor.py
exp/offline_search/rounds/r16/test_groot_servers.py
-k 'not same_identity_same_initial_state_across_processes'
-p no:cacheprovider
--basetemp=exp/r17_integration/tests/models/fastwam/receipts/final_legacy_tmp
--junitxml=exp/r17_integration/tests/models/fastwam/receipts/legacy.xml
--tb=short -q
```

**Exit 0; 91 passed, 6 skipped, 1 deselected; 20.67 s.** Six skips are manual localhost fake-proxy cases, all resolved in command 3. The deselected test launches real EGL and remains unexecuted.

3. Explicit fake-proxy socket run:

```bash
timeout -k 5s 40s env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  tests/serving/test_replica_proxy.py -m env_dependent --run-manual \
  -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/fastwam/receipts/final_proxy_tmp \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/proxy.xml --tb=short -q
```

**Exit 0; 6 passed, 23 deselected; 0.15 s.** Thus **146 distinct CPU cases passed** (49 + 91 + the six resolved skips), not a sum of repeated development runs. Existing pynvml/Swig warnings only. Ruff invocation was attempted and unavailable (`No module named ruff`); no lint-pass claim. `serve --help` also executed successfully without model imports/loading.

XML and source SHA snapshot: `tests/models/fastwam/receipts/{cpu,legacy,proxy}.xml`, `verification.json`. Earlier scratch remains for audit; no deletion/cleanup of others' files or processes.

## Hook table and evidence

| Hook / feature | Delivered and actually tested | Qualification boundary |
|---|---|---|
| LIBERO / MW loaders | Config separation, published 7/8/224×448/H32 preserved; MW 4D delta geometry; true C strict load semantics on tiny tensors; strong asset hashing | Real full model construction/strict released-parameter load GPU pending |
| Native stages / full / K | Explicit K10 full; native K1/2/10 grids; same-noise original vendor loop vs staged exact on both CPU stubs; CP1 HIT skips prefill/head; MISS reuses S1 | Real BF16 full/staged GPU parity pending |
| Keys / geometry | Policy pre-DiT source; checked native grid; adaptive 4×4; MW one real view, LIBERO fused two-view key; stock key lifecycle | Real token geometry, key cost and retrieval quality pending |
| Normalization / wire | C minmax helper bytes; native LIBERO processor/resize/toggle bytes, zero/tiny spans and gripper midpoint; no output clipping; aliases/missing views | Physical units/polarity beyond inherited controllers remain unqualified |
| Native resume / reset | Every default-grid suffix exact; wrong identity/index/time/shape/dtype/nonfinite rejected; native scheduler add_noise | Donor reuse and advanced stock WS warm dispatch pending |
| Batching / concurrency | Vector B2 S1/S2/S3 equals singles for both benchmarks/K1/2/10, no alias/cross-request noise; real BatchingCore worker/fault/snapshot tests; submit wrapper | Real B2 GPU memory/parity/throughput and GPU replica topology pending |
| Shared WS / bundles / lifecycle | Actual localhost two-client/two-YAML test; K1/2, same-ID idempotence/new generation rebind; bad binding/unknown bundle rejected; disconnect reset | Real model+worker WS qualification pending |
| Commit / five layers | Acknowledged commit10 tails with zero stages; commit up to H32 accepted, >32 rejected; shared library recipe/LOEO corrector, gripper exclusion, library tails and connection recipe isolation | Formal dev descent, full periodic/guard/fit campaign and SR pending P1/P2/P6/coordinator |
| Trace / library / debug | Debug bytes/RNG/forward counts invariant; true executed tokens/state/KV summary/native grid/snapshots; P6 real closed NPZ → reader → H5 exact for both fixtures | Real GPU capture/writer latency and complete artifact/replay campaign pending |
| CP2 / augmentation | Per-layer KV sequence mean summary; same seed K10/2/1 variants billed as extra 13 NFE; native resume variant hook; sealed offline loader | CP2 WS/online-RIT/shadow/routing advanced dispatch fails closed pending P1 |
| Split / blueprint | JSON-only CPU blueprint, no GPU/vendor objects | Split capability unsupported; P6 transport qualification future |

Capabilities distinguish CPU-supported hooks, implemented-but-unverified shared campaigns, and unsupported advanced combinations. No `qualified` real-GPU status is emitted. The generic shared factory rejects CP2/warm-reset/trace-routing advanced configurations; direct native hooks are available for future certified shared dispatch, not a fabricated uniform-flow fallback.

## Native / norm / feature / storage samples

- `fixtures/native_grids.json`: actual native scheduler float32/BF16 K1/2/10. K2 float32 times `[1000, 833.3333...]`, deltas `[-0.1666667..., -0.8333333...]`; BF16 times `[1000, 832]` with rounded native deltas. Updates use the scheduler's saved grid, never uniform time. Stock float snapshot keys encode **index/K only**.
- `fixtures/published_libero_cpu_probe.json`: real config/norm/header-only evidence, both norm-file SHA values, no weight hash or load claim.
- `fixtures/{libero,metaworld}_stub_manifest.json`, `*_capture.npz`, `*_snapshot.json`: clearly synthetic hidden8, key width128, normalized 32×7 / 32×4 and wire actions; K2 index1 snapshot, identities and RNG algorithm. Native dtype is recorded separately; numeric storage uses float32 copies and restore must cast to declared native dtype before strict resume.
- Real config predicts hidden3072, pooled `vision_0:49152` plus `robot_state:8/4`, native grids `(1,7,14)/(1,7,7)`. These are config-derived expectations enforced against runtime meta, **not measured GPU geometry**. LIBERO fields are fused; they do not denote two independent embeddings.
- `fixtures/unknown_cost_ledger.json`: all real Ckey/prefill/K10/K2/K1 timing values null; GPU B2, checkpoint quality and key quality unqualified. No π/GR timing ratios borrowed.

## Future GPU commands — prepared, not executed

Coordinator runs **one model/checkpoint at a time**, after training releases the GPU. Seal assets outside the GPU slot (streaming many GB takes time). Replace MW checkpoint placeholder and supply three saved, authorized native observations in NPZ (`image/state/prompt`, LIBERO also `image2`; MW originals 480²; no pickle). Replace diagnostic protocol with P2/coordinator sealed benchmark identity before formal serving. Current templates are diagnostic mechanics only.

```bash
# CPU/I/O sealing only, after fixing the descriptor; no model load.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.serve \
  --descriptor exp/r17_integration/models/fastwam/libero_descriptor.template.json \
  --seal-output exp/r17_integration/models/fastwam/libero_descriptor.sealed.json

# LIBERO slot: bounded <15 min, one resident checkpoint; no other process touched.
timeout -k 5s 840s flock -n /tmp/r17_gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.qualify \
  --descriptor exp/r17_integration/models/fastwam/libero_descriptor.sealed.json \
  --observations exp/r17_integration/models/fastwam/libero_fixed_obs.npz \
  --device cuda:0 --deadline-seconds 780 \
  --receipt exp/r17_integration/models/fastwam/libero_gpu_receipt.json
```

MW: run the same sealing/qualification commands in a separate slot with the `metaworld_` descriptor/observations/receipt names. The MW template requires `<COORDINATOR_IMMUTABLE_MT50_EXPORT>` replacement; it does not use C's mutable `policy_latest`. **Do not add `--batch2`: production B>1 is now rejected.** The CLI tests three fixed observations, explicit K10/2/1 vs native full loop with matched CPU noise, debug bytes and exact suffix resume; elapsed-with-load and tolerance are recorded. If load consumes the deadline, the receipt remains partial; outer timeout is the hard bound. LIBERO observations must contain both original256² frames. This CLI does not certify a physical episode, SR, pricing, GPU replicas or WS on GPU.

Optional real server for coordinator's separate ≤15-min mechanical episode slot:

```bash
# P3_OWNED_PORT is selected/free-checked by P6/coordinator; no reserved port assumed.
timeout -k 5s 840s flock -n /tmp/r17_gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.serve \
  --descriptor exp/r17_integration/models/fastwam/metaworld_descriptor.sealed.json \
  --device cuda:0 --port "$P3_OWNED_PORT" \
  --cache-config exp/r17_integration/models/fastwam/metaworld_bound.yaml \
  --manifest-output exp/r17_integration/models/fastwam/metaworld_serve_manifest.json
```

The bound YAML is generated after sealing/loading with P1 `binding_for(provider, protocol, num_steps=..., commit_horizon=10)`; `key_builder.type=adapter_native`, exact feature dimensions/keys, in-memory backend, `write_policy=never`, no legacy denoise_schedule. Optional `--max-batch-size 2`, P1/P6 `--recipe-factory module:function`, and `--capture-sink module:function` reuse shared paths. P2/P6/coordinator supplies the actual one-short-episode worker/transport command and own-process normal shutdown; there is no claim that a missing checkpoint/observation/YAML/worker receipt is already ready.

## Known limits and pending owners

| Item | Owner / next gate |
|---|---|
| Immutable final MW export, sealed benchmark identities and authorized fixed observations | Coordinator/C/P2; templates remain diagnostic until replaced |
| Real released LIBERO strict-load/full-staged parity; MW base+delta load and same-noise BF16 parity; actual token grids | P3 hooks ready, coordinator runs separate bounded G2 slots |
| Real B2 memory/parity, GPU stream/concurrent isolation, endpoint/replica qualification and Ckey/prefill/head pricing | Coordinator/P6/P3; CPU conformance is not GPU throughput evidence |
| One real MW and one real LIBERO mechanical episode with controls/cursor and manifest join | P2 + P6 + coordinator after native qualification |
| Native warm/CP2/trace twins/online-RIT/shadow/routing integration into stock advanced executor | P1 shared dispatch, P3 native hooks; requests recorded, configurations fail closed now |
| Tail capture of current raw state | P1 optional capture field request; stage-free state is already extracted; no extra model forward requested |
| Full debug/warmup/teacher artifacts, replay/search/analysis, LOEO/dev/test freeze, corrector/guard/periodic/descent campaign | P6/P1/P2; both native capture fixtures interoperate, full campaign remains pending |
| Pre-DiT fused-key retrieval quality, low-K SR and final-checkpoint quality/full500 | Coordinator/P6 G4; no small-smoke SR claim |
| Physical controller units/polarity beyond inherited metadata, all MW physical semantics/restore | P2/coordinator; no invented LIBERO grasp predicates |
| Split/RPC transport certificate | P6; CPU blueprint only, unsupported advanced capability |
| Lint | Environment/tooling: Ruff absent; all source imports and relevant CPU paths tested |

No outstanding routine permission question. Shared/new advanced limitations are explicit pending gates, not hidden success claims.


## Fix round 1

Read both Opus reports fully. No disagreement with B4/B5/B6/B7/B9/B10. All
changes below are in P3-owned paths; no GPU, real model tensor load, training
process/tmux operation, commit/push, frozen R16/harness or mw_finetune edit.

| Finding | Change | Regression/evidence | Status |
|---|---|---|---|
| B4 major | Stage1 executes native VAE→patchify→key pooling only. UMT5/proprio conditioning, time/text pre-DiT work, mask and video prefill move to stage2 MISS. A request-local delegation shim calls unchanged native pre_dit while reusing the already computed convolution; no resident method/global monkeypatch. The VAE lock no longer encloses UMT5. | `test_b4_key_does_not_encode_text_or_pre_dit_and_miss_reuses_patch` poisons the text entry point during key extraction, checks prompt/proprio-independent vision keys, and checks one text call on MISS and no repeated patchify. `test_b4_unchanged_native_pre_dit_reuses_identical_conv_tokens` executes unchanged vendor pre_dit on tiny real torch modules for both benchmarks and float32/bfloat16, exact full/staged parity and one convolution. Existing K1/2/10 parity and CP1 HIT tests pass; HIT now also asserts zero added context/pre-DiT calls. | Fixed; CPU certified. Real cost/weight parity pending GPU. |
| B5 LIBERO resolution | Both real cameras must be RGB uint8 original256×256; client224/128/480 frames reject. Each camera still uses original torch antialias bilinear256→224, then width concatenation. MW480→PIL224 unchanged. Wire resolution and resizing are in execution profile and preprocessing identity. | Six `test_b5_libero_rejects_client_resized_or_wrong_camera_resolution` cases; original vendor preprocessing equality now uses original256 frames. | Fixed. P2 owns simulator/client protocol receipt. |
| B6 released protocol | Explicit `LoadSpec.execution_protocol="released_libero"` and labelled descriptor template. Released profile: replan/commit32, K10, per-call explicit_seed42, settle10, caps spatial/object/goal/10=280/280/300/520, original256 frames. System profile remains replan5/semantic RNG/spatial220. Profiles produce different layout/preprocessing/model identities; wrong seed or reduced K in released profile rejects. Serve metadata and `--protocol-info` expose the label. | `test_b6_released_protocol_is_explicit_and_changes_layout_seed_and_binding` checks identities, caps, binding32, fixed seed rejection, K rejection and full32×7 output. New original-kernel/P1/P3 integration cases pass for both profiles: system actual34 controls with semantic seeds and commit10 tails; reference actual32 then2 with seed42 on both calls and terminal ack. | Model profile fixed. P2 reference spec/runner are now consumed in CPU fixtures; official-task sealing/physical execution and real reference-loader parity remain pending; no published97.6% SR claim. |
| B7 metadata | CP2/native_warm_resume/native_warm_reset become implemented_unverified. Production batch_stage1/2/3 become unsupported with the BF16 batch-composition reason. | `test_b7_advanced_and_batch_capabilities_fail_closed`; require() rejects the unqualified advanced hooks. | Fixed. Shared advanced executor qualification remains P1/P6. |
| B9 capture cost | Ordinary capture=True no longer computes CP2 means. Explicit capture_cp2=True is required. Captured native step/snapshot products have capture_enabled=True/pricing_eligible=False; explicit CP2 sampling is also unpriceable. Separate measurement runs capture off. | `test_b9_ordinary_capture_skips_cp2_and_marks_trace_unpriceable`: CP2 absent by default, explicitly present, unchanged KV/actions, flags correct. Measurement tests assert capture off. P1 ledger rejects debug pricing. | Fixed model hook; P6 must retain capture exclusion through campaign analysis. |
| B10 B>1 | Serving rejects every max-batch-size other than1 before model load. Default adapter/stage batcher rejects B>1 at each stage; production provider rejects a batch coordinator. No CLI/LoadSpec bypass. Vector implementation is retained only behind explicit allow_unqualified_batch=True used in CPU fixtures; metadata remains unsupported. | `test_b10_production_adapter_and_coordinator_reject_batch2`, `test_b10_serve_rejects_batch2_before_loading_weights`. Existing vector fixtures remain numerical-unit evidence only. | Fixed by disabling production B>1. |
| A5 analogous Fast-WAM metadata gap | Handshake now includes r17_model_spec and actual r17_samplers10/2/1 (released profile10 only) for P2 worker physical binding, plus execution profile and loader parity limitation. | `test_a5_serve_metadata_matches_worker_modelspec_contract`. | Fixed. |
| A2 / shared migration | Consume P1 local common.config and common.websocket_policy_server; stock src files restored by P1. No P3 changes to sealed source. | Default-off test plus `receipts/fix1_frozen_shared.json`: all four reviewed shared files equal Git HEAD SHA256. | Fixed integration import; R16 verify is P1-owned. |
| B1/B2 cross-package | Added reviewer LIBERO-context repro as regression: extended commit with old undeclared ack fails; original init7 is parsed; missing formal context fails. P3 does not create another benchmark runner. | `test_b1_b2_reviewer_libero_metadata_repro_fails_closed`, using actual shared factory. REQUESTS/P3_to_P1.md and P3_to_P2.md. | Shared fail-closed seam passes; P2 owns end-to-end LIBERO runner/spec/conductor certificate. |
| P4 collection request | Every test filename uses test_fastwam_ prefix and fixture/helper imports are fully qualified. | Combined P3+common collect-only:147 collected,0 errors. | Fixed. |

Owned diff: modified `staged.py,batcher.py,wire_policy.py,manifest.py,loader.py,
provider.py,serve.py,qualify.py,trace.py`; added `protocols.py,measure.py,
libero_released_descriptor.template.json`. Tests renamed to
`test_fastwam_native.py,test_fastwam_loader_wire.py,test_fastwam_batcher.py,
test_fastwam_serving.py`; added `test_fastwam_fix_round1.py`, native pre-DiT
source fixture and receipts. Modified own fixture/conftest/vendor-source receipt.
Updated own P3 requests to P1/P2/P4/P6 and this handback. No shared file edits.
Consumes unchanged r17.contracts.v1 and current additive r17.adapter_binding.v1
with P1 r17.recipe.v1/local config/WS/native seams; Stage1.data is opaque and
its text context is now intentionally deferred.

Default-off proof: Fast-WAM still requires explicit load/register; imports do
not import torch/JAX/vendor or register a provider. Default production batching
is disabled; CP2 summaries and phase observers are default off. Original HEAD
config/error/metadata/ack/action byte comparison and frozen pre-work hashes
pass. Source-parity receipt records all four sealed reviewed src files equal
HEAD, so this package does not require an R16 amendment.

### Actual CPU results

The common CPU prefix shown above is unchanged. These are actual runs with
CUDA hidden/JAX CPU; all scratch and outputs are in our owned receipts folder.

| Command after prefix | Exit | Pass / skip / deselect | Receipt |
|---|---:|---|---|
| `-m pytest exp/r17_integration/tests/models/fastwam -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix1_final_cpu_v3_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix1_cpu.xml --tb=short -q` | 0 | 72 passed,0 skipped,0 failed | fix1_cpu.xml |
| Original nine legacy/default-off targets listed above, same exclusion and options, with `--basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix1_legacy_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix1_legacy.xml` | 0 | 91 passed,6 skipped,1 deselected,0 failed | fix1_legacy.xml |
| `timeout -k 5s 40s` + CPU prefix + `-m pytest tests/serving/test_replica_proxy.py -m env_dependent --run-manual -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix1_proxy_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix1_proxy.xml --tb=short -q` | 0 | 6 passed,0 skipped,23 deselected | fix1_proxy.xml |
| `-m pytest exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common -p no:cacheprovider --collect-only --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix1_final_collection_v2_tmp -q` | 0 |147 collected,0 errors; collection only | command recorded in fix1_verification.json |

**169 distinct passing CPU tests** =72+91+6. The six legacy skips are resolved
by the proxy run. The one real EGL cross-process initial-state test remains
unrun; no renderer/GPU certificate is inferred. Earlier migration failures
were corrected before these final runs, not counted as passes.

Regression sensitivity was actually checked in six independent processes by
`receipts/fix1_defect_reinsertion.py B4|B5|B6|B7|B9|B10` under the CPU prefix.
Each reinserted its old defective behavior in memory and exited1 for the
expected assertion (B4:2 failures, B5:6, B6/B7/B9/B10:1 each;0 collection
errors). Receipts `fix1_red_B*.xml` preserve those12 expected failures. This is
**defect reinsertion, not a complete historical-source rerun**. No production
file was mutated. The same regressions pass on the fixed code in fix1_cpu.xml.

### Key/full cost split and samples

`receipts/fix1_cost_boundaries.json` contains actual CPU call counts and a
separate tiny-fixture timing sample for both benchmarks; those tiny timings
are explicitly ineligible as real model prices. Key path counts: VAE1,
patchify1, text0, pre-DiT0, prefill0, head0. K10 full: VAE1,patchify1,text1,
pre-DiT1,prefill1,head10. Full has no repeated patch convolution.

The analytical parameter×token estimates from review B4 give UMT5≈1.1776
TFLOP, video prefill≈0.9800 LIBERO /0.4900 MW TFLOP, K10 action≈0.6528 TFLOP.
Let V be the unknown VAE+patchify cost (pool/copy and omitted operations are
also unmeasured). Then old vs new key/full fractions are approximately
(V+1.1776)/(V+2.8104) → V/(V+2.8104) for LIBERO, and
(V+1.1776)/(V+2.3204) → V/(V+2.3204) for MW. **These are rough FLOP formulas,
not measured latency ratios or an asserted <0.1 bound.** Saved analytical
receipts leave V and real timing/ratio null. The full native computation stays
the same; its text cost moves off every HIT and outside the VAE lock.

Samples include current synthetic manifests with unsupported B>1 and corrected
advanced statuses, actual key/full call receipts, and the labelled released
execution profile in fix1_cost_boundaries.json. Native pre-DiT fixture is copied
from the unchanged vendor source and guarded by its source hash. Production
manifests/libraries must be resealed because code/preprocessing/layout identities
changed; earlier synthetic manifest/capture fixtures remain historical only.

### GPU-ready commands — not executed

Coordinator only, one model at a time after training releases the GPU. Seal
updated code/assets outside the GPU slot. Supply real saved paired256×256
LIBERO frames (or MW480×480), state and prompt in allow_pickle=False NPZ. No
synthesized wrist camera; no mutable checkpoint. Released descriptor needs
P2's sealed released BenchmarkSpec.identity, not the placeholder string.

```bash
# Diagnostic protocol inspection only; no model load, no GPU.
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.serve \
  --descriptor exp/r17_integration/models/fastwam/libero_released_descriptor.template.json --protocol-info

# Dedicated capture-off pricing slot (<15min including load).
# Reseal libero_descriptor.sealed.json using the earlier CPU sealing command first.
timeout -k 5s 840s flock -n /tmp/r17_gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.measure \
  --descriptor exp/r17_integration/models/fastwam/libero_descriptor.sealed.json \
  --observations exp/r17_integration/models/fastwam/libero_fixed_obs.npz \
  --device cuda:0 --deadline-seconds 780 --repeats 3 \
  --receipt exp/r17_integration/models/fastwam/libero_key_full_gpu.json
```

Run MW measurement in its own slot with metaworld_ paths and the final immutable
MT50 delta. The command reports synchronized B1 prepare/key/MISS/sample/wire
milliseconds, key/full ratio, and separate VAE/patchify/pooling/text_context/
pre-DiT/video-prefill/action-head timings, after one excluded warmup. These are
isolated adapter phases; controller/search/transport costs are outside this
measurement. Deadline exhaustion yields partial; external timeout is the hard
bound. Use the existing qualify command in a separate slot for native mechanics
and capture parity; its captured rows are not pricing evidence. Released
profile qualification uses K10/seed42 only. Neither command qualifies the raw
Wan loader against the published sidecar/Diffusers/HF loader or published SR.

Known limits/pending owners: coordinator/P3 G2 for real weight parity and measured
key/full costs; coordinator/P3 for raw-Wan versus released-sidecar equivalence;
P2 for sealed system/released LIBERO protocols, actual final partial execution
ack and simulator/conductor receipts; P1/P6 for advanced CP2/native-warm
executor dispatch; P6 for capture-off campaign pricing and closed model/physics
joins; coordinator for final immutable MT50 export. P1 now supports explicitly
negotiated system commit32 via allow_partial_commit=True plus episode
partial_tail_protocol=r17.partial_tail.v1; default binding rejects it, while
released profile replan32/commit32 needs no short-tail negotiation. Production
B>1 stays disabled until a separate numerical batch-composition certificate.

## Fix round 2

Read REVIEW_A2.md and REVIEW_B2.md completely and checked incoming P1/P2/P4/P6
requests during implementation. All changes below are P3-owned. No GPU use,
neural weight-body load, commit/push, running-process/tmux operation, frozen
R16/harness change or mw_finetune edit. Earlier sections describe earlier runs;
this section supersedes their descriptor, scope, measurement and loader claims.

| Finding | Change | Regression / actual evidence | Status |
|---|---|---|---|
| REVIEW_B2 N2 | Add --debug-directory, mutually exclusive with --capture-sink. Pass P6 make_capture_sink_factory(provider,directory) into the actual P1 connection factory. Close it in finally, including setup/server failures. | test_n2_serve_debug_uses_connection_sampler_and_closes drives serve.main with real P1/P6 cores at K2 and K1, separate bound sinks/grids, execution ack and closed capture manifests; normal and exceptional exits. test_n2_debug_and_global_sink_are_mutually_exclusive verifies CLI rejection before load. | Fixed; CPU tested. Real deployed GPU capture remains G2/G3. |
| REVIEW_B2 N3 | Add LoadSpec.benchmark_specs documents and validate them before checkpoint probing/body loading. Reconstruct P2 identity; verify allowlist coverage, embodiment, task table and manifest attestations. LIBERO system requires replan5/settle10/wire256/seed7 paired_semantic/caps220,280,300,520; released profile requires replan32/settle10/wire256/fixed42/caps280,280,300,520. MW requires canonical seed7/corner2/table/partitions/replan5/settle15/cap160. Register validated specs in P1. System seed label is explicit_benchmark_seed. | test_n3_*: three P2 manifest JSON roundtrips; six LIBERO parameter mismatches fail before a poisoned checkpoint probe; seven MW mismatches fail; system/reference mixing, forged identities/task tables, missing documents and diagnostic bypass limits reject. Existing original LIBERO kernel/P1/P3 profile tests pass after validating/registering their spec. | Fixed. Formal descriptors must supply actual P2 documents. |
| REVIEW_B2 N4 | Replace prompt-only scope with versioned SHA of effective CPU-prepared image and normalized proprio tensors, typed shapes and prompt. Compute before GPU transfer; no text encoder or GPU KV/latent copy for the identity. Native resume uses existing strict P1 NativeSnapshot.validate. | Seven test_n4_* cases cover each embodiment's image/state/prompt and the real LIBERO second camera. Identical independently prepared inputs resume exactly; changed conditioning rejects before head calls. Five same-prompt cases fail with old prompt-only scope reinserted. | Fixed native hook; shared warm dispatch remains unqualified/fail-closed. Old prompt-only snapshots are incompatible. |
| REVIEW_B2 N5 | key_ms now includes prepare+stage1. Report hit_path_ms=prepare+stage1+wire and both full-path shares; retrieval_search_ms is null/excluded. Label wire as full-output transform proxy and formal pricing as P1 host wall ledger. | Two deterministic-clock test_n5_* cases prove the numerator formulas and inequalities, with nonzero preparation/wire costs. Capture-off phase/call-count tests remain green. fix2_cost_boundaries.json has actual CPU fixture samples, explicitly ineligible as real model prices. | Fixed measurement definition. GPU timing and campaign retrieval cost still pending. |
| REVIEW_B2 B6 partial | Add loader_parity --header-only: execute actual raw-loader dispatch/P3 parameter-copy constructor and released from_pretrained constructor on meta tensors, substituting frozen sidecars/body reads. Check published config, all canonical trainable shapes/dtypes and K10 schedule. Add serial raw/released GPU output comparison command with strict serialization alias normalization and independently supplied released HF/Diffusers/tokenizer assets. | test_b6_actual_published_raw_released_loader_header_equivalence and fix2_loader_header.json: 1,651 unique parameter shapes, 24 inference config fields, dtypes and schedule match; zero tensor-body reads. Published header has 1,620 older block aliases and original/native key sets differ. test_b6_released_serialization_shim_preserves_values_and_is_strict verifies exact tensor values through alias rename and rejects missing/extra/wrong/duplicate tensors. Comparator detects frozen-output drift and uses exact token/mask equality; tolerances must be finite. Default-off import test passes. | CPU canonical config/header equivalence passed. Unmodified historical serialization loader NOT certified. Frozen VAE/T5/tokenizer/action numerical parity remains GPU pending. |
| Additional owned loader defect exposed by B6 | Raw FastWAM is a separate source tree, absent from the documented CPU PYTHONPATH. Explicit loading now temporarily resolves the sealed vendor src path and rejects raw modules from other locations; paths/loader functions restore. | The actual meta/header probe initially failed ModuleNotFoundError('fastwam'). It now passes with the standard CPU prefix. Audit asserts actual raw loader selection and restoration. fix2_loader_path_reinsertion.py reinstates the old scope in memory and reproduces the missing-package failure with zero body reads. | Fixed in loader.py; no environment or vendor edits. |
| REVIEW_A2 N2 / P1 migration | Owned recipe fixtures use complete canonical P2 tables/identity and actual canonical task_name/id, with partial library rows permitted. Consume P1's optional request task_table_identity without stripping it. | Both library/corrector/tail fixture cases and both original LIBERO kernel profiles pass. Combined P3/common suite passes. | Fixed fixture consumption; no local compatibility shim/shared edits. |
| REVIEW_B2 N1 cross-package qualification | Request P6 to replace invalid480/synthetic LIBERO inputs with real paired256 frames and keep B>1 disabled in slots. P3 does not edit qualification assets/specs. | REQUESTS/P3_to_P6.md records the failure prerequisites and descriptor migration. | Pending P6/coordinator confirmation before GPU slots. |

Owned diff this round: modified models/fastwam/{serve.py,loader.py,protocols.py,
staged.py,measure.py,libero_descriptor.template.json,
libero_released_descriptor.template.json,metaworld_descriptor.template.json};
added loader_parity.py and README.md. Modified owned conftest.py,
test_fastwam_serving.py and test_fastwam_fix_round1.py for current P1/P2 table
interfaces; added test_fastwam_fix_round2.py and fix2_* receipts/scripts. Updated
own P3_to_P1/P2/P6 requests and this handback. No other package edits.

Interface remains r17.contracts.v1 / adapter_binding.v1 / recipe.v1. LoadSpec's
benchmark_specs is an additive P3 descriptor field; optional task_table_identity
is the P1 append-only request field. Native conditioning scope uses
r17.fastwam.conditioning.v2 internally. Code/preprocessing identities changed:
reseal descriptors, models and libraries before qualification. Diagnostic
document omission requires diagnostic=true plus exclusively fake_* protocols;
formal identities cannot bypass validation using diagnostic=true.

Default-off proof: imports do not load/register a model, change source paths or
import torch/JAX/vendor. Raw import/bootstrap and serialization shims occur only
in explicit loading/qualification. Debug factory defaults to None, mutually
exclusive with the callback; CP2 summaries/phase observers still default off;
production B>1 stays disabled. Native/reference action-byte, capture/RNG and
upstream/default-off tests pass. fix2_frozen_shared.json proves the four sealed
src files match Git HEAD byte-for-byte. No R16 amendment is required by P3.

### Actual CPU commands and receipts

Use the same explicit CPU prefix for each Python command:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python
```

| Command after prefix | Exit | Actual pass / skip / deselect | Owned receipt |
|---|---:|---|---|
| -m pytest exp/r17_integration/tests/models/fastwam -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix2_final_cpu_v5_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix2_cpu.xml --tb=short -q | 0 | 105 passed,0 skipped,0 failed | fix2_cpu.xml |
| -m pytest exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix2_composition_final_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix2_composition.xml --tb=short -q | 0 | 214 passed,0 skipped,0 failed | fix2_composition.xml |
| Nine legacy/default-off targets below, -k 'not same_identity_same_initial_state_across_processes' -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix2_legacy_final_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix2_legacy.xml --tb=short -q | 0 | 92 passed,6 skipped,1 deselected,0 failed | fix2_legacy.xml |
| timeout -k 5s 40s + CPU prefix + -m pytest tests/serving/test_replica_proxy.py -m env_dependent --run-manual -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/fix2_proxy_tmp --junitxml=exp/r17_integration/tests/models/fastwam/receipts/fix2_proxy.xml --tb=short -q | 0 | 6 passed,0 skipped,23 deselected | fix2_proxy.xml |

Legacy/default-off targets:

```text
exp/r17_integration/tests/common/test_default_off.py
tests/serving/test_policy_identity.py
tests/serving/test_per_step_recorder.py
tests/serving/test_router_bundle_rebind.py
tests/serving/test_replica_proxy.py
tests/exp/metaworld/test_env.py
tests/exp/metaworld/test_runner.py
exp/offline_search/rounds/r16/test_conductor.py
exp/offline_search/rounds/r16/test_groot_servers.py
```

The six skips are the socket cases resolved by the explicit proxy run. The one
real EGL initial-state test remains unrun. Positive XML node identities deduped
across all four suites give **310 distinct passing CPU tests**, not the sum of
overlapping runs. fix2_verification.json records counts; early failing/partial
iterations are preserved and excluded from this claim. No real renderer or
neural weight/body/GPU execution is inferred from CPU passes.

Header audit also ran directly with the same prefix replacing .venv/bin/python
by /data/mw_finetune/envs/fastwam/bin/python:

```text
-m exp.r17_integration.models.fastwam.loader_parity --header-only
--checkpoint /data/mw_finetune/models/fastwam_libero --wan-root /data/xwam/wan22_5b
--receipt exp/r17_integration/tests/models/fastwam/receipts/fix2_loader_header.json
```

Exit0/status passed; meta/CPU only, zero weight-body reads. This is a bounded
config/header audit, not VAE/text output equality. The original key-set mismatch
is preserved in its released_serialization object rather than hidden by shape
canonicalization. GPU released comparison uses only a serialization rename shim,
then native strict load_state_dict; frozen loader/forward math stays unchanged.

Regression sensitivity: CPU prefix + receipts/fix2_defect_reinsertion.py
N2|N3|N4|N5|B6 ran in five independent processes, expected exit1. Their JUnit
receipts show respectively2/6/5/2/1 failures,0 errors. N2/N3/N4/N5 restore old
defective behaviors in memory; B6 bypasses the newly added comparison guard.
This is not a historical-source replay. The separate island invocation of
receipts/fix2_loader_path_reinsertion.py exits1 with expected missing raw package;
fix2_red_loader_path.json records it. No production source was mutated for red
runs. Fixed regressions pass in the final suite.

Samples: fix2_cost_boundaries.json contains actual tiny CPU fixture phase timings,
prepare-inclusive key/HIT shares, profiles and call counts. Warmup plus one K10
measurement gives VAE2/patchify2/text2/prefill2/head20; key phases remain only
VAE/patchify/pool. A full call has VAE1/patchify1/text1/prefill1/head10; every HIT
avoids text/prefill/head. Key/full and HIT/full samples are not GPU prices;
retrieval search is null/excluded and wire is a full-output transform proxy.
The original analytical FLOP estimates remain estimates, not measured latency.

### GPU-ready commands — NOT executed

Coordinator must supply final immutable exports, real paired256 LIBERO frames
(MW480 in its separate slot), complete P2 benchmark documents and fresh code/
asset seals. Published reference descriptor uses the released P2 spec, fixed42,
K10/replan32. Export local released Diffusers VAE/text assets and a separate
released UMT5 tokenizer before taking a GPU slot; no implicit download occurs.
Each command below is its own <=15min slot, one model resident at a time.

```bash
# Raw-Wan versus released HF/Diffusers/tokenizer numerical parity, serial loaders.
# Replace the two quoted asset placeholders with immutable local exports.
timeout -k 5s 840s flock -n /tmp/r17_gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.loader_parity \
  --descriptor exp/r17_integration/models/fastwam/libero_released_descriptor.sealed.json \
  --released-sidecars '<COORDINATOR_LOCAL_RELEASED_DIFFUSERS_EXPORT>' \
  --released-tokenizer '<COORDINATOR_LOCAL_RELEASED_UMT5_TOKENIZER>' \
  --observations exp/r17_integration/models/fastwam/libero_fixed_obs.npz \
  --device cuda:0 --deadline-seconds 780 --rtol 0.001 --atol 0.001 \
  --receipt exp/r17_integration/models/fastwam/libero_loader_parity_gpu.json

# Separate capture-off system-profile cost slot, after resealing system spec/code.
timeout -k 5s 840s flock -n /tmp/r17_gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/envs/fastwam/bin/python -m exp.r17_integration.models.fastwam.measure \
  --descriptor exp/r17_integration/models/fastwam/libero_descriptor.sealed.json \
  --observations exp/r17_integration/models/fastwam/libero_fixed_obs.npz \
  --device cuda:0 --deadline-seconds 780 --repeats 3 \
  --receipt exp/r17_integration/models/fastwam/libero_key_full_gpu.json
```

Parity compares two B1 observations, independent tokenizer IDs/masks exactly,
VAE latents/context/normalized/wire actions with stated tolerances. The parent
hashes released sidecars/tokenizer, descriptor and observations; raw descriptor
binds all raw assets/code. Each worker exits before the next loads. Mismatch or
deadline exhaustion fails the receipt. This does not certify historical loader
serialization, published SR, throughput or all inputs. The native qualify CLI
from the earlier handback remains a separate slot; do not pass --batch2. Run MW
measurement only in its separate final-export slot, with its canonical P2 doc.

Known limits / pending owners: P3+coordinator for frozen encoder/raw-versus-
released numerical parity, native GPU mechanics and capture-off prices; coordinator
for independent released sidecar/tokenizer exports and immutable MW delta; P2
for official LIBERO task-map export/real renderer and physical episode evidence;
P6/coordinator for REVIEW_B2 N1 qualification inputs/B1 slot correction and
per-connection capture deployment; P1/P6 for native warm/CP2 advanced executor
qualification and complete cost ledger consumption. Production B>1 stays
unsupported. No GPU/SR/full G2/G3 certificate is claimed by this round.
## 2026-10-10 owner batching task

This section supersedes earlier zero-GPU/B>1-unavailable statements only for the
owner's new explicit local Fast-WAM LIBERO batch study. Default B1, MetaWorld
and other models keep their previous gates. Shared interface remains **v1**;
no shared/frozen/DP/Cosmos/training source was edited by P3.

Owned additions: `models/fastwam/{batch_profile.py,batch_runtime.py,formal_launch.py,
formal_python.sh,formal_batch_server.py,BATCHING.md,measurements/}` and
`tests/models/fastwam/test_fastwam_batched_serving.py`, test-owned receipts.
Owned changes: `staged.py` explicit batch config separate from test bypass;
`batcher.py` request/row/stage receipts; `provider.py` connection-local native
clones bound to S07; `serve.py` default-off --max-batch/window/ledger; `README.md`.
Requests appended to `P3_to_P6.md` and `P3_to_P4.md`; incoming P4 request read
and its DP seam relayed to P6. No commit/push or other process/tmux operation.

Default-off proof: `create_runtime(...,max_batch=1)` returns None with no file,
thread, model mutation or metadata change. Replayed CPU actions are byte-equal
before/after. B2 still fails without an explicit runtime or test bypass. B1
formal-launch shim delegates directly to unchanged P6 main. The combined suite
also ran the original HEAD/default-off common comparison, native FP32/BF16
math cuts, debug and layout/tail checks. `git diff --exit-code` on the four
shared serving/cache files exited0. Current P3 Python source hashes match the
successful GPU descriptor.

CPU commands actually run (all CUDA hidden/JAX CPU, bytecode/cache writes off;
all scratch under owned `tests/models/fastwam/receipts/`):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common \
  -p no:cacheprovider --basetemp=exp/r17_integration/tests/models/fastwam/receipts/batching_final2_scratch \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/batching_final2.xml -q

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py \
  tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py exp/mw_finetune/tests \
  -k 'not same_identity_same_initial_state_across_processes' -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/fastwam/receipts/batching_legacy_scratch \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/batching_legacy.xml -q

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  tests/serving/test_replica_proxy.py -m env_dependent --run-manual -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/models/fastwam/receipts/batching_sockets_scratch \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/batching_sockets.xml -q
```

Results: **238passed,0skips/failures** (129 P3 +109 common); legacy
**89passed,6skipped,1EGL deselected**, then **6socket passes**, resolving all
six skips. The new file contributes19 passing checks. Initial full suite found
two debug-entry failures caused by passing coordinator=None to the default
registration mock; fixed by retaining the original B1 call. Earlier failed XML
is retained. Final real WS test verifies concurrent native batches, distinct
rows/actions/receipts, tail byte identity and no tail model forwards.

| Hook | Change/evidence | Scope |
|---|---|---|
| Native stages/S07 | Real VAE/prefill/head batch hooks, one worker/stage, producer events, private noise/KV/results | GPU B1/2/4/8/16 K10 replay passed |
| Provider/connection factory | Explicit runtime binds each cloned adapter to stock S07; sealed model/provider/YAML stays unchanged | CPU real WS + formal-entry test passed |
| Composition | Ordered semantic request/connection members, row/stage/batch ID/config identity in JSONL and per-decision response | GPU memberships + CPU isolation/tail tests passed |
| Norm/RNG/sampler | Existing native math/inverse and per-request CPU noise; K buckets retain native grid | CPU K1/2/10 isolation; GPU K10 tolerances measured |
| Serving/capability metadata | Default manifest still describes B1; explicit runtime advertises enabled_study and non-byte parity/quality pending | CPU metadata/default-off tests passed |
| Formal option | Local P3 shim delegates P6 seal/deployment/workers/conductor/process control, redirects only resident entry | CPU scoped restore + formal resident + live WS passed; physical formal command unexecuted |

GPU authorization was explicitly changed by the owner for this task. The local
DP campaign initially held the lock; P3 waited with `flock -w600` and did not
signal it. First attempt failed a changed-code seal before CUDA allocation.
Successful UUID-pinned run used an internal1200s deadline/outer1280s bound,
loaded in86.44s and exited normally in150.35s including load. Lock released
promptly; subsequent nvidia-smi showed0MiB and flock nonblocking succeeded.
No H100 connection or telemetry probe occurred.

Measured recommendation: **B8/window5ms**, **5.983decisions/s** versus
**1.100direct B1** = **5.437x** (5.970x versus queued B1). Mean/p95 latency
1.334/1.448s; allocated/reserved peak24.31/32.44GiB; mean/max power183/191W.
B1 head K10 is820.85ms of970.27ms instrumented wall (84.6%). Native full
B16 gives10.06x, but actual queued B16 gives only4.6% more throughput than B8
with72% more latency because the 5ms window fragments stages. B8 normalized
max/median/p95 abs errors0.0078125/0.0009765625/0.00390625; overall sweep max
0.015625. Raw receipts/provenance/tables/exact commands are in BATCHING.md and
`measurements/local_sweep{.json,.batches.jsonl,2.log}`. Recorded real frames are
dataset episode1272 frames0/1/2; no physical episode or SR claim.

GPU-ready commands not executed: **the new local B8 formal smoke command** in
BATCHING.md (one resident,16 workers,40GiB budget,1200s campaign), then a fresh
matching batched control/arms after smoke admission. Existing H100 B1 baseline
must stay unchanged and cannot be reused as paired control for B>1 arms.

Known limits/pending owners: **P6** for native formal launcher seam/H100 runtime
and attestation; **coordinator/P3** for H100 arrival-pattern/throughput/memory
qualification and actual B8 physical smoke/control quality, heterogeneous prompt
and reduced-K numeric envelopes; **P4** for DP batching/graph qualification
(their GPU queue may now acquire the released lock); **P6/Cosmos owner** for
new Cosmos StageBatcher. Local GPU sweep is in-process S07 connection adapters,
not GPU WS/network/renderer timing. Occupancy and pure CUDA kernel-time split
were not collected; phase event spans include launch gaps. Power samples are
short windows; VRAM allocator reserved peaks retain previous sweep pools.
No generic advanced warm/CP2/recipe batching or historical raw/released loader
numerical parity was certified by this new task.

## Normal-launcher B8 qualification and matched campaigns — 2026-10-10

This section supersedes the earlier local-only shim/H100-pending statements.
The owner authorized normal launcher changes and a bounded H100 smoke (≤15 GPU
minutes) after accepting B8/5ms arithmetic drift. Coordinator stopped B1 C10;
the failed B1 baseline is not a usable batched paired control. No baseline/study
campaign in the new command list was executed by P3.

### Files and concrete diff

- Owner-authorized shared ops seam: `ops/formal_eval.py` batch flags, campaign/
  seal/summary/receipt identity, control gate, qualified stream environment,
  attestation, physical VRAM and decision-composition audit; `formal_eval.sh`
  documents the opt-in; six dispatch lines in `formal_server.py` select owned
  Fast-WAM/DP entries. Existing B1 resident body is unchanged. Q's concurrent
  memory/recycling/multi-EGL changes were retained; P3 did not edit Q's files.
- Owned new `models/fastwam/formal_batching.py`: CPU-only treatment/control/
  resume/attestation/audit guards; `formal_qualification.py`: bounded startup
  replay on the existing resident, no second model load; `recover_smoke.py`:
  narrowly scoped CPU recovery with archive/code/closed-PID/transfer pins.
- Owned `formal_batch_server.py`: H100/local admission + hardware/source
  attestation, startup replay, same parent transfer manifest as normal serve,
  per-decision treatment identity; `formal_launch.py`: delegates to normal
  launcher. Native stages/batcher/RNG/S07 math unchanged.
- Owned `formal_commands.sh`, `BATCHING.md`, expanded eight CLI lists, command
  parser/schedule receipts, H100 summary, CPU XML/log/snapshot evidence under
  `models/fastwam/measurements/` (pre-edit ops snapshots use `.before`).
- Owned `tests/models/fastwam/test_fastwam_formal_batching.py` (41 cases), plus
  a close-snapshot/install regression in `test_fastwam_batched_serving.py`.
  Coordination updates: REQUESTS/P3_to_Q.md, P3_to_P4.md, P3_to_P6.md.

Interfaces: existing P1/P2/P6 contracts v1; BatchConfig/stage/decision receipt v1;
new **batch treatment v2** binds config and source identity, GPU family,
replicas/workers/node, renderer allowlist and preferred GPU, RAM/VRAM/restart/
recycle settings, one worker/stage and stage streams. Closed v1 smoke pins remain
readable, fresh resident startup checks the current implementation separately.

### Default-off and actual CPU results

`--max-batch 1` remains default. No batch fields/file/thread/queue/metadata on
B1; original action/manifest/metadata bytes match in P3 tests. Shared default-off
HEAD probe checks config/errors/ack/actions and opt-in import side effects.
Four tracked serving/cache files have zero git diff. Q's separate worker resource
default changes are outside P3's B1 numeric/default-off claim.

Every pytest below used `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src`, `.venv/bin/python`,
`-p no:cacheprovider`, owned unique `--basetemp`, and owned XML:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common \
  exp/r17_integration/ops/tests -p no:cacheprovider \
  --basetemp=exp/r17_integration/models/fastwam/measurements/cpu_formal_final_tmp \
  --junitxml=exp/r17_integration/models/fastwam/measurements/cpu_formal_final.xml -q
```

**Exit0:413 passed,0 skipped/failures/errors,29 subtests passed,219.46s**.
XML tests442 includes29 subtests; do not count442 plus29 again. Earlier full
pass398+29 reflects the pre-close-fix/Q resource snapshot; the first mistaken
test-path command exited4 before collecting tests, retained as cpu_formal_full.xml.

Final acceptance targets: `test_fastwam_formal_batching.py`,
`test_fastwam_batched_serving.py`, `tests/common/test_default_off.py`:
**exit0,62 passed,0 skips,9.74s**, `cpu_formal_acceptance.xml`.
The 62 overlap the full suite. Close-fix targeted pass58 is historical.

Legacy targets/commands from verification/CPU_CHECKS.md: policy_identity,
per_step_recorder, replica_proxy, router_bundle_rebind, MW env/runner,
`exp/mw_finetune/tests`, excluding the real EGL
`same_identity_same_initial_state_across_processes`:
**exit0,89 passed,6 initially skipped,1 deselected,7.82s**, cpu_formal_legacy.xml.
`tests/serving/test_replica_proxy.py -m env_dependent --run-manual` under an
explicit40s bound: **exit0,6 passed,23 deselected,0 skips**, cpu_formal_sockets.xml.
No training source/results were modified by those read-only tests.

`bash -n formal_eval.sh` / `formal_commands.sh`: exit0. All8 expanded commands
parsed through the current normal launcher; six requested K/C arms, B8/5ms,
N4 and timan107-only assertions PASS in `formal_commands_cpu.json`.

### Hook table and execution samples

| Hook | Change / retained framework path | Evidence |
|---|---|---|
| Resident selection | Normal ops entry dispatches explicit Fast/DP batch settings to owned hooks; B1 body unchanged | CPU local/H100 parsing/dispatch/default-off tests; H100 normal-entry load/attestation |
| Native S07 | Same VAE/prefill/flow head, per-request RNG/row KV/CPU producer events, one worker per stage | CPU K1/2/10 row/RNG isolation; GPU K10 direct B1/B8 + S07 replay |
| Paired control | Closed PASS official50 K10/C5, same GPU/config/source/topology/model/norm/protocol/suite coverage, pinned receipts | CPU reject B1/windows/GPU/workers/VRAM/recycle/failed control/model mismatch; resume guard; Q added semantics |
| Composition | Every serving decision includes config/member IDs/order/row/native stages and treatment pin; tail empty/zeroNFE | CPU malformed/mixed/tail cases;153 GPU physical decisions audited |
| Closure/transfer | Same parent manifest and SHA-attested P6 snapshot/install/allowed transfer stage | CPU regression + real closed H100 recovery, source archive/PID pins |
| Knobs/quality | Existing native K1/2/10 and C5..32 framework path, immutable model spec and default sampler | CPU full/horizon/steps/bundle suites; K10 physical GPU smoke only |

Sample: first physical decision belongs to spatial/task0/init0, explicit seed
3500044960, campaign P3_h100_fastwam_B8_smoke_20261010_1209; semantic request
and row in each stage receipt match. Original capture/cache disabled;
stage_calls[1,1,10], actualNFE10, H32/replan5, normalized action width7.
The model/norm/native-grid sealed examples are under that run's assets and
SERVER_ATTESTATIONS.json; native replay frames/seeds are1272:{0,1,2}/{42,43,44}.

### Actual H100 smoke and receipt repair

Executed normal launcher command (not a GPU-ready-only proposal):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src bash exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --server-node h100 --worker-node timan107 \
  --replicas 1 --workers 4 --worker-egl-device 1 --worker-max-episodes 8 \
  --worker-memory-mib 6144 --max-batch 8 --batch-wait-ms 5 \
  --port 23170 --driver-port 23179 --smoke --smoke-episodes 4 \
  --seconds 780 --episode-seconds 550 --retries 1 --memory-budget-gib 36 \
  --batch-qualification-observations /data/mw_finetune/qualification/observations/libero_original256_fix2_v1/fastwam_observations.npz \
  --run-name P3_h100_fastwam_B8_smoke_20261010_1209 --run
```

**Four physical episodes completed,4 successes/0 errors/0 retries;153 decisions**.
GPU resident287.00s /4.78min; all owned resident/proxy/driver/workers closed,
no GPU/episodes reexecuted. Driver192.21s /0.796 decisions/s includes setup/
shutdown, S3 actual mean1.50/max3 with only four workers. Native replay max-B8
**4.95dec/s,3.78× queuedB1 (1.31)**, max/median/p95 normalized deviation
**0.015625/0/0.00390625**, allocated/reserved24.17/27.60GiB. Eight-client replay
fragmented into5+3; direct fullB8 also passed8.397dec/s. Power includes concurrent
Cosmos and is not attributable to Fast-WAM alone. No SM occupancy certificate.

Initial launcher exit2/receiptPARTIAL was a missing P3 parent manifest in final
H100 server snapshot, after successful episodes and closure. Fixed owned entry;
regression fails on old code and passes now. CPU-only recovery verifies the
archived old implementation, CPU seal, closed PIDs, SHA snapshot/install,
required backends/workers and all153 per-decision composition receipts.
Original RECEIPT.json is untouched; **RECOVERED_RECEIPT.json PASS**. New parent
manifest and treatment-v2 bookkeeping were CPU-verified after the GPU run;
native model/stage/batcher/RNG/S07 code was unchanged. Do not call the first
unrecovered launcher exit0 or count another four GPU episodes.

Receipt root: `/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209`;
local `measurements/h100_smoke_summary.json` binds the results and archive pin.
Recovery exact command (already executed, CPU only):
`python -m exp.r17_integration.models.fastwam.recover_smoke --directory <root>`.
H100 Fast-WAM ports23170..79 only; existing Cosmos ports23240..49 untouched.
No local GPU allocation or interference with P4's /tmp/r17_gpu.lock measurement.

### Recommendations / commands not executed / limits and pending owners

Recommend joint schedule H1001/local1,16 workers per resident, N4/RAM6144MiB,
renderer VRAM1024MiB and H allowlist1,2,3 / local4,5,6. timan107 clean-node RAM
limit32 workers; full combined fleets wait for admission/Cosmos natural exit.
H100 can host2 solo residents at34GiB each once free, but scaling was not measured;
local48GB supports1. Avoid H1002/local1 needing48 workers. Actual B8 fill needs
monitoring; four-worker smoke was explicitly not a filled-B8 episode-rate test.

Exact eight commands and durations are in BATCHING.md / formal_commands.sh and
expanded *_commands.txt. H100 lane runs official four-suite2000-episode control
(4.4–8.9h), then C16 (0.8–1.6h). Local lane runs two-suite1000-episode control
(2.1–4.1h), then C10(1.0–2.1h),C20(0.5–1.0h),C32(0.3–0.7h),K2(0.7–1.3h),
K1(0.5–1.0h), each with --paired-control pointing to its own GPU control.
The local lane holds flock for every run. Enumeration of64 assignments balances
measured burst rates; projected combined elapsed5–10.5h after admission.
Durations assume50–100% suite caps and20% extra, not measured formal campaigns;
K1/K2 estimates use analytical step-cost scaling. No replica linear-scaling claim.
All these formal campaigns are **GPU-ready commands not executed** by P3.

Pending owners: **coordinator** for full paired controls/arms and success-rate/
cost conclusions,16-worker arrival/fill measurements and scheduling after RAM/
P4 lock admission; **P3/coordinator** for reduced-K GPU numeric envelope and
optional standalone2-resident H100 scaling; **P4** for DP GPU batching/quality
(normal hook only CPU-tested here); **D/P6** for any Cosmos stage-batching hook;
**Q/M** for continued worker-memory accounting/recycling qualification. Existing
advanced-library/G3 and released-loader numeric-parity limits retain scope.
No full success-rate-equivalence claim from4/4 smoke or three recorded frames.

Final post-Q legacy refresh also exits0:89 passed/6 skips/1 deselected in6.94s
(`cpu_formal_legacy_final.xml`), and explicit socket6 passed/23 deselected in0.13s
(`cpu_formal_sockets_final.xml`). Recovered receipt's original SHA,153 audited
compositions, closed ownership and zero GPU/episode rerun pins rechecked PASS.



## CUDA MPS option and bounded qualification — 2026-10-10

Owner authorized this narrow normal-launcher seam. New owned implementation:
`models/fastwam/ops/mps.py` (CPU-only lifecycle, private env, UUID binding,
PID/start verification, accounting, leased guard, exec wrapper, resume shim and
comparison scope); `mps_probe.py` (unchanged P4 actual resident load generator
and native Fast S07 caller); `mps_h100_check.py`, `mps_h100_probe.py`,
`mps_exec_check.py` (bounded explicit qualifiers). All auxiliary modules are
CPU/import-safe; GPU work requires explicit main execution. New regression
`tests/models/fastwam/test_mps.py`. Results, source snapshots and XML live in
owned `models/fastwam/measurements/mps_*`. Updated BATCHING.md and requests
P3_to_Q / P3_to_R. Shared edits ONLY formal_eval.py/sh: explicit --mps, campaign
config, leased guard before replicas, replica-only private environment / exec,
attested VRAM and receipt / teardown. Q's concurrent runtime_from, cwd and
prestaging changes were preserved; formal_resume/prestage and S/R files untouched.

Interface: native adapter/contracts v1 unchanged; opt-in MPS schema r17.mps.v1,
active-thread percentage100, existing Default mode required. Native loader root
source hashes exactly equal hs3 Fast B8 control; helpers live in a subdirectory,
while the normal runtime archive pins them. No shared cache/serving HEAD diff.
Default-off CLI common arguments, server environment and native source seal
match the pre-MPS snapshot. Off campaigns gain no mps field or daemon. Non-MPS
processes receive none of the new private pipe settings. Resume on/off mixing
is checked before attach; Q was asked to add it to direct shared resume semantics.
Mismatched-MPS paired controls are marked wall-time incomparable in the receipt.

CPU commands actually run (CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS=cpu,
PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=.:src:packages/openpi-client/src,
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1):

- `.venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib
  exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common
  exp/r17_integration/ops/tests --junitxml=.../mps_full_cpu_complete.xml`:
  **495passed,40subtests,0skips** (224.67s). Final new import-only
  regression additionally passed in the23-case focused suite; intermediate
  full run494passed/40subtests also passed.
- Same flags, target `tests/models/fastwam/test_mps.py`, XML mps_cpu_final3.xml:
  **23passed,0skips**, including actual launcher orchestration success/failure
  with fake services, PID/pipe tamper refusal, startup rollback, private client
  accounting, resource mode refusal, resume mix refusal, exec env sanitization,
  before-snapshot defaults and exact native code seal.
- Legacy targets from verification/CPU_CHECKS.md first group, excluding only
  same_identity_same_initial_state_across_processes: **89passed,6initial skips,
  1EGL deselected** (mps_legacy.xml). All29 replica-proxy tests, including the6
  env-dependent sockets, explicitly ran with --run-manual: **29passed,0skip**
  within mps_cpu_socket_final.xml's47passes (18MPS tests at that checkpoint).

Actual GPU tests: DP8cases (R1/B16,R2/B8 × off/on/off/on), Fast4passing cases
(R1/B8 off/on/off/on),32clients, real recorded observations, native K100/K10.
All eigen/graph references and batch ledgers preserved. DP native eager actions
are identical across every on/off pair; max batch wire deviation0.000345707.
Fast eager normalized references are identical; batched-vs-fixed-off-eager
max0.015625, median0.000488–0.000977, p950.003906. No physical SR test.
Pooled measured throughput: DP R2 MPS26.163 vs22.659 calls/s (1.1546x), pooled
p50 977 vs1138ms. DP R1 MPS13.653 vs14.381 (background changed). Fast R1 MPS
6.768 vs6.946 (0.9744x), p50 4404 vs4338ms. Recommendation: MPS optional for
DP two-replica B8 with a matched control; retain off for current R1/B16 DP and
single-replica Fast. H100 Fast remains unqualified; keep it off.

H100 mixed model attempt refused admission with14,408MiB free (need63,488MiB),
four existing contexts unchanged. No new model loaded. Tiny H100 private CUDA
check PASS (3.43s), both live Cosmos contexts unchanged. Actual local normal
MPS guard + exec wrapper CUDA check PASS (4.32s), correct private client PID and
496MiB including server overhead; removed inherited1MiB memory cap before16MiB
allocation, shutdown not forced, existing background GPU context retained.
No compute-mode write and no signals to non-owned GPU processes.

The background was not constant: original hs3 DP/Fast campaigns independently
closed PARTIAL with campaign-stopped/deadline receipts and all_owned_closed=true;
new H100 Fast replicas appeared. First DP-on pair preserved its original
background PID/start identity. DP background rates were off12.774/6.954 and
on6.357/0 calls/s; later DP/Fast pairs had0 old-background calls. No causal
claim about that decrease. MPS direct qualification receipts preserve actual
snapshots, including the original DP context's later external disappearance.
Conservative total probe windows **1,073seconds /17.9GPU-minutes across both**,
including failed setup attempts and checks, below20minutes. Every own process
and private daemon closed; failed descriptor/property attempts were corrected
and rerun, not used as throughput results. H100-owned processes never signalled.

Artifacts: `/data/mw_finetune/eval/p3mps_ab_1010`, p3mps_fw_ab_1010 (failed),
p3mps_fw2_ab_1010 (one pass, next round bounded), p3mps_fw3_1010 (three passes),
p3mps_exec_1010/receipt.json (real guard PASS). H100 p3mps_hcheck_1010 receipt
PASS and p3mps_h_ab_1010 receipt SKIP_MEMORY copied into measurements. Full
pooled table/numerics in measurements/mps_results.json; commands in BATCHING.md.

Study decision: **remaining MPS-on arms cannot pass the complete cost gate
against existing MPS-off timing baselines.** STUDY.md §3/§6.2 require comparable
service-ms/control-step plus model-call/actual-NFE ratios, not just count cost.
Existing SR/count baselines can remain subject to native/batch identity and
parity gates, but these replay tests do not prove SR invariance. Keep existing
arms off, or collect a matched MPS timing reference. No study/baseline edit.

Pending: Q direct shared resume mps field; R append results into RUN_CONFIGS;
coordinator H100 mixed throughput/parity once capacity is available (no forced
window or existing-context reconfiguration); coordinator physical quality check
for any new topology/MPS treatment. Use fresh runtime preparation and short
names (private pipe<=85bytes); avoid pre-MPS --runtime-from releases. This task
created no campaigns beyond its bounded measurement children and no git commits.

Final name-path fix: MPS auto-generated names are short enough for Unix sockets;
long custom names fail during CPU preparation before any process is created.
The focused naming/prestage/launcher regression XML is mps_cpu_naming_final.xml.
All default-off branches and Q edits preserved. Final source pins refreshed.

Final focused checks after the name-path fix and Q driver_mode publication:
**52passed,29subtests,0skips** (mps_cpu_naming_complete.xml), including all24
MPS regressions plus launcher/prestaging tests. The earlier23-case focused
receipt and495-case full suite remain preserved. No further GPU use.


Additional background audit: DP R2's first off/on pair had no external CUDA
context and gained14.9%. During the second on repetition another CUDA context
appeared (PID1890260,1486MiB at the final snapshot). Its call rate was not
captured; the old hs3 background counter remained0. Thus pooled1.1546x is an
observed ratio, not a two-repeat estimate under identical production load.
The second pair's16.1% gain has that qualification. Do not describe all later
zero old-background counters as an idle GPU. A further context appeared during
the tiny guard test and was retained. No additional context was signalled.

Final verified focused rerun:52passed/29subtests/0skips in6.34s (mps_cpu_verified.xml);
includes the CPU refusal of mismatched --mps --output / --run-name paths.
Final source hashes are in measurements/mps_final_receipt.json. No GPU work
was performed after the closed17.9-minute measurement budget.
## Exclusive H100 capacity tuning —2026-10-10

Owned edits: `models/fastwam/batch_runtime.py` extends the explicit opt-in cap
16→32; new CPU-import-safe `models/fastwam/ops/h100_capacity.py` implements
bounded48-client native/S07 replay, persistent residents across windows,
target-B16 VRAM admission, privateMPS, completion-window accounting and owned
process cleanup. Tests changed: `test_h100_capacity.py` (new),
`test_fastwam_batched_serving.py`, `test_fastwam_formal_batching.py`,
`test_mps.py`. Documentation appended to `BATCHING.md`, `P3_to_R.md` and
`P3_to_Q.md`; all evidence is under owned `measurements/h100_capacity_*` and
`tests/models/fastwam/receipts/h100_capacity_*`. No ops/shared/vendor/worker
implementation edits this turn. Native code digest changes deliberately;
fresh control/runtime preparation is required, then freeze it for every arm.
Interface remains R17 contracts v1, S07 and batch-treatment v2.

Default-off proof: existing/new tests compare normalized action bytes,
manifest/metadata and absence of files, queues/threads with max_batch1;
default-off CLI/server environment unchanged. Legacy cache/websocket/S07
tracked sources have zeroHEADdiff. Native forward/norm/weights unchanged.

CPU commands actually run, CUDA hidden, JAXcpu, OMP/OpenBLAS/MKLthreads1,
PYTHONDONTWRITEBYTECODE1 and PYTHONPATH`.:src:packages/openpi-client/src`:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib \
  exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common \
  exp/r17_integration/ops/tests \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/h100_capacity_full_cpu_final.xml
.venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib --run-manual \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py \
  tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py \
  -k 'not same_identity_same_initial_state_across_processes' \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/h100_capacity_legacy.xml
```

Final results: **539passed+43subtests,0skip**, full Fast-WAM/common/ops;
**71passed,0skip,1 realEGLtest deselected**, legacy including localhost proxy.
Focused regressions72passed; CLI/MPS71passed. Seven exact baseline/arm
commands separately passed normal parse_args and private-pipe path checking;
no campaign preparation/launch occurred. Initial full run had one obsolete
17-cap assertion, changed to33 and entire suite rerunPASS; two focused test
construction errors were fixed before final validation. Raw failed receipts
remain preserved; final source pins/CPU counts are in h100_capacity_source.json.

Hook table: normal formal_eval `--max-batch/--batch-wait-ms` accepts16/24/32 on
H100/local via the owned BatchConfig; S07 production stages/payloads unchanged;
native normalized-output/private RNG parity tested at16/24/32; MPS option uses
the previously implemented private leased guard/replica-only environment;
existing decision JSONL keeps actual stage membership/treatment identity;
all control/arms must match GPU/replicas/batch/window/MPS/fleet reservations.

GPU actually executed under the owner's new exclusive-H100 authorization:
singleR1 B16/24/32 at5/10ms, R2 B16 at5/10ms with MPS off/on. NativeK10,
systemC5,48 closed-loop clients,50ms think, three real LIBERO observations.
Full ten-row table and exact coordinator commands are in BATCHING.md;
recommended **R2/B16/5ms/MPS on:28.0 calls/decisions per second**, p50/p90
1679/1779ms, S3fill16/16, GPUbusy75.7% (not SMoccupancy),412.6W,
full32-action maxdrift0.015625, first5max0.0078125, median0,p950.00390625.
Capacity ratio8.24× vs owner-observed old3.4calls/s; actual campaign speedup
and success rate unmeasured. Same-config MPS gain12.9%. R3 not admitted:
allocated-workspace/context/headroom need~87,331MiB vs81,081MiB free.

Conservative total probe wall **410.17s/6.84GPUmin**, counting loads, warmup,
cleanup and two failed CPU-only packaging probes. Successful raw receipts:
`/data/mw_finetune/eval/p3_hcap_1010c` and
`/data/mw_finetune/eval/p3_hcap_multi_1010`; copied under owned measurements.
All own children/private MPS daemon closed; final H100contexts[],0MiB,0%,
compute modeDefault unchanged. No4090 GPU work/lock, no DP signals or changes.

Limits/owners: only three replay observations and K10, no simulator/SR,
websocket/proxy/dispatch or SMoccupancy measurement. Coordinator launches
fresh2000-episode official control and six matched arms; P3 supplied commands,
CPU-parsed only. R updates RUN_CONFIGS from P3_to_R. Q uses new code/control
and keeps privateMPS/fleet/runtime flags matched. Owner-selected hs4 W48
RAM2048/VRAM768/N4 passed read-only timan107RAM admission alongside liveDP32;
admission must rerun at launch, and M/Q retain recycling/memory monitoring.
No unmeasured campaign-duration or success-rate claim is made.

## Cache study Fast-WAM admission — 2026-10-10 (CPU only)

Owner scope: P6_to_P3 items1–3, actual B500 library path, eight YAML hot-switch
groups on hs6 Fast-WAM configuration. No GPU work, model-server/worker/MPS
daemon/dispatch start, existing-campaign signal, frozen-runtime edit, commit
or push occurred. CPU-only new preparations deployed source/assets to new
campaign directories; archived teacher/control sources were read only.

### Owned files and diff summary

- `models/fastwam/provider.py`: export configure_library_provider returning a
  same-ID/same-ModelSpec frozen replacement; attach identical lazy callback on
  live registration. Ordinary recipe-disabled inference is unchanged.
- `models/fastwam/ops/library_compatibility.py`: fail-closed source/content
  admission, exact physical weight/config/norm/tokenizer/vendor and key/wire/
  loader/protocol pins, complete native-geometry/transform comparison, B→A
  native bridge, native K10, executing-source fence and content-bound receipt.
- `models/fastwam/ops/library_evidence.py`: audited teacher/control/current
  content generations. Frozen teacher's native stage bodies match current
  AST exactly, excluding only exact-SHA-pinned constructor/batch admission
  additions. Empty Python3.12 type_params metadata normalized across3.11/3.12.
- `models/fastwam/ops/cache_preflight.py`: CPU metadata/fit transport through
  actual P6 stage_inputs/seal APIs; all16resident-bound YAMLs and640real B500
  queries; no teacher RGB transport or GPU model construction.
- `tests/models/fastwam/test_fastwam_cache_study.py`:29regressions/parameters,
  frozen-provider replacement, actual accepted control/new resident/source
  equivalence, every mismatch class denied, old archive proof, cross-Python
  AST fence, default-off byte identity, nativeB16 hit/tail/stall/cooldown and
  all8socket YAML switches over16persistent clients/192decisions.
- `tests/models/fastwam/test_mps.py`: the old MPS-only code seal now explicitly
  allows this later provider hook, and pins the entire new root source map.
- `models/fastwam/CACHE_STUDY.md`, README: status, evidence, exact CPU/GPU
  commands and limitations. Own receipts/measurements contain CPU outputs
  and copied metadata/fits only. REQUESTS/P3_to_P6.md records helper-resource
  correction, shared-test refresh and final handoff.

Interface: P1 contracts v1, existing default=None library_validator seam;
receipt r17.teacher_compatibility.v1, source proof r17.fastwam.library_sources.v1.
No shared contract edits or model norm/geometry/binding transplants.

Teacher checkpoint9f8304f7d001eec42e5c1774b424020f98049fc8e8e62449d4ce09686f807030;
accepted controlfa3f174e2301a675c5f3afab04114af5c359e49f66fa23c3601ae2877a62d929;
fresh residentf436df1f43a096f67b668e7154865b22543ca6fd6c7ef356bca7a0bbca891743.
All non-R17 content unchanged. Real key width49152+8raw state;64is fitted
projection width. Source receiptf0cba12ff45e85923da7bd119bab21b7ca325c504c2a5c7ae5e74ddcfbfe6565.
Future unreviewed source/weight generations fail instead of ignoring labels.

### P6 request → change → test → status

| Item | Change / proof | Status |
|---|---|---|
| 1 native cache/tails with resident batching | P6 clarified startup remains plain full.yaml; existing formal_batch_server factory handles hot-switch. Native16-client socket proof across8YAMLs retains same model/core, CP1only hits and empty tails | DONE CPU; no startup rejection relaxation |
| 2 content compatibility and frozen CPU/live hook | provider.py configure/registration and ops/library_compatibility/evidence; all16actual H100 CPU-sealed recipes accepted. Changed weights,norm,preprocess,pooling/grid,transform/code/protocol/K rejected | DONE CPU and actual physical asset seal |
| 3 accounting and GPU readiness | Native fixtures: HIT=(1,0,0)/0NFE; blind tail=(0,0,0)/0NFE; only confirmed Q3 stall gets=(1,1,10)/10NFE; fresh-anchor cooldown returns HIT. Runtime records all composition | DONE CPU; GPU smoke NOT EXECUTED by instruction |

### CPU commands and actual counts

All pytest commands used CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS=cpu,
OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,
PYTHONPATH=.:src:packages/openpi-client/src,.venv/bin/python,-p no:cacheprovider.
Results below are distinct commands, not added together as unique coverage.

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/models/fastwam \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/cache_study_fastwam_cpu.xml
#235passed,0failed/error/skip,34.15s; includes29cache regressions.

.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/models/fastwam exp/r17_integration/tests/common \
  exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/cache_study_full_cpu.xml
#588passed,2initial failures,0skip,533.80s. Both were stale Cosmos rejection
#expectations; owner edited them during this shared run. No Fast-WAM failures.

.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/ops/test_cache_study.py \
  -k metadata_fit_transport_and_cosmos_admission_are_cpu_only \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/cache_study_cosmos_refresh_cpu.xml
#Current tests2passed,31deselected,0skip,7.70s. Other355common/ops passed in full run.

.venv/bin/python -m pytest -q -p no:cacheprovider --run-manual \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py \
  tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py \
  -k 'not same_identity_same_initial_state_across_processes' \
  --junitxml=exp/r17_integration/tests/models/fastwam/receipts/cache_study_legacy_cpu.xml
#71passed,0skip,1realEGLtest deselected,7.17s.

.venv/bin/python -m exp.r17_integration.models.fastwam.ops.cache_preflight \
  --output exp/r17_integration/models/fastwam/measurements/cache_cpu_preflight_20261010_v4
#PASS:16YAMLs,640real B500queries,32official-A smoke episodes planned.

bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --prepare-only
#PASS CPU-only on H100, physical asset/source seal;16YAMLs admitted.
```

Default-off proof: test_no_library_io_or_math_change_on_default_off_provider
forbids library/source reads and compares normalized bytes/manifest; existing
B1 no-queue/file/metadata and native reference-loop tests passed. Source/CPU
aggregate receipt: tests/models/fastwam/receipts/cache_study_qualification_cpu.json.
Warnings were pynvml/Swig and DP transformer configuration, not skips/failures.

Initial local preflight used a different BLAS thread count and failed the
strict early-metric SHA. Re-running under P6 server_env's one-thread settings
passed; the artifact fence was not weakened. First H100 prep exposed3.12's
empty AST type_params field; fixed with semantic canonicalization and regression.
A subsequent bytes-constant serialization error was corrected before final
28/29cache and235full-owned CPU passes. Failed/raw attempts remain preserved;
only final PASS receipts are cited as qualification.

### Samples and actual matched CPU preparation

Local source/recipe sample: models/fastwam/measurements/cache_cpu_preflight_20261010_v4/
assets/cache_yamls/sp_chrg.yaml and l10_c.yaml; PREFLIGHT.json has per-task/
per-group corrector/Q3 decisions, resident/teacher identities and all source pins.
H100 actual native manifest and all16sealed YAMLs:
`/data/mw_finetune/eval/cfwp31010/assets/`.
Final PREPARED: `/data/mw_finetune/eval/cfwp31010/PREPARED.json`,statusPASS,
cpu_only=true,dispatch_started=false; timan107_MEMORY_ADMISSION.admitted=true.
Matched flags: H1002replicas,B16/5ms,MPSenabled metadata,group-commit driver;
timan10748,RAM2048/VRAM768MiB,N4,EGL0..7. No server/worker/MPS daemon started.

The exact helper's prior CPU PASS at mps_fastwam_libero_1010_165002_1cc26649
used RAM6144/defaultVRAM1024; it is a CPU model admission receipt but not a
matched fleet launch candidate. P6 owns correcting that helper; use cfwp31010.

### GPU-ready command (not executed) and limits

Coordinator after GPUs are available; planned32episodes, all8groups across
both suites, two/group/suite. Expected10–20GPUminutes is an unmeasured planning
estimate including resident loads/small smoke batches. CPU transport is extra.

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --prepared-run /data/mw_finetune/eval/cfwp31010 --run
```

Fresh equivalent flags/command are fully written in models/fastwam/CACHE_STUDY.md:
normal formal_eval.sh --model fastwam_libero --official-50 --cache-study STUDY.json
--server-node h100 --worker-node timan107 --workers48 --replicas2 --max-batch16
--batch-wait-ms5 --mps --worker-memory-mib2048 --worker-vram-mib768
--worker-max-episodes4 --worker-egl-devices0,..7 --memory-budget-gib70
--port23420 --driver-port23439 --seconds7200 --episode-seconds1800
--smoke --smoke-episodes2 --run. Those normal flags passed CPU parse_args.

Limits/pending owners: coordinator runs GPU smoke, checks stage/NFE/composition
and unchanged resident generations before promotion; GPU cache throughput,
numerical outputs and SR unmeasured. P6/coordinator must freeze the long horizon
NI selection before formal8000episodes; STUDY currently records fallbackC10.
If changed, regenerate/refit/reseal. P6 fixes helper worker reservations.
The static source receipt admits two explicit native content generations and
physically verified unchanged weights/producer; future source changes require
review/reseal. CUDA/Python dependency version or physical-policy parity beyond
previous batch qualification is not newly claimed by this CPU task.
