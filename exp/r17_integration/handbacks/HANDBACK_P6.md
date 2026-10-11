# HANDBACK P6 — debug/library/offline/analysis/deployment qualification

2026-10-09, shared checkout `/home/weiland/projects/openpi`, branch `Ziyang`.
P6 CPU implementation delivered. **142 distinct tests passed, zero skips/failures
in the final three runs. Zero GPU use, deployment, checkpoint loading, commits or pushes
by P6.** G3 real-model qualification remains pending; this is not an all-feature
GPU or policy-quality certificate.

## 1. Owned files and actual change

All implementation/test edits are within P6-owned paths. No P6 shared framework,
model, benchmark, R16, harness or training-tree edits. Existing shared-file changes
in this checkout belong to other packages; P6 did not revert or modify them.

| Owned directory | New files / purpose |
|---|---|
| `debug/` | `capture_bridge.py`: opt-in BlockWriter/StepTrace/physical transport bridge, DecisionSink, PhysicalSink, native-product ingestion, per-connection CaptureFactory. `reader_profiles.py`: checksum/closed/partial-aware NPZ/H5 readers, stock collected-H5 callback bridge, accepted-attempt and manifest joins. `validate.py`: v1 model/episode/decision identity and numeric serialization. `analysis_profiles.py`: geometry-driven decision/physical cards and HTML/PNG/CSV/JSON. |
| `search/` | `store.py`: READY dense/ragged read-only memmaps. `build_library.py`: closed complete nested libraries, split checks, active arrays, prev/next/progress, stock CacheEntry artifacts and receipts. `fit.py`: library read allowlist, LOEO, P1 math adapter, numeric fitted-array save/reload preserving memory order. `replay.py`: fit/reset/query and exact-result comparison. `freeze.py`, `audit.py`: immutable dependency and accepted-ledger audits. `analyze.py`: quality/missing/errors/task/group/cost/bootstrap and offline action scores/sweeps. |
| `ops/` | `plan.py`, `build_client_bundle.py`, `deploy.py`, `serve_registry.py`, `sync_back.py`, `smoke.py`: CPU blueprint/Unix paths, Python3.11 SHA payload, explicit tether lanes/retries, held ports/endpoint compatibility, quiescent streaming snapshots, opt-in deadline runner. `coverage.py`, `coverage.csv`, `coverage.json`: every INVENTORY ID. `cpu_sample.py`, `samples/cpu_pipeline_v1/`: retained synthetic pipeline. `RUNBOOK.md`: operational and qualification instructions. |
| `tests/debug/` | `fixtures.py`, `test_capture.py`, `test_framework_bridge.py`, `test_native_products.py`, `test_analysis.py`. |
| `tests/search/` | `test_search.py`. |
| `tests/ops/` | `test_ops.py`, `test_coverage.py`, `conftest.py`; final XML and verification/source/default-off receipts in `receipts/`. Scratch kept for audit, excluded from test collection. |
| Exceptions authorized by task | `REQUESTS/P6_to_P1.md` … `P6_to_P5.md`; this handback. |

28 new Python files, 3,104 lines including tests. Exact file hashes/line counts:
[`OWNED_FILES.json`](../tests/ops/receipts/OWNED_FILES.json). There is no tracked
P6 diff against HEAD because the whole R17 tree was initially untracked; these
are additions. The root `.gitignore` ignores JSON/NPZ/pickle results: coordinator
must explicitly retain selected receipts/sample artifacts when assembling the
final change. No staging/commit was performed.

## 2. Interface version and binding decisions

Consumes **`r17.contracts.v1`** from P1, including actual ModelSpec,
DecisionCapture.native_snapshots/model_capture, RequestContext and execution
callback. P6 envelope `r17.p6.v1`; store `r17.store.v1`; fit receipt
`r17.fit.v1`; fitted arrays `r17.fitted_recipe.v1`; freeze `r17.freeze.v1`.

`model_binding(spec, protocol, sampler=...)` retains the complete original
ModelSpec and an explicit **actual selected sampler**. A K2 capture can therefore
retain a model whose default is K100, while refusing admission into a K100
library. Normalization/layout/real cameras/key source/preprocess/schedule/code
metadata all participate in the binding. No model-name substitution or fake wrist.

Shared mechanisms are reused: BlockWriter, safe NPZ/JSONL, StreamSink filename
and spill protocol, optional H5TraceSink.record_step, DataCollector H5 schema,
stock CacheEntry/CachePayload/InMemoryBackend and P1 LayoutRecipe. No duplicate
five-layer equations or new WS/conductor protocol.

Stock loader drops unknown top-level fields. The artifact embeds `r17_binding`
**inside its model mapping**, as requested by P1. P6's verified loader also installs
the numeric-store manifest and actual native sampler. Native warm data remains
in that store; stock payloads are full-hit only, `native_warm_supported=False`.
Legacy uniform-flow metadata is not used to describe DDPM or shifted flow.

P1 now supplies RequestContext with execution acknowledgments. P6's local
CaptureFactory additionally creates a distinct DecisionSink per connection over
binding-shared writers. Same-bundle simultaneous seq0/ack interleaving passes.
Actual control counts arrive through an acknowledgment log; absent acks remain
pending/unknown rather than zero execution.

For P2's hashed benchmark protocol identity, library callers must explicitly use
`split_policy='mt50_init_v1'`. Literal MT50 protocol IDs select this automatically.
No benchmark is guessed from a content hash.

## 3. Default-off / legacy proof

`CaptureBridge.enabled=False` is default: no writer construction, directory,
array coercion/copy, forward or RNG call. `CaptureFactory.enabled=False` delegates
to the original P1 factory. No import registers a provider or starts a process.

Passing tests verify a poison-array off hook, fake forward/RNG/action-byte parity,
two actual stock WS clients with plain/debug bundles, same-bundle CPU thread
isolation, and zero model calls on acknowledged tails. Original writer/observer,
transport, per-step, identity and router tests also passed.

13 reused source files, including frozen harness/R16 files, were compared with
`git show HEAD:<path>` and were **byte-identical**. Detailed proof:
[`LEGACY_DEFAULT_OFF.json`](../tests/ops/receipts/LEGACY_DEFAULT_OFF.json).
This does not claim byte identity for shared files edited by P1. Passing legacy
regression tests constrain their default behavior, while P1 owns their certificate.

## 4. CPU commands and actual receipts

All tests used CUDA hidden, JAX CPU, BLAS/OMP=1, no bytecode/cacheprovider.
Final results are the following **three disjoint suites only**; earlier retries
and diagnostic XML are retained but excluded from totals.

| Final receipt | Exit / passed / skipped / failed | Scope |
|---|---|---|
| [`p6_handoff.xml`](../tests/ops/receipts/p6_handoff.xml) | 0 / **90** / 0 / 0; 7.90s | P6 four fake models × two benchmarks; actual P1 WS and layers; P2 physical callback; P4 saved CPU arrays; payload, sync, retry, port, audit and reports. |
| [`p6_legacy_handoff.xml`](../tests/ops/receipts/p6_legacy_handoff.xml) | 0 / **49** / 0 / 0; 11.77s | Existing identity/per-step/router, debug writer/observer, real localhost stream ACK/spill/fences, EGL lifecycle mocks and old snapshot tests. |
| [`p6_model_handoff.xml`](../tests/ops/receipts/p6_model_handoff.xml) | 0 / **3** / 0 / 0; 0.89s | P3 two-benchmark stub native capture through closed NPZ/H5; P5 stub capture through store, stock artifact, actual YAML preload and shared CP1 HIT. |

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search \
  exp/r17_integration/tests/ops -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_handoff \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_handoff.xml -q
```

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_legacy_handoff \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_legacy_handoff.xml -q
```

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/models/fastwam/test_native.py::test_native_capture_consumed_by_p6_closed_npz_h5_bridge \
  exp/r17_integration/tests/models/groot/test_p6_bridge.py -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_handoff_interop \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_model_handoff.xml -q
```

AST parsed all 28 owned Python files. CPU sample command exited0:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python \
  -m exp.r17_integration.ops.cpu_sample \
  --output exp/r17_integration/ops/samples/cpu_pipeline_v1
```

Use a **new** output/scratch directory for a later run; existing samples remain
immutable. Pytest warnings were pynvml/Swig deprecations, not GPU probes.
Source content at handoff is recorded in
[`SOURCE_HASHES.json`](../tests/ops/receipts/SOURCE_HASHES.json). Other agents are
still editing; recheck source hashes if rerunning after further integration.
P2 subsequently changed only `conductor/worker_entry.py` among those hashed
dependencies. Its actual isolated payload test was rerun: exit0, 1 passed,
0 skips/failures in 0.54s, receipt
[`p6_worker_refresh.xml`](../tests/ops/receipts/p6_worker_refresh.xml). This
repeat is excluded from the 142 distinct total. Command used the same CPU
environment above, target
`exp/r17_integration/tests/ops/test_ops.py::test_p2_actual_worker_payload_python311_model_free`,
`--basetemp=exp/r17_integration/tests/ops/scratch_worker_handoff`,
`--junitxml=exp/r17_integration/tests/ops/receipts/p6_worker_refresh.xml`,
`-p no:cacheprovider -q`. Current hashes and added P3/P5 interoperability
dependencies are in
[`SOURCE_HASHES_REFRESH.json`](../tests/ops/receipts/SOURCE_HASHES_REFRESH.json).
Aggregate counts and unique-case evidence are in
[`VERIFICATION_RECEIPT.json`](../tests/ops/receipts/VERIFICATION_RECEIPT.json).

## 5. Hook table / evidence limits

| Hook | Passing CPU evidence | Remaining owner |
|---|---|---|
| Model observe-only capture | P1 actual DecisionCapture, snapshots/RNG/model_capture; on/off bytes and stage counts unchanged. | P2/P3/P4/P5 actual checkpoint capture parity/G2. |
| Dynamic NPZ/H5 readers | All four fake model geometries × LIBERO/MW exact dtype/byte roundtrip; wrong checkpoint/norm/layout/camera/sampler rejected; stock DataCollector H5 callback tested. | Model owner supplies real feature-builder and attested source manifests. |
| Native physical sink | P2 actual runner loop with fake native model/data: float64, clipped/requested actions, reset/settled/terminal, early success **5+2** applied controls. | P2 real renderer/contact/goal maps/restore; raw predicates remain unsupported. |
| P4 trace product | Saved real-model **CPU** DP LIBERO/MW products consumed without loading model; integer grids/posterior uint8 preserved, K2 separated from defaultK100. | P4 real final MW export and GPU gate. |
| P3/P5 trace interoperability | P3 LIBERO/MW stubs preserve native K2 grid/keys/actions through shared writer/H5 without extra model calls; P5 stub preserves H16×32/active4, tokens/snapshots/ack, stock YAML preload yields exact CP1 HIT bytes. | P3/P5 real checkpoint/production geometry/BF16 GPU gate. |
| Library / stock artifact | Complete episode quotas, active channels, links/progress, READY/checksums, stock load and exact model/sampler receipt. | Coordinator real teacher collection; native warm executor model owners. |
| Shared fit | P1 fit_library and published fit_library_view; whole-episode LOEO; >=3/task corrector; gripper unchanged; numeric save/reload action/donor **byte parity**. | P1/model owners full legacy-math equivalence and real recipe+WS G3. |
| Replay / audit | fit/reset/query metadata/actions exact; real WS staged bytes; freeze mutations all rejected; exact accepted done/failed/error/late attempt handling. | Coordinator real recorded resident+WS campaign replay. |
| Analysis | Missing costs unknown; valid same-model normalization; paired task-data identity; dynamic active/gripper/native-grid cards and four export formats. | Coordinator GPU pricing/G4; P2 semantic analyzers. |
| Ops | Actual isolated P2 Python3.11 worker import with no model islands; shared transport loopback; EGL mocks; bounded exit75 retries/held ports/owned stop mocks; streaming SHA snapshot and idempotent/busy/tamper tests. | Coordinator remote deploy/network/renderer capacity; model owners cancellation-safe GPU hook. |

Coverage: **65/65 INVENTORY IDs** in [`coverage.csv`](../ops/coverage.csv) and
[`coverage.json`](../ops/coverage.json), classified F/M/B with reuse/hook/pending
scope and owner. Every GPU-qualified field is false. Single-camera second-view
ablation, π-MW H5 commit10, and GR K1 interior warm snapshots have explicit
not-applicable reasons. Cosmos-MW and new-model split stay pending.

## 6. Samples

[`SAMPLE_RECEIPT.json`](../ops/samples/cpu_pipeline_v1/SAMPLE_RECEIPT.json): six
synthetic complete episodes, **24 decisions**, 24 shared-corrector LOEO rows,
exact fit reload. No policy SR or measured GPU pricing claim.

- [`captured.h5`](../ops/samples/cpu_pipeline_v1/captured.h5): one real-view
  fixture key64/state4, H16×4 actions, native integer grid `[99,98,97,96,95]`,
  snapshot and private RNG bytes. Checkpoint/norm are explicit fixture identities.
- [`library/READY.json`](../ops/samples/cpu_pipeline_v1/library/READY.json): source
  SHA, binding, active arrays, complete episode rows and links/progress.
- [`fit/READY.json`](../ops/samples/cpu_pipeline_v1/fit/READY.json): read-only
  fitted P1 arrays, LOEO evidence and preserved C/Fortran memory order.
- [`analysis/report.html`](../ops/samples/cpu_pipeline_v1/analysis/report.html),
  `success.png`, `tasks.csv`, `report.json`: synthetic quality labels, missing
  measured timing unsupported, explicit unsupported physical semantics.

P4 producer sample sources remain in its owned model directory; P6's tests read
and roundtrip them without altering them or loading weights.

## 7. Future GPU-ready commands — NOT EXECUTED

The coordinator must allocate an exclusive slot after training ends, attest an
immutable checkpoint and model G2, and use one model per <=900s slot. P6's runner
defaults to a CPU dry run. It requires the coordinator's existing GPU-slot lock
and a model/benchmark qualified callable before execution.

```bash
MODEL_ISLAND_PYTHON -m exp.r17_integration.ops.smoke \
  --model MODEL_ID --checkpoint-sha IMMUTABLE_CHECKPOINT_SHA \
  --hook QUALIFIED_MODULE:qualify --gpu-slot-token COORDINATOR_GPU_SLOT_LOCK \
  --deadline 900 --receipt NEW_MODEL_G3_RECEIPT.json --execute
```

The callable is a pending model/benchmark handoff, **not a fabricated passing
implementation**. Its contract and bounded phase plan are in
[`RUNBOOK.md`](../ops/RUNBOOK.md): 0–3min load+warm; 3–8min two mechanical
full/retrieval episodes; 8–11min saved-observation shadow/native suffix with
extra cost; 11–13min H5→tiny artifact→new bundle→forced HIT; 13–15min owned
shutdown, SHA sync, accepted joins and reports. Timeout produces partial and
stops only the runner's owned child; forced stop is explicitly recorded.

Concrete first π-MW serving command for that hook (new reserved free port from
the owned receipt; all variables supplied by coordinator):

```bash
PI_ISLAND_PYTHON -m exp.r17_integration.models.pi05.serve \
  --checkpoint IMMUTABLE_PI_MW_CHECKPOINT --device cuda:0 \
  --port OWNED_FREE_PORT --config SEALED_PI_BUNDLE.yaml \
  --debug-directory NEW_PI_DEBUG_DIRECTORY
```

Concrete P4 G2 probe, if required before G3, also **not executed by P6**:

```bash
timeout -k 5s 900s DP_ISLAND_PYTHON -m exp.r17_integration.models.dp.probe \
  --checkpoint IMMUTABLE_DP_CHECKPOINT --domain metaworld --device cuda:0 \
  --deadline-seconds 840 --output NEW_DP_G2_DIRECTORY
```

Run separate slots for π-MW, GR-MW and each Fast/DP benchmark/checkpoint. This
does not schedule work or authorize GPU use today. Full500 quality, calibrated
>=50-sample pricing, batching/replicas/split throughput require later slots.

## 8. Known limits and pending owners

| Pending / limitation | Owner |
|---|---|
| G1/G2/G3 real GPU model+debug+library+resident recipe+hot-bundle+physical loop and cancellation-safe qualification callable; CPU fixtures do not establish SR or latency. | P1/P2/P3/P4/P5 + coordinator |
| Raw/normalized model snapshot dtype is preserved as supplied by producer. If a model trace adapter casts BF16 to float32, P6 cannot recover original BF16 bits; producer must supply dtype/bits for that qualification. | Model owners |
| Full twin trace, native warm-start/reset, CP2/online-RIT/shadow/routing and advanced combinations remain behind model/capability certificates; stock artifact warm is explicitly disabled. | P1 + model owners |
| Native MT50 contacts/goals/task predicates/grasp/drop/restore semantic analyzers and real EGL/rendering throughput are not certified. Raw cards retain unsupported/pending tool status. | P2/P6 + coordinator |
| Mixed-K debug in a default-bound writer is diagnostic only; library building checks actual sampler/grid. Use P6 CaptureFactory or sampler-separated captures for collection. | P2/P3/P4/P5 |
| Port handoff retains a release/bind race because stock WS does not inherit the reserved socket. Retry a new reservation if startup bind fails. | Coordinator/P1 |
| Remote deployment and tether transfers were mocked, not executed; actual client dependency island/SHA/EGL attestation required. Snapshot driver must honor `.eval.lock`. New content installs to a new SHA directory; conflicting local files are rejected. | Coordinator |
| Generic sweep/offline errors are not closed-loop SR. Advanced risk/oracle/profiler/physical forensic tool families need their semantic adapters and own receipts; matrix does not claim them all qualified. | P6/P1/P2 + coordinator |
| Cosmos-LIBERO optional stock-WS protocol bridge is a spec only. Cosmos-MW remains unusable. Split has only CPU blueprint/path tests and preserves existing restrictions. | Coordinator/D, P1/model owners |
| Immutable final teacher library/dev-freeze/test cohorts, complete500 outcomes, paired bootstraps and model-specific pricing. | Coordinator (G4) |

Incoming P1/P2/P3/P4/P5 requests were read repeatedly. P1's binding, native-capture
and acknowledgment requests are consumed; P2 callback/profile and P4 fixture
handoffs are CPU tested, including the final P3/P5 interoperability tests P6 ran.
Remaining model-GPU and advanced semantic qualifications
are explicitly assigned above. No running training process, tmux session,
prohibited data tree or GPU was touched by P6.

## Fix round 1

Read REVIEW_A and REVIEW_B completely, plus their supplied repros and updated
P1/P2/P3/P4/P5 requests. The initial sections above are historical; this section
supersedes their CPU counts, serving imports, compact artifact binding and P3
test filenames. **P6-owned fixes complete on CPU: 391 distinct passing test
cases across three disjoint suites, zero skips/failures. Zero GPU use, checkpoint
loading, remote deployment, commits or pushes.** GPU inflation/latency/SR and
out-of-scope qualification issues remain explicitly pending below.

| Finding | Owned change | Regression / actual witness | Status |
|---|---|---|---|
| A6: real per-decision cost analysis | `reader_profiles.cost_fields/merge_decision_ledgers` consume P1 `cost_ms`, `stage_ms/stage_timing_ms`, timing scope and P2 `server_timing.infer_ms`, identity-complete per-step rows and final client execution rows. `summarize` and `audit_campaign` accept these inputs; incomplete decision/ack coverage remains unsupported. | `test_review_costs::test_actual_server_timing_and_final_client_ack_are_consumed`; `test_review_runtime::test_p1_p2_actual_cost_ledger_and_terminal_ack_reach_analysis[False/True]` runs actual P1/P2 strict path with fake CPU weights/native env, measured clocks and **5+2** final controls; plain pricing available, captured pricing excluded. Real localhost WS terminal-control test now verifies final ack. | Fixed; GPU timing/pricing qualification pending coordinator. P1 owns timing producer/tail state-only transform; P2 owns terminal transport. |
| A5: server/worker debug deployment | Published `CoreCaptureFactory` and `make_capture_sink_factory(provider, directory)` with `.bridges/.close()`, episode metadata from built core, actual sampler binding and separate connection sinks. Existing CaptureFactory also defaults to core-aware metadata. `PhysicalCaptureFactory` requires attested `r17_model_spec`, actual sampler/spec or explicit K+resolver, protocol and provenance; debug client payload includes reader dependency. | `test_review_capture::test_core_aware_server_capture_uses_episode_metadata_and_actual_sampler`; `test_review_runtime::test_actual_groot_serve_assembly_publishes_worker_capture_metadata` executes actual GR serving assembly with stub CPU model; `test_review_worker_entry::test_production_worker_debug_uses_attested_sampler_and_server_episode` executes actual P2 worker main with mocked renderer/socket boundaries. Fresh installed Python3.11 payload writes/reads real float64 blocks without torch/JAX/model islands. | Fixed P6 bridge; P2/P5 consumed production hooks/metadata. Renderer/GPU/remote deployment unqualified. |
| A10 / A7 / B2: capture provenance joins | Canonical aliases must agree; model capture metadata cannot contradict RequestContext; physical metadata cannot overwrite event provenance. Cost/ack/journal joins use full campaign/arm/bundle/protocol/task/init/task-data/UID/attempt/seq. Default dispatch generation hashes full episode provenance while retaining stock `episode_key:dispatch_gen:decision_seq` wire format; wire collisions rejected. | Four wrong cost/ack identity cases, conflicting original-index alias, accepted UID+attempt with wrong generation, conflicting model metadata, and repeated UID across arms. Actual P2 strict run/worker deliberately use bundle_generation different from bundle_id and still produce exact model/physical joins; actual sampler K2 physical binding equals model binding, not default K4. | Fixed P6 validation/join/storage; P1/P2 own formal envelope/seed and LIBERO global task naming. |
| A11: tail zero-call audit | Audit checks `tail` and every `*_tail`, including `library_tail` and `cache_tail`, using tuple/list-independent zero stage calls and actual NFE. | Eight `test_every_tail_source_is_audited` cases cover extra vision and extra NFE across all four sources. **Four missed library/cache cases failed old code.** | Fixed. |
| B9: debug timing inflation | Bridge measures host preparation and original writer queue wait, marks capture enabled/ineligible. Analysis preserves reported core preparation/sink/server wall scopes, keeps unknown overhead as None, excludes any captured decision from whole price total even with erroneous eligible=True. Full reference must explicitly be plain/eligible too. No subtraction of debug overhead to invent plain timing. | `test_bridge_measures_preparation_and_marks_cost_as_capture`; captured candidate/reference and unknown-overhead regressions; actual capture-on/off P1/P2 CPU paths. Raw absolute diagnostic cost remains visible, normalized/pricing totals are None for captured/unknown inputs. | Fixed flags/host measurement and pricing exclusion. Actual GPU debug-vs-plain inflation measurement pending model owners/coordinator; core preparation excludes native stage capture and is labelled accordingly. |
| Q_to_P6: production BF16 observer failure | `numeric_arrays` stores lossless float32 values plus exact uint16 companion datasets; records include original name, native/storage/bit dtype and encoding by arrays/rawkeys namespace. Nested snapshot noise/RNG tensors use named numeric arrays, retaining scalar metadata. Torch import stays conditional on an existing BF16 tensor. | `test_bfloat16_capture` covers all65,536 bit patterns, noncontiguous tensors, companion collisions, closed NPZ/H5 snapshot/key/noise roundtrip, unchanged native tensors, action bytes and RNG. Old4 failed/1 passed; fixed5 passed. | Fixed CPU storage/observer boundary; real GPU BF16 timing/resume still pending model owners/coordinator. |
| A2 / A4 migration in P6-owned consumers | Config/server tests use P1 local `common.config` / `common.websocket_policy_server`; stock registry remains unchanged. Stock artifact compact nested binding now includes P1 `retrieval_model_identity`, keeps teacher sampler distinct from MISS K, and never enables native warm cross-K. | Actual stock YAML preload validates K4 teacher at K2 MISS with explicit teacher digest; P5 stock artifact/HIT regression rerun passed. P6 also reproduced nullable field_similarity parser failure and requested P1's fix; actual full YAML WS load now passes. R16 read-only verify and 17 source byte checks pass. | Fixed P6 consumers; P1 restored sealed shared files. |
| B3/B5/B6/B10/B11 evidence/ops scope | Runbook/matrix state DP K1 lossy research/quality pending, R17-LIBERO protocol differs from published Fast-WAM reference, GPU B>1 tolerance/batch composition unqualified, and synthetic views not paired-camera qualification. | Matrix still enumerates 65 IDs with every GPU flag false; direct incoming P4 request forwarded to P1/coordinator for qualification runtime fabricated wrist. | Documentation/status corrected. `qualification/` is outside P6 ownership; B11 runtime change assigned P1/coordinator/Q. |

Changed implementation: `debug/{capture_bridge,reader_profiles,validate}.py`,
`search/{analyze,audit,build_library}.py`, `ops/{build_client_bundle,coverage}.py`,
and `ops/RUNBOOK.md` plus generated coverage. Added six review regression files
in owned tests; updated existing fixtures/WS/report/payload tests. No P6 edits
to common, model, benchmark, conductor, frozen R16/harness, or training source.
There are now **34 owned Python files / 4,072 lines including tests**; exact
hashes/line counts are in
[`OWNED_FILES_FIX1_FINAL.json`](../tests/ops/receipts/OWNED_FILES_FIX1_FINAL.json).
Interface remains `r17.contracts.v1` / `r17.p6.v1`; P1's local
`r17.recipe.v1` and extended compact artifact binding are consumed.

Regression proof: the focused new tests ran against the old P6 implementation
before fixes: **20 failed, 4 passed, exit1**, retained
[`p6_fix1_red.xml`](../tests/ops/receipts/p6_fix1_red.xml). Failures include missed
library/cache tail checks and silent conflicting/colliding provenance, plus
missing cost-join/deployment/timing interfaces. Four existing tail variants
already behaved correctly. Later targeted/full intermediate failures exposed
shared-file migration and stale fixture provenance; those receipts are diagnostic
and excluded from final totals.

Q's later incoming BF16 repro is also resolved upstream in P6. Its separate old-code
run produced **4 failed / 1 passed / exit1**
([red receipt](../tests/ops/receipts/p6_fix1_bf16_red.xml)); the same five tests
then passed ([green receipt](../tests/ops/receipts/p6_fix1_bf16_green.xml)) and
are included once in the full 129 below. A P2 identity-guard update exposed stale
model/norm/layout strings in the worker CPU fixture; it now supplies the actual
server ModelSpec. The failed intermediate refresh is diagnostic, excluded; the
final full run passes the production guard. P1 added one more default-off test,
so the fresh legacy run has 259 cases. Historical successful runs are retained
but not added to these distinct counts.

| Final receipt | Actual result | Scope |
|---|---|---|
| [`p6_fix1_bf16_full.xml`](../tests/ops/receipts/p6_fix1_bf16_full.xml) | exit0, **129 passed / 0 skipped / 0 failed**, 10.17s | Full P6 debug/search/ops suite, including all review regressions and deployed worker/GR assembly. |
| [`p6_fix1_bf16_legacy.xml`](../tests/ops/receipts/p6_fix1_bf16_legacy.xml) | exit0, **259 passed / 0 skipped / 0 failed**, 22.45s | Existing writer/observer/transport, identity/per-step/router/worker, cache config/storage/schedule and P1 default-off checks. |
| [`p6_fix1_bf16_interop.xml`](../tests/ops/receipts/p6_fix1_bf16_interop.xml) | exit0, **3 passed / 0 skipped / 0 failed**, 0.85s | P3 two native NPZ/H5 bridge cases and P5 capture→stock YAML preload→HIT. |

All commands below were executed with GPUs hidden and CPU/BLAS settings:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search \
  exp/r17_integration/tests/ops -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_fix1_bf16_full \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_fix1_bf16_full.xml -q
```

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py \
  tests/cache/test_cache_storage.py tests/serving/test_policy_identity.py \
  tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_fix1_bf16_legacy \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_fix1_bf16_legacy.xml -q
```

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/models/fastwam/test_fastwam_native.py::test_native_capture_consumed_by_p6_closed_npz_h5_bridge \
  exp/r17_integration/tests/models/groot/test_p6_bridge.py -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/ops/scratch_fix1_bf16_interop \
  --junitxml=exp/r17_integration/tests/ops/receipts/p6_fix1_bf16_interop.xml -q
```

P3 renamed its test files to unique `test_fastwam_*` names. One attempted old
interop path produced exit4/no tests; not counted as passing or skipped. All 34
owned files were AST parsed. Seventeen reused/shared source files are exactly
HEAD bytes; proof in
[`LEGACY_DEFAULT_OFF_FIX1_FINAL.json`](../tests/ops/receipts/LEGACY_DEFAULT_OFF_FIX1_FINAL.json).
`python -m exp.offline_search.rounds.r16.analyze verify` was run read-only with
the same CPU environment and exited0: **DESIGN_VERIFIED; no outcome reads**.
Source paths were checked against the prohibited roots before that command.
Receipt: [`p6_fix1_r16_verify.json`](../tests/ops/receipts/p6_fix1_r16_verify.json).
No R16 result/source file was written.

Fresh synthetic pipeline was generated (exit0, same CPU environment):

```bash
.venv/bin/python -m exp.r17_integration.ops.cpu_sample \
  --output exp/r17_integration/ops/samples/cpu_pipeline_fix1_v1
```

Its [receipt](../ops/samples/cpu_pipeline_fix1_v1/SAMPLE_RECEIPT.json) records
6 complete synthetic episodes, 24 decisions, 24 LOEO rows, byte-exact fit reload,
updated stock binding, and HTML/PNG/CSV/JSON artifacts. Historical samples remain
unchanged. They are synthetic schema/math demonstrations, not SR/GPU costs or
real checkpoint certificates. The pre-BF16 sample retains its artifact hashes;
BF16-specific NPZ/H5 products are tested by the new regression. Native source
identities from older P4 diagnostic
samples are not substituted for its corrected sampler release.

Aggregate verification:
[`VERIFICATION_FIX1.json`](../tests/ops/receipts/VERIFICATION_FIX1.json);
source dependencies:
[`SOURCE_HASHES_FIX1_FINAL.json`](../tests/ops/receipts/SOURCE_HASHES_FIX1_FINAL.json).
Qdrant mock/config version and pynvml/Swig deprecation warnings appeared in
legacy tests; none is a GPU qualification receipt.

Remaining owners: model owners/coordinator must measure actual capture-off vs
capture-on GPU timing with fixed paired observations, retain host/GPU scopes and
batch membership, and run the previously documented <=900s qualification slots
one model at a time. **No GPU command was executed in this round.** P1/coordinator/Q
must replace fabricated wrist in out-of-scope `qualification/model_runtime.py`
with P4's true paired-frame helper and reseal assets (B11); P4/coordinator own K1
quality/descent admission. Existing pending native warm/twins/advanced routing,
semantic physical analyzers, remote rendering/deploy and full G4 experiments
remain pending as in section8. P6 does not claim those resolved by these CPU fixes.

## LIBERO library pipeline

2026-10-09, P6-owned implementation and CPU qualification. **428 distinct CPU
cases passed: 166 P6 +259 legacy/default-off +3 model interop, zero failures,
errors or skips. Zero GPU use, remote deployment, real collection, commit/push
by P6.** The deliverable is the generic library sibling
`bash exp/r17_integration/ops/library_campaign.sh --model BUNDLE --suite SUITE`.
It calls the existing formal resident/proxy/conductor launcher. No π0.5/GR00T
collection is added. [Operator details](../ops/LIBERO_LIBRARY.md),
[verification receipt](../tests/ops/receipts/LIBERO_LIBRARY_VERIFICATION.json).

Owned files and changes:

| Files | Change |
|---|---|
| `ops/library_alignment.py`, `ops/library_init.py` | Read-only historical B500/R10 audit, all per-array identities, SHA-only immutable private B input transport. |
| `ops/library_campaign.py`, `ops/library_campaign.sh`, `ops/LIBERO_LIBRARY.md` | One-suite teacher campaigns, two-episode smoke, dedicated233xx ports, full collection/build/subsets/fits, CPU `--finish`, Fast GPU A/B admission and optional measured-cost knob calibration. |
| `debug/library_capture.py` | Connection-local DecisionSink/CaptureBridge allowlist before serialization; exact paired uint8 adapter-input images; full-native/cache-off guard. |
| `search/teacher_library.py` | Bounded streaming500 parent, suite-local numeric IDs/links/progress/outcomes, row-index views, P1 fit/sealed reload, selected-library R11 knob products. |
| `debug/reader_profiles.py`, `search/fit.py` | Optional bounded NPZ cache (default None); compact LOEO receipts (default False); preserve explicitly supplied numeric library IDs. Existing stores retain their previous mapping. |
| `ops/formal_{eval,driver,server,workers}.py` | Default-off `campaign.library` branches using existing CPU seal, resident factory, proxy, dispatch, B probes, closed capture/sync and aggregation. Stock launcher behavior stays unchanged. |
| `tests/debug/test_library_capture.py`, `tests/search/{library_fixture,test_teacher_library}.py`, `tests/ops/{test_library_campaign,test_library_finish}.py` | 33 new behavioral cases, including actual formal-server composition with CUDA calls replaced by CPU stubs, B transport, all six real P1 fits and tiny closed sync/finish. |
| `ops/samples/libero_library_alignment/`, `tests/ops/receipts/library_*.xml`, `REQUESTS/P6_to_P1.md` | Actual audits, disk/sample/verification receipts; P1/D interface requests. |

The verification receipt hashes all18 listed code/test/document files and
AST-parses16 Python files. Interfaces remain `r17.contracts.v1`, `r17.p6.v1`,
`r17.store.v1`, `r17.fitted_recipe.v1`; new subset schema is
`r17.row_index_view.v1`. No P6 edits to src/common/models/benchmarks/conductor,
frozen R16/harness, training directories, or foreign processes/ports.

**B-pool provenance and alignment result.** Actual source directories:
`/home/weiland/projects/openpi/exp/common/data/db_init/libero/libero_spatial`
and `.../libero_10`. The historical records are
`exp/offline_search/closed_loop/ops/h100/bpool_<suite>.yaml`; their legacy field
`apool_dir` names these B inputs, while `init_pool` explicitly says B.
Spatial rollup SHA is
`f35ae5439028a3cf5279271b34db2b58a0ad842b53f58e7162ef17de5840d478`;
L10 is `fa993ce2042942a79941960eba23d9186f2b835c2fc0ba1e2f47e7d308d00b14`.
All20 input files match their pinned historical SHA. All1,000 selected arrays
have recorded dtype/shape/byte SHA; the worker receives the same files and
checks the same selected-array task-data hash before inference.

| Suite / receipt | Episode alignment | Missing GR indices recovered | Audit digest |
|---|---:|---:|---|
| [Spatial](../ops/samples/libero_library_alignment/libero_spatial.json) | 500/500 π =GR =ours; 500 arrays | 464, via episodes+rs SHA-bound R11 identity certificate | `db4603b84e34a79f44f4c4b3b5eec9582b349cb4a50f7d9472195cf0ae87c1ae` |
| [L10](../ops/samples/libero_library_alignment/libero_10.json) | 500/500 π =GR =ours; 500 arrays | 0 | `e20ad5f88ee967e2f5a8273783c52e7e82033cdc5d95d32eaead79655ee5092c` |

References are π `bpool_cs` and GR `bpool_all` beneath the owner's
`/home/weiland/trace_runs/offline_search_store/library/`. All24 R10 subset
files and their `permutation_by_task` match seed20261002 and both models.
The library ID is suite-local `task*50+init`, including L10 where the serving
canonical task IDs are30..39. Both B directories are installed so the stock
worker's spatial import/environment probe cannot fall back to A during L10.
One trial/attempt per init, retries0. Failed ordinary episodes are retained.

Input-byte equality is proved by the original pinned input-file bytes and the
same original array index. **Historical H5 does not record the full reset input
array**: simulator states are post-settle. Do not present them as independent
full-reset equality. The audit additionally checked all1,000 π physical H5
reset metadata records (task/init/seed7/settle10) read-only at the explicitly
referenced collection's matching `save_traj` root. No simulator/GPU reset or
A-pool fit read was performed. Reproduction, both executed exit0/PASS:

```bash
for suite in libero_spatial libero_10; do
  env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -m exp.r17_integration.ops.library_alignment --suite "$suite" \
    --trace-root /archive/openpi/ablation_study/cache_size/save_traj \
    --output "exp/r17_integration/ops/samples/libero_library_alignment/${suite}.json"
done
```

**Capture/build/fits.** Full native sampler is sealed from each bundle, with
cache/checkpoints/retrieval/recipe off: DP100/replan5, Fast10/replan5,
Cosmos5/replan16. Existing native benchmark/table identity is preserved;
B content is bound separately. Images are the two exact uint8 frames as received
by the R17 model adapter (`received_wire_uint8`), before its internal transforms.
Only FeatureSpec keys, normalized chunk, robot state and needed identities/
execution/native-count metadata reach the shared writer. Snapshots, RNG,
latents/video/text tensors, extra diagnostics and physical actions are dropped
at write time. Normalized actions suffice for library fitting. Accepted closed
outcomes, final acks and complete sequences supply success/progress/links.
Capture remains excluded from pricing; no GPU timing inflation claim is made.

Full parent output:
`/data/r17_library/<bundle>_<suite>/bpool/READY.json` and dense `.npy` arrays.
Views at `bpool/subsets/size_N/` contain only `VIEW.json` and `row_indices.npy`.
Sizes50/100/200/300/400/500 use unchanged R10 memberships. Each fit uses only
that view's keys/state/actions and complete episodes; images/outcomes cannot
be read through LibraryView. `bpool/fits/size_N/{fit.npz,READY.json}` is sealed
by save_fitted_recipe, containing P1's PCA/Mahalanobis/early metric,
LOEO-corrector/five-fold distance calibration/continuity scale and shared caps.
Guard n3/epsilon0.5 and the corrector motion-only cap are unchanged shared
parameters, not outcome-trained thresholds. Fitted recipes contain their
selected action/state inputs; the library views themselves copy no arrays.

At500, as R15, the fits include all B episodes but a B serving query must
exclude its complete own episode `suite_task_id*50+B-init`. A queries have no
own B episode even with equal integer indices. Numeric IDs are preserved in
LibraryData and LOEO training/schedule tests pass. Current P1
`query(exclude_episode=...)` returns raw retrieval before corrector/guard/knob;
full composed B500 serving and indexed-store loading remain P1 requests,
explicitly not a claimed live five-layer certificate.

`KNOB_PRODUCTS.json` defaults to pending_cost_qualification for IR25/32/40/45.
With a matching measured `--ir-costs` profile and optional sealed
`--miss-sampler`, P1 library_schedule/calibrate_periodic performs the exact R11
cap solve using only the selected size; infeasible targets are reported.
No π/GR coefficients or captured wall latency are reused. CPU synthetic-cost
calibration passes, but **new-model measured GPU IR profiles remain pending**.

**Disk estimates**, uint8 paired256 RGB, float32 keys/normalized chunks/state,
uncompressed maximum native decision counts. These are ceilings, not measured
rollout sizes. [Pinned manifest authorities and byte totals](../ops/samples/libero_library_alignment/DISK_ESTIMATES.json).
Parent overhead for JSON/headers and fit arrays is additional; budget the shown
3× working space for capture, staged archives and parent on weilandserver.
The initial H100 whole-run estimate is superseded by the bounded staging
implementation documented below; H100 no longer accumulates the whole run.

| Bundle × suite | Max decisions | Keys bytes / decision | Images bytes / decision | Parent GiB | Capture/transfer/build working GiB |
|---|---:|---:|---:|---:|---:|
| DP × spatial | 22,000 | 544 | 393,216 | 8.078 | 24.233 |
| DP × L10 | 52,000 | 544 | 393,216 | 19.093 | 57.278 |
| Fast-WAM × spatial | 22,000 | 196,640 | 393,216 | 12.105 | 36.314 |
| Fast-WAM × L10 | 52,000 | 196,640 | 393,216 | 28.611 | 85.833 |
| Cosmos × spatial | 7,000 | 2,084 | 393,216 | 2.580 | 7.741 |
| Cosmos × L10 | 16,500 | 2,084 | 393,216 | 6.082 | 18.246 |

**CPU evidence.** [Tiny synthetic sample receipt](../ops/samples/libero_library_alignment/CPU_PIPELINE_SAMPLE.json):
500 episodes,1,000 decisions,4×4 paired images,250 unsuccessful episodes,
all six fits and nested views, exact image bytes, complete LOEO exclusions and
selected distance folds. A separate two-episode synthetic test runs closed
SHA snapshot/install → actual pipeline finish/build. No real checkpoint,
renderer, GPU, success-rate or pricing qualification follows from these tests.

| Final receipt | Actual result |
|---|---|
| [P6 release](../tests/ops/receipts/library_p6_release2.xml) | exit0: 166 pass /0 skip /0 fail,44.07s |
| [Legacy/default-off](../tests/ops/receipts/library_legacy.xml) | exit0:259 pass /0 skip /0 fail,20.76s |
| [Model interop](../tests/ops/receipts/library_model_interop.xml) | exit0:3 pass /0 skip /0 fail,0.73s |

Commands executed from this checkout (use a fresh basetemp when repeating):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/ops/receipts/library_p6_release2.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_library_release2

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py \
  tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/library_legacy.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_library_legacy

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/models/fastwam/test_fastwam_native.py::test_native_capture_consumed_by_p6_closed_npz_h5_bridge \
  exp/r17_integration/tests/models/groot/test_p6_bridge.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/library_model_interop.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_library_interop
```

Default-off witnesses preserve the original400 A-plan, exact worker argv,
action bytes and forward counts; disabled allowlist touches no arrays/files.
All17 previously pinned legacy/shared/R16 source files remain HEAD-identical;
hashes are in LIBERO_LIBRARY_VERIFICATION.json. The first development run had
one collection/import error; the next exposed two incomplete request fixtures
and repeated manifest reads and was interrupted through its own test session.
Those diagnostic receipts are preserved and not counted as passing. Subsequent
focused/full runs pass. The final L10 formal-server regression first failed
because the decision NPZ used library episode0 but its execution JSON retained
global episode1500. `LibraryDecisionSink` now maps both to the same suite-local
ID before writing. The [red receipt](../tests/ops/receipts/library_ack_id_red.xml)
records the old-code failure; the166-pass release2 receipt includes the corrected
decision/ack join and default-off case. Shell syntax and CLI help also exited0. Warnings are
existing pynvml/Qdrant/Swig deprecations, not GPU activity.

**Exact coordinator commands — NOT executed.** From
`/home/weiland/projects/openpi`, after the active formal GPU fleets close.
Run models and suites sequentially; never use ports23150–23179. Each smoke has
2 complete episodes in that suite and a780s GPU campaign deadline, leaving
cleanup time within a15-minute GPU slot; CPU seal/deploy happens before GPU
startup. Smoke builds only its own tiny parent, so a passing closed capture
receipt is required before admitting the corresponding full collection.
Replace --run with --prepare-only for CPU sealing/deployment.

DP, local4090 +timan107:

```bash
(
set -e
for suite in libero_spatial libero_10; do
  flock -w 1 /tmp/r17_gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/dp/libero_policy_100000 --server-node local --worker-node timan107 \
    --port 23320 --driver-port 23339 --replicas 2 --workers 2 --memory-budget-gib 14 \
    --smoke --seconds 780 --episode-seconds 600 --run
done
for suite in libero_spatial libero_10; do
  flock -w 1 /tmp/r17_gpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/dp/libero_policy_100000 --server-node local --worker-node timan107 \
    --port 23320 --driver-port 23339 --replicas 12 --workers 24 --memory-budget-gib 14 \
    --seconds 172800 --episode-seconds 3600 --run
done
)
```

Fast-WAM commands below were superseded by the owner gate correction.
Use the exact G2 A/B, two-replica commands in **Fast-WAM gate = G2 A/B** at
the end of this handback. P3 released loader parity is optional; the earlier
mandatory-P3/one-replica commands are withdrawn.

Cosmos, H100 +timan108: sol D's HANDBACK_COSMOS.md landed during this task,
but its published bundle still declares debug_capture unsupported and rejects
stage2/sample/trace capture. **These commands remain blocked until D publishes
the minimal capture hook/capability; no guard bypass is installed.** Native K5,
16-control commit, seed195/environment0 remain unchanged.

```bash
(
set -e
for suite in libero_spatial libero_10; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/models/cosmos_libero --server-node h100 --worker-node timan108 \
    --port 23360 --driver-port 23379 --replicas 1 --workers 2 --memory-budget-gib 10 \
    --smoke --seconds 780 --episode-seconds 600 --run
done
for suite in libero_spatial libero_10; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/models/cosmos_libero --server-node h100 --worker-node timan108 \
    --port 23360 --driver-port 23379 --replicas 1 --workers 24 --memory-budget-gib 10 \
    --seconds 172800 --episode-seconds 3600 --run
done
)
```

Use a new run name for each launch (default supplies one). A full parent is
never overwritten. Campaign receipts/attempts/backend capture live under
`/data/r17_library/<bundle>_<suite>/campaigns/<run>/`; real parent/subsets/fits
will be under that bundle/suite's `bpool/`, never /home. A CPU finish can resume
from a passing closed campaign; see the operator document. Optional measured
IR profiles can be supplied then without repeating collection or refitting.

Pending owners: coordinator runs the two-suite smokes/full collections and
qualifies actual captures/remote B reset checks; P3/coordinator provide Fast
GPU A/B; D provides minimal Cosmos capture (forwarded through P1); P1 adopts
row-index stores and composed B500 LOEO serving; model owners/coordinator supply
capture-off native cost profiles for IR caps. These are recorded in
REQUESTS/P6_to_P1.md. No actual new-model library, GPU timing, H100 parity,
IR cost authority or live B500 five-layer serving is claimed by CPU tests.

## H100 bounded staging

2026-10-09, P6 storage review correction. H100's approximately90GB free space
was supplied by the owner; **P6 did not connect to H100, inspect its live disk,
deploy, start a GPU/process fleet, or touch a running campaign**. The same
coordinator smoke/full commands above now select bounded H100 staging
automatically. DP's local capture, final sync, parent and fitting behavior is
unchanged. Fast GPU A/B, Cosmos capture and measured IR gates remain unchanged.

Owned implementation: new `ops/library_staging.py`,
`debug/library_shards.py`, `tests/ops/test_library_staging.py`; additive H100
branches in `ops/formal_{eval,server}.py` and `ops/library_campaign.py`;
expanded actual formal-server CPU checks in
`tests/debug/test_library_capture.py`; an optional synthetic fixture episode
offset in `tests/search/library_fixture.py`; updated
[operator instructions](../ops/LIBERO_LIBRARY.md). No shared/model/src,
frozen R16, harness or training edits. Contracts stay v1; new receipt schemas
are `r17.teacher_shard.v1`, `r17.h100_shard_budget.v1`,
`r17.h100_shard_receipt.v1`, `r17.h100_shard_transfers.v1`.

**Bound and lifecycle.** A terminal execution acknowledgment closes/drains
that episode's existing allowlist writer. Only closed shards are eligible.
A coordinator background thread pulls them while the resident/proxy/conductor
campaign continues, using snapshot_run → existing SHA-named /tmp stage →
install_snapshot, with archive AND every member SHA verified. It fsyncs a
weilandserver receipt before acknowledging remote cleanup. Cleanup rechecks
the exact shard's immutable file hashes and owned archive paths, then removes
only that shard and its source/staged tar copies. It enumerates owned files;
there is no blanket recursive deletion. Existing local arrays and receipts
remain durable. The final server snapshot contains only remaining metadata;
the original whole-capture snapshot is no longer needed for H100 library runs.

| Control | Default / behavior |
|---|---|
| Total H100 capture/staging bound | **20,000,000,000 bytes** (decimal20GB,18.63GiB); shared across all replicas/connections by flock. |
| Episode reservation |256MiB, held while writing AND closed awaiting verified receipt; conservative array/JSON/header charges checked before enqueue. |
| Transfer headroom |1GiB, one transfer at a time; source tar plus allowed /tmp-stage tar. Each tar≤256MiB. |
| Control headroom |1GiB; metadata/receipts remain small for the fixed500-episode campaign. |
| Maximum concurrent reservations |66; a67th without verified receipts fails closed. Test high-water19,864,223,744 bytes≤20GB. |
| Before deployment/start |/data free≥20,000,000,000+10GiB (28.63GiB); /tmp free≥3GiB. Before model load as well as before coordinator spawn. |
| During run |/data free≥10GiB+256MiB; /tmp free≥3GiB. Checked at capture/ack writes and by the two-second background monitor. |
| Failure |Latched disk/quota/writer/transfer errors prevent qualification. Failed/partial campaigns never publish a parent; unverified remote data is retained within reservations. |
| Parent/fits host |Only weilandserver, under `/data/r17_library`; H100 launcher/finish is rejected by hostname. |

The previous Fast-WAM×L10 **85.833GiB working estimate applies to
weilandserver**, including all captures, parent and transfer/build allowance;
fit overhead remains additional. H100 needs the bounded20GB allowance, with
the separate10GiB free-space reserve. Existing checkpoint/runtime assets are
not duplicated into the parent on H100. Each new campaign uses a unique owned
directory. Existing running campaigns are not changed or migrated.

**Executed CPU evidence:**183 P6 +259 legacy/default-off +3 model bridge
cases = **445 distinct passes,0 failures/errors/skips**. Full P6 XML:
[h100_staging_p6.xml](../tests/ops/receipts/h100_staging_p6.xml),44.32s;
legacy [h100_staging_legacy.xml](../tests/ops/receipts/h100_staging_legacy.xml),18.16s;
interop [h100_staging_interop.xml](../tests/ops/receipts/h100_staging_interop.xml),0.72s.
The183 include15 new staging cases and two additional actual formal-server
compositions, all CPU fixtures. The latter compare action bytes and forward
counts with the unchanged local path. The first focused20 cases passed;
a later focused run exposed one incomplete launcher fixture (missing the
normal deployed server_campaign),22passed/1failed. The fixture was corrected;
the full183-case receipt passes. Diagnostic XML is preserved and not counted
as a passing release.

Behavioral regressions cover pre-deployment/pre-model/pre-spawn disk rejection;
low /data and /tmp during a run; quota across concurrent replicas/in-flight
buffers; refusal to transfer an open capture; oversize pre-write rejection;
corrupt archives/receipts and changed remote files retaining all remote copies;
durable receipt ordering; transfer-error propagation; incomplete terminal
acknowledgments; and H100 parent-host rejection. A two-episode CPU campaign
streams/drains captures while its simulated residency remains live, then
performs the original metadata SHA sync and builds the tiny parent locally,
retaining an unsuccessful episode. A separate native-sized numeric fixture
captures **104 decisions,49,152 fused vision keys,state8,actions32×7 and both
256×256 uint8 cameras**, proving Fast L10's maximum episode fits one256MiB
reservation. This is sizing/transport evidence, not a real Fast sampler or
GPU qualification.

[Current file hashes, byte proof and CPU sample receipts](../tests/ops/receipts/H100_BOUNDED_STAGING_VERIFICATION.json)
record actual closed-shard bytes, charges and quota high water. The17 pinned
legacy/shared/R16 sources remain byte-identical to HEAD. Reproduction, from
this checkout (use fresh basetemps):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/ops/receipts/h100_staging_p6.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_h100_staging_p6

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py \
  tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/h100_staging_legacy.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_h100_staging_legacy

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/models/fastwam/test_fastwam_native.py::test_native_capture_consumed_by_p6_closed_npz_h5_bridge \
  exp/r17_integration/tests/models/groot/test_p6_bridge.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/h100_staging_interop.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_h100_staging_interop
```

Remaining qualification: coordinator runs the already-listed two-suite smokes
after GPU availability and records actual H100 admission/throughput receipts.
H100 `SHARD_BUDGET.json`, local `H100_DISK_{ADMISSION,START}.json`, per-shard
`shard_transfers/*.receipt.json`, `H100_SHARD_TRANSFERS.json` and final
`RECEIPT.json.transfers.h100_shards` are the inspection points. A transfer
stall at shutdown has a60s thread-close deadline; incomplete drain returns
partial and blocks publication. Partial stages are retained, not automatically
retried into extra copies. Live bandwidth, disk pressure and crash recovery
remain unqualified on H100; no real staging/GPU success is claimed.

## Fast-WAM gate = G2 A/B

P6 gate correction, 2026-10-09 owner instruction. Admission now uses Q's exact
`slot_G2_fastwam_libero_A` and `_B` PASS receipts, supplied with
`--g2-a-receipt` and `--g2-b-receipt`. P3 raw-Wan/released-Diffusers loader
parity is an **optional extra**, via the existing paired `--ab-qualification`
and `--ab-descriptor` flags. It cannot substitute for G2 and is never required.
No GPU work, campaign start, model load, port operation or process inspection
was executed. H100 access was limited to reading/hashing the named historical
small JSON sidecars; running campaigns and their assets were not changed.

Owned changes:

| File | Change |
|---|---|
| `ops/library_campaign.py` | G2 CLI and fail-closed receipt audit before directories/tunnels/deployment; optional P3 flag pair validation. |
| `ops/formal_eval.py` | Compare both qualified manifests against the newly SHA-pulled CPU teacher seal before worker deployment/resident startup; write `GPU_AB_GATE.json`. H100 library guard acquired remotely before residents, monitored with existing owned leases, stopped after owned replicas. |
| `ops/formal_gpu_lock.py` | Stdlib guard using Q's existing execution-host `/tmp/r17_gpu.lock`; conflict rejects startup. No local coordinator lock. |
| `tests/ops/test_fastwam_g2_gate.py` |39 new CPU cases: exact slots/status/GPU requirement, nested wrappers, missing/mismatched/tampered pins, teacher geometry, CLI without P3, preserved staging, two replicas/ports, remote lock rejection/release/ordering. |
| `ops/LIBERO_LIBRARY.md` | Current gate, required binding, two-replica budget, H100-only exclusion and pending evidence. |
| `REQUESTS/P6_to_P1.md` | Request for Q/coordinator to publish historically bound receipts, preserving originals. |
| `tests/ops/receipts/FASTWAM_G2_{BINDING_INSPECTION,VERIFICATION}.json` | Actual input receipt/sidecar hashes, missing bindings, CPU and byte-identity evidence. |

Interface: existing ModelManifest v1 (`schema=r17.model_manifest.v1`, `spec`,
`content_digests`) must be attached as `model_manifest` to each native Q PASS
record, or `remote_receipt.model_manifest` for the coordinator wrapper. The
new gate receipt is `r17.fastwam.g2_library_gate.v1`. Both wrapper and nested
record must have exact slot, `status=PASS`, `gpu_executed=true`; checkpoint path
must match the teacher. The full weight/base/Wan/tokenizer/norm/code pin map
must equal new `load.json.expected_digests`. Recompute checkpoint identity
from the complete map and code digest from its code submap; compare checkpoint,
code and norm identities, layout, features and native sampler to sealed
`model_spec.json`. A/B must agree on the complete asset map and these identities.
There is no mutable-assets, directory-path, CPU-parity or status-only fallback.

**Actual supplied receipts remain blocked by missing bindings.** Both local
coordinator wrappers and nested Q records PASS with the requested slots and
checkpoint path. Neither contains the required recorded digests. For each
of A and B the exact missing fields are:

- `model_manifest.spec.checkpoint_identity`;
- `model_manifest.spec.code_digest`;
- `model_manifest.spec.norm_identity`;
- `model_manifest.content_digests`, including weight/code/norm asset SHA256s.

A's H100 `native.json` has its historical full model manifest; B's
`SERVER_READY.json` has the actual server spec/pid80420. Their observed
checkpoint/code/norm identities agree, but neither sidecar's SHA/identity
is bound in its PASS receipt. P6 did not retrospectively certify those files
by substituting today's assets. Q can publish bound receipts from recorded
historical artifacts, with original receipt hashes, if that association is
certifiable; otherwise the missing evidence requires Q's qualification.
[Read-only hashes and identity inspection](../tests/ops/receipts/FASTWAM_G2_BINDING_INSPECTION.json)
and [request to Q via P1](../REQUESTS/P6_to_P1.md) give exact paths and identities.
Matching a *fresh* teacher seal is still required after that repair; no passing
comparison with current native teacher weights is claimed by P6.

**H100 bounded staging is active for this bundle**, independent of G2/P3 flags.
All four suite/smoke/full CLI composition cases assert the unchanged default
20,000,000,000-byte campaign policy, automatic streaming, H100 server/timan108
workers and both replica ports. The full existing staging tests pass: closed
shards move through SHA-checked /tmp stages; exact remote copies are removed
only after durable verified receipt; free-space checks fail closed before and
during collection; parent/subsets/fits stay on weilandserver `/data/r17_library`.
Local DP's writer/build path and local lock usage remain unchanged.

Execution-host exclusion now uses an owned **H100** guard, sharing Q's existing
`/tmp/r17_gpu.lock` convention on that host. Existing formal process supervision
and coordinator renewal hold its expiring lease. It is registered before
residents, monitored for exit, and stopped after owned H100 replicas; readiness
must match owned pid/campaign. `H100_GPU_LOCK.json` and remote owned-process
receipt/cleanup entries attest it. CPU tests prove a held coordinator lock does
not conflict with the separate execution-host lock, and a busy H100 lock stops
before replicas. This guards cooperating launchers, not arbitrary training
processes that ignore the convention. Actual H100 locking/VRAM/throughput and
crash recovery remain coordinator qualifications, not CPU claims.

**Executed CPU release:**222 P6 +259 legacy/default-off +3 model bridge =
**484 distinct passes,0 failures/errors/skips**. The39 focused new cases pass
and are included in222, not counted twice. Initial focused run had12 failures
and53 passes from comparing serialized JSON lists with dataclass tuples;
canonical normalization corrected that comparison. Historical diagnostic XML
is retained; the complete final release passes. All17 pinned legacy/shared/R16
sources remain byte-identical to HEAD. Receipts:
[full P6](../tests/ops/receipts/fastwam_g2_p6.xml),
[legacy/default-off](../tests/ops/receipts/fastwam_g2_legacy.xml),
[bridge](../tests/ops/receipts/fastwam_g2_interop.xml),
[verification](../tests/ops/receipts/FASTWAM_G2_VERIFICATION.json).
Reproduction (CPU only; use fresh basetemps):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/ops/receipts/fastwam_g2_p6.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_fastwam_g2_p6

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py \
  tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/fastwam_g2_legacy.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_fastwam_g2_legacy

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/models/fastwam/test_fastwam_native.py::test_native_capture_consumed_by_p6_closed_npz_h5_bridge \
  exp/r17_integration/tests/models/groot/test_p6_bridge.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/fastwam_g2_interop.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_fastwam_g2_interop
```

Exact coordinator commands, from **weilandserver**, sequential suites, H100
inference +timan108 rendering. **Not executed.** With the receipts as supplied
these commands reject before deployment on the missing fields above. Once Q
publishes the bound receipts, use their concrete paths in the two flags (the
literal paths below are the owner's originals, not fabricated passing pins).
No local flock wrapper: the 4090/DP lock is never opened by these commands.
Proxy23340, backend23341/23342, driver23359; no23150–23179 ports. Two resident
replicas use a total64GiB cap (32GiB each), above the owner's approximately27GB
per-replica observation and below H10080GB; actual two-resident VRAM remains to
be measured in the smoke. Each suite smoke has two episodes and ≤900s deadline.
Both smokes must pass closed capture/build/backend coverage before either500
collection. Failed ordinary episodes remain in the500 parent.

```bash
(
set -e
cd /home/weiland/projects/openpi
for suite in libero_spatial libero_10; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model fastwam_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/models/fastwam_libero --server-node h100 --worker-node timan108 \
    --g2-a-receipt /data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_A_20261009_234134_2639853/receipt.json \
    --g2-b-receipt /data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_B_20261009_234224_2645346/receipt.json \
    --replicas 2 --workers 2 --memory-budget-gib 64 --port 23340 --driver-port 23359 \
    --smoke --seconds 900 --episode-seconds 600 --run
done
for suite in libero_spatial libero_10; do
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash exp/r17_integration/ops/library_campaign.sh --model fastwam_libero --suite "$suite" \
    --checkpoint /data/mw_finetune/models/fastwam_libero --server-node h100 --worker-node timan108 \
    --g2-a-receipt /data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_A_20261009_234134_2639853/receipt.json \
    --g2-b-receipt /data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_B_20261009_234224_2645346/receipt.json \
    --replicas 2 --workers 24 --memory-budget-gib 64 --port 23340 --driver-port 23359 \
    --seconds 172800 --episode-seconds 3600 --run
done
)
```

Pending owners: **Q/coordinator** historical receipt digest binding and actual
fresh teacher-seal comparison; coordinator two-replica GPU smoke/full/VRAM and
H100 real bounded-transfer qualification; **P3** optional released loader
parity remains separate and unexecuted. P1's prior indexed-store/composed B500
LOEO serving request remains pending; these collection commands do not claim
library policy quality or five-layer serving qualification.

## Shard transfer failure fix

2026-10-10, investigation of failed Cosmos spatial B500 collection
`teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be`.
**Root cause: closed-shard over-reservation exhausted the logical quota.**
The original code counted every active *and* closed-unreceived episode as
256MiB, then permanently latched any capacity rejection. H100's preserved
`servers/SHARD_BUDGET.json` contains exactly:

- 66 reservations:19 writing and47 closed;47 closed files total **50,765,494 bytes**;
- high water **19,864,223,744 bytes** (66×256MiB +two1GiB reserves);
- the67th full reservation would require **20,132,659,200 bytes**, above20e9;
- failure=`RuntimeError('H100 capture bound reached; verified shard receipts required')`;
- free /data **94,592,126,976 bytes**, free /tmp **28,466,307,072 bytes**;
- staged closed shard `63eed99ea1e836b2d4622571a2ff7d39981aa2b0ec4b3f7cb49b41ca91a664c3`.

The coordinator's17,716,740,096-byte high water was its last successful poll,
not the final H100 quota state.90 completed transfer acknowledgments were
summarized. The91st shard already had a durable local verified receipt and
SHA-checked installed files, with `partials=[]`; its two962,560-byte H100
archives matched that SHA. The latched quota failure blocks ShardBudget
construction in `acknowledge_shard`, so verification cannot release the queue.
The old coordinator log dropped the inner exception; the precise failing
pump call was not recorded there. Receipt ordering points to acknowledgment,
while the quota file directly proves the initiating failure. This was a
logical capacity failure, with ample physical free space; the current archive
passed pull/install checks and was a closed shard. No partial/open-shard or
failed pull explains this evidence.

[Preserved pre-cleanup evidence](../tests/ops/receipts/COSMOS_SHARD_FAILURE_EVIDENCE.json)
contains the original quota JSON/hash, staged files/hashes and original final
campaign receipt. It was captured before any cleanup. The failed campaign was
not reset, promoted or reused; its remote quota file remains byte-identical.

Owned implementation changes:

| File | Fix |
|---|---|
| `ops/library_staging.py` | Active episodes keep256MiB so they can finish; after drained closure, reserve actual immutable bytes +64KiB rounded to1MiB, capped at256MiB. New episodes wait outside flock for up to120s at capacity. Persistent stalls/disk/identity violations fail closed. No20GB increase. |
| `ops/library_staging.py` | Every pump operation records shard/step/archive paths, original cause/exception chain, stdout/stderr/returncode; prints and atomically writes `H100_SHARD_FAILURE.json`, marks summary FAIL. Background, direct drain and finish errors retain context; stderr still reports logging failures. |
| `ops/formal_eval.py` | Preserve complete command stdout/stderr on RuntimeError; final receipt includes structured `shard_transfer_failure`, and shutdown errors print their concrete cause. Success behavior is unchanged. |
| `ops/library_cleanup.py` | Explicit-file, SHA-verified cleanup for closed H100 library campaigns. Validate all paths/proofs and owned process closure before unlink; refuse live processes, symlinks, adjacent campaigns or changed files. No signals or capture removal. |
| `tests/ops/test_shard_failure_fix.py`, `test_library_cleanup.py` |22 new cases, including the production19-active/47-closed backlog crossing67, transfer-driven admission under quota pressure, bounded stalls, all pump failure steps, stderr, safe cleanup and adjacent DP preservation. |
| `tests/ops/test_library_staging.py`, `test_fastwam_g2_gate.py` | Existing capacity test requests immediate rejection explicitly; remote pump mock declares its failure field. |
| `ops/LIBERO_LIBRARY.md` | Current reservations/backpressure and diagnostic receipts. |

Closed unreceived *used bytes* remain accounted until a durable verified
receipt; only unused reservation is reclaimed at closure. Transfer reserve
remains1GiB and metadata reserve1GiB. Independent replicas share the same
flock. A capacity wait holds no quota lock, so the pump can snapshot/verify/
acknowledge and admit the waiting episode. Ordinary unsuccessful episodes are
still retained. No official A input or library identity/alignment changed.
The library hooks remain opt-in; the running local DP collection does not use
this H100 quota/pump and was not operated on.

**Actual cleanup:** four archives plus91 Q stage marker files, **95 files and
9,170,973 bytes removed**. All91 local installed shard manifests/files were
SHA-checked against their durable receipt, including the unacknowledged91st
shard. Local runtime/assets archives also matched the H100 copies. Exact four
archive paths, under this campaign only:

```text
/tmp/r17_formal/teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be/server/runtime.tar
/tmp/r17_formal/teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be/server/assets.tar
/tmp/r17_formal/teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be/server/shards/63eed99ea1e836b2d4622571a2ff7d39981aa2b0ec4b3f7cb49b41ca91a664c3.tar
/tmp/r17_Q_qualification/teacher_cosmos_libero_libero_spatial_500_20261009_235143_c24171be_shards/workers_e606aabae2331b8f06311c6b00e104073271cc1b87088c77ec13cecee399e578_a1a6d964b3484f6780004702b1c0c22e.tar
```

The91 removed `workers_<sha>_<nonce>.stage.json` files were in the same exact
Q `<campaign>_shards` directory. Both replicas and proxy had closed owned
process receipts; the cleanup additionally refused any matching live owned
PID. No process was signaled. The reviewed helper ran in a temporary Python
namespace against the existing private runtime's stdlib helpers; no runtime
snapshot, training asset or global environment was edited/deployed.
[Explicit plan](../tests/ops/receipts/COSMOS_SHARD_CLEANUP_PLAN.json),
[dry run](../tests/ops/receipts/COSMOS_SHARD_CLEANUP_DRY_RUN.json),
[exact applied file list/hash/bytes](../tests/ops/receipts/COSMOS_SHARD_CLEANUP_APPLIED.json),
[post-cleanup inventory](../tests/ops/receipts/COSMOS_SHARD_CLEANUP_AFTER.json).
Post-check finds **zero .tar/.part/.stage.json transfer files** in the two
campaign /tmp roots.13 control/log/receipt/seal files and empty directories
remain for diagnosis. All66 remote capture shards (19 incomplete,47 closed),
local91 verified captures and original failed quota/receipts remain. They are
not staging archives and were not blanket-deleted. DP files/processes/ports
were not accessed or changed by cleanup.

**Executed CPU release:**244 P6 +259 legacy/default-off = **503 distinct
passes,0 failures/errors/skips**.22 targeted new cases passed and are included
in244, not counted twice. The first focused run had66 passes/1 failure because
logging a summary-write failure added receipt_write_error only in memory;
the implementation now persists that field when the failure file remains
writable. Final targeted and full suites pass; diagnostic XML is retained.
17 legacy/shared/R16 sources remain byte-identical to HEAD. Evidence:
[targeted22](../tests/ops/receipts/shard_failure_fix_targeted.xml),
[full P6](../tests/ops/receipts/shard_failure_fix_p6.xml),
[legacy](../tests/ops/receipts/shard_failure_fix_legacy.xml),
[verification](../tests/ops/receipts/SHARD_FAILURE_FIX_VERIFICATION.json).
CPU commands (fresh basetemps on rerun):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/ops/receipts/shard_failure_fix_p6.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_shard_failure_fix_p6

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py \
  tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/shard_failure_fix_legacy.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_shard_failure_fix_legacy
```

**Exact relaunch command, coordinator only, not executed.** Run from
weilandserver after H100 availability. The launcher creates a new unique
campaign and SHA-deploys the corrected runtime; it does not resume the failed
quota state. Same native Cosmos5/replan16, B500, two resident replicas,24
workers and18GiB total budget as the failed run. Its actual closed replicas
recorded7,852MiB peak reserved each; the corrected collection/throughput
still needs this relaunch. No local4090 flock: execution-host exclusion is
acquired by the owned H100 guard. Ports remain23360/23361/23362/23379.

```bash
cd /home/weiland/projects/openpi
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_spatial \
  --checkpoint /data/mw_finetune/models/cosmos_libero --server-node h100 --worker-node timan108 \
  --replicas 2 --workers 24 --memory-budget-gib 18 --port 23360 --driver-port 23379 \
  --seconds 172800 --episode-seconds 3600 --run
```

No GPU work was started by P6. No commit/push, broad deletion or DP campaign
operation was performed. Coordinator owns the real relaunch/500-library/fit
receipt; CPU behavior and the authorized exact-stage cleanup are complete.

## Pre-decision retry + repair mode

**Change and ownership.** Added `debug/attempts.py`, `ops/library_retry.py`,
`ops/library_repair.py`, `tests/ops/test_library_retry_repair.py` and
`tests/search/test_teacher_retries.py`. Updated the owned library sinks,
`search/teacher_library.py`, `ops/{formal_driver,formal_eval,library_campaign}.py`,
the synthetic library fixture, `ops/LIBERO_LIBRARY.md` and this handback.
`REQUESTS/P6_to_P2.md` requests an optional shared retry predicate and accurate
worker exception progress receipts; the working implementation is an owned
launcher shim over P2's existing queue, fences, transport and journal.
Interface remains `r17.p6.v1`; new provenance schemas are
`r17.library_repair.v1` and `r17.library_input_intent.v1`. No conductor, model,
src, frozen R16, harness or mw_finetune source was edited.

**Retry rule.** At most **two retries / three total attempts per original UID**,
including attempts before repair. A worker error must report exactly zero
policy controls, no decision/ack rows, and verified absence of server capture
and synchronous model-input intents. Library sinks fsync an identity-only
intent before consuming model input, including a first forward that fails.
H100's persistent root intent survives verified shard removal, avoiding an
absence check that races a shard transfer. A normal unsuccessful outcome is
accepted once and kept. Any step, decision, input intent, unknown evidence or
unreported timeout makes the attempt terminal; it is never rerun. Parent
publication checks every prior contiguous attempt against all closed captures,
including unselected attempt rows, and seals the complete prior error records
under `episodes[*].prior_attempts` and
`metadata.pre_decision_infrastructure_attempts`.

**Real read-only preflight.** Original campaign:
`/data/r17_library/dp_libero_libero_spatial/campaigns/teacher_dp_libero_libero_spatial_500_20261010_000311_bdd4f22f`.
It contains **496 accepted non-error outcomes + four zero-step EGL errors**.
The repair selects exactly task6, B inits41/42/43/44, library IDs341/342/343/344,
at attempt2. All12 original local capture bindings are closed and drained;
their manifest/payload SHA checks pass. The selected original decision counts
equal their capture counts; the four errors have no per-step/ack/capture rows.
Evidence: [repair preflight](../tests/ops/receipts/DP_SPATIAL_REPAIR_PREFLIGHT.json).
A fresh native DP **CPU-only** seal matched all six original asset files
byte for byte, including weights, checkpoint/code/norm identities, sampler,
benchmark and full YAML: [reseal](../tests/ops/receipts/DP_SPATIAL_REPAIR_RESEAL.json).
This is input/repair admission evidence, not a completed real repair receipt.

**Repair mode.** `--repair FAILED_CAMPAIGN` explicitly launches only missing
non-error outcomes. It preserves the original semantic campaign, UID, YAML,
paired RNG seed, B archive bytes, B alignment, trial0 and library episode IDs.
Transport routes and fleet capacity can change. A fresh CPU seal is compared
before worker deployment or GPU inference; all six teacher files must match.
The original campaign remains read-only. New closed captures and original
verified captures form a new `merged/` collection with pinned provenance;
normal parent, nested row views and selected-size fits run on that collection.
The original FAIL receipt remains intact. Seal/checkpoint/alignment/B changes,
post-decision/step errors, missing dispatched worker results, duplicate outcomes,
capture corruption and exhausted retry budgets are rejected. Historic captures
without an input-intent fence allow only the known zero-step EGL framebuffer
initialization error, with no cost/ack/capture rows. Fast-WAM repair reuses
the original G2 A/B receipt paths and still runs the strict digest admission
gate. H100 repair keeps the20GB shard pump and H100-side lock; parent/fits
remain on weilandserver.

**CPU evidence.** Targeted repair/parent checks passed37/37 with no skips;
that count includes one existing finish test. A synthetic complete500 pipeline
keeps496 outcomes, recollects four episodes at attempt2, builds1000 rows with
both RGB cameras, cuts all six nested subsets, fits/seals each size, preserves
all250 unsuccessful outcomes and proves the original files unchanged. The
actual CampaignDriver/P2/stock queue tests cover retry, three-attempt exhaustion,
step/capture/input-intent/report blockers, absent evidence, ordinary unsuccessful
outcomes, timeout refusal and generation floors. Additional CLI tests verify
that a changed fresh weight seal blocks deployment, only the missing UIDs are
launched, and Fast-WAM repair still invokes G2 admission. Final release:
**281 P6 +259 legacy/default-off =540 distinct passes, zero failures/errors/skips**.
The76 gate/repair-focused passes are included in281, not counted twice.
All17 checked legacy/shared/R16 sources remain byte-identical to HEAD; new
retry/input-intent hooks are library-only, and the non-library timeout method
remains unchanged. Full receipts:
[P6](../tests/ops/receipts/predecision_retry_p6_release.xml),
[legacy](../tests/ops/receipts/predecision_retry_legacy.xml),
[focused76](../tests/ops/receipts/predecision_retry_gate_release.xml).
Source checks and exact repair admission evidence are recorded in
[verification](../tests/ops/receipts/PREDECISION_RETRY_VERIFICATION.json).
The initial targeted fixture failures were missing stock stage-completion
drives and optional journal error keys; the fixtures were corrected without
weakening production guards. Diagnostic XML remains available.

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --junitxml=exp/r17_integration/tests/ops/receipts/predecision_retry_p6_release.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_predecision_retry_p6_release

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py \
  tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py \
  tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py \
  exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py \
  exp/mw_finetune/tests/test_workers.py \
  --junitxml=exp/r17_integration/tests/ops/receipts/predecision_retry_legacy.xml \
  --basetemp=exp/r17_integration/tests/ops/scratch_predecision_retry_legacy
```

**Exact DP spatial repair command — coordinator only, after DP libero_10
finishes and the4090/library ports are free; not executed.** This local DP
command uses the local GPU lock, two replicas, four timan107 workers and a
900-second GPU campaign limit. It does not collect another500 trials.

```bash
cd /home/weiland/projects/openpi
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  flock -n /tmp/r17_gpu.lock \
  bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite libero_spatial \
  --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --server-node local --worker-node timan107 --replicas 2 --workers 4 \
  --memory-budget-gib 14 --port 23320 --driver-port 23339 \
  --seconds 900 --episode-seconds 600 \
  --repair /data/r17_library/dp_libero_libero_spatial/campaigns/teacher_dp_libero_libero_spatial_500_20261010_000311_bdd4f22f
```

If collection/merge passes but a later CPU finish is interrupted, use the
existing `--finish NEW_REPAIR_CAMPAIGN/merged` path. GPU/EGL repair and the real
500 parent/fit receipts remain coordinator-owned and unexecuted by P6. The P2
shared API request remains pending; the local shim is functional and tested.
No GPU/process/port operations, deletions, commit/push or writes to the running
Cosmos finish or DP libero_10 campaign were performed.

## Node-failure salvage

**Scope and interfaces.** Added CPU-only `library_campaign.sh --salvage
FAILED_CAMPAIGN` in `ops/library_campaign.py`, implemented by
`ops/library_salvage.py`. No inference, simulator, GPU lock, port, tunnel or
fleet deployment starts in this mode. `ops/library_repair.py` accepts its
separate closed derived collection. `debug/attempts.py` validates sealed
`discarded_node_failure` histories separately from zero-progress retry
receipts; `search/teacher_library.py` records these under
`metadata.quarantined_node_failure_attempts`. Existing pre-decision retry
rules and the default-off legacy path remain unchanged. New evidence schema:
`r17.node_failure_salvage.v1`; capture/model interfaces remain v1.

**Closure and transfer.** Salvage rechecks the original local driver and
receipt-owned H100/timan108 PID/start-tick identities, process groups and
campaign worker commands. A live or unreachable owner fails closed. It sends
only a stdlib helper to the existing sealed H100 runtime, without changing
that runtime. Each owned shard, including interrupted shards, uses the
existing SHA-checked `/tmp/r17_formal` → qualification `/tmp` stage. One archive
at a time is capped at256MiB, its two copies fit the existing1GiB transfer
reserve, and the campaign quota remains at or below20GB. Low `/tmp` free space,
changed payloads, foreign paths, missing receipts and checksum errors retain
the remote copy. A durable verified local receipt precedes remote cleanup;
partial bytes can be transferred as evidence but cannot qualify an episode.
Failures record and print shard, transfer step, cause and stderr in
`salvage/SALVAGE_FAILURE.json`. All operations are confined to this failed
Cosmos campaign. No Fast-WAM/DP process, port, lock or campaign file is changed.

**Episode admission.** Every retained episode requires its original unique
accepted non-error journal result and matching accepted attempt; a SHA-sealed,
closed/drained capture of the exact bundle and B input; contiguous decisions
matching the driver ledger; every exact-identity execution ack, including the
final terminal ack; and an executed-step sum equal to the worker outcome.
Ordinary unsuccessful teacher outcomes are kept. A checksum failure, torn
writer, missing decision/ack or inconsistent identity quarantines the entire
episode. An unknown torn driver tail conservatively quarantines the affected
collection instead of guessing its UID. Empty interrupted shards use their
original episode manifest to preserve attempt identity. Quarantine preserves
original attempt records and actual captured/reported decision counts; it
never converts a worker exception's reported `n_steps=0` into a claim of zero
actual progress. Repair reruns unattempted episodes at attempt1 and explicitly
discarded attempts at the next bounded generation, retaining semantic UIDs,
paired seed, B state/episode IDs and the original six-file teacher seal.
Quarantine SHA pins and the original receipt/journals/alignment/B archive are
rechecked before repair and merge. Original files remain read-only;
`salvage/` contains the documented derived outputs. H100 quota/receipt updates
and deletion of its verified copies are the authorized transfer cleanup.

**Actual Cosmos libero_10 recovery.** Original campaign:
`/data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3`.
P6 executed the CPU-only salvage and then reran it with the final receipt
audit. Both passed. **174 remaining shards (165 closed +9 interrupted)** were
SHA-verified locally and cleaned on H100 after durable receipt, removing
**445,020,802 bytes** of capture data. Original296 transferred shards plus
these174 were all reverified against their exact snapshot/install receipts
and every file SHA: **470 received shards** total. The closed derived
collection retains **461 complete accepted non-error episodes**, including
**19 ordinary unsuccessful teacher outcomes**. It discards **23 errored
attempts** and records **16 never-dispatched episodes**. Nine errored attempts
contain **66 captured decisions**, despite the original worker's reported
`n_steps=0`; all66 are quarantine evidence and excluded from library sources.
No teacher outcome or zero-progress attestation was fabricated.

The actual read-only repair admission **PASS** selects exactly suite-local
task9 (global39), B inits11–49, library IDs461–499: **23 at attempt2 +16 at
attempt1**. The original receipt remains FAIL with `all_owned_closed=false`;
its files, journals, alignment and B archive retain their pinned SHA values.
All16 current Cosmos model hook files match the original model seal. H100 now
has **zero capture bindings, zero source/relay shard archives and zero quota
shards** for this campaign. The remaining3,791,440 bytes under its `/data`
campaign and8,362,201 bytes under its formal `/tmp` stage are retained
deployment/log/receipt evidence; other campaigns are untouched.
Actual evidence: [receipt](../tests/ops/receipts/NODE_FAILURE_SALVAGE_ACTUAL.json);
full proof/transfer/drop/closure records and `REPAIR_ADMISSION.json` are in
`FAILED_CAMPAIGN/salvage/`. The latter is a CPU admission receipt, not a GPU
repair or500-library completion receipt.

**CPU regression evidence.** Final owned suite: **302 passed,0 failed,
0 errors,0 skipped**; [XML](../tests/ops/receipts/node_salvage_p6_sealed.xml).
Legacy/default-off: **259 passed,0 failed,0 errors,0 skipped**;
[XML](../tests/ops/receipts/node_salvage_legacy.xml). **561 distinct passes**;
the21 salvage regressions are included in the302, not added again:
[targeted XML](../tests/ops/receipts/node_salvage_targeted_sealed.xml). They use real
synthetic writer/capture/ack data and the actual SHA stage/install/cleanup
functions. They cover local and H100 salvage→repair→complete-parent joins,
ordinary failed outcomes, missing final ack/cost rows, torn captures/journals,
checksums, duplicate acks, step/coordinate disagreement, empty interrupted
shards, changed original/seal/proof/quarantine, live owners, low space,
foreign campaign paths, receipt-before-delete and idempotent acknowledgments.
Six changed Python files parse; all17 legacy source files remain byte-identical
to HEAD: [source proof](../tests/ops/receipts/NODE_SALVAGE_SOURCE_CHECKS.json).
CPU commands actually run:

```bash
cd /home/weiland/projects/openpi
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -p no:cacheprovider -q \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --basetemp=exp/r17_integration/tests/ops/scratch_node_salvage_sealed \
  --junitxml=exp/r17_integration/tests/ops/receipts/node_salvage_p6_sealed.xml
.venv/bin/python -m pytest -p no:cacheprovider -q \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_node_salvage_legacy \
  --junitxml=exp/r17_integration/tests/ops/receipts/node_salvage_legacy.xml
```

**Exact salvage command — CPU + transfer only, executed by P6.** Rerunning
is safe after completion: verified local captures/receipts are reused and the
original receipt stays FAIL/open. Use the derived directory for repair.

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_10 \
  --salvage /data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3
```

**Exact repair command — coordinator only, NOT executed.** Run after the
Fast-WAM H100 chain releases its execution-host lock. This uses the stock
H100 exclusivity, no local4090 `flock`, and Cosmos ports23360/23361/23362/23379.
It retains the original collection/episode deadlines; the planned repair
cohort, rather than another500 trials, bounds collection work. Fresh teacher
sealing must match all six original asset files before workers/GPU inference.

```bash
cd /home/weiland/projects/openpi
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_10 \
  --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
  --memory-budget-gib 18 --port 23360 --driver-port 23379 \
  --seconds 172800 --episode-seconds 3600 \
  --repair /data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3/salvage
```

The coordinator owns GPU repair and the final500 parent/subsets/fits receipt;
P6 has not started GPU work, committed or pushed.

## Transient node-offline tolerance

**Implemented and CPU-tested, 2026-10-10.** Read the original Fast-WAM
spatial campaign's receipt: its primary error was tether exit64 during a10s
timan108 control probe, explicitly `node_offline: status=STALE`, with
`all_owned_closed=false`. This was a control-plane rejection, not evidence of
a failed teacher decision. The old supervisor also expired unrenewed leases
after120s, so simply retrying a probe for five minutes would still stop workers.

Owned changes:

| File | Change |
|---|---|
| `ops/tether_retry.py` | New stdlib transport classifier, bounded retry, thread-safe append/fsync journal, campaign scope and shard context. |
| `ops/formal_eval.py` | Retry remote execution, lease/liveness controls, worker snapshot reads, transfers and owned tunnel cleanup; record events in final receipts; extend new remote leases and worker-close grace. Closure requires explicit `alive=false`, retaining historical supervisor errors separately. |
| `ops/formal_process.py` | Explicit bounded `lease_seconds`; old requests retain120s. New launcher requests use600s. Campaign hard deadlines still apply. |
| `ops/deploy.py` | Opt-in transport retry for immutable tether transfers; existing default API remains unchanged. |
| `ops/library_staging.py` | Propagate journal/context into pump thread; retry stage/pull/ack; persist the verified ack before deletion and recognize an exact replay after a lost reply; allow bounded capacity/shutdown waits to cover retries. |
| `ops/library_campaign.py` | Scope CPU salvage retry receipts to the derived `salvage/` directory; use retrying tunnel creation/cleanup for new campaigns. |
| `tests/ops/test_tether_retry.py` |34 new deterministic CPU regressions, including STALE recovery/expiry, cleanup closure, real failures, timeout controls, lost ack, bounded disk backpressure and worker payload isolation. |

Shared contracts remain v1. New local receipt schema is
`r17.tether_retry.v1`; this does not change model, conductor, capture or fit
interfaces. The worker payload explicitly includes the new stdlib retry module.

**Retry and receipt rules.** Each recognized offline/STALE transport rejection
gets a300s window after the first failed call, with2/4/8/16/30s backoff capped
at30s. Saturation is also retryable. Every blip prints the concrete stderr and
is durably appended to `TETHER_BLIPS.jsonl`, with node, operation, attempt,
elapsed time and shard/step context when applicable. Recovery and expiry are
recorded too. `TETHER_RETRY_RECEIPT.json` and the final collection
`RECEIPT.json.tether_retry` retain all events; H100 shard summaries also retain
the journal. Preparation and tunnel cleanup events join the same campaign log.

Explicit remote command failures, exit124 execution deadlines, checksum errors,
quota/free-space errors and teacher outcomes are never retried by this layer.
Lease `renew/status/stop`, immutable transfers, cached shard stage/ack and the
closed worker snapshot read can additionally retry recognized lost replies or
an outer tether-client timeout. An ambiguous `spawn` response is not replayed,
because starting a second process would be unsafe. Nested retry wrappers share
an exhaustion marker, so a300s expiry cannot restart another300s window.
An in-flight CLI call keeps its own execution timeout;300s bounds retry waits.

Remote failure cleanup uses these same retries. An explicit closed-process
reply makes `all_owned_closed=true` even when the process receipt retains an
earlier supervisor error. An unreachable node after the cleanup window remains
unconfirmed and fails closed. This behavior was tested with controlled CPU
responses, not exercised against live GPU campaigns.

The H10020,000,000,000-byte cap, free-space admission/continuous guards, closed
shard requirement and SHA-checked `/tmp` transfer remain enforced. Capacity
backpressure can wait390s without holding the quota flock, then fails closed.
Pump shutdown allows at most1080s for its three bounded stage/pull/ack operations
and their command timeouts. No unverified copy is deleted. A replay must exactly
match the durable received-shard receipt and committed quota state; tampering
latches capture closed. Parent/subsets/fits remain on weilandserver. DP retains
its local capture/build path.

**Final CPU evidence.**336 P6 +259 legacy/default-off = **595 distinct passed**,
zero failed/errors/skipped. The86 targeted checks are included in the336,
not additional passes. Earlier intermediate receipts are retained as history;
one intermediate test had an incorrect expectation after intentionally
tampering with a receipt: that integrity error correctly latched the quota,
so the subsequent access raises `capture stopped`. The final regression checks
that fail-closed behavior.

- [Final P6 XML](../tests/ops/receipts/tether_retry_p6_final.xml):336 passed,168.92s.
- [Final legacy XML](../tests/ops/receipts/tether_retry_legacy_final.xml):259 passed,17.76s.
- [Final targeted XML](../tests/ops/receipts/tether_retry_target_final_closed.xml):86 passed,15.56s.
- [Source/admission checks](../tests/ops/receipts/TETHER_RETRY_SOURCE_CHECKS.json):17 legacy files byte-identical to HEAD, seven Python files parse, all three coordinator commands parse, existing bound G2 A/B CPU admission passes. New/changed source SHA pins are recorded.

Executed CPU commands (the environment below applied to each pytest command):

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --basetemp=exp/r17_integration/tests/ops/scratch_tether_retry_p6_final \
  --junitxml=exp/r17_integration/tests/ops/receipts/tether_retry_p6_final.xml
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_tether_retry_legacy_final \
  --junitxml=exp/r17_integration/tests/ops/receipts/tether_retry_legacy_final.xml
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/ops/test_tether_retry.py \
  exp/r17_integration/tests/ops/test_shard_failure_fix.py exp/r17_integration/tests/ops/test_fastwam_g2_gate.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_tether_retry_target_final_closed \
  --junitxml=exp/r17_integration/tests/ops/receipts/tether_retry_target_final_closed.xml
```

**Exact Fast-WAM spatial salvage command — NOT executed.** CPU and H100
transfer only; process closure must be proved before draining the original
campaign. Original campaign evidence remains read-only; documented outputs go
under its `salvage/` directory. The cohort for repair comes from verified complete
episodes, not the approximate200 attempts reported in the coordinator log.

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model fastwam_libero --suite libero_spatial \
  --salvage /data/r17_library/fastwam_libero_libero_spatial/campaigns/teacher_fastwam_libero_libero_spatial_500_20261010_020305_a90d5a54
```

**Exact Fast-WAM spatial repair command — coordinator GPU launch, NOT executed.**
Use the derived closed receipt. The bound G2 paths below are the ones recorded
in the failed campaign and pass the CPU gate; these contain the required digest
bindings, unlike the earlier unbound qualification wrappers. Repair still
requires a fresh teacher seal identical to the original model/weight/code/norm,
alignment and B inputs before deployment. It re-runs only episodes lacking a
complete accepted result, then performs the normal parent/subsets/fits finish.

```bash
cd /home/weiland/projects/openpi
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model fastwam_libero --suite libero_spatial \
  --checkpoint /data/mw_finetune/models/fastwam_libero \
  --g2-a-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_A.json \
  --g2-b-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_B.json \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
  --memory-budget-gib 64 --port 23340 --driver-port 23359 \
  --seconds 172800 --episode-seconds 3600 \
  --repair /data/r17_library/fastwam_libero_libero_spatial/campaigns/teacher_fastwam_libero_libero_spatial_500_20261010_020305_a90d5a54/salvage
```

**Exact Fast-WAM libero_10 full command — coordinator GPU launch, NOT executed.**
Run after the spatial repair releases its H100 lock and ports.

```bash
cd /home/weiland/projects/openpi
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model fastwam_libero --suite libero_10 \
  --checkpoint /data/mw_finetune/models/fastwam_libero \
  --g2-a-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_A.json \
  --g2-b-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_B.json \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
  --memory-budget-gib 64 --port 23340 --driver-port 23359 \
  --seconds 172800 --episode-seconds 3600 --run
```

**Worker placement:** `--worker-node timan107` is valid with
`--server-node h100`; the server host and simulator worker host are independent.
Both combinations pass the CLI CPU regression. After DP libero_10's timan107
fleet finishes, replace only `--worker-node timan108` with
`--worker-node timan107` in either GPU command. Salvage still checks the original
timan108 process identities. Each launch runs the existing per-node simulator,
init-pool and endpoint preflight. Two replicas use23341/23342, stock proxy23340,
driver23359; both worker nodes reach the exposed policy/driver routes. H100
exclusivity remains the execution-host GPU guard; there is no local4090 `flock`.

Pending coordinator work: execute salvage, verify its selected/dropped cohort,
then run spatial repair and libero_10 collection/fit. The same tolerance will
be deployed for the pending Cosmos libero_10 repair. This change does not retrofit
sealed runtime code into running campaigns. No GPU work, live remote cleanup,
campaign transfer, process/port intervention, commit or push was performed here.

## H100 repair argv fix

Implemented and CPU-qualified. This section also covers local missing-result
salvage and reduced tether traffic. Contracts remain v1; teacher sampler,
checkpoint/norm identity, B coordinates, nested subsets and per-size fits retain
their existing rules. Coordinator commands below were **not executed**.

**Cause / evidence.** The failed Cosmos repair
`teacher_cosmos_libero_libero_10_repair_20261010_023755_54360429`
has a229,644-byte campaign JSON, including182KB repair provenance. Preflight
and asset transfer succeeded; `H100_DISK_START.json` was never produced. The
first H100 admission command put the entire campaign in one Python `-c`
argument, exceeding Linux's131,072-byte single-argument limit. Local DP skips
that branch. The500-coordinate regression reproduces actual `errno.E2BIG`
with the old argument, then reaches admission through a SHA-verified file
reference with short argv. No GPU process starts in that test.

**Owned files / hooks / regressions.** Paths below are relative to R17:

| Owned files | Change and passing CPU regression |
|---|---|
| `ops/remote_json.py`, `ops/formal_eval.py` | Large campaign/control cohorts/argument lists use content-addressed JSON through the campaign's existing /tmp ingress, with SHA verification after transfer and again at read time. References are cached.500-coordinate E2BIG, returned-SHA mismatch, tamper and symlink tests pass. |
| `ops/formal_process.py`, `ops/formal_eval.py` | One exec session renews/checks every owned service on a node, at least30s apart. The2s driver loop uses cached status. Missing labels fail closed. A120s CPU test makes8 sessions for2 nodes/5 services and renews all5 leases. |
| `ops/library_batch.py`, `ops/library_staging.py`, `ops/library_campaign.py` | New H100 library campaigns enable `h100_batch`: up to8 closed shards per SHA-checked /tmp archive/pull, with individual receipts. One exec acknowledges the last batch and prepares the next. Minimum4 shards, flush after120s or at finish. Two tar copies total at most1022MiB within the1GiB transfer reserve; /data staging uses hardlinks. Shared20GB quota and continuous free-space admission remain active. Corrupt archives/receipts/remote bytes, duplicate receipts, lost ack replay and tail flush tests pass. |
| `ops/sync_back.py` | Both snapshot scans preserve nested per-shard `TRANSFER_SHA256.json` files; only the root transfer manifest is excluded. Batch receipt verification passes. |
| `ops/library_capture_views.py`, `ops/library_salvage.py` | Local salvage proves closure, splits sealed shared bindings into per-episode metadata views over hardlinked blocks, keeps complete accepted outcomes including ordinary failures, and seals quarantine evidence. Missing results are discarded even if their captures/acks look complete. Intent-only progress remains evidence. H100 salvage recovers an interrupted closed batch before remaining shards. Original evidence remains read-only. |
| `debug/attempts.py`, `ops/library_repair.py`, `search/teacher_library.py` | `discarded_missing_worker_result` requires sealed quarantine/closure proof and no selected rows from that old attempt. Fresh generation and parent provenance distinguish quarantine from pre-decision infrastructure retries; no zero-progress claim is fabricated. Shared-binding salvage→repair→parent regression passes. |
| `ops/tether_retry.py`, `ops/deploy.py`, `ops/formal_eval.py` | `TETHER_CALLS.jsonl` fsyncs actual CLI calls including retries. Final receipts include per-node totals, observation time, average/minute and last-minute counts.300s transient retry and identity-bound cleanup remain; real command/SHA failures remain terminal. |
| `tests/ops/test_h100_argv_batch.py`, `test_local_library_salvage.py`, `test_library_salvage.py`, `test_tether_retry.py`, `test_fastwam_g2_gate.py`, `ops/RUNBOOK.md` | Behavioral regressions, control mocks and runbook updated.25 new regressions are included in the full suite. |

**Real DP evidence / quarantine.** Read-only inspection of
`teacher_dp_libero_libero_10_500_20261010_005446_9bd136a9`
confirms497 accepted non-error results, zero error attempts, and exactly three
dispatched UIDs lacking results: suite task30/B inits15,18,20. Local salvage
writes only documented `salvage/` outputs: closure proof, `capture_views/`,
`QUARANTINE.json`, sealed provenance, filtered driver ledgers and derived
collection/result receipts. Quarantine records entire excluded sources/UIDs,
observed capture/intent and driver counts, reasons, original attempts and SHA
pins. Shared array blocks remain evidence; filtered metadata indices exclude
every quarantined row from the parent. No missing result becomes an outcome.
Local errors after any step/decision and incomplete **returned** teacher
outcomes are refused; ordinary returned failures remain fixed. Repair retains
strict checkpoint/seal/B alignment checks. A source-only read verified all29
locally readable DP teacher Python/RECIPE digests against the original seal;
full weight/norm/B payload checks remain coordinator repair admission work.
P6 has not run salvage or repair on this real campaign.

**Tether rate / limits of measurement.** The closed failed Fast-WAM10 campaign
`teacher_fastwam_libero_libero_10_500_20261010_023853_9d2cd40c`
has54 verified shard receipts spanning925.519s:3.436 shard arrivals/minute
over53 intervals. Original code:9 status calls per2s loop,5 lease renewals
about every26s, one pending query per2s, and3 transfer calls per shard.
That gives **321.85 calls/minute**, a code/closed-receipt estimate excluding
RPC duration/startup calls. The original has no call telemetry or new retry
receipt; it does not demonstrate failure of the new300s tolerance.

A CPU fake-node replay of those54 arrival times through production functions
made122 calls over930s:64 node controls,46 batch-next execs,12 pulls with
4–5 shards each: **7.87 calls/minute**. Original transfer used54 pulls.
Real file-size metadata confirms even8 of these shards fit the bounded batch.
This is a CPU replay, not a measured live H100 after-run rate or proof of broker
heartbeat starvation. Future `TETHER_CALLS.jsonl`/receipt metrics enable that
comparison. Each control exec shares one session across the node's services;
batch-next combines ack and preparation. The installed noninteractive exec
interface is used; no persistent interactive agent or broker change is added.
Evidence: [closed trace](../tests/ops/receipts/TETHER_TRAFFIC_EVIDENCE.json),
[production CPU replay](../tests/ops/receipts/TETHER_TRAFFIC_CPU_REPLAY.json).

**CPU/default-off proof.** Final full P6: **361 passed**,193.742s; legacy:
**259 passed**,17.232s. Total **620 distinct passes**, zero failures/errors/skips.
The final rerun includes conservative tar/PAX/SHA-manifest sizing and cached
large argument references. All17 checked legacy/shared/R16 files remain byte
identical to HEAD. Batch staging is H100-library-only; DP local collection and
transfer retain their prior path. Receipts:
[P6 XML](../tests/ops/receipts/h100_argv_p6_final.xml),
[legacy XML](../tests/ops/receipts/h100_argv_legacy.xml),
[source checks](../tests/ops/receipts/H100_ARGV_SOURCE_CHECKS.json),
[commands parsed, not run](../tests/ops/receipts/H100_ARGV_COORDINATOR_COMMANDS.json).
Earlier diagnostic XMLs retain initial fixture failures; final XMLs pass.
Commands actually executed (CPU environment applied to both):

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops \
  --basetemp=exp/r17_integration/tests/ops/scratch_h100_argv_p6_final \
  --junitxml=exp/r17_integration/tests/ops/receipts/h100_argv_p6_final.xml
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_h100_argv_legacy \
  --junitxml=exp/r17_integration/tests/ops/receipts/h100_argv_legacy.xml
```

**Exact Cosmos repair relaunch — coordinator only, after H100 lock release;
not executed.** Use the original salvage, not the empty E2BIG repair campaign.
Fresh preparation deploys an immutable runtime containing these fixes. No
local4090 `flock`; H100 exclusivity remains its remote guard. Cosmos ports
23360–23379 remain separate from Fast-WAM23340–23359. `--worker-node timan107`
is valid with an H100 server too; below retains timan108 for Cosmos and reserves
timan107 for DP. GPU repair commands require the normal GPU-enabled coordinator
environment, not the CUDA-hidden CPU test export above.

```bash
cd /home/weiland/projects/openpi
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model cosmos_libero --suite libero_10 \
  --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --worker-node timan108 --replicas 2 --workers 24 \
  --memory-budget-gib 18 --port 23360 --driver-port 23379 \
  --seconds 172800 --episode-seconds 3600 \
  --repair /data/r17_library/cosmos_libero_libero_10/campaigns/teacher_cosmos_libero_libero_10_500_20261010_005553_3fa10df3/salvage
```

**Exact local DP10 salvage — CPU-only coordinator command, not executed.**

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite libero_10 \
  --salvage /data/r17_library/dp_libero_libero_10/campaigns/teacher_dp_libero_libero_10_500_20261010_005446_9bd136a9
```

**Exact local DP10 repair — coordinator only, after salvage PASS and4090
availability; not executed.** Selects only the3 missing UIDs;2 replicas/4
workers, local4090 lock, then normal parent/subsets/fits finish.

```bash
cd /home/weiland/projects/openpi
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  flock -n /tmp/r17_gpu.lock \
  bash exp/r17_integration/ops/library_campaign.sh --model dp_libero --suite libero_10 \
  --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --server-node local --worker-node timan107 --replicas 2 --workers 4 \
  --memory-budget-gib 14 --port 23320 --driver-port 23339 \
  --seconds 1800 --episode-seconds 900 \
  --repair /data/r17_library/dp_libero_libero_10/campaigns/teacher_dp_libero_libero_10_500_20261010_005446_9bd136a9/salvage
```

**Pending / limits.** Coordinator owns actual salvage/GPU repairs, real parent
and fit receipts and live call-rate measurement. Broker causality needs broker/
agent evidence. A node still offline after300s, real command/SHA failure,
corrupt/torn evidence or live owned process still fails closed. Quarantined
episodes remain excluded in full; every retained teacher outcome stays fixed.
No GPU work, remote command, real campaign transfer/cleanup/deletion, process/
port intervention, commit/push or original campaign edits were performed here.

## Cache study framework integration

**Scope / result (2026-10-10, CPU only).** Framework paths are implemented and
CPU checked for all eight groups × three model geometries × two suites. All48
YAML templates and their B500 fit/gate artifacts exist. Actual current-resident
admission remains blocked on the model hooks below; no GPU claim or campaign
launch is implied. The owner-authorized framework changes are confined to R17
common code, documented in `REQUESTS/P6_to_P1.md`; no `src/`, frozen R16/harness,
model-package or `exp/mw_finetune/` files were edited for this task.

**Rulings / freeze.** `OWNER_RULINGS.md` resolves evaluation to official **A500**
per suite, with B500 only for library/fitting. The R6/Q3 stall gate supersedes
legacy NoProgressGuard: confirmed slow calls the full native teacher, ambiguous
slow permits one extra LOOK, one fresh-anchor cooldown follows a stall-triggered
call. Ordinary lottery/MISS, periodic gap, target IR and knobs are off. Existing
batched controls/topologies are retained; no new controls. Complete40-task order
is preserved, including task30–39 for libero_10. DP/Fast short C5 has a distinct
C5 head; long is provisionally C10 pending registered NI freeze. Cosmos remains
C16, so ch→c, chr→cr, chg→cg, chrg→crg are logical aliases: smoke switches all8
labels, full study executes four physical groups. Full A plan: DP8,000 +
Fast8,000 + Cosmos4,000 =20,000 independent episodes.

**Files / concrete changes.**

- `common/config.py`, `common/provider.py`, `common/recipe.py`,
  `common/serving.py`, `common/layers/recipe.py`: default-off YAML miss modes,
  sealed fit/gate references, model-owned teacher compatibility callback,
  B→A protocol admission, frozen-fit reuse/corrector toggle, stall-call-only
  fallback, real executed-control notification and resident lifetime evidence.
  `make_bundle_validator(..., recipe_factory=None)` keeps the existing default;
  CPU admission can use functional model-owned hooks without mutating frozen
  providers or their registered identity.
- New `search/cache_recipe.py`: metadata/VIEW verification (complete episodes,
  selected view digest, B only), C-specific fitted recipe checks, SHA/stat
  immutability fences, original R6/Q3 controller at lottery probability0.
  Shared P1/R10 math remains the only retrieval/metric/corrector implementation.
- New `ops/cache_study.py`: generic generate/fit/audit/stage/seal/publish/plan/
  GroupCommit/aggregate paths; all study YAMLs acknowledged by stock proxy,
  conductor per-episode rebind and ordered groups on one resident fleet/model.
  Fits+metadata use the SHA-checked /tmp transport; teacher RGB is neither read
  nor shipped to serving. Native batched HIT runs stage1, tails run no stages,
  stall MISS runs all native stages. Actual final ACKs, activity and constant
  resident generations are audited, with unobserved activity explicitly marked.
- New `ops/cache_study_worker.py`: explicit process-local P2 cost/terminal-ACK
  evidence shim; normal worker/renderer loop and default cost rows stay intact.
  `ops/worker_lifetime.py`, `ops/formal_workers.py` apply it through recycling.
- `ops/formal_eval.py`, `ops/formal_driver.py`, `ops/formal_driver_fast.py`,
  `ops/formal_prestage.py`, `ops/driver_release.py`: opt-in study mode, CPU
  admission, proxy publication, group planning, durable driver and immutable
  runtime dependency pins. Study ports23400–23459 avoid existing campaign ports.
  Only study archives append the three original read-only R6 controller modules.
- New `ops/cache_study/AUDIT.md`, `ops/cache_study/smoke.sh`,
  `ops/cache_study/yaml/STUDY.json`,48 YAMLs, four sealed C5 fits and ten sealed
  Q3 artifacts (450MiB total new artifacts). All100 task/calibration combinations
  have at least four references; unavailable_tasks is empty in all ten READYs.
  Original B500 parents, subsets and size500 fits remain read-only.
- `tests/ops/test_cache_study.py`:33 CPU regressions. Existing
  `tests/debug/test_library_capture.py` graph mock now accepts the framework's
  sampler keyword; no production graph behavior change. Requests appended to
  P6_to_P1/P2/P3/P4/D. No commit/push or process/port intervention.

**Interface version / samples.** ModelAdapter/FeatureSpec/Layout/RequestContext
contracts remain v1; default RecipeConfig is `r17.recipe.v1` with legacy mode.
New artifacts: `r17.cache_study.v1`, `r17.cache_inputs.v1`,
`r17.cache_sealed.v1`, `r17.cache_stall.v1`, `r17.cache_summary.v1`,
`r17.teacher_compatibility.v1`. Model-owner receipt must be PASS with exact
teacher/resident retrieval identities and producer/weight/norm/layout evidence;
width equality alone never authorizes changed keys. `configure_library_provider`
returns a same-ID/same-ModelSpec provider, typically dataclasses.replace; normal
resident registration must install the identical hooks.

The48 short templates are under
`ops/cache_study/yaml/{dp_libero,fastwam_libero,cosmos_libero}/{sp,l10}/`:
`c.yaml`, `cr.yaml`, `ch.yaml`, `chr.yaml`, `cg.yaml`, `crg.yaml`, `chg.yaml`,
`chrg.yaml`. They use **never** (ungated) or **stall_only** (gated), checkpoints:{},
corrector as selected, legacy guard:false, periodic_gap/target_ir:null, native
sampler, sealed VIEW+fit(+stall) digests and explicit source/serving profiles.
Teacher-bound templates are reviewed here; CPU prepare produces actual
resident-bound `assets/cache_yamls/` before any dispatch and fails closed if a
model rejects the recipe or cannot attest teacher compatibility.

**Gap table / coordinator dispatch.** See `ops/cache_study/AUDIT.md` for every
source gap with file:line and `CACHE_STUDY_REAL_B500_AUDIT.json` for identities.

| Model | B500 load / C head | Corrector / Q3 gate | Per-YAML C | Hot-switch / batch / GroupCommit | Current resident admission |
|---|---|---|---|---|---|
| DP-T | fixed framework; **needs P4 VIEW reader** at admission.py:43,62 | fixed, actual owned B8/native scheduler CPU exercised | fixed C5/C10; NI freeze pending | fixed / CPU checked | **needs P4** teacher-v-control checkpoint/code/FeatureSpec compatibility |
| Fast-WAM | fixed | fixed, actual owned B16/native scheduler CPU exercised | fixed C5/C10; NI freeze pending | fixed / CPU checked | **needs P3** teacher-v-control checkpoint/code/FeatureSpec compatibility |
| Cosmos | fixed | fixed shared CPU path | ok native C16 aliases | fixed shared CPU path / B1 unchanged | **needs D** recipe admission at provider.py:6 and truthful capability evidence |

Thus **none of the three current releases is fully admitted for all eight
study YAMLs yet**. The shared framework no longer needs per-group Python edits
or replica restarts. Coordinator should dispatch P3, P4 and D against the
concrete requests, then freeze long C and schedule smoke. P2's permanent optional
cost/ACK hook is pending but does not block this explicit P6 evidence shim.

**CPU tests / receipts (actual executed).** Final broad P6 + common + durable
driver + Fast capacity run: **527 passed +3 subtests,0 failed,0 skipped**,285.42s,
`tests/ops/receipts/cache_study_verified.xml` and `.log`. All33 new regressions
are included. Separately legacy/default-off: **259 passed,0 failed,0 skipped**,
19.68s, `cache_study_legacy.xml`. Two default-off tests overlap these suites;
counts are not summed as distinct cases. Actual stock proxy localhost broadcast/
sticky/error tests: **6 passed,23 deselected,0 failed/skip**,0.13s,
`cache_stock_proxy.xml`. Real-library audit: **48 groups /192 real B500 fit
queries PASS**, all six parents, including confirmed-slow states and exact
C-specific fits: `CACHE_STUDY_REAL_B500_AUDIT.json`; it matches current study
manifest digest. This is teacher-bound CPU math, not current-resident/GPU
qualification. `CACHE_STUDY_SOURCE_CHECKS.json`:17 frozen/legacy files unchanged
from the previous HEAD-verified receipt. `CACHE_STUDY_COMMANDS_CPU.json`: all
three exact smoke command parsers/topologies and Fast native startup admission
PASS; script bash syntax PASS. Earlier failed test receipts remain preserved:
four stale graph mocks and two provider/error-contract issues were corrected;
final full rerun passes.

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/debug exp/r17_integration/tests/search \
  exp/r17_integration/tests/ops exp/r17_integration/tests/common \
  exp/r17_integration/ops/tests/test_driver_group_commit.py \
  exp/r17_integration/tests/models/fastwam/test_h100_capacity.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_cache_study_verified \
  --junitxml=exp/r17_integration/tests/ops/receipts/cache_study_verified.xml

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py \
  --basetemp=exp/r17_integration/tests/ops/scratch_cache_study_legacy \
  --junitxml=exp/r17_integration/tests/ops/receipts/cache_study_legacy.xml
```

Use new scratch/XML names for another run to retain these receipts. Shared
P1 layer tests match original AWM/Mahalanobis kernels, LOEO feature construction,
corrector math, gripper preservation and R16 serving conventions bitwise where
geometry agrees. C5 is explicitly a new C-specific port, not a claim of the old
R16 C10 action bytes. Frozen library identities and normalized teacher sampler
are independent of requested MISS grid; changed producer code still rejects.

**CPU fits/audit actually run, GPU-ready commands not executed.**

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m exp.r17_integration.ops.cache_study fit \
  --study exp/r17_integration/ops/cache_study/yaml/STUDY.json

env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m exp.r17_integration.ops.cache_study audit \
  --study exp/r17_integration/ops/cache_study/yaml/STUDY.json \
  --output exp/r17_integration/tests/ops/receipts/CACHE_STUDY_REAL_B500_AUDIT.json

# Coordinator only, AFTER model admission hooks; no commands below executed.
bash exp/r17_integration/ops/cache_study/smoke.sh dp --run
bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --run
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run
```

Smoke is32 episodes/model: two per group per suite, all8 logical groups on
one resident fleet, both suites. Same baseline replica/MPS/batch/worker settings:
DP local R2/B8/5ms/MPS,32 timan107 workers, budget40GiB, ports23400–23419;
Fast H100 R2/B16/5ms/MPS,48 timan107 workers, budget70GiB, ports23420–23439;
Cosmos H100 R2/B1,8 timan107 workers, budget20GiB, ports23440–23459. Fast/Cosmos
run sequentially. No local H100 flock: the formal launcher's H100 exclusivity
owns that GPU. `--print` is the script default and only prints the command;
`--prepare-only` requests CPU deployment/sealing. All three were parsed on CPU.
The smoke plan exceeds the old two-episode qualification size; cold load plus
32 episodes may exceed15min, so the bound is two hours including cleanup, not
a claimed15min measurement.

Require complete32 non-error accepted outcomes, proxy bundle ACKs, actual
per-decision source/NFE/stage accounting, corrector on/off and nonzero delta
activity, Q3 states/LOOK/calls and one-anchor cooldown, final server ACK joins,
and constant resident generation/PID (no restart) in SUMMARY/driver/server
receipts. Gated groups' only model calls are actual stall-triggered MISSes;
ungated groups must have zero action-head NFEs. If the two-episode cohort never
triggers a gate or has no nonzero corrector delta, record that GPU activity as
unexercised; do not fabricate it or add outcome-selected A episodes. CPU tests
separately force the trigger state to exercise the owned native queues/samplers.

**Known limits / pending owners.** P3/P4 must supply strict teacher compatibility
or request recollection; P4 must support actual size500 VIEW metadata; D must
admit Cosmos recipe and update evidence. Model registration/qualification files
were not edited here. Coordinator/Q owns final NI horizon selection, fresh
model seals, matched historical control identity merge, GPU smoke and full
launch. New C/gate artifacts use successful teacher B library LOEO empirical
calibration from original Q3; no independent new B validation rollout or A
calibration occurred. B-serving LOEO is not implemented by this study path;
A500 evaluation makes it unnecessary here. No performance/SR/GPU numerical
claim is made from CPU math or queue tests. Running campaigns, GPUs, original
libraries, forbidden nodes/paths, processes and existing ports were untouched.

## Cache-study matched smoke + resident full lanes (2026-10-10)

**Outcome / interface.** All three model admissions supplied by P3/P4/D are
now consumed by actual CPU preparations. P1 contracts remain v1; the optional
cache control/continuation CLI and receipts are `r17.cache_control.v1`,
`r17.cache_resume.v1`, `r17.cache_lane.v1`. No model package, src, original
teacher library, control campaign, preregistration, running process, GPU or
existing port was modified. No GPU work, commit/push or process signal occurred.
The old preceding section's P3/P4/D admission gaps are historical and resolved
by their later cache-study handbacks; GPU/cache quality remains unqualified.

### Files and concrete changes

| P6 owned file | Change |
|---|---|
| ops/cache_study/smoke.sh | Exact per-model worker RAM/VRAM, recycling/restart limits, driver/MPS topology, DP serialized startup, read-only control/runtime selection, short optional names; default prints only |
| ops/cache_control.py (new) | Complete matched-control proof, archive/native pins, explicit stale Cosmos outer-receipt recovery, immutable R17 cache-admission overlay over frozen control; simulator/client worker bytes retained |
| ops/cache_lane.py (new) | CPU prestage, one fleet/model across all groups/suites, H100 Fast→Cosmos order, scalar L selection and C-specific B-only artifacts, immutable lane plan, resume-safe cohort continuation |
| ops/cache_resume.py (new) | Accepted successful AND failed outcomes require complete decisions and terminal server ACK; torn/missing/error rows excluded with quarantine evidence; exact native/YAML/RNG/init UID pins; fresh closed-source segments |
| ops/cache_study.py | Stable original campaign namespace for resumed task UIDs, missing-cohort dispatch, source/current evidence join, resident generation checks per segment; regenerate fit/gate paths and identity when L changes |
| ops/formal_eval.py; formal_prestage.py | Default-off cache control/continuation paths, preparation seals and pin checks; full cache campaigns use Q GPU-first async evidence handoff |
| ops/queue_retry.py | Q's bounded startup retry also admits cache campaigns only after proved zero accepted work; reuse immutable native/YAML/worker inputs |
| ops/driver_release.py; formal_finalize.py | Cache-only CPU planner/resume refresh overlays; no native/model/worker change; final evidence uses current separately pinned cache aggregator |
| tests/ops/test_cache_lane.py (new) | Resource, controls, overlay/default-off, actual alternate-C fit, final ACK, torn/missing cohort, quarantine/readonly source, safe startup retry regressions |
| ops/tests/test_formal_finalize.py | Structural source audit parses one file snapshot; stale-import-line regression; all behavioral closure/transfer tests retained |
| ops/cache_study/FULL_RUN.md; AUDIT.md; ops/RUN_CONFIGS.md | Exact commands, resolved model-owner gaps, control/worker topology and planning ranges |
| ops/cache_study/CACHE_LANE_CPU_READY.json | Actual three-preparation seals, topology, all group500 counts, native/control/orchestration pins, GPU/dispatch false |

### CPU-prepared campaigns and baseline pins

| Model | Prepared campaign | Episodes / physical groups | Driver / fleet | timan107 workers RAM/VRAM MiB, recycle |
|---|---|---|---|---|
| DP | /data/mw_finetune/eval/csl10d | 8,000 /8 per suite | GroupCommit;4090R2/B8/5ms/MPS;startup1 |32×2048/576,N4 |
| Fast-WAM | /data/mw_finetune/eval/csl10f | 8,000 /8 per suite | GroupCommit;H100R2/B16/5ms/MPS;startup2 |48×2048/768,N4; explicit EGL0..7 |
| Cosmos | /data/mw_finetune/eval/csl10c | 4,000 /4 distinct per suite | legacy;H100R2/B1/MPSoff;startup1 |8×3072/768,N4; automatic EGL |

Each has16 sealed resident-bound YAMLs, `PREPARED.status=PASS`, CPU-only true,
GPU-bound/dispatch-started false, and no LAUNCH or execution receipt. The
independent readiness audit revalidated every preparation/orchestration/control
pin and all500/group counts. Both suites use official A; B never supplies
evaluation cells. DP32+Fast48 reserve54GiB renderer VRAM; after Fast closes,
DP32+Cosmos8 reserve24GiB. Runtime RAM/VRAM admission is rechecked at launch.
Q's MPS guard/control retry is automatic; no local4090 flock wraps H100.

DP prices against `hs5dp_ctl_142812`; Fast against complete merged
`hs6fw_res_152931`, native runtime `hs6fw_ctl_143332`. Cosmos uses merged K5
`formal_cosmos_libero_official2000_20261010_130639_a8ca2043`, specifically its
8-worker3072/768 treatment, rather than the fresh R1/12 template. Its original
outer receipt still says FAIL due stale rc2 handling. Current CPU
`driver_completion(control,0)` verifies2,000 outcomes; sealed RESUME_MERGE,
SUMMARY and CPU recovery attest its retained/continuation segments.
`CACHE_CONTROL.json` records the original error/status and merge SHA explicitly;
the original campaign remains read-only. No failure was silently relabeled.

The new framework cannot execute directly from a pre-cache control archive.
`CACHE_RUNTIME_OVERLAY.json` lists every reviewed R17 file overlaid on the
native control and pins both archives. Existing non-R17 control bytes stay
exact; only three original read-only Q3 dependencies are added. Model-owned
library validators still check native producer/weights/norm equivalence during
the actual CPU seal and live publication. `CACHE_WORKER_OVERLAY.json` proves
native simulator/client payload bytes exact, changing only P6 ops observers.
The DP planner was refreshed CPU-only after preparation; native assets,
runtime, workers and8,000 episode inputs stayed exact (`DRIVER_REFRESH.json`).

### Exact coordinator commands (GPU NOT executed)

```bash
bash exp/r17_integration/ops/cache_study/smoke.sh dp --run --run-name csmd1010
bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --run --run-name csmf1010
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run --run-name csmc1010

.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csl10dp --long-dp 10 --long-fast 10 --run
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --long-dp 10 --long-fast 10 --run
```

Smoke is32 episodes/model (2/group/suite), switches all8 logical labels on
one fleet, checks native gate-only MISS/NFE, corrector/gate/tail activity,
terminal ACK and resident lifetime. Unexercised activity stays unexercised.
Full H100 is one Fast fleet, then one Cosmos fleet; each remains resident
through both suites and every physical group. GPU-first handoff requires all
owned closure and GPU_RELEASED proof; CPU sealing overlaps the next model,
while the lane waits for all final evidence and fails on late sealing errors.
Q's rc2 missing-ACK recovery still requires complete accepted outcomes,
resolved retry errors, closed GroupCommit and confirmed worker fleet closure.

L stays **provisional10 for DP/Fast**, Cosmos16. The currently landed horizon
report still has pending arms/no certified long C; fallback10 is not NI proof.
`--long-dp N` / `--long-fast N` are the single model parameters. Set them in
CPU prestaging and repeat at dispatch. A new value needs a new lane directory:
heads/LOEO/caps and stall identity depend on C, so silently editing a prepared
YAML would violate the fit/data contract. Example (no GPU during preparation):

```bash
L_DP=10 L_FAST=10
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csnextdp --prefix csnext --long-dp "$L_DP" --long-fast "$L_FAST" --prepare-only
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csnexth --prefix csnext --long-dp "$L_DP" --long-fast "$L_FAST" --prepare-only
# Coordinator repeats with --run after reviewing the horizon freeze and smokes.
```

Repeat the same full lane command after a naturally closed campaign failure.
Complete successful AND failed outcomes are retained, never selected on SR.
Only missing/torn/error cells are dispatched in a fresh pinned segment; partial
rows remain excluded with source hashes/quarantine evidence. UID namespace,
native/YAML bytes, A init and seeds are inherited. Pending evidence/unclosed
fleets refuse resume. A new fleet on recovery is explicit; YAML hot-switches
within each segment retain residents. Startup faults with zero acceptance get
Q's at-most2 immutable retries. All-complete runs needing CPU evidence repair
are not silently rerun for different teacher outcomes.

Planning estimates (unmeasured cache GPU throughput): smokes DP10–20min,
Fast10–20min, Cosmos10–15min; full DP180–300min, Fast180–330min,
Cosmos120–260min, H100 lane300–590min plus cold load/evidence seal. Actual
native control driver times were55.66/64.40/123.22min per2,000 episodes;
4×/4×/2× reference scaling is223/258/246min. Key extraction, retrieval,
rendering, longer failed episodes and gate calls alter these projections;
update from smoke receipts. Full campaigns have a24h hard bound.

### CPU verification and limits

Broad P6/common/ops command (real receipt `cache_lane_full.xml`): **835 passed
+43 subtests,0 failed,0 skipped**,541.85s. Final changed-path +Q MPS/retry/
finalization suite (`cache_lane_release.xml`): **84 passed,0 failed,0 skipped**,
15.64s; includes28 cache-lane regressions. Legacy/default-off rerun
(`cache_lane_legacy.xml`): **259 passed,0 failed,0 skipped**,21.18s. Counts
overlap and are not summed as distinct coverage. The refreshed broad run
(`cache_lane_complete.xml`) closed with **845 passed +43 subtests,1 failed**,
500.73s: its structural source-inspection test used cached firstlineno after
shared source edits and read the wrong function. A fresh targeted run passed
1/1. The check now parses one source snapshot by function name, with a regression
reproducing stale import line numbers. Final entire ops suite plus all31 lane
regressions (`cache_lane_closed.xml`) passed **356 +43 subtests,0 failed,
0 skipped**,215.03s. The preceding updated changed-path run
(`cache_lane_ready.xml`) also passed100/100,18.53s. No production algorithm
change was needed for the source-inspection failure; intermediate receipts are
retained rather than relabeled as PASS. Final tests cover all changed runtime
paths and Q's current guard/retry/finalization behavior.

All commands used CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS=cpu,
PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,
MKL_NUM_THREADS=1,.venv/bin/python,-p no:cacheprovider. Commands:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops exp/r17_integration/tests/common exp/r17_integration/ops/tests --basetemp=exp/r17_integration/tests/ops/scratch_cache_lane_complete --junitxml=exp/r17_integration/tests/ops/receipts/cache_lane_complete.xml
.venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/tests/ops/test_cache_lane.py exp/r17_integration/ops/tests/test_formal_finalize.py exp/r17_integration/ops/tests/test_mps_queue_retry.py --basetemp=exp/r17_integration/tests/ops/scratch_cache_lane_release --junitxml=exp/r17_integration/tests/ops/receipts/cache_lane_release.xml
.venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/tests/ops/test_cache_lane.py exp/r17_integration/ops/tests --basetemp=exp/r17_integration/tests/ops/scratch_cache_lane_closed --junitxml=exp/r17_integration/tests/ops/receipts/cache_lane_closed.xml
.venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py --basetemp=exp/r17_integration/tests/ops/scratch_cache_lane_legacy --junitxml=exp/r17_integration/tests/ops/receipts/cache_lane_legacy.xml
```

Shell syntax passed; all three actual preparations and the20,000-cell
readiness audit passed on CPU. The intermediate alternate-C test receipt
(`cache_lane_horizon.xml`,22pass/1fail) exposed a test fixture lacking the
explicit partial-commit binding forDP C16; the corrected fixture now tests
real C15/C16/FastC20 fit seals and rejects a C10 substitution. It was not a
production partial-commit defect. Failed/intermediate receipts are preserved.
Known limits: no GPU model/cache smoke or performance/SR result here; horizon
NI freeze and launch remain coordinator/astra work. Original B fits/images
and calibration products stay untouched; new C heads use only B data. No
independent B validation rollout or A-based fitting was added.

Final smoke resource/timeout parser check: **4 passed,27 deselected,0 failed,
0 skipped**,0.15s (`cache_smoke_resources_final.xml`). Fast-WAM's episode
timeout now also matches the control3600s; DP/Cosmos retain1800s. The lane
CLI accepts registered L candidates DP10/15/16 and Fast10/16/20/32 only,
before any preparation. No further GPU/process action was taken.

## Urgent DP cache smoke startup routing + failure detection (2026-10-10)

**Closed result:** replacement `/data/mw_finetune/eval/csmdf1010` has sealed
RECEIPT + SUMMARY **PASS**, launcher rc0, 32/32 accepted episodes, 0 error
attempts,all_owned_closed=true. The full lane was not launched.

### Cause and ownership evidence

csmd1010 did **not** suffer an unexplained spontaneous worker-fleet death.
Its local replica0 published `servers/replica_0/ready.json` at18:11:29 CDT,
but the launcher selected `ziyanglin.com:23401` as its backend readiness
address. The fresh 234xx ports did not have the controls' working public
routing. It waited the 600s readiness bound before reporting
`TimeoutError('port ziyanglin.com:23401 did not become ready')`.
The driver never started. The timan107 private fleet receipt proves32
workers launched, 0 worker failures/recycles,and a 601.756s fleet lifetime;
individual `worker_*_generation_0.log` files show driver connection timeout
backoff. All32 workers were then SIGTERM-closed by the launcher's cleanup
(exit-15); the supervisor's child returned2 with fleet reason `interrupted`.
The reported empty main logs were not the per-worker log location.

Original receipt already had all_owned_closed=true when P6 inspected it.
P6 checked both local PID birth identities and all 34 remote identities
(32workers +fleet owner +supervisor): none remained alive. **No signals
were sent by P6 to csmd1010**, and no hs6fw_* or other campaign process was
touched. Evidence: `ops/cache_study/evidence/csmd1010_startup.json` binds the
original receipt and ready-file SHA256s,addresses,closure receipts and
remote private log diagnosis.

The observed workers stayed alive while awaiting the unavailable driver,
so a fleet-exit detector could not diagnose that particular routing wait.
A separate real gap in the detector was closed: it previously checked only
CLOSED+nonzero worker exit; FAIL,exit0-before-dispatch,and a live supervisor
with a dead fleet child could evade startup failure detection.

### Owned fixes and default-off behavior

- `ops/formal_eval.py`: opt-in cache studies allocate campaign-owned policy
  and driver tether routes for DP/Fast-WAM/Cosmos; local cache backend
  readiness/metadata checks use loopback. Proxy listener readiness precedes
  its public route. Failed/dead worker fleets and unexpected startup exit0
  fail on the next bounded 30s liveness probe,with private fleet receipt
  reason,exit code and worker failures in the launcher receipt/error.
- `ops/formal_workers.py`: cache-only startup diagnostic publishes fleet
  receipt,per-worker log glob,spawned count and awaiting-driver address.
- `ops/cache_study/smoke.sh`: DP `--run` and its printed command take
  `flock -n /tmp/r17_gpu.lock`; preparation/H100 paths do not take that
  local lock. All matched control resource configurations remain intact.
- `ops/cache_lane.py`: full 4090 launch takes the same local lock per
  campaign through owned GPU closure/handoff; H100/CPU preparation bypass
  it. CPU finalization waits happen after launch lock release.
- `ops/tests/test_cache_startup.py`: all three model routing regressions,
  byte-identical default-off route dictionaries/no tunnel allocation,
  supervisor/dead-child and startup exit0 cases,actual launcher failure
  propagation/owned cleanup with mocked remote I/O,and local-only lock
  exclusion/release. CPU fixture ports are ephemeral so they cannot collide
  with the authorized live smoke. No model/src/frozen/harness changes.

Existing undispatched csl10d/f/c preparations keep their native archives,
fits,worker payloads and control pins. The current prepared launcher binds
new endpoint/driver routes and rebuilds the episode endpoint plan at launch
(the existing prepared Cosmos path); no native re-preparation is needed.
Fresh H100 smoke helpers include the worker diagnostic as well. TUNNELS.json
records public transport provenance; controls remain read-only.

### CPU checks actually run

All CPU invocations used CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS=cpu,
PYTHONDONTWRITEBYTECODE=1 and OMP/OPENBLAS/MKL thread counts1,pytest
`-q -p no:cacheprovider`. Receipts in `tests/ops/receipts/`:

| Command targets | Receipt | Actual result |
|---|---|---|
| ops/tests + tests/ops + tests/debug + tests/search + tests/common | cache_startup_full.xml | **881 passed +43 subtests**,0 failed/skip,770.89s |
| ops/tests + tests/ops/test_cache_lane.py after final edits | cache_startup_ops_final.xml | **369 passed +43 subtests**,0 failed/skip,215.31s |
| ops/tests/test_cache_startup.py + ops/tests/test_formal_eval.py | cache_startup_cpu_final.xml | **27 passed +27 subtests**,0 failed/skip,7.84s |
| Previous handback's exact legacy/default-off target list | cache_startup_legacy.xml | **259 passed**,0 failed/skip,20.43s |

The intermediate `cache_startup_lock.xml` has 42 pass/2 fail: its two newly
written tests used the live smoke's234xx ports and hit their 30s admission
bound. The fixtures were corrected to ephemeral sockets. The final targeted,
operations,and full suites above pass; that intermediate receipt is retained.
`bash -n ops/cache_study/smoke.sh` passed. DP printed command starts with
flock; Fast-WAM printed command starts directly with Python.

### Authorized GPU smoke and accounting result

Executed only DP: `bash exp/r17_integration/ops/cache_study/smoke.sh dp
--run --run-name csmdf1010`. The run's separately recorded local lock
witness was acquired during CPU preparation,**before** the first resident
spawn,held through owned GPU cleanup,and is now RELEASED. Future helper
launches hold the lock directly through the new wrapper. Log:
`/home/weiland/projects/openpi_ext/scratch/r17_ops/cache_smoke_dp_fix.log`.

Exact proof: `ops/cache_study/evidence/csmdf1010_smoke.json` binds receipt,
summary,decision/journal,ACK,seals,tunnel and lock artifact SHA256s.
**16 suite/group cells × 2 episodes = 32**,all 2/2 accepted; driver 641.677s.
**1,253 decisions and1,253 execution ACKs,32 final terminal ACKs**,0 audit
violations. **1,227 FULL_HIT,26 MISS,2,600 NFE**. Every MISS was a
`stall_only` native DP100 call; all ungated groups have0 MISS/NFE. Source
counts:951library,275library_tail,26policy,1policy_tail. The audits include
tails and final ACKs. Corrector applied on 498 decisions; nonzero correction,
stall-call and extra-look activity all observed.

All16 YAMLs were acknowledged and dispatched across eight groups and both
suites with exactly two resident PIDs 3903175/3904487,one generation each;
`no_replica_restart_verified=true`. Worker recycling was planned N4 only,
not replica restart. Both replicas,proxy,driver,MPS guard and private daemon
are closed; remote owned_worker_survivors=[]; the 4090 lock is FREE. No full
lane dispatch or H100 GPU launch was performed.

**H100:** both Fast-WAM and Cosmos use the corrected generic cache route
allocation/liveness path. H100 smoke/full commands remain as in the matched
lane section; they never acquire the coordinator's4090 lock. The same
startup routing/dead-fleet defect is covered by CPU regressions for both.
Their GPU smoke/accounting qualification remains for the coordinator,not
claimed from the DP run. Coordinator may launch the full 4090 lane separately.


## Cache N4 slot reuse + accepted-cell resume

User-authorized live DP continuation on 2026-10-10; no H100 launch, stop,
signal, fleet change or active-campaign write. Interface contracts remain v1.

**Root cause and evidence.** Original `/data/mw_finetune/eval/csl10d` failed
at4001/8000 accepted. `RECEIPT.json` records RAM admission=true,
GPU replacement admission=false, zero worker_failures,992 planned N4 recycles
and1012 historical worker PIDs. The last planned recycle was slot27,
generation32,PID2399925,exit75. Original `Reservation.check()` performed a
new global NVIDIA/VRAM admission while the fleet still held that same slot,
and formal_workers abandoned the replacement after120 seconds. This added a
second headroom condition to an already admitted replacement. Old code
returned only a boolean and discarded the underlying NVIDIA/refusal details;
the retained receipt cannot distinguish delayed context release from a
CPU-starved NVIDIA query. RAM exhaustion and a worker crash are contradicted
by the receipt. The old worker log additionally raised
`ValueError('source archive must be a regular bounded file')` while staging
its uncompressed N4-generation archive; Q archive_stage's limit is512MiB.
That secondary exception changed the supervisor exit code to1.

The lane then tried a zero-acceptance startup retry of an accepted-results
campaign. Its rejection was correct; the missing behavior was explicit
continuation. CPU review also exposed a third defect: cache_resume required
`server_terminal_ack is True`, whereas production cache workers return a
structured server execution ACK. The old selector therefore retained0 of
4001 actual complete episodes.

**Owned fixes.**

| File | Result |
|---|---|
| ops/worker_gpu.py | Transfer the continuously held slot only after old PID closure/reap; verify slot index, physical device,UUID,VRAM budget and live reservation-owner birth identity; no fresh NVIDIA query or GPU admission |
| ops/formal_workers.py | Planned N4 reuses original RAM/GPU slot after child.wait; requests zero extra RAM/VRAM;600s bounded pending deadline; detailed last_replacement evidence; compress cache close archives with unchanged per-file SHA manifests |
| ops/cache_control.py | Rebuild a worker archive atomically from its sealed snapshot, changing exactly formal_workers.py and worker_gpu.py; preserve every other simulator/client/native byte; refresh only undispatched cache workers |
| ops/formal_prestage.py | Cold cache launch refreshes those two helpers under preparation lock before one-shot claim/startup snapshot; seal archive/refresh proof; running campaigns are refused |
| ops/cache_resume.py | Validate structured final ACK's decision/cursor/count/bundle/request identities, full contiguous decisions and matching execution counts/steps; retain legitimate failed outcomes; clone strict sealed missing-cell continuation |
| ops/queue_retry.py | Optional cache-only resume callback on accepted-cell failures; preserve native inputs, YAMLs and driver pins; default zero-acceptance path remains unchanged |
| ops/cache_lane.py | Resume failed accepted jobs at entry and mid-queue; unique continuation log names; advance through all remaining groups/suites/models; clear stale FAIL status on dispatch |
| ops/tests/test_recycle_resume.py; tests/ops/test_cache_lane.py | Real CPU exit75/reap, no fresh admission, live-PID/lost-lease/device rejection, actual structured ACK/tampering, two-file refresh/native parity, cold-only refresh, accepted-job continuation and FW→Cosmos advancement regressions |
| REQUESTS/P6_to_Q.md; ops/cache_study/evidence/csl10dp_resume_{launch,inputs,progress}.json | Coordinator caveat and SHA-bound original failure, source/setting/worker-overlay proof and live continuation evidence |

Generic N4 slot transfer applies to every model's formal worker fleet. The
cache payload refresh/accepted-cell resume/close compression are opt-in;
ordinary native/runtime-from snapshot inheritance stays byte-identical. No
src, model, conductor, Q qualification, frozen R16/harness or existing control
source was edited. The unrelated tracked logs/session_handoff.md diff belongs
to other work; P6 did not edit it.

**CPU validation actually run.** Commands use CUDA_VISIBLE_DEVICES='',
JAX_PLATFORMS=cpu,PYTHONDONTWRITEBYTECODE=1 and OMP/OPENBLAS/MKL threads1,
`.venv/bin/python -m pytest -q -p no:cacheprovider`. XMLs are in
`exp/r17_integration/tests/ops/receipts/`; scratch is isolated under /tmp.

| Targets | Receipt | Result |
|---|---|---|
| ops/tests + tests/ops full sweep | recycle_resume_fullops.xml |632 passed +43 subtests,0 failures/skips,701.66s |
| tests/debug + tests/search + tests/common | recycle_resume_debug_search_common.xml |255 passed,0 failures/skips,106.17s |
| Final test_recycle_resume + test_worker_gpu + test_mps_queue_retry + test_cache_lane | recycle_resume_final_gate.xml |85 passed,0 failures/skips,17.44s |
| Existing exact legacy/default-off target list from prior handback | recycle_resume_legacy.xml |259 passed,0 failures/skips,20.66s |

The full sweep began after the initial recycle/resume fix; structured ACK and
cold-refresh/queue-advancement regressions added during that sweep were then
covered by the final85-test gate. Counts overlap; do not sum them as unique.
The intermediate cold-refresh test failed once on attempting to overwrite an
immutable archive; the atomic archive rebuild fixed it and final gates passed.
Existing test_formal_finalize compressed-snapshot SHA roundtrip also passed
in the full sweep. No GPU test is implied by these CPU receipts.

Reproducible final regression and legacy commands:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/ops/tests/test_recycle_resume.py exp/r17_integration/ops/tests/test_worker_gpu.py exp/r17_integration/ops/tests/test_mps_queue_retry.py exp/r17_integration/tests/ops/test_cache_lane.py --basetemp=/tmp/p6_recycle_final_gate_1010 --junitxml=exp/r17_integration/tests/ops/receipts/recycle_resume_final_gate.xml
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py --basetemp=/tmp/p6_recycle_legacy_1010 --junitxml=exp/r17_integration/tests/ops/receipts/recycle_resume_legacy.xml
```

**DP live continuation.** Source audit now retains exactly4001 complete
accepted episodes, including ordinary unsuccessful outcomes; selects3999
missing cells; expected full plan8000. Four thousand spatial cells and one
libero_10 cell are retained. New segment contains499 c/libero_10 cells and500
each of cr/ch/chr/cg/crg/chg/chrg/libero_10. UID cohorts are disjoint and their
union equals the original8000 plan. Source partial rows are excluded by the
accepted UID whitelist and documented in CACHE_RESUME_QUARANTINE.json; no
partial outcome is accepted. Original campaign is read-only and SHA-pinned.

P6 launched the same prepared lane's explicit continuation:

```bash
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csl10dp --run
```

New run `/data/mw_finetune/eval/dpc8513584592`, lane parent PID262681,
start_ticks12201551. Detached owned launch receipt is in the evidence directory;
log `/home/weiland/projects/openpi_ext/scratch/r17_ops/cache_lane_4090_resume_1010.log`;
child log `/data/mw_finetune/eval/csl10dp/dpc8513584592_run.log`.
The CacheRunner holds /tmp/r17_gpu.lock through this run; observed HELD with
both residents alive(PIDs273674/275175). Original five owned local processes
and remote cleanup closure were verified dead/closed; no signals were needed
or sent. Runtime archive,checkpoint,controls,YAML selection,MPS,
R2/B8/5ms,startup width1,W32,2048/576MiB,N4 and restart budget2 remain identical
and are recorded in csl10dp_resume_inputs.json. Only the two owned recycle
helpers differ in worker payload. Actual progress/recycle/accounting snapshot
is in csl10dp_resume_progress.json. The lane remains running; it will execute
all remaining seven libero_10 groups after c and merge inherited plus new
ACK-complete outcomes into one8000-cell SUMMARY. This handback does not claim
full completion or future gate-group accounting before that summary seals.

**H100 lane safety / coordinator action.** No H100 or csl10f active files,
workers,signals or GPU commands were touched. Read-only receipt still shows
csl10f PARTIAL/running and Cosmos PREPARED. Its already-imported queue and
frozen worker fleet retain old code. If Fast-WAM completes normally, the
subsequent fresh formal_eval subprocess refreshes undispatched Cosmos helpers
before launch. If the current fleet naturally fails and all_owned_closed=true,
coordinator must restart the same lane to load the new automatic resume path:

```bash
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --run
```

Do not wrap H100 in the local4090 flock. The coordinator owns that restart;
P6 did not start an H100 watcher/continuation or interrupt the live lane.
Pending: complete live DP8000 result(P6/coordinator monitoring); reload old
H100 queue only after natural failure/coordinator scheduling. No additional
P3/P4/D model hook is required by these framework fixes.

### Final recycle file-descriptor correction and live evidence

The first32 actual N4 replacements in dpc8513584592 succeeded with
requested_extra_ram_bytes=0 and requested_extra_gpu_mib=0, same physical GPU
UUID/device/576MiB slot; worker_failures=[] and fleet status RUNNING.

A further concrete resource defect was found during live verification:
formal_workers retained every generation's log FD in a list. timan107's
supervisor soft RLIMIT_NOFILE was1024(hard1048576), and this source had1012
historical generations. Exhaustion is reproducible with genuine CPU worker
children; old code cannot complete80 recycles under a64-FD limit. New code
keeps at most one log handle per slot, waits for/reaps the old child and closes
that handle before spawning a replacement. All generation log files remain
on disk and in the SHA archive. This is another explanation consistent with
the historical discarded NVIDIA/refusal details; without those details we
do not claim to have recovered an original EMFILE stderr or exclude delayed
graphics cleanup/CPU-starved inventory. The held-slot re-admission and FD
growth defects are independently reproduced and fixed.

The already running DP payload was frozen before this last log-close change.
P6 therefore verified receipt-bound fleet PID2520110,birth339828082,current UID,
exact formal_workers command/launch_campaign.json and run_name/worker_run,
then used resource.prlimit to raise **only that owned supervisor's soft
NOFILE1024→8192**, leaving hard limit1048576. Observed FD count81. No signals,
RAM/VRAM changes, native model/YAML/episode changes or H100 fleet actions.
This bounded OS-resource workaround covers the remaining3999-cell frozen
segment; subsequent cold/resumed fleets receive the actual log-close fix.
Evidence: ops/cache_study/evidence/csl10dp_resume_nofile.json. The first
ownership check used the preparation campaign path rather than the launch
campaign path and refused before mutation; the successful operation checked
the actual receipt-bound launch_campaign.json. No unrelated PID was changed.

Final post-FD-fix CPU checks:

| Targets | Receipt | Actual result |
|---|---|---|
| test_recycle_resume,test_worker_gpu,test_mps_queue_retry,test_cache_lane,test_formal_finalize | recycle_resume_fd_release.xml |129 passed,0 failed/skipped,39.81s |
| Exact legacy/default-off targets above | recycle_resume_fd_legacy.xml |259 passed,0 failed/skipped,22.80s |

The new FD regression actually runs81 CPU children/generations under soft/hard
NOFILE64, verifies80 planned recycles/PASS and all81 preserved log files. An
intermediate lane fixture failed after concurrent lane changes required argv
seconds; it was updated to the real job contract and the final129 gate passed.
Concurrent shared lane additions (predecessor/size-count options) were preserved.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider exp/r17_integration/ops/tests/test_recycle_resume.py exp/r17_integration/ops/tests/test_worker_gpu.py exp/r17_integration/ops/tests/test_mps_queue_retry.py exp/r17_integration/tests/ops/test_cache_lane.py exp/r17_integration/ops/tests/test_formal_finalize.py --basetemp=/tmp/p6_recycle_fd_release_1010 --junitxml=exp/r17_integration/tests/ops/receipts/recycle_resume_fd_release.xml
```

Live DP count after that gate:298 new +4001 inherited =4299/8000 accepted,
zero errors. Earlier saved accounting sample:169 new episodes,
11355 matching decision/ACK pairs,169 final structured server ACKs,
library-only source/NFE0 in current ungated c group;32 successful N4 slot
transfers; lane alive and4090 lock HELD. These are live samples,not final
8000-cell qualification. The run remains active for the coordinator to monitor.
The H100 live fleet also retains its old log-handle behavior; P6 did not raise
its limits or modify it. If it closes with failure, use the H100 lane restart
command above to get bounded log handles and strict automatic continuation.

Latest exit-from-task observation (20:20CDT):347 new +4001 retained =
**4348/8000 accepted,0 new errors**, lane parent alive and intentionally left
running. Every selected task specification equals the original by UID.
Saved in ops/cache_study/evidence/csl10dp_resume_progress_latest.json.

## Worker evidence cleanup + disk guard

Owner's confirmed cause of the later20:2x–20:31 DP/Fast continuation failures
is timan107's shared root filesystem filling, including ENOSPC during stage
creation and an unavailable fleet receipt. The earlier RAM/EGL slot and log-FD
fixes remain separate changes. Read-only CPU measurements confirm that `/`,
`/tmp` and `/scratch/zixuans8/mw_r17/formal/runs` share device64769, total
227084713984 bytes (about212GiB). The coordinator's prior24 verified PASS-copy
removals and9GB provision staging are owner-provided evidence; P6 performed
no production deletion and did not change or signal either running lane.

Owned changes:

- New `ops/worker_disk.py`: stdlib-only `r17.worker_disk.v1` admission,
  unprivileged available blocks/inodes, nearest existing ancestor when the run
  is absent, run+stage checks, clear threshold/free-byte errors and receipt JSON.
- New `ops/worker_storage.py`: exact closed campaign cleanup by value, with a
  compact proof rather than a per-file manifest in argv. It reads the manifest
  from the SHA-verified archive, checks all remote file hashes and process
  identities, then uses ordinary `rm -- <files>` and bottom-up `rmdir -- <dirs>`.
- `ops/formal_eval.py`, `formal_prestage.py`, `formal_workers.py`: refuse low-space
  prepare before mkdir/push, recheck a prepared dispatch before its one-shot
  claim, check before each worker generation, pause replacements up to600s
  while retaining the admitted RAM/EGL slot, record disk/pause/recovery evidence.
- `ops/formal_finalize.py`, `sync_back.py`: attest archive size as well as SHA,
  seal `SEALED_EVIDENCE_RECEIPT.json` before deletion, keep public RECEIPT
  PENDING during cleanup, then publish PASS with the removal/refusal record.
  New detached CPU releases include the helpers. A legacy imported overlay
  without worker_storage explicitly retains remote evidence and still seals;
  it never imports unpinned working-tree helpers.
- `ops/cache_control.py`, `queue_retry.py`, `local_workers.py`: include
  worker_disk in fresh/narrowly refreshed worker payloads. Simulator/client,
  native model, server, episode, YAML and driver bytes stay pinned. Q's
  concurrent local/timan1 orchestration changes are preserved.
- `ops/library_campaign.py`: same configurable threshold on the library CLI.
  `formal_runtime.py`, `formal_resume.py`, `cache_resume.py`, `library_repair.py`:
  reject unfinished receipt publication as a source for pins/continuations.
- `ops/tests/test_worker_storage.py` (46 cases), `test_formal_finalize.py`,
  `test_recycle_resume.py`: cleanup/refusal, low-space admission/pause/deadline,
  four-node scope, UID/owner checks, by-value real CPU deletion and detached
  old/current overlay sealing. Launcher fixtures in `test_concurrent_fleets.py`,
  `test_mps_queue_retry.py`, `test_formal_eval.py` now provide disk admission;
  concurrent Q fixture/API updates were preserved.
- `REQUESTS/P6_to_Q.md`: RUN_CONFIGS pitfall, threshold, canonical paths and
  stable-seal publication guidance. Evidence lives under
  `ops/cache_study/evidence/{worker_storage_cpu,timan107_DISK_ADMISSION}.json`.

Interface: existing R17 model/benchmark/recipe contracts remain v1. The new
disk receipt is `r17.worker_disk.v1`; cleanup records are under
`RECEIPT.worker_storage_cleanup`. `scientific_evidence_seal` pins the immutable
sealed evidence receipt. `archive_size_bytes` supplements the existing SHA;
snapshot archive members/encoding and native inference behavior are unchanged.

Cleanup rule: automatic retirement requires a scientifically successful PASS,
confirmed closure, locally installed complete evidence, local archive SHA and
size equal to the remote attestation, exact canonical path and owning UID,
same gone fleet/worker identities, a free exclusive journal lock, and every
remote file equal to its sealed hash. Only `<runs>/<run>_<node>`, its exact
`.tar` and matching `.transfer.json` are removed. Local archives, installed
evidence, private dependencies/payloads/configs, process receipts and all
other campaigns/projects remain. Failed/pending/resumable campaigns are
retained. Refusals record RETAINED with the concrete reason/stderr and preserve
scientific PASS. Missing remote-size attestation or a different fallback/native
tar is retained rather than guessed away. New cache snapshots use the
finalizer's bounded8GiB stage in-process to avoid creating a second fallback
tar solely because of Q's512MiB qualification default; that default is restored.

Guard: default **8GiB =8589934592 unprivileged available bytes**, positive and
configurable via `--worker-min-free-gib N` in formal_eval/library_campaign.
Both run and `/tmp` stage filesystems must pass before deployment/dispatch.
The fleet checks its actual run filesystem before launch/recycle and records
free bytes, minimum, inode availability and pause/recovery events. Replacement
wait is bounded600s; it consumes zero new RAM/VRAM reservation. All repeated
fleet checks are local statvfs, adding no periodic tether calls. The same
logic applies to timan107, timan108, local direct workers and Q's newly wired
timan1 canonical run path. P6 contacted only timan107 for read-only CPU probes.

Hook table:

| Hook | DP/Fast/Cosmos | Default/native proof |
|---|---|---|
| Deployment/dispatch disk admission | generic, including local/timan1 | no model imports, no CUDA, no native payload edits |
| Generation/recycle disk pause | shared formal_workers;600s | held RAM/EGL slot reused; CPU children tests |
| Closed worker evidence retirement | shared CPU finalizer | complete local evidence retained;46 storage tests |
| Public scientific completion | stable PASS after cleanup | real CPU old/current overlay tests; native sentinel unchanged |

CPU verification (all with CUDA hidden and JAX_PLATFORMS=cpu):

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/ops/tests/test_worker_storage.py \
  exp/r17_integration/ops/tests/test_formal_finalize.py \
  exp/r17_integration/ops/tests/test_recycle_resume.py \
  exp/r17_integration/ops/tests/test_mps_queue_retry.py \
  exp/r17_integration/ops/tests/test_worker_gpu.py \
  exp/r17_integration/ops/tests/test_local_workers.py \
  exp/r17_integration/ops/tests/test_concurrent_fleets.py \
  --basetemp=/tmp/p6_storage_last \
  --junitxml=exp/r17_integration/tests/ops/receipts/worker_storage_last.xml
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/ops/tests exp/r17_integration/tests/ops --durations=12 \
  --basetemp=/tmp/p6_storage_fullops_final \
  --junitxml=exp/r17_integration/tests/ops/receipts/worker_storage_fullops_final.xml
.venv/bin/python -m pytest -q -p no:cacheprovider \
  exp/r17_integration/tests/common/test_default_off.py \
  tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py \
  tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_router_bundle_rebind.py \
  exp/offline_search/debug/tests/test_server_blocks.py exp/offline_search/debug/tests/test_server_observer.py \
  exp/offline_search/debug/tests/test_transport_debug.py exp/mw_finetune/tests/test_workers.py \
  --basetemp=/tmp/p6_storage_legacy_final \
  --junitxml=exp/r17_integration/tests/ops/receipts/worker_storage_legacy_final.xml
```

Completed focused gate: **167 passed,0 skipped,exit0,25.96s** (46 storage/disk
cases). Final legacy gate: **259 passed,0 skipped,exit0,22.25s**. First full
ops run:693 passed +40 passing subtests,3 failing model subtests and1 failing
launch fixture; those mocks lacked the new disk response, were corrected, and
passed the affected regression reruns. The clean final full ops run passed
**723 tests +43 subtests,0 failures,0 skipped,exit0,1427.32s**. JUnit reports
766 cases including those subtests. Receipts are `worker_storage_last.xml`,
`worker_storage_fullops_final.xml`, `worker_storage_legacy_final.xml` under
`tests/ops/receipts/`. Model/library size tests account for the longer CPU run;
no GPU was used. Native/default-off259 checks pass on the final sources.

Samples: `worker_storage_cpu.json` records two genuinely executed synthetic
verified cleanups (plain rm/rmdir), each6 files/1334 apparent bytes,582-byte
SHA-verified archive. Local archive/installed evidence and sibling campaign
files stayed intact. Production deletions=0 and gpu_executed=false. The20:51
read-only disk audit recorded6169714688 available bytes (5.746GiB), below the
8GiB admission threshold, with both run/stage on device64769; it created no
worker-node directory. Final read-only audit at **2026-10-10 21:32:18 CDT**:
**33328951296 bytes =31.040GiB free**, run/stage device64769, admission PASS
at8GiB. This increased space is an observation of external maintenance;
P6 still performed zero production deletions. Updated machine-readable
receipt: `ops/cache_study/evidence/timan107_DISK_ADMISSION.json`.

GPU commands: none needed or executed for this CPU storage fix. Existing smoke
and full-lane commands retain their controls/topology; a new dispatch needs
the worker disk admission to pass. No current lane, fleet, resident, lock,
port, process limit or frozen runtime was changed by this task.

Pending owners: **coordinator/Q** append the REQUESTS pitfall to RUN_CONFIGS;
the final measurement already passes the new space threshold. Active
frozen fleets retain their admitted code; do not restart them for this edit.
Legacy CPU overlays missing the helper retain evidence explicitly. Automatic
production retirement awaits the next qualifying closed campaign receipt;
P6 has CPU/synthetic receipts only. No P6 CPU/code work remains for this task.

## DP size500 local capacity continuation

This work preceded the subsequent CODE ONLY instruction. Authorized boundary
stop retained 1,734 accepted cells in dpc3464789869, with zero outstanding
accepted-attempt dispatches and all_owned_closed=true. Only verified DP owners
were signalled; H100 and its local workers were untouched. Capacity code is in
worker_topology, local_workers, cache_control, cache_resume, queue_retry,
cache_lane, formal_server, formal_eval, formal_finalize and cache_study;
cache_drain and tests/test_cache_segment_capacity are new. Explicit cache-only
orchestration leaves canonical native inputs, sampler, batch and YAMLs pinned;
P6's narrow DP admission bridge requires the recorded replica exception.
P4 permanent admission hook is requested in REQUESTS/P6_to_P4.md.

New running segment: dpc8713063796_r2_d1dd, local=33, timan107=0, 4xB8/5ms MPS,
startup width1, group-commit, N4, 4GiB RAM /576MiB renderer reservation and
8GiB server growth reservation. The requested 60 workers do not fit the measured
RAM/VRAM guards: the untouched H100 fleet already has16 local workers; local
admission allowed33. Four server processes measured104–112% single-core CPU,
so the requested <70% target was NOT met. Accepted rate17.8/min before and
56.8/min over379.1s after (359 additional cells,3.19x). At22:21:45 the new
segment had532 accepted, merged6,780/8,000, zero errors. The saved live audit
verified279 episodes,18,029 decisions/ACKs,279 final ACKs,16,196 library actions,
1,833 stall-policy actions,183,300 NFE, stable four resident PID births and
cg→crg hot-switch. These are bounded observations, not a final PASS receipt.
Evidence: ops/cache_study/evidence/dp_size500_capacity{,_result}.json,
dp_size500_live_audit.json; lane log cache_lane_4090_local1010.log.

CPU receipts cache_capacity_complete.xml:190 passed +3 subtests,0 skips;
cache_capacity_legacy_final.xml:259 passed,0 skips. A later focused run saw
57 pass/1 failure after Q changed the default local RAM reservation; the fixture
now explicitly pins its intended4096MiB instead of assuming a global default.
That fixture and the live-fleet change will be rechecked in this handback's next
section. All future experiment control belongs to the coordinator.

## Live fleet attach/drain (CODE ONLY)

Interface r17.live_fleets.v1 is default off. No live campaign was launched,
stopped, resized, signalled or edited, and no GPU work ran. New files:
ops/live_fleets.py, live_fleet_driver.py, live_fleet_worker.py, LIVE_FLEETS.md,
ops/tests/test_live_fleets.py. Small opt-in hooks: formal_eval, formal_workers,
formal_driver, formal_driver_fast, cache_driver_legacy, cache_study,
driver_release, group_commit, formal_finalize, formal_resume, cache_resume,
local_workers, timan1_workers, worker_storage. The prior capacity fixture now
explicitly pins its4096MiB local reservation; the cold-refresh fixture includes
Q's new sync_back helper. Coordination and hook collisions are documented in
REQUESTS/P6_to_Q.md. Q's renderer/resource/periodic-sync edits were preserved.

Prepare with formal_eval's existing frozen control/server arguments plus
--live-fleets. Exact archived workers.tar, scientific campaign and benchmark
inputs are reused for every attach; all node routes are pinned at initial
preparation. Client protocol checks fail before deployment if the supervisor or
GPU3/local authorization hook is missing. CLI requests are durable; only the
owning launcher acts. CPU preparation retains the parent's bounded retry/call
journal in a separate thread, leaving existing lease checks serviced. Every
fleet has a distinct evidence key even on the same node. Accepted worker census
and dispatch rows carry fleet_id/worker_node. Native TCP/results, generation
fences, group-commit fsync ordering, endpoint/bundle binding and native producers
are reused. Adding fleets can change MISS batch composition, and its resource
history is explicit. Replica counts and resident process births are unaffected.

Drain fences the next assignment under the registry/scheduler lock, then sends
an SHA-checked supervisor marker. Current episodes complete; next pull receives
SHUTDOWN; no replacements follow the marker, including normal N4 exits. Closure
must attest the supervisor, child and descendants gone. Finalization fences
attachment preparation and seals every started fleet's SHA/size/manifest,
registry, input pins and resource history. Missing attached evidence refuses
PASS. Source resume acceptance rules are unchanged; extra closed fleet source
pins prevent mutation/overlap. Cleanup now also holds Q's incremental sync lock
and excludes that control lock from evidence hashes (matching sync_back's
snapshot), retaining busy or failed runs.

CPU receipts (CUDA_VISIBLE_DEVICES empty, JAX_PLATFORMS cpu, one math thread,
PYTHONDONTWRITEBYTECODE=1, pytest -p no:cacheprovider):

- tests/ops/receipts/live_fleets_final.xml:275 passed +3 subtests,0 failures/skips,
  59.63s. Targets: ops/tests/test_live_fleets.py, test_cache_segment_capacity.py,
  tests/ops/test_cache_lane.py, ops/tests/test_local_workers.py,
  test_recycle_resume.py, test_cache_capacity.py, test_formal_finalize.py,
  test_driver_group_commit.py, test_completed_driver_drain.py,
  test_mps_queue_retry.py, test_worker_storage.py, test_timan1_recycle.py,
  test_concurrent_fleets.py.
- live_fleets_legacy.xml:259 passed,0 failures/skips,21.23s: shared default-off,
  cache config/schedule/storage, serving identity/recorder/router rebind,
  legacy debug blocks/observer/transport and MW worker fixtures.
- After the final retry-context/cause logging change, live_fleets_context.xml:
  42 passed +3 subtests,0 failures/skips,10.26s (live fleet +group-commit suites).
- New24 live-fleet tests include the real client bootstrap/SHA installer,
  production launcher preparation path, stock TCP driver on fake episodes,
  current-result commit followed by drain, real CPU N4 supervisor/no replacement,
  T1 UUID/device guard authorization, config/protocol/source tamper refusal,
  ambiguous-spawn ownership, attach/finalization fencing, same-node fleet final
  archive sealing, missing-evidence refusal, resume pins and a real held sync
  lock. These are synthetic CPU receipts, not production/GPU qualification.

Earlier failing CPU rounds are retained honestly: stream registration was fixed;
Q's concurrent writes removed T1/drain hooks, which were restored and protected
by a pinned protocol check; snapshot lock composition was fixed; and the old
GroupCommit call signature is preserved when live_fleets is absent. Source SHA
snapshot: tests/ops/receipts/live_fleets_sources.json. The baseline/default-off
path creates no fleet files or extra evidence stream, and no live driver method
is wrapped. Source/native worker payload and task-plan byte equality are tested.

Coordinator commands for a running **live-capable H100** campaign fw_live01:

```bash
.venv/bin/python -m exp.r17_integration.ops.live_fleets attach \
  --campaign /data/mw_finetune/eval/fw_live01 \
  --fleet-id t124 --node timan1 --workers 24 --run
.venv/bin/python -m exp.r17_integration.ops.live_fleets status \
  --campaign /data/mw_finetune/eval/fw_live01
.venv/bin/python -m exp.r17_integration.ops.live_fleets detach \
  --campaign /data/mw_finetune/eval/fw_live01 --fleet-id t124 --run
```

Use --node local or --node timan107 for those fleets; initial_timan107 drains
that original fleet. Do not wrap these worker-control requests in the4090 GPU
lock. They never relaunch an H100 server; the existing owner holds server/MPS
exclusivity. Read-only two-minute check: snapshot source SHAs/resident births,
submit, poll status at30/60/90/120s, require ATTACHED plus matching live driver
census and per-fleet dispatch→accepted-attempt joins, stable resident births;
then optional detach must stop new dispatches/replacements and eventually
produce DRAINED. Full steps and expected file checks are in ops/LIVE_FLEETS.md.
If startup/current episode exceeds2min, report PREPARING/DRAINING and continue
observing; do not force-stop anything.

Limits/pending: old frozen running campaigns lack this protocol and fail closed;
they need a coordinator-scheduled new segment prepared with --live-fleets.
No retroactive/in-place runtime upgrade is attempted. Q owns optional forwarding
of --live-fleets for newly prepared cache_lane jobs (requested; direct
formal_eval cache preparation already supports it). Shrink is by whole fleet.
Local/T1 installs support LIBERO; remote timan107/108 support the existing formal
benchmark payloads. Library teachers/native-parity-only are excluded. T1=24 is
an admission request, not a promise of free144GiB RAM/24GiB VRAM plus headroom;
the pinned GPU3/UUID/namespace guards remain mandatory. Actual production attach
and render/controller throughput are coordinator qualifications still pending.

Final drain hardening: supervisor spawn and drain-marker installation now share
an opt-in per-fleet lifecycle lock. A replacement cannot pass the drain check
then spawn after the marker commits. An empty fleet drained before startup
closes without requesting new RAM/VRAM. The live preparation thread preserves
the parent's ContextVar retry journal; attach/drain failures retain concrete
node/fleet/step/stderr. T1's6144MiB lease cannot be undercut by another node's
optional RAM override. Large resume cohorts are SHA-staged files, including a
20,000-UID CPU regression with every remote argv below32KiB; status omits the
large nested campaign copy. These refinements do not change the default path.

Final receipts after this hardening: live_fleets_drain_final.xml114 passed
+3 subtests,0 skips/failures,25.79s (live, recycle, T1, concurrent fleets,
group-commit, storage). The subsequent large-cohort check is recorded in
live_fleets_large_final.xml. The new live suite now contains27 CPU regressions;
previous275+3 and259 legacy receipts remain historical passing broad checks.
A transient fixture failure when Q stopped creating a sync lock by default was
resolved by having the busy-lock test explicitly create its own lock. No global
default was assumed or reset. Final tested source/test hashes are in
live_fleets_sources.json; coordinator should compare them before preparation.

Final all-change CPU rerun: **live_fleets_sealed.xml278 passed +3 subtests,
0 failures/skips,62.31s**; **live_fleets_legacy_sealed.xml259 passed,
0 failures/skips,20.20s**. The final standalone live suite is27 passed,
0 failures/skips,6.22s (live_fleets_large_final.xml). Source comparison after
both final suites reports no changes from live_fleets_sources.json. These final
receipts supersede the earlier intermediate counts above. No GPU or live-fleet
operation was performed. ops/LIVE_FLEETS.md contains coordinator commands and
the two-minute check; no current frozen campaign is claimed to have the feature.
