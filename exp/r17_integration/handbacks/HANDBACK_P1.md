# P1 handback — shared framework and layout-aware layers

2026-10-09, shared checkout `/home/weiland/projects/openpi`, branch `Ziyang`.
Interface: **`r17.contracts.v1`**, binding **`r17.adapter_binding.v1`**,
recipe **`r17.layout_recipe.v1`**. P1's shared CPU gate is qualified; real model
GPU/renderer/quality gates are not qualified. No GPU was used, no checkpoint
loaded, no commit/push, no other owner's process or training files modified.

## 1. Owned files and actual changes

| Owned area | Delivered change |
|---|---|
| `common/{__init__,contracts,request_rng}.py` | CPU-safe contracts, semantic request-local RNG, native snapshot/grid/dtype/conditioning binding, capability/manifest/receipt types |
| `common/{provider,serving}.py`, `common/INTERFACE.md` | Explicit provider/trace registration, strict opt-in config/artifact validation, stock connection assembly, staged full/K/CP1 dispatch, acknowledged tail, observe-only capture |
| `common/{query_view,fake_adapter,qualification,replicas}.py` | Outcome-free query facade, deterministic CPU staged adapter/batcher, bounded explicit qualification CLI, command-only replica plan reusing stock proxy |
| `common/layers/{__init__,math,recipe,guard,periodic,descent,storage_bridge}.py` | One implementation of AWM, only-no-progress, library-only LOEO corrector, deterministic periodic forcing and descent gate; P6 LibraryView fitter |
| `common/samples/fake_h{5,16,32}_{k2.yaml,model.json}` | Content-bound diagnostic layouts/configs; all three YAMLs actually loaded/validated with stock storage |
| `tests/common/` | Contracts/RNG/native suffix, real stock HIT/MISS/tail, two-client/two-bundle loopback, legacy byte certificate, frozen numerical parity, BatchingCore isolation, qualification, π/P6 bridge witness |
| `REQUESTS/P1_to_{P2,P3,P4,P5,P6}.md` | Published v1 seams, actual failing witnesses and resolved additive requests |
| `handbacks/HANDBACK_P1.md` | This handback |

Only four shared source files changed: `cache/config.py` (45 changed lines in
diff), `cache/types.py` (+24), `cache/storage_types.py` (+4), and
`serving/websocket_policy_server.py` (24 changed lines). Combined tracked shared
diff: **84 insertions, 13 deletions**. `git diff --check` passed.

Shared config adds optional `adapter_binding=None`; strict nested YAML parsing,
native key builder and artifact validation run only with that binding. Shared
types add a separate explicit native schedule registry. CachePayload appends
optional native metadata after every legacy positional argument. WebSocket adds
optional `bundle_validator=None`, invoked before storage build and before atomic
publication. Bound model islands enforce read-only runtime without importing
the Pi/JAX serving script. Original no-binding enforcement remains unchanged.

Generated pytest scratch and copied worker payloads remain inside owned
`tests/common/artifacts/`; its `.gitignore` keeps only listed receipts. Local
`collect_ignore` excludes scratch from test discovery. Do not recursively collect
all model tests together: independently owned model suites reuse test basenames.

## 2. Default-off certificate

`tests/common/test_default_off.py` runs two separate CPU subprocesses. The baseline
loads the original **four shared modules from `git show HEAD:<path>`**, without
changing the checkout; the current process loads the edited files. It compares
legacy config values/errors, metadata/control/action msgpack SHA256 and lifecycle
events. Assertions passed in the final common receipt. Only nondeterministic
`server_timing` and the newly appended optional config field are excluded from
that comparison; controller action/metadata/ack bytes are compared exactly.

Additional tests load the legacy positional CachePayload and simulate old
objects missing both appended attributes. Legacy schedule IDs remain unchanged;
native schedule IDs cannot alias legacy uniform-flow schedules. No provider is
registered on import, and contract/provider imports do not load torch/JAX/vendor.
The broad legacy suite below passed with hooks absent.

## 3. CPU commands and actual receipts

All pytest commands used this exact prefix from the checkout root:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest
```

Append each row's targets, `-p no:cacheprovider`,
`--basetemp=exp/r17_integration/tests/common/artifacts/<scratch>`,
`--junitxml=exp/r17_integration/tests/common/artifacts/<xml>.xml -q`.
Receipts are real JUnit output, not inferred capability claims. `cpu_receipt.json`
records the complete commands and counts. `source_hashes.json` records the
handoff source snapshot; this is a live shared checkout, not a freeze of others'
future edits.

| Run / targets | Scratch / XML | Exit / result |
|---|---|---|
| `exp/r17_integration/tests/common/test_*.py` | `g0_handback` / `g0.xml` | **0; 46 passed, 0 skipped**, 21.60s; includes all shared gates and the P1-owned provenance shim witness |
| `tests/serving/{test_policy_identity,test_per_step_recorder,test_replica_proxy,test_router_bundle_rebind}.py`, `tests/exp/metaworld/{test_env,test_runner}.py`, `exp/mw_finetune/tests`, `tests/conductor`, `exp/offline_search/rounds/r16/{test_conductor,test_groot_servers}.py`, `tests/cache/{test_config,test_denoise_schedule_config,test_phase3_validator,test_cache_storage}.py`; `-k 'not same_identity_same_initial_state_across_processes'` | `legacy` / `legacy.xml` | **0; 457 passed, 6 skipped, 1 deselected** |
| `tests/serving/test_replica_proxy.py -m env_dependent --run-manual` | `proxy` / `proxy.xml` | **0; 6 passed, 23 deselected**, all six localhost skips above resolved |
| `exp/r17_integration/tests/models/fastwam/test_*.py` | `fastwam_release` / `fastwam.xml` | **0; 49 passed, 0 skipped** |
| `exp/r17_integration/tests/models/dp/test_*.py` | `dp_release` / `dp.xml` | **0; 56 passed, 0 skipped** |
| `exp/r17_integration/tests/models/groot/test_*.py` | `groot_release` / `groot.xml` | **0; 43 passed, 0 skipped** |
| `tests/common/test_*.py`, `tests/benchmarks/test_*.py`, `tests/conductor/test_*.py`, `tests/models/pi05/test_*.py`, `tests/debug/test_*.py`, `tests/search/test_*.py`, `tests/ops/test_*.py`, all prefixed `exp/r17_integration/` | `cross_complete` / `cross.xml` | Latest broad bridge: **1; 183 passed, 1 failed, 0 skipped**; final standalone P2 probe lacks newly required arm provenance. All four dispatched episodes pass. Owned local shim reruns the same witness with complete probe provenance. |

An earlier broad bridge passed **180 tests** before P6 tightened the provenance
check; that output is historical, not a claim that the present unmodified P2
fixture passes. The failed current witness is retained and requested from P2.
The only intentionally unrun legacy test requires real EGL across processes.
No real MetaWorld renderer, GPU weights, external backend or quality experiment
was exercised. Model suite fixtures use CPU toy/vendor hooks as documented by
their owners. Warnings were package deprecations and DP transformer/JUnit
property warnings, not skipped qualifications.

Additional executed CLI (exit 0; receipt status `passed`):

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python \
  -m exp.r17_integration.common.qualification --fake --deadline-seconds 30 \
  --output exp/r17_integration/tests/common/artifacts/fake_qualification.json
```

Use a new output filename to rerun: receipts are never overwritten by this CLI.
`samples.json` records successful stock config/storage validation for all three
sample geometries. No source cleanup, dependency installation or deployment ran.

## 4. Framework/model/benchmark hook table

| Inventory area | Shared path and actual hook | Qualified scope / owner |
|---|---|---|
| S02/S09/S10/S11/S12 | Original WS factory/control/lifecycle → stock config/components/storage → CacheOrchestrator; explicit ModelSpec/bundle validator | CPU two clients/two generations, idempotent same-ID selection, unknown/bad binding rejection without destroying prior stack, disconnect cleanup. P1 framework; P2–P5 metadata/loaders |
| S04/S05 | `ReplicaPlan`, original `replica_proxy.run_proxy` | CPU command plan and six real localhost proxy tests; model process startup/P6 deployment pending |
| S06/S07 | Original BatchingCore with adapter-owned stage batcher | Fake S3 bucket/failure/event/no-alias and separate model CPU suites passed. Real GPU B2 and endpoint integration are model-owner gates; no generic full-stage batching claim |
| P01/P02/P03 | prepare → S1 → native named fields → stock CP1 check → S2/sample on MISS → one `to_wire` | FULL_HIT `(1,0,0)`, MISS `(1,1,K)`, tail `(0,0,0)` tested; exact real keys, finite normalized/wire geometry, dtype-preserving wire/capture |
| P05 / layer 1 | `LayoutRecipe.fit/query`, frozen PCA/action-aware metric/early and fresh-MISS continuity math | Independent original R2 parity for double-view top-k/kernel/chunk and early/stale/fresh cases; actual serving query now receives acknowledged prior normalized chunk and executed count |
| P06 / layer 2 | `NoProgressGuard`, `force_look` tail veto | Original K1 span arithmetic tested with blind gaps and MISS reset; no gripper/task constant |
| P07 / layer 3 | R10 library-only LOEO RFF384/ridge100/seed0, active-motion correction, five-fold distance scale | Original helper, feature, correction application and distance-scale parity; whole donor episode excluded, gripper preserved bitwise, padding irrelevant to active metric |
| P08 / layer 4 | Deterministic R11 fractional-gap coin/accepted V/M history; guard priority | Exact original coin/forcing parity tested; no worker/replica/order dependence |
| P09 / layer 5 | `TaskDescent` / `passes_gate` | CPU complete-cohort20 and first-failure gate; error/stale/duplicate/missing results excluded. P2 owns per-task attempt fences, queue, freeze and test authorization |
| P10/P11/P12 | Explicit provider `sampler_for_steps`, H-bounded commit, actual execution ack | K1/2/default NFE, native identity, cursor, terminal/stale/missing ack tested. Policy tails report reused FULL_HIT while source remains `policy_tail`; no invented H5 commit10 |
| Debug/capture/store | `DecisionCapture`, optional sink/trace provider, native SampleResult snapshots/capture | Observe-only CPU byte/NFE parity, P6 closed NPZ/H5 interop, current tail state, concurrent same-bundle ack identity. Model-specific fields P2–P5; writer/transport/readers P6; physical capture P2 |
| Advanced P04/CP2/warm/reset/trace variants/routing | Native snapshot/schedule contracts plus explicit capabilities | Direct native hooks are owned/tested by models; generic stock advanced dispatcher is intentionally rejected pending certification. No legacy flow time substituted for DDPM or shifted native time |

Default capture is off. Capture copies values from executed stages, invokes no
extra forward/RNG call, and does not fabricate absent S1/S2/S3 on a tail/HIT.
Normalized cache storage uses float32; wire dtype (including π float64 inverse)
is preserved exactly. Decision execution remains pending until the actual ack;
P6 records `executed_count=None` before the joined ack, rather than treating
zero as an observed control count. `ack['request']` identifies the original
decision even when concurrent clients both use sequence zero.

## 5. Native grid, feature, norm and storage samples

The delivered fake samples have one real diagnostic camera, keys
`vision_0:4, robot_state:4`, H5/16/32, action/state4, diagnostic scale2 inverse,
K2 grid `(1.0, 0.5)` and deltas `(-0.5,-0.5)`. Commit is 5/10/10 respectively.
Register `fake_provider('sample.h16.v1', fake_spec(horizon=16))` explicitly before
loading `common/samples/fake_h16_k2.yaml`. These are not real model artifacts.

| Consumed model hook | Real layout/key contract (CPU fixture only) | Native identity / transform |
|---|---|---|
| π–MW / P2 | H5, model width32, active4, state4, one vision32768 | π native K10/2/1 grids, original MW inverse once; formal replan5 means commit5 only |
| Fast-WAM / P3 | H32; MW one real view/state4/action4; LIBERO fused real two-view key/state8/action7 | Shifted flow native grid/deltas/config digest, benchmark-specific minmax/gripper handling |
| DP / P4 | H16; MW one real64/state4/action4; LIBERO two real64/state8/action7 | Native DDPM100/2/1 integer grids; DDIM separate config/eta identity; internal affine plus external quantile inverse |
| GR00T–MW / P5 | H16/storage32/active4/state4, one vision32768, Sawyer slot31 | Frozen K4/2/1 endpoints; bundles must match resident head K; no global per-request head mutation |

Each artifact must bind model/checkpoint/norm/layout/features/sampler/protocol.
An empty stock backend is valid for full/plain; populated retrieval requires
`artifact_meta.r17_binding`, preserved `artifact_meta.model.r17_binding`, or a
validated P6 `r17_manifest` whole-ModelSpec binding. Wrong geometry is rejected
before inverse transformation. Native snapshots bind checkpoint/norm/layout/
sampler/grid/index/native timestep/conditioning and tensor dtype. Portable native
numeric snapshots remain P6 versioned data, not legacy float-key warm payloads.

## 6. GPU-ready command — NOT EXECUTED

Coordinator only, after training releases a GPU; one immutable π checkpoint,
one process, one serial slot. Write `pi_descriptor.json` with
`{"checkpoint":"<approved immutable checkpoint>","checkpoint_identity":"<content SHA>","device":"cuda"}`
and at most eight fixed observations in a no-pickle NPZ. NPZ keys are wire
observation keys with a leading observation axis, including Unicode `prompt`.
These paths are operator-provided approved assets; do not substitute prohibited
paths. The operator chooses the single device and new owned output directory.

```bash
timeout --signal=TERM --kill-after=10s 14m \
  env CUDA_VISIBLE_DEVICES="${R17_DEVICE}" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src \
  "${R17_PI_PYTHON}" -m exp.r17_integration.common.qualification \
  --adapter-module exp.r17_integration.models.pi05.adapter \
  --provider-module exp.r17_integration.models.pi05.provider \
  --spec-json "${R17_RUN_DIR}/pi_descriptor.json" \
  --observations "${R17_RUN_DIR}/fixed_observations.npz" \
  --deadline-seconds 780 --output "${R17_RUN_DIR}/framework_gpu_receipt.json"
```

This executable probe checks K10/2/1 stock MISS vs staged helper, debug byte
parity and synthetic FULL_HIT/NFE using one shared resident model. H5/replan5
correctly marks extra tail not applicable. Loading is inside the outer timeout;
failure/incomplete receipt never qualifies a gate. The inner deadline starts
after loading. No model subprocess supervisor is spawned.

**Limit of this command:** it uses direct factories and a staged helper reference,
not upstream full Policy equivalence, live WS GPU, B2, physics, throughput or SR.
P2/model owners must run their bounded upstream/B2/WS slots separately; the CPU
WS/proxy receipts do not turn these into GPU-qualified claims. A timeout that
kills loading may leave no receipt; record it as partial, do not extend the slot.
LIBERO/MW and each new model require separate coordinator-allocated serial slots.

## 7. Known limits and pending owners

| Pending item | Owner / disposition |
|---|---|
| Final P2 standalone capture probe has empty arm; latest broad fixture fails P6 strict provenance | **P2**, concrete witness in `REQUESTS/P1_to_P2.md`. P1-owned `test_pi05_bridge.py` completes only fixture request defaults and reruns the original witness without editing P2 files or weakening validation |
| Native warm continue/reset, CP2, trace twins/augmentation, online RIT, routing/split advanced dispatcher end-to-end | **P1 + relevant model owner + P6**, follow-up G3. Current native contracts/direct hooks exist; generic config fails closed. Preserve stock combination restrictions until native executor receipts exist |
| Real checkpoint equivalence, native resume/capture and B1/B2 GPU parity, compiled/stream/cache-mutating vendor behavior | **P2/P3/P4/P5**, G2; zero GPU today |
| P2 numerical descent threshold may call shared `passes_gate` instead of its equivalent local arithmetic | **P2**, nonblocking consolidation request; attempt/freeze state remains benchmark-owned |
| Real renderer/task-state/goal certificates and complete physical capture | **P2**, benchmark gate; real EGL test intentionally excluded |
| External Qdrant, complete advanced search/judge family, live trace/resident combinations | **P1/P6 + hook owners**, outside the CP1 full-hit/miss certificate; not inferred from importability |
| Sealed real teacher library/dev/test stores, complete inventory-by-ID G3 audit, transport/analysis/deployment receipts | **P6**, consume shared fitter/provider and model captures; no quality claim from synthetic stores |
| Immutable final models, paired full500/library/dev-descent/test, pricing/SR/bootstrap | **Coordinator**, G4; no formal outcome inspected to tune library math |

## 8. Decisions

Kept the stock WS/control/proxy/cache/backend/search/judge/component path and
existing legacy entry restrictions. Provider registration and native binding
are explicit; model islands are not auto-imported. Rejected native advanced
combinations before publication instead of translating their schedules to
legacy uniform flow. Published additive v1 seams through owned REQUESTS files.
Used independent frozen source comparisons for numerical parity and actual
localhost/control/event tests for framework behavior. Routine fixture scratch
was kept; frozen R16, harness, training and other package source remained intact.

## Fix round 1

Read REVIEW_A and REVIEW_B in full and consumed incoming P2–P6 and Q requests.
All assigned P1 findings are fixed and covered by passing CPU regressions.
This section supersedes the original shared-source implementation description:
**all four sealed files now equal HEAD byte for byte; final shared-source diff
is zero**. No R16 amendment, frozen-file edit, commit/push, GPU use or external
process intervention occurred.

### Owned files and actual change

- New `common/config.py`: opt-in config subclass, strict nested YAML parser,
  complete stock validation with only the native key-builder extension,
  original backend/gate/judge/search/component factories.
- New `common/recipe.py`: bundle-local recipe schema validation, identities,
  sealed P6 teacher-store checking, immutable fit cache and fresh mutable forks.
- New `common/websocket_policy_server.py`: subclass/socket facade around the
  original handler, atomic native bundle publication and terminal execution
  controls. The original module still owns the bundle registry. No global
  function/class patching. Optional local client exposes terminal ack API.
- New `common/native.py`: native schedule registry and appended payload fields,
  isolated from sealed legacy types/storage classes.
- Updated `common/{contracts,provider,serving,qualification}.py`,
  `common/layers/{recipe,math}.py` and `common/INTERFACE.md`: strict context,
  per-bundle selection, teacher/MISS separation, execution protocol, measured
  cost ledger, core-aware capture factory, negotiated partial tails, and clear
  rejection of zero-scatter teacher metrics.
- Updated owned tests `conftest.py`, `test_contracts.py`, `test_default_off.py`,
  `test_layers.py`, `test_serving.py`, `test_pi05_bridge.py`; new
  `test_review_regressions.py`. Removed the old borrowed-fixture provenance
  monkeypatch. New sample YAMLs under `common/samples/` and receipts under
  `tests/common/artifacts/fix1_*`. Published migration/API notes in
  `REQUESTS/P1_to_P2.md` through `P1_to_P6.md`.
- Restored `src/openpi/cache/{config,types,storage_types}.py` and
  `src/openpi/serving/websocket_policy_server.py` to HEAD. The prior P1 shared
  patch is retained only as `artifacts/shared_before_fix.patch` for audit.

Interface stays **r17.contracts.v1**; recipe schema **r17.recipe.v1**;
execution declaration **r17.execution.v1**; optional partial-tail declaration
**r17.partial_tail.v1**. Native imports now come from `common.config`,
`common.native`, and the local WebSocket subclass. Existing stock callers
and stock imports are untouched.

### Finding → change → regression → status

Tests below refer to `tests/common/test_review_regressions.py` unless named
otherwise. Its **28 current tests** pass as part of the final full G0 suite.
These pin the behavior absent in the reviewed implementation: sealed bytes,
recipe-driven bundle selection, no default context, timing/capture fields,
terminal ack transport, and commit negotiation. Reviewer identity/commit
repros became strict original-index and nondivisible-commit witnesses.

| Finding | Change | Passing regression / evidence | Status |
|---|---|---|---|
| A #2 major, R16 seal | Restore four sealed sources; move hooks into local config/native/subclass | `test_sealed_shared_source_is_head_and_opt_in_types_are_local`; both `test_default_off.py` subprocess certificates; actual `r16.analyze verify` → `DESIGN_VERIFIED; no outcome reads` | Fixed; zero shared diff |
| A #3 major, bundle recipe/K/endpoint identity | Versioned YAML library/fit/teacher identity, guard/corrector/gap or target IR, retrieval k/kref, MISS K and commit; fit/state selected per bundle independently of K; generation IDs cannot change content | `test_full_and_plain_low_k_select_recipe_per_bundle`; six `test_recipe_identity_seals_each_arm_setting` cases; `test_real_ws_recipe_load_is_atomic_and_terminal_ack_is_recorded`; P2 endpoint/dispatch tests in passing cross suite | Shared fix complete; P2 consumes identity |
| A #4 shared, teacher versus MISS | Validate full-action teacher provenance separately from MISS/native grid; retained complete artifact ModelSpec verifies code/transform even when sealed stock loader drops unknown metadata | `test_teacher_library_is_independent_of_miss_grid_but_not_model`; `test_stock_action_artifact_teacher_provenance_remains_strict`; `test_actual_stock_loader_retained_model_spec_proves_code_provenance`; native snapshot rejection tests remain passing; GR00T suite59 passes | Fixed; P5 consumes |
| A #5 shared, connection capture | `capture_sink_factory(core)` receives actual bound sampler and per-connection core; P6 factory owns writer sharing/closure | `test_core_aware_sink_has_actual_bundle_sampler`; actual π/P6 WS/conductor capture witness and GR00T production capture tests | Shared fix complete |
| A #6 major, cost/final ack | Every decision publishes actual host wall total and stage ledger, including state-only tails; episode_end handles final executed ack without another infer; receipts/callbacks preserve original request and cursor_end | `test_actual_host_cost_stage_ledger_and_capture_pricing_exclusion`; `test_final_ack_binds_request_and_cursor_before_mutating_state`; real WS terminal count3/identity; cross suite contains P2/P6 joined ledgers | Fixed at framework seam; P2/P6 consume |
| A #7 / B2 common context | Recognize orig_init_state_idx; real bundles reject absent campaign/task/global id/original idx/seed/explicit seed before model work; fake diagnostic exemption is explicit and unavailable to real protocols | Six `test_missing_real_context_fails_before_any_model_work` cases; `test_original_init_alias_and_suite_identity_change_noise_and_periodic_coin` | Fixed; benchmark namespace/global ids owned by P2 |
| B1 shared extended commit | Require declared execution protocol; refuse missing next ack instead of silent downgrade; terminal ack validated against preceding available prefix | `test_extended_commit_declares_ack_and_refuses_silent_downgrade`; real WS terminal ack; P2 original LIBERO kernel tests in cross suite | Fixed; runner wiring P2 |
| A #9 nondivisible commit | Default commit must be multiple of replan. Explicit partial-capable binding plus episode handshake allows LIBERO32/replan5 and actual tail2, never silently applied to MW | Two `test_nondivisible_metaworld_commit_is_rejected` cases; `test_libero_commit32_requires_negotiated_partial_tail_and_ack` reaches cursor32 with one model plan | Fixed; documented opt-in exception |
| B9 shared pricing contamination | Any native capture or sink marks pricing_eligible=false; timing labelled host wall, capture preparation measured separately | `test_actual_host_cost_stage_ledger_and_capture_pricing_exclusion`; debug-on/off bytes/NFE tests; P6 pricing-exclusion tests in cross suite | Fixed; no GPU pricing claim |
| Incoming Q stationary cohort | Explicitly reject nonfinite/zero scatter with a data error; preserve original nondegenerate numerical equation | `test_q_stationary_teacher_metric_fails_with_explicit_data_error` covers identical direct rows and actual recipe fit; frozen-equation bitwise test passes | Fixed; no invented metric |
| Incoming P3/P5/P6 nullable YAML | Accept optional field_similarity=null; retain nested typo rejection | `test_nullable_yaml_and_unknown_nested_recipe_keys`; actual model/P6 hot-load suites | Fixed |

No assigned finding was rejected. Model-local observation aliases, tail state
normalization, CLI hooks, capability receipts and reader auditing remain with
P2–P6; their CPU suites consume these interfaces. Generic advanced native
dispatch remains an explicit qualification limit, rather than a supported
capability inferred from direct hook existence.

### Default-off proof and CPU commands

[fix1_seal_receipt.json](../tests/common/artifacts/fix1_seal_receipt.json)
records the actual verify command, exit0/stdout and four HEAD-equal SHA256s.
Two subprocess default-off tests compare errors, config/metadata, lifecycle
acks and action bytes to git HEAD, both without hooks and after importing all
opt-in modules. The latter proves imports do not patch stock factories.

All commands used the prefix from §3, with CUDA_VISIBLE_DEVICES empty,
JAX_PLATFORMS=cpu, no bytecode/cacheprovider and owned unique scratch paths.
Full exact targets/options, timestamps, exit codes, counts and XML digests are
in [fix1_cpu_receipt.json](../tests/common/artifacts/fix1_cpu_receipt.json).

| Actual command targets / options | XML basename | Result |
|---|---|---|
| `exp/r17_integration/tests/common/test_*.py` | `fix1_g0_release.xml` | **75 passed; 0 failed/errors/skipped**, exit0; includes all28 review/Q tests and both default-off proofs |
| Common + benchmarks + conductor + π0.5 + debug + search + ops, each `test_*.py` | `fix1_cross.xml` | **268 passed; 0 failed/errors/skipped**, exit0; predates the additional import-default-off and stationary tests, both included in final G0 |
| `test_review_regressions.py` before stationary addition | `fix1_regressions_final.xml` | **27 passed**, exit0 |
| Stationary-cohort witness + frozen numerical bitwise comparison | `fix1_stationary.xml` | **2 passed**, exit0 |
| Exact broad legacy targets in §3, same EGL deselection | `fix1_legacy.xml` | **457 passed, 6 skipped, 1 deselected**, exit0 |
| `tests/serving/test_replica_proxy.py -m env_dependent --run-manual` | `fix1_proxy.xml` | **6 passed, 23 deselected**, exit0; resolves those six socket skips |
| `exp/r17_integration/tests/models/fastwam/test_*.py` | `fix1_fastwam.xml` | **70 passed**, exit0 |
| `exp/r17_integration/tests/models/dp/test_*.py` | `fix1_dp.xml` | **79 passed**, exit0 |
| `exp/r17_integration/tests/models/groot/test_*.py` | `fix1_groot_final.xml` | **59 passed**, exit0 |
| `python -m exp.offline_search.rounds.r16.analyze verify` under the same CPU environment | seal JSON | **exit0**, `DESIGN_VERIFIED; no outcome reads` |

Counts overlap; they are not additive unique-test totals. Model suite counts
are P1's actual invocations, not claims about later tests added by other agents.
The EGL across-process test was not executed. Earlier failures are preserved
in fix1_early/smoke/g0/groot XML as superseded diagnostics: nullable parser and
probe lifecycle issues, the repaired P2 fixed-arm physical fixture, and stock
artifact retained-metadata compatibility. Final release receipts are clean.

The fake qualification CLI also exited0, using two fixed observations and
K4/2/1, capture parity, synthetic CP1 HIT and acknowledged tails:
`python -m exp.r17_integration.common.qualification --fake --deadline-seconds 30
--output exp/r17_integration/tests/common/artifacts/fix1_fake_qualification.json`.
This is diagnostic framework qualification, not model quality.

### Hook table and actual samples

| Framework hook | Final behavior / tested boundary |
|---|---|
| `config_from_dict/load_cache_config` | Local schema, complete stock checks; unknown nested keys and invalid store/fit/teacher identities fail before publication |
| `make_bundle_validator/provider` | Exact resident model/protocol, immutable retrieval model identity, explicit teacher sampler; native snapshot grid equality stays strict |
| `make_recipe_factory` | Library-only P6 fit, shared immutable arrays, fresh guard/memo; guard/corrector/periodic settings follow selected YAML |
| `bundle_identity` | Config/recipe/fit/store/K/commit identity; local output directory/path relocation exclusions only |
| `ConnectionPolicy` | Original CP1 orchestrator path, one S1, one inverse, actual NFE; strict request identity; ack cursor; no extra stage on tail |
| `capture_sink_factory(core)` | Actual sampler and episode resolver, no second model pass/RNG call; P6 owns closing shared writers |
| Local WS subclass | Original handler/lifecycle/select/health/timing/cleanup; atomic load and terminal ack extensions; immutable generation names |
| Decision cost fields | `cost_ms`, `stage_ms`/`stage_timing_ms`, host-wall scope/kind, bundle/recipe identity, capture/pricing flags; skipped stage0 is actual skipped work |

[fix1_samples.json](../tests/common/artifacts/fix1_samples.json) contains two
actual diagnostic decisions and terminal count3/cursor3 receipts, request
identity, stage/cost fields, wire shape/dtype and byte SHA. Generated sample
YAMLs are [fake_recipe_full.yaml](../common/samples/fake_recipe_full.yaml) and
[fake_plain_k2.yaml](../common/samples/fake_plain_k2.yaml). Register
`fake_provider('r17.fix1.sample')` explicitly before loading them; the teacher
store is the small sealed owned `artifacts/fix1_sample_library/library`.
Both sample decisions intentionally enable capture and are ineligible for
pricing. Store arrays are synthetic; they provide no SR evidence.

### GPU-ready commands, not executed

The bounded π command in §6 remains valid with the migrated local factories.
It now declares the execution protocol, uses complete explicit probe context
and acknowledges actual prefixes while comparing fixed-observation fresh plans.
Its outer **14-minute timeout** includes loading; incomplete/killed runs remain
partial. Run only after the coordinator releases a GPU, one immutable model
and one serial slot. No GPU command was executed in this fix round.

For each new model, the coordinator must use its owner-provided descriptor,
upstream reference and benchmark-specific qualification command (P3/P4/P5
handbacks), in a separate ≤15-minute slot. P1 direct-factory evidence cannot
certify upstream loader parity, GPU WS/batching, native advanced composition,
device timing or model quality. Capture-off pricing is a separate measurement.

### Known limits and pending owners

- **P2** owns formal suite/task namespace and global ids, per-task descent,
  endpoint exact bundle identity, runner/control acknowledgments and native
  physical benchmark semantics. CPU cross composition passes; real renderer
  and real model episodes await coordinator slots.
- **P3/P4/P5** own native GPU/upstream/loader/B2 certificates and accurate
  deployed capability manifests. The listed model tests use CPU fixtures.
- **P6** owns ledger joins/audit/pricing exclusion, store/fit receipts, writer
  closure and full G3 reports. Debug pricing is deliberately ineligible.
- **P1 + model hook owners + Q/P6** still need a certified native executor
  composition for advanced warm/reset, CP2/twins, shadow/routing and dynamic
  serving batch submission. Current generic configuration rejects these;
  direct hook tests are not full G3 support. Q's request is acknowledged, not
  answered by weakening validators. No full-G3/G4 certificate is issued here.
- **Coordinator** owns immutable assets, GPU scheduling, formal baseline/
  library/dev/test freeze and cost/SR experiments. No outcome-based tuning or
  frozen result mutation occurred.

Timings are real **host wall** measurements. Stage3 includes adapter sampling
and any enabled native capture work; no GPU synchronization is added to create
device-latency claims. `capture_overhead_ms` measures core CPU preparation
excluding the sink, while `capture_sink_ms` is in the response. It is not a
counterfactual estimate of all GPU capture overhead. Unavailable overhead is
None, and captured decisions cannot enter pricing. Execution counts come from
client truth; disconnects without an ack remain incomplete rather than guessed.

## Fix round 2

Read REVIEW_A2 and REVIEW_B2 completely, their A2 repros, and incoming requests.
P1's assigned N1/N2/N4 are fixed. This section supersedes the earlier target-IR
shortcut and timing semantics. Zero GPU use, no commit/push, no process control,
no frozen R16/harness/training edits. Four sealed sources remain HEAD-identical.
Other packages' edits and receipts are retained; P1 edited only its owned areas.

### Owned files, changes and interface version

- New [common/ir.py](../common/ir.py): R16 owner-IR formula, bound coefficient
  profile validation, library-only schedule calibration with guard/commit tails.
- New [common/benchmarks.py](../common/benchmarks.py): explicit canonical spec
  registry and digest; real protocols equal their complete BenchmarkSpec identity.
- `common/{config,contracts,provider,recipe,serving,websocket_policy_server}.py`:
  additive recipe/profile/table fields, request attestation, provider admission
  callbacks, immutable verified-store/fit/calibration caches, worker-thread bind,
  and queue/service timing. `fake_adapter.py` registers explicit fake tables and
  attests its zero model-lock wait. `qualification.py` excludes DP K1 from its
  default CP1 qualification profiles, selecting100/10/4/2.
- `tests/common/test_common_fix_round2.py`:34 regression cases;
  `reviewed_defect_witness.py`: scoped red/green mutation checks. Owned fixtures
  and existing round1 tests now use canonical metadata and explicit gap semantics.
- `common/INTERFACE.md`, three sample YAMLs, owned evidence/synthetic store,
  this handback, and REQUESTS/P1_to_P2–P6.md were updated. Complete source SHA
  and change-versus-fix1 list are in [fix2_source_hashes.json](../tests/common/artifacts/fix2_source_hashes.json).

Contracts remain **r17.contracts.v1** with append-only optional
RequestContext.task_table_identity and provider validation callbacks. Recipe and
binding schemas remain **r17.recipe.v1 / r17.adapter_binding.v1**; new identities
are **r17.task_table.v1**, **r17.ir_costs.v1**, **r17.ir_calibration.v1**.
Legacy/default bindings omit lossy-study fields; all hooks are opt-in.

### Finding → change → regression → status

Tests below are in `test_common_fix_round2.py` unless otherwise specified.

| Finding | Change | Passing test/evidence | Status |
|---|---|---|---|
| A2 N1 major | Remove `(1-target)/target`. Require model/MISS-sampler/protocol-bound normalized vision/MISS coefficients. Replay complete library episodes with LOEO donors, native periodic coins, guard and actual commit tails; solve R11 bracket0–40,26 iterations, setting rounded4 decimals. Record predicted IR/N/V/M and sealed calibration identity. Reject missing profiles and infeasible targets. | Six `test_n1_reproduces_r16_own_example_configs` cases call R16's own synthetic builder and actual analyzer; eight `test_n1_matches_r11_library_periodic_calibration` cases match exact R11 solve for π/GR IR45/40/32/25. IR25 reviewer witness, tail/guard replay, bad profile and prediction receipt tests pass. | Fixed at P1; no real-model pricing/quality claim |
| A2 N2 major | Require exact complete canonical ordered_tasks and task_table_identity from registered benchmark spec. Reject reordered/subset/stale/foreign tables before store access and before bundle publication/bind. Validate supplied request/episode table identity and task_name/task_id before model work. Disabled recipes may retain validated inert table metadata. | Four malformed-table cases; actual MT50 canonical table/version identity; enabled/plain mismatched name/index cases; attested request mismatch; stdlib-only import. Actual P2 π WS/conductor/worker cross tests pass. | Fixed; P5 recipe fixture migration pending |
| A2 N4 minor | Store checksums once per resolved path/content generation. Cache immutable verified Store, fitted arrays and calibration; metadata changes reject. Check metadata before/after initial checksum pass, including symlink targets. Prepare load and connection stacks in workers; stock handler commits prepared binding on event loop. | Single SHA pass across3 connections/factories; changed-generation and verification-race rejection; linked READY target mutation; event-loop worker barrier. Unchanged reviewer script reports each of3 binds hashes0 files/0 bytes. | Fixed; large real-store throughput unmeasured |
| A2 N5 shared follow-up | Consume adapter.drain_queue_wait_ms before infer/after each measured operation. Publish queue/service/stage service fields, explicit host scope and pricing_time_field. Absent instrumentation yields None and pricing-ineligible; GPU time remains None. | `test_n5_queue_wait_service_fields_and_missing_instrumentation`; P2 actual held-lock/client ledger witness in passing cross suite. | Shared hook fixed; P5 native GR instrumentation pending |
| B2 N6 common follow-up | Add provider binding_validator/config_validator, enforced by common binding/config/storage/load/bind paths. Strict optional allow_lossy_sampler plus nonempty sampler_study_label; expose study status in decision/capture. DP default CP1 probes omit K1. P4 owns role/teacher admission. | `test_b2_n6_provider_admission_hooks_apply_to_common_factory_and_capture`; actual P4 admission/teacher/entrypoint cases in114-pass suite. | Shared hooks fixed; P4 consumes |

R16 authorities are [cost.py:13](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/cost.py:13)
(full-reference coefficients and native K reduction),
[analyze.py:184](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/analyze.py:184)
(full-stage ledger IR), and [analyze.py:192](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/analyze.py:192)
(actual analyzer's coefficient-based episode pricing). R11's library-only knob
solve is [calibration.py:89](/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/calibration.py:89)
and [calibration.py:138](/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/calibration.py:138).
Formula: **owner IR=(vision·V+miss(K)·M)/N**. R16 intentionally preserves its
old rounded full-reference coefficients; regression uses its actual pricing
function, including GR's existing K8 reference, not R17 GR K4 measurements.

[Scoped mutation receipt](../tests/common/artifacts/fix2_reviewed_mutations/receipt.json)
records three reviewed mechanisms reintroduced **in memory only**: old gap
shortcut, missing canonical check, and always reopening/checksumming Store.
Each breaks its new regression; restoring current code passes the same three
checks with fresh assets. This is3 red/3 green, **not** a historical checkout
replay. Historical reviewer repro outputs remain in the review directories.
The low-level array-only LayoutRecipe API retains caller-defined numeric task
indices; serving safety is enforced at the recipe load/select/query boundary.

### CPU/default-off receipts and commands actually run

[fix2_cpu_receipt.json](../tests/common/artifacts/fix2_cpu_receipt.json) records
exact commands, exit codes, XML hashes, pass/fail/skip counts and CPU environment.
All tests used CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS=cpu, no pyc/cacheprovider,
and OMP/BLAS/MKL threads1. All scratch/XML is under tests/common/artifacts.

| Run / actual targets | Exit | Actual result | Receipt |
|---|---:|---|---|
| Full P1 `tests/common` | 0 |109 passed,0 skipped | fix2_g0_final.xml |
| Round2 regressions | 0 |34 passed,0 skipped | fix2_regressions_final.xml |
| P1/P2/P6 `tests/{common,benchmarks,conductor,models/pi05,debug,search,ops}` | 0 |346 passed,0 skipped | fix2_cross_final.xml |
| Fast-WAM `tests/models/fastwam` | 0 |104 passed,0 skipped | fix2_fastwam_final.xml |
| DP `tests/models/dp`, DP CPU island | 0 |114 passed,0 skipped | fix2_dp.xml |
| GR `tests/models/groot` | 1 |54 passed,7 failed,0 skipped | fix2_groot_final.xml; pending P5 canonical fixture migration |
| Legacy config/storage/schedule, serving identity/recorder/router, MW runner/env, conductor, GR/LIBERO, R16 conductor/server tests and default-off | 0 |1032 passed,10 skipped,1 EGL deselected | fix2_legacy.xml |
| Explicit env_dependent localhost proxy sockets | 0 |6 passed,0 skipped,23 deselected | fix2_proxy.xml |
| Fake framework CLI qualifier | 0 |status=passed; diagnostic only | fix2_fake_qualification.json |
| Scoped reviewed-mechanism mutation witness | 0 |3 expected red,3 restored green | fix2_reviewed_mutations/receipt.json |
| R16 analyze verify, repeated | 0 |DESIGN_VERIFIED; no outcome reads | fix2_seal_receipt.json |
| Unchanged reviewer bind/hash repro | 0 |3 binds,each0 files/0 bytes hashed | fix2_bind_repro.json |

Earlier failed attempts are retained, including a corrected common/conductor
pytest basename collision, P2 request field handoff, P3 raw-loader source-path
bootstrap and worker payload migration. No failed receipt is presented as a
passing release. The latest broad cross and Fast/DP results above supersede
those attempts; the7 GR failures remain open with P5.

Run P1 directly with the standard CPU prefix:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  exp/r17_integration/tests/common -p no:cacheprovider \
  --basetemp=exp/r17_integration/tests/common/artifacts/fix2_g0_final_scratch \
  --junitxml=exp/r17_integration/tests/common/artifacts/fix2_g0_final.xml -q
```

Use fresh owned scratch/XML names for another run. Exact legacy targets and
all other invocations are in the CPU receipt. No original stock source bytes
changed; the two subprocess HEAD/default-off certificates passed both alone
and in the full/cross/legacy runs. [Seal receipt](../tests/common/artifacts/fix2_seal_receipt.json)
records HEAD-equal SHA256 for each of the4 sealed files and actual verify output.

### Published hooks and samples

| Hook | Tested contract |
|---|---|
| register_benchmark/spec registry | CPU-safe, complete immutable task table; real key equals spec identity; unknown table rejects |
| validate_recipe / make_bundle_validator | Canonical table/profile/store/fit/teacher verification before atomic publication |
| bound_costs / owner_ir / calibration | Native MISS-sampler-bound coefficients; weighted counts; library-only prediction receipt |
| verified_store / make_recipe_factory | One checksum pass per immutable generation, shared immutable fits and fresh guard/memo |
| WS socket facade | Worker preparation followed by unchanged stock select/lifecycle commit; no per-bind array hashing |
| drain_queue_wait_ms | Per-operation model-lock wait; host service=wall−wait; missing instrumentation unsupported |
| Provider admission callbacks | Model-specific lossiness/role policy enforced even outside model's local factory |

[fake_recipe_full.yaml](../common/samples/fake_recipe_full.yaml) now declares a
bound diagnostic cost profile and canonical table; its tiny12-row cohort has
target.45/prediction.40833, explicitly recorded rather than claimed equal.
[fake_plain_k2.yaml](../common/samples/fake_plain_k2.yaml) carries inert canonical
metadata. Register fake_provider('r17.fix1.sample') for these examples.
[fix2_samples.json](../tests/common/artifacts/fix2_samples.json) contains actual
decision/capture/terminal count3 receipts with request and bundle identities.

[fake_recipe_owner_ir25.yaml](../common/samples/fake_recipe_owner_ir25.yaml) uses
the owned synthetic10×32-decision store. Register
fake_provider('r17.fix2.ir25.sample'). Its [calibration receipt](../tests/common/artifacts/fix2_ir25_calibration.json)
records target.25, prediction.2509375, gap7.0777, N320/V320/M38. Coefficients
.15/.85 are declared **diagnostic**, not measured Fake/model pricing. Finite
library quantization and unseen trajectories can differ from the target.

### GPU-ready commands and pending owners

The ≤14-minute π command in §6 remains available to the coordinator only,
with explicit immutable descriptor/fixed observations and a fresh output path.
P3/P4/P5 handbacks provide their own native/model-specific ≤15-minute serial
commands. **None was executed in this round.** DP's generic CP1 qualifier now
uses100/10/4/2; K1 is only a labelled plain study subject to P4 admission.
No GPU/service-price/SR/native advanced-composition certificate is issued.

- **P5:** migrate7 owned GR recipe fixtures/YAMLs to canonical table identity
  and new target-IR receipt semantics; instrument native shared model-lock wait.
  Until then GR recipe fixture validation fails closed and uninstrumented GR
  service pricing is unsupported. Requests and failing XML were handed back.
- **P2:** bind ir45/40/32/25 descent labels to the matching target/calibration
  identity. Existing canonical runner/manifest/endpoint and queue/client timing
  composition pass in the broad suite. Real renderer episodes remain unverified.
- **P6/coordinator:** verify real model/sampler/protocol cost authorities before
  supplying ir_costs; report actual owner IR from N/V/M. source_identity names
  a caller-verified authority; common checks identity/schema/numbers, not its
  scientific provenance. Never auto-transfer R16 LIBERO costs to MW/new grids.
- **P1/model owners/Q/P6:** advanced native warm/reset/CP2/twins/shadow/routing,
  automatic serving batching and full G3 qualification remain pending as before.
- **Coordinator:** release immutable checkpoints, schedule serial GPU slots,
  formal teacher/dev/test/SR studies and large-store performance measurements.

No assigned P1 finding was disputed. The remaining GR cross-package work is
explicitly tracked; P1 did not weaken canonical or lossiness gates to hide it.
