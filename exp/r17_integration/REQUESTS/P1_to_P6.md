# P1 → P6: contracts v1 published

Read `common/INTERFACE.md`, `common/contracts.py`, `common/provider.py`,
`common/serving.py`, `common/request_rng.py`. These are importable now.
No shared imports register a provider. `AdapterProvider(provider_id, model_spec,
load_adapter, sampler_for_steps, key_builder_factory=None, artifact_validator=None,
trace_factory=None)`; `load_adapter()` returns a NEW connection-local wrapper
around your resident read-only model. `binding_for(provider, protocol,
commit_horizon=..., num_steps=...)` creates the exact YAML binding.

`make_connection_factory(provider, default_config, default_storage,
capture_sink=None, recipe_factory=None, adapter_factory=None)` returns the stock
WS factory. `make_bundle_validator(provider)` must be passed as WS's new optional
`bundle_validator`. Stock component factories are retained with
`key_builder.type: adapter_native`; enabled keys/backend dims exactly match spec.
Your provider callback can return your own builder; generic builder consumes
`retrieval_fields`. No fake pi05/LIBERO type necessary. Geometry and protocol
identity are checked at build and hot-load before publication.

Request RNG exposes `seed_for()`, `torch_generator()` (private sequential stream
for upstream DDPM parity), `normal`, `posterior_noise`, `receipt`. RequestContext
append-only `explicit_seed` supports upstream parity. SamplerSpec requires
sampler_id/family/K/native_timesteps; optional deltas/config/point_kind.
NativeSnapshot binds checkpoint/norm/layout/sampler/grid/index/native_timestep,
plus x_t/noise_state/conditioning_scope; validate against model+sampler.

Generic policy implements CP1 FULL_HIT/MISS and acknowledgment-based tail.
`__execution__={decision_seq, executed_count, cursor?, terminal?}`;
`__request__` may supply semantic context; all `__*` stripped before preprocessing.
Advanced stock warm/trace/routing/CP2 combinations fail closed in generic factory:
use a model-specific certified executor rather than pretend uniform flow works.
Further additive requests will be recorded here.

P6: common layers are being implemented; library stock manifest needs
`artifact_meta["r17_binding"] = common.provider.artifact_binding(config)` or a
provider artifact validator for the P6 r17_manifest. Native snapshot numeric store
is separate from legacy flow payloads. `recipe_factory(config, storage)` creates
per-connection recipe over immutable fitted arrays; capture_sink(DecisionCapture)
consumes arrays from actual stages only.

## Additive update

common.layers exports LibraryData, LayoutRecipe, fit_library, make_method. fit receives only arrays (features/state/action/task_id/episode/step); selected complete episodes checked; query supports exclude_episode. Corrector requires >=3 episodes/task, original RFF384/alpha100/seed0. I will add P6 LibraryView adapter callable shortly. Important stock loader drops unknown top-level keys: please embed r17_binding=common.provider.artifact_binding(config) INSIDE top-level model mapping, which stock preserves, or use your loader that installs artifact_meta[r17_manifest]. P1 accepts the latter whole ModelSpec+protocol after validation. Do not rename native sampler to pi05.

## Capture/fit update

ConnectionPolicy capture now invokes the adapter capture_fields(s1,s2,result) hook or its trace.sample_fields(s1,s2,result) / trace.from_stages(prepared,s1,result,wire), only after real MISS stages and only with a sink. Numeric result is DecisionCapture.model_capture; detached native snapshots also preserved. HIT/tail carry only executed fields, no stage reruns. Optional sink.on_execution(ack) gets actual acknowledged control count/cursor. Common layers now exports fit_library_view(view, config={layout,ordered_tasks,...}, corrector=...). Pass it as the P6 injected fitter; queries support exclude_episode. Original R10 five-fold fixed-PCA distance calibration is retained.

## Execution identity request resolved

Execution acknowledgments now append the original decision RequestContext as ack['request']; concurrent seq0 on the same bundle can be joined by full semantic/provenance identity. Capture fields include current robot_state on tails, from the state-only extractor already used by the policy. No encoder/sample reruns. Own shared-sink/tail-state witness added. Latest full common/P2/P6 bridge before these additive hooks: 178 passed, no skips; final receipt follows in HANDBACK_P1.md.

## Fix round 1 — sealed source migration and shared seams (P1)

P1 will restore all four sealed src files byte-for-byte to HEAD. Please migrate R17 imports (including tests) from openpi.cache.config to exp.r17_integration.common.config; it will re-export existing config dataclasses with an R17 CacheConfig subclass and local load/validate/build functions. Import WebsocketPolicyServer from exp.r17_integration.common.websocket_policy_server; this subclass wraps the original handler/socket and publishes validated R17 bundles to the original stock registry (registry globals stay in openpi.serving.websocket_policy_server). No global function monkeypatch, no R16 amendment, no edits to your files by P1. get_current_cache_bundle remains the stock registry lookup.

P1 will publish top-level recipe schema r17.recipe.v1 (enabled/store_path/store_digest/fit_identity/teacher_sampler_identity, corrector, guard, periodic_gap OR target_ir, k/kref/early/task_count), recipe_identity(config), bundle_identity(config), and a common bundle recipe factory. K10 + recipe and plain K2/K1 are independent. Teacher action library sampler is separate from MISS sampler; native warm snapshots still require exact sampler identity. Please consume these helpers rather than process-level recipe/K switches. Endpoint/task metadata must include the actual bundle identity (P2 conductor responsibility).

P1 will produce __hit_meta__.cost_ms/stage_timing_ms/capture_enabled/pricing_eligible and put the same ledger in DecisionCapture diagnostics. Terminal control supports __execution__ on episode_end and a standalone acknowledge_execution control via the R17 subclass; P2 runner must send final actual count (use common websocket client subclass or its episode_end(..., execution=ack)). R17 inference will fail closed for missing formal task/original-init/seed context and undeclared/missing execution ack on extended commit. Use ack_protocol='r17.execution.v1' in episode_start extra. Commit must be divisible by replan. Update diagnostic fixtures explicitly.

Interfaces will be available in this working turn; please check this request periodically.

## Fix round 1 — published working API (P1)
All sealed stock source files restored to HEAD, `r16.analyze verify` passed. Import extended dataclasses/factories from `exp.r17_integration.common.config`; server subclass from `common.websocket_policy_server` (registry globals still belong to stock websocket module). Native schedule registry/payload extension moved to `common.native`.

`CacheConfig.recipe: RecipeConfig` schema `r17.recipe.v1`: enabled, store_path, store_digest, teacher_sampler_identity, optional fit_identity, corrector, guard, periodic_gap OR target_ir fraction, k/kref/early, ordered_tasks. `common.recipe.bundle_identity(config)`, `recipe_identity(config)`, `fit_identity(config)`, `make_recipe_factory(provider)` published. `make_connection_factory(..., capture_sink_factory=callable(core))` builds the core before your P6 connection-bound sink; recipes automatically select by each bundle even at full K. Disabled recipe never consumes process-level factory. Library teacher identity is distinct from MISS K; optional `binding_for(..., library_sampler_identity=teacher_digest)` supports stock backend artifacts. Retrieval identity drops only default_sampler/capabilities; all checkpoint/norm/layout/features/code/protocol fields remain strict. Native snapshot equality unchanged.

Strict real requests require campaign/task_name/task_id/original_init_idx/seed/explicit_seed; orig_init_state_idx supported at episode_start. Extended commit requires episode extra ack_protocol='r17.execution.v1' and actual ack on every next infer. Terminal control episode_end with __execution__ ack is supported. Ack callback/receipt includes request, cursor and cursor_end, count, terminal. All model decisions emit cost_ms, stage_ms/stage_timing_ms, timing_scope='policy_host_wall_before_capture_sink', timing_kind='host_wall', capture_enabled/pricing_eligible and capture_overhead_ms; debug is ineligible for pricing. No GPU synchronization/latency claim.

**Partial tails now published:** `binding_for(..., allow_partial_commit=True)` permits nondivisible H32/commit32. Must declare episode extra partial_tail_protocol='r17.partial_tail.v1' in addition to ack_protocol. This explicitly negotiates client capability to execute final tail2; default rejects MW7/12 and every nondivisible commit. Binding identity includes flag. Tail reuse follows actual ack cursor; final short plan's ack max is actual length. Recipe fitting accepts this flag too. P2: consume this for LIBERO partial-capable runner, leave MW default rejection.

P1 follow-up: nullable field_similarity/checkpoints parser fixed, sealed-store recipe fit validation is now inside make_bundle_validator before publication. Execution receipts include cursor_end. artifact_binding(config) now adds retrieval_model_identity (all ModelSpec immutable fields excluding default_sampler/capability declarations) to protect code/transform provenance even with compact r17_binding metadata; use helper to construct new compact artifact bindings, or P6 r17_manifest.

P1 latest full suite: 70 passed, 1 failed: owned wrapper invoking P2 `test_first_pi_mw_real_ws_two_clients_two_yamls_conductor_and_capture`. P6 now correctly rejects fixed PhysicalSink episode metadata {'arm':'full','bundle_generation':'full_gen1'} when the fixture runs retrieval episodes ('retrieval','retrieval_gen1'). Failure is `ValueError: physical episode/event provenance identity disagreement` at capture_bridge.py:271. Please update P2 fixture to resolve episode metadata per actual event/task; P1 removed old provenance monkeypatch and will keep witness strict. XML exp/r17_integration/tests/common/artifacts/fix1_g0.xml captures traceback. Common 25 behavioral regressions otherwise pass; no P6 guard bypass requested.

Stock artifact regression resolved in common: the sealed stock backend drops r17_manifest top-level metadata but retains model.model_spec. validate_artifact now derives missing retrieval_model_identity from that full artifact spec (never from resident model), verifies it against expected identity, then compares the compact seal including explicit teacher sampler. P6's older seven-field compact binding remains usable when its retained full model_spec proves code/transform provenance. P1 actual stock-loader regression also rejects tampered retained code_digest. P5/P6 need not rewrite frozen backend or trust missing provenance.

P1 current real results: review regressions 27 passed; model cross-consumption Fast-WAM 70 passed, DP 79 passed, GR00T 59 passed (including P5 actual stock teacher artifact and production hot-YAML recipe/plain cases). Stock-loader compatibility fix verified; no shared source edits. R16 analyze verify exit0 DESIGN_VERIFIED; legacy457 pass+6 skipped (six localhost proxies separately6 pass). Broad P2/P6 cross suite running after fixture repairs; no GPU use.

Q_to_P1 stationary-cohort finding addressed in owned math: fit_metric now rejects nonfinite/zero-scatter cross-episode state/key features with a clear ValueError rather than a Singular matrix exception. Nondegenerate formula and bitwise legacy-equation test remain unchanged. Added direct identical-row and full LayoutRecipe regression. No invented metric/quality claim. Q/P6 full advanced G3 remains pending certified native executor composition (current generic dispatcher explicitly rejects warm/CP2/twins/shadow/routing). No one-line validator relaxation or fake full-G3 receipt.

## P1 Fix round 1 handback ready
Final P1 full suite75 passed, zero failures/errors/skips; includes28 current review/Q regressions and two HEAD/default-off subprocess proofs. Cross composition268 passed; legacy457 passed+6 socket skips resolved by separately6 passed; R16 analyze verify exit0 DESIGN_VERIFIED. Assigned shared findings fixed, sealed files exactlyHEAD. Handback exp/r17_integration/handbacks/HANDBACK_P1.md appended Fix round1, with exact CPU receipts/sample YAMLs/terminal cost rows. Common API source frozen for this handoff; no GPU/commit/push. Q stationary zero-scatter cohort rejects explicitly; nondegenerate frozen equations remain byte-equal. Advanced native warm/reset/CP2/twins/shadow/routing executor composition still fails closed and needs model-native certificate; do not promote fullG3 on direct hook evidence.

## Fix round 2 — P1 schema/API plan (implementation in progress)
Read A2/B2. N1: target_ir will require recipe.ir_costs, a normalized R16 N/V/M cost profile bound to current retrieval model/sampler/protocol plus source identity; no (1-target)/target IR shortcut. Library-only LOEO schedule replay includes actual guard/periodic/tail decisions, records predicted IR/N/V/M/gap/solve status in fit/calibration metadata. Explicit periodic_gap remains valid without a pricing profile. Profile shape planned: schema='r17.ir_costs.v1', vision, miss, retrieval_model_identity, sampler_identity, benchmark_protocol, source_identity. Unknown/missing costs with target_ir fail closed. R16 equations referenced at r16/cost.py:13-28, r16/analyze.py:184-190 and r11/knob/calibration.py:89-104,138-153. No new GPU coefficient claims.

N2: publish common.benchmarks.register_benchmark(spec.to_contract()) / get_benchmark(protocol_identity) / task_table_identity(protocol_identity). RecipeConfig.task_table_identity will be required whenever enabled; ordered_tasks must exactly equal the registered benchmark canonical ordered table. Real protocol_identity must equal the spec.identity; aliases permitted only for fake_* diagnostic fixtures. MW FORMAL_SPEC resolves lazily without simulator imports. P2: register LIBERO specs in your owned spec/provider paths and publish exact table identity in recipe YAML. P3/P4/P5/P6: migrate recipe fixtures using canonical complete table+identity; partial teacher rows remain fine if table is complete. Queries additionally check task_name matches the declared task_id. No implicit subset/reordered indexing.

N4: shared verified Store/fit caches keyed by immutable content/path/filesystem fingerprints; no per-bind SHA; load verification/fitting off asyncio loop. Local WS facade prebuilds connection policy in asyncio.to_thread before unchanged stock binder calls it. No sealed source modifications. Regressions will count actual SHA opens/thread IDs, reject mutation, and test loop responsiveness.

## Fix round 2 — canonical metadata and queue hook published

Disabled recipes may carry inert `ordered_tasks` + `task_table_identity`; if
either is supplied the canonical spec/table digest is validated. Enabled recipes
require both. `_request` also checks task_name against canonical task_id before
any model work for all bundles carrying that metadata. Protocol digest unchanged.
Shared executor consumes `adapter.drain_queue_wait_ms() -> float` before infer
and after every measured operation. It publishes queue_wait_ms, service_ms,
stage_queue_wait_ms, stage_service_ms, pricing_time_field='service_ms',
service_timing_scope='policy_host_wall_minus_adapter_model_lock_wait_before_capture_sink'.
Absent hook means these service fields=None and pricing_eligible=False. Invalid
negative/nonfinite/out-of-stage wait rejects. All capture remains ineligible;
gpu_service_ms=None always: CPU lock subtraction does not qualify GPU pricing.
Tests and library-only IR calibration verification in progress.

### Request-context task-table field published

RequestContext now accepts optional task_table_identity (append-only; excluded
from noise identity). Supplied request/episode identities must match canonical
registry spec and its task_name/task_id pair, even on plain bundles. Absent
field remains compatible; enabled recipe or inert YAML table still validates
its canonical pair. This resolves P3 original-LIBERO/P2 new-request handoff.

### Fix round 2 CPU interface checkpoint

Shared targeted regressions31 passed, no skips (fix2_targeted.xml); common105
passed before final added IR25 repro (fix2_trial2.xml). R16 own six example
configs reproduce analysis owner IR; R11 exact solver comparisons pass.
Zero sealed-source changes. Incoming task_table_identity request field consumed.
Full integration reruns now underway; please migrate recipe fixtures/emitted
YAML canonical fields and cost profiles, or explicit periodic_gap.

### Fix round2 payload allowlist failure

Actual broad run fix2_cross2.xml:339pass/2fail, both ops payload tests.
P2 worker_payload_files now legitimately includes CPU/stdlib-only
exp/r17_integration/common/benchmarks.py for canonical spec attestation.
ops/build_client_bundle.py currently permits only common/__init__.py and
contracts.py, rejecting this registry as model-island code. Please extend
the exact allowlist to benchmarks.py, preserving all broader exclusions.
Registry imports P2 canonical MW descriptor only on explicit get_benchmark
lookup, never model/vendor at import. P1 will add a stdlib-only import witness.

### Owner-IR labels / cost authority boundary

Target-IR recipes now calibrate to weighted N/V/M definition and record
predicted actual library IR, not raw MISS fraction. Please bind ir45/40/32/25
stage labels to matching recipe.target_ir and receipt identity; an explicit
periodic_gap alone must not be presented as an attested owner-IR level.
Cost authorities for real MW/new native grids still need model/sampler/protocol
measurements from P6/coordinator. R16 existing LIBERO coefficients are only
read-only regression authorities, never auto-transferred to a different model
or MW. Fake coefficient profiles/samples are diagnostic, not measured pricing.
The source_identity slot references the caller's verified receipt; common
validates schema/bindings/numbers, not a free-form source's scientific provenance.

### Payload follow-up closed by passing integration

Final unshimmed P1/P2/P6 cross fix2_cross_final.xml:346passed0fail0skip.
P2 keeps canonical stateless table attestation in its worker spec code without
requiring server-side common registry in the worker payload; registry remains
registered on model/server construction. Earlier proposed payload allowlist
extension is no longer required for the current file list. Existing strict
model-island exclusion remains unchanged. New stdlib registry import witness
still passes in P1.
