# P6 bridge seams (v1, 2026-10-09)

P6 consumes, never recreates, P1 layer math. Contracts are absent at start.
Please publish a library-only fitting callable accepting a read-only library
view and `exclude_episode` for every training row, plus a method factory exposing
`fit(library)`, `reset(episode)`, `query(decision)`; query results should contain
normalized/wire actions, donor IDs/weights, guard, knob, stage calls and cost.
P6 `search.fit.fit_library(..., fitter=...)` and `search.replay.replay(...)`
are dependency-injected shims until this callable is published.

P6 storage binds the entire JSON-compatible ModelSpec (dataclasses accepted),
benchmark protocol, feature/layout/norm/sampler content and source hashes.
`search.build_library.stock_artifact()` produces stock CacheEntry/CachePayload
objects and append-only `r17_manifest`. Native snapshots/RNG stay in the
versioned store; please bind that manifest before enabling native warm resume.
No native DDPM/shifted-flow points are converted to legacy uniform flow.

For debug, call `CaptureBridge.capture(record, arrays, rawkeys=...)` with already
computed arrays. Bridge is opt-in and does no forward/RNG calls. A trace sink
can be passed and consumes the model owner's existing StepTrace. Shared
BlockWriter/NPZ/transport are reused, no shared patches requested yet.

Pending: cross-package live/resident/WS equivalence needs P1 method factory and
capture callback wired by P1. P6 tests fakes and stock backend independently.

## Additive execution acknowledgment request

`capture_sink.on_execution(ack)` currently has seq/count/cursor/bundle only.
Please append the associated `RequestContext` as `ack['request']` (dataclass
or mapping), preserving campaign/arm/attempt/init/generation from that decision.
One shared sink with concurrent clients on the SAME bundle otherwise cannot
join seq0 unambiguously. P6 joins ack logs by complete decision identity;
local shim rejects ambiguous ack instead of charging zero cost/execution.
`DecisionCapture.model_capture` and detached native_snapshots now consumed.
Stock artifacts embed r17_binding INSIDE top-level model mapping as requested.

Local shim now resolves the core use case: P6 CaptureFactory calls the published
P1 factory with a NEW DecisionSink per connection, while sharing BlockWriter by
complete model/protocol/actual-sampler binding. No monkeypatch or shared edit.
The request-context callback remains useful for callers deliberately using one
sink object globally, but is no longer a blocker for P6's factory path.

Resolved: P1 now appends `request` to on_execution; P6 accepts dataclass or dict.
The local factory remains useful to share writers by actual sampler binding.
# Fix round 1 — REVIEW_A #6/#7/#10; REVIEW_B B9

B11 cross-scope handoff: P4 reports qualification/model_runtime.py still creates
wrist image[::-1] instead of consuming true paired frames. qualification/ is
outside P6 ownership. Please forward to its coordinator/Q owner to use
models.dp.fixtures.libero_observation(frame=...) or sealed real paired NPZ.
B9 all debug/capture rows now excluded from P6 price totals and normalization,
including captured/unknown full reference receipts. Separate capture-off GPU
pricing is still needed for EVERY arm. No P6 synthetic wrist qualification claim.

Published: CoreCaptureFactory / make_capture_sink_factory(provider, directory),
returning callable(core) with .bridges/.close(); directory-only compatibility
also supported. Existing CaptureFactory now defaults to core._episode resolver
when none supplied. Measured bridge preparation host wall time and queue wait
remain diagnostic; bridge always marks capture_enabled=True, pricing_eligible=False.
Readers consume cost_ms/stage_ms/stage_timing_ms and server_timing.infer_ms;
actual timing_scope and total capture_overhead_ms should come from the core.
Migrating P6 config/server imports as your sealed-source request specifies.

Migration repro found: actual R17 WS load_cache_config with
yaml.safe_dump(dataclasses.asdict(CacheConfig(...))) returns error
"'NoneType' object has no attribute 'values'". common.config._strict_fields
does value.values() for field_similarity even when default value=None.
P6 regression test_framework_bridge::test_real_websocket_two_clients_debug_parity_and_capture
uses a complete YAML dataclass document and now reports the actual error.
Please treat nullable field_similarity/checkpoints mappings according to their
declared optional type instead of calling .values() on None. No P6 shared edits.

P6 is adding exact-identity cost/execution ledger merging, terminal-client-ack
recovery, tail auditing, and pricing exclusion for capture-contaminated rows.
Please publish measured `cost_ms`, `stage_ms` (actual executed stages),
`timing_scope`, `capture_enabled`/`pricing_eligible` and `capture_overhead_ms`
when measurable in DecisionCapture.diagnostics and `__hit_meta__`. Wall CPU
timings must be labelled host wall, not GPU stage latency; no GPU sync by P6.
Any native capture or sink enabled must set pricing_eligible=False (B9), including
capture work before sink invocation. Do not infer zero when timing unavailable.
Please accept terminal `__execution__` on episode_end and preserve request
identity in execution callbacks. P6 will consume explicit fields without reruns.

P6 will add `make_capture_sink_factory(provider, directory)` in its own
debug/capture_bridge.py: callable(core) creates connection-local DecisionSink
from core._episode and actual core.sampler, sharing writers by full binding.
Close callback exposes all writer receipts. Model serve entries may consume
this helper; P6 will not edit common or model serving files.

## Q BF16 observer boundary — upstream fix implemented

Consumed Q_to_P6's production BF16 serialization repro. P6 numeric_arrays now
copies BF16 observer values as float32 and stores exact uint16 companions named
`original_name__r17_bfloat16_bits`. Capture records carry `array_serialization`
by arrays/rawkeys namespace, including original name, native/storage/bit dtype,
companion name and encoding. Reserved companion collisions reject explicitly.
DecisionSink also stores nested native snapshot RNG tensors as numeric datasets
with `r17_array` references; scalar noise metadata is retained. Inference,
native-resume objects and RNG are untouched. Normal NumPy paths have no torch
import and no extra datasets/metadata. Q's local shim remains compatible but is
no longer required for P-owned sinks receiving original BF16 products.

Five CPU regressions: all65,536 BF16 bit patterns including NaN/Inf/signed zero,
noncontiguous tensors, collision rejection, closed NPZ/H5 snapshot+key+noise
roundtrip, immutable inference tensors and debug-on/off action/RNG bytes. Old
P6 ran4 failed/1 passed; upstream implementation ran5 passed. Receipts live in
tests/ops/receipts/p6_fix1_bf16_{red,green}.xml. Final full/legacy/interop reruns
are recorded in HANDBACK_P6.md. Please forward this upstream adoption to Q and
model owners; no GPU/BF16 production latency or native-resume certificate claimed.

## LIBERO teacher library — indexed views and 500 B LOEO serving

Owner now requests DP/Fast/Cosmos B500 teacher collections using the exact
π/GR B pool and R10 episode IDs/permutations. P6 implements a local B-pool plan
shim: preserve the sealed native LIBERO protocol/task table; explicit task
init_pool=B, partition/split=library and task_data_hash guard the different state
content (A and B both use original indices0..49). This is unrelated to the MW
index-partition rule; no shared partition semantics or files are changed.

Published search.teacher_library.RowIndexStore: subset VIEW.json + int64 row
indices only; .rows/.fields/.array expose the selected500-parent rows. New
fits use your existing LayoutRecipe math and P6 save_fitted_recipe. Please
consume this indexed-store schema in the shared recipe store loader when those
fitted libraries are admitted for serving; a fit may never read the other parent
rows. Parent immutable SHA + view SHA are sealed in fit receipts.

At500 every B init is in the library. As R15 requires, a B query must exclude
the complete own episode (suite-local task*50+B-init); A queries do not own a B
episode. P6 preserves these numeric IDs in LibraryData instead of remapping
hashes. Current LayoutRecipe.query(exclude_episode=...) returns raw retrieval
before corrector/guard/knob; P6 documents that limit rather than claiming full
five-layer LOEO serving. Please publish an opt-in composed LOEO serving hook
before future500 B policy qualification. No A arrays/query/outcome reads enter
these fits, and no GPU collection is executed by P6 now.

## LIBERO library follow-up — Cosmos bundle landed, minimal teacher capture pending

Read sol D's HANDBACK_COSMOS.md after it landed during implementation. The
published K5 bundle still declares debug_capture unsupported (and its native
capture path rejects it). Please forward to D: publish the minimal hook needed
for this owner-authorized collection: CaptureSpec(enabled=True,snapshots=False),
actual FeatureSpec retrieval fields, 9-D robot_state, and the native normalized
16x7 teacher chunk before to_wire; no future decodes/snapshots/resume required.
P6 captures both exact uint8 R17 adapter-input frames at the formal server,
filters extras before the shared writer, and performs no extra model forwards.
The new generic path preserves Cosmos seed195/environment0/replan16 protocol.
No capability guard is bypassed; Cosmos library launch remains CPU-seal blocked
until D implements/tests this producer and declares debug_capture supported.

Knob products now call common.ir.library_schedule + calibrate_periodic on each
selected row-index view and its sealed fit. The default emits explicit pending
cost receipts for IR25/32/40/45; --ir-costs plus an optional sealed --miss-sampler
consumes a matching measured new-model profile. No π/GR coefficients or wall
latency are imported. Base guard/corrector caps remain your unchanged defaults.
Row-index serving/whole-recipe B500 LOEO still needs the previously requested
P1 interface adoption, before admitting these libraries for serving.

Cosmos producer detail: common.serving also instantiates make_trace_adapter()
on the captured MISS even when snapshots=False. D must provide a minimal
CPU-detached capture adapter (or the published provider trace_factory), alongside
allowing capture in run_stage2/sample. Its current methods reject all three.
Normalized sample() actions are currently float32; the float64 physical inverse
is deliberately unnecessary for fitting and will not be written by this profile.

## Fast-WAM library G2 A/B digest binding — forward to Q/coordinator

Owner clarified that the library admission gate is Q's G2 A/B framework slots,
not P3 raw-Wan/released loader parity. P6 has replaced the mandatory P3 gate
with --g2-a-receipt / --g2-b-receipt and kept P3 parity as an optional extra.
Both supplied coordinator receipts PASS, but their nested r17.Q.slot.v1 PASS
records contain only checkpoint=/data/mw_finetune/models/fastwam_libero:

- slot_G2_fastwam_libero_A_20261009_234134_2639853/receipt.json
- slot_G2_fastwam_libero_B_20261009_234224_2645346/receipt.json

Both under /data/mw_finetune/qualification/runs on weilandserver. Missing in
each PASS record: model_manifest.spec.checkpoint_identity, code_digest,
norm_identity, and model_manifest.content_digests with weight/code/norm SHA256s.
The new gate consumes the published ModelManifest v1 schema inline in
remote_receipt.model_manifest (or model_manifest for a native Q receipt),
checks checkpoint_identity=canonical SHA256(content_digests), checks the code
submap hash, then compares all asset pins and model geometry/native sampler
to the fresh formal CPU teacher seal. No fallback to today's mutable assets.

Read-only H100 inspection found historical evidence, so this need not require
another GPU run if Q can certify the association to its recorded owned run:
A's native.json has the full ModelManifest; B's SERVER_READY.json records the
actual owned server model spec (pid80420). Neither sidecar's SHA nor identities
are bound in its PASS receipt. Their observed checkpoint/code/norm identities
agree: 9f8304f7d001eec42e5c1774b424020f98049fc8e8e62449d4ce09686f807030 /
47f2dc2af4459f59deb145d84807a617ce67868f625c978444ecfccdb247e17e /
8672adf02916fdcdd002c98d6d3b4b261fd88b99c22034acc873edcfc05d57f4.
Please publish bound receipts from the historical run artifacts, preserving
the original receipts and recording their hashes. Do not reseal current files
and assert retrospectively they were qualified. P6 has not edited Q files,
started a GPU process, or silently promoted unbound PASS status to admission.

## 2026-10-10 cache study framework lead authorization

Latest owner task explicitly says framework parts yourself. P6 therefore adds
only default-off R17 common RecipeConfig miss_mode/fitted/stall/protocol fields,
shared sealed-fit factory dispatch, no-ordinary-MISS enforcement and execution
hook, and AdapterProvider.library_validator=None. No src/ or R16 edits. New
recipe fields omitted from identity when absent to preserve default identities.
Model hooks requested from P3/P4/D. Exact Q3 controller reuse lives in P6 search/;
legacy guard/periodic math and defaults remain unchanged. Tests/handback follow.

Final CPU-seal seam: make_bundle_validator(provider, *, recipe_factory=None)
keeps its default exactly as before. This allows a functional model-owned
configure_library_provider(provider) callback to supply CPU recipe admission
without mutating the frozen AdapterProvider or replacing its registry entry.
The callback must return a same-ID/same-ModelSpec provider; CPU seal separately
checks its config_validator. Normal resident registration owns the same hook.
