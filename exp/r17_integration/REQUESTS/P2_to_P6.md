# P2 → P6: physical/execution capture seam

## Fix round 2 — N5 ledger service time and N2 task table

P2 requests P1/P5 instrumentation for queue_wait_ms/service_ms/stage_queue_wait_ms/stage_service_ms, native timing scope and pricing_time_field. Runner cost_row will preserve these and full server timing; unknown queue waits cannot silently become zero. Please have summarize/audit select attested service_ms for timing normalization; host cost_ms remains latency diagnostics, GPU timing remains unsupported. Preserve the service/queue split in model capture and receipts and document the pricing field. Canonical task_table_identity (digest ordered canonical names) will be published in benchmark/episode metadata and emitted π configs, with P1 enforcing exact recipe order. No cross-package edits by P2.

## Fix round 2 — cross payload regression / published timing
Own full P2 suite106 passed; legacy451 passed+6 socket skips separately6 passed. Cross342 collected:340 passed,2 failed in tests/ops/test_ops.py lines51/65: build_client_bundle allowlist rejects new CPU-only common/benchmarks.py required by explicit canonical table registration. Current P2 copied-file Python3.11 import probe passes with that module and no torch/JAX/simulator. Please allow this exact client-safe module (imports threading/contracts, lazily canonical MW descriptor) or advise server-only registry separation. XML cpu_fix2_cross2.xml retained; no allowlist weakening by P2. N5 P1 hook is now published and π supplies zero adapter lock wait; actual held CPU lock witness subtracts wait from service_ms. P2 persists gpu_service_ms=None, stage_queue_wait_ms/stage_service_ms, service_timing_scope and pricing_time_field. Please use service_ms only for attested timing normalization, keep host cost_ms diagnostic, and primary owner IR N/V/M coefficients separate.

Local shim resolved the payload dependency without changing your allowlist: workers only reconstruct/validate descriptors (load_benchmark_spec(register=False)); server/provider loaders and conductor plan register tables explicitly. Client runner/adapter constructors no longer touch the server registry; canonical identity remains a pure descriptor calculation. common/benchmarks.py removed from CLIENT_ONLY_FILES. P3/P4 loaders already call register_benchmark, π model_spec registers its own protocol; CPU LIBERO fixture now models that explicit server initialization. Rerunning payload/cross. You need not weaken your allowlist. Service-pricing analyzer selection still pending your area.

Final round2 unshimmed broad346 passed0failed0skipped (cpu_fix2_cross4.xml); full P2 108 passed. Your payload tests both pass without allowlist changes. P2 native timing bridge and GR canonical emission bridge are opt-in CPU-tested APIs; production P5 hook still pending. Ledger keeps cost_ms host latency, service_ms separate; pricing_time_field='service_ms'. Current summarize still sums cost_ms even when service is attested, so P2 handback explicitly warns against calling that ratio service pricing until your consumer selects service_ms. Actual measured coefficient/IR/currency/GPU price remains unqualified. Please consume or own the pending analysis change; no P6 edits from P2.

P2 implements benchmarks/physical_mt50.py NativeMetaWorldPhysicalAdapter.capture(obs=None, info=None, requested_action=None, applied_action=None) → mapping {schema, arrays, metadata, capabilities}. Arrays are copied numeric float64 except integer IDs. Native env.model/env.data only; task predicates and restore remain unsupported/uncertified. Contact forces use native mj_contactForce or injected CPU fixture. Full native entity catalog retained; reach-v3 explicit goal position available when _target_pos exists, certification is CPU fixture only until real GPU/renderer smoke.

benchmarks/metaworld.py BenchmarkAdapter.make_episode_runner(client_factory, capture_sink=None, execution_ack=None) callbacks receive mappings with episode identity and event reset/settled/decision/control/terminal. execution_ack receives each completed decision's actual controls, last cursor, requested/applied, terminal. Debug defaults off, no extra inference or render. P6 owns writing/transport; P2 does not write H5/NPZ.

Update: common v1 now published/consumed; benchmarks/spec.py::BenchmarkSpec.to_contract is the source for protocol identity. models/pi05/adapter.py::Pi05TraceAdapter.capture consumes DecisionCapture without forward/RNG. PhysicalSink in P6 now matches event mappings. Client-only explicit file list and fresh-process import probe live in conductor/worker_entry.py::CLIENT_ONLY_FILES/import_probe. Physical control records carry actual decision_seq/control_idx; reset/settled/terminal events retain metadata without invented decision indexes. Samples are provided by P2 tests/benchmarks; real goal-map/renderer certificate remains pending, and restore_certified=False.

Client-only payload list expanded to ALL modules imported by openpi.conductor.__init__, including agent/driver/scheduler/health/monitor. An isolated `-I` copied-file probe caught the omission; corrected list lives in worker_entry.py. Debug-enabled worker additionally needs P6 debug modules and existing offline_search.debug writer/schema packages (P6 owns that optional payload expansion). Worker default capture stays off. `--debug-directory` explicitly composes P6 PhysicalSink/CaptureBridge and closes writer at exit. EpisodeResult execution actions are wire-safe lists; physical NPZ retains numeric float64 arrays.

Server live hook now exposed by models/pi05/serve.py --debug-directory, via connection-local sink_factory(core) → P6 DecisionSink(bridge, core._episode). R17 runner (framework_context=True only) enriches original episode_start extra metadata with actual task_uid/arm/model/norm/layout/protocol/task-data hash; canonical A decision_seed is passed through explicit_seed to P1. The native sampler capture_fields provides prefix_embeddings/prefix_mask/normalized_state/model_input_base_rgb/initial_noise and NativeSnapshot tuple without extra calls. Please ensure DecisionSink serializes DecisionCapture.model_capture/native_snapshots (currently it writes normalized/wire/state/grid and retrieval keys only), or label unsupported storage fields in P6 handback. P2 CPU E2E now exercises real P6 model writer + reader alongside physical writer.
## Final P2 receipt / sampler-bound capture resolver

Final owned CPU suite: `tests/conductor/cpu_p2_binding_release.xml`, **51 passed,
zero failed/skipped**. `handbacks/HANDBACK_P2.md` includes actual commands and
future unexecuted ≤15-minute slots. P1's dtype fix is used directly and the final
standalone WS probe has full nonempty provenance. Real WS test now checks model
writer groups `{K10,K2}` and each group's actual NFE/sampler binding.

The live π serve hook uses `models/pi05/provider.py::make_capture_sink_factory`:
core-aware sampler/model binding → your CaptureBridge/DecisionSink, with
`lambda request: dict(core._episode)` as connection-local resolver. This is a
thin local shim because CaptureFactory currently receives only a request resolver,
which cannot obtain original task UID/task-data bytes from core episode metadata.
Please publish a core-aware CaptureFactory resolver/attach_core hook; P2 can then
remove this binding-selection shim. P2 does not implement any writer/storage math.

Clean samples under `tests/models/pi05/samples/native_k2/` are labelled synthetic
CPU and use actual K2 binding, native snapshots and copied float64 physics;
earlier outer sample artifacts are diagnostic only. Builder supports `--output`
and verified completed reuse. Your hashed protocol build requirement
`split_policy='mt50_init_v1'` is recorded in the handback. Real EGL/goal/restore,
closed actual-checkpoint library/accepted result sync remain coordinator/P2/P6
qualification, not claimed by these fixture receipts.


## Fix round 1 — LIBERO and terminal receipts
P2 implementing LIBERO canonical suite/task names with global task IDs 0..39 (suite order spatial/object/goal/10), actual original init index and explicit paired decision seed. Existing kernel/env retained via adapter. Both runners send final actual-count ack on episode_end and keep identity-complete cost/ack rows (canonical bundle_generation from task extra). Worker will consume PhysicalCaptureFactory metadata and injected sampler_resolver; please publish callable signature. Please consume cursor_end or last_cursor+1 and preserve source/timings. Recipe fitting for LIBERO uses all canonical names, no suite-local IDs.

Consumed CoreCaptureFactory/PhysicalCaptureFactory. MW cost and execution rows now carry full task/bundle/attempt/original idx identity; generation is t.extra.bundle_generation everywhere. Terminal ack goes through stock _send_ctrl with __execution__, cursor_end explicit in result receipts. Stage_ms/timing_scope/capture/pricing/server_timing preserved. Real WS LIBERO commit10/15/32 CPU tests pass and sum final server acks to34.

## Final cross regression needing P6 fixture correction
P2 now fails closed before worker physical capture if task model_identity/norm_identity/layout_identity differs from attested r17_model_spec (prevents a mixed-model global queue silently writing wrong binding). Own tests pass75. Latest P1/P2/P6 cross:273 pass,1 fail at tests/debug/test_review_worker_entry.py:64: fixture uses task() helper with model_identity=model/norm_identity=norm/layout_identity=layout but metadata uses model_spec(m,n) (or corresponding actual values). Please update your test's EpisodeTask extra to actual endpoint spec checkpoint/norm/layout identities. Do not weaken the production guard. Full traceback in tests/conductor/cpu_fix1_cross_sealed.xml. This is P6-owned fixture; P2 has not edited it. P2 will re-run after update.

Observed P6-owned worker fixture now uses spec.checkpoint/norm/layout identities; starting full unshimmed cross rerun. P2 no foreign edits. This resolves the production guard compatibility issue once the receipt passes.

Final unshimmed P1/P2/P6 cross:280 passed0failed0skipped in60.89s. Owned P2:76 passed0failed0skipped. P6 fixture correction resolved; actual sampler/model physical binding guard retained. XML tests/conductor/cpu_fix1_cross_release2.xml. Cost/terminal ack/P6 writer reader join tested, no GPU.

Final A7 follow-up: both R17 MW constructor/factory default framework_context=True; explicit False remains byte-parity path. Full final owned77/cross281/legacy451 pass, zero failures. Cross XML cpu_fix1_cross_release4.xml includes your default-constructor physical capture test; no additional fixture edits or guard bypasses. Previous release2 receipts remain historical. No GPU.
