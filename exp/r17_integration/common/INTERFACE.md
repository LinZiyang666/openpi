# P1 published interface: r17.contracts.v1

`contracts.py` is CPU-safe (stdlib only); imports register nothing. Model packages
export their loader and register an `AdapterProvider` explicitly. Registration
accepts provider ID, immutable `ModelSpec`, zero-argument adapter loader,
`sampler_for_steps(k)`, optional key-builder/artifact/trace factories. Loaders
must create a connection-local adapter with a shared read-only model underneath.

`binding_for(provider, protocol, commit_horizon=..., num_steps=...)` generates
the exact `adapter_binding` YAML mapping. Use `key_builder.type: adapter_native`,
exact real `FeatureSpec.fields` for enabled keys and vector dimensions, and
`write_policy: {type: never}`. Native sampler identity replaces legacy
`denoise_schedule`. No provider is registered by default. Stock full-hit/miss
CP1 gate/search/judge/backend factories are retained. Advanced stock flow
recipes need their model-specific certified factory; unsupported combinations
are rejected instead of being silently approximated.

`make_connection_factory(provider, default_config, default_storage, ...)`
returns a callable for the opt-in `common.websocket_policy_server.WebsocketPolicyServer`
subclass of the unchanged existing WebSocket server. Import the extended
`CacheConfig` and config/storage factories from `common.config`, and native
schedule registry/payload extensions from `common.native`. Pass
`bundle_validator=make_bundle_validator(provider)` to that server. Bundles are
looked up by the requested ID, with new generation IDs on reload. Capture sinks
consume `DecisionCapture`, after real stages; no debug forward or RNG call.

The observation envelope reserves `__debug__`, `__request__`, `__execution__`.
The first two contain diagnostic data and semantic `RequestContext` overrides.
`__execution__` acknowledges the preceding decision using `decision_seq` and
`executed_count`; it is stripped before preprocessing. Cursor advances only
with that acknowledgment. Without acknowledgment tail reuse is disabled.
Optional `capture_sink.on_execution(ack)` receives the original decision's
`RequestContext` as `ack['request']`, allowing shared sinks to join concurrent
sequence numbers. Tail captures include the current state-only `robot_state`
field; no vision/encoder is rerun.

Real bundles require campaign, task_name, global task_id, original init index,
seed and explicit_seed, supplied through episode metadata and/or __request__.
`orig_init_state_idx` is an episode metadata alias for original_init_idx.
Missing identity fails before model work. Extended commits require episode
metadata `ack_protocol: r17.execution.v1`; a missing next-decision ack fails
loudly. Only fake diagnostic protocols may set diagnostic_context=true to
exercise mechanical calls without a benchmark envelope.

Commits must be multiples of replan within [replan,H]. A client which handles
short final plans may opt into `binding_for(..., allow_partial_commit=True)`
and declare `partial_tail_protocol: r17.partial_tail.v1` as well as ack_protocol.
This supports LIBERO commit32/replan5; standard MW never opts into this mode.
`episode_end` accepts `__execution__` for the final decision; its receipt and
capture callback include actual count, cursor/cursor_end, terminal status and
original request identity. No extra infer is needed. The optional local client
subclass exposes episode_end(execution=...) and acknowledge_execution(...).

`recipe: RecipeConfig` uses `r17.recipe.v1`: enabled (default false), store_path,
store_digest, teacher_sampler_identity, optional checked fit_identity,
corrector, guard, periodic_gap OR target_ir (fraction), k/kref/early and
ordered_tasks. `make_recipe_factory(provider)` opens and verifies a sealed P6
store, fits immutable arrays, and forks mutable state per connection.
`make_bundle_validator` verifies/fits enabled stores before publication.
Recipe selection is independent of MISS K; a disabled recipe never consumes
an injected process recipe. `target_ir` means R16 owner IR:
`(vision*V + miss(K)*M)/N`. It requires `ir_costs` from
`common.ir.bound_costs(config, vision=..., miss=..., source_identity=...)`,
bound to immutable retrieval model, MISS sampler, benchmark and cost authority.
The factory calibrates gap on library-only LOEO schedule replay (guard and
actual commit tails included), using R11's [0,40] bracket and26 bisections.
Missing/identity-mismatched profiles and infeasible targets reject. The caller
supplies a verified cost authority; a declared coefficient profile alone is
not a measured real-model pricing receipt. Fit evidence and decisions
carry `ir_calibration` with setting, predicted IR, actual N/V/M and identity;
finite library quantization means a target need not equal its prediction.
R16 authorities: `r16/cost.py:13-28`, `r16/analyze.py:184-197`;
calibration authority: `r11/knob/calibration.py:89-104,138-153`.

Register canonical specs through `common.benchmarks.register_benchmark(spec)`
(P2 specs may supply `to_contract()`). Real protocol IDs equal spec identity.
Enabled recipes require the complete exact `ordered_tasks` and
`task_table_identity(protocol)`; a selected teacher cohort may be a subset,
but its task table may not be. Disabled recipes may retain validated inert
table metadata. MW has a CPU-only canonical resolver; LIBERO is explicitly
registered by the benchmark/model owner. RequestContext accepts optional
task_table_identity; supplied identity and task_name/task_id pair are checked
before model work. Enabled/inert recipe tables always check the pair.

Verified stores are cached by resolved path/content digest; changed inode,
size, mtime/ctime or mode rejects the immutable generation. SHA verification
and fitting happen once per content/fit identity. Hot load and connection
construction run in workers through the socket facade, while the unchanged
stock handler commits the prepared stack on its event loop. Bind checks use
file metadata, never whole-store array hashes.
`bundle_identity(config)` seals all semantic config, including recipe/fit/library
identity, K and commit. Generation IDs cannot be rebound to different content.
Episode bundle_identity, if supplied, must match the connection. P2 binds exact
identity in endpoint compatibility and episode dispatch.

Full-action teacher libraries and MISS sampler grids have separate identities.
For compact stock artifacts, set library_sampler_identity explicitly (or enable
a recipe with teacher_sampler_identity); build metadata with artifact_binding.
Retrieval model identity includes code/checkpoint/norm/layout/features/transform/
protocol/island, excluding default_sampler and capability receipt declarations.
Native snapshot and warm continuation equality remain strict across grids.

Every decision includes measured cost_ms and stage_ms/stage_timing_ms in
__hit_meta__ and DecisionCapture.diagnostics. timing_kind=host_wall and
timing_scope=policy_host_wall_before_capture_sink: no GPU synchronization or
device-latency guarantee. State-only work is timed even for zero-network tails.
`adapter.drain_queue_wait_ms()` optionally attests per-operation model-lock
wait. The core publishes queue_wait_ms, service_ms, stage_queue_wait_ms,
stage_service_ms, service_timing_scope and pricing_time_field='service_ms'.
Absent instrumentation produces None service fields and pricing_eligible=False.
This is host service time; gpu_service_ms remains None and concurrent GPU
contention is not removed. Owner IR comes from N/V/M and sealed coefficients.
Capture preparation overhead is measured separately, sink duration appears in
the response, and all capture-enabled/sink-enabled decisions are ineligible
for pricing. `capture_sink_factory(core)` receives the bound core and actual
sampler; P6 provides a closeable per-connection writer factory.

Providers may supply additive `binding_validator(binding,sampler)` and
`config_validator(config)` callbacks, enforced by the common binding/config
factories before YAML publication and bind. Optional allow_lossy_sampler
(strict bool) requires nonempty sampler_study_label; both are omitted by
default. Opted-in decision metadata/capture carries the study label and lossy
status. P4 enforces DP's plain-study-only policy and rejects K1 teachers;
generic DP qualification defaults to100/10/4/2, excluding the K1 study.

Normalized results have `(H, layout.action_width)`, wire results `(H, valid_action_dim)`.
Single-camera/fused-camera fields are real features only. Native snapshots bind
checkpoint, norm, layout, sampler/grid, index, native timestep and conditioning;
`RequestRNG` supplies independent semantic initial/posterior streams without
arm/attempt/worker/replica dependence. Adapters preserving an upstream sequential
generator may use `torch_generator()` and must capture its continuation state.

Shared algorithms and fit data APIs live in `layers/` and `query_view.py`;
P6 owns artifact serialization, P2 owns benchmark/task descent integration.
