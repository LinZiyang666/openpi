# Native physical capture / accepted episode join

P6 accepts a JSON metadata record plus named numeric arrays, with native qpos,
qvel/contact force float64 and requested/applied actions recorded separately.
Preserve `capabilities` entries as {status, reason}; missing goal/predicate/restore
semantics remain unsupported. No `.sim` or task-text interpretation in P6.

Please supply fixtures and wire `CaptureBridge.capture_physical(record, arrays)`
or shared StreamSink for physical records. Identity includes campaign, arm,
bundle_generation, benchmark_protocol, task_name, orig_init_idx, task_data_hash,
task_uid, attempt, decision_seq, control_idx. Accepted joins admit only terminal
accepted successful-result-status attempts (success false is a valid episode).
Errors, rejected and late attempts remain diagnostic rows, outside formal SR.

Client bundle builder takes an explicit list of client-only files; P2 please
publish that list and import probe entrypoint. EGL locking remains in the existing
worker/env implementation; P6 never launches renderer or probes GPUs today.

## Resolved / current integration notes

P6 consumes P2's actual callback and worker file profile. CPU test now runs the
P2 native runner through PhysicalSink (reset/settled/7 controls/terminal), checks
float64, requested/applied clipping and actual 5+2 execution. Payload closure is
imported in a fresh Python3.11 process without torch/JAX/vendor/simulator imports.
DecisionSink now preserves DecisionCapture.model_capture and native_snapshots,
including private RNG bytes; P1 on_execution includes its RequestContext.

Use CaptureFactory for a writer per exact active sampler binding when serving
K10/K2/K1 bundles. One CaptureBridge bound to default K10 may retain reduced-K
diagnostics, but those records must not build a K10 teacher library. The builder
rejects actual grid/sampler mismatch. For hashed FORMAL_SPEC.identity, pass
split_policy='mt50_init_v1' to build_library; labels alone don't describe init
boundaries. Remaining real renderer/goal/restore/G1/G3 qualification is yours
with coordinator. No extra inference/rendering in P6.
# Fix round 1 — REVIEW_A #5/#6/#10; REVIEW_B B1/B2

P6 actual worker main regression now passes: native fake env + P1 staged core,
mocked renderer/socket boundaries, real PhysicalCaptureFactory attachment from
production worker entry. K2 physical binding == K2 model binding, canonical
generation deliberately differs from bundle ID, 5+2 final acknowledgments,
7 float64 physical controls, exact full episode identity. No renderer/GPU claim.
P1 reported your shared E2E fixture still fixes PhysicalSink metadata to full
while dispatching retrieval. Please use {} or event-aware episode resolver in
your owned fixture, not a P6 provenance-guard bypass (A10).

Published now: `make_capture_sink_factory(provider, directory)` (directory-only
compatibility is also supported), returning callable(core) with .bridges/.close().
`PhysicalCaptureFactory(directory, server_metadata, sampler_resolver=None,
enabled=True)` is callable(event), has .bridges/.close(), and accepts event
['sampler'] as full SamplerSpec mapping or event['num_steps'] with resolver(k).
Reduced K requires full explicit sampler or resolver; explicit default K may
use attested default_sampler. Each event must include full nonempty episode
identity and canonical bundle_generation. Sampler identity disagreement fails.
P6 added reader_profiles.py to optional debug client payload import closure.

Cost consumption published: `debug.reader_profiles.merge_decision_ledgers`
and `search.analyze.summarize` / `search.audit.audit_campaign` accept
cost_rows=runner per_step_rows and execution_rows=client ack rows. Summary rows
are ignored, _kind=execution_ack automatically split. All decision/cost/ack rows
require decision_seq and full episode identity, or a sealed episode_manifests
entry keyed by task_uid+attempt; no sequence-only joining. Final last_cursor+1
and server cursor_end are validated. capture_enabled rows excluded from pricing.

P6 will publish core-aware `make_capture_sink_factory(provider, directory)`
and `PhysicalCaptureFactory(directory, server_metadata, sampler_resolver=...)`.
Physical factory consumes explicit `r17_model_spec` and each event's actual
sampler (`sampler` full spec, or explicit num_steps + injected resolver), grouping
writers by full model/protocol/actual sampler binding. No K guessing/default
binding for a reduced-K task. Worker must pass canonical bundle_generation,
actual sampler identity/spec and episode identity into every capture event.

Please provide decision_seq and FULL episode identity in cost_rows/per_step_rows,
measured cost_ms/stage_ms or server_timing.infer_ms with explicit timing_scope,
capture_enabled/pricing_eligible. Final client execution_ack must reach server
episode_end (and remain in result rows). P6 reader can also overlay exact client
execution_ack rows, including last_cursor -> exclusive cursor_end conversion.
P6 will reject cost/ack joins that disagree on bundle/task-data/attempt identity.

## Library pre-decision retry / repair, 2026-10-10

Owner permits at most two infrastructure retries before any executed policy
step or captured decision; an ordinary unsuccessful outcome is never retried.
P6's opt-in formal driver now applies this policy before delegating results to
your stock queue, seeds original UID dispatch generations for a four-episode
repair, and makes unreported timeouts fatal for library campaigns. Non-library
driver behavior is unchanged. The owned library sink writes a synchronous
input-intent fence before inference, closing the stock WorkerLoop exception
path's loss of partial n_steps/per_step_rows. Parent provenance retains every
eligible error and rejects capture rows from any prior attempt.

Requested future P2 API: optional driver retry predicate receiving the raw
result and trusted progress/capture evidence, and worker exception receipts
that preserve actual controls/decision count. P6 uses an owned launcher shim
until that shared hook is published; no P2/src file was edited. Historic repair
without the intent fence accepts only the explicit zero-step EGL framebuffer
initialization failure, with no per-step/capture rows and a fully closed cohort.


## 2026-10-10 cache-study evidence allowlist

benchmarks/execution.py cost_row discards corrector/Q3 stall activity, donor rows
and native batch composition. Final LIBERO ACK response is checked for success
but the persisted local last receipt is not marked with the server terminal ACK.
P6 adds an explicit process-local ops/cache_study_worker evidence shim, enabled
only for campaign.cache_study, around your existing cost_row/end_episode (no
new rollout/model loop). It preserves default bytes, adds the study metadata
and attaches the validated final server ACK. Please eventually expose an
optional cost-row extension/terminal-ACK sink instead of the local shim. No
P2 files edited. New launcher/tests consume your standard runner/receipt path.
