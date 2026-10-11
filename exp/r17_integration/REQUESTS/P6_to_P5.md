# GR MW trace handoff

P6 bridge expects actual one-camera fields + 4 active state/action channels,
normalized H16 chunk, wire chunk, native schedule and identity. No fake wrist.
Preserve original widths and active-channel layout. Please expose an already
computed trace fixture and callback integration; H5 sink/NPZ shared writers are
reused. K4/2/1 endpoint identities must remain distinct. P6 qualification consumes
G2 receipts; no GPU probes today.

## CPU handoff consumed

P6 ran `tests/models/groot/test_p6_bridge.py` with CUDA hidden/JAX CPU;
it passed, including original YAML backend preload, nested model binding,
full normalized H16x32 / active4 geometry and exact shared CP1 HIT bytes.
Receipt `tests/ops/receipts/p6_model_handoff.xml` contains it plus two distinct
P3 bridge cases (3 passed, 0 skips/failures). Native real BF16/checkpoint and
renderer qualifications remain yours/coordinator's future GPU gate.
# Fix round 1 — REVIEW_A #5/#10

Resolved compact artifact migration: stock_artifact now embeds
common.recipe.retrieval_model_identity(s) in model.r17_binding. Own regression
loads original stock YAML preload with K4 teacher and K2 MISS, preserving teacher
digest distinct from served sampler; artifact provenance validates. P6 reran
your test_p6_bridge plus P3 two native NPZ/H5 cases: 3 passed, 0 skips/failures.
No native snapshot cross-K resume enabled. Actual GR serving assembly and P2
worker main debug attachment also pass P6 CPU regressions with strict identities.

P6 helper is published now: make_capture_sink_factory(provider, directory)
returns callable(core), .bridges and .close(); directory-only form is accepted.
CPU regression verifies core episode UID/task-data and native K2 binding,
exact terminal executed count, plus worker sampler-separated K2/K4 captures.

Please expose metadata['r17_model_spec']=asdict(provider.model_spec), and add
--debug-directory wiring to P6's upcoming core-aware
debug.capture_bridge.make_capture_sink_factory(provider, directory). Callable
sink_factory(core) derives actual core.sampler/protocol and core._episode;
factory.close() seals each sampler-bound writer. No writer code needed in P5.
P2 worker consumes the same r17_model_spec through P6 PhysicalCaptureFactory.
P6 will test these deployment bridges under CPU; no shared/model edits by P6.

