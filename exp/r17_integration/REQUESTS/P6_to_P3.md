# Model trace handoff

Please expose JSON-compatible ModelSpec plus already computed named retrieval
fields/actions/state/native_grid/snapshot/RNG arrays. P6 consumes these through
debug.capture_bridge and stores arbitrary field widths (including fused keys)
without camera duplication. Decision ledger must include actual stage calls,
actual NFE, cursor/executed_count and capability statuses. Native snapshot metadata
includes identity, loop index, native t, RNG algorithm/state, posterior noise.
No model-specific forward in P6. Cross-model GPU receipts remain your G2 gate.

## CPU handoff consumed

P6 ran your two benchmark instances of
`test_native_capture_consumed_by_p6_closed_npz_h5_bridge` with CUDA hidden/JAX
CPU. Both passed; shared receipt `tests/ops/receipts/p6_model_handoff.xml`
contains these plus P5's separate bridge test (3 passed, 0 skips/failures).
Native K2 stays distinct from default K10. P6 handback retains G2/G3 GPU limits.

## 2026-10-10 cache study — coordinator dispatch P3

Owner requests eight YAML groups, DP/Fast/Cosmos, A500 evaluation against B500
libraries. P6 is implementing framework-only miss_mode=never|stall_only,
sealed fit loading, the exact R6/Q3 stall controller (nominal lottery=0), YAML
publication, resident reuse, and group-commit dispatch. P6 does not edit models.

1. `models/fastwam/formal_batch_server.py:76` rejects a nonempty recipe at
startup; its shared factory already binds runtime adapters on hot-switch.
Please admit `campaign.cache_study` explicitly, preserve default B1/plain
behavior, attest study mode, and exercise CP1-only hits and blind tails with
BatchRuntime.wrap_connection_factory at B16. No weights reload or re-registration.
Alternatively expose your resident factory composition to P6's generic entry.
2. Library manifests include all adapter source hashes in checkpoint identity.
The B libraries were built before several batching changes. Please implement
`AdapterProvider.library_validator(config, teacher_binding)` (new default=None
framework seam): fail closed unless weight/norm, feature producer/preprocessing,
action-space/output transform and geometry match. Return a content-bound
`{schema:r17.teacher_compatibility.v1,status:PASS,teacher_identity:<common.recipe.retrieval_model_identity(teacher)>,resident_identity:<same resident>,evidence:<receipt/path and source hashes>}`.
No blanket ignoring code_digest/checkpoint identity. Pure cohort admission
changes B→official A are separately proven by P6 using complete native profiles.
3. GPU smoke later: resident 2×B16+MPS H100, all eight YAMLs, two episodes/group.
Need a CPU model-adapter test proving HIT=(1,0,0), tail=(0,0,0), gate-only native
K10 call. No GPU launch requested. Report exact model hook status to coordinator.

Clarification after tracing actual launcher: P6 starts the existing full.yaml
plain resident, then publishes/selects all study recipes. Thus the startup
recipe rejection needs no change for this study. Please focus on the explicit
teacher compatibility hook and CPU B16 cache-hit/tail behavior. P6 implements
its own generic final batch audit (CP1 library hit has stage1 only; tails none;
only stall policy calls have all stages), replacing the plain-only audit in
this opt-in mode. The service still starts once and does not reload weights.

Teacher vs accepted B16 control metadata differs in checkpoint_identity,
code_digest and feature_spec as well as allowed protocols (both suites). These
are strict blockers, not just a code-label nuisance. Please attest producer/
weights/norm equivalence or require recollection; never transplant 64D keys
by width alone. Optional configure_library_provider(provider) export in
provider.py is called by P6 CPU sealing, because qualification.assets uses a
synthetic Fast provider. Attach the same validator on resident registration.

Provider is frozen: configure_library_provider(provider) MUST RETURN a
same-ID/same-ModelSpec dataclasses.replace(provider, library_validator=...).
Do not mutate it. The CPU sealer passes that factory to shared bundle validation;
it retains the original registered provider and never changes resident weights.
The normal registered resident must carry the identical model-owned hook.

## Matched cache lanes resolved — P6, 2026-10-10

Fast-WAM CPU admission is consumed by /data/mw_finetune/eval/csl10f (8,000 A cells,16 sealed YAMLs). Helper now matches hs6 R2/B16/5ms/MPS/GroupCommit and timan10748 at2048/768/N4/EGL0..7. Complete control hs6fw_res_152931 pins native runtime hs6fw_ctl_143332 with reviewed R17 admission overlay. No GPU launched. No further P3 dispatch required before coordinator smoke. See ops/cache_study/FULL_RUN.md and CACHE_LANE_CPU_READY.json.
