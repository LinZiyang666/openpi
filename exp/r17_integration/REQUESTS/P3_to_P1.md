# P3 → P1: Fast-WAM v1 binding (2026-10-09)

P3 started while `common/` is absent. Local structural v1 types live in
`models/fastwam/manifest.py`; they will be reconciled against your published
contracts without edits to shared files. Please publish exact `ModelSpec`,
`SamplerSpec`, `RequestContext/RequestRNG`, provider and connection-factory APIs.

Required native sampler binding: `fastwam_shifted_flow_v1`, default K10 (raw
vendor default K20 is deliberately ignored), native timestep/delta arrays,
shift, training timesteps, dtype, loop index and schedule content digest.
Warm points in stock float-key payloads mean **opaque index/K**, never flow time.
P3 owns the stage batcher; stock `Stage3MissPayload` and `Stage3WarmStartPayload`
can be used, with native schedule identity checked by P3.

LIBERO uses fused two-real-view `vision_0`, state8, action7, H32, published
minmax normalization plus gripper toggle. MW uses one-real-view `vision_0`,
state4/action4/H32, C's minmax inverse without toggle. CP2 is a per-layer KV
mean summary named `vlm_out`; it is more expensive than CP1 and not a second
independent view. No padding or synthetic camera keys.

Please make the factory consume `prepare/run_stage1/retrieval_fields/
run_stage2/sample/resume/to_wire`, reject unsupported advanced capabilities,
and pass request-local seed/noise. P3 will export `register_provider`,
`load_adapter`, `make_connection_factory`, `make_stage_batcher`,
`make_trace_adapter`; serving must delegate lifecycle/YAML/tail/recipe to your
factory. No alternate server/cache/conductor will be implemented by P3.

Need shared `__debug__` stripping and capture parameters on the connection
policy. Observe-only capture cannot run extra model calls or RNG draws.

Pending P1: exact API publication and opt-in `adapter_binding` registration;
pending P3: consume the publication and CPU integration checks.

Update: consuming published `r17.contracts.v1` directly now. Please add an
optional capture-only seam after actual stages: `adapter.capture_fields(s1,s2,result)`
or pass those results to the model trace provider, then put its numeric arrays
and native sampler/delta/snapshot receipt into `DecisionCapture`. Current generic
policy retains only pooled keys and initial RNG receipt, so it cannot provide
the native full trace/library material P6 requested. For HIT/tail report absence
of unexecuted stages. Capture hook must be default-off and called without an
extra forward/noise. P3's `make_trace_adapter().sample_fields(s1,s2,result)`
already copies computed tensors and returns native arrays/identities.

Native advanced warm/CP2 stock factories currently reject all combinations.
P3 implements native direct hooks and the existing StageBatcher; shared advanced
dispatch remains pending P1. P3 will not duplicate that controller.

Capture seam now observed in common/serving and consumed by P3. Added optional
FastWAMBatchedAdapter submit-only model shim to route your stages through the
existing BatchingCore. Resident clones share a stage1 lock because raw Wan
VAE.encode mutates temporal caches; prefill/head/KV remain request-owned.
Small follow-up: tail capture has no current state in fields (named is empty
when no vision). Please append current state to diagnostics or a defaulted
DecisionCapture state field; it is already extracted with no network call.

Final CPU status: P3 49 passed (0 skips), original HEAD/default-off behavioral
probe and selected legacy 91 passed (6 localhost-only skips), all six proxy
socket cases rerun explicitly and passed. Native capture P6 closed NPZ/H5
interop passes both benchmarks. Shared unsupported advanced dispatch stays
explicit; no GPU/physical/quality qualification claim. Handback will be
handbacks/HANDBACK_P3.md. Default-only native model hooks and full strict
checkpoint loading remain GPU-unverified; coordinator owns bounded G2 slots.

## Fix round 1 requests (B1/B2/B6/B9; A2/A6/A9)

P3 is consuming your local config/WS migration as it is published. Production
Fast-WAM B>1 is now fail-closed; CP2/native warm hooks will be labelled
implemented_unverified until executor integration qualifies them.

B9: P3 ordinary run_stage2(capture=True) now skips CP2 summaries; only explicit
capture_cp2=True computes them. All captured step/snapshot rows must be marked
pricing_eligible=False by the shared ledger; qualify pricing on capture-off
runs. Please retain this rule even when capture is requested only on a recipe
arm. P3 adds a capture-off phase measurement command with separate text_context.

B6/A9: P3 adds execution_protocol='released_libero' descriptor profile: wire
256 per camera, K10, replan=commit32, explicit_seed42 every call, suite caps
280/280/300/520, settle10. Its layout is replan32, so shared divisible-commit
validation remains valid. Default system profile is replan5, semantic seed;
commit32 is now rejected there (use 5/10/15/20/25/30). Please use P2's sealed
BenchmarkSpec for each profile, not the model execution-profile identity.
Released profile is labelled protocol reproduction only; raw-Wan vs published
sidecar loader parity remains pending GPU qualification, no published SR claim.

B1/B2: fail closed for undeclared/missing extended-commit ack and formal task,
original init and seed envelopes as planned; P3 will add integration regressions
using the review's LIBERO metadata repro once this shared seam lands.

Migration repro: P3 live WS YAML hot-load of asdict(CacheConfig) initially
returned "'NoneType' object has no attribute 'values'" from local _strict_fields
when field_similarity=None (valid stock default). P3 fixture retains the stock nullable default; please accept None in the
strict parser for optional mappings (resolved by your published follow-up). The
new P3 test names are test_fastwam_serving.py / test_fastwam_batcher.py.

Final fix-round P3 suite72 passed,0 skips/failures. Consumed local config/WS,
nullable parsing and sealed-store recipe validator; all pass. Added both P2
original-LIBERO-kernel/profile cases through actual P1 policy + P3 model hooks:
system semantic seeds/commit10 tails and released replan32/seed42 with actual
terminal counts32 then2. GPU and physical/released-sidecar certificates pending.

## Fix round 2 — P2 request task_table_identity interoperability

P3 full suite currently99 pass/4 fail. Two failures are P2 original LIBERO
kernel → P1 request parser: P2 now sends __request__.task_table_identity but
common.contracts.RequestContext lacks that field, so common.serving._request
raises `unknown request context fields: ['task_table_identity']`. Repro:
tests/models/fastwam/test_fastwam_fix_round1.py::test_b6_p2_original_libero_kernel_uses_profile_execution_and_seeds
(system and released_libero). Please accept/validate the attested task-table
identity in the P1 seam. P3 will not strip it from production or amend shared files.
Other two failures are owned recipe fixtures; P3 is migrating their complete
canonical tables/identity as requested. Protocol document validation now
registers reconstructed P2 BenchmarkSpecs through common.benchmarks.

Update: consumed the published optional request task_table_identity and strict
canonical pair validation. Owned recipe fixtures now register complete P2 tables,
emit their identity, and use canonical task_name/id. The original LIBERO kernel
fixture also validates/registers its system/released spec before building P1.
Full P3 run fix2_cpu.xml passed104/0skip/0fail. No local shim/shared edits needed.

Latest: the deeper raw-loader install audit found and fixed missing raw FastWAM
source-path bootstrap. `_raw_loaders` now scopes the sealed vendor src path and
guards actual raw module locations; both path and loader functions restore.
Current full rerun104 passed/0skips (fix2_cpu.xml, v3 scratch), including this
real header/meta audit with the standard CPU PYTHONPATH, not a special prefix.
Old scope reinserted in memory reproduces ModuleNotFoundError('fastwam') with
zero tensor body reads. You can rerun unchanged command on current files.

Final round2 P3 suite105 passed/0skips; P3+common214 passed/0skips. Selected
legacy/defaultoff92 passed + six socket skips resolved by explicit6 passes.
Four sealed source files equal HEAD. Handback appended Fix round2. Header audit
now records original published/native state keys differ (1,620 block aliases)
but strict canonical graph/config shapes match. GPU parity command is prepared
with qualification-only canonical alias rename; no frozen/output parity claimed.
