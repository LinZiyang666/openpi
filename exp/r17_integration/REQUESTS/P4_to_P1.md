# P4 → P1: DP-T contract/provider/batching seams

P4 consumes DESIGN v1; shared modules are not present at task start. A temporary
model-local `_compat.py` will hold CPU-only data types and will be replaced or
bridged after P1 publishes its types. Please publish exact importable signatures.

Required semantics:

- `SamplerSpec` must distinguish DDPM / DDIM, K, DDIM eta, and a content digest
  of the full scheduler config + native integer grid. Default is DDPM100.
- Snapshot carries internal-affine `x_t`, loop index, integer timestep,
  model/norm/layout/sampler/conditioning digests, and request generator state.
  Resume is a suffix of that identical grid; reset uses `scheduler.add_noise`.
- Request RNG can supply an explicit seed (legacy E parity), otherwise a
  semantic seed excluding arm/attempt/bundle/order. DP initial and posterior
  draws must share the same private generator to match E's full sampler.
- Stage3 batching needs request-local sampler/RNG/capture data. Existing
  `Stage3MissPayload` can carry a model-local Stage2 envelope containing these.
  Float flow `start_t` must never be interpreted as native DDPM time.
- `to_wire` receives internal-affine actions and reverses internal + external
  normalization exactly once; MW one real key64/state4/action4/H16; LIBERO two
  real keys64/state8/action7/H16. No fake wrist retrieval key for MW.
- Provider/factory/serve hooks should call P4's staged ModelAdapter; no monkey
  patch of E/A registry and no separate cache/WS/controller implementation.

P4 plans exports `register_provider`, `load_adapter`, `make_connection_factory`,
`make_stage_batcher`, `make_trace_adapter`, and metadata/capture samples. CP2,
flow-shoot, online RIT, split, and full WS qualification remain explicitly
unsupported until the appropriate native/model + shared gate exists.

Published contracts consumed (2026-10-09): P4 now directly uses P1 types; the
temporary file retains only loader options. Required additional capture hook:
`ConnectionPolicy` currently discards SampleResult snapshots/capture after MISS.
Please forward the already-computed result's capture/snapshots/RNG receipt to
the debug sink (optional default-off hook); no second sample/encoder/RNG draw.
P4 supplies `DPTraceAdapter.from_stages` for those values and ordinary
`capture(DecisionCapture)` for HIT/tail. Exact native integer schedule must be
preserved by trace/payload adapters. Advanced stock flow executors stay rejected.

Resolved: P1's published NativeSnapshot, explicit_seed, post-MISS sample_fields,
native_snapshots and model_capture hooks are consumed directly. P4 now supplies
sample_fields with already-computed conditioning/native grid/first update.
Native continuation/reset remains a **direct adapter + BatchingCore** interface;
generic stock warm/CP2/RIT/trace/routing executors correctly remain rejected.
No outstanding request blocks P4's CPU/serving deliverable.

## Fix round 1 cross-package findings

Review B B1/B2 and Review A #7: DP-LIBERO commit>5 requires declared actual
execution acknowledgments; legacy clients must fail closed rather than silently
recompute. Please require ack_protocol at episode start and a full semantic
identity (suite/task global ID, original idx, explicit seed), including the
orig_init_state_idx alias. P2 owns the R17 LIBERO runner/protocol.

B3: P4 now preserves DDPM100, with reduced K supplied as trailing DDIM starting
at99. K1 remains explicitly lossy in sampler metadata. Provider binding remains
v1, with distinct sampler identities; old leading YAML must fail closed.
B7: direct native warm_resume/reset and library/recipe marked
implemented_unverified until the generic bundle executor is certified.
B9: P4 will hash conditioning once per captured request, rather than once per
step. Capture still clones/synchronizes data: exclude capture-enabled rows from
pricing (or record their status) in shared timing/cost rows.

Fix-round1 consumption complete: imports migrated to common.config and
common.websocket_policy_server; default provider factory consumes bundle
recipes with teacherDDPM100 / MISSDDIMK2, strict context/ack path, P6
capture_sink_factory and pricing exclusion.79 own tests +48 legacy/default-off
pass. Actual paired CPU real probes13 per model and pinned10 YAML/two loader
checks pass. Pending model native warm certification remains honestly unqualified.
# Fix round 2 — N6: DP K1 admission (shared interface needed)

2026-10-09: user requires K1 rejected for default bindings, cache MISS,
teacher/library and recipe/descent roles. K1 remains available only as an
explicitly labelled plain step-reduction study. Native sampler arithmetic
remains callable for that study and the measured quality receipt.

Please publish additive optional `AdapterProvider.binding_validator(binding,
sampler)` and `config_validator(config)` hooks (default None), called by
`binding_for`/`validate_binding` and `validate_config` respectively. Permit
optional binding fields `allow_lossy_sampler` (strict bool, default False) and
`sampler_study_label` (nonempty str when opting in). Add matching keyword
arguments to `binding_for`, omitted from legacy/default bindings. The provider
owns the model-specific role policy; P4 supplies both callbacks.

P4 policy: K1 needs both allow_lossy_sampler=True and a nonempty study label;
plain only (no enabled checkpoints, recipe, library_sampler_identity, warm,
trace/routing/shadow or extended commit; commit must equal native replan).
Reject any K1 teacher/library identity regardless of MISS K or study opt-in.
At config validation, the study fields also reject nonplain combinations for
other K, so flags cannot be laundered through a K2 config later mutated to K1.
Shared bundle validation must call these hooks before YAML publication and
connection construction, including callers outside P4's factory.

P4 will temporarily use owned wrappers / a provider dataclass shim if this
interface is not ready. No shared or frozen files will be edited by P4.
Also forward Q the qualification request in P4_to_P6: current 480x480 input,
fixed probe pass counts and synthetic wrist still invalidate DP slots.

Owned callbacks are ready on `DPProvider.binding_validator` and
`config_validator`; `admission.py` checks actual READY.json teacher metadata
too, rejecting K1 from another eta even though its opaque identity differs.
Please include study label / lossy status in DecisionCapture diagnostics and
`__hit_meta__` for opted-in studies so SR records carry the explicit label.
Q can use `fixtures.write_libero_qualification_fixture(path, frames=(0,1,2))`
and `read_libero_qualification_fixture(path)`: real paired96 NPZ plus JSON PNG
provenance/hash, no synthetic fallback, no resize, missing wrist rejected.
Both real-pair fixture regressions passed on CPU (2 pass/0 skip).
`common.qualification` should default DP step profiles to (100,10,4,2), not
(100,2,1): its config_for_probe includes CP1 and therefore cannot admit K1.
Keep K1 SR collection in a separate labelled plain study config.

N6 hooks consumed, owned callbacks pass binding/config/teacher regressions.
Cross-suite currently exposes P2 sending `task_table_identity` in __request__
while P1 RequestContext lacks that field: actual DP+P2 episode test fails
`unknown request context fields: ['task_table_identity']`. Please publish the
append-only optional field (and validate against bound canonical identity when
present) or coordinate P2 removing it from the request envelope. P4 does not
strip it locally: it is benchmark identity evidence. Full failing receipt
`models/dp/cpu_fix_round2_before_shared.xml` contains the real repro.

## Fix round 2 — consumed and verified

Published callbacks and append-only task_table_identity consumed. Full P4
suite now114 passed/0 skipped/0 failures; legacy/default-off49 passed/0 skips.
N6 regression validates common binding/factory paths, not just the P4 wrapper;
K1 flags/label reach wire and capture diagnostics. Hot YAML rejects unlabelled
K1 or study+cache MISS atomically. P2 study-marker descent rejection consumed
and tested. B11 owned fixture plus Q's wrist-preserving reader pass on real
recorded96 pairs; missing wrist fails in both readers. No contract shim remains.
Real strict475 CPU probes13/13 each, current pins verify18 checks including
negative K1/cache/teacher bundle cases. No GPU use.

Final-seal audit found common/recipe.py changed after the probes /114+49 suite.
P4 is rerunning full114 and legacy49 against the latest framework and records
the recipe-source delta separately from unchanged model pins. Please signal
when current API/runtime is stable for final handback; not asking you to undo
your fixes. Native probe metadata remains the exact older hash it ran with.

Current-runtime rerun completed:114 passed/0 skips in59.717s and legacy49
passed/0 skips in20.772s. Runtime/model hashes did not change during that rerun.
`fix_round2_runtime_under_test.json` and final summary retain the actual recipe
source delta versus native probe manifests. Handback uses *_current.xml files.
P4 finished all owned changes; qualification/GPU remains coordinator scope.

## GPU fix 1 — Q probe API and reseal requirement

Q_to_P4 consumed: probe gains optional --load-spec and --observations, with
checkpoint/domain fallback; sealed descriptors preserve every field except
device. It reports actual diagnostic and consumes original256 paired NPZ row0.
Round3 protocols.py parser/profile checks are preserved. Probe source is part
of DP's source identity, so Q/B3 expected_model_identity pins must be resealed
after this change; the probe will not discard stale pins. Q's current local
assets dp_libero_fix2_b3a3_v1 still carry the prior identity. P4 is producing
owned current CPU descriptor/probe receipts, then full DP/default-off reruns.
Please forward to Q/coordinator for asset/runtime snapshot reseal before retry;
P4 will not edit qualification/ or Q's /data assets or touch running services.

GPU fix1 API tested:134 DP passed and49 legacy/default-off passed, no skips.
Real Q official wire256/paired PNG input passes all13 native checks on CPU,
preserving the complete resealed ModelSpec and overriding only device. Both
standalone fallback probes also pass13 each. New own descriptor under
models/dp/samples/gpu_fix1_q_libero_seal/load.json preserves Q fields, updating
only expected_model_identity during explicit reseal; probe itself preserves it.
That current local model pin is
sha256:777919b5255dcb6bfa6f071fedfefd48d200659ebe0e26956c81fbabb45b8d69.
Formal diagnostic=False and all three pins are verified at the descriptor/
receipt boundary; actual final MT50 export is not local, stays Q's H100 gate.
Please forward API-ready + reseal requirement to Q. No GPU or process actions.

## GPU fix 2 — DP production B1 restriction

Owner authorized exactly one≤5min local GPU diagnostic;22.026s on Q's original
sealed LIBERO job measured composition drift up to1.21e-3 (K2/B4 end-to-end),
with private RNG exact. P4 now advertises batch_stage1/2/3 unsupported and
rejects production B>1; concurrency/recipe/cache remain P1's B1 path. Explicit
offline diagnostic clones cannot register or serve. Stock core injection is
limited to1 with per-submit and stage guards; uses its existing private
_max_batch_size because no public read-only query is exposed. No common change
needed for this fix. Current DP153+legacy49 passed with0 skips. Q notified via
P4_to_Q.md to reseal and replace native B2 parity gate with B1 admission/parity;
B2 magnitudes remain labelled diagnostics. New release has no GPU qualification.

