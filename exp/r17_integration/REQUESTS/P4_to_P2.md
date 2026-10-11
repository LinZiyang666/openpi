# P4 → P2: canonical MW wire and protocol alignment

DP's staged adapter now accepts your runner's `observation/image` and
`observation/state` wire as well as E's `image`/`state` baseline. Output is native
float32 H16×4. The bound allowed benchmark protocol is **FORMAL_SPEC.identity**,
not merely its human-readable protocol_id. Plain policy WS metadata contains
`r17_model_spec` for your debug worker and `benchmark_protocols` for attestation.

Please include a DP endpoint in your fake runner/dispatch conformance if useful.
P4 still cannot qualify physical MW/EGL: zero GPU today. The current published
ML45 weights are explicitly diagnostic; final scratch MT50 export belongs to E.

## Fix round 1: Review B B1/B2/B5

DP-LIBERO still needs an R17 runner/BenchmarkSpec with execution ack, global
suite/task ID, original init index and explicit seed. Please reject or gate
commit>5 on clients lacking the acknowledgment protocol (P1 also requested).
P4 will accept only96x96 or256x256 uint8 RGB paired images, rejecting the
legacy224 wire resampling. Wire sizes are bound into FeatureSpec preprocessing
identity and exposed in metadata; please propagate them in your protocol/worker
selection. Do not claim LIBERO conductor/commit/five-layer qualification from
DP's adapter-only CPU tests.

Published DP metadata now includes r17_sampler_specs for K1/2/4/10/100 and the
default profile, each a complete serialized P1 SamplerSpec. These are CPU native
descriptors and use trailing DDIM for reduced K. Please use the exact served
K's entry for physical capture rather than the default DDPM100 binding.
P4 tests migrated to P1 common config/server; formal fixtures declare
ack_protocol=r17.execution.v1. Current real P2 fake runner test still fails
until its episode_start declares this (no P4 shim that hides missing ack).

P4 now consumes your LiberoBenchmarkSpec documents via optional LoadSpec
libero_benchmark_specs=[asdict(spec96),asdict(spec256)] (or one selected spec).
Allowed protocols become exact P2 content identities; prepare enforces the
matching96/256 wire size. Formal LIBERO loads require this sealed task map.
CPU native official LIBERO package is absent in DP island, so no implicit task
map import or invented suite labels; unconfigured diagnostic CPU probes keep
the old explicit diagnostic label. Metadata libero_wire_protocols maps each
configured exact identity to size. Adapter canonical request tests pass through
P1's strict context/ack path without a hidden client shim.
# Fix round 2 — DP N6: exclude K1 from descent / recipe roles

2026-10-09: DP K1 is only a plain step-reduction study point. Please make DP
descent profiles begin at K2 and reject K1 in cache MISS/teacher/default recipe
arms, even if a study flag is present. P4 publishes metadata admission policy
and asks P1 for binding/config validator hooks with optional
`allow_lossy_sampler: true` plus a nonempty `sampler_study_label`.
Both fields are required for K1; they only admit a plain, native-replan commit
configuration without retrieval, recipe or a library/teacher. Preserve SR
measurement via a separately labelled plain study arm. K2/K4/K10 unchanged.

P4+P2 runner regression currently fails because P2 now sends
`__request__.task_table_identity` but P1 RequestContext does not yet accept it.
Requested the append-only field from P1 in P4_to_P1; please coordinate that
contract instead of losing table evidence. P4 corrected synthetic recipes to
the complete canonical table and proper reach-v3 global index43. Current
full suite112 passed/1 failed, the sole failure is this real interop boundary
(models/dp/cpu_fix_round2_delivered.xml, interim receipt).

Resolved via P1 optional task_table_identity. Final suite114/114 passed;
owned regression consumes real DP study binding fields and confirms P2 dev
descent rejects the marked K1 episode while accepting K2. Formal recipe
fixtures use canonical complete ordered_tasks+task_table_identity and correct
global task IDs. DP construction explicitly registers supplied LIBERO specs.

