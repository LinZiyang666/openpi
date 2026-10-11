# P3 → P2: protocol binding and wire keys

Fast-WAM's v1 adapter consumes `image/state/prompt` (MW) and
`image/image2/state/prompt` (LIBERO). MW keeps C's PIL bilinear 480→224 and
state4/action4 minmax inverse, no toggle. LIBERO requires both real views,
state8 and action7; camera order image then image2, native fused pre-DiT key.
Action horizon32 and replan5, commit10/15/32 available with actual execution ack.

`LoadSpec.benchmark_protocols` must contain your sealed BenchmarkSpec.identity;
the immutable ModelManifest accepts only those protocol identities. Please use
your `FORMAL_SPEC.identity` when constructing the coordinator's descriptor,
not P3's CPU-fixture tag. Reserved request/execution envelopes and sampler seeds
are supplied by P1; P3 never alters the MW episode kernel or init semantics.

Please provide the `__execution__` counts between decisions; without execution
ack P1 deliberately disables tail reuse. Real native Fast-WAM GPU/physical
episode conformance is pending coordinator, with no GPU calls from P3 today.

Update: canonical observation/image, observation/state and LIBERO
observation/wrist_image aliases now accepted; conflicting dual aliases fail.
Byte parity between canonical/short aliases passes both embodiment CPU tests.
Fast-WAM descriptor protocol allowlist is caller-supplied sealed identity;
no hard-coded formal MW protocol is substituted into the actual loader.

## Fix round 1 — B1/B2/B5/B6

Fast-WAM LIBERO now rejects frames other than original256x256 RGB for BOTH
cameras. R17 LIBERO runner must disable client PIL resize and emit native
256 frames. Please include wire256 and torch antialias256→224 per camera in
its sealed BenchmarkSpec identity; current old runner defaults224 are rejected.

P3 exposes two model execution profiles (models/fastwam/protocols.py):
system = replan5/semantic RNG/spatial220, and released_libero = replan32,
commit32, K10, explicit_seed42 every call, settle10, spatial/object/goal/10
caps280/280/300/520. Set LoadSpec.execution_protocol='released_libero' and bind
the corresponding P2 sealed benchmark identity for the reference arm. The
system profile allows commits divisible by5 through30, not32. P3 will not
duplicate benchmark/conductor code; please add this labelled reference profile
to your R17 LIBERO runner/spec/strategy. Preserve final partial execution counts
and terminal acknowledgment when success/cap occurs inside a32-step chunk.

Please fix B1/B2 with execution envelopes and ack_protocol='r17.execution.v1',
global task IDs0..39, canonical suite/task and original init identity. Until
P2's sealed profiles/runner and physical parity exist, LIBERO framework
capabilities remain CPU-tested model hooks, not an end-to-end certificate.

Update: metadata now includes r17_model_spec, r17_execution_protocol and
r17_samplers mapping strings10/2/1 to actual native SamplerSpecs (released
profile only10). Your physical capture resolver can select the actual bound K.
P1's explicit allow_partial_commit negotiation is now available for system
H32/commit32 when the runner declares partial_tail_protocol. Default profile
binding still rejects nondivisible commits; released profile replan32 naturally
accepts commit32 without short-tail negotiation.

Published fixed-seed reference spec consumed in two P3 CPU regressions via your
unchanged-original-kernel fixture + actual P1 policy/P3 hooks. System protocol
executes34 actual controls with semantic seeds and tail reuse. Released profile
executes32 then2 on early done, both seed42, final terminal ack closes policy.
All72 P3 tests pass. Formal official-task sealing and physical GPU qualification
remain coordinator/P2, and raw-Wan vs published-sidecar equivalence remains G2.

## Fix round 2 — load-time benchmark/profile validation (B2 N3)

P3 is adding LoadSpec.benchmark_specs (tuple of P2 manifest/asdict documents).
Every formal benchmark_protocols identity must have exactly one matching spec;
the reconstructed P2 identity and execution parameters are checked before any
checkpoint body/hash/model loading. LIBERO system requires seed7 paired_semantic,
replan5/settle10/wire256/caps220,280,300,520; released_libero requires seed42 fixed,
replan32/settle10/wire256/caps280,280,300,520. MW requires the canonical FORMAL_SPEC.
Please include those documents in real serve descriptors. Only explicitly
diagnostic fake_* protocol descriptors can omit them. No shared changes requested.
