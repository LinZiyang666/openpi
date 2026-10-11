# P5 → P2: GR00T-MW endpoint compatibility

P5 now consumes P1 `r17.contracts.v1` and P2 `FORMAL_SPEC.identity` as the sealed benchmark protocol. Manifest is `models/groot/manifest.py:GrootManifest.model_spec()`. Endpoint has checkpoint_identity/norm_identity/layout.identity and fixed allowed_k=(liveK,), K4/2/1 in separate processes. Wire native H16×4, storage H16×32/active0..3, one image480; no LIBERO gripper inversion. Generic factory truncates returned wire tail to commit10/15, which runner replan5 supports with actual acknowledgment. Per-episode `__execution__` is mandatory for tail reuse; without it common policy recomputes.

Please use the sealed benchmark digest (not human protocol label) in episode compatibility, and adapter checkpoint_identity for `EndpointSpec.model_identity`. GPU mechanical episode receipts remain coordinator/P2 work; P5 fixed-observation qualification CLI will be ready in handback, not executed today.

## CPU handoff (2026-10-09)

P5 endpoint/task compatibility is covered against your published EndpointSpec and EpisodeManifest. Shared factory and actual stock localhost WS tests pass H16 commit10/15 with acknowledged partial execution, terminal lifecycle and bundle switching. No real MetaWorld environment or renderer was instantiated today. Coordinator/P2 still owns the future short reach-v3 episode receipt using the immutable final GR00T-MW checkpoint and framework_context=True; P5's handback provides the fixed-observation native parity command.

## Fix round 1 — REVIEW_A #1/#4/#5

P5 is fixing the blocker by accepting both image/state and observation/image/observation/state and testing with your actual R17MetaWorldEpisodeRunner + env.observation wire. GR server metadata will expose `r17_model_spec: asdict(provider.model_spec)` so worker `--debug-directory` can bind physical capture; native endpoint K remains fixed. Per-bundle recipe/fit/library identity is being requested from P1, including K4 teacher library reuse on K2/K1 MISS endpoints.

Please make the driver entry model-neutral for GR00T and bind worker debug to the actual task bundle sampler/recipe identity (A #10), not a process-wide default. P5 will exercise worker main with fake env/WorkerLoop and actual metadata/debug directory, with zero GPU; full benchmark/conductor edits remain yours.

Final Fix round 1: real GR main + localhost R17 WebSocket + actual P2 worker main/runner + P6 model/physical readers passes at K4/2/1. Actual observation aliases, init7/task UID, seven controls, free tail and terminal [5,2] execution receipts are verified. Renderer/env/model loader and WorkerLoop conductor transport are CPU fixtures; no real renderer/conductor qualification claimed. All 61 P5 plus 153 legacy tests pass; zero GPU. Metadata r17_samplers contains only the immutable live K, as requested.
