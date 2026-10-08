# B-pool recording requests

Created before designing or evaluating a look-half arm. No launch is requested of the explorer; the coordinator owns recording and launch decisions.

Priority: record the unchanged five-layer baseline on B-pool dev initial states disjoint from the served library. Start with pi05 l10 and spatial, R10 nested sizes 50 and 200, then GR00T equivalents if feasible. Use all ten tasks and ten dev episodes per task per cell-size. For each, knob off and guard-aware periodic target IR .25 are useful paired drivers. If time is limited, prioritize pi05 l10 size50 and spatial size50, knob off.

At each real cache look, record both key_v0 and key_v1, rs/raw_state, task, episode/init identity, step, has_vision, previous executed action and hit history, selected rows/weights, pre/post-correction cache chunks, guard reason, progress state, layer-4 due state and actual policy calls. Record policy shadow chunks on a deterministic subsample of looks if affordable, with shadow cost excluded from deployment IR. Save camera token features/images on that same subsample if possible. Please preserve the causal trajectory; counterfactual wrist decisions should initially remain shadow decisions.

These recordings test half-view retrieval agreement, guard/corrector behavior, and distribution shift on cache-driven states. They cannot establish SR of a changed controller. A separately frozen paired B-pool dev pilot will be specified in EXPERIMENTS.md after offline analysis.

For size500 no within-parent held-out B initial states remain: use additional B-pool dev states outside all served episodes, and save an explicit disjointness manifest. Do not substitute test A or silently reuse served initial states.

Cost request: measure pi05 full, wrist-only, agentview-only, and second-camera completion with warm batch-one request semantics (including full-look fallback after a half look). GR00T needs a validated actual one-camera path and matching measurements before an IR claim. The local GPU is currently unavailable to this explorer (`nvidia-smi` cannot communicate with its driver); CPU offline work continues.

No numerical outcomes or results from another R12 explorer should be sent to this directory.
# DIRECTIVE_2 follow-up (post-freeze, 2026-10-03)

The saturated-serving compute prices are settled; no additional latency benchmark is requested. Current B recordings suffice for the fixed-state cache/guard screens in `directive2/`.

Before claiming a deployable GR00T result, please provide a B-only numerical-path pilot with exact encoder input pixels (or original RGB plus the exact transform), both joint and genuinely single-camera keys for the same observation, and policy chunks from joint versus completed separate-camera prefixes using the same policy noise. The existing 56px pictures permit only reconstructed-image sensitivity tests. GR00T B libraries have no original RGB arrays; we cannot honestly call those reconstructions a single-camera re-encoding of the library. A matched single-camera B library fit, or a verified robust representation on exact pixels, is required.

This matters on rejected half proposals too: acquiring the other camera restores information, but separately encoded GR00T prefixes need not equal the joint-prefix full control. Compare both full-information numerical paths at the same compute-share price. Do not count their request latency difference as a method or saving.

For controller evaluation, use the concrete B-first arms and fixed parameters in the revised `EXPERIMENTS.md`; this request is a handoff only. I have not launched or sent any evaluation job.
